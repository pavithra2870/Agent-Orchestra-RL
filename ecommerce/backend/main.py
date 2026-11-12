"""FastAPI backend (port 8000). Start with:  python -m ecommerce.backend.main

Serves the JSON API under /api and, if the React app has been built (ecommerce/frontend/dist),
the dashboard at /. Streamlit (port 8501) is a separate app and is not touched.
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ecommerce.backend.jobs import JobError, JobManager
from ecommerce.backend.metric_help import METRIC_HELP
from ecommerce.backend.orchestrator import ChatRequest, run_chat, run_replay, summarize_events
from ecommerce.backend.services import CHAT_AGENTS, Services
from ecommerce.evaluation.splits import get_splits
from ecommerce.sim.scenarios import generate_scenario

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
REPLAY_SPLITS = ("iid", "heldout_combos", "tool_failures", "missing_info", "out_of_scope", "returns", "policy", "paraphrase", "train", "validation")


class ChatBody(BaseModel):
    message: str = Field(min_length=1, max_length=600)
    customer_id: str = Field(pattern=r"^CUST-\d{4}$")
    agent: str = "dqn"
    language_mode: str = Field(default="auto", pattern="^(auto|template|llm)$")
    seed: int = Field(default=0, ge=0, le=10_000)
    context: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, str]]] = Field(default=None, max_length=3)   # last 3 messages {role, text}


class InspectBody(BaseModel):
    order_id: str = Field(pattern=r"^ORD-\d{4}$")
    result: str = Field(pattern="^(passed|failed)$")


class ReplayBody(BaseModel):
    scenario_id: str = Field(pattern=r"^[a-z_]+-\d{5}$")
    agent: str
    run_seed: int = Field(default=0, ge=0, le=100)


class TrainBody(BaseModel):
    agent: str
    seeds: List[int] = [0]
    episodes: Optional[int] = None
    variant: Optional[str] = None
    hp: Dict[str, float] = {}
    eval_iid: int = 100


class EvalBody(BaseModel):
    llm: int = 0


def create_app(svc: Optional[Services] = None, warm_up: bool = True) -> FastAPI:
    svc = svc or Services()
    jobs = JobManager(svc)
    app = FastAPI(title="E-commerce support agent API", version="1.0")
    app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                       allow_methods=["*"], allow_headers=["*"])
    app.state.svc, app.state.jobs = svc, jobs
    episodes_cache: Dict[str, Any] = {"mtime": None, "df": None}

    if warm_up:
        threading.Thread(target=lambda: svc.parser("rules"), daemon=True).start()

    # ------------------------------------------------------------------ meta
    @app.get("/api/health")
    def health() -> Dict[str, Any]:
        return {"ok": True, "llm_available": svc.llm_available(), "llm_model": svc.llm.model,
                "agents_available": [k for k in CHAT_AGENTS if svc.agent_available(k)],
                "has_eval_results": (svc.results_dir / "eval" / "summary.json").exists()}

    @app.get("/api/agents")
    def agents() -> List[Dict[str, Any]]:
        return svc.list_agents()

    @app.get("/api/metrics/help")
    def metrics_help() -> Dict[str, Any]:
        return METRIC_HELP

    @app.get("/api/customers")
    def customers() -> List[Dict[str, Any]]:
        w = svc.demo_world()
        out = []
        for c in list(w.customers.values())[:40] + [c for c in w.customers.values() if c.customer_id > "CUST-0060"]:
            out.append({"customer_id": c.customer_id, "name": c.name, "orders": [
                {"order_id": o, "items": w.orders[o].items, "status": w.orders[o].status,
                 "payment_method": w.orders[o].payment_method, "payment_state": w.orders[o].payment_state,
                 "total": w.orders[o].total, "expected_delivery": w.orders[o].expected_delivery,
                 "delivered_on": w.orders[o].delivered_on, "return_state": w.orders[o].return_state}
                for o in c.order_ids]})
        return out

    @app.get("/api/demo/changes")
    def demo_changes() -> Dict[str, Any]:
        return {"changed_orders": svc.changed_orders()}

    @app.get("/api/demo/returns")
    def demo_returns() -> Dict[str, Any]:
        """Returns waiting for the supply-chain team's inspection (demo buttons: passed / failed)."""
        return {"waiting": svc.returns_waiting()}

    @app.post("/api/demo/inspect")
    def demo_inspect(body: InspectBody) -> Dict[str, Any]:
        try:
            return svc.inspect_return(body.order_id, body.result == "passed")
        except KeyError:
            raise HTTPException(404, "unknown order")
        except ValueError as e:
            raise HTTPException(409, str(e))

    @app.get("/api/demo/claims")
    def demo_claims() -> Dict[str, Any]:
        """Audit list of damage claims (instant refunds are reviewed afterwards by staff)."""
        return {"claims": svc.damage_claims()}

    @app.post("/api/demo/restore")
    def demo_restore() -> Dict[str, Any]:
        """Testing aid: put the demo database back to its original values."""
        return {"restored_orders": svc.restore_demo_world()}

    @app.get("/api/examples")
    def examples() -> List[Dict[str, Any]]:
        w = svc.demo_world()
        picks: List[Dict[str, Any]] = []
        for c in w.customers.values():
            if not c.order_ids:
                continue
            o = w.orders[c.order_ids[0]]
            if o.status == "delayed" and o.payment_method == "COD" and o.payment_state == "pending" and not o.out_for_delivery:
                picks.append({"label": "Delayed order + switch COD to UPI (the brief's example)", "customer_id": c.customer_id,
                              "message": f"My order {o.order_id} is delayed. Can you check where it is and change my payment method from cash on delivery to UPI?"})
                break
        multi = next((c for c in w.customers.values() if len(c.order_ids) > 1), None)
        if multi:
            picks.append({"label": "Missing order id (agent must ask)", "customer_id": multi.customer_id,
                          "message": "where is my order? also i want to cancel it"})
        refundable = next((c for c in w.customers.values() if c.order_ids and w.orders[c.order_ids[0]].status in ("lost", "returned")
                           and w.orders[c.order_ids[0]].payment_state == "paid" and w.orders[c.order_ids[0]].total <= 5000), None)
        if refundable:
            oid = refundable.order_ids[0]
            picks.append({"label": "Refund for a lost order", "customer_id": refundable.customer_id,
                          "message": f"hi, i want a refund for {oid}"})
        from datetime import date, timedelta

        today = date.fromisoformat(w.today)
        cfg = w.cfg

        def delivered_days(o):
            return (today - date.fromisoformat(o.delivered_on)).days if o.status == "delivered" and o.delivered_on else None

        def pick_return(pred):
            for c in w.customers.values():
                for oid in c.order_ids:
                    o = w.orders[oid]
                    d = delivered_days(o)
                    if d is not None and o.payment_state == "paid" and o.return_state is None and pred(c, o, d):
                        return c, o
            return None, None

        c, o = pick_return(lambda c, o, d: d <= cfg.damage_report_window_days and o.total <= cfg.instant_refund_cap
                           and len(c.claims) < cfg.damage_claims_limit and len(c.order_ids) == 1)
        if c:
            picks.append({"label": "Damaged item, small order (instant refund or exchange)", "customer_id": c.customer_id,
                          "message": f"the item in my order {o.order_id} arrived broken"})
        c, o = pick_return(lambda c, o, d: d <= cfg.damage_report_window_days and o.total > cfg.instant_refund_cap and len(c.order_ids) == 1)
        if c:
            picks.append({"label": "Damaged item, big order (staff will contact you)", "customer_id": c.customer_id,
                          "message": f"my order {o.order_id} arrived damaged, i want a replacement"})
        c, o = pick_return(lambda c, o, d: d <= cfg.return_window_days and o.total <= cfg.high_value_threshold and len(c.order_ids) == 1)
        if c:
            picks.append({"label": "Changed my mind, return it", "customer_id": c.customer_id,
                          "message": f"i don't want the item in {o.order_id} anymore, i want to return it"})
        any_c = next(iter(w.customers.values()))
        picks.append({"label": "Policy question (no order needed)", "customer_id": any_c.customer_id,
                      "message": "what happens if my order is lost?"})
        picks.append({"label": "Out of scope (must hand to a human)", "customer_id": any_c.customer_id,
                      "message": "can you recommend a good smartphone?"})
        picks.append({"label": "Unusual wording", "customer_id": any_c.customer_id,
                      "message": f"the courier hasn't shown up with {any_c.order_ids[0]}, any update? plus i'd like to scrap it"})
        return picks

    # ------------------------------------------------------------------ chat
    def _persist(events: List[Dict[str, Any]]) -> None:
        if events and any(e["type"] == "done" for e in events):
            svc.db.save_run(summarize_events(events), events)

    def _guard(agent: str) -> None:
        if agent not in CHAT_AGENTS or not svc.agent_available(agent):
            raise HTTPException(404, f"agent '{agent}' is not available (train it first)")

    @app.post("/api/chat/run")
    def chat_run(body: ChatBody) -> Dict[str, Any]:
        _guard(body.agent)
        events = list(run_chat(svc, ChatRequest(**body.model_dump())))
        _persist(events)
        return {"run_id": events[0].get("run_id"), "events": events}

    @app.get("/api/chat/stream")
    def chat_stream(message: str = Query(min_length=1, max_length=600), customer_id: str = Query(pattern=r"^CUST-\d{4}$"),
                    agent: str = "dqn", language_mode: str = Query("auto", pattern="^(auto|template|llm)$"),
                    seed: int = Query(0, ge=0, le=10_000), context: Optional[str] = None, history: Optional[str] = None,
                    delay_ms: int = Query(0, ge=0, le=2000)) -> StreamingResponse:
        _guard(agent)
        try:
            ctx = json.loads(context) if context else None
            hist = json.loads(history)[-3:] if history else None
        except json.JSONDecodeError:
            raise HTTPException(400, "context and history must be valid JSON")
        req = ChatRequest(message, customer_id, agent, language_mode, seed, ctx, hist)

        def gen() -> Iterator[str]:
            events: List[Dict[str, Any]] = []
            try:
                for ev in run_chat(svc, req):
                    events.append(ev)
                    yield f"data: {json.dumps(ev)}\n\n"
                    if delay_ms:
                        time.sleep(delay_ms / 1000)
            finally:
                _persist(events)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------------ runs
    @app.get("/api/runs")
    def runs(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), agent: Optional[str] = None,
             status: Optional[str] = None, q: Optional[str] = None, kind: Optional[str] = None) -> Dict[str, Any]:
        return svc.db.list_runs(limit, offset, agent, status, q, kind)

    @app.get("/api/runs/{run_id}")
    def run_detail(run_id: str) -> Dict[str, Any]:
        r = svc.db.get_run(run_id)
        if r is None:
            raise HTTPException(404, "run not found")
        return r

    # ------------------------------------------------------------------ results
    def _need(*parts: str) -> Any:
        data = svc.read_json(*parts)
        if data is None:
            raise HTTPException(404, f"{'/'.join(parts)} not found. Run the training / evaluation first.")
        return data

    @app.get("/api/results/summary")
    def results_summary() -> Any:
        return _need("eval", "summary.json")

    @app.get("/api/results/breakdowns")
    def results_breakdowns() -> Any:
        return _need("eval", "breakdowns.json")

    @app.get("/api/results/training")
    def results_training() -> Any:
        return {"summary": _need("eval", "training_summary.json"), "curves": _need("eval", "curves.json"),
                "ppo_tuning": svc.read_json("training", "ppo_tuning.json"),
                "logged_dataset": svc.read_json("..", "data", "logged_dataset.json")}

    @app.get("/api/results/language")
    def results_language() -> Any:
        return {"router": _need("language", "router_eval.json"), "pipeline": svc.read_json("eval", "pipeline.json"),
                "knowledge": svc.read_json("language", "knowledge_eval.json")}

    @app.get("/api/results/llm_benchmark")
    def results_llm() -> Any:
        return _need("eval", "llm_benchmark.json")

    def _episodes() -> pd.DataFrame:
        f = svc.results_dir / "eval" / "episodes.csv.gz"
        if not f.exists():
            raise HTTPException(404, "episodes.csv.gz not found. Run the evaluation first.")
        mt = f.stat().st_mtime
        if episodes_cache["mtime"] != mt:
            episodes_cache["df"] = pd.read_csv(f).fillna({"failure_reasons": ""})
            episodes_cache["mtime"] = mt
        return episodes_cache["df"]

    @app.get("/api/results/failures")
    def failures(agent: str, split: Optional[str] = None, reason: Optional[str] = None, seed: Optional[int] = None,
                 limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)) -> Dict[str, Any]:
        df = _episodes()
        df = df[(df["agent_key"] == agent) & (~df["success"].astype(bool))]
        if split:
            df = df[df["split"] == split]
        if seed is not None:
            df = df[df["seed"] == seed]
        if reason:
            df = df[df["failure_reasons"].str.contains(re.escape(reason))]
        counts: Dict[str, int] = {}
        for s in df["failure_reasons"]:
            for r in str(s).split(";"):
                if r:
                    key = r.split(":")[0] if r.startswith(("violation", "missed_goal", "missed_escalation")) else r
                    counts[key] = counts.get(key, 0) + 1
        cols = ["scenario_id", "split", "type", "combo", "seed", "failure_reasons", "steps", "reward", "terminal", "expected_terminal"]
        return {"total": int(len(df)), "by_reason": counts, "rows": df[cols].iloc[offset:offset + limit].to_dict("records")}

    @app.post("/api/replay")
    def replay(body: ReplayBody) -> Dict[str, Any]:
        _guard(body.agent)
        split, idx = body.scenario_id.rsplit("-", 1)
        if split not in REPLAY_SPLITS:
            raise HTTPException(400, "unknown split")
        sc = generate_scenario(get_splits(svc.settings)[split], int(idx), svc.settings)
        events = list(run_replay(svc, sc, body.agent, run_seed=body.run_seed))
        _persist(events)
        return {"scenario": sc.public(), "events": events}

    # ------------------------------------------------------------------ jobs
    @app.post("/api/jobs/train")
    def job_train(body: TrainBody) -> Dict[str, Any]:
        try:
            return jobs.submit_training(body.model_dump())
        except (JobError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.post("/api/jobs/evaluate")
    def job_eval(body: EvalBody) -> Dict[str, Any]:
        try:
            return jobs.submit_evaluation(body.model_dump())
        except (JobError, ValueError) as e:
            raise HTTPException(400, str(e))

    @app.get("/api/jobs")
    def job_list() -> List[Dict[str, Any]]:
        return jobs.list()

    @app.get("/api/jobs/{job_id}")
    def job_get(job_id: str) -> Dict[str, Any]:
        j = jobs.get(job_id)
        if j is None:
            raise HTTPException(404, "job not found")
        return j

    @app.post("/api/jobs/{job_id}/cancel")
    def job_cancel(job_id: str) -> Dict[str, Any]:
        j = jobs.cancel(job_id)
        if j is None:
            raise HTTPException(404, "job not found")
        return j

    # ------------------------------------------------------------------ dashboard files
    if FRONTEND_DIST.exists():
        app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="dashboard")
    return app


def main() -> None:
    import uvicorn

    uvicorn.run(create_app(), host="127.0.0.1", port=8000, log_level="info")


if __name__ == "__main__":
    main()
