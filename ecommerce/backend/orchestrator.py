"""Runs one conversation step by step and yields an event for every stage.

Event types, in order: start, parse, state, (decision, tool, state)*, reply, evaluation, done.
The dashboard animates these events; the same list is saved to SQLite so any run can be replayed.

Two entry points share one loop:
  run_chat    a live customer message (parser -> scenario built from the demo database)
  run_replay  a scenario from an evaluation split (used by the failure drill-down)

The RL policy chooses every tool. The language model (if enabled) only reads the message and
rewords the reply; both of its outputs are checked.
"""

from __future__ import annotations

import copy
import time
import zlib
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterator, List, Optional

import numpy as np

from ecommerce.agents.base import Agent
from ecommerce.backend.db import new_id
from ecommerce.backend.services import Services
from ecommerce.env.env import EcomSupportEnv
from ecommerce.env.state import ACTIONS, NUM_ACTIONS, state_view
from ecommerce.evaluation.evaluator import expected_outcomes
from ecommerce.language.parser import ParsedRequest, extract_return_kind, extract_return_reason
from ecommerce.language.orders import STATE_CHANGING_GOALS, Resolution, is_no, is_yes, resolve_order, would_change
from ecommerce.language.reply import METHOD_LABEL, assumption_text, build_facts, confirm_reply, confirm_return_reply, write_reply
from ecommerce.sim import rules
from ecommerce.sim.policy import build_topics
from ecommerce.sim.scenarios import Scenario, scenario_from_world
from ecommerce.sim.tools import order_summaries

SCORE_KIND = {"ppo": "probability"}


@dataclass
class ChatRequest:
    message: str
    customer_id: str
    agent: str = "dqn"
    language_mode: str = "auto"              # auto (LLM if it is running, else template) | template | llm
    seed: int = 0
    context: Optional[Dict[str, Any]] = None  # returned by a previous run that asked a question
    history: Optional[List[Dict[str, str]]] = None  # the last 3 messages, {role: user|agent, text}


def _scores(agent: Agent, obs: np.ndarray, mask: np.ndarray) -> Optional[Dict[str, Any]]:
    raw = agent.action_scores(obs, mask)
    if raw is None:
        return None
    raw = np.asarray(raw, dtype=float)
    items = [{"action": ACTIONS[i], "value": round(float(raw[i]), 3)} for i in np.flatnonzero(mask) if not np.isnan(raw[i])]
    items.sort(key=lambda d: -d["value"])
    return {"kind": SCORE_KIND.get(agent.name, "q_value"), "top": items[:6]}


def _diff(before: Dict[str, Dict[str, Any]], after: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for oid, b in before.items():
        for field_, old in b.items():
            new = after[oid][field_]
            if new != old:
                out.append({"order_id": oid, "field": field_, "before": old, "after": new})
    return out


def episode_events(svc: Services, scenario: Scenario, agent: Agent, live: bool, run_seed: int, language_mode: str,
                   parsed: Optional[ParsedRequest] = None, extra_question: Optional[str] = None,
                   resolution: Optional[Resolution] = None) -> Iterator[Dict[str, Any]]:
    env = EcomSupportEnv(svc.settings, enforce_mask=agent.uses_mask)
    obs, info = env.reset(options={"scenario": scenario, "live": live, "run_seed": run_seed})
    agent.reset(scenario if agent.needs_scenario else None)
    owned = scenario.world.customers[scenario.customer_id].order_ids
    before = {o: dict(v) for o, v in scenario.world.snapshot().items() if o in owned}
    yield {"type": "state", "step": 0, "state": state_view(env.state),
           "legal": [ACTIONS[i] for i in np.flatnonzero(info["action_mask"])]}

    total, done, n = 0.0, False, 0
    while not done:
        mask = info["action_mask"] if agent.uses_mask else np.ones(NUM_ACTIONS, dtype=bool)
        scores = _scores(agent, obs, mask)
        a = agent.act(obs, mask, info)
        n += 1
        yield {"type": "decision", "step": n, "action": ACTIONS[a], "policy": agent.name, "scores": scores,
               "legal": [ACTIONS[i] for i in np.flatnonzero(mask)]}
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
        yield {"type": "tool", "step": n, **info["trace_step"], "reward": round(r, 3)}
        yield {"type": "state", "step": n, "state": state_view(env.state),
               "legal": [ACTIONS[i] for i in np.flatnonzero(info["action_mask"])]}

    terminal = info["terminal"]
    after = {o: dict(v) for o, v in env.world.snapshot().items() if o in owned}
    changes = _diff(before, after)
    if live and changes:
        # a live chat really changes the demo database; "Restore original data" on the dashboard undoes everything
        svc.commit_orders({c["order_id"]: copy.deepcopy(env.world.orders[c["order_id"]]) for c in changes},
                          {scenario.customer_id: copy.deepcopy(env.world.customers[scenario.customer_id].claims)})
    name = scenario.world.customers[scenario.customer_id].name
    facts = build_facts(name, env.trace, terminal, sorted(scenario.parsed_goals))
    if terminal == "awaiting_customer":
        if extra_question == "which_method":
            facts["question"]["type"] = "which_method"
            current = scenario.world.orders[facts["order_id"]].payment_method if facts.get("order_id") in scenario.world.orders else None
            methods = [m for m in ("UPI", "CARD") if m != current]
            facts["question"]["methods"] = methods
            facts["question"]["options"] = [{"label": METHOD_LABEL[m], "text": METHOD_LABEL[m]} for m in methods]
        elif extra_question in RETURN_OPTIONS:
            facts["question"]["type"] = extra_question
            facts["question"]["reason"] = scenario.return_reason
            facts["question"]["options"] = RETURN_OPTIONS[extra_question]
        else:
            quoted = scenario.order_id_in_text
            not_theirs = quoted is not None and quoted not in owned
            needs_order = bool(set(scenario.parsed_goals) - {"policy_info"})
            if needs_order and (not_theirs or (env.state.order_candidate is None and len(owned) > 1)):
                facts["question"]["type"] = "which_order"
                facts["question"]["orders"] = list(owned)
                facts["question"]["detail"] = order_summaries(scenario.world, scenario.customer_id)
                facts["question"]["rejected_order"] = quoted if not_theirs else None
                if resolution is not None and not not_theirs and resolution.how in ("ambiguous", "none_qualifies"):
                    keep = resolution.candidates
                    facts["question"]["orders"] = list(keep)
                    facts["question"]["detail"] = [d for d in facts["question"]["detail"] if d["order_id"] in keep]
                    facts["question"]["how"], facts["question"]["why"] = resolution.how, resolution.reason
    elif resolution is not None and resolution.assumed and facts.get("order_id") == resolution.order_id:
        facts["assumption"] = assumption_text(resolution.order_id, facts.get("items", []), resolution.reason)
    elif resolution is not None and resolution.how == "history" and facts.get("order_id") == resolution.order_id:
        facts["assumption"] = assumption_text(resolution.order_id, facts.get("items", []), resolution.reason)
    reply = write_reply(facts, mode=language_mode, client=svc.llm)
    q = facts.get("question") or {}
    if terminal == "awaiting_customer" and q.get("type") == "which_order" and "options" not in q:
        q["options"] = [{"label": f"{d['order_id']} · {', '.join(d['items'])}", "text": d["order_id"]} for d in q.get("detail", [])]
    yield {"type": "reply", "text": reply.text, "mode": reply.mode, "problems": reply.problems,
           "llm_draft": reply.llm_draft, "llm_seconds": round(reply.llm_seconds, 2),
           "options": q.get("options", []) if terminal == "awaiting_customer" else []}

    ev = env.last_eval
    if ev is not None:
        exp = {g: {"kind": e.kind, "reason": e.reason} for g, e in expected_outcomes(scenario).items()}
        yield {"type": "evaluation", "result": ev.to_dict(), "expected": exp,
               "judged_against": "the goals the parser found (live messages have no labels)" if live else "the scenario's ground truth"}
    yield {"type": "done", "terminal": terminal, "success": None if ev is None else ev.success, "reward": round(total, 3),
           "steps": env.state.steps, "cost": round(env.state.cost, 2), "latency": round(env.state.latency, 2),
           "database_changes": changes, "facts": facts, "trace": [t.to_dict() for t in env.trace],
           "agent": agent.name}


def _effective_mode(svc: Services, requested: str) -> str:
    """auto = the local LLM when it is running, otherwise templates. Templates are always the safe fallback."""
    if requested == "auto":
        return "llm" if svc.llm_available() else "template"
    return requested


def _emit_early(parsed: ParsedRequest, resolution: Resolution, t0: float, text: str, context: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """A reply that needs no RL steps (a yes/no question to the customer): parse, reply, done."""
    yield {"type": "parse", **parsed.to_dict(), "resolution": resolution.to_dict()}
    yield {"type": "reply", "text": text, "mode": "template", "problems": [], "llm_draft": None, "llm_seconds": 0.0,
           "options": [{"label": "Yes", "text": "yes"}, {"label": "No", "text": "no"}]}
    yield {"type": "done", "terminal": "awaiting_customer", "success": None, "reward": 0.0, "steps": 0, "cost": 0.0, "latency": 0.0,
           "database_changes": [], "facts": {"confirm": True}, "trace": [], "agent": "confirmation step",
           "seconds": round(time.time() - t0, 2), "context": context}


# what the customer sees as buttons for the three return questions (each text is understood by the parser on its own)
RETURN_OPTIONS = {
    "return_reason": [{"label": "It arrived damaged or faulty", "text": "It arrived damaged or faulty"},
                      {"label": "I just don't want it", "text": "I don't want it, I changed my mind"}],
    "return_kind": [{"label": "Exchange for a new one", "text": "Exchange for a new one"},
                    {"label": "Return it and get a refund", "text": "Return it and get a refund"}],
    "return_declaration": [{"label": "Yes, I confirm", "text": "Yes, I confirm"}, {"label": "No", "text": "No"}],
}


def _next_return_question(ret: Dict[str, Any]) -> Optional[str]:
    """What the customer still has to tell us before a return can be checked (always one question at a time)."""
    if ret.get("reason") is None:
        return "return_reason"
    if ret.get("kind") is None:
        return "return_kind"
    if ret.get("declared") is None:
        return "return_declaration"
    return None


def _fill_return(ret: Dict[str, Any], asking: Optional[str], parsed: ParsedRequest, text: str) -> bool:
    """Take what the customer just wrote into `ret`. Returns True when the answer to the pending question was found."""
    answered = False
    if ret.get("reason") is None:
        ret["reason"] = parsed.return_reason or extract_return_reason(text)
        answered = answered or (asking == "return_reason" and ret["reason"] is not None)
    if ret.get("kind") is None:
        ret["kind"] = parsed.return_kind or extract_return_kind(text)
        answered = answered or (asking == "return_kind" and ret["kind"] is not None)
    if ret.get("declared") is None and asking == "return_declaration":
        if is_yes(text):
            ret["declared"], answered = True, True
        elif is_no(text):
            ret["declared"], answered = False, True
    return answered


def run_chat(svc: Services, req: ChatRequest, run_id: Optional[str] = None) -> Iterator[Dict[str, Any]]:
    run_id = run_id or new_id("run")
    world = svc.demo_world().copy()
    if req.customer_id not in world.customers:
        yield {"type": "error", "message": f"unknown customer {req.customer_id}"}
        return
    mode = _effective_mode(svc, req.language_mode)
    history = [{"role": m.get("role", "user"), "text": str(m.get("text", ""))} for m in (req.history or [])][-3:]
    agent = svc.agent(req.agent, req.seed)
    cust = world.customers[req.customer_id]
    orders = [world.orders[o] for o in cust.order_ids]
    by_id = {o.order_id: o for o in orders}
    today, cfg = date.fromisoformat(world.today), world.cfg
    t0 = time.time()
    yield {"type": "start", "run_id": run_id, "message": req.message, "customer_id": req.customer_id, "customer": cust.name,
           "agent": req.agent, "language_mode": mode, "language_requested": req.language_mode,
           "followup": bool(req.context), "history_used": len(history)}

    ctx = req.context or {}
    exclude: List[str] = list(ctx.get("exclude", []))
    confirmed = False
    parser = svc.parser(mode)
    ret: Dict[str, Any] = dict(ctx.get("return", {}))
    asking = ctx.get("asking")

    # ---- 1. what does the customer want?  (goals, payment method, return answers, policy topic)
    if ctx.get("confirm") and is_yes(req.message):
        # The customer said yes to "do you want me to ... ORD-X?": go ahead with exactly what we asked about.
        goals, method, confirmed = tuple(ctx["goals"]), ctx.get("target_method"), True
        parsed = ParsedRequest(goals, ctx["order_id"], method, 0.9, 0.95, "confirmation")
        parsed.policy_topic = ctx.get("policy_topic")
        parsed.notes.append("the customer confirmed the order we proposed")
    else:
        if ctx.get("confirm") and is_no(req.message):
            exclude = list(dict.fromkeys(exclude + [ctx["order_id"]]))
        parsed = parser.parse(req.message, history)
        answered = False
        if ctx and asking and "return_item" in ctx.get("goals", ()):
            # the message answers a question we asked about a return ("exchange", "yes", "it arrived damaged"): keep the goals
            answered = _fill_return(ret, asking, parsed, req.message)
            if answered:
                parsed.goals = tuple(ctx["goals"])
                parsed.scope_score, parsed.parse_conf = max(parsed.scope_score, 0.9), max(parsed.parse_conf, 0.9)
                parsed.source += " + answer to our question"
                parsed.notes.append(f"this message answers our question ({asking}); the goals from the earlier message are kept")
        if ctx and not answered and not parsed.goals and not parsed.policy_topic:
            # A short reply to our question ("the other one", "where it is"): keep what they asked for before.
            parsed.goals = tuple(ctx.get("goals", ()))
            parsed.target_method = parsed.target_method or ctx.get("target_method")
            parsed.scope_score, parsed.parse_conf = max(parsed.scope_score, 0.9), max(parsed.parse_conf, 0.9)
            parsed.source = parsed.source + " + earlier goals"
            parsed.notes.append("no new request in this message, so the goals from the earlier message are kept")
        elif ctx and not answered and parsed.goals:
            parsed.parse_conf = max(parsed.parse_conf, 0.9) if parsed.order_id else parsed.parse_conf
            parsed.target_method = parsed.target_method or ctx.get("target_method")
        if "return_item" in parsed.goals and not answered:
            _fill_return(ret, None, parsed, req.message)          # reason / exchange-or-refund the customer already mentioned
        if parsed.policy_topic is None and ctx.get("policy_topic") and "policy_info" in parsed.goals:
            parsed.policy_topic = ctx["policy_topic"]
    if "policy_info" in parsed.goals and ctx.get("policy_done"):
        parsed.goals = tuple(g for g in parsed.goals if g != "policy_info")          # already answered earlier in this conversation

    # ---- 2. which order?  (named / history / only one / inferred from the goal)
    resolution = Resolution(parsed.order_id, "named", reason="you named it") if confirmed else resolve_order(
        req.message, parsed.goals, parsed.target_method, orders, today, cfg, history, exclude)
    if (not confirmed and not ctx.get("confirm") and ctx.get("order_id") and
            resolution.how in ("inferred", "ambiguous", "none_qualifies", "no_goals")):
        # the order was settled in the earlier turn (the customer named it); keep it while we collect the missing piece
        resolution = Resolution(ctx["order_id"], "history", reason="the order we were just talking about")
    if exclude and resolution.how == "none_qualifies":
        resolution.reason = resolution.reason.replace("none of your orders", "none of your other orders")
    if resolution.order_id and not parsed.order_id:
        parsed.order_id = resolution.order_id
        parsed.notes.append(f"order {resolution.order_id}: {resolution.reason}")
    if resolution.assumed and resolution.order_id:
        parsed.parse_conf = max(parsed.parse_conf, 0.8)

    extra_question = None
    if "payment_change" in parsed.goals and not parsed.target_method:
        parsed.parse_conf = min(parsed.parse_conf, 0.45)
        parsed.notes.append("payment change requested but no new payment method was named, so the agent must ask")
        extra_question = "which_method"

    # return / exchange / damage claim: the customer's own answers are needed (one question at a time)
    return_known = True
    if "return_item" in parsed.goals:
        nxt = _next_return_question(ret)
        res_order = by_id.get(resolution.order_id)
        if nxt and res_order is not None and (res_order.status != "delivered" or res_order.return_state is not None
                                              or res_order.payment_state != "paid"):
            # Nothing the customer could say would change the answer (not delivered, already in a return, not paid), so
            # the questions are skipped: the agent checks the order and explains why it cannot be returned.
            ret.update(reason=ret.get("reason") or "change_of_mind", kind=ret.get("kind") or "refund",
                       declared=True if ret.get("declared") is None else ret["declared"])
            nxt = None
            parsed.notes.append("this order cannot be returned whatever the answers are, so the return questions were skipped")
        return_known = nxt is None
        if nxt and resolution.order_id in by_id and extra_question is None:
            extra_question = nxt
            parsed.notes.append(f"a return needs the customer's answers first; still missing: {nxt.replace('return_', '')}")

    carry = {"goals": list(parsed.goals), "target_method": parsed.target_method, "customer_id": req.customer_id, "exclude": exclude,
             "return": ret, "policy_topic": parsed.policy_topic}
    if ctx.get("policy_done"):
        carry["policy_done"] = True

    # ---- 3. confirm BEFORE anything that would change money or the order (refund, cancel, payment change, return)
    # Only when it would really happen: if the order is not eligible, the agent just explains and nothing needs confirming.
    if not confirmed and resolution.order_id in by_id:
        o = by_id[resolution.order_id]
        acts = [g for g in parsed.goals if g in STATE_CHANGING_GOALS and g != "return_item"
                and would_change(g, o, parsed.target_method, cfg, today)]
        policy_text = ""
        if "policy_info" in parsed.goals and parsed.policy_topic:
            policy_text = build_topics(svc.settings)[parsed.policy_topic][1]
        if acts:
            amount = o.total if "refund" in acts else None
            text = confirm_reply(cust.name, o.order_id, o.items, acts, resolution.reason, parsed.target_method,
                                 assumed=resolution.assumed, refund_amount=amount)
            if policy_text:
                text = text.replace(f"Hi {cust.name.split()[0]}, ", f"Hi {cust.name.split()[0]}, {policy_text} ", 1)
            ctx_out = {**carry, "goals": [g for g in carry["goals"] if g != "policy_info"], "confirm": True, "order_id": o.order_id}
            yield from _emit_early(parsed, resolution, t0, text, ctx_out)
            return
        if "return_item" in parsed.goals and return_known:
            e = rules.return_eligibility(o, world.customers.get(req.customer_id), ret["reason"], ret["kind"], bool(ret["declared"]),
                                         cfg, today)
            if e.allowed:
                text = confirm_return_reply(cust.name, o.order_id, o.items, ret["reason"], ret["kind"], e.reason, o.total)
                ctx_out = {**carry, "goals": [g for g in carry["goals"] if g != "policy_info"], "confirm": True, "order_id": o.order_id}
                yield from _emit_early(parsed, resolution, t0, text, ctx_out)
                return

    yield {"type": "parse", **parsed.to_dict(), "resolution": resolution.to_dict()}
    scenario = scenario_from_world(
        world, req.customer_id, req.message, parsed.goals, target_method=parsed.target_method,
        order_id_in_text=parsed.order_id, parse_conf=parsed.parse_conf, scope_score=parsed.scope_score,
        scenario_type="live" if parsed.goals else "out_of_scope", seed=zlib.crc32(f"{req.message}|{req.seed}".encode()) % (2**31), scenario_id=run_id,
        policy_topic=parsed.policy_topic, return_reason=ret.get("reason"), return_kind=ret.get("kind"),
        return_declared=bool(ret.get("declared")), return_info_given=return_known,
        prior_clarifications=svc.settings.env.max_clarifications if (parsed.off_topic and not parsed.goals) else 0,
    )
    for ev in episode_events(svc, scenario, agent, True, req.seed, mode, parsed, extra_question, resolution):
        if ev["type"] == "done":
            ev["seconds"] = round(time.time() - t0, 2)
            if ev["terminal"] == "awaiting_customer":
                firm = resolution.how in ("named", "reference", "history", "single_order")
                ctx_out = {**carry, "order_id": parsed.order_id if firm else None}
                if extra_question and extra_question.startswith("return_"):
                    ctx_out["asking"] = extra_question
                if any(t.get("action") == "lookup_policy" and t.get("ok") for t in ev.get("trace", [])):
                    ctx_out["policy_done"] = True
                    ctx_out["goals"] = [g for g in ctx_out["goals"] if g != "policy_info"]
                ev["context"] = ctx_out
        yield ev


def run_replay(svc: Services, scenario: Scenario, agent_key: str, seed: int = 0, run_seed: int = 0,
               language_mode: str = "template") -> Iterator[Dict[str, Any]]:
    agent = svc.agent(agent_key, seed)
    yield {"type": "start", "run_id": scenario.scenario_id, "message": scenario.text, "customer_id": scenario.customer_id,
           "customer": scenario.world.customers[scenario.customer_id].name, "agent": agent_key,
           "language_mode": language_mode, "followup": False, "scenario": scenario.public()}
    yield {"type": "parse", "goals": list(scenario.parsed_goals), "order_id": scenario.order_id_in_text,
           "target_method": scenario.target_method, "scope_score": round(scenario.scope_score, 3),
           "parse_conf": round(scenario.parse_conf, 3), "source": "scenario (ground-truth parser)", "notes": [],
           "router": None, "llm_goals": None, "dropped_goals": []}
    yield from episode_events(svc, scenario, agent, False, run_seed, language_mode)


def summarize_events(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Flatten the event list into the columns stored in the runs table."""
    start = next((e for e in events if e["type"] == "start"), {})
    parse = next((e for e in events if e["type"] == "parse"), {})
    reply = next((e for e in events if e["type"] == "reply"), {})
    ev = next((e for e in events if e["type"] == "evaluation"), None)
    done = next((e for e in events if e["type"] == "done"), {})
    return {
        "run_id": start.get("run_id"), "created_at": time.time(), "kind": "replay" if "scenario" in start else "chat",
        "name": start.get("run_id"), "agent": start.get("agent"), "language_mode": start.get("language_mode"),
        "customer_id": start.get("customer_id"), "message": start.get("message"),
        "status": "failed" if done.get("success") is False else ("waiting" if done.get("terminal") == "awaiting_customer" else "ok"),
        "success": None if done.get("success") is None else int(done["success"]), "reward": done.get("reward"),
        "steps": done.get("steps"), "cost": done.get("cost"), "latency": done.get("latency"),
        "terminal": done.get("terminal"), "reply": reply.get("text"), "parsed_json": {k: v for k, v in parse.items() if k != "type"},
        "eval_json": ev["result"] if ev else None, "config_json": {"language_mode": start.get("language_mode")},
        "notes": "",
    }
