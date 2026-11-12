"""End-to-end pipeline evaluation: real router instead of the oracle parser.

For each message in the held-out paraphrase split (phrase set B) we
  1. parse the text (router; optionally LLM + router),
  2. give the agent what the PARSER found (goals, order id, payment method, confidence),
  3. judge the result against the scenario's TRUE goals.
The gap between this and the oracle-parser numbers is the cost of language errors.

  python -m ecommerce.evaluation.run_pipeline_eval
Writes ecommerce/results/eval/pipeline.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from typing import Any, Dict, List

import numpy as np

from ecommerce.agents.baselines import RuleBasedAgent
from ecommerce.config import RESULTS_DIR, get_settings
from ecommerce.evaluation.metrics import summarize
from ecommerce.evaluation.run_eval import _agent_for, _available
from ecommerce.evaluation.runner import evaluate_agent, oracle_rewards
from ecommerce.evaluation.splits import get_splits
from ecommerce.language.parser import MessageParser
from ecommerce.language.reply import build_facts, check_reply, template_reply, write_reply
from ecommerce.language.router_store import get_router
from ecommerce.sim.scenarios import Scenario, generate_scenarios

N = 200
AGENTS = ["rule_based", "q_learning", "sarsa", "dqn", "ppo", "cql"]


def apply_parse(sc: Scenario, p) -> Scenario:
    conf = p.parse_conf
    if "payment_change" in p.goals and not p.target_method:
        conf = min(conf, 0.45)
    return dataclasses.replace(sc, parsed_goals=tuple(p.goals), parse_conf=conf, scope_score=p.scope_score,
                               order_id_in_text=p.order_id, parsed_target=p.target_method, use_parsed_target=True)


def run_pipeline_eval(n: int = N, llm_replies: int = 0, progress=print) -> Dict[str, Any]:
    settings = get_settings()
    scenarios = generate_scenarios(get_splits(settings)["paraphrase"], n, settings)
    oracle = oracle_rewards(scenarios, settings)
    parser = MessageParser(get_router(), mode="rules")
    parsed = [parser.parse(s.text) for s in scenarios]
    truth_ok = [set(p.goals) == set(s.true_goals) for p, s in zip(parsed, scenarios)]
    piped = [apply_parse(s, p) for s, p in zip(scenarios, parsed)]
    out: Dict[str, Any] = {
        "messages": n, "phrase_set": "B (held-out wording)", "parser": "router (rules mode)",
        "goal_set_exact_match": float(np.mean(truth_ok)),
        "low_confidence_when_wrong": float(np.mean([p.parse_conf < 0.6 for p, ok in zip(parsed, truth_ok) if not ok])) if not all(truth_ok) else None,
        "agents": {},
    }
    for key in AGENTS:
        seeds = _available(key)
        if not seeds:
            continue
        rows_o, rows_p = [], []
        for seed in seeds:
            agent = _agent_for(key, seed)
            rows_o.append(summarize(evaluate_agent(agent, scenarios, k=2, settings=settings), k=2, oracle=oracle))
            rows_p.append(summarize(evaluate_agent(agent, piped, k=2, settings=settings), k=2, oracle=oracle))
        pick = lambda rows, m: {"mean": float(np.mean([r[m] for r in rows])), "std": float(np.std([r[m] for r in rows]))}
        reasons: Dict[str, float] = {}
        for r in rows_p:
            for k_, v in r["failure_reasons"].items():
                reasons[k_] = reasons.get(k_, 0.0) + v / len(rows_p)
        out["agents"][key] = {
            "oracle_parser": {m: pick(rows_o, m) for m in ("success_rate", "avg_reward", "violation_rate")},
            "router_parser": {m: pick(rows_p, m) for m in ("success_rate", "pass_k", "avg_reward", "violation_rate", "clarify_recall", "avg_steps")},
            "cost_of_language_errors": float(np.mean([a["success_rate"] - b["success_rate"] for a, b in zip(rows_o, rows_p)])),
            "failure_reasons_with_parser": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:6]),
        }
        progress(f"{key}: oracle parser {out['agents'][key]['oracle_parser']['success_rate']['mean']:.3f} "
                 f"-> router parser {out['agents'][key]['router_parser']['success_rate']['mean']:.3f}")

    # reply layer on the same messages, using the rule-based agent's real runs (template mode)
    from ecommerce.env.env import EcomSupportEnv

    flagged = 0
    env, agent, samples = EcomSupportEnv(settings), RuleBasedAgent(), []
    for sc in piped[: min(n, 120)]:
        obs, info = env.reset(options={"scenario": sc})
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(agent.act(obs, info["action_mask"]))
            done = term or trunc
        facts = build_facts(sc.world.customers[sc.customer_id].name, env.trace, info["terminal"], sorted(sc.parsed_goals))
        problems = check_reply(template_reply(facts), facts)
        flagged += bool(problems)
    out["reply_template_flagged"] = {"replies": min(n, 120), "flagged_by_checker": flagged}

    if llm_replies:
        from ecommerce.language.llm import OllamaClient

        client = OllamaClient()
        if client.available():
            t0, raw_flagged, fallbacks, shipped_flagged, ex = time.time(), 0, 0, 0, []
            for sc in piped[:llm_replies]:
                obs, info = env.reset(options={"scenario": sc})
                done = False
                while not done:
                    obs, _, term, trunc, info = env.step(agent.act(obs, info["action_mask"]))
                    done = term or trunc
                facts = build_facts(sc.world.customers[sc.customer_id].name, env.trace, info["terminal"], sorted(sc.parsed_goals))
                rep = write_reply(facts, mode="llm", client=client)
                raw_flagged += bool(rep.problems and rep.llm_draft)
                fallbacks += rep.mode == "template(fallback)"
                shipped_flagged += bool(check_reply(rep.text, facts))
                if len(ex) < 6:
                    ex.append({"message": sc.text, "shipped": rep.text, "mode": rep.mode, "problems": rep.problems, "llm_draft": rep.llm_draft})
            out["reply_llm"] = {"replies": llm_replies, "llm_drafts_flagged": raw_flagged, "fell_back_to_template": fallbacks,
                                "shipped_flagged": shipped_flagged, "seconds_per_reply": (time.time() - t0) / llm_replies, "examples": ex}
    (RESULTS_DIR / "eval").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "eval" / "pipeline.json").write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--llm-replies", type=int, default=0)
    a = ap.parse_args()
    run_pipeline_eval(a.n, a.llm_replies)
