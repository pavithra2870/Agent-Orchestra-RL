"""Compare three ways to read a message on the held-out wording (phrase set B):
router only, LLM only (raw phi4-mini), and LLM + router cross-check.

The LLM is slow on a laptop CPU, so this runs on a sample (default 60 messages)."""

from __future__ import annotations

import time
from typing import Any, Dict

import numpy as np

from ecommerce.evaluation.splits import get_splits
from ecommerce.language.llm import OllamaClient
from ecommerce.language.parser import MessageParser, SCHEMA, SYSTEM
from ecommerce.language.router import IntentRouter
from ecommerce.sim.scenarios import generate_scenarios
from ecommerce.sim.world import GOALS


def _stats(pred_sets, truth_sets, confs=None) -> Dict[str, Any]:
    exact = np.array([set(p) == set(t) for p, t in zip(pred_sets, truth_sets)])
    halluc = np.mean([len(set(p) - set(t)) > 0 for p, t in zip(pred_sets, truth_sets)])
    missed = np.mean([len(set(t) - set(p)) > 0 for p, t in zip(pred_sets, truth_sets)])
    out = {"n": len(exact), "exact_match": float(exact.mean()), "invented_goal_rate": float(halluc),
           "missed_goal_rate": float(missed)}
    if confs is not None:
        c = np.array(confs)
        out["low_conf_catches_errors"] = float(np.mean(c[~exact] < 0.6)) if (~exact).any() else None
        out["low_conf_false_alarms"] = float(np.mean(c[exact] < 0.6)) if exact.any() else None
    return out


def run_llm_eval(n: int = 60) -> Dict[str, Any]:
    client = OllamaClient()
    if not client.available():
        return {"available": False, "error": client.last_error}
    sp = get_splits()
    val = generate_scenarios(sp["language_val"], 500)
    router = IntentRouter()
    router.tune_threshold([s.text for s in val], [s.true_goals for s in val])
    router.fit_calibration([s.text for s in val], [s.true_goals for s in val])
    test = generate_scenarios(sp["paraphrase"], n)
    truth = [s.true_goals for s in test]

    client.chat_json(SYSTEM, 'Message: "warm up"', SCHEMA)                  # load the model once
    router_only = [router.route(s.text) for s in test]
    parser = MessageParser(router, mode="llm", llm=client)
    t0 = time.time()
    parsed = [parser.parse(s.text) for s in test]
    wall = time.time() - t0
    raw_llm = [p.llm_goals if p.llm_goals is not None else p.goals for p in parsed]
    return {
        "available": True, "model": client.model, "messages": n, "phrase_set": "B (held-out wording)",
        "router_only": _stats([o.goals for o in router_only], truth, [o.parse_conf for o in router_only]),
        "llm_only_raw": _stats(raw_llm, truth),
        "llm_plus_router": _stats([p.goals for p in parsed], truth, [p.parse_conf for p in parsed]),
        "fallbacks_to_router": int(sum(p.source == "rules(fallback)" for p in parsed)),
        "goals_dropped_by_guard": int(sum(len(p.dropped_goals) for p in parsed)),
        "avg_llm_seconds": float(np.mean([p.llm_seconds for p in parsed])), "wall_seconds": wall,
        "examples": [{"text": s.text, "truth": list(s.true_goals), "router": list(o.goals), "llm_raw": list(rl),
                      "final": list(p.goals), "conf": round(p.parse_conf, 2), "notes": p.notes}
                     for s, o, rl, p in zip(test, router_only, raw_llm, parsed)
                     if set(p.goals) != set(s.true_goals) or set(rl) != set(s.true_goals)][:15],
    }


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(run_llm_eval(int(sys.argv[1]) if len(sys.argv) > 1 else 40), indent=2))
