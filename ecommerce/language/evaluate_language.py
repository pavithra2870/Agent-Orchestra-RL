"""Measure the language layer. Protocol (no leakage):
  examples  = phrase set A   (what the router compares against)
  tune/cal  = phrase set C   (threshold + confidence calibration)
  test      = phrase set B   (never used for anything but this final number)

  python -m ecommerce.language.evaluate_language            # router only (fast)
  python -m ecommerce.language.evaluate_language --llm 60   # also LLM parser on 60 test messages (slow)
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from ecommerce.config import RESULTS_DIR
from ecommerce.evaluation.splits import get_splits
from ecommerce.language.router import IntentRouter
from ecommerce.sim.scenarios import generate_scenarios
from ecommerce.sim.text import example_sentences
from ecommerce.sim.world import GOALS

OUT_DIR = RESULTS_DIR / "language"


def score_router(router: IntentRouter, messages: Sequence[str], truth: Sequence[Tuple[str, ...]]) -> Dict[str, Any]:
    outs = [router.route(m) for m in messages]
    exact = [set(o.goals) == set(g) for o, g in zip(outs, truth)]
    oos_true = [len(g) == 0 for g in truth]
    oos_pred = [len(o.goals) == 0 for o in outs]
    tp = sum(a and b for a, b in zip(oos_true, oos_pred))
    per_goal = {}
    for goal in GOALS:
        tp_g = sum(goal in o.goals and goal in g for o, g in zip(outs, truth))
        fp = sum(goal in o.goals and goal not in g for o, g in zip(outs, truth))
        fn = sum(goal not in o.goals and goal in g for o, g in zip(outs, truth))
        per_goal[goal] = {"precision": tp_g / max(1, tp_g + fp), "recall": tp_g / max(1, tp_g + fn)}
    conf = np.array([o.parse_conf for o in outs])
    ok = np.array(exact)
    return {
        "n": len(messages), "exact_match": float(ok.mean()),
        "oos_precision": tp / max(1, sum(oos_pred)), "oos_recall": tp / max(1, sum(oos_true)),
        "per_goal": per_goal,
        "mean_conf_when_correct": float(conf[ok].mean()) if ok.any() else None,
        "mean_conf_when_wrong": float(conf[~ok].mean()) if (~ok).any() else None,
        "mean_best_similarity": float(np.mean([max(c.sim for c in o.clauses) for o in outs])),
        "low_conf_catches_errors": float(np.mean(conf[~ok] < 0.6)) if (~ok).any() else None,
        "low_conf_false_alarms": float(np.mean(conf[ok] < 0.6)) if ok.any() else None,
    }


def run_router_eval(n_val: int = 500, n_test: int = 500) -> Dict[str, Any]:
    sp = get_splits()
    val = generate_scenarios(sp["language_val"], n_val)
    test = generate_scenarios(sp["paraphrase"], n_test)
    train_text = generate_scenarios(sp["validation"], n_test)          # phrase set A, same wording as the examples
    vm, vg = [s.text for s in val], [s.true_goals for s in val]
    tm, tg = [s.text for s in test], [s.true_goals for s in test]
    am, ag = [s.text for s in train_text], [s.true_goals for s in train_text]

    router = IntentRouter()
    tuned = router.tune_threshold(vm, vg)
    cal = router.fit_calibration(vm, vg)
    res: Dict[str, Any] = {"embedder": router.embedder.name, "tuned": tuned, "calibration": cal,
                           "examples_A_only": {"seen_wording_A": score_router(router, am, ag),
                                               "tuning_wording_C": score_router(router, vm, vg),
                                               "held_out_wording_B": score_router(router, tm, tg)}}
    # same thresholds and calibration, but more example sentences (A + C). B is still untouched.
    ex = example_sentences("A")
    exC = example_sentences("C")
    merged = {k: ex[k] + exC[k] for k in ex}
    router2 = IntentRouter(examples=merged, in_scope_threshold=router.tau_in)
    router2.calibrator = router.calibrator
    res["examples_A_plus_C"] = {"held_out_wording_B": score_router(router2, tm, tg)}
    # the deployed router = A + C examples with the thresholds/calibration tuned on C (see router_store.py)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "router_params.json").write_text(json.dumps({**router2.export_params(), "examples": "A+C"}, indent=2))
    res["errors_on_B"] = [{"text": s.text, "truth": list(s.true_goals), "predicted": list(router.route(s.text).goals)}
                          for s in test if set(router.route(s.text).goals) != set(s.true_goals)][:25]
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=int, default=0, help="also evaluate the LLM parser on this many test messages")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t = time.time()
    res = run_router_eval()
    res["seconds"] = time.time() - t
    if a.llm:
        from ecommerce.language.evaluate_llm import run_llm_eval
        res["llm_parser"] = run_llm_eval(a.llm)
    (OUT_DIR / "router_eval.json").write_text(json.dumps(res, indent=2))
    for k in ("seen_wording_A", "tuning_wording_C", "held_out_wording_B"):
        r = res["examples_A_only"][k]
        print(f"{k:20s} exact={r['exact_match']:.3f} oos_recall={r['oos_recall']:.2f} best_sim={r['mean_best_similarity']:.3f}")
    r = res["examples_A_plus_C"]["held_out_wording_B"]
    print(f"{'A+C examples on B':20s} exact={r['exact_match']:.3f} oos_recall={r['oos_recall']:.2f}")


if __name__ == "__main__":
    main()
