"""Measure the policy-question path: topic retrieval and how well the router tells a policy question from a request.

A policy question is the goal `policy_info`. The router decides that it IS one (measured in evaluate_language.py, per goal);
this script measures the second step, which topic the question is about (the language layer picks the topic by comparing the
question with example questions, then the environment hands it to the `lookup_policy` tool), and the two ways the first
step can go wrong:
  - a real policy question the router did not call a policy question (the agent then misses it)
  - an ordinary request the router wrongly called a policy question (the agent then answers a policy nobody asked for)

Held-out test = TEST_QUESTIONS (phrase set B). Ordinary requests = the paraphrase split (phrase set B) without a policy goal.

  python -m ecommerce.language.evaluate_knowledge
Writes ecommerce/results/language/knowledge_eval.json
"""

import json

import numpy as np

from ecommerce.config import RESULTS_DIR
from ecommerce.evaluation.splits import get_splits
from ecommerce.language.knowledge import TEST_QUESTIONS, KnowledgeBase
from ecommerce.language.router_store import get_router
from ecommerce.sim.scenarios import generate_scenarios


def main() -> None:
    r = get_router()
    kb = KnowledgeBase(r.embedder)
    tst_q = [(t, k) for k, qs in TEST_QUESTIONS.items() for t in qs]
    got_topic = [kb.best(t)[0] for t, _ in tst_q]
    router_says_policy = ["policy_info" in r.route(t).goals for t, _ in tst_q]
    others = [s.text for s in generate_scenarios(get_splits()["paraphrase"], 400) if "policy_info" not in s.true_goals and s.true_goals]
    wrongly = ["policy_info" in r.route(t).goals for t in others]
    res = {"test": {"policy_questions": len(tst_q),
                    "router_recognises_policy_question": float(np.mean(router_says_policy)),
                    "topic_accuracy": float(np.mean([a == k for a, (_, k) in zip(got_topic, tst_q)])),
                    "both_right": float(np.mean([a == k and ok for a, (_, k), ok in zip(got_topic, tst_q, router_says_policy)])),
                    "ordinary_requests": len(others),
                    "ordinary_requests_called_policy": float(np.mean(wrongly))},
           "topics": sorted(TEST_QUESTIONS),
           "misses": [{"text": t, "got_topic": a, "expected": k, "router_said_policy": ok}
                      for (t, k), a, ok in zip(tst_q, got_topic, router_says_policy) if a != k or not ok]}
    (RESULTS_DIR / "language").mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "language" / "knowledge_eval.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res["test"], indent=2))
    for m in res["misses"]:
        print("miss:", m)


if __name__ == "__main__":
    main()
