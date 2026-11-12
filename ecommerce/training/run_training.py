"""Command-line training driver.

  python -m ecommerce.training.run_training --suite full --seeds 0 1 2
  python -m ecommerce.training.run_training --agents dqn --seeds 0 --episodes 2000 --hp '{"lr": 0.0005}'

--suite full  = main methods (q_learning, sarsa, dqn, ppo, cql) + ablations:
                dqn/ppo with outcome-only reward ("sparse"), dqn/ppo with no guardrail ("nomask"),
                cql with no conservative penalty ("alpha0").
Progress is printed as JSON lines (and appended to --progress-file) so the dashboard can show it live.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ecommerce.config import get_settings
from ecommerce.training import trainer


def make_progress(path: Optional[str]):
    f = open(path, "a", encoding="utf-8") if path else None

    def emit(event: Dict[str, Any]) -> None:
        line = json.dumps({"t": round(time.time(), 2), **event})
        print(line, flush=True)
        if f:
            f.write(line + "\n")
            f.flush()
    return emit


def run_suite(seeds: List[int], episodes: Optional[int], emit) -> List[Dict[str, Any]]:
    settings = get_settings()
    sparse = settings.model_copy(deep=True)
    sparse.reward.goal_handled_bonus = 0.0
    out: List[Dict[str, Any]] = []
    jobs = []
    for seed in seeds:
        for name in ("q_learning", "sarsa", "dqn", "ppo"):
            jobs.append(dict(kind="online", name=name, seed=seed))
        jobs.append(dict(kind="cql", name="cql", seed=seed))
        for name in ("dqn", "ppo"):
            jobs.append(dict(kind="online", name=name, seed=seed, tag="sparse", settings=sparse))
            jobs.append(dict(kind="online", name=name, seed=seed, tag="nomask", masked=False))
        jobs.append(dict(kind="cql", name="cql", seed=seed, tag="alpha0", hp={"alpha": 0.0}))
    for j in jobs:
        emit({"event": "start", "agent": j["name"], "tag": j.get("tag", ""), "seed": j["seed"]})
        if j["kind"] == "cql":
            s = trainer.train_cql(seed=j["seed"], hp=j.get("hp"), settings=settings, progress=emit,
                                  tag=j.get("tag", ""))
        else:
            s = trainer.train_online(j["name"], j["seed"], episodes, settings=j.get("settings", settings),
                                     progress=emit, tag=j.get("tag", ""), masked=j.get("masked", True))
        emit({"event": "done", **s})
        out.append(s)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["full"], default=None)
    ap.add_argument("--agents", nargs="*", default=[])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--hp", type=str, default=None, help="JSON dict of hyperparameter overrides")
    ap.add_argument("--variant", choices=["sparse", "nomask", "robust"], default=None,
                    help="ablation: outcome-only reward ('sparse'), no guardrail ('nomask'), or stress scenarios mixed into training ('robust')")
    ap.add_argument("--tag", type=str, default=None, help="run-name suffix; keeps these runs separate from the official ones")
    ap.add_argument("--eval-iid", type=int, default=0, help="after training, evaluate the best checkpoint on N iid test scenarios")
    ap.add_argument("--progress-file", type=str, default=None)
    a = ap.parse_args(argv)
    emit = make_progress(a.progress_file)
    if a.suite == "full":
        run_suite(a.seeds, a.episodes, emit)
        return 0
    hp = json.loads(a.hp) if a.hp else None
    settings = get_settings()
    if a.variant == "sparse":
        settings = settings.model_copy(deep=True)
        settings.reward.goal_handled_bonus = 0.0
    for name in a.agents:
        for seed in a.seeds:
            emit({"event": "start", "agent": name, "seed": seed})
            tag = a.tag or a.variant or ""
            if name == "cql":
                s = trainer.train_cql(seed=seed, hp=hp, progress=emit, tag=tag)
            else:
                s = trainer.train_online(name, seed, a.episodes, hp=hp, progress=emit, settings=settings,
                                         tag=tag, masked=a.variant != "nomask",
                                         mix="robust" if a.variant == "robust" else "standard")
            emit({"event": "done", **s})
            if a.eval_iid:
                from ecommerce.evaluation.runner import evaluate_agent, oracle_rewards
                from ecommerce.evaluation.splits import get_splits
                from ecommerce.evaluation.metrics import summarize
                from ecommerce.sim.scenarios import generate_scenarios
                sc = generate_scenarios(get_splits(settings)["iid"], a.eval_iid, settings)
                res = evaluate_agent(trainer.load_trained(name, seed, "best", tag), sc, k=1, settings=settings)
                m = summarize(res, k=1, oracle=oracle_rewards(sc, settings))
                emit({"event": "evaluated", "agent": name, "seed": seed, "split": "iid", "scenarios": a.eval_iid,
                      "success_rate": m["success_rate"], "avg_reward": m["avg_reward"], "regret": m["regret"],
                      "violation_rate": m["violation_rate"]})
    return 0


if __name__ == "__main__":
    sys.exit(main())
