"""Evaluate every agent on every test split and write the files the dashboard and report read.

  python -m ecommerce.evaluation.run_eval                 # all agents, k=4 repeats per scenario
  python -m ecommerce.evaluation.run_eval --llm 60        # also benchmark the LLM agent on 60 scenarios

Outputs in ecommerce/results/eval/ :
  summary.json          metrics per agent x split (mean and std over training seeds)
  breakdowns.json       success by scenario type and by goal combination
  training_summary.json sample efficiency (episodes to 90% validation success) and final scores
  curves.json           validation learning curves per method (mean over seeds)
  episodes.csv.gz       one row per episode (first repeat) for failure drill-down
  llm_benchmark.json    (optional) LLM agent vs the others on the same small sample
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ecommerce.agents.baselines import RandomAgent, RuleBasedAgent
from ecommerce.config import RESULTS_DIR, get_settings
from ecommerce.evaluation.metrics import breakdown, summarize
from ecommerce.evaluation.runner import evaluate_agent, oracle_rewards
from ecommerce.evaluation.splits import EVAL_SIZES, get_splits
from ecommerce.sim.scenarios import generate_scenarios
from ecommerce.training import trainer

EVAL_DIR = RESULTS_DIR / "eval"
ENV_SPLITS = ["iid", "heldout_combos", "tool_failures", "missing_info", "out_of_scope", "returns", "policy"]
SEEDS = (0, 1, 2)
K = 4

# key -> (label, kind, family, tag)   kind: baseline | online | offline | ablation
RL_AGENTS = {
    "q_learning": ("Q-learning", "online", "q_learning", ""),
    "sarsa": ("SARSA", "online", "sarsa", ""),
    "dqn": ("DQN", "online", "dqn", ""),
    "ppo": ("PPO", "online", "ppo", ""),
    "cql": ("CQL (offline)", "offline", "cql", ""),
    "dqn_sparse": ("DQN, outcome-only reward", "ablation", "dqn", "sparse"),
    "ppo_sparse": ("PPO, outcome-only reward", "ablation", "ppo", "sparse"),
    "dqn_robust": ("DQN, trained with tool-failure scenarios", "improvement", "dqn", "robust"),
    "ppo_robust": ("PPO, trained with tool-failure scenarios", "improvement", "ppo", "robust"),
    "dqn_nomask": ("DQN, no guardrail", "ablation", "dqn", "nomask"),
    "ppo_nomask": ("PPO, no guardrail", "ablation", "ppo", "nomask"),
    "cql_alpha0": ("CQL without conservative penalty", "ablation", "cql", "alpha0"),
}
BASELINES = {
    "random": ("Random (legal actions)", "baseline"),
    "random_no_guardrail": ("Random, no guardrail", "ablation"),
    "rule_based": ("Rule-based playbook", "baseline"),
}


def _agent_for(key: str, seed: int):
    if key == "random":
        return RandomAgent(masked=True, seed=seed)
    if key == "random_no_guardrail":
        return RandomAgent(masked=False, seed=seed)
    if key == "rule_based":
        return RuleBasedAgent()
    _, _, family, tag = RL_AGENTS[key]
    return trainer.load_trained(family, seed, "best", tag)


def _available(key: str) -> List[int]:
    if key in BASELINES:
        return [0]
    _, _, family, tag = RL_AGENTS[key]
    return [s for s in SEEDS if trainer.model_path(family, s, "best", tag).exists()]


def _mean_std(rows: List[Dict[str, Any]], metric: str) -> Optional[Dict[str, float]]:
    vals = [r[metric] for r in rows if r.get(metric) is not None and not (isinstance(r[metric], float) and np.isnan(r[metric]))]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}


SCALAR_METRICS = ["success_rate", "pass_k", "goal_completion", "violation_rate", "blocked_per_episode", "avg_steps",
                  "avg_cost", "avg_latency", "avg_reward", "regret", "clarify_recall", "clarify_precision",
                  "escalate_recall", "escalate_precision", "unneeded_escalation_rate", "tool_failure_rate"]


def run_main_eval(progress=print) -> Dict[str, Any]:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    settings = get_settings()
    splits = get_splits(settings)
    scenarios = {s: generate_scenarios(splits[s], EVAL_SIZES[s], settings) for s in ENV_SPLITS}
    oracle = {s: oracle_rewards(scenarios[s], settings) for s in ENV_SPLITS}
    out: Dict[str, Any] = {"meta": {"k": K, "sizes": {s: EVAL_SIZES[s] for s in ENV_SPLITS}, "seeds": list(SEEDS),
                                    "splits": ENV_SPLITS, "generated": time.strftime("%Y-%m-%d %H:%M:%S")}, "agents": {}}
    breakdowns: Dict[str, Any] = {}
    episode_rows: List[Dict[str, Any]] = []
    all_keys = list(BASELINES) + list(RL_AGENTS)
    for key in all_keys:
        label, kind = (BASELINES[key] if key in BASELINES else RL_AGENTS[key][:2])
        seeds = _available(key)
        if not seeds:
            progress(f"skip {key}: no trained models")
            continue
        per_seed: Dict[int, Dict[str, Dict[str, Any]]] = {}
        pooled: Dict[str, List[Dict[str, Any]]] = {s: [] for s in ENV_SPLITS}
        for seed in seeds:
            agent = _agent_for(key, seed)
            per_seed[seed] = {}
            for split in ENV_SPLITS:
                res = evaluate_agent(agent, scenarios[split], k=K, settings=settings)
                per_seed[seed][split] = summarize(res, k=K, oracle=oracle[split])
                pooled[split] += res
                episode_rows += [{**{k_: v for k_, v in r.items() if k_ not in ("failure_reasons", "trace")},
                                  "failure_reasons": ";".join(r["failure_reasons"]), "agent_key": key, "seed": seed}
                                 for r in res if r["run_seed"] == 0]
            progress(f"{key} seed {seed}: " + ", ".join(f"{s}={per_seed[seed][s]['success_rate']:.3f}" for s in ENV_SPLITS))
        entry: Dict[str, Any] = {"label": label, "kind": kind, "seeds": seeds, "splits": {}}
        for split in ENV_SPLITS:
            rows = [per_seed[s][split] for s in seeds]
            entry["splits"][split] = {m: _mean_std(rows, m) for m in SCALAR_METRICS}
            entry["splits"][split]["success_ci95"] = rows[0]["success_ci95"] if len(rows) == 1 else [
                float(np.mean([r["success_ci95"][0] for r in rows])), float(np.mean([r["success_ci95"][1] for r in rows]))]
            entry["splits"][split]["failure_reasons"] = {
                reason: float(np.mean([r["failure_reasons"].get(reason, 0.0) for r in rows]))
                for reason in sorted({x for r in rows for x in r["failure_reasons"]})}
        overall_rows = []
        for seed in seeds:
            n_tot = sum(per_seed[seed][s]["episodes"] for s in ENV_SPLITS)
            overall_rows.append({m: sum(per_seed[seed][s][m] * per_seed[seed][s]["episodes"] for s in ENV_SPLITS
                                        if per_seed[seed][s].get(m) is not None and not np.isnan(per_seed[seed][s][m])) / n_tot
                                 for m in SCALAR_METRICS})
        entry["overall"] = {m: _mean_std(overall_rows, m) for m in SCALAR_METRICS}
        out["agents"][key] = entry
        breakdowns[key] = {
            "by_type": breakdown([r for s in ENV_SPLITS for r in pooled[s]], "type"),
            "by_combo": breakdown([r for s in ("iid", "heldout_combos", "returns", "policy") for r in pooled[s]], "combo"),
            "by_return_outcome": breakdown([r for r in pooled["returns"] if r["return_outcome"]], "return_outcome"),
        }
    (EVAL_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    (EVAL_DIR / "breakdowns.json").write_text(json.dumps(breakdowns, indent=2))
    pd.DataFrame(episode_rows).to_csv(EVAL_DIR / "episodes.csv.gz", index=False, compression="gzip")
    return out


def run_training_summary() -> Dict[str, Any]:
    """Sample efficiency and learning curves from the saved training logs."""
    summaries: Dict[str, Any] = {}
    curves: Dict[str, Any] = {}
    for key, (label, kind, family, tag) in RL_AGENTS.items():
        rows, vals = [], []
        for seed in SEEDS:
            d = trainer.run_dir(family, seed, tag)
            if (d / "summary.json").exists():
                rows.append(json.loads((d / "summary.json").read_text()))
                vals.append(pd.read_csv(d / "val_log.csv"))
        if not rows:
            continue
        reached = [r["episodes_to_90pct_val"] for r in rows if r["episodes_to_90pct_val"] is not None]
        steps_to_90 = []
        for r, v in zip(rows, vals):
            hit = v[v["val_success"] >= 0.9]
            if len(hit):
                steps_to_90.append(float(hit.iloc[0]["env_steps"]))
        unit = "gradient steps" if family == "cql" else "episodes"
        summaries[key] = {
            "label": label, "kind": kind, "unit": unit, "seeds": len(rows),
            "best_val_success": _mean_std(rows, "best_val_success"),
            "final_val_success": _mean_std(rows, "final_val_success"),
            "to_90pct": {"reached_in": f"{len(reached)}/{len(rows)} seeds", "mean": float(np.mean(reached)) if reached else None,
                         "env_steps_mean": float(np.mean(steps_to_90)) if steps_to_90 else None},
            "train_seconds": _mean_std(rows, "train_seconds"), "total_budget": rows[0]["episodes"],
        }
        n = min(len(v) for v in vals)
        curves[key] = {"label": label, "unit": unit, "x": vals[0]["episode"].iloc[:n].tolist(),
                       "env_steps": vals[0]["env_steps"].iloc[:n].tolist(),
                       "mean": np.mean([v["val_success"].iloc[:n].to_numpy() for v in vals], axis=0).round(4).tolist(),
                       "std": np.std([v["val_success"].iloc[:n].to_numpy() for v in vals], axis=0).round(4).tolist()}
    (EVAL_DIR / "training_summary.json").write_text(json.dumps(summaries, indent=2))
    (EVAL_DIR / "curves.json").write_text(json.dumps(curves))
    return summaries


def run_llm_benchmark(n: int, progress=print, refresh_others_only: bool = False) -> Dict[str, Any]:
    """refresh_others_only=True keeps the saved LLM results and only recomputes the other agents on the same sample
    (used after retraining, so the slow LLM calls are not repeated)."""
    from ecommerce.agents.llm_agent import LLMAgent
    from ecommerce.language.llm import OllamaClient

    client = OllamaClient()
    if not refresh_others_only and not client.available():
        return {"available": False, "error": client.last_error}
    settings = get_settings()
    splits = get_splits(settings)
    per = {"iid": int(n * 0.5), "heldout_combos": int(n * 0.15), "missing_info": int(n * 0.2)}
    per["out_of_scope"] = n - sum(per.values())
    sample = [s for name, c in per.items() for s in generate_scenarios(splits[name], EVAL_SIZES[name], settings)[:c]]
    orc = oracle_rewards(sample, settings)
    if refresh_others_only:
        out = json.loads((EVAL_DIR / "llm_benchmark.json").read_text())
        out["others"] = {}
    else:
        llm = LLMAgent(client)
        t = time.time()
        res = evaluate_agent(llm, sample, k=1, settings=settings, progress=lambda i: progress(f"llm {i}/{len(sample)}") if i % 10 == 0 else None)
        out = {"available": True, "model": client.model, "scenarios": len(sample), "mix": per,
               "llm": {**summarize(res, k=1, oracle=orc), **llm.stats(), "wall_seconds": time.time() - t}, "others": {}}
        out["llm_failures"] = pd.Series([x for r in res for x in r["failure_reasons"]]).value_counts().to_dict()
    for key in ("rule_based", "random", "dqn", "ppo", "cql", "q_learning", "sarsa"):
        seeds = _available(key)
        if not seeds:
            continue
        r = evaluate_agent(_agent_for(key, seeds[0]), sample, k=1, settings=settings)
        out["others"][key] = summarize(r, k=1, oracle=orc)
    (EVAL_DIR / "llm_benchmark.json").write_text(json.dumps(out, indent=2))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", type=int, default=0)
    ap.add_argument("--skip-main", action="store_true")
    ap.add_argument("--refresh-llm-others", type=int, default=0, help="recompute only the non-LLM rows of the LLM benchmark (give the sample size)")
    a = ap.parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if not a.skip_main:
        t = time.time()
        run_main_eval()
        run_training_summary()
        print(f"main evaluation done in {time.time() - t:.0f}s")
    if a.refresh_llm_others:
        run_llm_benchmark(a.refresh_llm_others, refresh_others_only=True)
        print("refreshed the comparison rows of the LLM benchmark")
    if a.llm:
        r = run_llm_benchmark(a.llm)
        print("llm benchmark:", json.dumps({k: v for k, v in r.items() if k in ("available", "scenarios")}))


if __name__ == "__main__":
    main()
