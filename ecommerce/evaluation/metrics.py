"""Turn episode results into the metrics we report (see REPORT / dashboard 'Metrics help')."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

CLARIFY_TYPES = ("missing_order", "wrong_order", "low_confidence")


def _bootstrap_ci(per_scenario: np.ndarray, n_boot: int = 1000, seed: int = 0) -> List[float]:
    if len(per_scenario) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_scenario), size=(n_boot, len(per_scenario)))
    means = per_scenario[idx].mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def summarize(results: List[Dict[str, Any]], k: int = 4, oracle: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """Aggregate episode results. `k` is used for pass^k (needs >= k runs per scenario)."""
    if not results:
        return {}
    by_scn: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        by_scn[r["scenario_id"]].append(r)
    per_scn_success = np.array([np.mean([x["success"] for x in v]) for v in by_scn.values()])
    runs_per = min(len(v) for v in by_scn.values())
    kk = min(k, runs_per)
    pass_k = float(np.mean([all(x["success"] for x in sorted(v, key=lambda z: z["run_seed"])[:kk])
                            for v in by_scn.values()]))

    n = len(results)
    mean = lambda key: float(np.mean([r[key] for r in results]))   # noqa: E731
    needed = [r for r in results if r["clar_needed"]]
    asked = [r for r in results if r["clar_asked"] > 0]
    exp_esc = [r for r in results if r["expected_terminal"] == "escalate"]
    did_esc = [r for r in results if r["terminal"] == "escalate"]

    reasons = Counter()
    for r in results:
        for fr in r["failure_reasons"]:
            reasons[fr.split(":")[0] if fr.startswith(("violation", "missed_goal", "missed_escalation")) else fr] += 1

    out: Dict[str, Any] = {
        "episodes": n, "scenarios": len(by_scn), "runs_per_scenario": runs_per,
        "success_rate": float(per_scn_success.mean()), "success_ci95": _bootstrap_ci(per_scn_success),
        f"pass_k": pass_k, "k": kk,
        "goal_completion": mean("goal_completion"),
        "violation_rate": float(np.mean([r["violations"] > 0 for r in results])),
        "violations_per_episode": mean("violations"),
        "blocked_per_episode": mean("blocked"),
        "avg_steps": mean("steps"), "avg_cost": mean("cost"), "avg_latency": mean("latency"),
        "avg_reward": mean("reward"),
        "clarify_recall": _safe_div(sum(r["clar_asked"] > 0 for r in needed), len(needed)),
        "clarify_precision": _safe_div(sum(r["clar_needed"] for r in asked), len(asked)),
        "escalate_recall": _safe_div(sum(r["terminal"] == "escalate" for r in exp_esc), len(exp_esc)),
        "escalate_precision": _safe_div(sum(r["expected_terminal"] == "escalate" for r in did_esc), len(did_esc)),
        "unneeded_escalation_rate": float(np.mean([r["unneeded_esc"] for r in results])),
        "tool_failure_rate": mean("tool_failures"),
        "failure_reasons": {k_: v / n for k_, v in reasons.most_common()},
    }
    if oracle:
        out["regret"] = float(np.mean([oracle[r["scenario_id"]] - r["reward"] for r in results
                                       if r["scenario_id"] in oracle]))
    return out


def breakdown(results: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, float]]:
    """Success rate and count grouped by 'type' or 'combo' (A7: never report only the overall number)."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        groups[str(r[key])].append(r)
    return {g: {"success_rate": float(np.mean([x["success"] for x in v])), "n": len(v)} for g, v in sorted(groups.items())}
