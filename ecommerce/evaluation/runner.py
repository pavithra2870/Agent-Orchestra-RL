"""Run agents on scenarios and collect one result dict per episode."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from ecommerce.agents.base import Agent
from ecommerce.agents.baselines import OracleAgent
from ecommerce.config import Settings, get_settings
from ecommerce.env.env import EcomSupportEnv
from ecommerce.env.state import NUM_ACTIONS
from ecommerce.evaluation.evaluator import expected_outcomes
from ecommerce.sim.scenarios import Scenario


def run_episode(
    agent: Agent,
    scenario: Scenario,
    settings: Optional[Settings] = None,
    enforce_mask: Optional[bool] = None,
    run_seed: int = 0,
    failure_multiplier: Optional[float] = None,
    with_trace: bool = False,
    with_scores: bool = False,
    env: Optional[EcomSupportEnv] = None,
) -> Dict[str, Any]:
    settings = settings or get_settings()
    enforce_mask = agent.uses_mask if enforce_mask is None else enforce_mask
    env = env or EcomSupportEnv(settings, enforce_mask=enforce_mask)
    env.enforce_mask = enforce_mask
    options: Dict[str, Any] = {"scenario": scenario, "run_seed": run_seed}
    if failure_multiplier is not None:
        options["failure_multiplier"] = failure_multiplier
    obs, info = env.reset(options=options)
    agent.reset(scenario if agent.needs_scenario else None)

    total, done, scores = 0.0, False, []
    while not done:
        mask = info["action_mask"] if agent.uses_mask else np.ones(NUM_ACTIONS, dtype=bool)
        if with_scores:
            sc = agent.action_scores(obs, mask)
            scores.append(None if sc is None else np.asarray(sc, dtype=float).round(3).tolist())
        a = agent.act(obs, mask, info)
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc

    ev = info["eval"]
    out = {
        "scenario_id": scenario.scenario_id, "split": scenario.split, "type": scenario.scenario_type,
        "combo": "+".join(scenario.true_goals) or "none", "run_seed": run_seed, "agent": agent.name,
        "success": bool(ev["success"]), "reward": total, "steps": info["steps"], "cost": info["cost"],
        "latency": info["latency"], "terminal": ev["terminal"], "expected_terminal": ev["expected_terminal"],
        "goal_completion": ev["goal_completion_rate"], "violations": len(ev["violations"]),
        "blocked": ev["blocked_attempts"], "failure_reasons": ev["failure_reasons"],
        "clar_needed": ev["clarification_needed"], "clar_asked": ev["clarifications_asked"],
        "unneeded_clar": ev["unneeded_clarification"], "unneeded_esc": ev["unneeded_escalation"],
        "tool_failures": ev["tool_failures"],
        # what should happen to a return request (e.g. instant_refund_ok, return_ok, late_damage_report, outside_return_window)
        "return_outcome": (lambda e: f"{e.kind}:{e.reason}")(expected_outcomes(scenario)["return_item"]) if "return_item" in scenario.true_goals else "",
    }
    if with_trace:
        out["trace"] = [t.to_dict() for t in env.trace]
        out["eval"] = ev
        if with_scores:
            out["scores"] = scores
    return out


def evaluate_agent(
    agent: Agent,
    scenarios: Iterable[Scenario],
    k: int = 1,
    settings: Optional[Settings] = None,
    enforce_mask: Optional[bool] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> List[Dict[str, Any]]:
    settings = settings or get_settings()
    enforce = agent.uses_mask if enforce_mask is None else enforce_mask
    env = EcomSupportEnv(settings, enforce_mask=enforce)
    results = []
    for i, sc in enumerate(scenarios):
        for rs in range(k):
            results.append(run_episode(agent, sc, settings, enforce, run_seed=rs, env=env))
        if progress:
            progress(i + 1)
    return results


def oracle_rewards(scenarios: Iterable[Scenario], settings: Optional[Settings] = None) -> Dict[str, float]:
    """Best achievable reward per scenario: the oracle plan with tool failures switched off."""
    settings = settings or get_settings()
    oracle = OracleAgent()
    env = EcomSupportEnv(settings, enforce_mask=True)
    return {sc.scenario_id: run_episode(oracle, sc, settings, failure_multiplier=0.0, env=env)["reward"]
            for sc in scenarios}
