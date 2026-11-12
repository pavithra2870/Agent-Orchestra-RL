"""Non-learning agents: random (lower bound), rule-based (hand-written), oracle (upper bound)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ecommerce.agents.base import Agent
from ecommerce.env.state import ACTION_INDEX, NUM_ACTIONS, decode
from ecommerce.sim.scenarios import Scenario
from ecommerce.sim.world import GOALS

I = ACTION_INDEX


class RandomAgent(Agent):
    """Uniform random. masked=True picks only legal actions; masked=False ignores the guardrail (ablation)."""

    def __init__(self, masked: bool = True, seed: int = 0):
        self.masked = masked
        self.uses_mask = masked
        self.name = "random" if masked else "random_no_guardrail"
        self._rng = np.random.default_rng(seed)

    def act(self, obs, mask, info=None) -> int:
        if self.masked:
            return int(self._rng.choice(np.flatnonzero(mask)))
        return int(self._rng.integers(0, NUM_ACTIONS))


class RuleBasedAgent(Agent):
    """Hand-written support playbook. Uses only what the state tells it (no ground truth)."""

    name = "rule_based"

    def __init__(self, scope_threshold: float = 0.38):
        self.scope_threshold = scope_threshold

    def act(self, obs, mask, info=None) -> int:
        f = decode(obs)
        can = lambda tool: bool(mask[I[tool]])   # noqa: E731
        escalate, finish, clarify = I["escalate_to_human"], I["finish_task"], I["request_clarification"]
        requested = [g for g in GOALS if f[f"requested:{g}"] > 0.5]

        if not requested and f["scope_score"] < self.scope_threshold:
            return escalate
        if f["low_confidence"] > 0.5 and f["clarifications_used"] == 0 and can("request_clarification"):
            return clarify
        if not requested:
            return clarify if can("request_clarification") else escalate
        # A policy question needs no order: look the policy up first. If only a policy was asked, we are done after it.
        if "policy_info" in requested:
            if f["handled:policy_info"] < 0.5 and f["needs_escalation:policy_info"] < 0.5 and can("lookup_policy"):
                return I["lookup_policy"]
        policy_stuck = f["needs_escalation:policy_info"] > 0.5
        requested = [g for g in requested if g != "policy_info"]
        if not requested:
            return escalate if policy_stuck else finish
        if f["customer_verified"] < 0.5:
            return I["lookup_customer"] if can("lookup_customer") else escalate
        if f["order_rejected"] > 0.5 or f["order_ambiguous"] > 0.5:
            return clarify if can("request_clarification") else escalate
        if f["order_verified"] < 0.5:
            return I["lookup_order"] if can("lookup_order") else escalate

        needs_human = policy_stuck
        for g in requested:
            if f[f"handled:{g}"] > 0.5:
                continue
            if f[f"needs_escalation:{g}"] > 0.5:
                needs_human = True
                continue
            if g == "payment_change" and f["payment_method_missing"] > 0.5:
                return clarify if can("request_clarification") else escalate
            if g == "return_item" and f["return_info_missing"] > 0.5:
                return clarify if can("request_clarification") else escalate        # reason, exchange or refund, declaration
            tool = self._next_tool(g, f)
            if tool and can(tool):
                return I[tool]
            needs_human = True
        if needs_human or any(f[f"needs_escalation:{g}"] > 0.5 for g in requested):
            return escalate
        return finish

    @staticmethod
    def _next_tool(goal: str, f: Dict[str, float]) -> Optional[str]:
        if goal == "delivery_status":
            return "check_shipment"
        if goal == "payment_change":
            if f["pay_elig_known"] < 0.5:
                return "check_payment_change_eligibility"
            return "update_payment_method" if f["pay_elig_yes"] > 0.5 else None
        if goal == "refund":
            if f["refund_elig_known"] < 0.5:
                return "check_refund_eligibility"
            return "initiate_refund" if f["refund_elig_yes"] > 0.5 else None
        if goal == "return_item":
            if f["return_elig_known"] < 0.5:
                return "check_return_eligibility"
            return "process_return" if f["return_elig_yes"] > 0.5 else None
        if f["cancel_elig_known"] < 0.5:
            return "check_cancel_eligibility"
        return "cancel_order" if f["cancel_elig_yes"] > 0.5 else None


class OracleAgent(Agent):
    """Knows the hidden truth and follows the cheapest correct plan. Used only to measure regret."""

    name = "oracle"
    needs_scenario = True

    def __init__(self):
        self._plan: List[str] = []
        self._i = 0

    def reset(self, scenario: Scenario = None) -> None:
        from ecommerce.evaluation.evaluator import expected_outcomes, expected_terminal

        sc = scenario
        exp = expected_outcomes(sc)
        plan: List[str] = []
        if sc.scenario_type == "out_of_scope":
            self._plan, self._i = ["escalate_to_human"], 0
            return
        order_goals = [g for g in sc.true_goals if g != "policy_info"]
        if sc.scenario_type in ("missing_order", "wrong_order", "low_confidence") or                 ("return_item" in sc.true_goals and not sc.return_info_given):
            if order_goals:
                plan.append("request_clarification")
        if "policy_info" in sc.true_goals and exp["policy_info"].reason != "tool_down":
            plan.append("lookup_policy")          # after any clarification: a goal the parser missed only becomes legal then
        if order_goals:
            plan += ["lookup_customer", "lookup_order"]
        for g in sc.true_goals:
            e = exp[g]
            if g == "policy_info":
                continue
            if g == "return_item":
                if e.reason != "tool_down":
                    plan.append("check_return_eligibility")
                    if e.kind == "done":
                        plan.append("process_return")
            elif g == "delivery_status":
                if e.kind == "done":
                    plan.append("check_shipment")
            elif g == "payment_change":
                if e.reason != "tool_down":
                    plan.append("check_payment_change_eligibility")
                    if e.kind == "done":
                        plan.append("update_payment_method")
            elif g == "refund":
                if not (e.kind == "escalate" and e.reason == "tool_down"):
                    plan.append("check_refund_eligibility")
                    if e.kind == "done":
                        plan.append("initiate_refund")
            elif g == "cancel_order":
                if e.reason != "tool_down":
                    plan.append("check_cancel_eligibility")
                    if e.kind == "done":
                        plan.append("cancel_order")
        plan.append("escalate_to_human" if expected_terminal(sc, exp) == "escalate" else "finish_task")
        self._plan, self._i = plan, 0

    def act(self, obs, mask, info=None) -> int:
        while self._i < len(self._plan):
            a = I[self._plan[self._i]]
            self._i += 1
            if mask[a]:
                return a
        return I["finish_task"]
