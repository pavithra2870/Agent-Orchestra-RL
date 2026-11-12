"""The agent's view of the world (the MDP state) and its vector encoding.

State s_t (what the policy is told):
  - which goals the parser found, which are handled, which need a human
  - what has been verified (customer, order) and what the tools have revealed
  - failures so far per tool, retries, steps and cost used
  - how sure the parser/router was (parse confidence, scope score)
  - the last action taken

The state contains NO ground truth the agent could not know (no hidden eligibility,
no true goals). Facts only appear after a tool call has returned them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from ecommerce.config import Settings
from ecommerce.sim.tools import DB_TOOLS
from ecommerce.sim.world import GOALS, PAYMENT_METHODS, STATUSES

ACTIONS = DB_TOOLS + ("request_clarification", "escalate_to_human", "finish_task")
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)

# tools that need a verified order before they may be used
ORDER_TOOLS = (
    "check_shipment", "get_payment_details", "check_payment_change_eligibility", "update_payment_method",
    "check_refund_eligibility", "initiate_refund", "check_cancel_eligibility", "cancel_order",
    "check_return_eligibility", "process_return",
)

# Which tools a goal needs (first = information step, last = action step)
GOAL_TOOLS = {
    "delivery_status": ("check_shipment",),
    "payment_change": ("check_payment_change_eligibility", "update_payment_method"),
    "refund": ("check_refund_eligibility", "initiate_refund"),
    "cancel_order": ("check_cancel_eligibility", "cancel_order"),
    "return_item": ("check_return_eligibility", "process_return"),
    "policy_info": ("lookup_policy",),
}


@dataclass
class EpisodeState:
    parsed_goals: Set[str]
    parse_conf: float
    scope_score: float
    low_conf_threshold: float
    max_steps: int
    cost_budget: float
    max_clarifications: int
    max_consecutive_failures: int

    customer_verified: bool = False
    customer_order_ids: Optional[List[str]] = None
    order_candidate: Optional[str] = None
    order_ambiguous: bool = False
    order_rejected: bool = False
    order_verified: bool = False

    shipment: Optional[dict] = None
    payment: Optional[dict] = None
    pay_elig: Optional[str] = None
    refund_elig: Optional[str] = None
    cancel_elig: Optional[str] = None
    return_elig: Optional[str] = None
    done: Dict[str, bool] = field(default_factory=lambda: {"payment_change": False, "refund": False, "cancel_order": False,
                                                           "return_item": False, "policy_info": False})

    steps: int = 0
    cost: float = 0.0
    latency: float = 0.0
    clarifications_used: int = 0
    fail_counts: Dict[str, int] = field(default_factory=lambda: {t: 0 for t in DB_TOOLS})
    total_failures: int = 0
    last_action: Optional[int] = None
    last_failed: bool = False
    # False while a payment change was requested but the customer has not named the new method (feeds the action
    # mask and the observation: the agent has to ask). Same idea for a return: reason, exchange-or-refund and the
    # customer's declaration (unused / not damaged by them) all come from the customer, never from the agent.
    method_known: bool = True
    return_info_known: bool = True

    # ---- derived knowledge --------------------------------------------------
    def locked(self, tool: str) -> bool:
        return self.fail_counts.get(tool, 0) >= self.max_consecutive_failures

    def goal_handled(self, goal: str) -> bool:
        if goal == "delivery_status":
            return self.shipment is not None
        if goal == "payment_change":
            return self.done["payment_change"] or self.pay_elig == "ineligible"
        if goal == "refund":
            return self.done["refund"] or self.refund_elig == "ineligible"
        if goal == "return_item":
            return self.done["return_item"] or self.return_elig == "ineligible"
        if goal == "policy_info":
            return self.done["policy_info"]
        return self.done["cancel_order"] or self.cancel_elig == "ineligible"

    def goal_needs_escalation(self, goal: str) -> bool:
        """True when this goal cannot be finished by the agent and a human must take over."""
        if self.goal_handled(goal):
            return False
        tools = GOAL_TOOLS[goal]
        info_tool = tools[0]
        if goal in ("delivery_status", "policy_info"):
            return self.locked(info_tool)
        act_tool = tools[1]
        elig = {"payment_change": self.pay_elig, "refund": self.refund_elig, "cancel_order": self.cancel_elig,
                "return_item": self.return_elig}[goal]
        if goal in ("refund", "return_item") and elig == "needs_human":
            return True
        if elig is None:
            return self.locked(info_tool)
        if elig == "eligible":
            return self.locked(act_tool)
        return False

    def unresolved_order(self) -> bool:
        return self.order_ambiguous or self.order_rejected

    def handled_set(self) -> Set[str]:
        return {g for g in self.parsed_goals if self.goal_handled(g)}


# ---------------------------------------------------------------------------
# Vector encoding
# ---------------------------------------------------------------------------
def _feature_names() -> List[str]:
    names: List[str] = []
    names += [f"requested:{g}" for g in GOALS]
    names += [f"handled:{g}" for g in GOALS]
    names += [f"needs_escalation:{g}" for g in GOALS]
    names += ["customer_verified", "order_known", "order_verified", "order_ambiguous", "order_rejected"]
    names += ["clarifications_used"]
    names += ["shipment_known"] + [f"shipment_status:{s}" for s in STATUSES]
    names += ["payment_known"] + [f"payment_method:{m}" for m in PAYMENT_METHODS]
    names += ["pay_elig_known", "pay_elig_yes", "refund_elig_known", "refund_elig_yes", "refund_elig_human",
              "cancel_elig_known", "cancel_elig_yes", "return_elig_known", "return_elig_yes", "return_elig_human"]
    names += ["payment_method_missing", "return_info_missing"]
    names += ["last_failed", "consecutive_failures", "total_failures"]
    names += ["steps_used", "cost_used"]
    names += ["parse_conf", "scope_score", "low_confidence"]
    names += [f"last_action:{a}" for a in ACTIONS]
    names += [f"fail_count:{t}" for t in DB_TOOLS]
    return names


FEATURES = _feature_names()
FEATURE_INDEX = {n: i for i, n in enumerate(FEATURES)}
OBS_DIM = len(FEATURES)


def encode(s: EpisodeState) -> np.ndarray:
    v = np.zeros(OBS_DIM, dtype=np.float32)

    def put(name: str, value: float) -> None:
        v[FEATURE_INDEX[name]] = value

    for g in GOALS:
        put(f"requested:{g}", float(g in s.parsed_goals))
        put(f"handled:{g}", float(g in s.parsed_goals and s.goal_handled(g)))
        put(f"needs_escalation:{g}", float(g in s.parsed_goals and s.goal_needs_escalation(g)))
    put("customer_verified", float(s.customer_verified))
    put("order_known", float(s.order_candidate is not None))
    put("order_verified", float(s.order_verified))
    put("order_ambiguous", float(s.order_ambiguous))
    put("order_rejected", float(s.order_rejected))
    put("clarifications_used", s.clarifications_used / max(1, s.max_clarifications))
    if s.shipment is not None:
        put("shipment_known", 1.0)
        put(f"shipment_status:{s.shipment['status']}", 1.0)
    if s.payment is not None:
        put("payment_known", 1.0)
        put(f"payment_method:{s.payment['method']}", 1.0)
    if s.pay_elig is not None:
        put("pay_elig_known", 1.0)
        put("pay_elig_yes", float(s.pay_elig == "eligible"))
    if s.refund_elig is not None:
        put("refund_elig_known", 1.0)
        put("refund_elig_yes", float(s.refund_elig == "eligible"))
        put("refund_elig_human", float(s.refund_elig == "needs_human"))
    if s.cancel_elig is not None:
        put("cancel_elig_known", 1.0)
        put("cancel_elig_yes", float(s.cancel_elig == "eligible"))
    if s.return_elig is not None:
        put("return_elig_known", 1.0)
        put("return_elig_yes", float(s.return_elig == "eligible"))
        put("return_elig_human", float(s.return_elig == "needs_human"))
    put("payment_method_missing", float("payment_change" in s.parsed_goals and not s.method_known))
    put("return_info_missing", float("return_item" in s.parsed_goals and not s.return_info_known))
    put("last_failed", float(s.last_failed))
    put("consecutive_failures", max(s.fail_counts.values()) / max(1, s.max_consecutive_failures))
    put("total_failures", min(1.0, s.total_failures / 8.0))
    put("steps_used", s.steps / s.max_steps)
    put("cost_used", min(1.0, s.cost / s.cost_budget))
    put("parse_conf", s.parse_conf)
    put("scope_score", s.scope_score)
    put("low_confidence", float(s.parse_conf < s.low_conf_threshold))
    if s.last_action is not None:
        put(f"last_action:{ACTIONS[s.last_action]}", 1.0)
    for t in DB_TOOLS:
        put(f"fail_count:{t}", min(1.0, s.fail_counts[t] / max(1, s.max_consecutive_failures)))
    return v


def decode(obs: np.ndarray) -> Dict[str, float]:
    """Readable view of an observation vector (used by the rule-based agent and the dashboard)."""
    return {n: float(obs[i]) for i, n in enumerate(FEATURES)}


def state_view(s: EpisodeState) -> Dict[str, object]:
    """Human-readable snapshot of the state for the dashboard (same facts the policy sees)."""
    def goal_status(g: str) -> str:
        if s.goal_handled(g):
            return "handled"
        return "needs_human" if s.goal_needs_escalation(g) else "pending"

    return {
        "goals": {g: goal_status(g) for g in GOALS if g in s.parsed_goals},
        "customer_verified": s.customer_verified,
        "order_id": s.order_candidate,
        "order_verified": s.order_verified,
        "order_unclear": s.order_ambiguous or s.order_rejected,
        "facts": {
            "shipment_status": s.shipment["status"] if s.shipment else None,
            "payment_change_check": s.pay_elig, "refund_check": s.refund_elig, "cancel_check": s.cancel_elig,
            "return_check": s.return_elig, "policy_looked_up": s.done["policy_info"],
        },
        "steps": s.steps, "max_steps": s.max_steps, "cost": round(s.cost, 2), "latency": round(s.latency, 2),
        "parse_conf": round(s.parse_conf, 2), "scope_score": round(s.scope_score, 2),
        "locked_tools": [t for t in DB_TOOLS if s.locked(t)], "clarifications_used": s.clarifications_used,
    }
