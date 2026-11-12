"""Guardrail layer 2: action masking (hard rules in code, not learned).

The policy can only pick from the legal actions. A state-changing tool is legal only
when (a) the customer actually asked for it, (b) the order is verified as theirs,
and (c) the eligibility check came back "eligible". Retries are capped: after
`max_consecutive_failures` failures in a row a tool is locked.

The same function is used in three places:
  - the environment (to reject illegal actions),
  - the agents (to avoid choosing them / to mask learning targets),
  - the safety tests (to prove no violation can happen while masking is on).
"""

from __future__ import annotations

import numpy as np

from ecommerce.env.state import ACTION_INDEX, ACTIONS, NUM_ACTIONS, ORDER_TOOLS, EpisodeState


def legal_mask(s: EpisodeState) -> np.ndarray:
    m = np.zeros(NUM_ACTIONS, dtype=bool)

    def allow(tool: str, cond: bool) -> None:
        m[ACTION_INDEX[tool]] = bool(cond) and not s.locked(tool)

    allow("lookup_customer", not s.customer_verified)
    allow("lookup_order", s.customer_verified and not s.order_verified and s.order_candidate is not None
          and not s.order_rejected)

    ov = s.order_verified
    allow("check_shipment", ov and s.shipment is None)
    allow("get_payment_details", ov and s.payment is None)
    # A payment change needs the NEW method to be named by the customer; the agent may never pick one itself.
    allow("check_payment_change_eligibility", ov and s.pay_elig is None and "payment_change" in s.parsed_goals
          and s.method_known)
    allow("update_payment_method", ov and "payment_change" in s.parsed_goals and s.pay_elig == "eligible"
          and not s.done["payment_change"] and s.method_known)
    allow("check_refund_eligibility", ov and s.refund_elig is None)
    allow("initiate_refund", ov and "refund" in s.parsed_goals and s.refund_elig == "eligible"
          and not s.done["refund"])
    allow("check_cancel_eligibility", ov and s.cancel_elig is None)
    allow("cancel_order", ov and "cancel_order" in s.parsed_goals and s.cancel_elig == "eligible"
          and not s.done["cancel_order"])
    # A return needs the customer's answers (reason, exchange or refund, declaration); the agent has to ask for them.
    allow("check_return_eligibility", ov and s.return_elig is None and "return_item" in s.parsed_goals and s.return_info_known)
    allow("process_return", ov and "return_item" in s.parsed_goals and s.return_elig == "eligible"
          and not s.done["return_item"] and s.return_info_known)
    # A policy lookup needs no order and changes nothing.
    allow("lookup_policy", "policy_info" in s.parsed_goals and not s.done["policy_info"])

    m[ACTION_INDEX["request_clarification"]] = s.clarifications_used < s.max_clarifications
    m[ACTION_INDEX["escalate_to_human"]] = True
    m[ACTION_INDEX["finish_task"]] = True
    return m


def explain_blocked(s: EpisodeState, action: str) -> str:
    """Plain-language reason an action is not allowed (shown on the dashboard)."""
    if s.locked(action):
        return f"{action} failed {s.max_consecutive_failures} times in a row, so it is locked (retry limit)"
    if action == "lookup_order":
        if not s.customer_verified:
            return "look up the customer first"
        if s.order_candidate is None or s.order_rejected:
            return "no verified order id yet; the customer must tell us which order"
    if action in ("check_payment_change_eligibility", "update_payment_method") and not s.method_known:
        return "the customer has not said which payment method they want, so the agent must ask"
    if action in ("check_return_eligibility", "process_return") and not s.return_info_known:
        return "the customer has not yet said why, whether they want an exchange or a refund, or confirmed the condition of the item, so the agent must ask"
    if action == "lookup_policy":
        return "the customer did not ask a policy question, or it was already answered"
    if action in ("update_payment_method", "initiate_refund", "cancel_order", "process_return"):
        goal = {"update_payment_method": "payment_change", "initiate_refund": "refund",
                "cancel_order": "cancel_order", "process_return": "return_item"}[action]
        if goal not in s.parsed_goals:
            return "the customer did not ask for this"
        if not s.order_verified:
            return "the order is not verified as belonging to this customer"
        return "the eligibility check has not passed yet"
    if action in ACTIONS and action not in ("request_clarification", "escalate_to_human", "finish_task"):
        if not s.order_verified:
            return "the order is not verified as belonging to this customer"
        return "this information is already known or not applicable"
    if action == "request_clarification":
        return "the clarification limit was reached"
    return "not allowed in this state"


_MUTATION_GOAL = {"update_payment_method": "payment_change", "initiate_refund": "refund", "cancel_order": "cancel_order",
                  "process_return": "return_item"}


def safety_ok(s: EpisodeState, action: str) -> bool:
    """The real safety rules only (not the efficiency rules in the mask).

    - any order-level tool needs a verified order;
    - a state-changing tool also needs the customer to have asked for it and an "eligible" check result.
    """
    if action in ORDER_TOOLS and not s.order_verified:
        return False
    if action in _MUTATION_GOAL:
        goal = _MUTATION_GOAL[action]
        elig = {"payment_change": s.pay_elig, "refund": s.refund_elig, "cancel_order": s.cancel_elig,
                "return_item": s.return_elig}[goal]
        known = (s.method_known if goal == "payment_change" else s.return_info_known if goal == "return_item" else True)
        return goal in s.parsed_goals and elig == "eligible" and known
    if action in ("check_payment_change_eligibility",) and not s.method_known:
        return False
    if action == "check_return_eligibility" and not s.return_info_known:
        return False
    return True
