"""Which of the customer's orders does a message mean?  Deterministic code, not the LLM.

Order of evidence (strongest first):
  1. the customer NAMED it: full id, last four digits, "the lost one", an item word, "the latest/oldest order";
  2. the conversation: "the other one", "it / that one" (only with the last few messages as history);
  3. the customer has only ONE order;
  4. INFERENCE from what they want, using the same business rules the tools enforce:
       where is it?    -> orders still on their way
       refund          -> orders the refund rules allow (or that need a human)
       cancel          -> orders still being processed
       change payment  -> orders that are unpaid, open and not out for delivery
     exactly one such order -> assume it (and say so); several -> ask, listing only those;
     none -> ask, and say that none of the orders looks eligible.

Safety: an ASSUMED order is never used for a refund, cancel or payment change without the customer's yes
(that confirmation lives in the orchestrator).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence

from ecommerce.config import WorldCfg
from ecommerce.language.parser import extract_order_id, resolve_order_reference
from ecommerce.sim import rules
from ecommerce.sim.world import Order

STATE_CHANGING_GOALS = ("payment_change", "refund", "cancel_order", "return_item")
ORDER_GOALS_SKIP = ("policy_info",)          # a policy question is about the shop, not about one of the customer's orders
IN_TRANSIT = ("processing", "shipped", "delayed")
ORD_RE = re.compile(r"ORD-\d{4}")
OTHER_RE = re.compile(r"\b(?:the )?other(?: one)?\b|\banother(?: one)?\b|\bthe second(?: one)?\b", re.I)
RECEIVED_RE = re.compile(r"\b(?:already )?(?:received|got|arrived|delivered)\b", re.I)
ANAPHORA_RE = re.compile(r"\b(?:it|that one|this one|the same(?: order| one)?|same order)\b", re.I)
LATEST_RE = re.compile(r"\b(?:latest|newest|most recent|last|recent)\s+(?:order|one|purchase)\b", re.I)
OLDEST_RE = re.compile(r"\b(?:oldest|earliest|first)\s+(?:order|one|purchase)\b", re.I)
YES_RE = re.compile(r"^\s*(?:yes|yeah|yep|yup|sure|ok|okay|correct|right|that'?s (?:it|right|correct)|go ahead|please do|do it|confirm(?:ed)?)\b", re.I)
NO_RE = re.compile(r"^\s*(?:no|nope|nah|not that|wrong|cancel that|don'?t)\b", re.I)


@dataclass
class Resolution:
    order_id: Optional[str]
    how: str                      # named | reference | single_order | history | inferred | ambiguous | none_qualifies | no_goals | no_order_needed
    candidates: List[str] = field(default_factory=list)      # ids to list when asking
    reason: str = ""              # plain-words explanation (shown to the customer and on the dashboard)
    assumed: bool = False         # True only for "inferred"
    confirm_needed: bool = False  # assumed order + a state-changing goal

    def to_dict(self) -> Dict[str, Any]:
        return {"order_id": self.order_id, "how": self.how, "candidates": self.candidates, "reason": self.reason,
                "assumed": self.assumed, "confirm_needed": self.confirm_needed}


def is_yes(text: str) -> bool:
    return bool(YES_RE.match(text)) and not NO_RE.match(text)


def is_no(text: str) -> bool:
    return bool(NO_RE.match(text))


def _qualifies(goal: str, o: Order, target: Optional[str], cfg: WorldCfg, today: date) -> bool:
    if goal == "delivery_status":
        return o.status in IN_TRANSIT
    if goal == "payment_change":
        t = target or ("UPI" if o.payment_method != "UPI" else "CARD")
        return rules.payment_change_eligibility(o, t).allowed
    if goal == "refund":
        return rules.refund_eligibility(o, cfg, today).status in (rules.ELIGIBLE, rules.NEEDS_HUMAN)
    if goal == "cancel_order":
        return rules.cancel_eligibility(o).allowed
    if goal == "return_item":
        # delivered, paid, no return started yet, and still inside the longer of the two windows
        if o.status != "delivered" or not o.delivered_on or o.payment_state != "paid" or o.return_state is not None:
            return False
        return (today - date.fromisoformat(o.delivered_on)).days <= max(cfg.return_window_days, cfg.damage_report_window_days)
    return False


def would_change(goal: str, o: Order, target: Optional[str], cfg: WorldCfg, today: date) -> bool:
    """Would carrying out this goal on this order really change the database? (Used to decide when to ask yes / no.)"""
    if goal == "refund":
        return rules.refund_eligibility(o, cfg, today).status == rules.ELIGIBLE
    if goal == "payment_change":
        return bool(target) and rules.payment_change_eligibility(o, target).allowed
    if goal == "cancel_order":
        return rules.cancel_eligibility(o).allowed
    return False


def _recent_or_lost(o: Order, today: date) -> bool:
    """Fallback for "where is it" when nothing is in transit: a lost order, or one delivered in the last 3 days."""
    if o.status == "lost":
        return True
    return bool(o.status == "delivered" and o.delivered_on and (today - date.fromisoformat(o.delivered_on)).days <= 3)


REASON = {
    "delivery_status": "your only order that is still on its way",
    "refund": "your only order that can be refunded",
    "cancel_order": "your only order that can still be cancelled",
    "payment_change": "your only order whose payment method can still be changed",
    "return_item": "your only delivered order that can still be returned",
}
NONE_REASON = {
    "delivery_status": "none of your orders is still on its way",
    "refund": "none of your orders looks eligible for a refund right now",
    "cancel_order": "none of your orders can still be cancelled (they have already shipped or closed)",
    "payment_change": "none of your orders can have its payment method changed right now",
    "return_item": "none of your orders can be returned right now (it must be delivered, paid, not already in a return, and within the return window)",
}


def _history_ids(history: Sequence[Dict[str, str]], roles: Sequence[str]) -> List[str]:
    out: List[str] = []
    for m in history:
        if m.get("role") in roles:
            out += ORD_RE.findall(m.get("text", "").upper())
    return out


def resolve_order(
    text: str,
    goals: Sequence[str],
    target_method: Optional[str],
    orders: Sequence[Order],
    today: date,
    cfg: WorldCfg,
    history: Sequence[Dict[str, str]] = (),
    exclude: Sequence[str] = (),
) -> Resolution:
    by_id = {o.order_id: o for o in orders}
    order_goals = [g for g in goals if g not in ORDER_GOALS_SKIP]
    if goals and not order_goals:
        return Resolution(None, "no_order_needed", [], "a policy question does not need an order")
    goals = order_goals
    summaries = [{"order_id": o.order_id, "status": o.status, "items": o.items} for o in orders]
    low = text.lower()

    # 1. named by the customer
    explicit = extract_order_id(text)
    if explicit:
        return Resolution(explicit, "named", reason="you named it")
    if OTHER_RE.search(text):
        listed = [i for i in dict.fromkeys(_history_ids(history, ("agent",))) if i in by_id]
        pool = [by_id[i] for i in (listed or list(by_id))]
        if RECEIVED_RE.search(text):
            pool = [o for o in pool if o.status != "delivered"]
        pool = [o for o in pool if o.order_id not in exclude]
        if len(pool) == 1:
            return Resolution(pool[0].order_id, "history", reason="the other order we were talking about")
    ref = resolve_order_reference(text, summaries)
    if ref:
        return Resolution(ref, "reference", reason="matched what you wrote")
    if orders and LATEST_RE.search(text):
        return Resolution(max(orders, key=lambda o: o.placed_on).order_id, "reference", reason="your most recent order")
    if orders and OLDEST_RE.search(text):
        return Resolution(min(orders, key=lambda o: o.placed_on).order_id, "reference", reason="your oldest order")

    # 2. the conversation ("it", "that one")
    if ANAPHORA_RE.search(low) and history:
        seen = [i for i in _history_ids(history, ("agent", "user")) if i in by_id and i not in exclude]
        if seen:
            return Resolution(seen[-1], "history", reason="the order we were just talking about")

    # 3. only one order
    if len(orders) == 1:
        return Resolution(orders[0].order_id, "single_order", reason="your only order")

    all_ids = [o.order_id for o in orders]
    if not orders:
        return Resolution(None, "none_qualifies", [], "there are no orders on your account")
    if not goals:
        return Resolution(None, "no_goals", all_ids, "")

    # 4. inference from what they want
    pool = [o for o in orders if o.order_id not in exclude]

    def matching(goal: str) -> List[Order]:
        hits = [o for o in pool if _qualifies(goal, o, target_method, cfg, today)]
        if goal == "delivery_status" and not hits:
            hits = [o for o in pool if _recent_or_lost(o, today)]
        return hits

    per_goal = {g: matching(g) for g in goals}
    all_goals = [o for o in pool if all(o in per_goal[g] for g in goals)]
    any_goal = [o for o in pool if any(o in per_goal[g] for g in goals)]
    for cands, why in ((all_goals, None), (any_goal, "the only order that fits part of what you asked")):
        if len(cands) == 1:
            reason = why or (REASON[goals[0]] if len(goals) == 1 else "the only order that fits everything you asked")
            confirm = any(g in STATE_CHANGING_GOALS for g in goals)
            return Resolution(cands[0].order_id, "inferred", [cands[0].order_id], reason, assumed=True, confirm_needed=confirm)
        if len(cands) > 1:
            return Resolution(None, "ambiguous", [o.order_id for o in cands],
                              "these are the orders that fit" if len(goals) == 1 else "these orders fit what you asked")
    why = " and ".join(NONE_REASON[g] for g in goals[:2])
    return Resolution(None, "none_qualifies", all_ids, why)
