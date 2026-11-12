"""Business rules. Pure functions: (order, config) -> decision. No randomness, no side effects.

These are the "policies of the shop". Tools use them to answer eligibility questions,
and the evaluator uses them to work out what SHOULD have happened.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

from ecommerce.config import WorldCfg
from ecommerce.sim.world import Customer, Order

ELIGIBLE = "eligible"
INELIGIBLE = "ineligible"
NEEDS_HUMAN = "needs_human"


@dataclass(frozen=True)
class Eligibility:
    status: str          # eligible | ineligible | needs_human
    reason: str

    @property
    def allowed(self) -> bool:
        return self.status == ELIGIBLE


def payment_change_eligibility(order: Order, target_method: str) -> Eligibility:
    if order.status in ("delivered", "returned", "lost", "cancelled"):
        return Eligibility(INELIGIBLE, "order_closed")
    if order.payment_state != "pending":
        return Eligibility(INELIGIBLE, "already_paid")
    if target_method == order.payment_method:
        return Eligibility(INELIGIBLE, "same_method")
    if order.out_for_delivery:
        return Eligibility(INELIGIBLE, "out_for_delivery")
    return Eligibility(ELIGIBLE, "ok")


def refund_eligibility(order: Order, cfg: WorldCfg, today: date) -> Eligibility:
    if order.payment_state == "refunded":
        return Eligibility(INELIGIBLE, "already_refunded")
    if order.payment_state != "paid":
        return Eligibility(INELIGIBLE, "not_paid")
    if order.return_state == "requested":
        return Eligibility(INELIGIBLE, "return_in_progress")
    if order.return_state == "inspection_failed":
        return Eligibility(INELIGIBLE, "inspection_failed")
    qualifies = False
    if order.status in ("lost", "returned"):
        qualifies = True
    elif order.status == "delivered":
        # A delivered item is never refunded straight away: the customer starts a return (or reports damage) and the
        # refund follows once the returned item has passed inspection (see sim/inspection.py).
        return Eligibility(INELIGIBLE, "return_required")
    elif order.status == "delayed":
        qualifies = order.delay_days >= cfg.delay_refund_days
    if not qualifies:
        return Eligibility(INELIGIBLE, "outside_policy")
    if order.total > cfg.high_value_threshold:
        return Eligibility(NEEDS_HUMAN, "high_value_needs_approval")
    return Eligibility(ELIGIBLE, "ok")


def cancel_eligibility(order: Order) -> Eligibility:
    if order.status == "processing":
        return Eligibility(ELIGIBLE, "ok")
    if order.status == "cancelled":
        return Eligibility(INELIGIBLE, "already_cancelled")
    return Eligibility(INELIGIBLE, "already_shipped_or_closed")


# ---------------------------------------------------------------------------------------------
# Returns, exchanges and damage claims
# ---------------------------------------------------------------------------------------------
# Eligibility.reason codes for return_eligibility:
#   eligible    instant_refund_ok   damaged, small order, refund chosen: refund now, subject to review
#               return_ok           the customer sends the item back; refund after inspection passes
#               exchange_ok         the customer sends the item back; replacement after inspection passes
#   needs_human late_damage_report | above_instant_limit | claim_history | high_value_needs_approval
#   ineligible  not_delivered | already_claimed | not_paid | already_refunded | declaration_not_confirmed |
#               outside_return_window
def recent_claims(customer: Optional[Customer], cfg: WorldCfg, today: date) -> int:
    if customer is None:
        return 0
    since = today - timedelta(days=cfg.claims_period_days)
    return sum(1 for c in customer.claims if date.fromisoformat(c["date"]) >= since)


def return_eligibility(order: Order, customer: Optional[Customer], reason: str, kind: str, declared: bool,
                       cfg: WorldCfg, today: date) -> Eligibility:
    """Can this order be returned / exchanged / claimed as damaged, and does a human need to look at it?

    `reason` is "damaged" or "change_of_mind", `kind` is "exchange" or "refund", `declared` is the customer's answer to
    "is it unused and in its original packaging?" (change of mind) or "did it arrive damaged and you did not damage
    it?" (damaged). Only the customer's own words give these three; the agent never invents them."""
    if order.status != "delivered" or not order.delivered_on:
        return Eligibility(INELIGIBLE, "not_delivered")            # only courier-confirmed deliveries
    if order.payment_state == "refunded":
        return Eligibility(INELIGIBLE, "already_refunded")
    if order.payment_state != "paid":
        return Eligibility(INELIGIBLE, "not_paid")
    if order.return_state is not None:
        return Eligibility(INELIGIBLE, "already_claimed")          # one claim per order
    if not declared:
        return Eligibility(INELIGIBLE, "declaration_not_confirmed")
    days = (today - date.fromisoformat(order.delivered_on)).days
    if reason == "damaged":
        if days > cfg.damage_report_window_days:
            return Eligibility(NEEDS_HUMAN, "late_damage_report")
        if order.total > cfg.instant_refund_cap:
            return Eligibility(NEEDS_HUMAN, "above_instant_limit")
        if recent_claims(customer, cfg, today) >= cfg.damage_claims_limit:
            return Eligibility(NEEDS_HUMAN, "claim_history")
        return Eligibility(ELIGIBLE, "instant_refund_ok" if kind == "refund" else "exchange_ok")
    if days > cfg.return_window_days:
        return Eligibility(INELIGIBLE, "outside_return_window")
    if order.total > cfg.high_value_threshold:
        return Eligibility(NEEDS_HUMAN, "high_value_needs_approval")
    return Eligibility(ELIGIBLE, "return_ok" if kind == "refund" else "exchange_ok")
