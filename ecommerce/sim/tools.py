"""The tool layer: what the agent can actually do to the world.

The RL policy only chooses WHICH tool to call. The arguments (customer id, order id,
target payment method) are filled in by the environment from verified context, so the
policy cannot invent or mistype an id.

Every state-changing tool re-checks the business rules itself ("defense in depth"),
so even a bug elsewhere cannot make the database inconsistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from ecommerce.config import Settings, get_settings
from ecommerce.sim import policy, rules
from ecommerce.sim.world import World

DB_TOOLS = (
    "lookup_customer",
    "lookup_order",
    "check_shipment",
    "get_payment_details",
    "check_payment_change_eligibility",
    "update_payment_method",
    "check_refund_eligibility",
    "initiate_refund",
    "check_cancel_eligibility",
    "cancel_order",
    "check_return_eligibility",
    "process_return",
    "lookup_policy",
)
STATE_CHANGING_TOOLS = ("update_payment_method", "initiate_refund", "cancel_order", "process_return")


def order_summaries(world: World, customer_id: str) -> List[Dict[str, Any]]:
    """What a customer can recognise their orders by: id, items and order date (their own account data)."""
    c = world.customers[customer_id]
    return [{"order_id": o, "items": list(world.orders[o].items), "placed_on": world.orders[o].placed_on} for o in c.order_ids]


@dataclass
class ToolResult:
    ok: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    cost: float = 0.0
    latency: float = 0.0
    mutated: bool = False


class ToolBox:
    def __init__(
        self,
        world: World,
        rng: np.random.Generator,
        settings: Optional[Settings] = None,
        failure_multiplier: float = 1.0,
        down_tools: Iterable[str] = (),
    ):
        self.world = world
        self.rng = rng
        self.settings = settings or get_settings()
        self.failure_multiplier = failure_multiplier
        self.down_tools = set(down_tools)
        names = list(self.settings.tools.keys())
        self._cost_factor = {n: float(rng.uniform(0.85, 1.15)) for n in names}
        self._latency_factor = {n: float(rng.uniform(0.85, 1.15)) for n in names}
        self.today = date.fromisoformat(world.today)

    def cost_latency(self, name: str):
        cfg = self.settings.tools[name]
        return cfg.cost * self._cost_factor[name], cfg.latency * self._latency_factor[name]

    def call(self, name: str, **args: Any) -> ToolResult:
        cost, latency = self.cost_latency(name)
        if name in self.down_tools:
            return ToolResult(False, error="service_unavailable", cost=cost, latency=latency)
        p_fail = min(0.9, self.settings.tools[name].failure_prob * self.failure_multiplier)
        if p_fail > 0 and self.rng.random() < p_fail:
            return ToolResult(False, error="timeout", cost=cost, latency=latency)
        handler = getattr(self, f"_{name}")
        result: ToolResult = handler(**args)
        result.cost, result.latency = cost, latency
        return result

    # ---- read-only tools -------------------------------------------------
    def _lookup_customer(self, customer_id: str) -> ToolResult:
        c = self.world.customers.get(customer_id)
        if c is None:
            return ToolResult(False, error="customer_not_found")
        return ToolResult(True, data={"customer_id": c.customer_id, "name": c.name, "order_ids": list(c.order_ids),
                                      "orders": order_summaries(self.world, customer_id)})

    def _owned_order(self, customer_id: str, order_id: str):
        order = self.world.orders.get(order_id)
        if order is None:
            return None, "order_not_found"
        if order.customer_id != customer_id:
            return None, "permission_denied"
        return order, None

    def _lookup_order(self, customer_id: str, order_id: str) -> ToolResult:
        order, err = self._owned_order(customer_id, order_id)
        if err:
            return ToolResult(False, error=err)
        return ToolResult(True, data={"order_id": order.order_id, "items": order.items, "total": order.total,
                                      "placed_on": order.placed_on})

    def _check_shipment(self, customer_id: str, order_id: str) -> ToolResult:
        order, err = self._owned_order(customer_id, order_id)
        if err:
            return ToolResult(False, error=err)
        return ToolResult(True, data={"order_id": order_id, "status": order.status,
                                      "expected_delivery": order.expected_delivery,
                                      "delay_days": order.delay_days, "delivered_on": order.delivered_on,
                                      "tracking": order.tracking})

    def _get_payment_details(self, customer_id: str, order_id: str) -> ToolResult:
        order, err = self._owned_order(customer_id, order_id)
        if err:
            return ToolResult(False, error=err)
        return ToolResult(True, data={"order_id": order_id, "method": order.payment_method,
                                      "state": order.payment_state, "total": order.total})

    def _check_payment_change_eligibility(self, customer_id: str, order_id: str, target_method: str) -> ToolResult:
        order, err = self._owned_order(customer_id, order_id)
        if err:
            return ToolResult(False, error=err)
        e = rules.payment_change_eligibility(order, target_method)
        return ToolResult(True, data={"order_id": order_id, "target_method": target_method,
                                      "status": e.status, "reason": e.reason})

    def _check_refund_eligibility(self, customer_id: str, order_id: str) -> ToolResult:
        order, err = self._owned_order(customer_id, order_id)
        if err:
            return ToolResult(False, error=err)
        e = rules.refund_eligibility(order, self.world.cfg, self.today)
        return ToolResult(True, data={"order_id": order_id, "status": e.status, "reason": e.reason})

    def _check_cancel_eligibility(self, customer_id: str, order_id: str) -> ToolResult:
        order, err = self._owned_order(customer_id, order_id)
        if err:
            return ToolResult(False, error=err)
        e = rules.cancel_eligibility(order)
        return ToolResult(True, data={"order_id": order_id, "status": e.status, "reason": e.reason})

    def _check_return_eligibility(self, customer_id: str, order_id: str, reason: str, kind: str, declared: bool) -> ToolResult:
        order, err = self._owned_order(customer_id, order_id)
        if err:
            return ToolResult(False, error=err)
        e = rules.return_eligibility(order, self.world.customers.get(customer_id), reason, kind, declared, self.world.cfg, self.today)
        return ToolResult(True, data={"order_id": order_id, "status": e.status, "reason": e.reason, "return_reason": reason,
                                      "return_kind": kind, "total": order.total})

    def _lookup_policy(self, topic: str) -> ToolResult:
        """Read-only. Returns the written policy for a topic; touches no order and needs no order id."""
        entry = policy.build_topics(self.settings).get(topic)
        if entry is None:
            return ToolResult(False, error="unknown_topic")
        t, answer = entry
        return ToolResult(True, data={"topic": topic, "title": t.title, "source": t.source, "answer": answer})

    # ---- state-changing tools (re-check the rules before touching the DB) -------
    def _update_payment_method(self, customer_id: str, order_id: str, target_method: str) -> ToolResult:
        order, err = self._owned_order(customer_id, order_id)
        if err:
            return ToolResult(False, error=err)
        if not rules.payment_change_eligibility(order, target_method).allowed:
            return ToolResult(False, error="not_allowed")
        order.payment_method = target_method
        return ToolResult(True, data={"order_id": order_id, "new_method": target_method}, mutated=True)

    def _initiate_refund(self, customer_id: str, order_id: str) -> ToolResult:
        order, err = self._owned_order(customer_id, order_id)
        if err:
            return ToolResult(False, error=err)
        if not rules.refund_eligibility(order, self.world.cfg, self.today).allowed:
            return ToolResult(False, error="not_allowed")
        order.payment_state = "refunded"
        order.refund_id = f"RF-{order_id[-4:]}"
        return ToolResult(True, data={"order_id": order_id, "refund_id": order.refund_id, "amount": order.total},
                          mutated=True)

    def _cancel_order(self, customer_id: str, order_id: str) -> ToolResult:
        order, err = self._owned_order(customer_id, order_id)
        if err:
            return ToolResult(False, error=err)
        if not rules.cancel_eligibility(order).allowed:
            return ToolResult(False, error="not_allowed")
        order.status = "cancelled"
        if order.payment_state == "paid":
            order.payment_state = "refunded"
            order.refund_id = f"RF-{order_id[-4:]}"
        return ToolResult(True, data={"order_id": order_id, "status": "cancelled"}, mutated=True)

    def _process_return(self, customer_id: str, order_id: str, reason: str, kind: str, declared: bool) -> ToolResult:
        """Open the return / exchange, or (small damaged order, refund chosen) refund straight away.

        Nothing is refunded for a normal return here: the refund is released later, by the inspection step."""
        order, err = self._owned_order(customer_id, order_id)
        if err:
            return ToolResult(False, error=err)
        customer = self.world.customers.get(customer_id)
        e = rules.return_eligibility(order, customer, reason, kind, declared, self.world.cfg, self.today)
        if not e.allowed:
            return ToolResult(False, error="not_allowed")
        order.return_kind, order.return_reason, order.return_id = kind, reason, f"RT-{order_id[-4:]}"
        if reason == "damaged":
            customer.claims.append({"order_id": order_id, "date": self.today.isoformat(), "amount": order.total,
                                    "mode": "instant_refund" if e.reason == "instant_refund_ok" else "exchange"})
        if e.reason == "instant_refund_ok":
            order.payment_state, order.refund_id, order.return_state = "refunded", f"RF-{order_id[-4:]}", "instant_refund"
            return ToolResult(True, data={"order_id": order_id, "mode": "instant_refund", "refund_id": order.refund_id,
                                          "amount": order.total, "return_id": order.return_id}, mutated=True)
        order.return_state = "requested"
        return ToolResult(True, data={"order_id": order_id, "mode": "exchange_requested" if kind == "exchange" else "return_requested",
                                      "return_id": order.return_id, "return_kind": kind}, mutated=True)
