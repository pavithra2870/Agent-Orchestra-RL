"""What happens AFTER a return is opened: the item is collected and a human supply-chain team inspects it.

This is not something the RL agent does. It is a step of the shop's process, simulated here:
  - the dashboard has "inspection passed / failed" buttons for the demo;
  - evaluation uses a seeded outcome (`seeded_inspection`) when it needs one.

Passed: a refund return releases the refund, an exchange return ships the replacement.
Failed: nothing is refunded or replaced and the item is sent back to the customer.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ecommerce.sim.world import Order

INSPECTION_FAIL_PROB = 0.15            # used only by the seeded outcome


def complete_inspection(order: Order, passed: bool) -> Order:
    """Apply the inspection result to an order whose return is waiting (mutates and returns it)."""
    if order.return_state != "requested":
        raise ValueError(f"{order.order_id} has no return waiting for inspection")
    if not passed:
        order.return_state = "inspection_failed"           # the item goes back to the customer, no refund
        return order
    order.return_state = "inspection_passed"
    if order.return_kind == "refund":
        order.payment_state = "refunded"
        order.refund_id = f"RF-{order.order_id[-4:]}"
    else:
        order.exchange_id = f"EX-{order.order_id[-4:]}"
    return order


def seeded_inspection(order_id: str, seed: int = 0) -> bool:
    """Deterministic pass / fail for an order id (85% pass), for evaluation runs."""
    rng = np.random.default_rng([seed, int(order_id[-4:])])
    return bool(rng.random() >= INSPECTION_FAIL_PROB)
