"""Seeded fake e-commerce database: customers, orders, shipments, payments.

Everything is generated from a seed, so the same seed always gives the same world.
The world is the single source of truth: tools read and change it, and the
independent evaluator compares its final state with what should have happened.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

from ecommerce.config import WorldCfg, get_settings

GOALS = ("delivery_status", "payment_change", "refund", "cancel_order", "return_item", "policy_info")
RETURN_REASONS = ("damaged", "change_of_mind")
RETURN_KINDS = ("exchange", "refund")
# return_state: None | requested (waiting for the supply-chain team) | inspection_passed | inspection_failed | instant_refund
STATUSES = ("processing", "shipped", "delayed", "delivered", "returned", "lost", "cancelled")
PAYMENT_METHODS = ("COD", "UPI", "CARD")
PAYMENT_STATES = ("pending", "paid", "refunded")
TODAY = date(2026, 10, 17)

FIRST_NAMES = ["Divya", "Arjun", "Meera", "Rohan", "Isha", "Karan", "Anika", "Vikram", "Sana", "Nikhil",
               "Priya", "Aditya", "Neha", "Rahul", "Tara", "Imran", "Lakshmi", "Manish", "Pooja", "Sameer"]
LAST_NAMES = ["Shah", "Iyer", "Nair", "Khan", "Rao", "Das", "Mehta", "Gupta", "Bose", "Patel", "Singh", "Reddy"]
CATALOG = ["Socks (2 pairs)", "Running shoes", "Phone case", "Water bottle", "Backpack", "Desk lamp",
           "Bluetooth speaker", "Notebook set", "Yoga mat", "T-shirt", "Coffee mug", "Wireless mouse",
           "Table fan", "Kitchen knife set", "Bedsheet", "Headphones"]


@dataclass
class Order:
    order_id: str
    customer_id: str
    items: List[str]
    status: str
    placed_on: str
    expected_delivery: str
    delay_days: int
    delivered_on: Optional[str]
    payment_method: str
    payment_state: str
    total: float
    out_for_delivery: bool
    tracking: List[Dict[str, str]]
    refund_id: Optional[str] = None
    return_state: Optional[str] = None
    return_kind: Optional[str] = None
    return_reason: Optional[str] = None
    return_id: Optional[str] = None
    exchange_id: Optional[str] = None


@dataclass
class Customer:
    customer_id: str
    name: str
    order_ids: List[str] = field(default_factory=list)
    claims: List[Dict[str, Any]] = field(default_factory=list)      # damage claims: {order_id, date, amount, mode}


@dataclass
class World:
    seed: int
    today: str
    customers: Dict[str, Customer]
    orders: Dict[str, Order]
    cfg: WorldCfg

    def copy(self) -> "World":
        return copy.deepcopy(self)

    def snapshot(self) -> Dict[str, Dict[str, object]]:
        """The mutable parts of every order, used to show before/after differences."""
        return {
            oid: {
                "status": o.status,
                "payment_method": o.payment_method,
                "payment_state": o.payment_state,
                "refund_id": o.refund_id,
                "return_state": o.return_state,
                "return_kind": o.return_kind,
                "exchange_id": o.exchange_id,
            }
            for oid, o in self.orders.items()
        }


_INITIAL_STATUS_P = np.array([0.22, 0.20, 0.18, 0.25, 0.07, 0.05])
_INITIAL_STATUS_P = _INITIAL_STATUS_P / _INITIAL_STATUS_P.sum()


def _iso(d: date) -> str:
    return d.isoformat()


def _make_order(rng: np.random.Generator, order_id: str, customer_id: str) -> Order:
    status = str(rng.choice(STATUSES[:6], p=_INITIAL_STATUS_P))
    method = str(rng.choice(PAYMENT_METHODS, p=[0.45, 0.30, 0.25]))
    n_items = int(rng.integers(1, 4))
    items = [str(x) for x in rng.choice(CATALOG, size=n_items, replace=False)]
    total = float(np.clip(round(math.exp(rng.normal(7.4, 0.9))), 150, 20000))

    delay_days, delivered_on, out_for_delivery = 0, None, False
    if status == "processing":
        placed = TODAY - timedelta(days=int(rng.integers(0, 3)))
        expected = placed + timedelta(days=5)
    elif status == "shipped":
        placed = TODAY - timedelta(days=int(rng.integers(2, 7)))
        expected = TODAY + timedelta(days=int(rng.integers(0, 4)))
        out_for_delivery = bool(rng.random() < 0.35)
    elif status == "delayed":
        delay_days = int(rng.integers(1, 13))
        expected = TODAY - timedelta(days=delay_days)
        placed = expected - timedelta(days=int(rng.integers(4, 8)))
        out_for_delivery = bool(rng.random() < 0.15)
    elif status == "delivered":
        delivered = TODAY - timedelta(days=int(rng.integers(0, 15)))
        delivered_on = _iso(delivered)
        expected = delivered
        placed = delivered - timedelta(days=int(rng.integers(3, 8)))
    elif status == "returned":
        delivered = TODAY - timedelta(days=int(rng.integers(8, 21)))
        delivered_on = _iso(delivered)
        expected = delivered
        placed = delivered - timedelta(days=int(rng.integers(3, 8)))
    else:  # lost
        expected = TODAY - timedelta(days=6)
        placed = expected - timedelta(days=int(rng.integers(4, 8)))
        delay_days = 6

    if method == "COD":
        payment_state = "paid" if status in ("delivered", "returned") else "pending"
    else:
        payment_state = "paid"

    tracking = [{"date": _iso(placed), "event": "Order placed"}]
    if status in ("shipped", "delayed", "delivered", "returned", "lost"):
        tracking.append({"date": _iso(placed + timedelta(days=1)), "event": "Shipped from warehouse"})
    if status == "delayed":
        tracking.append({"date": _iso(TODAY - timedelta(days=1)), "event": "Delayed at regional hub"})
    if status == "lost":
        tracking.append({"date": _iso(TODAY - timedelta(days=3)), "event": "Package missing at sorting centre"})
    if status in ("delivered", "returned") and delivered_on:
        tracking.append({"date": delivered_on, "event": "Delivered"})
    if status == "returned":
        tracking.append({"date": _iso(TODAY - timedelta(days=3)), "event": "Return received"})

    return Order(
        order_id=order_id, customer_id=customer_id, items=items, status=status,
        placed_on=_iso(placed), expected_delivery=_iso(expected), delay_days=delay_days,
        delivered_on=delivered_on, payment_method=method, payment_state=payment_state,
        total=total, out_for_delivery=out_for_delivery, tracking=tracking,
    )


def generate_world(seed: int, cfg: Optional[WorldCfg] = None) -> World:
    cfg = cfg or get_settings().world
    rng = np.random.default_rng(seed)
    customers: Dict[str, Customer] = {}
    orders: Dict[str, Order] = {}
    used_ids = set()
    names = set()
    for i in range(cfg.n_customers):
        cid = f"CUST-{i + 1:04d}"
        while True:
            name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
            if name not in names:
                names.add(name)
                break
        customers[cid] = Customer(customer_id=cid, name=name)
        n_orders = int(rng.integers(1, cfg.max_orders_per_customer + 1))
        for _ in range(n_orders):
            while True:
                oid = f"ORD-{int(rng.integers(1000, 10000))}"
                if oid not in used_ids:
                    used_ids.add(oid)
                    break
            orders[oid] = _make_order(rng, oid, cid)
            customers[cid].order_ids.append(oid)
    # Past damage claims. A separate random stream, so adding them never changes the orders generated above.
    crng = np.random.default_rng([seed, 99])
    for c in customers.values():
        u = crng.random()
        n_claims = 2 if u < 0.12 else (1 if u < 0.27 else 0)
        for k in range(n_claims):
            c.claims.append({"order_id": f"ORD-{int(crng.integers(1000, 10000))}",
                             "date": _iso(TODAY - timedelta(days=int(crng.integers(1, 26)))),
                             "amount": float(int(crng.integers(80, 480))), "mode": "instant_refund"})
    return World(seed=seed, today=_iso(TODAY), customers=customers, orders=orders, cfg=cfg)


# ---------------------------------------------------------------------------------------------------------------
# Demo-only customers for trying the return / damage flows by hand. Added to the DEMO world only (never to the
# worlds used for training or evaluation), so nothing measured depends on them. Order ids and names are fixed.
# ---------------------------------------------------------------------------------------------------------------
def _delivered_order(order_id: str, customer_id: str, items: List[str], total: float, days_ago: int, method: str = "UPI") -> Order:
    d = TODAY - timedelta(days=days_ago)
    return Order(order_id=order_id, customer_id=customer_id, items=items, status="delivered",
                 placed_on=_iso(d - timedelta(days=4)), expected_delivery=_iso(d), delay_days=0, delivered_on=_iso(d),
                 payment_method=method, payment_state="paid", total=total, out_for_delivery=False,
                 tracking=[{"date": _iso(d - timedelta(days=4)), "event": "Order placed"},
                           {"date": _iso(d - timedelta(days=3)), "event": "Shipped from warehouse"}, {"date": _iso(d), "event": "Delivered"}])


def add_showcase_customers(world: World) -> World:
    claim = lambda n: {"order_id": f"ORD-{8000 + n}", "date": _iso(TODAY - timedelta(days=3 * n + 2)), "amount": 150.0, "mode": "instant_refund"}   # noqa: E731
    spec = [
        ("CUST-0061", "Ishaan Verma", [_delivered_order("ORD-9001", "CUST-0061", ["Face cream"], 180.0, 1)], []),
        ("CUST-0062", "Tanvi Kulkarni", [_delivered_order("ORD-9002", "CUST-0062", ["Bluetooth speaker", "Headphones"], 6400.0, 1, "CARD")], []),
        ("CUST-0063", "Rishi Pillai", [_delivered_order("ORD-9003", "CUST-0063", ["T-shirt"], 650.0, 4)], []),
        ("CUST-0064", "Zoya Ansari", [_delivered_order("ORD-9004", "CUST-0064", ["Coffee mug"], 220.0, 1)], [claim(1), claim(2)]),
        ("CUST-0065", "Dev Malhotra", [_delivered_order("ORD-9005", "CUST-0065", ["Notebook set"], 300.0, 12)], []),
    ]
    for cid, name, orders, claims in spec:
        world.customers[cid] = Customer(customer_id=cid, name=name, order_ids=[o.order_id for o in orders], claims=claims)
        for o in orders:
            world.orders[o.order_id] = o
    return world
