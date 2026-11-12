"""Shared test helpers: hand-built worlds so tests do not depend on random generation."""

from __future__ import annotations

import numpy as np
import pytest

from ecommerce.config import get_settings
from ecommerce.sim.scenarios import scenario_from_world
from ecommerce.sim.world import Customer, Order, World, TODAY


def make_order(order_id="ORD-1111", customer_id="CUST-0001", status="delayed", method="COD", state="pending",
               total=1200.0, delay_days=7, delivered_on=None, out_for_delivery=False) -> Order:
    return Order(order_id=order_id, customer_id=customer_id, items=["Socks (2 pairs)"], status=status,
                 placed_on="2026-10-01", expected_delivery="2026-10-10", delay_days=delay_days,
                 delivered_on=delivered_on, payment_method=method, payment_state=state, total=total,
                 out_for_delivery=out_for_delivery, tracking=[{"date": "2026-10-01", "event": "Order placed"}])


def make_world(orders=None, extra_customer=True) -> World:
    cfg = get_settings().world
    orders = orders or [make_order()]
    customers = {"CUST-0001": Customer("CUST-0001", "Divya Shah", [o.order_id for o in orders if o.customer_id == "CUST-0001"])}
    if extra_customer:
        customers["CUST-0002"] = Customer("CUST-0002", "Arjun Iyer", [o.order_id for o in orders if o.customer_id == "CUST-0002"])
    return World(seed=0, today=TODAY.isoformat(), customers=customers, orders={o.order_id: o for o in orders}, cfg=cfg)


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture
def divya():
    """The example from the project brief: delayed COD order, wants status + switch to UPI."""
    world = make_world([make_order()])
    return scenario_from_world(world, "CUST-0001", "My order is delayed. Check where it is and change COD to UPI",
                               ("delivery_status", "payment_change"), target_method="UPI",
                               order_id_in_text="ORD-1111", scenario_type="standard", scenario_id="divya")


def run_plan(scenario, plan, enforce_mask=True, run_seed=0, failure_multiplier=0.0):
    """Execute a fixed list of action names and return (env, last_info, total_reward)."""
    from ecommerce.env.env import EcomSupportEnv
    from ecommerce.env.state import ACTION_INDEX

    env = EcomSupportEnv(enforce_mask=enforce_mask)
    env.reset(options={"scenario": scenario, "run_seed": run_seed, "failure_multiplier": failure_multiplier})
    total, info = 0.0, {}
    for name in plan:
        _, r, term, trunc, info = env.step(ACTION_INDEX[name])
        total += r
        if term or trunc:
            break
    return env, info, total


def pytest_configure(config):
    """Third-party deprecation notices (starlette test client, torch) are not about our code."""
    for pattern in ("ignore::DeprecationWarning", "ignore::PendingDeprecationWarning", "ignore:.*requires_grad.*:UserWarning",
                    "ignore:Using .httpx. with .starlette.testclient.*"):
        config.addinivalue_line("filterwarnings", pattern)
