"""World, rules, tools and the scenario generator."""

import numpy as np
import pytest

from ecommerce.config import get_settings
from ecommerce.evaluation.splits import get_splits
from ecommerce.sim import rules
from ecommerce.sim.scenarios import ALL_COMBOS, HELDOUT_COMBOS, TRAIN_COMBOS, generate_scenario, generate_scenarios
from ecommerce.sim.tools import ToolBox
from ecommerce.sim.world import generate_world
from ecommerce.tests.conftest import make_order, make_world
from datetime import date

CFG = get_settings().world
TODAY = date(2026, 10, 17)


def test_world_is_deterministic_and_consistent():
    a, b = generate_world(7), generate_world(7)
    assert a.snapshot() == b.snapshot()
    assert generate_world(8).snapshot() != a.snapshot()
    for o in a.orders.values():
        assert o.order_id in a.customers[o.customer_id].order_ids
        if o.payment_method != "COD":
            assert o.payment_state == "paid"
        if o.status in ("delivered", "returned"):
            assert o.delivered_on is not None


@pytest.mark.parametrize("order_kwargs,target,expected", [
    (dict(status="delayed", method="COD", state="pending"), "UPI", "eligible"),
    (dict(status="processing", method="COD", state="pending"), "CARD", "eligible"),
    (dict(status="delayed", method="UPI", state="paid"), "CARD", "ineligible"),          # already paid
    (dict(status="delivered", method="COD", state="paid", delivered_on="2026-10-15"), "UPI", "ineligible"),
    (dict(status="shipped", method="COD", state="pending", out_for_delivery=True), "UPI", "ineligible"),
    (dict(status="shipped", method="COD", state="pending"), "COD", "ineligible"),         # same method
])
def test_payment_change_rules(order_kwargs, target, expected):
    assert rules.payment_change_eligibility(make_order(**order_kwargs), target).status == expected


@pytest.mark.parametrize("order_kwargs,expected", [
    (dict(status="lost", method="UPI", state="paid"), "eligible"),
    (dict(status="delivered", method="UPI", state="paid", delivered_on="2026-10-15"), "ineligible"),     # delivered: return first
    (dict(status="delivered", method="UPI", state="paid", delivered_on="2026-09-20"), "ineligible"),     # 27 days
    (dict(status="delayed", method="UPI", state="paid", delay_days=6), "eligible"),
    (dict(status="delayed", method="UPI", state="paid", delay_days=2), "ineligible"),
    (dict(status="delayed", method="COD", state="pending", delay_days=9), "ineligible"),                 # nothing paid
    (dict(status="lost", method="UPI", state="paid", total=9000.0), "needs_human"),                       # high value
    (dict(status="lost", method="UPI", state="refunded"), "ineligible"),
])
def test_refund_rules(order_kwargs, expected):
    assert rules.refund_eligibility(make_order(**order_kwargs), CFG, TODAY).status == expected


def test_cancel_rules():
    assert rules.cancel_eligibility(make_order(status="processing")).allowed
    assert not rules.cancel_eligibility(make_order(status="shipped")).allowed
    assert not rules.cancel_eligibility(make_order(status="cancelled")).allowed


def test_tools_enforce_ownership_and_rules():
    w = make_world([make_order(), make_order("ORD-2222", "CUST-0002")])
    tb = ToolBox(w, np.random.default_rng(0), failure_multiplier=0.0)
    assert tb.call("lookup_order", customer_id="CUST-0001", order_id="ORD-2222").error == "permission_denied"
    assert tb.call("lookup_order", customer_id="CUST-0001", order_id="ORD-9999").error == "order_not_found"
    assert tb.call("lookup_order", customer_id="CUST-0001", order_id="ORD-1111").ok
    # defense in depth: update refuses an ineligible change even when called directly
    w.orders["ORD-1111"].payment_state = "paid"
    r = tb.call("update_payment_method", customer_id="CUST-0001", order_id="ORD-1111", target_method="UPI")
    assert not r.ok and r.error == "not_allowed" and w.orders["ORD-1111"].payment_method == "COD"


def test_tool_failures_are_seeded_and_down_tools_always_fail():
    w = make_world()
    results = lambda seed: [ToolBox(w, np.random.default_rng(seed), failure_multiplier=8.0).call(
        "check_shipment", customer_id="CUST-0001", order_id="ORD-1111").ok for _ in range(1)]
    assert results(3) == results(3)
    tb = ToolBox(w, np.random.default_rng(0), failure_multiplier=0.0, down_tools=["check_shipment"])
    assert tb.call("check_shipment", customer_id="CUST-0001", order_id="ORD-1111").error == "service_unavailable"


def test_scenarios_are_deterministic():
    spec = get_splits()["train"]
    a, b = generate_scenario(spec, 11), generate_scenario(spec, 11)
    assert (a.text, a.order_id, a.true_goals, a.scenario_type) == (b.text, b.order_id, b.true_goals, b.scenario_type)
    assert a.world.snapshot() == b.world.snapshot()


def test_heldout_combos_are_disjoint_from_training():
    assert not set(HELDOUT_COMBOS) & set(TRAIN_COMBOS)
    assert set(HELDOUT_COMBOS) | set(TRAIN_COMBOS) == set(ALL_COMBOS)
    sc = generate_scenarios(get_splits()["heldout_combos"], 80)
    assert all(tuple(s.true_goals) in HELDOUT_COMBOS for s in sc if s.scenario_type != "out_of_scope")


def test_splits_do_not_share_seeds():
    splits = get_splits()
    seeds = [s.base_seed for s in splits.values()]
    assert len(set(seeds)) == len(seeds)
    ids = {n: {s.scenario_id for s in generate_scenarios(spec, 30)} for n, spec in splits.items()}
    names = list(ids)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not ids[names[i]] & ids[names[j]]


def test_scenario_types_and_text_properties():
    sc = generate_scenarios(get_splits()["train"], 400)
    types = {s.scenario_type for s in sc}
    assert types == {"standard", "missing_order", "wrong_order", "low_confidence", "out_of_scope"}
    for s in sc:
        owned = s.world.customers[s.customer_id].order_ids
        if s.scenario_type == "missing_order":
            assert s.order_id_in_text is None and len(owned) > 1
        if s.scenario_type == "wrong_order":
            assert s.order_id_in_text not in owned
        if s.scenario_type == "out_of_scope":
            assert not s.true_goals and s.scope_score < 0.38
        if s.scenario_type == "standard" and len(owned) > 1:
            assert s.order_id_in_text == s.order_id


def test_eligible_and_ineligible_cases_are_balanced():
    sc = generate_scenarios(get_splits()["train"], 600)
    pc = [rules.payment_change_eligibility(s.world.orders[s.order_id], s.target_method).allowed
          for s in sc if "payment_change" in s.true_goals]
    assert 0.3 < np.mean(pc) < 0.7
