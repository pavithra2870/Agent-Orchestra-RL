"""Returns, exchanges, damage claims, inspection and the policy tool: the rules and the tools, case by case."""

from datetime import date, timedelta

import numpy as np
import pytest

from ecommerce.config import get_settings
from ecommerce.sim import rules
from ecommerce.sim.inspection import complete_inspection, seeded_inspection
from ecommerce.sim.tools import ToolBox
from ecommerce.sim.world import Customer, generate_world
from ecommerce.tests.conftest import make_order, make_world

CFG = get_settings().world
TODAY = date(2026, 10, 17)


def delivered(days_ago=1, total=300.0, **kw):
    o = make_order("ORD-1111", status="delivered", method="UPI", state="paid", total=total,
                   delivered_on=(TODAY - timedelta(days=days_ago)).isoformat(), delay_days=0, **kw)
    return o


def elig(order, reason="damaged", kind="refund", declared=True, claims=()):
    c = Customer("CUST-0001", "Divya Shah", ["ORD-1111"], list(claims))
    return rules.return_eligibility(order, c, reason, kind, declared, CFG, TODAY)


def claim(days_ago):
    return {"order_id": "ORD-0001", "date": (TODAY - timedelta(days=days_ago)).isoformat(), "amount": 200.0, "mode": "instant_refund"}


# ---------------- the agreed rules
def test_small_damaged_item_reported_in_time_is_refunded_right_away():
    e = elig(delivered(1, 300))
    assert (e.status, e.reason) == (rules.ELIGIBLE, "instant_refund_ok")


def test_small_damaged_item_with_exchange_opens_an_exchange():
    assert elig(delivered(1, 300), kind="exchange").reason == "exchange_ok"


@pytest.mark.parametrize("days,total,claims,reason", [
    (3, 300, (), "late_damage_report"),                 # later than 48 hours
    (1, 900, (), "above_instant_limit"),                # above the 500 rupee cap (either choice)
    (1, 300, (claim(5), claim(20)), "claim_history"),   # two damage claims in the last 30 days
])
def test_damage_claims_that_need_a_human(days, total, claims, reason):
    for kind in ("refund", "exchange"):
        e = elig(delivered(days, total), kind=kind, claims=claims)
        assert (e.status, e.reason) == (rules.NEEDS_HUMAN, reason), kind


def test_old_claims_do_not_count_after_the_period():
    assert elig(delivered(1, 300), claims=(claim(40), claim(50))).reason == "instant_refund_ok"
    assert elig(delivered(1, 300), claims=(claim(5),)).reason == "instant_refund_ok"      # one is below the limit


def test_change_of_mind_return_has_a_7_day_window_and_no_instant_refund():
    assert elig(delivered(5, 300), reason="change_of_mind").reason == "return_ok"
    assert elig(delivered(5, 300), reason="change_of_mind", kind="exchange").reason == "exchange_ok"
    e = elig(delivered(8, 300), reason="change_of_mind")
    assert (e.status, e.reason) == (rules.INELIGIBLE, "outside_return_window")
    assert elig(delivered(1, 9000), reason="change_of_mind").status == rules.NEEDS_HUMAN       # high value


def test_safeguards():
    assert elig(make_order(status="shipped", state="paid", method="UPI")).reason == "not_delivered"
    assert elig(delivered(1), declared=False).reason == "declaration_not_confirmed"
    o = delivered(1)
    o.return_state = "requested"
    assert elig(o).reason == "already_claimed"
    o = delivered(1)
    o.payment_state = "refunded"
    assert elig(o).reason == "already_refunded"
    o = make_order(status="delivered", method="COD", state="pending", delivered_on=(TODAY - timedelta(days=1)).isoformat())
    assert elig(o).reason == "not_paid"


def test_a_delivered_item_is_never_refunded_by_the_plain_refund_tool():
    e = rules.refund_eligibility(delivered(1, 300), CFG, TODAY)
    assert (e.status, e.reason) == (rules.INELIGIBLE, "return_required")
    lost = make_order(status="lost", method="UPI", state="paid")
    assert rules.refund_eligibility(lost, CFG, TODAY).allowed, "lost orders are unchanged"


# ---------------- the tools
def toolbox(order, claims=()):
    world = make_world([order])
    world.customers["CUST-0001"].claims.extend(claims)
    return world, ToolBox(world, np.random.default_rng(0), failure_multiplier=0.0)


def test_instant_refund_changes_the_order_and_records_the_claim():
    order = delivered(1, 300)
    world, tb = toolbox(order)
    r = tb.call("process_return", customer_id="CUST-0001", order_id="ORD-1111", reason="damaged", kind="refund", declared=True)
    assert r.ok and r.data["mode"] == "instant_refund" and r.mutated
    o = world.orders["ORD-1111"]
    assert (o.payment_state, o.return_state, o.refund_id) == ("refunded", "instant_refund", "RF-1111")
    assert len(world.customers["CUST-0001"].claims) == 1
    again = tb.call("process_return", customer_id="CUST-0001", order_id="ORD-1111", reason="damaged", kind="refund", declared=True)
    assert not again.ok and again.error == "not_allowed", "one claim per order"


def test_a_return_only_opens_the_request_and_refunds_nothing():
    order = delivered(5, 300)
    world, tb = toolbox(order)
    r = tb.call("process_return", customer_id="CUST-0001", order_id="ORD-1111", reason="change_of_mind", kind="refund", declared=True)
    assert r.ok and r.data["mode"] == "return_requested"
    o = world.orders["ORD-1111"]
    assert (o.return_state, o.payment_state, o.refund_id) == ("requested", "paid", None)
    assert world.customers["CUST-0001"].claims == [], "change of mind is not a damage claim"
    # while it waits, a plain refund is refused
    assert tb.call("check_refund_eligibility", customer_id="CUST-0001", order_id="ORD-1111").data["reason"] == "return_in_progress"


def test_the_tool_refuses_what_the_rules_forbid():
    order = delivered(1, 900)
    world, tb = toolbox(order)
    r = tb.call("process_return", customer_id="CUST-0001", order_id="ORD-1111", reason="damaged", kind="refund", declared=True)
    assert not r.ok and world.orders["ORD-1111"].return_state is None
    other = tb.call("process_return", customer_id="CUST-0002", order_id="ORD-1111", reason="damaged", kind="refund", declared=True)
    assert other.error == "permission_denied"


# ---------------- inspection (a step of the shop's process, not of the agent)
def open_return(kind):
    order = delivered(5, 300)
    world, tb = toolbox(order)
    tb.call("process_return", customer_id="CUST-0001", order_id="ORD-1111", reason="change_of_mind", kind=kind, declared=True)
    return world.orders["ORD-1111"]


def test_passed_inspection_releases_the_refund():
    o = complete_inspection(open_return("refund"), passed=True)
    assert (o.return_state, o.payment_state, o.refund_id) == ("inspection_passed", "refunded", "RF-1111")


def test_passed_inspection_ships_the_replacement_for_an_exchange():
    o = complete_inspection(open_return("exchange"), passed=True)
    assert (o.return_state, o.exchange_id, o.payment_state) == ("inspection_passed", "EX-1111", "paid")


def test_failed_inspection_refunds_nothing_and_a_later_refund_request_is_declined():
    o = complete_inspection(open_return("refund"), passed=False)
    assert (o.return_state, o.payment_state, o.refund_id) == ("inspection_failed", "paid", None)
    e = rules.refund_eligibility(o, CFG, TODAY)
    assert (e.status, e.reason) == (rules.INELIGIBLE, "inspection_failed")


def test_inspection_needs_a_waiting_return_and_the_seeded_outcome_is_stable():
    with pytest.raises(ValueError):
        complete_inspection(delivered(1), passed=True)
    assert [seeded_inspection(f"ORD-{n}", 0) for n in range(1000, 1010)] == [seeded_inspection(f"ORD-{n}", 0) for n in range(1000, 1010)]
    rate = np.mean([seeded_inspection(f"ORD-{n}", 0) for n in range(1000, 3000)])
    assert 0.8 < rate < 0.9


# ---------------- policy tool and data
def test_policy_tool_returns_text_built_from_the_config_and_touches_nothing():
    world, tb = toolbox(delivered(1))
    before = world.snapshot()
    d = tb.call("lookup_policy", topic="damaged_item").data
    assert "48 hours" in d["answer"] and "500 rupees" in d["answer"] and d["title"]
    assert "7 days" in tb.call("lookup_policy", topic="returns_exchange").data["answer"]
    assert tb.call("lookup_policy", topic="nope").error == "unknown_topic"
    assert world.snapshot() == before


def test_new_data_does_not_change_the_existing_demo_world():
    w = generate_world(2026)
    o = w.orders["ORD-6759"]
    assert (o.status, o.payment_method, o.payment_state) == ("delayed", "COD", "pending")
    assert sum(len(c.claims) for c in w.customers.values()) > 0
    assert generate_world(2026).customers["CUST-0003"].claims == w.customers["CUST-0003"].claims, "same seed, same history"
