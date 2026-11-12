"""Order inference: one test per situation, plus a property test over generated worlds."""

from datetime import date

import pytest

from ecommerce.config import get_settings
from ecommerce.language.orders import STATE_CHANGING_GOALS, _qualifies, is_no, is_yes, resolve_order
from ecommerce.sim.world import generate_world
from ecommerce.tests.conftest import make_order

TODAY = date(2026, 10, 17)
CFG = get_settings().world


def res(text, goals, orders, target=None, history=(), exclude=()):
    return resolve_order(text, goals, target, orders, TODAY, CFG, history, exclude)


def o(oid, status="shipped", method="COD", state="pending", placed="2026-10-01", **kw):
    x = make_order(oid, "CUST-0001", status=status, method=method, state=state, **kw)
    x.placed_on = placed
    return x


def test_single_order_is_simply_used():
    r = res("where is my order", ["delivery_status"], [o("ORD-1111")])
    assert (r.order_id, r.how, r.assumed) == ("ORD-1111", "single_order", False)


# ---------------- where is it? -> orders still on their way
def test_delivery_picks_the_only_order_still_on_its_way():
    r = res("where is my order", ["delivery_status"], [o("ORD-1111", "shipped"), o("ORD-2222", "delivered", "UPI", "paid", delivered_on="2026-09-20")])
    assert (r.order_id, r.how, r.assumed, r.confirm_needed) == ("ORD-1111", "inferred", True, False)
    assert "still on its way" in r.reason


def test_delivery_with_two_orders_in_transit_asks_and_lists_only_those():
    r = res("where is my order", ["delivery_status"], [o("ORD-1111", "shipped"), o("ORD-2222", "processing"),
                                                       o("ORD-3333", "delivered", "UPI", "paid", delivered_on="2026-09-01")])
    assert r.how == "ambiguous" and r.order_id is None and r.candidates == ["ORD-1111", "ORD-2222"]


def test_delivery_when_nothing_is_in_transit_asks():
    r = res("where is my order", ["delivery_status"], [o("ORD-1111", "delivered", "UPI", "paid", delivered_on="2026-09-01"),
                                                       o("ORD-2222", "returned", "UPI", "paid", delivered_on="2026-09-02")])
    assert r.how == "none_qualifies" and "still on its way" in r.reason and r.candidates == ["ORD-1111", "ORD-2222"]


def test_delivery_falls_back_to_a_lost_order_when_nothing_is_in_transit():
    r = res("where is my order", ["delivery_status"], [o("ORD-1111", "lost", "UPI", "paid"),
                                                       o("ORD-2222", "delivered", "UPI", "paid", delivered_on="2026-09-01")])
    assert (r.order_id, r.how) == ("ORD-1111", "inferred")


# ---------------- refund
def test_refund_picks_the_only_refundable_order_and_needs_confirmation():
    r = res("i want a refund", ["refund"], [o("ORD-1111", "lost", "UPI", "paid"), o("ORD-2222", "shipped", "COD", "pending")])
    assert (r.order_id, r.assumed, r.confirm_needed) == ("ORD-1111", True, True)


def test_refund_two_candidates_asks_and_a_high_value_order_counts_as_a_candidate():
    r = res("refund please", ["refund"], [o("ORD-1111", "lost", "UPI", "paid"), o("ORD-2222", "lost", "UPI", "paid", total=9000.0)])
    assert r.how == "ambiguous" and set(r.candidates) == {"ORD-1111", "ORD-2222"}


def test_refund_when_no_order_qualifies_asks_and_says_why():
    r = res("refund please", ["refund"], [o("ORD-1111", "shipped", "COD", "pending"), o("ORD-2222", "processing", "UPI", "paid")])
    assert r.how == "none_qualifies" and "refund" in r.reason


def test_refund_ignores_unpaid_cod_even_if_lost():
    r = res("refund please", ["refund"], [o("ORD-1111", "lost", "COD", "pending"), o("ORD-2222", "lost", "UPI", "paid")])
    assert r.order_id == "ORD-2222"


# ---------------- cancel
def test_cancel_picks_the_only_order_still_processing():
    r = res("cancel my order", ["cancel_order"], [o("ORD-1111", "processing"), o("ORD-2222", "shipped")])
    assert (r.order_id, r.how, r.confirm_needed) == ("ORD-1111", "inferred", True)


def test_cancel_when_everything_has_shipped_asks():
    r = res("cancel my order", ["cancel_order"], [o("ORD-1111", "shipped"), o("ORD-2222", "delayed")])
    assert r.how == "none_qualifies" and "cancelled" in r.reason


# ---------------- payment change
def test_payment_change_skips_paid_out_for_delivery_and_same_method_orders():
    orders = [o("ORD-1111", "shipped", "COD", "pending", out_for_delivery=True), o("ORD-2222", "processing", "COD", "pending"),
              o("ORD-3333", "processing", "UPI", "paid")]
    r = res("switch to UPI", ["payment_change"], orders, target="UPI")
    assert (r.order_id, r.how) == ("ORD-2222", "inferred")
    r2 = res("switch to UPI", ["payment_change"], [o("ORD-2222", "processing", "UPI", "pending"), o("ORD-3333", "processing", "UPI", "paid")], target="UPI")
    assert r2.how == "none_qualifies"


# ---------------- several goals in one message
def test_two_goals_pick_the_one_order_that_fits_both():
    orders = [o("ORD-1111", "shipped", "COD", "pending"), o("ORD-2222", "delivered", "UPI", "paid", delivered_on="2026-09-01")]
    r = res("where is it and change payment to UPI", ["delivery_status", "payment_change"], orders, target="UPI")
    assert (r.order_id, r.how, r.confirm_needed) == ("ORD-1111", "inferred", True) and "everything" in r.reason


def test_two_goals_that_fit_different_orders_ask_instead_of_guessing():
    orders = [o("ORD-1111", "shipped", "COD", "pending"), o("ORD-2222", "lost", "UPI", "paid")]
    r = res("where is it and refund it", ["delivery_status", "refund"], orders)
    assert r.how == "ambiguous" and set(r.candidates) == {"ORD-1111", "ORD-2222"}


# ---------------- the customer names it
ORDERS3 = [o("ORD-4175", "shipped", placed="2026-10-14"),
           o("ORD-4749", "lost", "UPI", "paid", placed="2026-10-07"), o("ORD-3679", "processing", placed="2026-10-12")]
ORDERS3[0].items, ORDERS3[1].items, ORDERS3[2].items = ["Wireless mouse"], ["Phone case", "Backpack"], ["Notebook set"]


@pytest.mark.parametrize("text,expected", [
    ("where is ORD-4749", "ORD-4749"), ("where is order 4749", "ORD-4749"), ("my lost order", "ORD-4749"),
    ("the wireless mouse", "ORD-4175"), ("the notebook", "ORD-3679"), ("my latest order", "ORD-4175"), ("my oldest order", "ORD-4749"),
])
def test_what_the_customer_names_beats_inference(text, expected):
    r = res(text, ["delivery_status"], ORDERS3)
    assert r.order_id == expected and not r.assumed


# ---------------- the conversation (last few messages)
HIST_LIST = [{"role": "user", "text": "where is my order"},
             {"role": "agent", "text": "I can see more than one order. Which one?\n- ORD-6405: Backpack\n- ORD-5992: Bedsheet"}]


def test_the_other_one_after_i_received_one():
    orders = [o("ORD-6405", "delivered", "UPI", "paid", delivered_on="2026-10-10"), o("ORD-5992", "shipped")]
    r = res("have already received one. am asking bout the other", ["delivery_status"], orders, history=HIST_LIST)
    assert (r.order_id, r.how) == ("ORD-5992", "history")


def test_it_refers_to_the_order_we_just_talked_about():
    hist = [{"role": "user", "text": "where is ORD-4749"}, {"role": "agent", "text": "Order ORD-4749 is currently lost."}]
    r = res("and refund it too", ["refund"], ORDERS3, history=hist)
    assert (r.order_id, r.how) == ("ORD-4749", "history")
    assert res("and refund it too", ["refund"], ORDERS3).how in ("inferred", "ambiguous", "none_qualifies"), "no history, no anaphora"


def test_the_other_one_without_a_clear_answer_does_not_guess():
    orders = [o("ORD-1111", "shipped"), o("ORD-2222", "shipped"), o("ORD-3333", "shipped")]
    hist = [{"role": "agent", "text": "Which one? ORD-1111, ORD-2222, ORD-3333"}]
    r = res("the other one", ["delivery_status"], orders, history=hist)
    assert r.order_id is None


# ---------------- no goals, corrections, yes / no
def test_no_goals_and_several_orders_asks_what_and_which():
    r = res("hi", [], ORDERS3)
    assert r.how == "no_goals" and len(r.candidates) == 3


def test_after_a_no_the_rejected_order_is_not_assumed_again():
    orders = [o("ORD-1111", "shipped"), o("ORD-2222", "delivered", "UPI", "paid", delivered_on="2026-09-01")]
    assert res("where is my order", ["delivery_status"], orders).order_id == "ORD-1111"
    again = res("no, not that one", ["delivery_status"], orders, exclude=["ORD-1111"])
    assert again.order_id is None


@pytest.mark.parametrize("text,yes,no", [("yes", True, False), ("yes please do", True, False), ("go ahead", True, False), ("ok", True, False),
                                         ("no", False, True), ("no, ORD-4749", False, True), ("not that one", False, True), ("where is it", False, False)])
def test_yes_and_no(text, yes, no):
    assert is_yes(text) is yes and is_no(text) is no


# ---------------- properties over many generated worlds
@pytest.mark.parametrize("goal", ["delivery_status", "refund", "cancel_order", "payment_change"])
def test_inference_never_picks_a_wrong_order(goal):
    checked = 0
    for seed in range(25):
        w = generate_world(seed)
        for c in w.customers.values():
            if len(c.order_ids) < 2:
                continue
            orders = [w.orders[i] for i in c.order_ids]
            r = resolve_order("my order", [goal], "UPI" if goal == "payment_change" else None, orders, TODAY, CFG)
            checked += 1
            if r.how == "inferred":
                assert r.order_id in c.order_ids and r.assumed
                assert (goal in STATE_CHANGING_GOALS) == r.confirm_needed
                q = _qualifies(goal, w.orders[r.order_id], "UPI" if goal == "payment_change" else None, CFG, TODAY)
                assert q or goal == "delivery_status"
            elif r.how == "ambiguous":
                assert len(r.candidates) > 1 and all(i in c.order_ids for i in r.candidates)
            elif r.how == "none_qualifies" and goal != "delivery_status":
                assert not any(_qualifies(goal, x, "UPI" if goal == "payment_change" else None, CFG, TODAY) for x in orders)
    assert checked > 30


# ---------------- Group B goals
def test_a_policy_question_needs_no_order():
    r = res("what is your refund policy?", ["policy_info"], ORDERS3)
    assert (r.order_id, r.how) == (None, "no_order_needed")
    mixed = res("what is the cancellation rule, and cancel my order", ["policy_info", "cancel_order"], [o("ORD-1111", "processing"), o("ORD-2222", "shipped")])
    assert (mixed.order_id, mixed.how) == ("ORD-1111", "inferred"), "the policy part is ignored when picking the order"


def test_return_picks_the_only_delivered_order_that_can_still_be_returned():
    orders = [o("ORD-1111", "delivered", "UPI", "paid", delivered_on="2026-10-15"), o("ORD-2222", "shipped"),
              o("ORD-3333", "delivered", "UPI", "paid", delivered_on="2026-09-01")]          # too old
    r = res("i want to return my order", ["return_item"], orders)
    assert (r.order_id, r.how, r.confirm_needed) == ("ORD-1111", "inferred", True)
    none = res("i want to return my order", ["return_item"], [o("ORD-2222", "shipped"), o("ORD-3333", "processing")])
    assert none.how == "none_qualifies" and "returned right now" in none.reason
