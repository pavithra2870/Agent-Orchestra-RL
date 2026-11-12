"""Hand-built 'golden' scenarios: the expected outcome is known in advance, so any change to the
environment, rules, guardrails or evaluator that breaks behaviour shows up here."""

import numpy as np
import pytest

from ecommerce.config import get_settings
from ecommerce.env.env import EcomSupportEnv
from ecommerce.env.state import ACTION_INDEX, ACTIONS, FEATURE_INDEX, OBS_DIM, decode, encode
from ecommerce.guardrails.action_mask import legal_mask
from ecommerce.sim.scenarios import scenario_from_world
from ecommerce.tests.conftest import make_order, make_world, run_plan

R = get_settings().reward
HAPPY = ["lookup_customer", "lookup_order", "check_shipment", "check_payment_change_eligibility",
         "update_payment_method", "finish_task"]


def test_divya_happy_path_succeeds_and_changes_the_database(divya):
    env, info, total = run_plan(divya, HAPPY)
    assert info["success"] is True
    assert env.world.orders["ORD-1111"].payment_method == "UPI"
    assert info["eval"]["violations"] == [] and info["eval"]["goal_completion_rate"] == 1.0
    assert divya.world.orders["ORD-1111"].payment_method == "COD", "the scenario's own world must never be modified"


def test_reward_equals_documented_formula(divya):
    env, info, total = run_plan(divya, HAPPY)
    cost = sum(t.cost for t in env.trace)
    latency = sum(t.latency for t in env.trace)
    expected = -(R.lam_cost * cost + R.mu_latency * latency) + 2 * R.goal_handled_bonus + R.success_bonus
    assert total == pytest.approx(expected)


def test_ineligible_change_is_declined_correctly_and_update_is_blocked():
    w = make_world([make_order(method="UPI", state="paid")])
    sc = scenario_from_world(w, "CUST-0001", "switch to card", ("payment_change",), "CARD", "ORD-1111")
    env, info, _ = run_plan(sc, ["lookup_customer", "lookup_order", "check_payment_change_eligibility",
                                 "update_payment_method", "finish_task"])
    assert info["success"] is True                                  # declined correctly
    assert env.trace[3].blocked and "eligibility" in env.trace[3].note
    assert env.world.orders["ORD-1111"].payment_method == "UPI"


def test_skipping_the_eligibility_check_is_blocked_with_guardrail_and_a_violation_without_it(divya):
    plan = ["lookup_customer", "lookup_order", "update_payment_method", "finish_task"]
    env, info, _ = run_plan(divya, plan, enforce_mask=True)
    assert env.trace[2].blocked and env.world.orders["ORD-1111"].payment_method == "COD"
    env2, info2, total2 = run_plan(divya, plan, enforce_mask=False)
    assert info2["success"] is False
    assert any(v["type"] == "state_change_without_check" for v in info2["eval"]["violations"])
    assert total2 < 0


def test_unrequested_state_change_is_a_violation():
    w = make_world([make_order(status="processing")])
    sc = scenario_from_world(w, "CUST-0001", "where is my order", ("delivery_status",), None, "ORD-1111")
    env, info, _ = run_plan(sc, ["lookup_customer", "lookup_order", "check_cancel_eligibility", "cancel_order",
                                 "check_shipment", "finish_task"], enforce_mask=False)
    assert any(v["type"] == "unrequested_state_change" for v in info["eval"]["violations"])
    assert not info["success"]


def test_out_of_scope_must_be_handed_to_a_human():
    w = make_world()
    sc = scenario_from_world(w, "CUST-0001", "tell me a joke", (), None, None, 0.4, 0.1, scenario_type="out_of_scope")
    assert run_plan(sc, ["escalate_to_human"])[1]["success"] is True
    info = run_plan(sc, ["finish_task"])[1]
    assert not info["success"] and "missing_escalation" in info["eval"]["failure_reasons"]


def test_high_value_refund_needs_a_human_and_refund_is_blocked():
    w = make_world([make_order(status="lost", method="UPI", state="paid", total=9000.0)])
    sc = scenario_from_world(w, "CUST-0001", "refund please", ("refund",), None, "ORD-1111")
    env, info, _ = run_plan(sc, ["lookup_customer", "lookup_order", "check_refund_eligibility",
                                 "initiate_refund", "escalate_to_human"])
    assert env.trace[3].blocked and info["success"] is True
    assert env.world.orders["ORD-1111"].payment_state == "paid"
    assert not run_plan(sc, ["lookup_customer", "lookup_order", "check_refund_eligibility", "finish_task"])[1]["success"]


def test_retry_limit_locks_a_tool_and_forces_handover():
    w = make_world()
    sc = scenario_from_world(w, "CUST-0001", "where is my order", ("delivery_status",), None, "ORD-1111",
                             down_tools=("check_shipment",))
    plan = ["lookup_customer", "lookup_order"] + ["check_shipment"] * 3
    env, info, _ = run_plan(sc, plan, failure_multiplier=0.0)
    assert not legal_mask(env.state)[ACTION_INDEX["check_shipment"]], "locked after 3 failures in a row"
    assert decode(encode(env.state))["needs_escalation:delivery_status"] == 1.0, "the state must tell the agent a human is needed"
    assert run_plan(sc, plan + ["escalate_to_human"])[1]["success"] is True
    assert not run_plan(sc, plan + ["finish_task"])[1]["success"]


def test_missing_order_requires_a_question_and_lookup_is_blocked_until_then():
    w = make_world([make_order("ORD-1111"), make_order("ORD-1112", status="shipped")])
    sc = scenario_from_world(w, "CUST-0001", "where is my order", ("delivery_status",), None, None,
                             scenario_type="missing_order", order_id="ORD-1112")
    env = EcomSupportEnv()
    env.reset(options={"scenario": sc})
    env.step(ACTION_INDEX["lookup_customer"])
    assert env.state.order_ambiguous and not legal_mask(env.state)[ACTION_INDEX["lookup_order"]]
    env.step(ACTION_INDEX["request_clarification"])
    assert legal_mask(env.state)[ACTION_INDEX["lookup_order"]] and env.state.order_candidate == "ORD-1112"


def test_unneeded_clarification_is_penalised_and_flagged(divya):
    env, info, total = run_plan(divya, ["request_clarification"] + HAPPY)
    assert info["eval"]["unneeded_clarification"] is True
    _, _, clean = run_plan(divya, HAPPY)
    assert total < clean


def test_step_limit_truncates_and_fails(divya):
    env, info, _ = run_plan(divya, ["lookup_order"] * 20)
    assert info["terminal"] == "max_steps" and not info["success"] and "step_limit" in info["eval"]["failure_reasons"]


def test_observation_is_well_formed_and_hides_ground_truth(divya):
    env = EcomSupportEnv()
    obs, info = env.reset(options={"scenario": divya})
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32 and obs.min() >= 0 and obs.max() <= 1
    d = decode(obs)
    assert d["shipment_known"] == 0 and d["pay_elig_known"] == 0 and d["requested:payment_change"] == 1
    obs2 = env.step(ACTION_INDEX["lookup_customer"])[0]
    obs3 = env.step(ACTION_INDEX["lookup_order"])[0]
    obs4 = env.step(ACTION_INDEX["check_shipment"])[0]
    assert decode(obs4)["shipment_known"] == 1 and decode(obs4)["handled:delivery_status"] == 1
    assert decode(obs4)["shipment_status:delayed"] == 1


def test_same_seed_same_trace_and_failures_vary_with_run_seed():
    w = make_world()
    sc = scenario_from_world(w, "CUST-0001", "where is my order", ("delivery_status",), None, "ORD-1111", seed=5)
    plan = ["lookup_customer", "lookup_order", "check_shipment", "finish_task"]
    t1 = [t.ok for t in run_plan(sc, plan, failure_multiplier=6.0, run_seed=1)[0].trace]
    t2 = [t.ok for t in run_plan(sc, plan, failure_multiplier=6.0, run_seed=1)[0].trace]
    assert t1 == t2
    outcomes = {tuple(t.ok for t in run_plan(sc, plan, failure_multiplier=6.0, run_seed=s)[0].trace) for s in range(20)}
    assert len(outcomes) > 1


# ---------------- Group B: returns, damage claims and policy questions
from datetime import date, timedelta

TODAY_ = date(2026, 10, 17)


def _delivered_world(days_ago=1, total=300.0, claims=()):
    o = make_order(status="delivered", method="UPI", state="paid", total=total, delay_days=0,
                   delivered_on=(TODAY_ - timedelta(days=days_ago)).isoformat())
    w = make_world([o])
    w.customers["CUST-0001"].claims.extend(claims)
    return w


def _return_scenario(world, reason="damaged", kind="refund", declared=True, goals=("return_item",)):
    return scenario_from_world(world, "CUST-0001", "it arrived broken", goals, None, "ORD-1111", return_reason=reason,
                               return_kind=kind, return_declared=declared, return_info_given=False)


RETURN_PLAN = ["request_clarification", "lookup_customer", "lookup_order", "check_return_eligibility", "process_return", "finish_task"]


def test_small_damaged_item_is_refunded_right_away_and_judged_correct():
    env, info, _ = run_plan(_return_scenario(_delivered_world()), RETURN_PLAN)
    o = env.world.orders["ORD-1111"]
    assert info["success"] is True and (o.return_state, o.payment_state) == ("instant_refund", "refunded")
    assert env.world.customers["CUST-0001"].claims, "the claim is recorded"


def test_the_agent_cannot_check_or_open_a_return_before_the_customer_has_answered():
    sc = _return_scenario(_delivered_world())
    env, info, _ = run_plan(sc, ["lookup_customer", "lookup_order", "check_return_eligibility", "process_return", "finish_task"])
    assert env.trace[2].blocked and "must ask" in env.trace[2].note and env.trace[3].blocked
    assert env.world.orders["ORD-1111"].return_state is None and info["success"] is False
    assert decode(encode(env.state))["return_info_missing"] == 1.0


def test_a_return_waits_for_inspection_and_pays_nothing_now():
    env, info, _ = run_plan(_return_scenario(_delivered_world(5), reason="change_of_mind"), RETURN_PLAN)
    o = env.world.orders["ORD-1111"]
    assert info["success"] is True and (o.return_state, o.payment_state) == ("requested", "paid")


def test_a_big_damaged_order_must_go_to_a_human_and_nothing_changes():
    sc = _return_scenario(_delivered_world(1, 900))
    ok, info_ok, _ = run_plan(sc, ["request_clarification", "lookup_customer", "lookup_order", "check_return_eligibility", "escalate_to_human"])
    assert info_ok["success"] is True and ok.world.orders["ORD-1111"].return_state is None
    wrong, info_w, _ = run_plan(sc, ["request_clarification", "lookup_customer", "lookup_order", "check_return_eligibility", "process_return", "finish_task"])
    assert wrong.trace[4].blocked, "the mask refuses process_return when the check says a human must decide"
    assert info_w["success"] is False and "missing_escalation" in info_w["eval"]["failure_reasons"]


def test_a_declined_declaration_is_declined_without_a_change():
    env, info, _ = run_plan(_return_scenario(_delivered_world(), declared=False),
                            ["request_clarification", "lookup_customer", "lookup_order", "check_return_eligibility", "finish_task"])
    assert info["success"] is True and env.world.orders["ORD-1111"].return_state is None


def test_a_policy_question_needs_no_order_and_changes_nothing():
    w = make_world([make_order(), make_order("ORD-2222")])
    sc = scenario_from_world(w, "CUST-0001", "what is your refund policy?", ("policy_info",), None, None, policy_topic="refund_policy")
    env, info, _ = run_plan(sc, ["lookup_policy", "finish_task"])
    assert info["success"] is True and env.world.snapshot() == w.snapshot()
    assert env.trace[0].data["topic"] == "refund_policy" and "refund" in env.trace[0].data["answer"].lower()
    env2, info2, _ = run_plan(sc, ["finish_task"])
    assert info2["success"] is False and "missed_goal:policy_info" in info2["eval"]["failure_reasons"]
    asked, _, _ = run_plan(sc, ["lookup_policy", "lookup_policy"])
    assert asked.trace[1].blocked, "no repeated lookups"


def test_a_wrong_policy_topic_is_not_a_success():
    from ecommerce.evaluation.evaluator import evaluate

    w = make_world([make_order()])
    sc = scenario_from_world(w, "CUST-0001", "x", ("policy_info",), None, None, policy_topic="lost_order")
    env, info, _ = run_plan(sc, ["lookup_policy", "finish_task"])
    assert info["success"] is True
    sc.policy_topic = "refund_policy"            # the customer really asked about refunds; the trace shows "lost_order"
    res = evaluate(sc, env.world, env.trace, "finish")
    assert not res.success and "missed_goal:policy_info" in res.failure_reasons


def test_a_refund_of_a_delivered_item_without_a_return_is_a_violation_in_the_evaluator():
    from ecommerce.evaluation.evaluator import evaluate

    w = _delivered_world()
    sc = scenario_from_world(w, "CUST-0001", "refund", ("refund",), None, "ORD-1111")
    env, info, _ = run_plan(sc, ["lookup_customer", "lookup_order", "check_refund_eligibility", "initiate_refund", "finish_task"], enforce_mask=False)
    assert env.world.orders["ORD-1111"].payment_state == "paid", "the tool itself refuses (return_required)"
    broken = env.world.copy()
    broken.orders["ORD-1111"].payment_state = "refunded"           # what a bug in a tool would look like
    res = evaluate(sc, broken, env.trace, "finish")
    assert any(v["type"] == "refund_without_return" for v in res.violations)


def test_observation_and_actions_after_group_b():
    assert OBS_DIM == len(decode(np.zeros(OBS_DIM))) == 85 and len(ACTIONS) == 16
    assert {"check_return_eligibility", "process_return", "lookup_policy"} <= set(ACTIONS)
