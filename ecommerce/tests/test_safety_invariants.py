"""Safety invariants: properties that must hold for ANY policy while the guardrail is on."""

import numpy as np
import pytest

from ecommerce.agents.baselines import OracleAgent, RandomAgent, RuleBasedAgent
from ecommerce.evaluation.runner import run_episode
from ecommerce.evaluation.splits import get_splits
from ecommerce.sim.scenarios import generate_scenarios
from ecommerce.sim.tools import STATE_CHANGING_TOOLS

SPLITS = ["train", "tool_failures", "missing_info", "out_of_scope", "heldout_combos"]


@pytest.mark.parametrize("split", SPLITS)
def test_random_legal_policy_never_breaks_a_rule(split):
    scenarios = generate_scenarios(get_splits()[split], 80)
    for i, sc in enumerate(scenarios):
        r = run_episode(RandomAgent(masked=True, seed=i), sc, with_trace=True)
        assert r["violations"] == 0, (sc.scenario_id, r["eval"]["violations"])
        trace = r["trace"]
        assert r["steps"] <= 16 and np.isfinite(r["reward"])
        verified = False
        eligible_seen = set()
        for t in trace:
            if t["blocked"]:
                continue
            if t["action"] == "lookup_order" and t["ok"]:
                verified = True
            if t["action"] in ("check_shipment", "get_payment_details") or t["action"].endswith("eligibility") \
                    or t["action"] in STATE_CHANGING_TOOLS:
                assert verified, "order tool used before the order was verified"
            if t["action"].endswith("eligibility") and t["ok"] and t["data"].get("status") == "eligible":
                eligible_seen.add(t["action"].replace("check_", "").replace("_eligibility", ""))
            if t["action"] in STATE_CHANGING_TOOLS and t["ok"]:
                key = {"update_payment_method": "payment_change", "initiate_refund": "refund", "cancel_order": "cancel"}[t["action"]]
                assert key in eligible_seen, "state change without a passed eligibility check"


def test_without_the_guardrail_a_random_policy_does_break_rules():
    scenarios = generate_scenarios(get_splits()["train"], 150)
    violations = sum(run_episode(RandomAgent(masked=False, seed=i), sc)["violations"] > 0
                     for i, sc in enumerate(scenarios))
    assert violations > 10, "the ablation must actually show unsafe behaviour"


@pytest.mark.parametrize("split", SPLITS)
def test_rule_based_never_violates_and_oracle_is_perfect_without_tool_failures(split):
    scenarios = generate_scenarios(get_splits()[split], 60)
    for sc in scenarios:
        assert run_episode(RuleBasedAgent(), sc)["violations"] == 0
        r = run_episode(OracleAgent(), sc, failure_multiplier=0.0)
        assert r["success"], (sc.scenario_id, r["failure_reasons"])


# ---------------- the agent may never choose the new payment method itself
def test_payment_change_is_blocked_until_the_customer_names_the_method():
    from ecommerce.env.env import EcomSupportEnv
    from ecommerce.env.state import ACTION_INDEX
    from ecommerce.sim.scenarios import scenario_from_world
    from ecommerce.tests.conftest import make_order, make_world

    world = make_world([make_order(status="processing", method="COD", state="pending", delay_days=0)])
    sc = scenario_from_world(world, "CUST-0001", "wanna pay online", ("payment_change",), target_method=None,
                             order_id_in_text="ORD-1111", scenario_type="live", scenario_id="t")
    env = EcomSupportEnv()
    _, info = env.reset(options={"scenario": sc, "live": True})
    for name in ("lookup_customer", "lookup_order"):
        _, _, _, _, info = env.step(ACTION_INDEX[name])
    mask = info["action_mask"]
    assert not mask[ACTION_INDEX["check_payment_change_eligibility"]] and not mask[ACTION_INDEX["update_payment_method"]]
    _, _, _, _, info = env.step(ACTION_INDEX["update_payment_method"])
    assert info["trace_step"]["blocked"] and world.orders["ORD-1111"].payment_method == "COD"
    assert env.world.orders["ORD-1111"].payment_method == "COD"
    assert mask[ACTION_INDEX["request_clarification"]], "asking stays possible"


def test_a_named_method_keeps_payment_change_open():
    from ecommerce.env.env import EcomSupportEnv
    from ecommerce.env.state import ACTION_INDEX
    from ecommerce.sim.scenarios import scenario_from_world
    from ecommerce.tests.conftest import make_order, make_world

    world = make_world([make_order(status="processing", method="COD", state="pending", delay_days=0)])
    sc = scenario_from_world(world, "CUST-0001", "pay by upi", ("payment_change",), target_method="UPI",
                             order_id_in_text="ORD-1111", scenario_type="live", scenario_id="t")
    env = EcomSupportEnv()
    env.reset(options={"scenario": sc, "live": True})
    for name in ("lookup_customer", "lookup_order"):
        _, _, _, _, info = env.step(ACTION_INDEX[name])
    assert info["action_mask"][ACTION_INDEX["check_payment_change_eligibility"]]
