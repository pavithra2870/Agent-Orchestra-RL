"""The metrics code itself must be right, because every table and chart depends on it."""

import numpy as np
import pytest

from ecommerce.evaluation.metrics import breakdown, summarize


def ep(sid, run, success, **kw):
    base = dict(scenario_id=sid, run_seed=run, success=success, reward=1.0 if success else -2.0, steps=5, cost=3.0,
                latency=4.0, terminal="finish", expected_terminal="finish", goal_completion=float(success),
                violations=0, blocked=0, failure_reasons=[] if success else ["missed_goal:refund"], clar_needed=False,
                clar_asked=0, unneeded_clar=False, unneeded_esc=False, tool_failures=0, type="standard", combo="refund")
    base.update(kw)
    return base


def test_pass_k_requires_every_repeat_to_succeed():
    results = [ep("a", 0, True), ep("a", 1, True), ep("b", 0, True), ep("b", 1, False)]
    s = summarize(results, k=2)
    assert s["success_rate"] == pytest.approx((1.0 + 0.5) / 2)
    assert s["pass_k"] == pytest.approx(0.5) and s["k"] == 2


def test_confidence_interval_brackets_the_mean():
    results = [ep(f"s{i}", 0, i % 4 != 0) for i in range(80)]
    s = summarize(results, k=1)
    lo, hi = s["success_ci95"]
    assert lo <= s["success_rate"] <= hi and hi - lo < 0.25


def test_regret_is_oracle_minus_agent_reward():
    results = [ep("a", 0, True, reward=4.0), ep("b", 0, False, reward=-1.0)]
    s = summarize(results, k=1, oracle={"a": 6.0, "b": 6.0})
    assert s["regret"] == pytest.approx(((6 - 4) + (6 + 1)) / 2)


def test_clarification_and_escalation_precision_recall():
    results = [
        ep("a", 0, True, clar_needed=True, clar_asked=1),
        ep("b", 0, False, clar_needed=True, clar_asked=0),
        ep("c", 0, False, clar_needed=False, clar_asked=2, unneeded_clar=True),
        ep("d", 0, True, terminal="escalate", expected_terminal="escalate"),
        ep("e", 0, False, terminal="escalate", expected_terminal="finish", unneeded_esc=True),
    ]
    s = summarize(results, k=1)
    assert s["clarify_recall"] == pytest.approx(0.5) and s["clarify_precision"] == pytest.approx(0.5)
    assert s["escalate_recall"] == pytest.approx(1.0) and s["escalate_precision"] == pytest.approx(0.5)
    assert s["unneeded_escalation_rate"] == pytest.approx(0.2)


def test_failure_reasons_are_counted_by_category():
    results = [ep("a", 0, False, failure_reasons=["missed_goal:refund", "violation:state_change_without_check"]),
               ep("b", 0, True)]
    s = summarize(results, k=1)
    assert s["failure_reasons"]["missed_goal"] == pytest.approx(0.5)
    assert s["failure_reasons"]["violation"] == pytest.approx(0.5)


def test_breakdown_reports_per_group_numbers():
    results = [ep("a", 0, True, type="standard"), ep("b", 0, False, type="standard"), ep("c", 0, True, type="out_of_scope")]
    b = breakdown(results, "type")
    assert b["standard"] == {"success_rate": 0.5, "n": 2} and b["out_of_scope"]["success_rate"] == 1.0
