import numpy as np
import pytest

from src.envs.tool_orchestration_env import ToolOrchestrationEnv, decode_observation
from src.envs.tools import TOOL_INDEX, STOP_ACTION, NUM_TOOLS
from src.envs.tasks import TASK_REGISTRY, NUM_TASKS
from src.envs.oracle import OracleSolver


def make_env(**kwargs):
    return ToolOrchestrationEnv(seed=123, **kwargs)


def test_action_and_observation_spaces():
    env = make_env()
    assert env.action_space.n == NUM_TOOLS + 1
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert obs.shape == (NUM_TASKS + 2 * NUM_TOOLS + 2,)


def test_reset_is_reproducible_under_fixed_seed():
    env1 = make_env()
    env2 = make_env()
    obs1, _ = env1.reset(seed=7)
    obs2, _ = env2.reset(seed=7)
    np.testing.assert_array_equal(obs1, obs2)
    # same seed -> same episode noise -> identical trajectory for the same actions
    action = 0
    r1 = env1.step(action)
    r2 = env2.step(action)
    assert r1[1] == r2[1]  # reward
    np.testing.assert_array_equal(r1[0], r2[0])  # obs


def test_tool_call_deducts_cost_and_latency():
    env = make_env(force_deterministic_success=True)
    obs, info = env.reset(seed=1, options={"task_id": 2})  # process_refund
    action = TOOL_INDEX["verify_identity"]
    obs, reward, terminated, truncated, info = env.step(action)
    assert info["cost"] > 0
    assert info["latency"] > 0
    assert env.remaining_budget < TASK_REGISTRY[2].max_budget
    assert not terminated  # process_refund also needs payment_api


def test_reward_formula_matches_manual_calculation_when_deterministic():
    lam, mu, success_reward, failure_penalty = 0.1, 0.2, 10.0, 2.0
    env = ToolOrchestrationEnv(
        success_reward=success_reward, lam=lam, mu=mu, failure_penalty=failure_penalty,
        seed=5, force_deterministic_success=True,
    )
    obs, info = env.reset(seed=5, options={"task_id": 0})  # check_order_status: db_lookup alone succeeds
    action = TOOL_INDEX["db_lookup"]
    tool_cost = env._cost_factor[action] * 1.0  # base_cost of db_lookup is 1.0
    tool_latency = env._latency_factor[action] * 1.0  # base_latency of db_lookup is 1.0
    obs, reward, terminated, truncated, info = env.step(action)
    expected_reward = -lam * tool_cost - mu * tool_latency + success_reward * 1.0  # quality=1.0 path
    assert terminated
    assert info["success"] is True
    assert reward == pytest.approx(expected_reward, rel=1e-6)


def test_success_path_detection_multi_tool_task():
    env = make_env(force_deterministic_success=True)
    obs, info = env.reset(seed=2, options={"task_id": 2})  # process_refund: verify_identity + payment_api
    obs, reward, terminated, truncated, info = env.step(TOOL_INDEX["verify_identity"])
    assert not terminated
    obs, reward, terminated, truncated, info = env.step(TOOL_INDEX["payment_api"])
    assert terminated
    assert info["success"] is True


def test_stop_action_ends_episode_as_failure():
    env = make_env()
    obs, info = env.reset(seed=3, options={"task_id": 0})
    obs, reward, terminated, truncated, info = env.step(STOP_ACTION)
    assert terminated
    assert not truncated
    assert info["success"] is False
    assert info["reason"] == "stop"
    assert reward == pytest.approx(-env.failure_penalty)


def test_budget_exhaustion_terminates_episode():
    env = make_env(force_deterministic_success=False)
    # escalate_complaint has a generous budget; force failure by always failing tools
    # via a task with a tight budget and an expensive tool: fraud_check_and_block.
    obs, info = env.reset(seed=4, options={"task_id": 7})
    done = False
    steps = 0
    while not done and steps < 20:
        obs, reward, terminated, truncated, info = env.step(TOOL_INDEX["escalate_human"])
        done = terminated or truncated
        steps += 1
    assert done
    assert info["reason"] in ("budget_exhausted", "task_success", "max_steps")


def test_max_steps_truncates_episode():
    env = make_env(force_deterministic_success=True)
    obs, info = env.reset(seed=9, options={"task_id": 9})  # full_account_review, needs 4 distinct tools
    # Repeatedly call a tool NOT in the success path so it never succeeds the task,
    # until max_steps truncation kicks in.
    task = TASK_REGISTRY[9]
    done = False
    steps = 0
    while not done:
        obs, reward, terminated, truncated, info = env.step(TOOL_INDEX["send_email"])
        done = terminated or truncated
        steps += 1
        assert steps <= task.max_steps + 1
    assert truncated
    assert info["reason"] == "max_steps"


def test_invalid_action_raises():
    env = make_env()
    env.reset(seed=1)
    with pytest.raises(AssertionError):
        env.step(999)


def test_decode_observation_roundtrip():
    env = make_env()
    obs, info = env.reset(seed=11, options={"task_id": 5})
    decoded = decode_observation(obs)
    assert decoded["task_id"] == 5
    assert decoded["succeeded_mask"].sum() == 0
    assert decoded["budget_norm"] == pytest.approx(1.0)


def test_oracle_solver_returns_valid_path_for_every_task():
    solver = OracleSolver(success_reward=10.0, lam=0.1, mu=0.1)
    plans = solver.solve_all()
    assert len(plans) == NUM_TASKS
    for task in TASK_REGISTRY:
        plan = plans[task.task_id]
        assert plan.best_path in task.success_paths
        assert plan.expected_cost > 0
        assert plan.expected_latency > 0
