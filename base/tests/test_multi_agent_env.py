import pytest

from src.envs.multi_agent_env import MultiAgentToolOrchestrationEnv
from src.envs.tasks import TASK_REGISTRY


def make_env(**kwargs):
    return MultiAgentToolOrchestrationEnv(seed=42, **kwargs)


def test_reset_returns_dict_obs_for_both_roles():
    env = make_env()
    obs_dict, info = env.reset(seed=1, options={"task_id": 2})
    assert set(obs_dict.keys()) == {"triage", "execution"}
    assert obs_dict["triage"].shape == (env.obs_dim,)
    (obs_dict["triage"] == obs_dict["execution"]).all()


def test_light_tool_action_does_not_require_execution_action():
    env = make_env(force_deterministic_success=True)
    env.reset(seed=1, options={"task_id": 0})  # check_order_status: db_lookup alone succeeds
    light_idx = env.LIGHT_TOOLS.index("db_lookup")
    obs_dict, reward_dict, terminated, truncated, info = env.step({"triage": light_idx})
    assert terminated
    assert info["success"] is True
    assert reward_dict["triage"] == reward_dict["execution"]


def test_delegate_requires_and_uses_execution_action():
    env = make_env(force_deterministic_success=True)
    env.reset(seed=1, options={"task_id": 2})  # process_refund: verify_identity + payment_api (both heavy)
    heavy_idx = env.HEAVY_TOOLS.index("verify_identity")
    obs_dict, reward_dict, terminated, truncated, info = env.step(
        {"triage": env.DELEGATE_ACTION, "execution": heavy_idx}
    )
    assert info["delegated"] is True
    assert info["tool"] == "verify_identity"
    assert not terminated
    assert env.cumulative_cost > 0


def test_delegate_without_execution_action_raises():
    env = make_env()
    env.reset(seed=1, options={"task_id": 2})
    with pytest.raises(AssertionError):
        env.step({"triage": env.DELEGATE_ACTION})


def test_stop_action_ends_episode_as_failure():
    env = make_env()
    env.reset(seed=1, options={"task_id": 0})
    obs_dict, reward_dict, terminated, truncated, info = env.step({"triage": env.TRIAGE_STOP_ACTION})
    assert terminated
    assert info["success"] is False
    assert reward_dict["triage"] == pytest.approx(-env.failure_penalty)


def test_full_task_via_delegation_and_light_tools():
    env = make_env(force_deterministic_success=True)
    env.reset(seed=1, options={"task_id": 2})  # process_refund: verify_identity + payment_api
    obs_dict, reward_dict, terminated, truncated, info = env.step(
        {"triage": env.DELEGATE_ACTION, "execution": env.HEAVY_TOOLS.index("verify_identity")}
    )
    assert not terminated
    obs_dict, reward_dict, terminated, truncated, info = env.step(
        {"triage": env.DELEGATE_ACTION, "execution": env.HEAVY_TOOLS.index("payment_api")}
    )
    assert terminated
    assert info["success"] is True


def test_light_and_heavy_tool_sets_partition_all_tools():
    env = make_env()
    all_tools = set(env.LIGHT_TOOLS) | set(env.HEAVY_TOOLS)
    assert len(env.LIGHT_TOOLS) + len(env.HEAVY_TOOLS) == len(all_tools)  # no overlap
    from src.envs.tools import TOOL_NAMES
    assert all_tools == set(TOOL_NAMES)
