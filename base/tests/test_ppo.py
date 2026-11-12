import numpy as np

from src.envs.tool_orchestration_env import ToolOrchestrationEnv
from src.agents.ppo_agent import PPOAgent, RolloutBuffer


def test_ppo_act_and_act_greedy_return_valid_actions():
    env = ToolOrchestrationEnv(seed=1)
    obs, _ = env.reset(seed=1)
    agent = PPOAgent(seed=0)
    action, log_prob, value = agent.act(obs)
    assert env.action_space.contains(action)
    assert isinstance(log_prob, float)
    assert isinstance(value, float)
    greedy_action = agent.act_greedy(obs)
    assert env.action_space.contains(greedy_action)


def test_ppo_gae_shapes():
    env = ToolOrchestrationEnv(seed=2, force_deterministic_success=True)
    agent = PPOAgent(seed=0)
    obs, _ = env.reset(seed=2, options={"task_id": 0})
    rollout = RolloutBuffer()
    for _ in range(20):
        action, log_prob, value = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rollout.add(obs, action, log_prob, value, reward, done)
        obs = next_obs
        if done:
            obs, _ = env.reset(options={"task_id": 0})
    advantages, returns = agent.compute_gae(rollout, last_value=0.0)
    assert advantages.shape == (20,)
    assert returns.shape == (20,)


def test_ppo_update_runs_without_error_and_changes_weights():
    env = ToolOrchestrationEnv(seed=3, force_deterministic_success=True)
    agent = PPOAgent(seed=0, minibatch_size=8, update_epochs=2)
    obs, _ = env.reset(seed=3, options={"task_id": 0})
    rollout = RolloutBuffer()
    for _ in range(32):
        action, log_prob, value = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rollout.add(obs, action, log_prob, value, reward, done)
        obs = next_obs
        if done:
            obs, _ = env.reset(options={"task_id": 0})

    before = agent.network.policy_head.weight.clone()
    stats = agent.update(rollout, last_value=0.0)
    after = agent.network.policy_head.weight
    assert not np.allclose(before.detach().numpy(), after.detach().numpy())
    assert "policy_loss" in stats and "value_loss" in stats
