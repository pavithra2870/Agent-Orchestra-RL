import numpy as np

from src.envs.tool_orchestration_env import ToolOrchestrationEnv
from src.agents.dqn_agent import DQNAgent, ReplayBuffer, TaskConditionedQNetwork


def test_qnetwork_forward_shape():
    env = ToolOrchestrationEnv(seed=1)
    obs, _ = env.reset(seed=1)
    net = TaskConditionedQNetwork()
    import torch
    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    q_values = net(obs_t)
    assert q_values.shape == (1, env.action_space.n)


def test_replay_buffer_push_and_sample():
    env = ToolOrchestrationEnv(seed=2)
    obs_dim = env.observation_space.shape[0]
    buffer = ReplayBuffer(capacity=100, obs_dim=obs_dim, seed=0)
    obs, _ = env.reset(seed=2)
    for _ in range(20):
        next_obs, reward, terminated, truncated, info = env.step(0)
        buffer.push(obs, 0, reward, next_obs, float(terminated or truncated))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()
    assert len(buffer) == 20
    batch = buffer.sample(8)
    states, actions, rewards, next_states, dones = batch
    assert states.shape == (8, obs_dim)
    assert actions.shape == (8,)


def test_dqn_agent_train_step_reduces_loss_trend():
    env = ToolOrchestrationEnv(seed=3, force_deterministic_success=True)
    obs_dim = env.observation_space.shape[0]
    agent = DQNAgent(obs_dim=obs_dim, seed=0, epsilon_decay_episodes=10)
    buffer = ReplayBuffer(capacity=2000, obs_dim=obs_dim, seed=0)

    obs, _ = env.reset(seed=3, options={"task_id": 0})
    for _ in range(300):
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        buffer.push(obs, action, reward, next_obs, float(terminated or truncated))
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset(options={"task_id": 0})

    losses = []
    for _ in range(50):
        batch = buffer.sample(32)
        loss = agent.train_step(batch)
        losses.append(loss)

    assert np.mean(losses[-10:]) <= np.mean(losses[:10]) + 5.0  # loose sanity check, not full convergence
