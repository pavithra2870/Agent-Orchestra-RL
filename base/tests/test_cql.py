import numpy as np
import torch

from src.envs.tool_orchestration_env import ToolOrchestrationEnv
from src.agents.cql_agent import CQLAgent
from src.agents.rule_based import RuleBasedAgent
from src.utils.trajectory_logger import BehaviorPolicy, generate_logged_dataset


def _small_dataset():
    env = ToolOrchestrationEnv(seed=1)
    agent = RuleBasedAgent()
    policies = [BehaviorPolicy("rule_based", agent.act, agent.reset, 1.0)]
    dataset, records = generate_logged_dataset(env, policies, n_episodes=50, seed=1)
    return dataset


def test_generate_logged_dataset_shapes():
    dataset = _small_dataset()
    n = len(dataset["states"])
    assert n > 0
    assert dataset["actions"].shape == (n,)
    assert dataset["rewards"].shape == (n,)
    assert dataset["next_states"].shape == dataset["states"].shape
    assert dataset["dones"].shape == (n,)


def test_cql_train_step_runs_and_act_greedy_valid():
    dataset = _small_dataset()
    agent = CQLAgent(seed=0)
    n = len(dataset["states"])
    rng = np.random.default_rng(0)
    for _ in range(30):
        idx = rng.integers(0, n, size=16)
        batch = (
            torch.as_tensor(dataset["states"][idx], dtype=torch.float32),
            torch.as_tensor(dataset["actions"][idx], dtype=torch.long),
            torch.as_tensor(dataset["rewards"][idx], dtype=torch.float32),
            torch.as_tensor(dataset["next_states"][idx], dtype=torch.float32),
            torch.as_tensor(dataset["dones"][idx], dtype=torch.float32),
        )
        stats = agent.train_step(batch)
        assert "loss" in stats

    env = ToolOrchestrationEnv(seed=2)
    obs, _ = env.reset(seed=2)
    action = agent.act_greedy(obs)
    assert env.action_space.contains(action)
