"""DQN agent with a learned task embedding, experience replay, and a target
network -- a from-scratch PyTorch implementation (stable-baselines3 is not
installed in this environment; see the Phase 2 report for why a from-scratch
implementation was chosen instead of adding that dependency).

=== "Richer state" design note (Phase 2 spec item 1) ===
Rather than building a separate environment wrapper that changes the
observation format, the task-id portion of ToolOrchestrationEnv's existing
flat observation (a one-hot block) is decoded back to an integer task id and
fed through an `nn.Embedding` layer *inside* the Q-network. The embedding
vector is a small, dense, *learned* representation of the task -- trained
end-to-end via the TD loss, exactly matching the spec's "simple learned
embedding" option. This keeps the environment's public observation API
identical to Phase 1 (so the tabular baselines and DQN can share one env),
while still giving the function approximator a genuinely richer,
higher-dimensional, learned state representation instead of a raw one-hot.
"""

import random
from collections import deque
from typing import Deque, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.envs.tools import NUM_TOOLS
from src.envs.tasks import NUM_TASKS

NUM_ACTIONS = NUM_TOOLS + 1


class TaskConditionedQNetwork(nn.Module):
    def __init__(self, num_tasks: int = NUM_TASKS, num_tools: int = NUM_TOOLS, embed_dim: int = 8, hidden_dim: int = 64, include_metadata: bool = True):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_tools = num_tools
        self.include_metadata = include_metadata
        self.task_embedding = nn.Embedding(num_tasks, embed_dim)
        
        # OLD: attempted_mask + succeeded_mask + budget_norm + steps_norm = 2*num_tools + 2
        # NEW: + required_tools + min_path_len_norm + num_paths_norm + task_max_budget_norm + task_max_steps_norm = num_tools + 4
        if include_metadata:
            rest_dim = 2 * num_tools + 2 + num_tools + 4  # 3*num_tools + 6
        else:
            rest_dim = 2 * num_tools + 2
        
        self.net = nn.Sequential(
            nn.Linear(embed_dim + rest_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_tools + 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        task_onehot = obs[:, : self.num_tasks]
        task_id = torch.argmax(task_onehot, dim=1)
        rest = obs[:, self.num_tasks :]
        embed = self.task_embedding(task_id)
        x = torch.cat([embed, rest], dim=1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, seed: int = None):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.buffer: Deque[Tuple] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def __len__(self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        batch = self._rng.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.as_tensor(np.array(states), dtype=torch.float32),
            torch.as_tensor(np.array(actions), dtype=torch.long),
            torch.as_tensor(np.array(rewards), dtype=torch.float32),
            torch.as_tensor(np.array(next_states), dtype=torch.float32),
            torch.as_tensor(np.array(dones), dtype=torch.float32),
        )


class DQNAgent:
    name = "dqn"

    def __init__(
        self,
        obs_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 800,
        target_update_freq: int = 500,  # gradient steps between hard target syncs
        embed_dim: int = 8,
        hidden_dim: int = 64,
        include_metadata: bool = True,  # NEW: whether to expect metadata features
        seed: int = None,
    ):
        torch.manual_seed(seed if seed is not None else 0)
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.target_update_freq = target_update_freq
        self.include_metadata = include_metadata

        self.q_network = TaskConditionedQNetwork(embed_dim=embed_dim, hidden_dim=hidden_dim, include_metadata=include_metadata)
        self.target_network = TaskConditionedQNetwork(embed_dim=embed_dim, hidden_dim=hidden_dim, include_metadata=include_metadata)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self._rng = np.random.default_rng(seed)
        self.episode_count = 0
        self.gradient_steps = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.episode_count / max(1, self.epsilon_decay_episodes))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def start_episode(self) -> None:
        self.episode_count += 1

    def select_action(self, obs: np.ndarray, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon():
            return int(self._rng.integers(0, NUM_ACTIONS))
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(obs_t)
            return int(torch.argmax(q_values, dim=1).item())

    def train_step(self, batch) -> float:
        states, actions, rewards, next_states, dones = batch

        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = F.smooth_l1_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.gradient_steps += 1
        if self.gradient_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.q_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)
