"""PPO (Proximal Policy Optimization) agent -- from-scratch PyTorch
implementation, consistent with the DQN choice in Phase 2 (no
stable-baselines3 dependency; see the Phase 3 report).

Shares the "task-embedding inside the network" design from
src/agents/dqn_agent.py: the actor-critic trunk decodes the task one-hot
block of ToolOrchestrationEnv's observation back to an id and embeds it,
so the environment's observation API stays identical across all agents.

Single-environment, sequential-episode rollout collection (no vectorized
envs) -- simpler to read and reason about, and this environment is cheap
enough that wall-clock isn't a concern. GAE(lambda) is used for advantage
estimation; episode boundaries reset the GAE recursion (no bootstrapping
across a terminal transition).
"""

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from src.envs.tools import NUM_TOOLS
from src.envs.tasks import NUM_TASKS

NUM_ACTIONS = NUM_TOOLS + 1


class ActorCriticNetwork(nn.Module):
    """`num_tools` sizes the observation parsing (attempted/succeeded masks
    are always over the environment's full tool set, even in the multi-agent
    env where a given role can only *act* on a subset of them). `num_actions`
    sizes the policy head independently, since triage and execution have
    different action spaces over that same shared observation."""

    def __init__(
        self,
        num_tasks: int = NUM_TASKS,
        num_tools: int = NUM_TOOLS,
        num_actions: Optional[int] = None,
        embed_dim: int = 8,
        hidden_dim: int = 64,
        include_metadata: bool = True,  # NEW: whether to expect metadata features
    ):
        super().__init__()
        self.num_tasks = num_tasks
        self.include_metadata = include_metadata
        if num_actions is None:
            num_actions = num_tools + 1
        
        # OLD: attempted_mask + succeeded_mask + budget_norm + steps_norm = 2*num_tools + 2
        # NEW: + required_tools + min_path_len_norm + num_paths_norm + task_max_budget_norm + task_max_steps_norm = num_tools + 4
        if include_metadata:
            rest_dim = 2 * num_tools + 2 + num_tools + 4  # 3*num_tools + 6
        else:
            rest_dim = 2 * num_tools + 2
        
        self.task_embedding = nn.Embedding(num_tasks, embed_dim)
        self.trunk = nn.Sequential(
            nn.Linear(embed_dim + rest_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden_dim, num_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        task_onehot = obs[:, : self.num_tasks]
        task_id = torch.argmax(task_onehot, dim=1)
        rest = obs[:, self.num_tasks :]
        embed = self.task_embedding(task_id)
        return self.trunk(torch.cat([embed, rest], dim=1))

    def forward(self, obs: torch.Tensor):
        features = self._features(obs)
        logits = self.policy_head(features)
        value = self.value_head(features).squeeze(-1)
        return logits, value


class RolloutBuffer:
    """Accumulates one rollout (n_steps of single-environment interaction)."""

    def __init__(self):
        self.observations: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []

    def add(self, obs, action, log_prob, value, reward, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.observations)


class PPOAgent:
    name = "ppo"

    def __init__(
        self,
        num_tools: int = NUM_TOOLS,
        num_actions: Optional[int] = None,
        hidden_dim: int = 64,
        embed_dim: int = 8,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        vf_coef: float = 0.5,
        ent_coef: float = 0.01,
        update_epochs: int = 4,
        minibatch_size: int = 64,
        include_metadata: bool = True,  # NEW: whether to expect metadata features
        seed: Optional[int] = None,
    ):
        torch.manual_seed(seed if seed is not None else 0)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.include_metadata = include_metadata

        self.network = ActorCriticNetwork(
            num_tools=num_tools, 
            num_actions=num_actions, 
            embed_dim=embed_dim, 
            hidden_dim=hidden_dim,
            include_metadata=include_metadata
        )
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self._rng = np.random.default_rng(seed)

    def act(self, obs: np.ndarray):
        """Stochastic action for rollout collection. Returns (action, log_prob, value)."""
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, value = self.network(obs_t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            return int(action.item()), float(dist.log_prob(action).item()), float(value.item())

    def act_greedy(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, _ = self.network(obs_t)
            return int(torch.argmax(logits, dim=1).item())

    def compute_gae(self, rollout: RolloutBuffer, last_value: float):
        """Generalized Advantage Estimation. `last_value` bootstraps the final
        step if the rollout was cut off mid-episode (0.0 if it ended on a
        terminal transition)."""
        n = len(rollout)
        advantages = np.zeros(n, dtype=np.float32)
        gae = 0.0
        values = rollout.values + [last_value]
        for t in reversed(range(n)):
            not_done = 1.0 - float(rollout.dones[t])
            delta = rollout.rewards[t] + self.gamma * values[t + 1] * not_done - values[t]
            gae = delta + self.gamma * self.gae_lambda * not_done * gae
            advantages[t] = gae
        returns = advantages + np.array(rollout.values, dtype=np.float32)
        return advantages, returns

    def update(self, rollout: RolloutBuffer, last_value: float):
        advantages, returns = self.compute_gae(rollout, last_value)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs = torch.as_tensor(np.array(rollout.observations), dtype=torch.float32)
        actions = torch.as_tensor(np.array(rollout.actions), dtype=torch.long)
        old_log_probs = torch.as_tensor(np.array(rollout.log_probs), dtype=torch.float32)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
        returns_t = torch.as_tensor(returns, dtype=torch.float32)

        n = len(rollout)
        indices = np.arange(n)
        total_policy_loss, total_value_loss, total_entropy = 0.0, 0.0, 0.0
        n_updates = 0

        for _ in range(self.update_epochs):
            self._rng.shuffle(indices)
            for start in range(0, n, self.minibatch_size):
                mb_idx = indices[start : start + self.minibatch_size]
                mb_idx_t = torch.as_tensor(mb_idx, dtype=torch.long)

                logits, values = self.network(obs[mb_idx_t])
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions[mb_idx_t])
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - old_log_probs[mb_idx_t])
                surr1 = ratio * advantages_t[mb_idx_t]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t[mb_idx_t]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = ((values - returns_t[mb_idx_t]) ** 2).mean()

                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                total_policy_loss += float(policy_loss.item())
                total_value_loss += float(value_loss.item())
                total_entropy += float(entropy.item())
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / n_updates,
            "value_loss": total_value_loss / n_updates,
            "entropy": total_entropy / n_updates,
        }

    def save(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)

    def load(self, path: str) -> None:
        self.network.load_state_dict(torch.load(path, map_location="cpu"))
