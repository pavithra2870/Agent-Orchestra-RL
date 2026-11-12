"""Conservative Q-Learning (CQL), trained purely on a fixed logged dataset --
no live environment interaction during training (the standard offline-RL
setup; Kumar et al., 2020).

CQL modifies the standard Q-learning (Bellman) loss by adding a regularizer
that pushes down Q-values for actions *not* seen in the dataset at a given
state, and pushes up Q-values for the action the dataset actually took:

    L_CQL = alpha * E_s[ logsumexp_a Q(s, a) - Q(s, a_data) ]  +  L_Bellman

The logsumexp term approximates "how much probability mass would a
Q-greedy policy put on out-of-distribution actions"; penalizing it counters
the classic offline-RL failure mode where bootstrapping off of unseen
(state, action) pairs overestimates their value, causing the learned policy
to drift to actions the data never demonstrated. Reuses the same
TaskConditionedQNetwork architecture as DQN so the two are directly
comparable.
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.agents.dqn_agent import TaskConditionedQNetwork


class CQLAgent:
    name = "cql"

    def __init__(
        self,
        lr: float = 1e-3,
        gamma: float = 0.95,
        alpha: float = 1.0,  # weight on the CQL conservative regularizer
        target_update_freq: int = 500,
        embed_dim: int = 8,
        hidden_dim: int = 64,
        include_metadata: bool = True,  # NEW: whether to expect metadata features
        seed: int = None,
    ):
        torch.manual_seed(seed if seed is not None else 0)
        self.gamma = gamma
        self.alpha = alpha
        self.target_update_freq = target_update_freq
        self.include_metadata = include_metadata

        self.q_network = TaskConditionedQNetwork(embed_dim=embed_dim, hidden_dim=hidden_dim, include_metadata=include_metadata)
        self.target_network = TaskConditionedQNetwork(embed_dim=embed_dim, hidden_dim=hidden_dim, include_metadata=include_metadata)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.gradient_steps = 0

    def act_greedy(self, obs: np.ndarray) -> int:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            q_values = self.q_network(obs_t)
            return int(torch.argmax(q_values, dim=1).item())

    def train_step(self, batch) -> dict:
        states, actions, rewards, next_states, dones = batch

        all_q = self.q_network(states)  # (B, num_actions)
        q_data = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(dim=1).values
            targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        bellman_loss = F.smooth_l1_loss(q_data, targets)

        # Conservative regularizer: logsumexp over all actions minus the
        # dataset action's Q-value, averaged over the batch.
        logsumexp_q = torch.logsumexp(all_q, dim=1)
        conservative_loss = (logsumexp_q - q_data).mean()

        loss = bellman_loss + self.alpha * conservative_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.gradient_steps += 1
        if self.gradient_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {
            "loss": float(loss.item()),
            "bellman_loss": float(bellman_loss.item()),
            "conservative_loss": float(conservative_loss.item()),
        }

    def save(self, path: str) -> None:
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        self.q_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)
