"""Tabular TD-learning baselines: Q-learning (off-policy) and SARSA (on-policy).

Both share the same epsilon-greedy Q-table machinery; they differ only in
their bootstrap target (see `TabularQLearningAgent.update` vs
`SARSAAgent.update`). State is the discretized tuple from
src.utils.discretization.discretize_state.
"""

from collections import defaultdict
from typing import Tuple
import pickle

import numpy as np

from src.envs.tools import NUM_TOOLS

NUM_ACTIONS = NUM_TOOLS + 1  # tools + STOP

StateKey = Tuple[int, Tuple[int, ...], int, int]


class _TabularTDAgentBase:
    name = "tabular_td_base"

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_episodes: int = 4000,
        seed: int = None,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.q_table = defaultdict(lambda: np.zeros(NUM_ACTIONS, dtype=np.float64))
        self._rng = np.random.default_rng(seed)
        self.episode_count = 0

    def epsilon(self) -> float:
        frac = min(1.0, self.episode_count / max(1, self.epsilon_decay_episodes))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def start_episode(self) -> None:
        self.episode_count += 1

    def select_action(self, state: StateKey, greedy: bool = False) -> int:
        if not greedy and self._rng.random() < self.epsilon():
            return int(self._rng.integers(0, NUM_ACTIONS))
        q_values = self.q_table[state]
        return int(np.argmax(q_values))

    def update(self, state, action, reward, next_state, terminated, next_action=None):
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        """Save Q-table to disk."""
        # Convert defaultdict to regular dict for pickling
        q_table_dict = dict(self.q_table)
        save_data = {
            'q_table': q_table_dict,
            'episode_count': self.episode_count,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay_episodes': self.epsilon_decay_episodes
        }
        with open(path, 'wb') as f:
            pickle.dump(save_data, f)
    
    def load(self, path: str) -> None:
        """Load Q-table from disk."""
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        
        # Restore Q-table as defaultdict
        loaded_dict = save_data['q_table']
        self.q_table = defaultdict(lambda: np.zeros(NUM_ACTIONS, dtype=np.float64))
        self.q_table.update(loaded_dict)
        
        # Restore training state
        self.episode_count = save_data['episode_count']
        self.alpha = save_data['alpha']
        self.gamma = save_data['gamma']
        self.epsilon_start = save_data['epsilon_start']
        self.epsilon_end = save_data['epsilon_end']
        self.epsilon_decay_episodes = save_data['epsilon_decay_episodes']


class TabularQLearningAgent(_TabularTDAgentBase):
    """Off-policy TD control: bootstraps from max_a' Q(s', a')."""

    name = "q_learning"

    def update(self, state, action, reward, next_state, terminated, next_action=None):
        current_q = self.q_table[state][action]
        target = reward
        if not terminated:
            target += self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)


class SARSAAgent(_TabularTDAgentBase):
    """On-policy TD control: bootstraps from Q(s', a') for the action the
    policy actually takes next (requires next_action to be passed in)."""

    name = "sarsa"

    def update(self, state, action, reward, next_state, terminated, next_action=None):
        current_q = self.q_table[state][action]
        target = reward
        if not terminated:
            assert next_action is not None, "SARSA update requires next_action"
            target += self.gamma * self.q_table[next_state][next_action]
        self.q_table[state][action] = current_q + self.alpha * (target - current_q)
