"""Tabular Q-learning (off-policy) and SARSA (on-policy) with action masking.

The Q-table needs a small hashable state, so the observation vector is compressed
(see `discretize`). This is a deliberate baseline simplification: it drops the shipment
status, payment method and last-action detail, and bins the continuous numbers.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from typing import Optional, Tuple

import numpy as np

from ecommerce.agents.base import Agent
from ecommerce.env.state import FEATURE_INDEX, NUM_ACTIONS
from ecommerce.sim.world import GOALS

_BINARY = (
    [f"requested:{g}" for g in GOALS]
    + [f"handled:{g}" for g in GOALS]
    + [f"needs_escalation:{g}" for g in GOALS]
    + ["customer_verified", "order_known", "order_verified", "order_ambiguous", "order_rejected",
       "shipment_known", "payment_known", "pay_elig_known", "pay_elig_yes", "refund_elig_known",
       "refund_elig_yes", "refund_elig_human", "cancel_elig_known", "cancel_elig_yes", "return_elig_known",
       "return_elig_yes", "return_elig_human", "payment_method_missing", "return_info_missing", "last_failed",
       "low_confidence"]
)
_BIN_IDX = [FEATURE_INDEX[n] for n in _BINARY]
_I = FEATURE_INDEX


def discretize(obs: np.ndarray) -> Tuple:
    bits = tuple(int(obs[i] > 0.5) for i in _BIN_IDX)
    steps = min(3, int(obs[_I["steps_used"]] * 4))
    cost = min(2, int(obs[_I["cost_used"]] * 3))
    scope = 0 if obs[_I["scope_score"]] < 0.38 else (1 if obs[_I["scope_score"]] < 0.6 else 2)
    clar = int(round(obs[_I["clarifications_used"]] * 2))
    fails = min(3, int(round(obs[_I["consecutive_failures"]] * 3)))
    return bits + (steps, cost, scope, clar, fails)


class _TabularBase(Agent):
    def __init__(self, alpha=0.2, gamma=0.97, eps_start=1.0, eps_end=0.05, eps_decay_episodes=2500, seed=0):
        self.alpha, self.gamma = alpha, gamma
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_episodes
        self.q = defaultdict(lambda: np.zeros(NUM_ACTIONS))
        self._rng = np.random.default_rng(seed)
        self.episode = 0
        self.training = False

    def epsilon(self) -> float:
        frac = min(1.0, self.episode / max(1, self.eps_decay))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _greedy(self, key, mask) -> int:
        q = np.where(mask, self.q[key], -np.inf)
        best = np.flatnonzero(q == q.max())          # random tie-break so unseen states do not always pick action 0
        return int(self._rng.choice(best))

    def act(self, obs, mask, info=None) -> int:
        key = discretize(obs)
        if self.training and self._rng.random() < self.epsilon():
            return int(self._rng.choice(np.flatnonzero(mask)))
        return self._greedy(key, mask)

    def action_scores(self, obs, mask):
        return np.where(mask, self.q[discretize(obs)], np.nan)

    def save(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"q": dict(self.q), "episode": self.episode, "alpha": self.alpha, "gamma": self.gamma}, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            d = pickle.load(f)
        self.q = defaultdict(lambda: np.zeros(NUM_ACTIONS), d["q"])
        self.episode, self.alpha, self.gamma = d["episode"], d["alpha"], d["gamma"]
        self.training = False


class QLearningAgent(_TabularBase):
    name = "q_learning"

    def update(self, obs, a, r, next_obs, done, next_mask, next_a=None):
        key, nkey = discretize(obs), discretize(next_obs)
        target = r if done else r + self.gamma * np.max(np.where(next_mask, self.q[nkey], -np.inf))
        self.q[key][a] += self.alpha * (target - self.q[key][a])


class SarsaAgent(_TabularBase):
    name = "sarsa"

    def update(self, obs, a, r, next_obs, done, next_mask, next_a=None):
        key, nkey = discretize(obs), discretize(next_obs)
        target = r if done else r + self.gamma * self.q[nkey][next_a]
        self.q[key][a] += self.alpha * (target - self.q[key][a])
