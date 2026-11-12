"""DQN with experience replay, a target network and action masking.

Masking matters in two places: (1) when acting, only legal actions are chosen; (2) when
building the learning target, the max over next-state actions only counts LEGAL actions.
Without (2) the agent would "bootstrap" from actions it is never allowed to take.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ecommerce.agents.base import Agent, masked_argmax
from ecommerce.agents.nn import mask_logits, mlp, to_tensor
from ecommerce.env.state import NUM_ACTIONS, OBS_DIM


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        self.buf = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(self, *transition) -> None:
        self.buf.append(transition)

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, n: int):
        idx = self.rng.integers(0, len(self.buf), size=n)
        s, a, r, s2, d, m2 = zip(*[self.buf[i] for i in idx])
        return (to_tensor(np.stack(s)), to_tensor(a, torch.long), to_tensor(r), to_tensor(np.stack(s2)),
                to_tensor(d), to_tensor(np.stack(m2), torch.bool))


class DQNAgent(Agent):
    name = "dqn"

    def __init__(self, hidden=128, lr=1e-3, gamma=0.97, batch_size=64, buffer_size=50_000, target_update=250,
                 eps_start=1.0, eps_end=0.05, eps_decay_episodes=2500, warmup=500, double=False, seed=0):
        torch.manual_seed(seed)
        self.hp = dict(hidden=hidden, lr=lr, gamma=gamma, batch_size=batch_size, buffer_size=buffer_size,
                       target_update=target_update, eps_start=eps_start, eps_end=eps_end,
                       eps_decay_episodes=eps_decay_episodes, warmup=warmup, double=double, seed=seed)
        self.gamma, self.batch_size, self.target_update, self.warmup, self.double = gamma, batch_size, target_update, warmup, double
        self.eps_start, self.eps_end, self.eps_decay = eps_start, eps_end, eps_decay_episodes
        self.q = mlp(OBS_DIM, hidden, NUM_ACTIONS)
        self.target = mlp(OBS_DIM, hidden, NUM_ACTIONS)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size, seed)
        self._rng = np.random.default_rng(seed)
        self.episode, self.grad_steps, self.training = 0, 0, False

    def epsilon(self) -> float:
        frac = min(1.0, self.episode / max(1, self.eps_decay))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def q_values(self, obs) -> np.ndarray:
        with torch.no_grad():
            return self.q(to_tensor(obs).unsqueeze(0)).squeeze(0).numpy()

    def act(self, obs, mask, info=None) -> int:
        if self.training and self._rng.random() < self.epsilon():
            return int(self._rng.choice(np.flatnonzero(mask)))
        return masked_argmax(self.q_values(obs), mask)

    def action_scores(self, obs, mask):
        return np.where(mask, self.q_values(obs), np.nan)

    def remember(self, s, a, r, s2, done, mask2) -> None:
        self.buffer.push(s, a, r, s2, float(done), mask2)

    def train_step(self) -> Optional[float]:
        if len(self.buffer) < max(self.batch_size, self.warmup):
            return None
        s, a, r, s2, d, m2 = self.buffer.sample(self.batch_size)
        q = self.q(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.double:
                best = mask_logits(self.q(s2), m2).argmax(dim=1, keepdim=True)
                nxt = self.target(s2).gather(1, best).squeeze(1)
            else:
                nxt = mask_logits(self.target(s2), m2).max(dim=1).values
            y = r + self.gamma * nxt * (1.0 - d)
        loss = F.smooth_l1_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.grad_steps += 1
        if self.grad_steps % self.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({"state": self.q.state_dict(), "hp": self.hp, "obs_dim": OBS_DIM, "n_actions": NUM_ACTIONS}, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["obs_dim"] == OBS_DIM and ck["n_actions"] == NUM_ACTIONS, "checkpoint was trained on a different state layout"
        self.q.load_state_dict(ck["state"])
        self.target.load_state_dict(ck["state"])
        self.training = False
