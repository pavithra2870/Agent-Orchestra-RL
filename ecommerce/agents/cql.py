"""Conservative Q-Learning (offline). Trains ONLY on a fixed log of past conversations.

Loss = TD error (target max over LEGAL next actions) + alpha * (logsumexp_a Q(s,a) - Q(s, a_logged)).
The second term pushes down Q-values of actions the log never shows, so the policy does not
get over-optimistic about untried actions. The logsumexp only runs over legal actions.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

from ecommerce.agents.base import Agent, masked_argmax
from ecommerce.agents.nn import mask_logits, mlp, to_tensor
from ecommerce.env.state import NUM_ACTIONS, OBS_DIM


class OfflineDataset:
    def __init__(self, obs, act, rew, next_obs, done, mask, next_mask):
        self.obs, self.act, self.rew = to_tensor(obs), to_tensor(act, torch.long), to_tensor(rew)
        self.next_obs, self.done = to_tensor(next_obs), to_tensor(done)
        self.mask, self.next_mask = to_tensor(mask, torch.bool), to_tensor(next_mask, torch.bool)

    def __len__(self) -> int:
        return len(self.obs)

    @classmethod
    def load(cls, path: str) -> "OfflineDataset":
        d = np.load(path)
        return cls(d["obs"], d["act"], d["rew"], d["next_obs"], d["done"], d["mask"], d["next_mask"])

    def sample(self, n: int, rng: np.random.Generator):
        i = torch.as_tensor(rng.integers(0, len(self), size=n))
        return (self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i], self.mask[i], self.next_mask[i])


class CQLAgent(Agent):
    name = "cql"

    def __init__(self, hidden=128, lr=5e-4, gamma=0.97, alpha=1.0, batch_size=128, target_update=200, seed=0):
        torch.manual_seed(seed)
        self.hp = dict(hidden=hidden, lr=lr, gamma=gamma, alpha=alpha, batch_size=batch_size,
                       target_update=target_update, seed=seed)
        self.gamma, self.alpha, self.batch_size, self.target_update = gamma, alpha, batch_size, target_update
        self.q = mlp(OBS_DIM, hidden, NUM_ACTIONS)
        self.target = mlp(OBS_DIM, hidden, NUM_ACTIONS)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=lr)
        self._rng = np.random.default_rng(seed)
        self.steps = 0
        self.training = False

    def q_values(self, obs) -> np.ndarray:
        with torch.no_grad():
            return self.q(to_tensor(obs).unsqueeze(0)).squeeze(0).numpy()

    def act(self, obs, mask, info=None) -> int:
        return masked_argmax(self.q_values(obs), mask)

    def action_scores(self, obs, mask):
        return np.where(mask, self.q_values(obs), np.nan)

    def train_step(self, data: OfflineDataset) -> Dict[str, float]:
        s, a, r, s2, d, m, m2 = data.sample(self.batch_size, self._rng)
        q_all = self.q(s)
        q_data = q_all.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            nxt = mask_logits(self.target(s2), m2).max(dim=1).values
            y = r + self.gamma * nxt * (1.0 - d)
        td = F.smooth_l1_loss(q_data, y)
        cons = (torch.logsumexp(mask_logits(q_all, m), dim=1) - q_data).mean()
        loss = td + self.alpha * cons
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), 10.0)
        self.opt.step()
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target.load_state_dict(self.q.state_dict())
        return {"loss": loss.item(), "td": td.item(), "conservative": cons.item()}

    def save(self, path: str) -> None:
        torch.save({"state": self.q.state_dict(), "hp": self.hp, "obs_dim": OBS_DIM, "n_actions": NUM_ACTIONS}, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["obs_dim"] == OBS_DIM and ck["n_actions"] == NUM_ACTIONS, "checkpoint was trained on a different state layout"
        self.q.load_state_dict(ck["state"])
        self.target.load_state_dict(ck["state"])
        self.training = False
