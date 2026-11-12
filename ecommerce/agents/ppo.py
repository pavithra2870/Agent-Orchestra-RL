"""PPO (clipped objective, GAE) with action masking.

The policy's logits for illegal actions are set to a huge negative number, so they get
probability ~0 both when sampling and when computing the probability ratio in the update.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from ecommerce.agents.base import Agent, masked_argmax
from ecommerce.agents.nn import mask_logits, mlp, to_tensor
from ecommerce.env.state import NUM_ACTIONS, OBS_DIM


class ActorCritic(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.trunk = mlp(OBS_DIM, hidden, hidden, layers=1)
        self.act = nn.Sequential(nn.ReLU(), nn.Linear(hidden, NUM_ACTIONS))
        self.val = nn.Sequential(nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        h = self.trunk(x)
        return self.act(h), self.val(h).squeeze(-1)


class PPOAgent(Agent):
    name = "ppo"

    def __init__(self, hidden=128, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2, ent_coef=0.01,
                 vf_coef=0.5, update_epochs=4, minibatch=128, rollout_steps=1024, seed=0):
        torch.manual_seed(seed)
        self.hp = dict(hidden=hidden, lr=lr, gamma=gamma, gae_lambda=gae_lambda, clip_eps=clip_eps,
                       ent_coef=ent_coef, vf_coef=vf_coef, update_epochs=update_epochs, minibatch=minibatch,
                       rollout_steps=rollout_steps, seed=seed)
        self.gamma, self.lam, self.clip, self.ent, self.vf = gamma, gae_lambda, clip_eps, ent_coef, vf_coef
        self.epochs, self.minibatch, self.rollout_steps = update_epochs, minibatch, rollout_steps
        self.net = ActorCritic(hidden)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self._rng = np.random.default_rng(seed)
        self.reset_rollout()
        self.training = False

    # ---- acting -----------------------------------------------------------
    def _dist(self, obs, mask):
        logits, value = self.net(to_tensor(obs).unsqueeze(0))
        logits = mask_logits(logits, to_tensor(mask, torch.bool).unsqueeze(0))
        return torch.distributions.Categorical(logits=logits), value

    def act(self, obs, mask, info=None) -> int:
        with torch.no_grad():
            dist, _ = self._dist(obs, mask)
            if self.training:
                return int(dist.sample().item())
            return masked_argmax(dist.probs.squeeze(0).numpy(), mask)

    def act_train(self, obs, mask):
        with torch.no_grad():
            dist, value = self._dist(obs, mask)
            a = dist.sample()
            return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    def action_scores(self, obs, mask):
        with torch.no_grad():
            dist, _ = self._dist(obs, mask)
            return np.where(mask, dist.probs.squeeze(0).numpy(), np.nan)

    # ---- learning ---------------------------------------------------------
    def reset_rollout(self) -> None:
        self.buf: Dict[str, List] = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done", "mask")}

    def store(self, obs, act, logp, val, rew, done, mask) -> None:
        for k, v in zip(self.buf, (obs, act, logp, val, rew, float(done), mask)):
            self.buf[k].append(v)

    def ready(self) -> bool:
        return len(self.buf["obs"]) >= self.rollout_steps

    def update(self, last_value: float) -> Dict[str, float]:
        b = self.buf
        n = len(b["obs"])
        rew, val, done = np.array(b["rew"]), np.array(b["val"] + [last_value]), np.array(b["done"])
        adv = np.zeros(n)
        gae = 0.0
        for t in reversed(range(n)):
            nonterminal = 1.0 - done[t]
            delta = rew[t] + self.gamma * val[t + 1] * nonterminal - val[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            adv[t] = gae
        ret = adv + val[:-1]
        obs, act = to_tensor(np.stack(b["obs"])), to_tensor(b["act"], torch.long)
        old_logp, mask = to_tensor(b["logp"]), to_tensor(np.stack(b["mask"]), torch.bool)
        adv_t, ret_t = to_tensor(adv), to_tensor(ret)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        for _ in range(self.epochs):
            perm = self._rng.permutation(n)
            for i in range(0, n, self.minibatch):
                idx = torch.as_tensor(perm[i:i + self.minibatch])
                logits, value = self.net(obs[idx])
                dist = torch.distributions.Categorical(logits=mask_logits(logits, mask[idx]))
                ratio = torch.exp(dist.log_prob(act[idx]) - old_logp[idx])
                s1 = ratio * adv_t[idx]
                s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_t[idx]
                p_loss = -torch.min(s1, s2).mean()
                v_loss = (value - ret_t[idx]).pow(2).mean()
                ent = dist.entropy().mean()
                loss = p_loss + self.vf * v_loss - self.ent * ent
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
                stats = {"policy_loss": p_loss.item(), "value_loss": v_loss.item(), "entropy": ent.item()}
        self.reset_rollout()
        return stats

    def value_of(self, obs) -> float:
        with torch.no_grad():
            return float(self.net(to_tensor(obs).unsqueeze(0))[1].item())

    def save(self, path: str) -> None:
        torch.save({"state": self.net.state_dict(), "hp": self.hp, "obs_dim": OBS_DIM, "n_actions": NUM_ACTIONS}, path)

    def load(self, path: str) -> None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        assert ck["obs_dim"] == OBS_DIM and ck["n_actions"] == NUM_ACTIONS, "checkpoint was trained on a different state layout"
        self.net.load_state_dict(ck["state"])
        self.training = False
