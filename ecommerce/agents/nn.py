"""Small shared neural-net helpers for DQN / PPO / CQL."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

NEG_INF = -1e9


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


def to_tensor(x, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def mask_logits(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set illegal actions to a very large negative number so they are never chosen or counted."""
    return torch.where(mask, values, torch.full_like(values, NEG_INF))
