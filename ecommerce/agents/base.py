"""Common agent interface for every method (random, rule-based, tabular, DQN, PPO, CQL, LLM, oracle)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class Agent:
    name = "agent"
    needs_scenario = False      # only the oracle needs the hidden ground truth
    uses_mask = True            # False for the "no guardrail" ablation

    def reset(self, scenario: Any = None) -> None:
        pass

    def act(self, obs: np.ndarray, mask: np.ndarray, info: Optional[Dict[str, Any]] = None) -> int:
        raise NotImplementedError

    def action_scores(self, obs: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
        """Optional per-action scores (Q-values or probabilities) shown on the dashboard."""
        return None


def masked_argmax(values: np.ndarray, mask: np.ndarray) -> int:
    v = np.where(mask, values, -np.inf)
    return int(np.argmax(v))
