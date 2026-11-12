"""Discretizes ToolOrchestrationEnv's continuous observation into a small,
hashable state for tabular Q-learning / SARSA.

Design note: the full observation includes both an "attempted" mask and a
"succeeded" mask per tool (12 bits total), plus task id, budget and steps.
Keeping both masks blows up the tabular state space to ~10 * 2^12 * bins^2,
which is too sparse to learn from in a modest number of episodes. Since only
*successful* calls matter for task completion, the tabular state drops the
attempted mask and keeps only the succeeded mask -- failed attempts still
show up indirectly via consumed budget/steps. This is a deliberate baseline
simplification; DQN/PPO in later phases consume the full observation.

NEW (metadata update): With task metadata features enabled, tabular methods
now include the required_tools mask as part of the discretized state. This
provides semantic information about which tools are needed for the task,
enabling better generalization to unseen tasks. The required_tools mask is
deterministic per task, so it doesn't increase state space exponentially.
"""

from typing import Tuple

import numpy as np

from src.envs.tool_orchestration_env import decode_observation


def discretize_state(
    obs: np.ndarray,
    budget_bins: int = 5,
    steps_bins: int = 5,
    include_metadata: bool = True,
) -> Tuple:
    """Discretize observation into a hashable state for tabular methods.
    
    Args:
        obs: Raw observation from environment
        budget_bins: Number of bins for budget discretization
        steps_bins: Number of bins for steps discretization
        include_metadata: Whether observation includes task metadata features
    
    Returns:
        Tuple representing discretized state:
        - If include_metadata=True: (task_id, succeeded_mask, budget_bin, steps_bin, required_tools_mask)
        - If include_metadata=False: (task_id, succeeded_mask, budget_bin, steps_bin)
    """
    decoded = decode_observation(obs, include_metadata=include_metadata)
    succeeded_mask = tuple(int(x) for x in decoded["succeeded_mask"])
    budget_bin = min(budget_bins - 1, int(decoded["budget_norm"] * budget_bins))
    steps_bin = min(steps_bins - 1, int(decoded["steps_norm"] * steps_bins))
    
    # Base state components
    state_components = [decoded["task_id"], succeeded_mask, budget_bin, steps_bin]
    
    # Add metadata features if available
    if include_metadata and "required_tools" in decoded:
        required_tools_mask = tuple(int(x) for x in decoded["required_tools"])
        state_components.append(required_tools_mask)
    
    return tuple(state_components)
