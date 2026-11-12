"""RuleBasedAgent: a hand-written if/else heuristic baseline.

The heuristic always attempts the *first-defined* success path for a task
(TASK_REGISTRY order, not the cheapest/highest-quality one -- a static
heuristic wouldn't know that without being told), calling its tools in
order. If a tool call fails, it retries the same tool up to `max_retries`
times before giving up on the whole task (STOP). This rigidity -- no
adaptive replanning, no picking a better path when the "obvious" one keeps
failing -- is intentional: it is the point of contrast against the learned
agents in the comparison plots.
"""

from typing import Dict

from src.envs.tools import TOOL_INDEX, STOP_ACTION
from src.envs.tasks import TASK_REGISTRY
from src.envs.tool_orchestration_env import decode_observation


class RuleBasedAgent:
    name = "rule_based"

    def __init__(self, max_retries: int = 2):
        self.max_retries = max_retries
        self._plan = None
        self._plan_idx = 0
        self._retries_on_current = 0
        self._last_attempted_mask = None

    def reset(self) -> None:
        self._plan = None
        self._plan_idx = 0
        self._retries_on_current = 0
        self._last_attempted_mask = None

    def act(self, obs) -> int:
        decoded = decode_observation(obs)
        task_id = decoded["task_id"]

        if self._plan is None:
            self._plan = TASK_REGISTRY[task_id].success_paths[0].sequence
            self._plan_idx = 0
            self._retries_on_current = 0

        # Did our most recent attempted tool succeed? Advance the plan if so.
        attempted_mask = decoded["attempted_mask"]
        succeeded_mask = decoded["succeeded_mask"]
        if self._plan_idx < len(self._plan):
            current_tool_idx = TOOL_INDEX[self._plan[self._plan_idx]]
            if succeeded_mask[current_tool_idx] > 0:
                self._plan_idx += 1
                self._retries_on_current = 0
            elif self._last_attempted_mask is not None and attempted_mask[current_tool_idx] > self._last_attempted_mask[current_tool_idx]:
                # We just attempted the current tool and it failed.
                self._retries_on_current += 1
                if self._retries_on_current > self.max_retries:
                    self._last_attempted_mask = attempted_mask.copy()
                    return STOP_ACTION
        self._last_attempted_mask = attempted_mask.copy()

        if self._plan_idx >= len(self._plan):
            # Plan already satisfied; env should have terminated, but if not, stop.
            return STOP_ACTION

        return TOOL_INDEX[self._plan[self._plan_idx]]
