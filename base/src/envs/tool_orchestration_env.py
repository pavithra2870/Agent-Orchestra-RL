"""ToolOrchestrationEnv: a gymnasium environment for agentic tool-use orchestration.

=== MDP formulation ===

State s_t:
    A flat vector concatenating:
      - one-hot encoding of the current task template (which instruction/context
        the agent is trying to resolve),
      - a multi-hot "attempted" mask over tools (has this tool been called at
        least once so far, regardless of outcome),
      - a multi-hot "succeeded" mask over tools (has this tool call *succeeded*
        at least once so far),
      - remaining budget, normalized by the task's max_budget,
      - remaining steps, normalized by the task's max_steps.
    This matches the spec: "task embedding/one-hot + history of tools called
    so far + remaining budget".

Action a_t:
    Discrete(NUM_TOOLS + 1): call one of the mock tools, or STOP (commit to an
    answer / give up on the task right now).

Transition:
    Calling a tool costs a (per-episode-noised) cost and latency, and succeeds
    with a (per-episode-noised) probability. Cost/latency are paid whether or
    not the call succeeds -- this mirrors real tool use, where you pay for
    the API call even if it errors or returns nothing useful. A task is
    solved the instant the sequence of *successful* tool calls contains one
    of the task's defined SuccessPath sequences as a subsequence; the episode
    then terminates immediately (success does not require an explicit STOP).

Reward r_t (the project reward formula):
    r_t = -lambda * cost_t - mu * latency_t                         (every tool-call step)
        + success_reward * quality                                  (only on the step that
                                                                       satisfies a success path)
        - failure_penalty                                            (only on a terminal step
                                                                       that is NOT a success:
                                                                       STOP, budget exhausted,
                                                                       or max_steps reached)
    lambda, mu, success_reward and failure_penalty are all configurable.

Episode termination:
    - terminated=True, success  -> a success path's sequence is satisfied.
    - terminated=True, failure  -> the agent chose STOP, or remaining budget <= 0.
    - truncated=True, failure   -> max_steps for the task template is reached.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.envs.tools import TOOL_REGISTRY, TOOL_NAMES, NUM_TOOLS, STOP_ACTION
from src.envs.tasks import TASK_REGISTRY, NUM_TASKS, get_task_metadata_stats


class ToolOrchestrationEnv(gym.Env):
    """Simulated agentic tool-use orchestration environment."""

    metadata = {"render_modes": ["human"], "render_fps": 1}

    # --- reward hyperparameters (all overridable via constructor) ---
    DEFAULT_SUCCESS_REWARD = 10.0
    DEFAULT_LAMBDA = 0.1     # cost weight
    DEFAULT_MU = 0.1         # latency weight
    DEFAULT_FAILURE_PENALTY = 2.0

    # --- per-episode noise magnitudes ---
    COST_NOISE_LOW, COST_NOISE_HIGH = 0.85, 1.15
    LATENCY_NOISE_LOW, LATENCY_NOISE_HIGH = 0.85, 1.15
    SUCCESS_PROB_NOISE_STD = 0.03
    SUCCESS_PROB_MIN, SUCCESS_PROB_MAX = 0.05, 0.99

    def __init__(
        self,
        success_reward: float = DEFAULT_SUCCESS_REWARD,
        lam: float = DEFAULT_LAMBDA,
        mu: float = DEFAULT_MU,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        seed: Optional[int] = None,
        force_deterministic_success: bool = False,
        render_mode: Optional[str] = None,
        include_task_metadata: bool = True,  # NEW: flag to enable rich task features
    ):
        super().__init__()
        self.success_reward = success_reward
        self.lam = lam
        self.mu = mu
        self.failure_penalty = failure_penalty
        self.force_deterministic_success = force_deterministic_success
        self.render_mode = render_mode
        self.include_task_metadata = include_task_metadata

        self.num_tools = NUM_TOOLS
        self.num_tasks = NUM_TASKS

        self.action_space = spaces.Discrete(self.num_tools + 1)  # +1 for STOP
        
        # OLD observation: task_onehot(10) + attempted(6) + succeeded(6) + budget_norm(1) + steps_norm(1) = 24
        # NEW observation with metadata: + required_tools(6) + min_path_len(1) + num_paths(1) + task_max_budget(1) + task_max_steps(1) = 34
        if self.include_task_metadata:
            obs_dim = self.num_tasks + self.num_tools + self.num_tools + 2 + self.num_tools + 4
        else:
            obs_dim = self.num_tasks + self.num_tools + self.num_tools + 2
        
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
        
        # Cache task metadata statistics for normalization
        self._metadata_stats = get_task_metadata_stats()

        # Episode state, populated in reset().
        self.task = None
        self.attempted_mask = np.zeros(self.num_tools, dtype=np.float32)
        self.succeeded_mask = np.zeros(self.num_tools, dtype=np.float32)
        self.successful_sequence: List[str] = []
        self.call_history: List[Tuple[str, bool, float, float]] = []  # (tool, success, cost, latency)
        self.remaining_budget = 0.0
        self.steps_taken = 0
        self.cumulative_cost = 0.0
        self.cumulative_latency = 0.0
        self._cost_factor = np.ones(self.num_tools)
        self._latency_factor = np.ones(self.num_tools)
        self._success_delta = np.zeros(self.num_tools)

        # Seed via the gymnasium-standard mechanism so self.np_random is set.
        self.reset(seed=seed)

    # ------------------------------------------------------------------ #
    # Core gym API
    # ------------------------------------------------------------------ #
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)  # sets self.np_random deterministically from `seed`

        options = options or {}
        task_id = options.get("task_id")
        if task_id is None:
            task_id = int(self.np_random.integers(0, self.num_tasks))
        self.task = TASK_REGISTRY[task_id]

        self.attempted_mask = np.zeros(self.num_tools, dtype=np.float32)
        self.succeeded_mask = np.zeros(self.num_tools, dtype=np.float32)
        self.successful_sequence = []
        self.call_history = []
        self.remaining_budget = float(self.task.max_budget)
        self.steps_taken = 0
        self.cumulative_cost = 0.0
        self.cumulative_latency = 0.0

        self._sample_episode_noise()

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"invalid action {action}"
        assert self.task is not None, "call reset() before step()"

        terminated = False
        truncated = False
        reward = 0.0
        info: Dict[str, Any] = {"success": False, "reason": None, "tool": None}

        if action == STOP_ACTION:
            terminated = True
            reward -= self.failure_penalty
            info["reason"] = "stop"
        else:
            tool = TOOL_REGISTRY[action]
            cost = float(tool.base_cost * self._cost_factor[action])
            latency = float(tool.base_latency * self._latency_factor[action])
            eff_p = float(
                np.clip(
                    tool.base_success_prob + self._success_delta[action],
                    self.SUCCESS_PROB_MIN,
                    self.SUCCESS_PROB_MAX,
                )
            )
            success = True if self.force_deterministic_success else bool(self.np_random.random() < eff_p)

            self.attempted_mask[action] = 1.0
            self.cumulative_cost += cost
            self.cumulative_latency += latency
            self.remaining_budget -= cost
            self.steps_taken += 1
            self.call_history.append((tool.name, success, cost, latency))
            info["tool"] = tool.name

            reward -= self.lam * cost + self.mu * latency

            if success:
                self.succeeded_mask[action] = 1.0
                self.successful_sequence.append(tool.name)

            best_quality = self._check_success_paths()
            if best_quality is not None:
                terminated = True
                reward += self.success_reward * best_quality
                info["success"] = True
                info["quality"] = best_quality
                info["reason"] = "task_success"
            elif self.remaining_budget <= 0:
                terminated = True
                reward -= self.failure_penalty
                info["reason"] = "budget_exhausted"
            elif self.steps_taken >= self.task.max_steps:
                truncated = True
                reward -= self.failure_penalty
                info["reason"] = "max_steps"

        info.update(
            cost=self.cumulative_cost,
            latency=self.cumulative_latency,
            steps=self.steps_taken,
            task_id=self.task.task_id,
            task_name=self.task.name,
        )

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "human":
            return
        print(
            f"[{self.task.name}] step={self.steps_taken} "
            f"budget_left={self.remaining_budget:.2f} "
            f"succeeded={[TOOL_NAMES[i] for i in range(self.num_tools) if self.succeeded_mask[i]]} "
            f"history={[c[0] for c in self.call_history]}"
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _sample_episode_noise(self) -> None:
        """Draw per-episode multiplicative cost/latency noise and an additive
        success-probability perturbation for each tool, using the env's seeded
        RNG so episodes are reproducible under a fixed seed while still
        varying from episode to episode (per spec: "with some randomness per
        episode")."""
        self._cost_factor = self.np_random.uniform(
            self.COST_NOISE_LOW, self.COST_NOISE_HIGH, size=self.num_tools
        )
        self._latency_factor = self.np_random.uniform(
            self.LATENCY_NOISE_LOW, self.LATENCY_NOISE_HIGH, size=self.num_tools
        )
        self._success_delta = self.np_random.normal(
            0.0, self.SUCCESS_PROB_NOISE_STD, size=self.num_tools
        )

    def _check_success_paths(self) -> Optional[float]:
        """Return the highest quality among success paths satisfied by the
        successful-call history so far (as an ordered subsequence), or None
        if no path is satisfied yet."""
        best = None
        for path in self.task.success_paths:
            if self._is_subsequence(path.sequence, self.successful_sequence):
                if best is None or path.quality > best:
                    best = path.quality
        return best

    @staticmethod
    def _is_subsequence(required: Tuple[str, ...], actual: List[str]) -> bool:
        idx = 0
        for tool_name in actual:
            if idx < len(required) and tool_name == required[idx]:
                idx += 1
        return idx == len(required)

    def _get_obs(self) -> np.ndarray:
        task_onehot = np.zeros(self.num_tasks, dtype=np.float32)
        task_onehot[self.task.task_id] = 1.0
        budget_norm = np.clip(self.remaining_budget / self.task.max_budget, 0.0, 1.0)
        steps_norm = np.clip(
            (self.task.max_steps - self.steps_taken) / self.task.max_steps, 0.0, 1.0
        )
        
        # Base observation (backward compatible)
        obs_components = [
            task_onehot,
            self.attempted_mask,
            self.succeeded_mask,
            np.array([budget_norm, steps_norm], dtype=np.float32),
        ]
        
        # Add task metadata if enabled (NEW for generalization to unseen tasks)
        if self.include_task_metadata:
            required_tools = self.task.get_required_tools(TOOL_NAMES)
            min_path_len_norm = self.task.get_min_path_length() / self._metadata_stats['max_max_path_length']
            num_paths_norm = self.task.get_num_success_paths() / self._metadata_stats['max_num_paths']
            task_max_budget_norm = self.task.max_budget / self._metadata_stats['max_budget_global']
            task_max_steps_norm = self.task.max_steps / self._metadata_stats['max_steps_global']
            
            obs_components.extend([
                required_tools,
                np.array([min_path_len_norm, num_paths_norm, task_max_budget_norm, task_max_steps_norm], dtype=np.float32),
            ])
        
        obs = np.concatenate(obs_components).astype(np.float32)
        return obs

    def _get_info(self) -> Dict[str, Any]:
        return {
            "task_id": self.task.task_id,
            "task_name": self.task.name,
            "cost": self.cumulative_cost,
            "latency": self.cumulative_latency,
            "steps": self.steps_taken,
        }


def decode_observation(obs: np.ndarray, num_tasks: int = NUM_TASKS, num_tools: int = NUM_TOOLS, include_metadata: bool = True) -> Dict[str, Any]:
    """Inverse of ToolOrchestrationEnv._get_obs, used by baseline agents and
    the tabular-state discretizer so they don't need to duplicate the layout.
    
    Args:
        obs: Observation vector from environment
        num_tasks: Number of tasks in the environment
        num_tools: Number of tools in the environment
        include_metadata: Whether the observation includes task metadata features
    
    Returns:
        Dictionary with decoded observation components
    """
    task_onehot = obs[:num_tasks]
    attempted_mask = obs[num_tasks : num_tasks + num_tools]
    succeeded_mask = obs[num_tasks + num_tools : num_tasks + 2 * num_tools]
    budget_norm, steps_norm = obs[num_tasks + 2 * num_tools : num_tasks + 2 * num_tools + 2]
    
    result = {
        "task_id": int(np.argmax(task_onehot)),
        "attempted_mask": attempted_mask,
        "succeeded_mask": succeeded_mask,
        "budget_norm": float(budget_norm),
        "steps_norm": float(steps_norm),
    }
    
    if include_metadata:
        offset = num_tasks + 2 * num_tools + 2
        required_tools = obs[offset : offset + num_tools]
        min_path_len_norm, num_paths_norm, task_max_budget_norm, task_max_steps_norm = obs[offset + num_tools : offset + num_tools + 4]
        result.update({
            "required_tools": required_tools,
            "min_path_len_norm": float(min_path_len_norm),
            "num_paths_norm": float(num_paths_norm),
            "task_max_budget_norm": float(task_max_budget_norm),
            "task_max_steps_norm": float(task_max_steps_norm),
        })
    
    return result
