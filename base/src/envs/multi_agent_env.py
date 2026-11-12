"""MultiAgentToolOrchestrationEnv: a 2-agent extension of ToolOrchestrationEnv
(Phase 4, Module 7 MARL).

=== Design (documented so it can be cited/critiqued in the write-up) ===

Two specialized roles share one task episode:
  - triage:    decides, each turn, whether to call one of the "light" tools
               itself, DELEGATE the turn to the execution agent, or STOP.
  - execution: only acts on a turn triage delegates to it, and only chooses
               *which* "heavy" tool to call.

Tool partition (a deliberate, defensible split -- not arbitrary):
  LIGHT_TOOLS = db_lookup, search_api, send_email   (cheap, low-stakes;
                triage can call these directly)
  HEAVY_TOOLS = payment_api, escalate_human, verify_identity  (higher-stakes
                / specialist actions; only reachable via delegation)

Delegation is not free: choosing DELEGATE adds a fixed DELEGATION_COST /
DELEGATION_LATENCY on top of whatever heavy tool the execution agent picks,
modeling the coordination overhead of handing off work to a specialist
(spec: "delegation incurring its own cost/latency").

Both agents observe the *same* flat observation as the single-agent env
(task one-hot + attempted/succeeded masks over all 6 tools + budget/steps) --
i.e. this is a shared-observation, decentralized-action cooperative MARL
setup, not a partially-observable one. That is a deliberate simplification
(communication/partial-observability protocols are out of scope here); it is
also why a *shared team reward* is used (both agents get the same per-step
reward) rather than a split/negotiated reward -- they are jointly optimizing
one task outcome, so credit assignment beyond "both did well when the task
succeeded cheaply" is not modeled.

This class intentionally does NOT subclass gymnasium.Env: multi-agent step()
takes a dict of per-role actions and returns dicts of per-role
observations/rewards (the same shape PettingZoo's ParallelEnv API uses),
which doesn't fit gymnasium's single-agent Env contract.

Task success detection, reward decomposition (-lambda*cost - mu*latency,
+success_reward*quality, -failure_penalty), episode termination, and
per-episode noise sampling all reuse the exact same logic as
ToolOrchestrationEnv for a fair, apples-to-apples comparison against the
single-agent baselines.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.envs.tools import TOOL_REGISTRY, TOOL_INDEX, NUM_TOOLS, TOOL_NAMES
from src.envs.tasks import TASK_REGISTRY, NUM_TASKS, get_task_metadata_stats
from src.envs.tool_orchestration_env import ToolOrchestrationEnv


class MultiAgentToolOrchestrationEnv:
    LIGHT_TOOLS: Tuple[str, ...] = ("db_lookup", "search_api", "send_email")
    HEAVY_TOOLS: Tuple[str, ...] = ("payment_api", "escalate_human", "verify_identity")

    DELEGATE_ACTION = len(LIGHT_TOOLS)       # triage action id
    TRIAGE_STOP_ACTION = len(LIGHT_TOOLS) + 1  # triage action id
    TRIAGE_NUM_ACTIONS = len(LIGHT_TOOLS) + 2  # light tools + DELEGATE + STOP
    EXECUTION_NUM_ACTIONS = len(HEAVY_TOOLS)

    DELEGATION_COST = 0.5
    DELEGATION_LATENCY = 0.5

    DEFAULT_SUCCESS_REWARD = ToolOrchestrationEnv.DEFAULT_SUCCESS_REWARD
    DEFAULT_LAMBDA = ToolOrchestrationEnv.DEFAULT_LAMBDA
    DEFAULT_MU = ToolOrchestrationEnv.DEFAULT_MU
    DEFAULT_FAILURE_PENALTY = ToolOrchestrationEnv.DEFAULT_FAILURE_PENALTY

    def __init__(
        self,
        success_reward: float = DEFAULT_SUCCESS_REWARD,
        lam: float = DEFAULT_LAMBDA,
        mu: float = DEFAULT_MU,
        failure_penalty: float = DEFAULT_FAILURE_PENALTY,
        seed: Optional[int] = None,
        force_deterministic_success: bool = False,
        include_task_metadata: bool = True,  # NEW: flag to enable rich task features
    ):
        self.success_reward = success_reward
        self.lam = lam
        self.mu = mu
        self.failure_penalty = failure_penalty
        self.force_deterministic_success = force_deterministic_success
        self.include_task_metadata = include_task_metadata

        self.num_tools = NUM_TOOLS
        self.num_tasks = NUM_TASKS
        
        # OLD observation: 24 dimensions
        # NEW observation with metadata: 34 dimensions
        if self.include_task_metadata:
            self.obs_dim = self.num_tasks + 2 * self.num_tools + 2 + self.num_tools + 4
        else:
            self.obs_dim = self.num_tasks + 2 * self.num_tools + 2

        # Cache task metadata statistics for normalization
        self._metadata_stats = get_task_metadata_stats()

        self.task = None
        self.attempted_mask = np.zeros(self.num_tools, dtype=np.float32)
        self.succeeded_mask = np.zeros(self.num_tools, dtype=np.float32)
        self.successful_sequence: List[str] = []
        self.remaining_budget = 0.0
        self.steps_taken = 0
        self.cumulative_cost = 0.0
        self.cumulative_latency = 0.0
        self._cost_factor = np.ones(self.num_tools)
        self._latency_factor = np.ones(self.num_tools)
        self._success_delta = np.zeros(self.num_tools)
        self.np_random = None

        self.reset(seed=seed)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None or self.np_random is None:
            self.np_random = np.random.default_rng(seed)

        options = options or {}
        task_id = options.get("task_id")
        if task_id is None:
            task_id = int(self.np_random.integers(0, self.num_tasks))
        self.task = TASK_REGISTRY[task_id]

        self.attempted_mask = np.zeros(self.num_tools, dtype=np.float32)
        self.succeeded_mask = np.zeros(self.num_tools, dtype=np.float32)
        self.successful_sequence = []
        self.remaining_budget = float(self.task.max_budget)
        self.steps_taken = 0
        self.cumulative_cost = 0.0
        self.cumulative_latency = 0.0

        self._cost_factor = self.np_random.uniform(0.85, 1.15, size=self.num_tools)
        self._latency_factor = self.np_random.uniform(0.85, 1.15, size=self.num_tools)
        self._success_delta = self.np_random.normal(0.0, 0.03, size=self.num_tools)

        obs = self._get_obs()
        info = self._get_info()
        return {"triage": obs, "execution": obs}, info

    def step(self, action_dict: Dict[str, int]):
        triage_action = action_dict["triage"]
        execution_action = action_dict.get("execution")
        assert 0 <= triage_action < self.TRIAGE_NUM_ACTIONS, f"invalid triage action {triage_action}"
        assert self.task is not None, "call reset() before step()"

        terminated = False
        truncated = False
        reward = 0.0
        info: Dict[str, Any] = {"success": False, "reason": None, "tool": None, "delegated": False}

        if triage_action == self.TRIAGE_STOP_ACTION:
            terminated = True
            reward -= self.failure_penalty
            info["reason"] = "stop"
        elif triage_action == self.DELEGATE_ACTION:
            assert execution_action is not None and 0 <= execution_action < self.EXECUTION_NUM_ACTIONS, \
                f"invalid execution action {execution_action}"
            info["delegated"] = True
            tool_name = self.HEAVY_TOOLS[execution_action]
            reward += self._apply_tool_call(
                tool_name, extra_cost=self.DELEGATION_COST, extra_latency=self.DELEGATION_LATENCY
            )
            info["tool"] = tool_name
        else:
            tool_name = self.LIGHT_TOOLS[triage_action]
            reward += self._apply_tool_call(tool_name)
            info["tool"] = tool_name

        if triage_action != self.TRIAGE_STOP_ACTION:
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
            cost=self.cumulative_cost, latency=self.cumulative_latency, steps=self.steps_taken,
            task_id=self.task.task_id, task_name=self.task.name,
        )

        obs = self._get_obs()
        obs_dict = {"triage": obs, "execution": obs}
        reward_dict = {"triage": reward, "execution": reward}  # shared team reward
        return obs_dict, reward_dict, terminated, truncated, info

    # ------------------------------------------------------------------ #
    def _apply_tool_call(self, tool_name: str, extra_cost: float = 0.0, extra_latency: float = 0.0) -> float:
        tool_idx = TOOL_INDEX[tool_name]
        tool = TOOL_REGISTRY[tool_idx]
        cost = float(tool.base_cost * self._cost_factor[tool_idx]) + extra_cost
        latency = float(tool.base_latency * self._latency_factor[tool_idx]) + extra_latency
        eff_p = float(
            np.clip(
                tool.base_success_prob + self._success_delta[tool_idx],
                ToolOrchestrationEnv.SUCCESS_PROB_MIN,
                ToolOrchestrationEnv.SUCCESS_PROB_MAX,
            )
        )
        success = True if self.force_deterministic_success else bool(self.np_random.random() < eff_p)

        self.attempted_mask[tool_idx] = 1.0
        self.cumulative_cost += cost
        self.cumulative_latency += latency
        self.remaining_budget -= cost
        self.steps_taken += 1
        if success:
            self.succeeded_mask[tool_idx] = 1.0
            self.successful_sequence.append(tool_name)

        return -self.lam * cost - self.mu * latency

    def _check_success_paths(self) -> Optional[float]:
        best = None
        for path in self.task.success_paths:
            if ToolOrchestrationEnv._is_subsequence(path.sequence, self.successful_sequence):
                if best is None or path.quality > best:
                    best = path.quality
        return best

    def _get_obs(self) -> np.ndarray:
        task_onehot = np.zeros(self.num_tasks, dtype=np.float32)
        task_onehot[self.task.task_id] = 1.0
        budget_norm = np.clip(self.remaining_budget / self.task.max_budget, 0.0, 1.0)
        steps_norm = np.clip((self.task.max_steps - self.steps_taken) / self.task.max_steps, 0.0, 1.0)
        
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
        
        return np.concatenate(obs_components).astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "task_id": self.task.task_id, "task_name": self.task.name,
            "cost": self.cumulative_cost, "latency": self.cumulative_latency, "steps": self.steps_taken,
        }
