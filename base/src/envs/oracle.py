"""OracleSolver: computes the expected-optimal tool sequence per task, using
the environment's *known* simulation parameters (we control the simulation,
so this is legitimate).

Assumption / simplification (documented so it can be cited/critiqued in the
write-up): because each tool call succeeds only with some probability, an
agent following a given success path may need to retry a tool multiple times
before it succeeds. Under a geometric-retry model, the expected number of
attempts to first success with per-attempt success probability p is 1/p, so
the expected cost/latency of "eventually getting this tool to succeed" is
base_cost/p and base_latency/p respectively. The oracle evaluates each of a
task's defined SuccessPath options under this expectation and returns the
path maximizing expected reward. It ignores per-episode noise (it reasons
about the *base* tool parameters), which is appropriate since the oracle is
meant as a stable reference point for regret, not a per-episode optimal
solver.
"""

from dataclasses import dataclass
from typing import Dict, Tuple

from src.envs.tools import TOOL_REGISTRY, TOOL_INDEX
from src.envs.tasks import TaskTemplate, TASK_REGISTRY, SuccessPath


@dataclass(frozen=True)
class OraclePlan:
    task_id: int
    task_name: str
    best_path: SuccessPath
    expected_cost: float
    expected_latency: float
    expected_value: float  # expected reward under the reward formula


class OracleSolver:
    """Computes, for each task template, the success path with the highest
    expected reward under the environment's reward formula, using expected
    retry-cost accounting. Used as the reference point for the
    "regret vs. oracle" evaluation metric.
    """

    def __init__(self, success_reward: float, lam: float, mu: float):
        self.success_reward = success_reward
        self.lam = lam
        self.mu = mu

    def _expected_path_cost_latency(self, sequence: Tuple[str, ...]) -> Tuple[float, float]:
        expected_cost = 0.0
        expected_latency = 0.0
        for tool_name in sequence:
            tool = TOOL_REGISTRY[TOOL_INDEX[tool_name]]
            expected_cost += tool.base_cost / tool.base_success_prob
            expected_latency += tool.base_latency / tool.base_success_prob
        return expected_cost, expected_latency

    def solve(self, task: TaskTemplate) -> OraclePlan:
        best_plan = None
        for path in task.success_paths:
            exp_cost, exp_latency = self._expected_path_cost_latency(path.sequence)
            exp_value = self.success_reward * path.quality - self.lam * exp_cost - self.mu * exp_latency
            if best_plan is None or exp_value > best_plan.expected_value:
                best_plan = OraclePlan(
                    task_id=task.task_id,
                    task_name=task.name,
                    best_path=path,
                    expected_cost=exp_cost,
                    expected_latency=exp_latency,
                    expected_value=exp_value,
                )
        return best_plan

    def solve_all(self) -> Dict[int, OraclePlan]:
        return {task.task_id: self.solve(task) for task in TASK_REGISTRY}

    def _expected_path_cost_latency_multi_agent(self, sequence: Tuple[str, ...]) -> Tuple[float, float]:
        """Same expected-retry accounting as `_expected_path_cost_latency`,
        but adds MultiAgentToolOrchestrationEnv's fixed delegation overhead
        to every tool in the sequence classified as a HEAVY_TOOL (every
        retry of a heavy tool re-incurs the delegation cost, since each
        attempt is its own delegate-then-call turn)."""
        from src.envs.multi_agent_env import MultiAgentToolOrchestrationEnv as MAEnv

        expected_cost = 0.0
        expected_latency = 0.0
        for tool_name in sequence:
            tool = TOOL_REGISTRY[TOOL_INDEX[tool_name]]
            extra_cost = MAEnv.DELEGATION_COST if tool_name in MAEnv.HEAVY_TOOLS else 0.0
            extra_latency = MAEnv.DELEGATION_LATENCY if tool_name in MAEnv.HEAVY_TOOLS else 0.0
            expected_cost += (tool.base_cost + extra_cost) / tool.base_success_prob
            expected_latency += (tool.base_latency + extra_latency) / tool.base_success_prob
        return expected_cost, expected_latency

    def solve_multi_agent(self, task: TaskTemplate) -> OraclePlan:
        best_plan = None
        for path in task.success_paths:
            exp_cost, exp_latency = self._expected_path_cost_latency_multi_agent(path.sequence)
            exp_value = self.success_reward * path.quality - self.lam * exp_cost - self.mu * exp_latency
            if best_plan is None or exp_value > best_plan.expected_value:
                best_plan = OraclePlan(
                    task_id=task.task_id, task_name=task.name, best_path=path,
                    expected_cost=exp_cost, expected_latency=exp_latency, expected_value=exp_value,
                )
        return best_plan

    def solve_all_multi_agent(self) -> Dict[int, OraclePlan]:
        return {task.task_id: self.solve_multi_agent(task) for task in TASK_REGISTRY}
