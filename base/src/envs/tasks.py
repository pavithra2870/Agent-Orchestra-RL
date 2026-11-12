"""Task template definitions for ToolOrchestrationEnv.

A task is "solved" once the successfully-completed tools (in the order they
succeeded) contain one of the task's `SuccessPath.sequence` lists as a
*subsequence* (order matters, extra/interleaved calls are allowed but cost
extra). Several tasks define more than one success path so the same task can
be resolved in multiple ways with different cost/quality tradeoffs -- e.g.
a cheap "informational" resolution vs. a more complete (and more expensive)
one that yields a larger success bonus via `quality`.
"""

from dataclasses import dataclass
from typing import Tuple, Set
import numpy as np


@dataclass(frozen=True)
class SuccessPath:
    sequence: Tuple[str, ...]  # required tool names, in order (subsequence match)
    quality: float  # multiplier on success_reward; 1.0 = fullest/best resolution
    label: str  # human readable description, for logging only


@dataclass(frozen=True)
class TaskTemplate:
    task_id: int
    name: str
    success_paths: Tuple[SuccessPath, ...]
    max_budget: float
    max_steps: int

    def get_required_tools(self, tool_names: Tuple[str, ...]) -> np.ndarray:
        """Return multi-hot encoding of tools that appear in ANY success path.
        
        This gives the agent semantic information about which tools are
        potentially useful for this task, enabling generalization to unseen tasks.
        
        Args:
            tool_names: Tuple of all tool names in order (from TOOL_NAMES)
        
        Returns:
            Binary array of shape (num_tools,) where 1 = tool appears in some success path
        """
        required = set()
        for path in self.success_paths:
            required.update(path.sequence)
        
        tool_to_idx = {name: i for i, name in enumerate(tool_names)}
        encoding = np.zeros(len(tool_names), dtype=np.float32)
        for tool in required:
            if tool in tool_to_idx:
                encoding[tool_to_idx[tool]] = 1.0
        return encoding
    
    def get_min_path_length(self) -> int:
        """Return the length of the shortest success path."""
        return min(len(path.sequence) for path in self.success_paths)
    
    def get_max_path_length(self) -> int:
        """Return the length of the longest success path."""
        return max(len(path.sequence) for path in self.success_paths)
    
    def get_num_success_paths(self) -> int:
        """Return the number of alternative success paths."""
        return len(self.success_paths)
    
    def get_avg_quality(self) -> float:
        """Return the average quality across all success paths."""
        return sum(path.quality for path in self.success_paths) / len(self.success_paths)
    
    def get_max_quality(self) -> float:
        """Return the maximum quality achievable for this task."""
        return max(path.quality for path in self.success_paths)


TASK_REGISTRY: Tuple[TaskTemplate, ...] = (
    TaskTemplate(
        task_id=0,
        name="check_order_status",
        success_paths=(
            SuccessPath(("db_lookup",), 1.0, "checked via internal database"),
            SuccessPath(("search_api",), 0.9, "checked via external search"),
        ),
        max_budget=6.0,
        max_steps=4,
    ),
    TaskTemplate(
        task_id=1,
        name="resolve_billing_issue",
        success_paths=(
            SuccessPath(("verify_identity", "db_lookup"), 0.7, "informational resolution only"),
            SuccessPath(("verify_identity", "db_lookup", "payment_api"), 1.0, "refund issued"),
        ),
        max_budget=15.0,
        max_steps=6,
    ),
    TaskTemplate(
        task_id=2,
        name="process_refund",
        success_paths=(
            SuccessPath(("verify_identity", "payment_api"), 1.0, "refund processed"),
        ),
        max_budget=12.0,
        max_steps=5,
    ),
    TaskTemplate(
        task_id=3,
        name="update_contact_email",
        success_paths=(
            SuccessPath(("verify_identity", "send_email"), 1.0, "email updated and confirmed"),
        ),
        max_budget=8.0,
        max_steps=5,
    ),
    TaskTemplate(
        task_id=4,
        name="escalate_complaint",
        success_paths=(
            SuccessPath(("escalate_human",), 0.7, "escalated directly, unverified"),
            SuccessPath(("verify_identity", "escalate_human"), 1.0, "escalated with verified identity"),
        ),
        max_budget=16.0,
        max_steps=6,
    ),
    TaskTemplate(
        task_id=5,
        name="search_product_info",
        success_paths=(
            SuccessPath(("db_lookup",), 0.9, "found in local catalog"),
            SuccessPath(("search_api",), 1.0, "found via external search"),
        ),
        max_budget=6.0,
        max_steps=4,
    ),
    TaskTemplate(
        task_id=6,
        name="verify_and_notify",
        success_paths=(
            SuccessPath(("verify_identity", "send_email"), 1.0, "verified and notified"),
        ),
        max_budget=8.0,
        max_steps=5,
    ),
    TaskTemplate(
        task_id=7,
        name="fraud_check_and_block",
        success_paths=(
            SuccessPath(("verify_identity", "db_lookup", "escalate_human"), 1.0, "fraud confirmed and blocked"),
        ),
        max_budget=20.0,
        max_steps=7,
    ),
    TaskTemplate(
        task_id=8,
        name="password_reset",
        success_paths=(
            SuccessPath(("verify_identity", "send_email"), 1.0, "reset link sent"),
        ),
        max_budget=8.0,
        max_steps=5,
    ),
    TaskTemplate(
        task_id=9,
        name="full_account_review",
        success_paths=(
            SuccessPath(
                ("verify_identity", "db_lookup", "search_api", "escalate_human"),
                1.0,
                "complete account review",
            ),
        ),
        max_budget=25.0,
        max_steps=8,
    ),
)

TASK_NAMES: Tuple[str, ...] = tuple(t.name for t in TASK_REGISTRY)
NUM_TASKS = len(TASK_REGISTRY)


def get_task_metadata_stats():
    """Return statistics about task metadata across the registry.
    
    Used for normalization in the environment observation.
    """
    from src.envs.tools import TOOL_NAMES
    
    max_min_path_length = max(t.get_min_path_length() for t in TASK_REGISTRY)
    max_max_path_length = max(t.get_max_path_length() for t in TASK_REGISTRY)
    max_num_paths = max(t.get_num_success_paths() for t in TASK_REGISTRY)
    max_budget = max(t.max_budget for t in TASK_REGISTRY)
    max_steps = max(t.max_steps for t in TASK_REGISTRY)
    
    return {
        'max_min_path_length': max_min_path_length,
        'max_max_path_length': max_max_path_length,
        'max_num_paths': max_num_paths,
        'max_budget_global': max_budget,
        'max_steps_global': max_steps,
    }
