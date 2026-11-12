"""Mock tool definitions for ToolOrchestrationEnv.

Each Tool has a *base* cost, latency and success probability. At the start of
every episode the environment perturbs these base values with small
per-episode noise (see ToolOrchestrationEnv._sample_episode_noise) so that
repeated episodes are not identical, while still being reproducible under a
fixed seed.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Tool:
    name: str
    base_cost: float
    base_latency: float
    base_success_prob: float


# Order here fixes the action index of every tool: action `i` calls
# TOOL_REGISTRY[i]. The STOP action is always the last action, i.e.
# index == len(TOOL_REGISTRY).
TOOL_REGISTRY: Tuple[Tool, ...] = (
    Tool(name="db_lookup", base_cost=1.0, base_latency=1.0, base_success_prob=0.95),
    Tool(name="search_api", base_cost=2.0, base_latency=2.0, base_success_prob=0.85),
    Tool(name="payment_api", base_cost=5.0, base_latency=3.0, base_success_prob=0.90),
    Tool(name="escalate_human", base_cost=10.0, base_latency=8.0, base_success_prob=0.99),
    Tool(name="send_email", base_cost=0.5, base_latency=1.0, base_success_prob=0.98),
    Tool(name="verify_identity", base_cost=1.5, base_latency=2.0, base_success_prob=0.90),
)

TOOL_NAMES: Tuple[str, ...] = tuple(t.name for t in TOOL_REGISTRY)
TOOL_INDEX = {name: i for i, name in enumerate(TOOL_NAMES)}
NUM_TOOLS = len(TOOL_REGISTRY)
STOP_ACTION = NUM_TOOLS  # discrete action id reserved for "stop and answer"
