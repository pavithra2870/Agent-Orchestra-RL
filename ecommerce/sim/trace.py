"""One record per environment step. The evaluator and the dashboard replay both read this."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class TraceStep:
    step: int
    action: str
    args: Dict[str, Any] = field(default_factory=dict)
    blocked: bool = False          # the guardrail refused the action (it was not executed)
    precond_ok: bool = True        # were the safety preconditions satisfied when it was chosen?
    ok: bool = True                # did the tool succeed?
    error: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    cost: float = 0.0
    latency: float = 0.0
    mutated: bool = False
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
