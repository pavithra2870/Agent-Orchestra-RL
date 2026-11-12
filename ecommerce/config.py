"""Typed configuration loader. YAML values override the defaults in ecommerce/config/ecommerce.yaml."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import BaseModel

ECOM_ROOT = Path(__file__).parent
CONFIG_PATH = ECOM_ROOT / "config" / "ecommerce.yaml"
RESULTS_DIR = ECOM_ROOT / "results"


class WorldCfg(BaseModel):
    n_customers: int = 60
    max_orders_per_customer: int = 3
    high_value_threshold: float = 5000.0
    refund_window_days: int = 7
    delay_refund_days: int = 5
    return_window_days: int = 7             # change-of-mind returns: days after delivery
    damage_report_window_days: int = 2      # damage must be reported within 48 hours of delivery
    instant_refund_cap: float = 500.0       # damage claims up to this order total may be handled without a human
    damage_claims_limit: int = 2            # this many damage claims inside the period sends the next one to a human
    claims_period_days: int = 30


class ToolCfg(BaseModel):
    cost: float
    latency: float
    failure_prob: float


class EnvCfg(BaseModel):
    max_steps: int = 16
    cost_budget: float = 30.0
    max_clarifications: int = 2
    max_consecutive_tool_failures: int = 3


class RewardCfg(BaseModel):
    lam_cost: float = 0.1
    mu_latency: float = 0.05
    goal_handled_bonus: float = 1.0
    success_bonus: float = 5.0
    failure_penalty: float = 2.0
    violation_penalty: float = 5.0
    blocked_action_penalty: float = 0.5
    unneeded_clarification_penalty: float = 0.3


class ScenarioCfg(BaseModel):
    type_probs: Dict[str, float]
    typo_prob: float = 0.15
    failure_multiplier_stress: float = 3.0
    down_tool_prob_stress: float = 0.2


class GuardrailCfg(BaseModel):
    in_scope_threshold: float = 0.38
    margin_threshold: float = 0.04
    low_confidence_threshold: float = 0.6


class LLMCfg(BaseModel):
    ollama_url: str = "http://localhost:11434"
    model: str = "phi4-mini:latest"
    timeout_seconds: float = 120.0
    temperature: float = 0.0


class Settings(BaseModel):
    world: WorldCfg
    tools: Dict[str, ToolCfg]
    env: EnvCfg
    reward: RewardCfg
    scenarios: ScenarioCfg
    guardrails: GuardrailCfg
    llm: LLMCfg


@lru_cache(maxsize=1)
def get_settings(path: Optional[str] = None) -> Settings:
    with open(path or CONFIG_PATH, "r", encoding="utf-8") as f:
        return Settings(**yaml.safe_load(f))
