"""LLM tool-calling benchmark (phi4-mini via Ollama), in the style of tau-bench's baseline agents.

It sees the SAME information as the RL agents (a readable version of the state, and the list of
currently legal actions) and picks one action per step. It is a benchmark, not part of the system:
the product's decisions are made by the RL policy.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import numpy as np

from ecommerce.agents.base import Agent
from ecommerce.env.state import ACTIONS, decode
from ecommerce.language.llm import OllamaClient
from ecommerce.sim.world import GOALS

TOOL_HELP = {
    "lookup_customer": "verify the customer record (always first)",
    "lookup_order": "verify the order exists and belongs to the customer",
    "check_shipment": "get shipment status / expected delivery date",
    "get_payment_details": "get the order's payment method and state",
    "check_payment_change_eligibility": "check whether the payment method may be changed",
    "update_payment_method": "change the payment method (only after the check says eligible)",
    "check_refund_eligibility": "check whether a refund is allowed",
    "initiate_refund": "issue the refund (only after the check says eligible)",
    "check_cancel_eligibility": "check whether the order may be cancelled",
    "cancel_order": "cancel the order (only after the check says eligible)",
    "check_return_eligibility": "check whether a return, exchange or damage claim is allowed (needs the customer's answers first)",
    "process_return": "open the return or exchange, or refund a small damaged order (only after the check says eligible)",
    "lookup_policy": "look up the shop's written policy when the customer asks a policy question (no order needed)",
    "request_clarification": "ask the customer a question when the order or request is unclear (or a return needs their answers)",
    "escalate_to_human": "hand over to a human (out of scope, needs approval, repeated tool failure)",
    "finish_task": "end the conversation when every requested goal is handled",
}
SYSTEM = ("You are the decision module of an e-commerce support agent. Each turn you pick ONE next action.\n"
          "Policy: verify the customer, then the order; for each requested goal check eligibility first, then act only if eligible; "
          "if a check says ineligible just explain (do not act); if something is unclear ask for clarification; "
          "if the request is out of scope, a refund needs human approval, or a tool keeps failing, escalate; "
          "finish only when all requested goals are handled.\n"
          "Actions:\n" + "\n".join(f"- {k}: {v}" for k, v in TOOL_HELP.items()))


def describe_state(obs: np.ndarray, mask: np.ndarray) -> str:
    f = decode(obs)
    req = [g for g in GOALS if f[f"requested:{g}"] > 0.5]
    handled = [g for g in req if f[f"handled:{g}"] > 0.5]
    need_h = [g for g in req if f[f"needs_escalation:{g}"] > 0.5]
    def elig(prefix: str) -> str:
        if f[f"{prefix}_elig_known"] < 0.5:
            return "unknown"
        if prefix == "refund" and f["refund_elig_human"] > 0.5:
            return "needs human approval"
        return "eligible" if f[f"{prefix}_elig_yes"] > 0.5 else "ineligible"
    status = next((k.split(":")[1] for k in f if k.startswith("shipment_status:") and f[k] > 0.5), "unknown")
    last = next((k.split(":")[1] for k in f if k.startswith("last_action:") and f[k] > 0.5), "none")
    lines = [
        f"Requested goals: {req or 'none found'}",
        f"Parser confidence {f['parse_conf']:.2f} (low={bool(f['low_confidence'])}); scope score {f['scope_score']:.2f}",
        f"Customer verified: {bool(f['customer_verified'])}; order verified: {bool(f['order_verified'])}; "
        f"order unclear: {bool(f['order_ambiguous'] or f['order_rejected'])}",
        f"Shipment status: {status}",
        f"Payment-change check: {elig('pay')}; refund check: {elig('refund')}; cancel check: {elig('cancel')}; return check: {elig('return')}",
        f"Customer still has to say which payment method: {bool(f['payment_method_missing'])}; customer still has to answer the return questions: {bool(f['return_info_missing'])}",
        f"Handled: {handled}; need a human: {need_h}",
        f"Last action: {last} (failed={bool(f['last_failed'])}); steps used {f['steps_used'] * 16:.0f}/16",
        f"LEGAL ACTIONS NOW: {[ACTIONS[i] for i in np.flatnonzero(mask)]}",
    ]
    return "\n".join(lines)


class LLMAgent(Agent):
    name = "llm_phi4_mini"

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or OllamaClient()
        self.calls = self.invalid = 0
        self.seconds = 0.0

    def act(self, obs, mask, info=None) -> int:
        legal = [ACTIONS[i] for i in np.flatnonzero(mask)]
        schema = {"type": "object", "properties": {"action": {"type": "string", "enum": legal}}, "required": ["action"]}
        data, secs = self.client.chat_json(SYSTEM, describe_state(obs, mask) + "\nChoose the next action.", schema, max_tokens=30)
        self.calls += 1
        self.seconds += secs
        name = data.get("action") if isinstance(data, dict) else None
        if name not in legal:
            self.invalid += 1
            return ACTIONS.index("escalate_to_human")           # safe fallback if the model fails
        return ACTIONS.index(name)

    def stats(self) -> Dict[str, Any]:
        return {"calls": self.calls, "invalid_outputs": self.invalid, "avg_seconds_per_call": self.seconds / max(1, self.calls)}
