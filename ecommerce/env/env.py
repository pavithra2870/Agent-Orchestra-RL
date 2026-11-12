"""EcomSupportEnv: multi-goal customer-support environment (gymnasium API).

=== MDP formulation ===
State s_t     : see ecommerce/env/state.py (goals requested / handled / need a human, what the tools
                have revealed, failures per tool, steps and cost used, parser confidence, last action).
                No hidden ground truth is ever put in the state.
Action a_t    : one of 16 discrete actions: 13 tools, request_clarification, escalate_to_human, finish_task.
                The policy picks the tool; the environment fills in the arguments (customer id, order id,
                target payment method, the customer's answers about a return, the policy topic) from verified context.
Legal actions : given by ecommerce/guardrails/action_mask.py. With enforce_mask=True an illegal action is
                refused (not executed) and costs a small penalty; with enforce_mask=False it is executed
                (used only for the "no guardrail" ablation to show what the mask prevents).
Transition    : a tool runs on the simulated database (cost/latency noise and random transient failures are
                seeded). request_clarification asks the (simulated) customer: if something really was
                missing, the answer reveals the true order and goals. escalate_to_human and finish_task end
                the episode. Reaching max_steps truncates it.

=== Reward (every term is configurable in config/ecommerce.yaml) ===
  r_t = - lam_cost * cost_t - mu_latency * latency_t           every action (efficiency)
        + goal_handled_bonus * (#goals newly handled)          the moment a requested goal is handled
        - blocked_action_penalty                               if the guardrail refused the action
        - unneeded_clarification_penalty                       asking when nothing was missing
  at the end of the episode, from the INDEPENDENT evaluator (database state + trace, not the agent's claims):
        + success_bonus                                        all goals correct, right ending, no violations
        - failure_penalty                                      otherwise
        - violation_penalty * (#safety violations)             executed rule breaks (0 when masking is on)

Why mostly outcome-based: a 2026 study of customer-service tool agents found that naive dense per-turn
rewards can hurt performance, so the main signal is whether the final database state is right.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ecommerce.config import Settings, get_settings
from ecommerce.env.state import ACTION_INDEX, ACTIONS, NUM_ACTIONS, OBS_DIM, EpisodeState, encode
from ecommerce.evaluation.evaluator import EvalResult, evaluate
from ecommerce.guardrails.action_mask import explain_blocked, legal_mask, safety_ok
from ecommerce.sim.scenarios import Scenario
from ecommerce.sim.tools import STATE_CHANGING_TOOLS, ToolBox
from ecommerce.sim.trace import TraceStep
from ecommerce.sim.world import World

TRANSIENT_ERRORS = ("timeout", "service_unavailable")


class EcomSupportEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        settings: Optional[Settings] = None,
        enforce_mask: bool = True,
        scenario_sampler: Optional[Callable[[np.random.Generator], Scenario]] = None,
    ):
        super().__init__()
        self.settings = settings or get_settings()
        self.enforce_mask = enforce_mask
        self.scenario_sampler = scenario_sampler
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)

        self.scenario: Optional[Scenario] = None
        self.world: Optional[World] = None
        self.toolbox: Optional[ToolBox] = None
        self.state: Optional[EpisodeState] = None
        self.trace: List[TraceStep] = []
        self.live = False
        self.finished = False
        self.last_eval: Optional[EvalResult] = None
        self.target_method: Optional[str] = None

    # ------------------------------------------------------------------ API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)
        options = options or {}
        sc = options.get("scenario")
        if sc is None:
            if self.scenario_sampler is None:
                raise ValueError("pass options={'scenario': ...} or set a scenario_sampler")
            sc = self.scenario_sampler(self.np_random)
        self.scenario = sc
        self.live = bool(options.get("live", False))
        self.world = sc.world.copy()
        run_seed = int(options.get("run_seed", 0))
        tool_rng = np.random.default_rng([sc.seed % (2**31), run_seed])
        self.toolbox = ToolBox(self.world, tool_rng, self.settings,
                               failure_multiplier=float(options.get("failure_multiplier", sc.failure_multiplier)),
                               down_tools=sc.down_tools)
        e = self.settings.env
        self.state = EpisodeState(
            parsed_goals=set(sc.parsed_goals), parse_conf=sc.parse_conf, scope_score=sc.scope_score,
            low_conf_threshold=self.settings.guardrails.low_confidence_threshold,
            max_steps=e.max_steps, cost_budget=e.cost_budget, max_clarifications=e.max_clarifications,
            max_consecutive_failures=e.max_consecutive_tool_failures,
            order_candidate=sc.order_id_in_text,
        )
        self.trace = []
        self.finished = False
        self.last_eval = None
        self.target_method = sc.parsed_target if sc.use_parsed_target else sc.target_method
        self.state.method_known = not ("payment_change" in sc.parsed_goals and self.target_method is None)
        self.state.return_info_known = not ("return_item" in sc.parsed_goals and not sc.return_info_given)
        self.state.clarifications_used = min(int(sc.prior_clarifications), self.state.max_clarifications)
        return encode(self.state), {"action_mask": legal_mask(self.state), "scenario_id": sc.scenario_id}

    def action_mask(self) -> np.ndarray:
        return legal_mask(self.state)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.state is not None and not self.finished, "call reset() first / episode already finished"
        sc, s, cfg = self.scenario, self.state, self.settings.reward
        a = int(action)
        name = ACTIONS[a]
        legal = bool(legal_mask(s)[a])
        ts = TraceStep(step=len(self.trace) + 1, action=name, precond_ok=safety_ok(s, name))
        before_handled = s.handled_set()
        reward = 0.0
        terminal: Optional[str] = None

        if not legal and self.enforce_mask:
            ts.blocked, ts.ok, ts.error = True, False, "blocked_by_guardrail"
            ts.note = explain_blocked(s, name)
            reward -= cfg.blocked_action_penalty
        else:
            if name == "request_clarification":
                terminal = self._clarify(ts)
                if ts.data.get("needed") is False:
                    reward -= cfg.unneeded_clarification_penalty
            elif name == "escalate_to_human":
                terminal = "escalate"
            elif name == "finish_task":
                terminal = "finish"
            else:
                self._run_tool(name, ts)
            cost, latency = self.toolbox.cost_latency(name)
            if not ts.cost:
                ts.cost, ts.latency = cost, latency
            s.cost += ts.cost
            s.latency += ts.latency
            reward -= cfg.lam_cost * ts.cost + cfg.mu_latency * ts.latency

        s.steps += 1
        s.last_action = a
        self.trace.append(ts)

        newly = s.handled_set() - before_handled
        reward += cfg.goal_handled_bonus * len(newly)

        truncated = False
        if terminal is None and s.steps >= s.max_steps:
            terminal, truncated = "max_steps", True

        info: Dict[str, Any] = {"trace_step": ts.to_dict(), "terminal": terminal, "legal": legal}
        if terminal is not None:
            self.finished = True
            if terminal != "awaiting_customer":
                ev = evaluate(sc, self.world, self.trace, terminal)
                self.last_eval = ev
                reward += cfg.success_bonus if ev.success else -cfg.failure_penalty
                reward -= cfg.violation_penalty * len(ev.violations)
                info.update(success=ev.success, eval=ev.to_dict())
        obs = encode(s)
        info["action_mask"] = legal_mask(s)
        info.update(cost=s.cost, latency=s.latency, steps=s.steps)
        return obs, float(reward), terminal is not None and not truncated, truncated, info

    # -------------------------------------------------------------- internals
    def _clarify(self, ts: TraceStep) -> Optional[str]:
        sc, s = self.scenario, self.state
        needed = self._clarification_needed()
        s.clarifications_used += 1
        if self.live:
            ts.note = "asked the customer a question and is waiting for the reply"
            ts.data = {"needed": needed}
            return "awaiting_customer"
        if needed:
            s.order_candidate = sc.order_id
            s.order_ambiguous = s.order_rejected = False
            s.parsed_goals = set(sc.true_goals)
            self.target_method = sc.target_method
            s.method_known = True
            s.return_info_known = True
            s.parse_conf = 0.95
            s.scope_score = max(s.scope_score, 0.8)
            ts.note = "customer answered: order and goals are now clear"
            ts.data = {"needed": True}
        else:
            ts.note = "nothing was missing, the question was unnecessary"
            ts.data = {"needed": False}
        return None

    def _clarification_needed(self) -> bool:
        """Was something really missing? Judged from the ground truth, not from what the agent has looked up."""
        sc, s = self.scenario, self.state
        owned = set(self.world.customers[sc.customer_id].order_ids)
        needs_order = bool(s.parsed_goals - {"policy_info"})          # a policy question needs no order
        order_unclear = needs_order and ((s.order_candidate is None and len(owned) > 1) or
                                         (s.order_candidate is not None and s.order_candidate not in owned))
        method_missing = "payment_change" in s.parsed_goals and self.target_method is None
        return_missing = "return_item" in s.parsed_goals and not s.return_info_known
        return bool(order_unclear or (needs_order and (s.order_ambiguous or s.order_rejected)) or method_missing
                    or return_missing or s.parsed_goals != set(sc.true_goals))

    def _run_tool(self, name: str, ts: TraceStep) -> None:
        sc, s = self.scenario, self.state
        order_id = s.order_candidate or sc.order_id_in_text or sc.order_id
        args: Dict[str, Any] = {"customer_id": sc.customer_id}
        if name != "lookup_customer":
            args["order_id"] = order_id
        if name == "lookup_policy":
            args = {"topic": sc.policy_topic}                     # chosen by the language layer, not by the agent
        if name in ("check_return_eligibility", "process_return"):
            args.update(reason=sc.return_reason, kind=sc.return_kind, declared=sc.return_declared)
        if name in ("check_payment_change_eligibility", "update_payment_method"):
            # With the mask on, a missing method can never reach this line (see guardrails/action_mask.py);
            # the "UPI" default only exists for the no-guardrail ablation.
            args["target_method"] = self.target_method or "UPI"
        ts.args = args
        res = self.toolbox.call(name, **args)
        ts.ok, ts.error, ts.data, ts.cost, ts.latency, ts.mutated = res.ok, res.error, res.data, res.cost, res.latency, res.mutated

        if not res.ok:
            s.last_failed = True
            if res.error in TRANSIENT_ERRORS:
                s.fail_counts[name] += 1
                s.total_failures += 1
            elif name == "lookup_order" and res.error in ("order_not_found", "permission_denied"):
                s.order_rejected, s.order_candidate = True, None
            return

        s.last_failed = False
        s.fail_counts[name] = 0
        d = res.data
        if name == "lookup_customer":
            s.customer_verified, s.customer_order_ids = True, d["order_ids"]
            if s.order_candidate is None:
                if len(d["order_ids"]) == 1:
                    s.order_candidate = d["order_ids"][0]
                else:
                    s.order_ambiguous = True
        elif name == "lookup_order":
            s.order_verified, s.order_ambiguous, s.order_rejected = True, False, False
        elif name == "check_shipment":
            s.shipment = d
        elif name == "get_payment_details":
            s.payment = d
        elif name == "check_payment_change_eligibility":
            s.pay_elig = d["status"]
        elif name == "check_refund_eligibility":
            s.refund_elig = d["status"]
        elif name == "check_cancel_eligibility":
            s.cancel_elig = d["status"]
        elif name == "check_return_eligibility":
            s.return_elig = d["status"]
        elif name == "process_return":
            s.done["return_item"] = True
        elif name == "lookup_policy":
            s.done["policy_info"] = True
        elif name == "update_payment_method":
            s.done["payment_change"] = True
        elif name == "initiate_refund":
            s.done["refund"] = True
        elif name == "cancel_order":
            s.done["cancel_order"] = True
