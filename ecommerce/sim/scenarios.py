"""Seeded scenario generator.

A scenario = one customer message + the (hidden) true world state behind it.
The same (split, index) always produces the same scenario, so tests and
evaluations are reproducible.

Scenario types
  standard        everything clear
  missing_order   customer has several orders and does not say which one -> must ask
  wrong_order     customer quotes an order id that is not theirs        -> must ask
  low_confidence  the parser is unsure / missed a goal                   -> must ask
  out_of_scope    not something the shop's tools can do                  -> hand to a human
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ecommerce.config import Settings, WorldCfg, get_settings
from ecommerce.sim import rules
from ecommerce.sim.text import render_message
from ecommerce.sim.policy import TOPIC_KEYS
from ecommerce.sim.world import GOALS, Customer, Order, World, generate_world
from datetime import date, timedelta

ALL_COMBOS: List[Tuple[str, ...]] = [
    ("delivery_status",), ("payment_change",), ("refund",), ("cancel_order",),
    ("delivery_status", "payment_change"), ("delivery_status", "refund"), ("delivery_status", "cancel_order"),
    ("payment_change", "refund"), ("delivery_status", "payment_change", "refund"),
    # Group B: returns / damage claims and policy questions
    ("return_item",), ("policy_info",), ("delivery_status", "return_item"), ("policy_info", "return_item"),
    ("policy_info", "refund"), ("policy_info", "cancel_order"), ("policy_info", "delivery_status"),
]
HELDOUT_COMBOS = [("delivery_status", "cancel_order"), ("payment_change", "refund"),
                  ("delivery_status", "payment_change", "refund"),
                  ("delivery_status", "return_item"), ("policy_info", "delivery_status")]
_NEW_GOALS = {"return_item", "policy_info"}
# The older goal combinations appear twice in the training mix so the new return / policy requests (about 30% of
# training scenarios) do not crowd out what the agents already do well.
TRAIN_COMBOS = [c for c in ALL_COMBOS if c not in HELDOUT_COMBOS for _ in range(1 if _NEW_GOALS & set(c) else 2)]

RELEVANT_TOOLS = {
    "delivery_status": ["check_shipment"],
    "payment_change": ["check_payment_change_eligibility", "update_payment_method"],
    "refund": ["check_refund_eligibility", "initiate_refund"],
    "cancel_order": ["check_cancel_eligibility", "cancel_order"],
    "return_item": ["check_return_eligibility", "process_return"],
    "policy_info": ["lookup_policy"],
}


@dataclass
class Scenario:
    scenario_id: str
    seed: int
    world: World
    customer_id: str
    order_id: str
    true_goals: Tuple[str, ...]
    target_method: Optional[str]
    scenario_type: str
    text: str
    phrasing_set: str
    order_id_in_text: Optional[str]
    parsed_goals: Tuple[str, ...]
    parse_conf: float
    scope_score: float
    failure_multiplier: float = 1.0
    down_tools: Tuple[str, ...] = ()
    split: str = ""
    # what the parser thought the new payment method was (only used when use_parsed_target is True)
    parsed_target: Optional[str] = None
    use_parsed_target: bool = False
    # Group B. The TRUE answers of a customer asking for a return / damage claim, and the policy topic of a policy
    # question. return_info_given=False means the parser did not get them from the text, so the agent must ask
    # (the answers are revealed by the clarification). The declaration is never in the text, so it is always False
    # at the start of a return.
    policy_topic: Optional[str] = None
    return_reason: Optional[str] = None
    return_kind: Optional[str] = None
    return_declared: bool = True
    return_info_given: bool = True
    # questions already used up before this episode starts (live chat only): a clearly off-topic message arrives with the
    # question budget used, so the agent hands it over instead of asking
    prior_clarifications: int = 0

    def public(self) -> Dict[str, object]:
        """JSON-friendly view (no world object)."""
        return {
            "scenario_id": self.scenario_id, "split": self.split, "type": self.scenario_type,
            "text": self.text, "customer_id": self.customer_id, "true_goals": list(self.true_goals),
            "parsed_goals": list(self.parsed_goals), "parse_conf": round(self.parse_conf, 3),
            "scope_score": round(self.scope_score, 3), "down_tools": list(self.down_tools),
            "failure_multiplier": self.failure_multiplier, "policy_topic": self.policy_topic,
            "return_reason": self.return_reason, "return_kind": self.return_kind, "return_declared": self.return_declared,
        }


@dataclass
class SplitSpec:
    name: str
    base_seed: int
    combos: List[Tuple[str, ...]]
    phrasing_set: str = "A"
    type_probs: Optional[Dict[str, float]] = None
    failure_multiplier: float = 1.0
    down_prob: float = 0.0
    typo_prob: Optional[float] = None


def _outcome_matches(order: Order, goal: str, want: str, target: Optional[str], world: World) -> bool:
    today = date.fromisoformat(world.today)
    if goal in ("delivery_status", "return_item", "policy_info"):
        return True          # return outcomes are built into the order afterwards (_shape_return)
    if goal == "payment_change":
        e = rules.payment_change_eligibility(order, target or "UPI")
    elif goal == "refund":
        e = rules.refund_eligibility(order, world.cfg, today)
    else:
        e = rules.cancel_eligibility(order)
    return e.status == want


def _shape_return(world: World, customer: Customer, order: Order, want: str, reason: str, rng: np.random.Generator) -> bool:
    """Edit this scenario's own copy of the order so that a return request has the wanted outcome
    (eligible / needs_human / ineligible). Returns the customer's declaration (False only for one ineligible case)."""
    cfg, today = world.cfg, date.fromisoformat(world.today)
    cap, hv = int(cfg.instant_refund_cap), int(cfg.high_value_threshold)

    def make_delivered(days: int, total: int, paid: bool = True) -> None:
        d = today - timedelta(days=days)
        order.status, order.delivered_on, order.expected_delivery, order.delay_days = "delivered", d.isoformat(), d.isoformat(), 0
        order.out_for_delivery, order.total = False, float(total)
        order.payment_method = "UPI" if rng.random() < 0.6 else "CARD"
        order.payment_state = "paid" if paid else "pending"
        order.return_state = order.return_kind = order.return_reason = order.return_id = order.exchange_id = None
        order.refund_id = None
        order.tracking = [{"date": (d - timedelta(days=3)).isoformat(), "event": "Order placed"}, {"date": d.isoformat(), "event": "Delivered"}]

    small = lambda: int(rng.integers(100, cap + 1))             # noqa: E731
    recent = lambda: {"order_id": f"ORD-{int(rng.integers(1000, 10000))}", "date": (today - timedelta(days=int(rng.integers(1, 20)))).isoformat(),   # noqa: E731
                      "amount": float(small()), "mode": "instant_refund"}
    declared = True
    if want == "eligible":
        customer.claims = customer.claims[: max(0, cfg.damage_claims_limit - 1)]
        if reason == "damaged":
            make_delivered(int(rng.integers(0, cfg.damage_report_window_days + 1)), small())
        else:
            make_delivered(int(rng.integers(0, cfg.return_window_days + 1)), int(rng.integers(150, hv)))
    elif want == "needs_human":
        if reason == "damaged":
            sub = str(rng.choice(["late", "big", "history"]))
            if sub == "late":
                make_delivered(int(rng.integers(cfg.damage_report_window_days + 1, 12)), small())
            elif sub == "big":
                make_delivered(int(rng.integers(0, cfg.damage_report_window_days + 1)), int(rng.integers(cap + 50, 15000)))
            else:
                make_delivered(int(rng.integers(0, cfg.damage_report_window_days + 1)), small())
                customer.claims = [recent() for _ in range(cfg.damage_claims_limit)]
        else:
            make_delivered(int(rng.integers(0, cfg.return_window_days + 1)), int(rng.integers(hv + 100, 20000)))
    else:
        subs = ["not_delivered", "declined", "claimed", "unpaid"] + (["window"] if reason == "change_of_mind" else [])
        sub = str(rng.choice(subs))
        if sub == "not_delivered":
            if order.status == "delivered":
                order.status, order.delivered_on = "shipped", None
        elif sub == "declined":
            make_delivered(int(rng.integers(0, cfg.damage_report_window_days + 1)), small())
            declared = False
        elif sub == "claimed":
            make_delivered(int(rng.integers(0, cfg.damage_report_window_days + 1)), small())
            order.return_state = str(rng.choice(["requested", "inspection_failed"]))
        elif sub == "unpaid":
            make_delivered(int(rng.integers(0, cfg.damage_report_window_days + 1)), small(), paid=False)
            order.payment_method = "COD"
        else:
            make_delivered(int(rng.integers(cfg.return_window_days + 1, 20)), small())
    return declared


def _sample_scenario_type(rng: np.random.Generator, probs: Dict[str, float]) -> str:
    names = list(probs.keys())
    p = np.array([probs[n] for n in names], dtype=float)
    return str(rng.choice(names, p=p / p.sum()))


def generate_scenario(spec: SplitSpec, index: int, settings: Optional[Settings] = None) -> Scenario:
    settings = settings or get_settings()
    rng = np.random.default_rng(spec.base_seed * 100_003 + index)
    probs = spec.type_probs or settings.scenarios.type_probs
    s_type = _sample_scenario_type(rng, probs)
    typo_prob = settings.scenarios.typo_prob if spec.typo_prob is None else spec.typo_prob

    true_goals: Tuple[str, ...] = () if s_type == "out_of_scope" else tuple(
        spec.combos[int(rng.integers(0, len(spec.combos)))])
    if true_goals == ("policy_info",):
        s_type = "standard"            # a policy question needs no order, so there is nothing to ask for
    target_method = str(rng.choice(["UPI", "CARD"], p=[0.7, 0.3])) if "payment_change" in true_goals else None
    wants = {}
    for g in true_goals:
        if g == "refund":
            wants[g] = str(rng.choice(["eligible", "ineligible", "needs_human"], p=[0.5, 0.35, 0.15]))
        elif g in ("payment_change", "cancel_order"):
            wants[g] = "eligible" if rng.random() < 0.5 else "ineligible"

    # Group B: a policy topic, and the customer's true answers for a return. Drawn AFTER the older random draws, and
    # only when needed, so scenarios without these goals are generated exactly as before.
    policy_topic = str(rng.choice(TOPIC_KEYS)) if "policy_info" in true_goals else None
    return_reason = return_kind = None
    return_declared, kind_in_text = True, ""
    if "return_item" in true_goals:
        return_reason = str(rng.choice(["damaged", "change_of_mind"]))
        return_kind = str(rng.choice(["exchange", "refund"]))
        kind_in_text = return_kind if rng.random() < 0.5 else ""
        wants["return_item"] = str(rng.choice(["eligible", "ineligible", "needs_human"], p=[0.45, 0.30, 0.25]))

    mini = WorldCfg(**{**settings.world.model_dump(), "n_customers": 4})
    best = None
    for _ in range(200):
        w = generate_world(int(rng.integers(0, 2**31 - 1)), mini)
        c = w.customers["CUST-0001"]
        if s_type == "missing_order" and len(c.order_ids) < 2:
            continue
        o = w.orders[c.order_ids[int(rng.integers(0, len(c.order_ids)))]]
        best = (w, c, o)
        if all(_outcome_matches(o, g, wants[g], target_method, w) for g in wants):
            break
    world, customer, order = best
    if "return_item" in true_goals:
        return_declared = _shape_return(world, customer, order, wants["return_item"], return_reason, rng)

    other_orders = [o for o in world.orders.values() if o.customer_id != customer.customer_id]
    multi = len(customer.order_ids) > 1
    if s_type == "missing_order":
        order_id_in_text = None
    elif s_type == "wrong_order":
        order_id_in_text = other_orders[0].order_id if other_orders else "ORD-0000"
    elif s_type == "out_of_scope":
        order_id_in_text = None
    else:
        order_id_in_text = order.order_id if (multi or rng.random() < 0.6) else None
    order_ref = f"order {order_id_in_text}" if order_id_in_text and rng.random() < 0.7 else (
        order_id_in_text or "my order")

    text = render_message(true_goals, target_method or "UPI", order_ref, spec.phrasing_set, rng, typo_prob,
                          return_reason or "damaged", kind_in_text, policy_topic or "")

    if s_type == "out_of_scope":
        parsed, conf, scope = (), float(rng.uniform(0.2, 0.5)), float(rng.uniform(0.05, 0.3))
    elif s_type == "low_confidence":
        keep = [g for g in true_goals if rng.random() < 0.5]
        if len(keep) == len(true_goals):
            keep = keep[:-1]
        parsed, conf, scope = tuple(keep), float(rng.uniform(0.2, 0.5)), float(rng.uniform(0.5, 0.9))
    else:
        parsed, conf, scope = true_goals, float(rng.uniform(0.75, 0.99)), float(rng.uniform(0.55, 0.95))

    down: Tuple[str, ...] = ()
    if spec.down_prob > 0 and rng.random() < spec.down_prob:
        pool = [t for g in true_goals for t in RELEVANT_TOOLS[g]] if true_goals and rng.random() < 0.7 else \
            [t for ts in RELEVANT_TOOLS.values() for t in ts]
        down = (str(rng.choice(pool)),)

    return Scenario(
        scenario_id=f"{spec.name}-{index:05d}", seed=spec.base_seed * 100_003 + index, world=world,
        customer_id=customer.customer_id, order_id=order.order_id, true_goals=true_goals,
        target_method=target_method, scenario_type=s_type, text=text, phrasing_set=spec.phrasing_set,
        order_id_in_text=order_id_in_text, parsed_goals=parsed, parse_conf=conf, scope_score=scope,
        failure_multiplier=spec.failure_multiplier, down_tools=down, split=spec.name,
        policy_topic=policy_topic, return_reason=return_reason, return_kind=return_kind, return_declared=return_declared,
        return_info_given=("return_item" not in true_goals),
    )


def generate_scenarios(spec: SplitSpec, n: int, settings: Optional[Settings] = None) -> List[Scenario]:
    return [generate_scenario(spec, i, settings) for i in range(n)]


def scenario_from_world(
    world: World,
    customer_id: str,
    text: str,
    parsed_goals: Tuple[str, ...],
    target_method: Optional[str] = None,
    order_id_in_text: Optional[str] = None,
    parse_conf: float = 0.95,
    scope_score: float = 0.9,
    scenario_type: str = "live",
    true_goals: Optional[Tuple[str, ...]] = None,
    order_id: Optional[str] = None,
    seed: int = 0,
    failure_multiplier: float = 1.0,
    down_tools: Tuple[str, ...] = (),
    scenario_id: str = "custom",
    policy_topic: Optional[str] = None,
    return_reason: Optional[str] = None,
    return_kind: Optional[str] = None,
    return_declared: bool = True,
    return_info_given: bool = True,
    prior_clarifications: int = 0,
) -> Scenario:
    """Build a scenario around an existing world (live chat, golden tests).

    For a live message there are no labels, so the goals the parser found are treated as the
    customer's goals. The order the customer means is the quoted one, or their only order.
    """
    customer = world.customers[customer_id]
    if order_id is None:
        order_id = order_id_in_text if order_id_in_text in world.orders else customer.order_ids[0]
    return Scenario(
        scenario_id=scenario_id, seed=seed, world=world, customer_id=customer_id, order_id=order_id,
        true_goals=tuple(parsed_goals if true_goals is None else true_goals), target_method=target_method,
        scenario_type=scenario_type, text=text, phrasing_set="live", order_id_in_text=order_id_in_text,
        parsed_goals=tuple(parsed_goals), parse_conf=parse_conf, scope_score=scope_score,
        failure_multiplier=failure_multiplier, down_tools=down_tools, split="live",
        policy_topic=policy_topic, return_reason=return_reason, return_kind=return_kind, return_declared=return_declared,
        return_info_given=return_info_given, prior_clarifications=prior_clarifications,
    )
