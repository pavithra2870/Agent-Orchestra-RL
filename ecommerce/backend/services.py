"""Shared services for the API: database, router, parser, agent registry, demo world."""

from __future__ import annotations

import copy
import dataclasses
import json
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ecommerce.agents.base import Agent
from ecommerce.agents.baselines import RandomAgent, RuleBasedAgent
from ecommerce.backend.db import Database
from ecommerce.config import RESULTS_DIR, Settings, get_settings
from ecommerce.language.llm import OllamaClient
from ecommerce.language.parser import MessageParser
from ecommerce.language.router_store import get_router
from ecommerce.sim.inspection import complete_inspection
from ecommerce.sim.world import Order, World, add_showcase_customers, generate_world
from ecommerce.training import trainer

DEMO_WORLD_SEED = 2026


def original_demo_world() -> World:
    """The seeded demo database before any chat touched it (plus the demo-only showcase customers)."""
    return add_showcase_customers(generate_world(DEMO_WORLD_SEED))

# key -> (label, family, tag, description)
CHAT_AGENTS: Dict[str, Dict[str, str]] = {
    "dqn": {"label": "DQN", "family": "dqn", "tag": "", "about": "Deep Q-network, trained online with experience replay."},
    "ppo": {"label": "PPO", "family": "ppo", "tag": "", "about": "Policy-gradient agent (clipped objective, GAE), trained online."},
    "cql": {"label": "CQL (offline)", "family": "cql", "tag": "", "about": "Learned only from a log of past conversations."},
    "q_learning": {"label": "Q-learning", "family": "q_learning", "tag": "", "about": "Tabular off-policy baseline."},
    "sarsa": {"label": "SARSA", "family": "sarsa", "tag": "", "about": "Tabular on-policy baseline."},
    "rule_based": {"label": "Rule-based playbook", "family": "", "tag": "", "about": "Hand-written support playbook (no learning)."},
    "random": {"label": "Random (legal actions)", "family": "", "tag": "", "about": "Lower bound: random choice among allowed actions."},
}


class Services:
    def __init__(self, db_path: Optional[Path] = None, results_dir: Optional[Path] = None,
                 settings: Optional[Settings] = None):
        self.settings = settings or get_settings()
        self.results_dir = Path(results_dir or RESULTS_DIR)
        self.db = Database(db_path or (self.results_dir / "experiments.sqlite"))
        # Changes made by live chats are kept (so a cancel or a refund is still there on the next message) in a small
        # file next to the run database. "Restore original data" deletes it and regenerates the world from its seed.
        self.state_path = self.db.path.parent / "demo_state.json"
        self.db.mark_interrupted_jobs()
        self._agents: Dict[str, Agent] = {}
        self._parsers: Dict[str, MessageParser] = {}
        self._world: Optional[World] = None
        self._lock = threading.RLock()
        self.llm = OllamaClient()
        self._llm_ok, self._llm_checked = False, 0.0

    def llm_available(self, max_age: float = 30.0) -> bool:
        """Cached for a few seconds: asking a busy local model every time would make /api/health slow."""
        if time.time() - self._llm_checked > max_age:
            self._llm_ok, self._llm_checked = self.llm.available(), time.time()
        return self._llm_ok

    # ---- demo world -----------------------------------------------------------
    def demo_world(self) -> World:
        """The live demo database: the original seeded world plus every change made by earlier chats."""
        with self._lock:
            if self._world is None:
                w = original_demo_world()
                saved = self._load_changes()
                for oid, d in saved.get("orders", {}).items():
                    if oid in w.orders:
                        w.orders[oid] = Order(**d)
                for cid, claims in saved.get("claims", {}).items():
                    if cid in w.customers:
                        w.customers[cid].claims = claims
                self._world = w
            return self._world

    def _load_changes(self) -> Dict[str, Dict[str, Any]]:
        try:
            d = json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            return {}
        if d and "orders" not in d and "claims" not in d:         # the first file format was just the orders
            d = {"orders": d}
        return d

    def commit_orders(self, changed: Dict[str, Order], claims: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> None:
        """Keep the orders (and the customer's damage-claim history) a live chat changed, in memory and on disk."""
        if not changed and not claims:
            return
        with self._lock:
            w = self.demo_world()
            saved = self._load_changes()
            saved.setdefault("orders", {})
            saved.setdefault("claims", {})
            for oid, order in changed.items():
                w.orders[oid] = order
                saved["orders"][oid] = dataclasses.asdict(order)
            for cid, lst in (claims or {}).items():
                w.customers[cid].claims = lst
                saved["claims"][cid] = lst
            self._write_state(saved)

    def _write_state(self, saved: Dict[str, Any]) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(saved, indent=1), encoding="utf-8")
        tmp.replace(self.state_path)

    def changed_orders(self) -> List[Dict[str, Any]]:
        """Orders that differ from the original seeded data, for the dashboard."""
        with self._lock:
            original, now = original_demo_world(), self.demo_world()
            out = []
            for oid, o in now.orders.items():
                a, b = dataclasses.asdict(original.orders[oid]), dataclasses.asdict(o)
                diff = {k: {"original": a[k], "now": b[k]} for k in a if a[k] != b[k]}
                if diff:
                    out.append({"order_id": oid, "customer_id": o.customer_id, "changes": diff})
            return out

    def restore_demo_world(self) -> int:
        """Back to the original seeded data. Returns how many orders had been changed."""
        with self._lock:
            n = len(self.changed_orders())
            if self.state_path.exists():
                self.state_path.unlink()
            self._world = None
            return n

    # ---- inspection of returns (a step of the shop's process, done by a human team; demo buttons on the dashboard)
    def returns_waiting(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [{"order_id": o.order_id, "customer_id": o.customer_id, "customer": self.demo_world().customers[o.customer_id].name,
                     "items": o.items, "total": o.total, "kind": o.return_kind, "reason": o.return_reason, "return_id": o.return_id}
                    for o in self.demo_world().orders.values() if o.return_state == "requested"]

    def inspect_return(self, order_id: str, passed: bool) -> Dict[str, Any]:
        with self._lock:
            w = self.demo_world()
            if order_id not in w.orders:
                raise KeyError(order_id)
            order = copy.deepcopy(w.orders[order_id])
            complete_inspection(order, passed)              # raises ValueError when no return is waiting
            self.commit_orders({order_id: order})
            if order.return_state == "inspection_failed":
                message = "Inspection failed: no refund and no replacement; the item is sent back to the customer."
            elif order.return_kind == "exchange":
                message = f"Inspection passed: the replacement ({order.exchange_id}) is sent."
            else:
                message = f"Inspection passed: the refund ({order.refund_id}) of {order.total:.0f} rupees is released."
            return {"order_id": order_id, "return_state": order.return_state, "payment_state": order.payment_state,
                    "refund_id": order.refund_id, "exchange_id": order.exchange_id, "message": message}

    def damage_claims(self) -> List[Dict[str, Any]]:
        """Audit list: every damage claim on record, with who and how much (instant refunds are reviewed afterwards)."""
        with self._lock:
            return [{"customer_id": c.customer_id, "customer": c.name, **claim}
                    for c in self.demo_world().customers.values() for claim in c.claims]

    # ---- parser / agents --------------------------------------------------------
    def parser(self, mode: str) -> MessageParser:
        with self._lock:
            if mode not in self._parsers:
                self._parsers[mode] = MessageParser(get_router(), mode="llm" if mode == "llm" else "rules", llm=self.llm)
            return self._parsers[mode]

    def agent_available(self, key: str) -> bool:
        spec = CHAT_AGENTS.get(key)
        if spec is None:
            return False
        if not spec["family"]:
            return True
        return trainer.model_path(spec["family"], 0, "best", spec["tag"]).exists()

    def agent(self, key: str, seed: int = 0) -> Agent:
        spec = CHAT_AGENTS.get(key)
        if spec is None or not self.agent_available(key):
            raise KeyError(f"agent '{key}' is not available")
        with self._lock:
            cache_key = f"{key}:{seed}"
            if cache_key not in self._agents:
                if key == "rule_based":
                    self._agents[cache_key] = RuleBasedAgent()
                elif key == "random":
                    self._agents[cache_key] = RandomAgent(masked=True, seed=seed)
                else:
                    self._agents[cache_key] = trainer.load_trained(spec["family"], 0, "best", spec["tag"])
            ag = self._agents[cache_key]
        if key == "random":
            ag = RandomAgent(masked=True, seed=seed)       # reproducible per run
        return ag

    def list_agents(self) -> List[Dict[str, Any]]:
        out = []
        for key, spec in CHAT_AGENTS.items():
            entry = {"key": key, "label": spec["label"], "about": spec["about"], "available": self.agent_available(key),
                     "hyperparameters": None, "training": None}
            if spec["family"]:
                d = trainer.run_dir(spec["family"], 0, spec["tag"])
                if (d / "config.json").exists():
                    entry["hyperparameters"] = json.loads((d / "config.json").read_text()).get("hyperparameters")
                if (d / "summary.json").exists():
                    entry["training"] = json.loads((d / "summary.json").read_text())
            out.append(entry)
        return out

    # ---- results files ------------------------------------------------------------
    def read_json(self, *parts: str) -> Optional[Any]:
        p = self.results_dir.joinpath(*parts)
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
