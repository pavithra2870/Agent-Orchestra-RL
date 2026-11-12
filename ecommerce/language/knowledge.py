"""Topic matching for POLICY questions ("what happens if my order is lost?").

The policy texts live in ecommerce/sim/policy.py (they are what the tool `lookup_policy` returns). This module only
finds the closest topic for a question, by comparing it with example questions (embeddings), the same way the router
finds goals. The environment hands that topic to the tool; the RL agent decides whether and when to call it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ecommerce.config import Settings, get_settings
from ecommerce.language.embedder import Embedder
from ecommerce.sim.policy import TEST_QUESTIONS, TUNING_QUESTIONS, Topic, build_topics   # noqa: F401  (re-exported)


class KnowledgeBase:
    def __init__(self, embedder: Embedder, settings: Optional[Settings] = None):
        self.embedder = embedder
        self.topics = build_topics(settings)
        self._vecs = {k: embedder.encode(list(tp.examples)) for k, (tp, _) in self.topics.items()}

    def similarities(self, text: str) -> Dict[str, float]:
        v = self.embedder.encode([text])[0]
        return {k: float(np.max(vs @ v)) for k, vs in self._vecs.items()}

    def best(self, text: str) -> Tuple[Optional[str], float]:
        sims = self.similarities(text)
        key = max(sims, key=sims.get)
        return key, sims[key]

    def answer(self, key: str) -> str:
        return self.topics[key][1]

    def title(self, key: str) -> str:
        return self.topics[key][0].title

    def source(self, key: str) -> str:
        return self.topics[key][0].source


ORDER_ID = re.compile(r"\bORD-\d{4}\b", re.I)


def is_policy_question(kb: KnowledgeBase, text: str, goal_similarity: float, threshold: float, margin: float) -> Tuple[Optional[str], float]:
    """A message is a policy question when it is close to a known topic, closer than to any transactional request,
    and does not quote an order id (a message about a specific order is a transaction)."""
    if ORDER_ID.search(text):
        return None, 0.0
    key, sim = kb.best(text)
    if sim >= threshold and sim >= goal_similarity + margin:
        return key, sim
    return None, sim
