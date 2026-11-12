"""Intent router (guardrail layer 1): message -> goals, scope score, confidence.

How it works (no learning of the RL policy happens here):
  1. strip the greeting and split the message into clauses ("... and also ...");
  2. embed every clause and compare it (cosine similarity) with example sentences for each
     goal and for "out of scope";
  3. a clause becomes a goal only if its best class is a goal AND the similarity is above a
     threshold; otherwise it is ignored / treated as out of scope;
  4. a small calibrated model turns (similarity, margin, ...) into a confidence in [0, 1].

The thresholds are tuned on a VALIDATION set built from phrase set A, and reported on the
held-out phrase set B (see ecommerce/language/evaluate_language.py).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ecommerce.config import get_settings
from ecommerce.language.embedder import Embedder, get_embedder
from ecommerce.sim.text import PHRASES, example_sentences
from ecommerce.sim.world import GOALS

CLASSES = GOALS + ("out_of_scope",)
GREETING_RE = re.compile(r"^(?:hi there|hello there|hello|hey team|good evening|excuse me|dear support|hi)\s*,\s*", re.I)
SPLIT_RE = re.compile(r"\s*(?:[.;?!]+\s+|,\s*also\s+|\bwhile you'?re at it,?\s*|\band on top of that,?\s*|"
                      r"\badditionally,?\s*|\band also\b|\bas well as\b|\bby the way,?\s*|\band then\b|\bplus\b|"
                      r"\balso\b|\band\b)\s*", re.I)


ORDER_ID_RE = re.compile(r"\b(?:order\s+)?ORD-\d{4}\b", re.I)


LOOSE_ORDER_RE = re.compile(r"\b(?:order|ord)\b[\s#:.-]*\d{4}\b|#\d{4}\b", re.I)


def normalise(text: str) -> str:
    """An order id says nothing about intent, so replace it with the words the example sentences use.
    Handles ORD-4175, "order 4175", "ord 4175", "order #4175" and "#4175"."""
    return LOOSE_ORDER_RE.sub("my order", ORDER_ID_RE.sub("my order", text))


_LETTERS = "abcdefghijklmnopqrstuvwxyz"
_WORD_RE = re.compile(r"[A-Za-z']+")


def _edits1(w: str) -> set:
    """All strings one typing slip away from w (Norvig): a missing, extra, wrong or swapped letter."""
    sp = [(w[:i], w[i:]) for i in range(len(w) + 1)]
    return ({a + b[1:] for a, b in sp if b} | {a + b[1] + b[0] + b[2:] for a, b in sp if len(b) > 1}
            | {a + c + b[1:] for a, b in sp if b for c in _LETTERS} | {a + c + b for a, b in sp for c in _LETTERS})


def repair_typos(text: str, vocab: Counter) -> str:
    """Fix single-slip typos ("oredr", "whree", "moeny") using only words that occur in the router's example sentences.
    A word that is already known, short, or has no unique-enough neighbour is left alone."""
    def fix(m: "re.Match") -> str:
        w = m.group(0)
        lw = w.lower()
        if len(lw) < 4 or lw in vocab:
            return w
        cands = [c for c in _edits1(lw) if c in vocab]
        return max(cands, key=lambda c: vocab[c]) if cands else w
    return _WORD_RE.sub(fix, text)


def split_clauses(text: str) -> List[str]:
    text = GREETING_RE.sub("", normalise(text).strip())
    parts = [p.strip(" .,?!") for p in SPLIT_RE.split(text)]
    return [p for p in parts if len(p.split()) >= 2] or [text.strip()]


@dataclass
class ClauseResult:
    text: str
    label: str
    sim: float
    margin: float
    sims: Dict[str, float]


@dataclass
class RouterOutput:
    goals: Tuple[str, ...]
    scope_score: float
    parse_conf: float
    in_scope: bool
    clauses: List[ClauseResult] = field(default_factory=list)
    features: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["goals"] = list(self.goals)
        return d


class LogisticCalibrator:
    """sigmoid(w . x + b). Stored as plain numbers so it can be saved as JSON (no pickle)."""

    def __init__(self, coef, intercept: float):
        self.coef, self.intercept = np.asarray(coef, dtype=float), float(intercept)

    def predict_proba(self, X) -> np.ndarray:
        z = np.asarray(X, dtype=float) @ self.coef + self.intercept
        p = 1.0 / (1.0 + np.exp(-z))
        return np.stack([1 - p, p], axis=1)


class IntentRouter:
    def __init__(self, embedder: Optional[Embedder] = None, examples: Optional[Dict[str, List[str]]] = None,
                 in_scope_threshold: Optional[float] = None, top_k: int = 2):
        self.examples = examples or example_sentences("A")
        corpus = [s for v in self.examples.values() for s in v]
        self.embedder = embedder or get_embedder(corpus)
        self.top_k = top_k
        self.tau_in = in_scope_threshold if in_scope_threshold is not None else get_settings().guardrails.in_scope_threshold
        self._ex_vecs = {c: self.embedder.encode(self.examples[c]) for c in CLASSES}
        # words of the example sentences: the only vocabulary used to repair typos in a customer message
        self.vocab = Counter(w.lower() for v in self.examples.values() for sent in v for w in _WORD_RE.findall(sent))
        self.repair = True
        self.calibrator = None      # set by fit_calibration()

    # ---- scoring ----------------------------------------------------------
    def _class_sims(self, vec: np.ndarray) -> Dict[str, float]:
        out = {}
        for c in CLASSES:
            s = np.sort(self._ex_vecs[c] @ vec)[::-1]
            out[c] = float(np.mean(s[: self.top_k]))
        return out

    def route(self, text: str) -> RouterOutput:
        clauses = split_clauses(repair_typos(text, self.vocab) if self.repair else text)
        vecs = self.embedder.encode(clauses)
        results: List[ClauseResult] = []
        for t, v in zip(clauses, vecs):
            sims = self._class_sims(v)
            ordered = sorted(sims.items(), key=lambda kv: -kv[1])
            (best, s1), (_, s2) = ordered[0], ordered[1]
            results.append(ClauseResult(t, best, s1, s1 - s2, sims))

        goals: List[str] = []
        for r in results:
            if r.label != "out_of_scope" and r.sim >= self.tau_in and r.label not in goals:
                goals.append(r.label)
        goal_sims = [max(r.sims[g] for g in GOALS) for r in results]
        scope = float(max(goal_sims))
        accepted = [r for r in results if r.label != "out_of_scope" and r.sim >= self.tau_in]
        feats = {
            "min_sim": float(min((r.sim for r in accepted), default=scope)),
            "min_margin": float(min((r.margin for r in accepted), default=min(r.margin for r in results))),
            "scope": scope,
            "n_ignored": float(sum(1 for r in results if r not in accepted and r.label != "out_of_scope")),
            "n_goals": float(len(goals)),
        }
        conf = self._confidence(feats, bool(goals))
        ordered_goals = tuple(g for g in GOALS if g in goals)
        return RouterOutput(ordered_goals, scope, conf, in_scope=bool(goals), clauses=results, features=feats)

    # ---- confidence ---------------------------------------------------------
    @staticmethod
    def _vector(f: Dict[str, float]) -> List[float]:
        return [f["min_sim"], f["min_margin"], f["scope"], f["n_ignored"], f["n_goals"]]

    def _confidence(self, feats: Dict[str, float], has_goals: bool) -> float:
        if self.calibrator is None:
            return float(np.clip(0.5 + 2.0 * feats["min_margin"] + 0.5 * (feats["min_sim"] - self.tau_in), 0.0, 1.0))
        return float(self.calibrator.predict_proba(np.array([self._vector(feats)]))[0, 1])

    def fit_calibration(self, messages: Sequence[str], true_goal_sets: Sequence[Tuple[str, ...]]) -> Dict[str, float]:
        """Learn P(goal set is exactly right | router features) with logistic regression."""
        from sklearn.linear_model import LogisticRegression

        self.calibrator = None
        X, y = [], []
        for m, truth in zip(messages, true_goal_sets):
            out = self.route(m)
            X.append(self._vector(out.features))
            y.append(int(set(out.goals) == set(truth)))
        y = np.array(y)
        if y.min() == y.max():
            return {"calibration_samples": float(len(y)), "positive_rate": float(y.mean())}
        lr = LogisticRegression(C=5.0, max_iter=1000).fit(np.array(X), y)
        self.calibrator = LogisticCalibrator(lr.coef_[0], lr.intercept_[0])
        return {"calibration_samples": float(len(y)), "positive_rate": float(y.mean())}

    def tune_threshold(self, messages: Sequence[str], true_goal_sets: Sequence[Tuple[str, ...]],
                       grid: Sequence[float] = tuple(np.arange(0.15, 0.81, 0.02))) -> Dict[str, float]:
        """Pick the in-scope threshold that gives the best exact-match accuracy on validation messages."""
        best = (-1.0, self.tau_in)
        for t in grid:
            self.tau_in = float(t)
            acc = np.mean([set(self.route(m).goals) == set(g) for m, g in zip(messages, true_goal_sets)])
            if acc > best[0] + 1e-9:
                best = (float(acc), float(t))
        self.tau_in = best[1]
        return {"tau_in": best[1], "validation_exact_match": best[0]}

    # ---- save / load the tuned settings ----------------------------------------
    def export_params(self) -> Dict:
        return {"tau_in": self.tau_in, "top_k": self.top_k, "embedder": self.embedder.name,
                "calibrator": None if self.calibrator is None else
                {"coef": self.calibrator.coef.tolist(), "intercept": self.calibrator.intercept}}

    def load_params(self, params: Dict) -> "IntentRouter":
        self.tau_in = float(params["tau_in"])
        cal = params.get("calibrator")
        self.calibrator = LogisticCalibrator(cal["coef"], cal["intercept"]) if cal else None
        return self
