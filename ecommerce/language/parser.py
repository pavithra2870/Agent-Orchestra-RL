"""Message -> ParsedRequest (goals, order id, target payment method, scope score, confidence).

Two modes:
  "rules"   embedding router for goals + regex for order id / payment method   (fast, always available)
  "llm"     local LLM (phi4-mini) extracts a strict JSON request, then the embedding router CROSS-CHECKS it:
              - a goal the LLM invented but the router finds no support for is dropped (hallucination guard)
              - if the LLM and the router disagree, the confidence is capped low so the agent asks first
            If the LLM is down or returns invalid JSON, we fall back to "rules".
The LLM never chooses tools. It only reads the message.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple


from ecommerce.config import get_settings
from ecommerce.language.knowledge import KnowledgeBase
from ecommerce.language.llm import OllamaClient
from ecommerce.language.router import IntentRouter, RouterOutput
from ecommerce.sim.world import GOALS

ORDER_RE = re.compile(r"\bORD-\d{4}\b", re.I)
METHOD_PATTERNS = (("UPI", re.compile(r"\bupi\b", re.I)),
                   ("CARD", re.compile(r"\b(?:credit |debit |my )?card\b", re.I)))
SUPPORT_FLOOR = 0.28            # a goal the LLM names needs at least this similarity to some clause
LOW_CONF_CAP = 0.45

SCHEMA = {
    "type": "object",
    "properties": {
        "goals": {"type": "array", "items": {"type": "string", "enum": list(GOALS)}},
        "order_id": {"type": ["string", "null"]},
        "target_payment_method": {"type": ["string", "null"], "enum": ["UPI", "CARD", "COD", None]},
        "in_scope": {"type": "boolean"},
    },
    "required": ["goals", "order_id", "target_payment_method", "in_scope"],
}

SYSTEM = """You read customer-support messages for an online shop and output JSON.
Allowed goals (only these):
- delivery_status: the customer wants to know where an order is, its status or delivery date
- payment_change: the customer wants to change how an order is paid (for example cash on delivery to UPI or card)
- refund: the customer wants their money back for an order that was lost, never arrived, is very late or was cancelled. Wanting to send an item back is NOT a refund: that is return_item
- cancel_order: the customer wants an order cancelled
- return_item: the customer received an item and wants to return it, exchange it, or report that it is damaged or faulty
- policy_info: the customer asks a general question about the shop's policy (refunds, returns, cancellation, lost or damaged orders), not about one of their own orders
Rules:
- Include a goal ONLY if the customer clearly asks for it. Do not guess or add extra goals.
- If the message is not about one of these (recommendations, weather, jokes, passwords, addresses, accounts, general questions) return an empty goals list and in_scope false.
- If earlier messages are shown, read the NEW message in their light: a short reply like "where is it" or "and refund it too" refers to what was discussed. Output goals for the NEW message only.
- order_id is the id like ORD-1234 if the customer wrote one, otherwise null.
- target_payment_method is UPI, CARD or COD only when the customer names a new payment method, otherwise null.
Examples:
Message: "hello, where's my parcel ORD-1234?"
{"goals":["delivery_status"],"order_id":"ORD-1234","target_payment_method":null,"in_scope":true}
Message: "please check delivery of ORD-1111 and switch the payment to card"
{"goals":["delivery_status","payment_change"],"order_id":"ORD-1111","target_payment_method":"CARD","in_scope":true}
Message: "the lamp is broken, i want to send it back"
{"goals":["return_item"],"order_id":null,"target_payment_method":null,"in_scope":true}
Message: "what is your policy on returns?"
{"goals":["policy_info"],"order_id":null,"target_payment_method":null,"in_scope":true}
Message: "can you tell me a story about dragons"
{"goals":[],"order_id":null,"target_payment_method":null,"in_scope":false}"""


@dataclass
class ParsedRequest:
    goals: Tuple[str, ...]
    order_id: Optional[str]
    target_method: Optional[str]
    scope_score: float
    parse_conf: float
    source: str                                  # rules | llm+router | rules(fallback)
    router: Optional[RouterOutput] = None
    llm_goals: Optional[Tuple[str, ...]] = None
    dropped_goals: List[str] = field(default_factory=list)
    llm_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)
    policy_topic: Optional[str] = None       # set when "policy_info" is a goal: the closest policy topic (retrieval)
    policy_sim: float = 0.0
    return_reason: Optional[str] = None      # damaged | change_of_mind, only when the customer's words say so
    return_kind: Optional[str] = None        # exchange | refund, only when the customer's words say so
    off_topic: bool = False                  # no goal AND every part of the message is closest to the "out of scope" examples

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["goals"] = list(self.goals)
        d["llm_goals"] = list(self.llm_goals) if self.llm_goals is not None else None
        d["router"] = self.router.to_dict() if self.router else None
        return d


def extract_order_id(text: str) -> Optional[str]:
    m = ORDER_RE.search(text)
    return m.group(0).upper() if m else None


DAMAGED_RE = re.compile(r"\b(?:broken|damaged|damage|defective|defect|faulty|cracked|smashed|shattered|torn|dented?|crushed|scratched|"
                        r"leak(?:s|ed|ing)?|malfunction\w*|not working|stopped working|isn'?t working|(?:does not|doesn'?t|do not|don'?t) (?:work|turn on)|"
                        r"dead on arrival|no sound|bad shape)\b", re.I)
MIND_RE = re.compile(r"\b(?:(?:don'?t|do not) (?:like|want|need)|changed my mind|no longer (?:want|need)|not what i (?:expected|wanted|ordered)|"
                     r"not for me|isn'?t for me|doesn'?t suit|wrong (?:size|colou?r)|didn'?t meet|by mistake|not happy|(?:doesn'?t|does not|don'?t) fit|"
                     r"fit .* is wrong|decided to return|rather return)\b", re.I)
EXCHANGE_RE = re.compile(r"\b(?:exchange\w*|replac\w+|swap\w*|new one|different one|another one|replacement)\b", re.I)
REFUND_RE = re.compile(r"\b(?:refund\w*|money back|reimburs\w*|paid back|funds)\b", re.I)


NEVER_GOT_IT_RE = re.compile(r"\b(?:lost|never (?:arrived|came|reached|got)|not (?:arrive|arrived|received)|didn'?t (?:arrive|come|show)|late|delayed|missing|cancel\w*)\b", re.I)


RETURN_CUE_RE = re.compile(r"\b(?:return\w*|send(?:ing)? (?:it |this |them )?back|give (?:it )?back|exchange\w*)\b", re.I)
ALREADY_RETURNED_RE = re.compile(r"\b(?:(?:i|we)(?:'ve| have)|have|has|had|already|just)\s+(?:\w+\s+)?(?:returned|sent (?:it |this |them |the \w+ )?back|given (?:it |this |them )?back)\b|\b(?:returned|sent back) (?:it|this|them|my|the)\b.*\b(?:refund|money|paid)\b", re.I)


def extract_return_reason(text: str) -> Optional[str]:
    """Only what the customer's words say: damage wording, or a change-of-mind wording; otherwise None (the agent asks)."""
    if DAMAGED_RE.search(text):
        return "damaged"
    return "change_of_mind" if MIND_RE.search(text) else None


def extract_return_kind(text: str) -> Optional[str]:
    ex, rf = bool(EXCHANGE_RE.search(text)), bool(REFUND_RE.search(text))
    return "exchange" if ex and not rf else ("refund" if rf and not ex else None)


def extract_method(text: str) -> Optional[str]:
    for name, pat in METHOD_PATTERNS:
        if pat.search(text):
            return name
    return None


def _llm_prompt(text: str, history: Optional[List[Dict[str, str]]]) -> str:
    """The new message, preceded by the last few messages when there are any (kept short on purpose)."""
    if not history:
        return f'Message: "{text}"'
    lines = [f"- {'customer' if m.get('role') == 'user' else 'agent'}: {str(m.get('text', ''))[:300]}" for m in history[-3:]]
    return "\n".join(["Earlier messages (oldest first):"] + lines + [f'New message: "{text}"'])


class MessageParser:
    def __init__(self, router: IntentRouter, mode: str = "rules", llm: Optional[OllamaClient] = None):
        assert mode in ("rules", "llm")
        self.router, self.mode = router, mode
        self._kb: Optional[KnowledgeBase] = None
        self.llm = llm or (OllamaClient() if mode == "llm" else None)

    def _knowledge(self) -> KnowledgeBase:
        if self._kb is None:
            self._kb = KnowledgeBase(self.router.embedder)
        return self._kb

    def _fill_extras(self, p: ParsedRequest, text: str) -> ParsedRequest:
        """Things read from the customer's words that go with a goal: why / what they want for a return, the policy topic."""
        if not p.goals:
            # "I found nothing I can do" must reach the agent as an unsure parse with a low scope score, the way out-of-scope
            # messages look in training. The raw router is very confident (about 0.94) that an off-topic message is off-topic,
            # which the agents had never seen, so the learned ones wandered instead of handing over.
            if p.parse_conf > LOW_CONF_CAP or p.scope_score > 0.3:
                p.notes.append("no goal found: the parse is passed on as unsure, so the agent hands the message to a human")
            p.parse_conf, p.scope_score = min(p.parse_conf, LOW_CONF_CAP), min(p.scope_score, 0.3)
            p.off_topic = bool(p.router and p.router.clauses and all(c.label == "out_of_scope" for c in p.router.clauses))
        if "refund" in p.goals and "return_item" not in p.goals and DAMAGED_RE.search(text) and not NEVER_GOT_IT_RE.search(text):
            # "it arrived broken, I want my money back" is a damage claim (the customer chose the refund), not a plain refund
            p.goals = tuple(sorted(({g for g in p.goals if g != "refund"} | {"return_item"}), key=GOALS.index))
            p.notes.append("damage wording with a request for money back is handled as a damage claim")
        if "return_item" in p.goals and ALREADY_RETURNED_RE.search(text):
            # "I have returned it, I want my refund": the item is already back, so this is a refund request, not a new return
            p.goals = tuple(sorted(({g for g in p.goals if g != "return_item"} | {"refund"}), key=GOALS.index))
            p.notes.append("the customer says the item has already been returned, so this is a refund request, not a new return")
        if "return_item" in p.goals and "refund" in p.goals and not NEVER_GOT_IT_RE.search(text):
            # "return it and get my money back" is ONE request: the return flow already includes the refund
            p.goals = tuple(g for g in p.goals if g != "refund")
            p.notes.append("return and refund in one message are handled as a return (the refund follows the inspection)")
        if "return_item" in p.goals:
            p.return_reason, p.return_kind = extract_return_reason(text), extract_return_kind(text)
        if "policy_info" in p.goals:
            p.policy_topic, p.policy_sim = self._knowledge().best(text)
        return p

    def parse(self, text: str, history: Optional[List[Dict[str, str]]] = None) -> ParsedRequest:
        """history = the last few messages ({role: user|agent, text}), oldest first. Only the LLM uses it."""
        r = self.router.route(text)
        order_id, method = extract_order_id(text), extract_method(text)
        base = ParsedRequest(r.goals, order_id, method, r.scope_score, r.parse_conf, "rules", router=r)
        self._fill_extras(base, text)
        if self.mode != "llm":
            return base
        data, secs = self.llm.chat_json(SYSTEM, _llm_prompt(text, history), SCHEMA)
        base.llm_seconds = secs
        if data is None or not isinstance(data.get("goals"), list):
            base.source = "rules(fallback)"
            base.notes.append(f"LLM unavailable or invalid ({self.llm.last_error}); used the router only")
            return base

        llm_goals = tuple(g for g in GOALS if g in data["goals"])
        support = {g: max(c.sims[g] for c in r.clauses) for g in GOALS}
        kept = [g for g in llm_goals if support[g] >= SUPPORT_FLOOR]
        notes: List[str] = []
        if "refund" in kept and "return_item" in r.goals and not NEVER_GOT_IT_RE.search(text) and                 ("refund" not in r.goals and not REFUND_RE.search(text) or RETURN_CUE_RE.search(text)):
            # "I want to return it" is a return, not a request for money back: trust the router's reading of the words
            kept = [("return_item" if g == "refund" else g) for g in kept]
            notes.append("the LLM read a return as a refund; the customer did not ask for money back, so it was treated as a return")
        dropped = [g for g in llm_goals if g not in kept and not (g == "refund" and "return_item" in kept)]
        final = tuple(g for g in GOALS if g in kept)
        agree = set(llm_goals) == set(r.goals)
        agree = agree or set(kept) == set(r.goals)
        conf = r.parse_conf if agree else min(r.parse_conf, LOW_CONF_CAP)
        if dropped:
            notes.append(f"dropped goals with no router support: {dropped}")
        if not agree:
            notes.append(f"LLM {list(llm_goals)} and router {list(r.goals)} disagree, so confidence is capped")
        order = extract_order_id(text) or (data.get("order_id") if data.get("order_id") and ORDER_RE.fullmatch(str(data["order_id"])) else None)
        # The new payment method counts only if the customer's own words name it. A method the LLM "inferred" from
        # something vague like "pay online" is ignored, so the agent asks instead of assuming.
        method = extract_method(text)
        if data.get("target_payment_method") and not method:
            notes.append(f"the LLM guessed payment method {data['target_payment_method']}, but the customer did not name one, so it was ignored")
        scope = r.scope_score if final else min(r.scope_score, 0.3)
        return self._fill_extras(ParsedRequest(final, order, method, scope, conf, "llm+router", router=r, llm_goals=llm_goals,
                                               dropped_goals=dropped, llm_seconds=secs, notes=notes), text)


_STATUS_SYNONYMS = {"late": "delayed", "delayed": "delayed", "lost": "lost", "missing": "lost", "delivered": "delivered",
                    "arrived": "delivered", "returned": "returned", "shipped": "shipped", "processing": "processing",
                    "cancelled": "cancelled", "canceled": "cancelled"}


def resolve_order_reference(text: str, orders: List[Dict[str, Any]]) -> Optional[str]:
    """Find which of a customer's orders a message refers to. Returns an id only when exactly one order matches.

    Accepts the full id ("ORD-4749"), the last four digits ("4749"), a status word ("the lost order"),
    or a word from an item name ("the speaker"). Several matches or none -> None (the agent will ask)."""
    explicit = extract_order_id(text)
    if explicit:
        return explicit
    low = text.lower()
    digits = set(re.findall(r"\b\d{4}\b", text))
    by_digits = [o["order_id"] for o in orders if any(o["order_id"].endswith(d) for d in digits)]
    if len(set(by_digits)) == 1:
        return by_digits[0]
    wanted = {status for word, status in _STATUS_SYNONYMS.items() if re.search(rf"\b{word}\b", low)}
    by_status = [o["order_id"] for o in orders if o["status"] in wanted]
    if wanted and len(by_status) == 1:
        return by_status[0]
    words = set(re.findall(r"[a-z]{4,}", low))
    by_item = [o["order_id"] for o in orders if any(w in words for item in o["items"] for w in re.findall(r"[a-z]{4,}", item.lower()))]
    if len(set(by_item)) == 1:
        return by_item[0]
    return None
