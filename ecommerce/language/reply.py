"""Reply writing (guardrail layer 3: output checks).

The reply is built only from VERIFIED facts taken from the tool results in the trace.
  template mode : fixed sentences filled with those facts (cannot invent anything)
  llm mode      : phi4-mini rewrites the same facts in friendlier words, then `check_reply`
                  looks for anything not in the facts; if it finds a problem we ship the template instead.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Set

from ecommerce.config import get_settings
from ecommerce.language.llm import OllamaClient
from ecommerce.sim.trace import TraceStep

_W = get_settings().world
REASON_TEXT = {
    "order_closed": "the order has already been completed or closed",
    "already_paid": "it has already been paid",
    "same_method": "it is already set to that payment method",
    "out_for_delivery": "the order is already out for delivery",
    "outside_policy": "it is outside our refund policy",
    "not_paid": "no payment has been made yet",
    "already_refunded": "it has already been refunded",
    "already_shipped_or_closed": "it has already shipped",
    "already_cancelled": "it has already been cancelled",
    "high_value_needs_approval": "refunds of this size need approval from a human agent",
    # returns, exchanges and damage claims
    "return_required": "an item that has reached you is refunded only after you send it back and it passes our inspection",
    "return_in_progress": "a return for it is already in progress",
    "inspection_failed": "the item you sent back did not pass our inspection, so it cannot be refunded and has been sent back to you",
    "not_delivered": "it has not reached you yet",
    "already_claimed": "a return, exchange or claim has already been opened for it",
    "declaration_not_confirmed": "the condition of the item was not confirmed",
    "outside_return_window": f"it is past the {_W.return_window_days}-day return window",
    "late_damage_report": f"the damage was reported more than {_W.damage_report_window_days * 24} hours after delivery",
    "above_instant_limit": "the order value is above what I can handle without a review",
    "claim_history": "there have been several recent damage claims on your account",
}
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
          "november", "december"]


@dataclass
class ReplyResult:
    text: str
    mode: str                                     # template | llm | template(fallback)
    problems: List[str] = field(default_factory=list)     # what the checker found in the LLM draft
    llm_draft: Optional[str] = None
    facts: Dict[str, Any] = field(default_factory=dict)
    llm_seconds: float = 0.0


def _ok(trace: Sequence[TraceStep], tool: str) -> List[TraceStep]:
    return [s for s in trace if s.action == tool and s.ok and not s.blocked]


def build_facts(customer_name: str, trace: Sequence[TraceStep], terminal: str, requested: Sequence[str],
                scenario_type: str = "") -> Dict[str, Any]:
    """Collect only what the tools confirmed."""
    facts: Dict[str, Any] = {"customer": customer_name.split()[0], "terminal": terminal, "outcomes": {},
                             }
    cust = _ok(trace, "lookup_customer")
    if cust:
        facts["orders"] = cust[-1].data.get("order_ids", [])
        facts["orders_detail"] = cust[-1].data.get("orders", [])
    order = _ok(trace, "lookup_order")
    if order:
        facts["order_id"] = order[-1].data["order_id"]
        facts["items"] = order[-1].data["items"]
    ship = _ok(trace, "check_shipment")
    out: Dict[str, Any] = facts["outcomes"]
    if "delivery_status" in requested:
        out["delivery_status"] = {"kind": "info", **ship[-1].data} if ship else {"kind": "missing"}
    if "policy_info" in requested:
        pol = _ok(trace, "lookup_policy")
        out["policy_info"] = {"kind": "info", **pol[-1].data} if pol else {"kind": "missing"}
    specs = {"payment_change": ("check_payment_change_eligibility", "update_payment_method"),
             "refund": ("check_refund_eligibility", "initiate_refund"),
             "cancel_order": ("check_cancel_eligibility", "cancel_order"),
             "return_item": ("check_return_eligibility", "process_return")}
    for goal, (check, act) in specs.items():
        if goal not in requested:
            continue
        done, checks = _ok(trace, act), _ok(trace, check)
        if done:
            out[goal] = {"kind": "done", **done[-1].data}
        elif checks and checks[-1].data.get("status") == "ineligible":
            out[goal] = {"kind": "declined", "reason": checks[-1].data.get("reason")}
        elif checks and checks[-1].data.get("status") == "needs_human":
            out[goal] = {"kind": "needs_human", "reason": checks[-1].data.get("reason")}
        else:
            out[goal] = {"kind": "missing"}
    # The shop's rule numbers are facts only when an explanation uses them. Offering them always made the LLM add unrelated
    # sentences ("you have 7 days to return any damaged items") to replies that had nothing to do with returns.
    if any(o.get("reason") in ("outside_return_window", "late_damage_report") for o in out.values() if isinstance(o, dict)):
        facts["rules"] = {"return_window_days": _W.return_window_days, "damage_report_hours": _W.damage_report_window_days * 24}
    failed_tools = sorted({s.action for s in trace if not s.ok and not s.blocked and s.error in ("timeout", "service_unavailable")})
    facts["tool_trouble"] = failed_tools
    if terminal == "awaiting_customer":
        rejected = next((s.args.get("order_id") for s in trace if s.action == "lookup_order" and not s.ok and not s.blocked
                         and s.error in ("order_not_found", "permission_denied")), None)
        several = bool(facts.get("orders")) and len(facts["orders"]) > 1
        # Ask about what is really missing: the order (unknown or not theirs) or, when the order is already
        # verified, what the customer wants done with it.
        qtype = "which_order" if rejected else ("what_do_you_need" if facts.get("order_id") else ("which_order" if several else "more_detail"))
        if not requested and not facts.get("order_id"):
            qtype = "more_detail"            # nothing was understood and no order is known, so "which order?" would be a non-answer
        facts["question"] = {"orders": facts.get("orders", []), "detail": facts.get("orders_detail", []),
                             "rejected_order": rejected, "type": qtype}
    return facts


def _because(o: Dict[str, Any], default: str) -> str:
    return REASON_TEXT.get(o.get("reason"), default)


def _action_sentence(goal: str, kind: str, o: Dict[str, Any], oid: str) -> str:
    """One sentence for a payment / refund / cancel outcome. Branches are evaluated lazily on purpose."""
    if goal == "payment_change":
        if kind == "done":
            return f"The payment method for {oid} is now {o.get('new_method')}."
        if kind == "declined":
            return f"I couldn't change the payment method for {oid} because {_because(o, 'it is not allowed')}."
        return "A human agent needs to review the payment change." if kind == "needs_human" else "I wasn't able to complete the payment change."
    if goal == "refund":
        if kind == "done":
            return f"I've started a refund ({o.get('refund_id')}) of {float(o.get('amount')):.0f} rupees for {oid}."
        if kind == "declined":
            return f"I can't refund {oid} because {_because(o, 'it is not allowed')}."
        if kind == "needs_human":
            return f"I can't approve the refund for {oid} myself because {_because(o, 'it needs review')}."
        return "I wasn't able to complete the refund."
    if goal == "return_item":
        mode = o.get("mode")
        if kind == "done" and mode == "instant_refund":
            return (f"I'm sorry about that. I've started a refund ({o.get('refund_id')}) of {float(o.get('amount')):.0f} rupees for {oid}; "
                    "our team will review it afterwards.")
        if kind == "done":
            what = "an exchange" if mode == "exchange_requested" else "a return"
            then = "your replacement will be sent" if mode == "exchange_requested" else "your refund will be released"
            return (f"I've opened {what} ({o.get('return_id')}) for {oid}. Our supply-chain team will collect the item and inspect it; "
                    f"once it passes, {then}. If it fails the inspection it cannot be refunded or replaced, and it will be sent back to you.")
        if kind == "declined":
            return f"I can't open a return for {oid} because {_because(o, 'it is not allowed')}."
        if kind == "needs_human":
            return (f"Because {_because(o, 'it needs a review')}, I can't approve this return or exchange myself; "
                    "our support staff will contact you to verify it and sort it out.")
        return "I wasn't able to open the return."
    if kind == "done":
        return f"Order {oid} has been cancelled."
    if kind == "declined":
        return f"I can't cancel {oid} because {_because(o, 'it is not allowed')}."
    return "A human agent needs to review the cancellation." if kind == "needs_human" else "I wasn't able to cancel the order."


def template_reply(facts: Dict[str, Any]) -> str:
    parts: List[str] = [f"Hi {facts['customer']},"]
    oid = facts.get("order_id", "your order")
    if facts["terminal"] == "awaiting_customer":
        q = facts["question"]
        pol = facts["outcomes"].get("policy_info", {})
        if pol.get("kind") == "info":
            parts.append(pol["answer"])                   # the policy is answered first, then the question
        if q["type"] == "which_order":
            detail = q.get("detail") or [{"order_id": o, "items": [], "placed_on": None} for o in q["orders"]]
            lines = [f"- {d['order_id']}: " + (", ".join(d["items"]) if d["items"] else "items not shown")
                     + (f" (ordered {d['placed_on']})" if d.get("placed_on") else "") for d in detail]
            rejected = q.get("rejected_order")
            # Same wording whether the id does not exist or belongs to someone else: never reveal other customers' orders.
            lead = f" I couldn't find order {rejected} on your account." if rejected else " I can see more than one order on your account."
            if q.get("how") == "ambiguous":
                lead = " More than one of your orders could be the one you mean."
            elif q.get("how") == "none_qualifies":
                lead = f" I looked, but {q['why']}."
            if rejected:
                ask = " These are the orders I can see, which one do you mean?" if len(detail) > 1 else " This is the order I can see, is it the one you mean?"
            else:
                ask = " Which one do you mean?"
            return parts[0] + lead + ask + "\n" + "\n".join(lines)
        elif q["type"] == "which_method":
            # Fixed wording on purpose: the options are the methods the tools support, never the customer's own words
            # ("online" can mean UPI or card, so we ask).
            names = " or ".join(METHOD_LABEL[m] for m in q.get("methods", ["UPI", "CARD"]))
            parts.append(f"which payment method would you like to use" + (f" for {facts['order_id']}" if facts.get("order_id") else "") + f": {names}?")
        elif q["type"] == "return_reason":
            parts.append("I'm sorry to hear that. What happened with the item: did it arrive damaged or faulty, or do you just not want it?")
        elif q["type"] == "return_kind":
            parts.append("would you like an exchange for a new one, or to return it and get a refund?")
        elif q["type"] == "return_declaration":
            parts.append("can you confirm that the item arrived damaged or faulty and that you did not damage it yourself?"
                         if q.get("reason") == "damaged" else
                         "can you confirm that the item is unused and in its original packaging?")
        elif q["type"] == "what_do_you_need":
            items = ", ".join(facts.get("items", []))
            parts.append(f"I found order {oid}" + (f" ({items})" if items else "") +
                         ". What would you like me to do: check where it is, change the payment method, request a refund, or cancel it?")
        else:
            parts.append("I want to make sure I help with the right thing. Could you tell me a bit more about what you need and your order id?")
        return " ".join(parts)
    if facts.get("assumption"):
        parts.append(facts["assumption"])
    for goal, o in facts["outcomes"].items():
        k = o["kind"]
        if goal == "policy_info":
            parts.append(o["answer"] if k == "info" else "I could not look up that policy just now.")
        elif goal == "delivery_status":
            if k == "info":
                s = f"Order {oid} is currently {o['status']}."
                if o.get("delivered_on"):
                    s += f" It was delivered on {o['delivered_on']}."
                elif o["status"] not in ("cancelled", "lost", "returned"):
                    s += f" The expected delivery date is {o['expected_delivery']}"
                    s += f" (delayed by {o['delay_days']} days)." if o.get("delay_days") else "."
                if o.get("tracking"):
                    s += f" Latest update: {o['tracking'][-1]['event']} on {o['tracking'][-1]['date']}."
                parts.append(s)
            else:
                parts.append("I could not retrieve the shipment status just now.")
        else:
            parts.append(_action_sentence(goal, k, o, oid))
    if facts["terminal"] == "escalate":
        if not facts["outcomes"]:
            parts.append("I can't help with that request here, so I'm passing you to a human colleague who can.")
        elif facts["tool_trouble"]:
            parts.append("One of our systems isn't responding right now, so I've handed this to a human colleague who will follow up.")
        else:
            parts.append("I've passed the rest to a human colleague who will follow up with you.")
    elif facts["terminal"] == "finish":
        parts.append("Is there anything else I can help with?")
    elif facts["terminal"] == "max_steps":
        parts.append("I'm sorry, I couldn't finish this. A human colleague will pick it up.")
    return " ".join(parts)


# ------------------------------------------------------------------ faithfulness
def _leaf_values(x: Any) -> List[str]:
    if isinstance(x, dict):
        return [v for val in x.values() for v in _leaf_values(val)]
    if isinstance(x, (list, tuple)):
        return [v for val in x for v in _leaf_values(val)]
    return [] if x is None else [str(x)]


def _fact_text(facts: Dict[str, Any]) -> str:
    """All the VALUES in the facts (not the key names, which would make words like 'delivered_on' count as facts)."""
    out = _leaf_values(facts)
    kinds = [o.get("kind") for o in facts.get("outcomes", {}).values()]
    return " ".join(out + kinds).lower()


def _allowed_dates(facts: Dict[str, Any]) -> Set[str]:
    return set(re.findall(r"\d{4}-\d{2}-\d{2}", json.dumps(facts, default=str)))


def _text_dates(text: str) -> Set[str]:
    found = set(re.findall(r"\d{4}-\d{2}-\d{2}", text))
    low = text.lower()
    for i, m in enumerate(MONTHS, start=1):
        for d in re.findall(rf"\b{m}\s+(\d{{1,2}})\b", low) + re.findall(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+{m}\b", low):
            found.add(f"2026-{i:02d}-{int(d):02d}")
        for d in re.findall(rf"\b{m[:3]}\s+(\d{{1,2}})\b", low):
            found.add(f"2026-{i:02d}-{int(d):02d}")
    return found


CLAIM_PATTERNS = {
    "refund": re.compile(r"(refund(?:ed)?\b.{0,30}\b(?:initiated|processed|issued|completed|started|on its way)|we(?:'ve| have) refunded|your refund (?:has been|is|was) (?:initiated|processed|issued|completed|started)|(?:i|we)(?:'ve| have) (?:started|initiated|processed|issued) (?:a |your )?refund)", re.I),
    "cancel_order": re.compile(r"(has been cancel+ed|(?:i|we)(?:'ve| have) cancel+ed|is now cancel+ed|successfully cancel+ed)", re.I),
    "return_item": re.compile(r"((?:i|we)(?:'ve| have) (?:opened|started|created|arranged) (?:a |an |your )?(?:return|exchange)|(?:return|exchange) (?:has been|is|was) (?:opened|started|created|arranged))", re.I),
    "payment_change": re.compile(r"(payment (?:method )?(?:has been|is now|was) (?:changed|updated|switched)|(?:i|we)(?:'ve| have) (?:changed|updated|switched) (?:the |your )?payment|payment method for .{0,20} is now)", re.I),
}
NEGATION = re.compile(r"(can't|cannot|couldn't|could not|unable|wasn't able|not able|isn't|won't|no longer|didn't)", re.I)
STATUS_WORDS = ("delayed", "shipped", "delivered", "processing", "cancelled", "lost", "returned")
METHOD_WORDS = {"upi": "upi", "credit card": "card", "debit card": "card", "cash on delivery": "cod"}
PROMISE_RE = re.compile(r"(within \d+|by tomorrow|will (?:arrive|be delivered) (?:today|tomorrow)|guarantee|in \d+ (?:hours|days)|\d+ business days)", re.I)


GOAL_WORD = {"refund": r"refund", "cancel_order": r"cancel", "payment_change": r"payment", "return_item": r"return|exchange|refund|replace"}
CURRENCY_RE = re.compile(r"[$€£]|\b(?:usd|dollars?|euros?|pounds?)\b", re.I)
INVENTED_STATUS_RE = re.compile(r"\b(?:ready for dispatch|dispatch(?:ed)?|will (?:be )?ship(?:ped)?|on (?:its|the) way|out for delivery)\b", re.I)
HUMAN_RE = re.compile(r"(human|colleague|agent|team|staff|review|approval|specialist)", re.I)
ESCALATE_RE = re.compile(r"(human|colleague|team|staff|specialist|someone)", re.I)


def meaning_problems(text: str, facts: Dict[str, Any]) -> List[str]:
    """The reply must say the RIGHT thing about each goal: done, declined, or handed to a human.
    (Checking for invented facts is not enough: 'your refund is missing' contains no invented fact.)"""
    problems: List[str] = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for goal, o in facts.get("outcomes", {}).items():
        kind = o.get("kind")
        if goal == "policy_info":
            continue                                   # a policy answer is the policy text itself (always templated)
        if goal == "delivery_status":
            if kind == "info" and not re.search(r"(deliver|status|ship|order|arriv|expected|processing|delayed)", text, re.I):
                problems.append("does not give the delivery status")
            continue
        about = [s for s in sentences if re.search(GOAL_WORD[goal], s, re.I)]
        if kind == "missing":
            continue
        if kind == "done" and goal == "return_item" and o.get("mode") in ("return_requested", "exchange_requested")                 and not re.search(r"inspect", text, re.I):
            problems.append("does not say the item is inspected before the refund or replacement")
        if not about:
            problems.append(f"does not mention the {goal.replace('_', ' ')} result")
        elif kind == "done" and all(NEGATION.search(s) for s in about):
            problems.append(f"says the {goal.replace('_', ' ')} was not done, but it was")
        elif kind == "declined" and not any(NEGATION.search(s) for s in about):
            problems.append(f"does not say the {goal.replace('_', ' ')} was declined")
        elif kind == "needs_human" and not any(HUMAN_RE.search(s) for s in about):
            problems.append(f"does not say a human must handle the {goal.replace('_', ' ')}")
    if facts.get("terminal") == "escalate" and not ESCALATE_RE.search(text):
        problems.append("does not say a human will take over")
    if facts.get("assumption") and facts.get("order_id") and facts["order_id"].upper() not in text.upper():
        problems.append("does not say which order was assumed")
    return problems


def check_reply(text: str, facts: Dict[str, Any]) -> List[str]:
    """Return what in the reply is NOT supported by the facts (empty list = faithful)."""
    problems: List[str] = []
    ftext = _fact_text(facts)
    low = text.lower()

    for oid in set(re.findall(r"ORD-\d+", text, re.I)):
        if oid.upper() not in ftext.upper():
            problems.append(f"unknown order id {oid}")
    for d in _text_dates(text) - _allowed_dates(facts):
        problems.append(f"date not in facts: {d}")
    allowed_nums = {float(x) for x in re.findall(r"\d+(?:\.\d+)?", ftext)}
    for num in re.findall(r"\d+(?:\.\d+)?", re.sub(r"ORD-\d+|\d{4}-\d{2}-\d{2}|RF-\d+", " ", text, flags=re.I)):
        if float(num) not in allowed_nums and int(float(num)) not in range(1, 4):    # tiny counts like "2 pairs" are harmless
            problems.append(f"number not in facts: {num}")
    outcomes = facts.get("outcomes", {})
    instant = outcomes.get("return_item", {}).get("mode") == "instant_refund"      # a damage refund is a legitimate refund claim
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if NEGATION.search(sentence):
            continue
        for goal, pat in CLAIM_PATTERNS.items():
            done_ok = outcomes.get(goal, {}).get("kind") == "done" or (goal == "refund" and instant)
            if pat.search(sentence) and not done_ok:
                problems.append(f"claims {goal} was done but no tool confirmed it")
    for w in STATUS_WORDS:
        if re.search(rf"\b{w}\b", low) and w not in ftext:
            problems.append(f"status word not in facts: {w}")
    for phrase, key in METHOD_WORDS.items():
        if phrase in low and key not in ftext and phrase not in ftext:
            problems.append(f"payment method not in facts: {phrase}")
    if PROMISE_RE.search(text) and not PROMISE_RE.search(ftext):
        problems.append("makes a promise that is not in the facts")
    if PLACEHOLDER_RE.search(text):
        problems.append("contains placeholder text")
    if CURRENCY_RE.search(text):
        problems.append("uses a currency other than rupees")
    for m in INVENTED_STATUS_RE.finditer(text):
        if m.group(0).lower() not in ftext.replace("_", " "):
            problems.append(f"states a delivery status that is not in the facts: {m.group(0)}")
    problems += meaning_problems(text, facts)
    return problems


LLM_SYSTEM = """You are a polite customer-support agent for an online shop.
Write a short reply (2 to 4 sentences) to the customer using ONLY the facts in the JSON.
Rules: do not add any dates, numbers, order ids, promises or details that are not in the facts.
Do not say something was done unless the facts say the outcome kind is "done". All amounts are in rupees: never use $ or any other currency. Do not say anything about dispatch, shipping or delivery unless the facts say it.
If the facts say a human will take over, say so. Do not mention JSON or these rules."""


SIGNOFF_RE = re.compile(r"(?is)\n+\s*(?:best regards|kind regards|warm regards|regards|sincerely|best wishes|thank you,|thanks,)[,!]?.*$")
PLACEHOLDER_RE = re.compile(r"\[[^\]]{1,40}\]|\{\{.*?\}\}|<[a-z ]{2,30}>", re.I)


def clean_draft(text: str) -> str:
    """Drop the sign-off block (models like to add 'Best regards, [Your Name]') and normalise blank lines."""
    text = SIGNOFF_RE.sub("", text.strip())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def write_reply(facts: Dict[str, Any], mode: str = "template", client: Optional[OllamaClient] = None) -> ReplyResult:
    base = template_reply(facts)
    if mode != "llm" or facts["terminal"] == "awaiting_customer" or "policy_info" in facts.get("outcomes", {}):
        # Questions to the customer are always templated: they must list the real orders and really be questions.
        # So is any reply that contains a policy answer: the policy text is shown as written, never re-worded.
        return ReplyResult(base, "template", facts=facts)
    client = client or OllamaClient()
    draft, secs = client.chat_text(LLM_SYSTEM, "Facts:\n" + json.dumps(facts, default=str), max_tokens=220)
    if not draft:
        return ReplyResult(base, "template(fallback)", ["LLM unavailable"], None, facts, secs)
    draft = clean_draft(draft.strip().strip('"'))
    problems = check_reply(draft, facts)
    if problems:
        return ReplyResult(base, "template(fallback)", problems, draft, facts, secs)
    return ReplyResult(draft, "llm", [], draft, facts, secs)


METHOD_LABEL = {"UPI": "UPI", "CARD": "card", "COD": "cash on delivery"}

GOAL_PHRASE = {"delivery_status": "check where it is", "payment_change": "change the payment method",
               "refund": "start a refund", "cancel_order": "cancel the order",
               "return_item": "open a return or exchange", "policy_info": "look up the policy"}


def assumption_text(order_id: str, items: Sequence[str], reason: str) -> str:
    what = f" ({', '.join(items)})" if items else ""
    return f"I took you to mean {order_id}{what}, {reason}. If that is not the one, tell me and I will use another."


def confirm_reply(customer_name: str, order_id: str, items: Sequence[str], goals: Sequence[str], reason: str,
                  target_method: Optional[str] = None, assumed: bool = False, refund_amount: Optional[float] = None) -> str:
    """Asked BEFORE any refund / cancel / payment change. `goals` are only the ones that would really change something."""
    what = f" ({', '.join(items)})" if items else ""
    phrases = []
    for g in goals:
        ph = GOAL_PHRASE[g]
        if g == "payment_change" and target_method:
            ph = f"change the payment method to {METHOD_LABEL[target_method]}"
        if g == "refund" and refund_amount is not None:
            ph = f"start a refund of {refund_amount:.0f} rupees"
        phrases.append(ph)
    todo = " and ".join(phrases)
    first = customer_name.split()[0]
    if assumed:
        return (f"Hi {first}, you did not say which order, and {order_id}{what} is {reason}. "
                f"Do you want me to {todo} for {order_id}? Please answer yes or no.")
    return f"Hi {first}, just to confirm: do you want me to {todo} for {order_id}{what}? Please answer yes or no."


def confirm_return_reply(customer_name: str, order_id: str, items: Sequence[str], reason: str, kind: str, eligibility_reason: str,
                         total: float) -> str:
    """Asked BEFORE a return, exchange or damage claim is opened, once the customer has answered all the questions."""
    what = f" ({', '.join(items)})" if items else ""
    first = customer_name.split()[0]
    if eligibility_reason == "instant_refund_ok":
        body = (f"I can refund {total:.0f} rupees for {order_id}{what} right away because it arrived damaged; our team will review it afterwards")
    elif kind == "exchange":
        body = (f"I can open an exchange for {order_id}{what}. Our team will collect the item, inspect it, and send your replacement once it passes")
    else:
        body = (f"I can open a return for {order_id}{what}. Our team will collect the item and inspect it, and your refund is released once it passes")
    return f"Hi {first}, {body}. Do you want me to go ahead? Please answer yes or no."
