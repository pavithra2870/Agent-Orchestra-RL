"""Customer message generator.

Three disjoint phrase sets are kept on purpose:
  set "A": used for training and for building the router's example sentences
  set "C": different wording, used only to TUNE and CALIBRATE the router (validation)
  set "B": different wording, only ever used as the final held-out paraphrase test
so "it handles new phrasings" is measured on wording the system has never seen or been tuned on.

Placeholders: {o} = order reference, {m} = target payment method.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

PHRASES: Dict[str, Dict[str, List[str]]] = {
    "A": {
        "delivery_status": [
            "where is {o}?", "can you check the status of {o}?", "{o} is delayed, when will it arrive?",
            "i want to track {o}", "what is the delivery date for {o}?", "has {o} shipped yet?",
            "my package hasn't arrived, where is {o}", "please tell me the current status of {o}",
            "how long will {o} take to reach me?", "{o} still has not been delivered",
        ],
        "payment_change": [
            "change the payment method of {o} to {m}", "i want to pay by {m} instead of cash on delivery for {o}",
            "can i switch the payment for {o} from cod to {m}?", "please update payment on {o} to {m}",
            "use {m} for {o} instead", "i'd like to change how i pay for {o}, make it {m}",
            "switch {o} to {m} payment", "is it possible to pay {o} with {m} rather than cash?",
            "modify the payment option on {o} to {m}", "move the payment of {o} from cash on delivery to {m}",
        ],
        "refund": [
            "i want a refund for {o}", "please refund {o}", "i need my money back for {o}",
            "can i get a refund on {o}?", "refund {o} to my account",
            "i would like my money back for {o}", "{o} never arrived, refund me",
            "i want to claim a refund for {o}", "please process a refund for {o}",
            "give me a refund, {o} is not what i ordered",
        ],
        "cancel_order": [
            "cancel {o}", "i want to cancel {o}", "please cancel {o} before it ships", "can you cancel {o} for me?",
            "i don't want {o} anymore, cancel it", "stop {o}, i want to cancel", "please call off {o}",
            "need to cancel {o} today", "cancel the purchase {o}", "i changed my mind, cancel {o}",
        ],
        "out_of_scope": [
            "can you recommend a good smartphone?", "what's the weather like today?", "tell me a joke",
            "how do i become a seller on your platform?", "i want to change my delivery address and phone number",
            "my account was hacked, please help", "do you have this product in blue?", "what are your store timings?",
        ],
    },
    "B": {
        # Group B rewrite: the first version of this set was used up while developing the new classes (its errors were
        # read to improve the examples), so a fresh, never-inspected set replaces it as the held-out test.
        "delivery_status": [
            "could you tell me when {o} will reach me", "i have not received {o} yet, any news?", "track my parcel {o} please",
            "is {o} still on the way?", "what is the current location of {o}?", "my delivery {o} is overdue, can you look into it",
            "give me the shipping status for {o}", "how far along is {o}?", "has {o} been dispatched?",
            "i keep waiting for {o}, where is it now?",
        ],
        "payment_change": [
            "i'd like to pay online for {o} with {m} rather than cash", "set the payment on {o} to {m}",
            "switch the way {o} is paid from cod to {m}", "can i settle {o} using {m} instead?",
            "please convert {o} to a {m} payment", "{o} should be prepaid by {m}", "change my payment option for {o} to {m}",
            "i want to use {m} for {o}, not cash", "make the payment for {o} {m} please", "i'd like {o} to be charged to my {m}",
        ],
        "refund": [
            "i'd like to get my money back for {o}", "please issue a refund on {o}", "kindly refund the amount of {o}",
            "i'm asking for a refund of {o}", "give me back the money for {o}", "i want {o} refunded",
            "{o} never reached me, return my money", "i need to recover the money i paid for {o}",
            "the order {o} is a no-show, please refund it", "send me a refund for {o}",
        ],
        "cancel_order": [
            "i want {o} to be cancelled", "cancel {o} immediately", "please withdraw my order {o}",
            "i don't need {o} anymore, stop it", "can you stop the shipment of {o}?", "terminate my order {o}",
            "go ahead and cancel {o}", "i wish to cancel {o}", "scrap {o} please", "{o} must not be sent, cancel it",
        ],
        "out_of_scope": [
            "what time do you open?", "can you tell me a fun fact?", "i need help setting up my account",
            "do you sell gift cards?", "how is the weather today?", "tell me a joke about cats",
            "can i change my username?", "which brands do you stock?",
        ],
    },
    "C": {
        "delivery_status": [
            "what's happening with {o}? it hasn't turned up", "can i get a tracking update for {o}?",
            "where has {o} got to?", "i ordered {o} days ago and still nothing, any news?",
            "please let me know the whereabouts of {o}", "is there an expected arrival date for {o}?",
            "how far along is the shipping on {o}?", "{o} was meant to be here already, what's the situation?",
        ],
        "payment_change": [
            "can you make {o} a {m} payment instead of cash?", "i'd prefer to pay with {m} for {o} now",
            "alter the payment mode for {o} to {m}", "please set {o} to be paid by {m}",
            "i don't want to hand over cash for {o}, {m} instead", "could the billing for {o} switch over to {m}?",
            "swap cash on delivery for {m} on {o}", "{o} should be paid through {m} from now on",
        ],
        "refund": [
            "i'd like the money for {o} returned", "can you give me my payment back for {o}?",
            "{o} isn't acceptable, i want a full refund", "please reverse my payment on {o}",
            "i'm requesting repayment for {o}", "return the funds i spent on {o}",
            "make a refund of the amount i paid for {o}", "i've been charged for {o} and want that money returned",
        ],
        "cancel_order": [
            "please stop {o} from being sent", "i've decided against {o}, cancel the order",
            "get rid of {o}, i don't want it", "can {o} be cancelled?", "i'd like to retract {o}",
            "cancel my purchase of {o} please", "do not ship {o}, i want it cancelled",
            "i made an error ordering {o}, remove it",
        ],
        "out_of_scope": [
            "do you sell laptops?", "how do i reset my account password?",
            "which payment methods do you accept for new orders?", "can i talk to your manager?",
            "what's the best way to contact you by phone?", "i'd like to subscribe to your newsletter",
        ],
    },
}

# ---- Group B: more example sentences for the router (sets A and C), written independently of the held-out set B
EXTRA_A: Dict[str, List[str]] = {
    "delivery_status": [
        "when should i expect {o} to arrive?", "can you give me an update on {o}?", "{o} is taking too long, what is going on?",
        "has {o} left the warehouse yet?", "i'm still waiting for {o}", "tell me where {o} is right now",
    ],
    "payment_change": [
        "i'd rather not pay cash for {o}, can i use {m}?", "can the payment for {o} be {m} instead?",
        "i want {o} to be paid via {m}", "update the payment mode of {o} to {m} please",
        "is it fine to pay {o} online using {m}?", "change {o} from cash on delivery to {m}",
    ],
    "refund": [
        "pay me back for {o}", "i want the amount for {o} credited back to me", "i should get my money back for {o}",
        "please give me back what i paid for {o}", "{o} didn't show up, i want the money back", "credit my account for {o}",
    ],
    "cancel_order": [
        "i want to stop {o}", "cancel the order {o} for me", "please don't deliver {o}, cancel it",
        "i made a mistake with {o}, cancel it please", "can you remove {o} from processing?", "kindly cancel {o}",
    ],
    "out_of_scope": [
        "do you ship internationally?", "how do i change my account email?", "can you help me pick a gift?",
        "what is the price of the red backpack?", "i'd like to know about your loyalty programme", "can you recommend a good laptop?",
    ],
}
EXTRA_RETURN_A: Dict[str, List[str]] = {
    "damaged": ["the item i received in {o} is defective", "i got {o} but the item inside is faulty",
                "the product delivered in {o} stopped functioning", "unfortunately {o} came with a broken part"],
    "change_of_mind": ["i'd like to send back the item in {o}", "the item in {o} is not suitable, i want to return it",
                       "i want to give up on {o}, how do i send it back", "can i return {o}? it is too small"],
}
EXTRA_C: Dict[str, List[str]] = {
    "delivery_status": ["any progress on {o}?", "i'd like to see the tracking for {o}", "how soon will {o} get to me?"],
    "payment_change": ["could {o} be billed to my {m} instead of cash?", "i'd like to prepay {o} with {m}", "move {o} over to {m} payment"],
    "refund": ["i expect my money for {o} back", "please return what i paid on {o}", "i want to be paid back for {o}"],
    "cancel_order": ["could you cancel {o} for me?", "i don't want {o} sent anymore", "please annul {o}"],
    "out_of_scope": ["how do i delete my account?", "who is your ceo?", "what is your return address for suppliers?"],
}

# ---- Group B: returns / damage claims and policy questions ---------------------------------------------------
# reason -> sentences (the reason is what the customer says; the router example sets use both reasons)
RETURN_PHRASES: Dict[str, Dict[str, List[str]]] = {
    "A": {
        "damaged": [
            "{o} arrived damaged", "the product from {o} is broken", "my item in {o} is defective",
            "the item in {o} stopped working after a day", "{o} came with a cracked screen", "the thing i got in {o} is faulty",
            "i received {o} and the item is torn", "the parcel {o} was crushed and the item is damaged",
            "{o} arrived with a dent in the product", "the item from {o} does not work properly, it is defective",
        ],
        "change_of_mind": [
            "i want to return {o}", "i don't like the product in {o}, i want to send it back", "please arrange a return for {o}",
            "i would like to return {o}, it is not what i expected", "can i send {o} back? i changed my mind",
            "the item in {o} is the wrong size for me, i want to return it", "i don't want the item from {o} anymore, how do i return it",
            "return {o} please",
        ],
    },
    "C": {
        "damaged": [
            "the item in {o} came broken", "my delivery {o} has a defect", "what i received in {o} isn't working",
            "the {o} package arrived with the product smashed", "something is wrong with the item from {o}, it is faulty",
            "the item from {o} was dead on arrival", "{o} contains a damaged product", "the stuff in {o} arrived all scratched up",
        ],
        "change_of_mind": [
            "the item in {o} isn't for me, i'd like to give it back", "i'd rather return the item from {o}",
            "i want to send back {o}, i no longer want it", "{o} didn't meet my expectations, i'd like to return it",
            "please start a return on {o}", "i've decided to return {o}", "i bought {o} by mistake and want to return it",
            "the fit of the item in {o} is wrong, can i return it",
        ],
    },
    "B": {
        "damaged": [
            "{o} showed up with a broken item", "the item in {o} has stopped working", "{o} is defective",
            "the product in {o} is damaged", "something is broken in {o}", "{o} came damaged", "my item in {o} is faulty and unusable",
        ],
        "change_of_mind": [
            "i'd like to return the item in {o}", "i changed my mind about {o}, can i send it back?", "please let me return {o}",
            "i don't need the item from {o}, i want to return it", "the item in {o} doesn't fit, i'd like to return it",
            "{o} is not as i imagined, i want to return it",
        ],
    },
}
# what the customer may add about what they want done (exchange or refund); often left out, then the agent asks
KIND_PHRASES: Dict[str, Dict[str, List[str]]] = {
    "A": {"exchange": ["i want an exchange", "please send me a new one", "i'd like a replacement"],
          "refund": ["i want my money back", "please refund me", "i want a refund"]},
    "C": {"exchange": ["exchange it for a new one please", "i want it replaced"],
          "refund": ["refund me instead", "just give me my money back"]},
    "B": {"exchange": ["i want a new one", "please exchange it", "send me a replacement"],
          "refund": ["i want my money back", "refund me please", "i'd like a refund for it"]},
}


def _policy_bank(phrasing_set: str) -> Dict[str, List[str]]:
    from ecommerce.sim import policy

    if phrasing_set == "A":
        return {k: list(tp.examples) for k, (tp, _) in policy.build_topics().items()}
    return {k: list(v) for k, v in (policy.TUNING_QUESTIONS if phrasing_set == "C" else policy.TEST_QUESTIONS).items()}


def _fill_group_b() -> None:
    for reason, lst in EXTRA_RETURN_A.items():
        RETURN_PHRASES["A"][reason] += lst
    for goal, lst in EXTRA_A.items():
        PHRASES["A"][goal] += lst
    for goal, lst in EXTRA_C.items():
        PHRASES["C"][goal] += lst
    for ph in ("A", "B", "C"):
        PHRASES[ph]["return_item"] = [p for lst in RETURN_PHRASES[ph].values() for p in lst]
        PHRASES[ph]["policy_info"] = [q for lst in _policy_bank(ph).values() for q in lst]
    # the "damaged and wants money back" mixes are real customer wording and belong to the return class
    PHRASES["A"]["return_item"] += [f"{d.format(o='my order')}, {k}" for d, k in zip(RETURN_PHRASES["A"]["damaged"][:3], KIND_PHRASES["A"]["refund"])]
    PHRASES["C"]["return_item"] += [f"{d.format(o='my order')}, {k}" for d, k in zip(RETURN_PHRASES["C"]["damaged"][:2], KIND_PHRASES["C"]["refund"])]


CONNECTORS = {
    "A": [" and also ", ", also ", " and ", ". Also, ", ". And "],
    "B": [" plus ", " and on top of that, ", ". Additionally, ", " while you're at it, "],
    "C": [", and also ", " as well as ", ". By the way, ", " and then "],
}
GREETINGS = {
    "A": ["hi, ", "hello, ", "", "", "hey team, "],
    "B": ["good evening, ", "hi there, ", "", "excuse me, "],
    "C": ["hello there, ", "hi, ", "", "dear support, "],
}
METHOD_WORDS = {"UPI": ["UPI", "upi"], "CARD": ["card", "my credit card", "debit card"], "COD": ["cash on delivery"]}
_fill_group_b()


def add_typos(text: str, rng: np.random.Generator, prob: float) -> str:
    """Swap two neighbouring letters in a few words, like real chat messages."""
    if rng.random() >= prob:
        return text
    words = text.split(" ")
    candidates = [i for i, w in enumerate(words) if len(w) > 4 and w.isalpha()]
    for i in rng.choice(candidates, size=min(2, len(candidates)), replace=False) if candidates else []:
        w = words[int(i)]
        j = int(rng.integers(1, len(w) - 2))
        words[int(i)] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


def render_message(
    goals,
    target_method: str,
    order_ref: str,
    phrasing_set: str,
    rng: np.random.Generator,
    typo_prob: float = 0.0,
    return_reason: str = "damaged",
    return_kind_in_text: str = "",
    policy_topic: str = "",
) -> str:
    bank = PHRASES[phrasing_set]
    method_word = str(rng.choice(METHOD_WORDS.get(target_method, [target_method])))
    parts = []
    for g in goals:
        if g == "return_item":
            phrase = str(rng.choice(RETURN_PHRASES[phrasing_set][return_reason])).format(o=order_ref)
            if return_kind_in_text:
                phrase += ", " + str(rng.choice(KIND_PHRASES[phrasing_set][return_kind_in_text]))
            parts.append(phrase)
            continue
        if g == "policy_info":
            parts.append(str(rng.choice(_policy_bank(phrasing_set)[policy_topic])))
            continue
        phrase = str(rng.choice(bank[g]))
        parts.append(phrase.format(o=order_ref, m=method_word))
    if not parts:
        return str(rng.choice(bank["out_of_scope"]))
    text = parts[0]
    for p in parts[1:]:
        text += str(rng.choice(CONNECTORS[phrasing_set])) + p
    text = str(rng.choice(GREETINGS[phrasing_set])) + text
    return add_typos(text, rng, typo_prob)


def example_sentences(phrasing_set: str = "A") -> Dict[str, List[str]]:
    """Example sentences per goal (order placeholders filled with plain words) for the router."""
    out: Dict[str, List[str]] = {}
    for g, phrases in PHRASES[phrasing_set].items():
        out[g] = [p.format(o="my order", m="UPI") for p in phrases]
    return out
