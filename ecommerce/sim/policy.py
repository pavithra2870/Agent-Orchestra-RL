"""The shop's written policies (the content of the knowledge-base tool `lookup_policy`).

Answers are generated from the SAME numbers the rules enforce (config/ecommerce.yaml), so a policy answer can never
contradict what the tools do. Nothing here needs a model: the topic is chosen elsewhere (the language layer, by
comparing the question with the example questions below) and handed to the tool; the agent decides WHETHER and WHEN
to look a policy up, never what the policy says.

Three disjoint question sets, like the router's phrase sets:
  Topic.examples     phrase set A: used to match questions to topics
  TUNING_QUESTIONS   phrase set C: used only to tune the matching threshold
  TEST_QUESTIONS     phrase set B: only ever used for the final held-out test
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from ecommerce.config import Settings, get_settings


@dataclass(frozen=True)
class Topic:
    key: str
    title: str
    examples: Tuple[str, ...]          # phrase set A
    source: str                        # which rule the answer comes from


TOPIC_KEYS = ("lost_order", "refund_policy", "cancellation_policy", "payment_change_policy", "delayed_order",
              "returns_exchange", "damaged_item")

TUNING_QUESTIONS: Dict[str, Tuple[str, ...]] = {
    "lost_order": ("my parcel never came, will i get compensated?", "what do you do when a package disappears?"),
    "refund_policy": ("under what conditions do you give money back?", "when am i entitled to a refund?"),
    "cancellation_policy": ("until when can i call off an order?", "is cancelling possible once it is dispatched?"),
    "payment_change_policy": ("under what circumstances can i swap how i pay?", "is it allowed to move from cash to online payment?"),
    "delayed_order": ("my shipment is running behind, what are my options?", "what happens when delivery takes longer than promised?"),
    "returns_exchange": ("can i send something back if i just don't want it?", "what are the rules for giving an item back?"),
    "damaged_item": ("if something shows up broken, what are my options?", "what do you do when an order arrives with a defect?"),
}
TEST_QUESTIONS: Dict[str, Tuple[str, ...]] = {
    # a fresh set (the first version was used up while developing the new classes)
    "lost_order": ("what is the policy if my package gets lost?", "if a parcel disappears on the way, what do you do?", "do i get paid back for a lost shipment?"),
    "refund_policy": ("explain how your refunds work", "under what rules do you refund orders?", "am i entitled to my money back, in general?"),
    "cancellation_policy": ("how does cancelling an order work?", "is there a time limit for cancelling?", "what are the conditions to cancel an order?"),
    "payment_change_policy": ("can the way i pay be changed later?", "what rules apply to switching to upi?", "how do i change from cash to card payment?"),
    "delayed_order": ("what is the rule for late deliveries?", "what compensation do i get for a delayed order?", "my parcel could be late, what's your policy?"),
    "returns_exchange": ("what is your policy on sending items back?", "how long do i have to return a product?", "is it possible to return something i bought?"),
    "damaged_item": ("what if what i bought arrives broken?", "how do damaged items get handled?", "will you replace an item that comes defective?"),
}

# more example questions per topic (phrase set A), added to each Topic's own examples
EXTRA_EXAMPLES: Dict[str, Tuple[str, ...]] = {
    "lost_order": ("what happens when a parcel is lost?", "what do you do about packages that never arrive?", "will i get my money if the order is lost?"),
    "refund_policy": ("when do you give refunds?", "what must be true for me to get a refund?", "tell me about your refund rules"),
    "cancellation_policy": ("what is the rule for cancelling an order?", "up to what point can an order be cancelled?", "is there a deadline for cancelling an order?"),
    "payment_change_policy": ("what are the rules about changing how i pay?", "am i allowed to switch my payment after ordering?", "when can the payment method of an order be changed?"),
    "delayed_order": ("what is your policy when an order arrives late?", "what can i do if my package is running late?", "do i get anything if delivery is delayed?"),
    "returns_exchange": ("what is the process to return an item?", "do you accept returns?", "can i get a different item in exchange?"),
    "damaged_item": ("what do i do if my order arrives broken?", "how do you handle faulty items?", "can i get a replacement for a damaged product?"),
}


def build_topics(settings: Optional[Settings] = None) -> Dict[str, Tuple[Topic, str]]:
    """topic key -> (Topic, answer text). Numbers come from the configuration."""
    w = (settings or get_settings()).world
    hv, late, rw = int(w.high_value_threshold), w.delay_refund_days, w.return_window_days
    hours, cap, lim, per = w.damage_report_window_days * 24, int(w.instant_refund_cap), w.damage_claims_limit, w.claims_period_days
    topics = {
        "lost_order": (Topic("lost_order", "Lost orders", (
            "what happens if my order is lost?", "my order is lost, what now?", "what if the package never arrives and is lost?",
            "do you refund lost orders?", "what is the policy for lost parcels?"), "refund rules: status lost"),
            f"If an order is lost, you are eligible for a refund of what you paid. Refunds above {hv} rupees need approval from a human colleague. "
            "Orders paid by cash on delivery that never arrived have nothing to refund, because nothing was paid. "
            "If you tell me which order it is, I can check it for you."),
        "refund_policy": (Topic("refund_policy", "Refund policy", (
            "what is your refund policy?", "when can i get a refund?", "how do refunds work?",
            "am i eligible for a refund?", "what are the conditions for a refund?"), "refund rules"),
            f"A paid order can be refunded if it is lost, if it was returned to us, or if it is delayed by at least {late} days. "
            f"A delivered item is refunded after you return it and it passes our inspection (see the return policy). "
            f"Refunds above {hv} rupees need approval from a human colleague. Nothing is refunded on cash-on-delivery orders that have not been paid. "
            "Tell me the order and I can check whether it qualifies."),
        "cancellation_policy": (Topic("cancellation_policy", "Cancellation policy", (
            "can i cancel my order after it ships?", "until when can i cancel an order?", "what is your cancellation policy?",
            "is cancelling allowed once it is shipped?", "how long do i have to cancel?"), "cancel rules"),
            "An order can be cancelled only while it is still being processed. Once it has shipped it can no longer be cancelled here. "
            "If it is already paid, a cancelled order is refunded."),
        "payment_change_policy": (Topic("payment_change_policy", "Changing the payment method", (
            "can i change my payment method?", "how do i switch from cash on delivery to upi?", "what is the rule for changing payment?",
            "can i pay online instead of cash on delivery?", "is changing the payment mode allowed?"), "payment change rules"),
            "You can change the payment method (for example from cash on delivery to UPI or card) only on orders that are not paid yet, "
            "are still open, and are not already out for delivery. Tell me the order and the new method and I will check it."),
        "delayed_order": (Topic("delayed_order", "Delayed orders", (
            "what happens if my order is delayed?", "my order is late, what can i do?", "what if delivery takes too long?",
            "what is the policy for delayed delivery?", "how long before a late order gets a refund?"), "refund rules: status delayed"),
            f"If your order is delayed, I can check the latest tracking update. If it is already paid and delayed by {late} days or more, you can ask for a refund. "
            "Tell me the order and I will look it up."),
        "returns_exchange": (Topic("returns_exchange", "Returns and exchanges", (
            "can i return an item?", "how do returns work?", "what is your return policy?",
            "how many days do i have to return something?", "can i exchange my product for another?"), "return rules"),
            f"You can return an item within {rw} days of delivery, at no charge, if it is unused and in its original packaging; I will ask you to confirm that. "
            "Our team collects the item and inspects it. If it passes, you get your refund (or a replacement, if you prefer an exchange). "
            f"If it fails the inspection it is not refunded and is sent back to you. Returns on orders above {hv} rupees need approval from a human colleague."),
        "damaged_item": (Topic("damaged_item", "Damaged or faulty items", (
            "what if my item arrives damaged?", "my product is broken, what is the policy?", "what is your policy on damaged goods?",
            "how do i report a defective item?", "what happens if i receive a faulty product?"), "damage claim rules"),
            f"I am sorry if something arrived damaged. Please report it within {hours} hours of delivery. You can choose an exchange for a new item, or to return it and get a refund, "
            f"and I will ask you to confirm that it arrived damaged and that you did not damage it. For orders up to {cap} rupees a refund starts right away and is checked afterwards. "
            f"Larger orders, late reports, and customers with {lim} or more damage claims in the last {per} days are reviewed first by our support staff, who will contact you."),
    }
    return {k: (Topic(t.key, t.title, t.examples + EXTRA_EXAMPLES[k], t.source), text) for k, (t, text) in topics.items()}
