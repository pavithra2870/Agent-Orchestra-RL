"""Language layer: clause splitting, router sanity, slot extraction, reply faithfulness."""

import pytest

from ecommerce.agents.baselines import OracleAgent, RuleBasedAgent
from ecommerce.env.env import EcomSupportEnv
from ecommerce.evaluation.runner import run_episode
from ecommerce.evaluation.splits import get_splits
from ecommerce.language.parser import MessageParser, extract_method, extract_order_id
from ecommerce.language.reply import build_facts, check_reply, clean_draft, template_reply, write_reply
from ecommerce.language.router import IntentRouter, split_clauses
from ecommerce.sim.scenarios import generate_scenarios


@pytest.fixture(scope="module")
def router():
    return IntentRouter()


def test_split_clauses_handles_greetings_and_connectors():
    assert split_clauses("hi, where is my order and also cancel it") == ["where is my order", "cancel it"]
    assert len(split_clauses("hello there, check order ORD-1234. Additionally, refund it")) == 2
    assert split_clauses("tell me a joke") == ["tell me a joke"]


def test_slot_extraction():
    assert extract_order_id("please cancel ord-4821 today") == "ORD-4821"
    assert extract_order_id("cancel my order") is None
    assert extract_method("switch it to upi please") == "UPI"
    assert extract_method("pay with my credit card instead") == "CARD"
    assert extract_method("where is it") is None


@pytest.mark.parametrize("text,goals", [
    ("where is my order ORD-1234?", {"delivery_status"}),
    ("i want a refund for order ORD-1234", {"refund"}),
    ("cancel order ORD-1234 please", {"cancel_order"}),
    ("check the status of ORD-1234 and also switch the payment to UPI", {"delivery_status", "payment_change"}),
])
def test_router_finds_obvious_goals(router, text, goals):
    assert set(router.route(text).goals) == goals


def test_router_flags_unrelated_messages_and_is_deterministic(router):
    out = router.route("tell me a joke")
    assert out.goals == () and not out.in_scope
    assert router.route("where is my order").parse_conf == router.route("where is my order").parse_conf


def test_router_beats_chance_on_unseen_wording(router):
    sc = generate_scenarios(get_splits()["paraphrase"], 120)
    acc = sum(set(router.route(s.text).goals) == set(s.true_goals) for s in sc) / len(sc)
    assert acc > 0.75


def test_rules_parser_returns_a_complete_request(router):
    p = MessageParser(router).parse("hello, switch order ORD-2222 to UPI and where is it?")
    assert set(p.goals) == {"delivery_status", "payment_change"} and p.order_id == "ORD-2222" and p.target_method == "UPI"
    assert p.source == "rules" and 0.0 <= p.parse_conf <= 1.0


def _facts_from_run(scenario):
    env = EcomSupportEnv()
    agent = OracleAgent()
    obs, info = env.reset(options={"scenario": scenario, "failure_multiplier": 0.0})
    agent.reset(scenario)
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(agent.act(obs, info["action_mask"]))
        done = term or trunc
    name = scenario.world.customers[scenario.customer_id].name
    return build_facts(name, env.trace, info["terminal"], scenario.true_goals), env


def test_template_replies_are_always_faithful():
    for sc in generate_scenarios(get_splits()["iid"], 150) + generate_scenarios(get_splits()["missing_info"], 60):
        facts, _ = _facts_from_run(sc)
        reply = template_reply(facts)
        assert check_reply(reply, facts) == [], (sc.scenario_id, reply, check_reply(reply, facts))


def test_checker_catches_invented_facts(divya):
    facts, _ = _facts_from_run(divya)
    good = template_reply(facts)
    assert check_reply(good, facts) == []
    bad_cases = {
        "invented date": good + " It will arrive on 2026-12-25.",
        "invented order": good + " Also your order ORD-9999 is on the way.",
        "false refund claim": good + " Your refund has been processed.",
        "false cancel claim": good + " I've cancelled the order as well.",
        "invented status": "Hi Divya, your order was delivered yesterday.",
        "invented promise": good + " It will be there within 2 hours.",
        "invented amount": good + " We charged you 4999 rupees.",
    }
    for label, text in bad_cases.items():
        assert check_reply(text, facts), f"checker missed: {label}"


def test_llm_reply_falls_back_to_template_when_llm_is_down(divya):
    from ecommerce.language.llm import OllamaClient

    facts, _ = _facts_from_run(divya)
    res = write_reply(facts, mode="llm", client=OllamaClient(url="http://127.0.0.1:9", timeout=1.0))
    assert res.mode == "template(fallback)" and res.text == template_reply(facts)


def test_clarification_question_mentions_the_customers_orders():
    sc = [s for s in generate_scenarios(get_splits()["missing_info"], 40) if s.scenario_type == "missing_order"][0]
    env = EcomSupportEnv()
    obs, info = env.reset(options={"scenario": sc, "live": True})
    from ecommerce.env.state import ACTION_INDEX
    env.step(ACTION_INDEX["lookup_customer"])
    _, _, term, _, info = env.step(ACTION_INDEX["request_clarification"])
    assert info["terminal"] == "awaiting_customer"
    facts = build_facts(sc.world.customers[sc.customer_id].name, env.trace, "awaiting_customer", sc.parsed_goals)
    text = template_reply(facts)
    assert all(o in text for o in sc.world.customers[sc.customer_id].order_ids)


def test_signoffs_and_placeholders_are_removed_or_rejected(divya):
    facts, _ = _facts_from_run(divya)
    draft = "Hi Divya, your order is delayed.\n\nBest regards,\n[Your Name]"
    assert clean_draft(draft) == "Hi Divya, your order is delayed."
    assert "contains placeholder text" in check_reply("Hi Divya, [Customer Name] your order is delayed.", facts)
    assert "contains placeholder text" not in check_reply(template_reply(facts), facts)


ORDERS = [
    {"order_id": "ORD-4175", "status": "delivered", "items": ["Desk lamp"]},
    {"order_id": "ORD-4749", "status": "lost", "items": ["Bluetooth speaker", "Yoga mat"]},
    {"order_id": "ORD-3679", "status": "delayed", "items": ["Backpack"]},
]


def test_orders_are_found_from_natural_references():
    from ecommerce.language.parser import resolve_order_reference as r
    assert r("for the lost order (4749)", ORDERS) == "ORD-4749"
    assert r("it is 4749", ORDERS) == "ORD-4749"
    assert r("the order with ord-3679", ORDERS) == "ORD-3679"
    assert r("my delayed order please", ORDERS) == "ORD-3679"
    assert r("the speaker one", ORDERS) == "ORD-4749"
    assert r("my order", ORDERS) is None, "several orders and no clue: the agent must ask"
    assert r("the one from last week", ORDERS) is None


def test_meaning_flips_are_rejected_even_when_no_fact_is_invented():
    from ecommerce.language.reply import meaning_problems
    facts = {"customer": "Pooja", "terminal": "finish", "outcomes": {"refund": {"kind": "done", "refund_id": "RF-1", "amount": 500.0}}}
    assert meaning_problems("Pooja, your refund RF-1 of 500 rupees has been started.", facts) == []
    assert meaning_problems("Pooja, it appears that the refund is missing.", facts) == []  # no negation words: caught below by claim check
    declined = {"customer": "Pooja", "terminal": "finish", "outcomes": {"refund": {"kind": "declined", "reason": "outside_policy"}}}
    assert meaning_problems("Great news, your refund is on its way!", declined)
    assert not meaning_problems("I can't refund this order because it is outside our refund policy.", declined)
    human = {"customer": "Pooja", "terminal": "escalate", "outcomes": {"refund": {"kind": "needs_human", "reason": "x"}}}
    assert meaning_problems("Your refund is being looked at.", human)
    assert not meaning_problems("A human colleague must approve this refund.", human)


def test_questions_to_the_customer_are_never_left_to_the_llm():
    from ecommerce.language.llm import OllamaClient
    facts = {"customer": "Pooja", "terminal": "awaiting_customer", "outcomes": {},
             "question": {"type": "which_order", "orders": ["ORD-4175", "ORD-4749", "ORD-3679"]}}

    class Boom(OllamaClient):
        def chat_text(self, *a, **k):
            raise AssertionError("the LLM must not be called for a question")

    r = write_reply(facts, mode="llm", client=Boom())
    assert r.mode == "template" and "?" in r.text and all(o in r.text for o in facts["question"]["orders"])


ORDERS = [
    {"order_id": "ORD-4175", "status": "delivered", "items": ["Desk lamp"]},
    {"order_id": "ORD-4749", "status": "lost", "items": ["Bluetooth speaker", "Yoga mat"]},
    {"order_id": "ORD-3679", "status": "delayed", "items": ["Backpack"]},
]


def test_orders_are_found_from_natural_references():
    from ecommerce.language.parser import resolve_order_reference as r
    assert r("for the lost order (4749)", ORDERS) == "ORD-4749"
    assert r("where is my order 4749", ORDERS) == "ORD-4749"
    assert r("the order with ord-3679", ORDERS) == "ORD-3679"
    assert r("my delayed order please", ORDERS) == "ORD-3679"
    assert r("the speaker one", ORDERS) == "ORD-4749"
    assert r("my order", ORDERS) is None
    assert r("the one from last week", ORDERS) is None


def test_meaning_flips_are_rejected():
    from ecommerce.language.reply import meaning_problems
    declined = {"customer": "Pooja", "terminal": "finish", "outcomes": {"refund": {"kind": "declined", "reason": "outside_policy"}}}
    assert meaning_problems("Great news, your refund is on its way!", declined)
    assert not meaning_problems("I can't refund this order because it is outside our refund policy.", declined)
    done = {"customer": "Pooja", "terminal": "finish", "outcomes": {"refund": {"kind": "done", "refund_id": "RF-1", "amount": 500.0}}}
    assert not meaning_problems("Your refund RF-1 of 500 rupees has been started.", done)
    assert meaning_problems("Sorry, I cannot refund this.", done)
    human = {"customer": "Pooja", "terminal": "escalate", "outcomes": {"refund": {"kind": "needs_human", "reason": "x"}}}
    assert meaning_problems("Your refund is being looked at.", human)
    assert not meaning_problems("A human colleague must approve this refund.", human)


def test_questions_to_the_customer_are_never_left_to_the_llm():
    from ecommerce.language.llm import OllamaClient
    facts = {"customer": "Pooja", "terminal": "awaiting_customer", "outcomes": {},
             "question": {"type": "which_order", "orders": ["ORD-4175", "ORD-4749", "ORD-3679"]}}

    class Boom(OllamaClient):
        def chat_text(self, *a, **k):
            raise AssertionError("the LLM must not be called for a question")

    r = write_reply(facts, mode="llm", client=Boom())
    assert r.mode == "template" and "?" in r.text and all(o in r.text for o in facts["question"]["orders"])


def test_which_order_question_shows_the_items_so_the_customer_can_recognise_them():
    sc = [s for s in generate_scenarios(get_splits()["missing_info"], 60) if s.scenario_type == "missing_order"][0]
    env = EcomSupportEnv()
    env.reset(options={"scenario": sc, "live": True})
    from ecommerce.env.state import ACTION_INDEX
    env.step(ACTION_INDEX["lookup_customer"])
    env.step(ACTION_INDEX["request_clarification"])
    facts = build_facts(sc.world.customers[sc.customer_id].name, env.trace, "awaiting_customer", sc.parsed_goals)
    text = template_reply(facts)
    for oid in sc.world.customers[sc.customer_id].order_ids:
        assert oid in text
        for item in sc.world.orders[oid].items:
            assert item in text, f"{item} should be shown next to {oid}"


@pytest.mark.parametrize("text", ["where is ord 4175", "where is order 4175", "where is order #4175", "where is #4175", "where is ORD-4175"])
def test_every_way_of_writing_an_order_reference_is_cleaned_before_embedding(text):
    from ecommerce.language.router import normalise
    assert normalise(text) == "where is my order"


def test_when_the_order_is_known_the_agent_asks_what_to_do_not_which_order():
    sc = [s for s in generate_scenarios(get_splits()["missing_info"], 80) if s.scenario_type == "low_confidence"][0]
    env = EcomSupportEnv()
    env.reset(options={"scenario": sc, "live": True})
    from ecommerce.env.state import ACTION_INDEX
    for a in ("lookup_customer", "lookup_order", "request_clarification"):
        env.step(ACTION_INDEX[a])
    facts = build_facts(sc.world.customers[sc.customer_id].name, env.trace, "awaiting_customer", sc.parsed_goals)
    assert facts["question"]["type"] == "what_do_you_need"
    text = template_reply(facts)
    assert "What would you like me to do" in text and "Which one do you mean" not in text and sc.order_id in text


# ---------------- Group B: returns / damage claims and policy questions are goals of their own
import pytest as _pytest


@_pytest.mark.parametrize("text,goal", [
    ("the mouse is not working, i want to return the order or exchange", "return_item"),
    ("my phone came damaged, what can i do", "return_item"),
    ("i want to send this back, i don't like it", "return_item"),
    ("the lamp does not turn on, i'd like to send it back", "return_item"),
    ("i want a refund for my lost order", "refund"),
    ("my order never arrived, refund me", "refund"),
    ("please cancel my order", "cancel_order"),
    ("where is my order", "delivery_status"),
    ("what happens if my order is lost?", "policy_info"),
    ("how do returns and exchanges work?", "policy_info"),
])
def test_the_router_separates_returns_refunds_and_policy_questions(text, goal):
    from ecommerce.language.router_store import get_router

    assert goal in get_router().route(text).goals


def test_return_wording_is_read_by_regex_only_from_what_the_customer_wrote():
    from ecommerce.language.parser import extract_return_kind, extract_return_reason

    assert extract_return_reason("my parcel arrived broken") == "damaged"
    assert extract_return_reason("i changed my mind, i don't want it") == "change_of_mind"
    assert extract_return_reason("i want to return it") is None, "no reason given: the agent asks"
    assert extract_return_kind("please send a replacement") == "exchange"
    assert extract_return_kind("i want my money back") == "refund"
    assert extract_return_kind("exchange it or give me a refund") is None, "both named: the agent asks"


def test_typos_are_repaired_with_words_from_the_example_sentences_only():
    from collections import Counter

    from ecommerce.language.router import repair_typos

    vocab = Counter("where is my order status check parcel".split())
    assert repair_typos("whree is my oredr", vocab) == "where is my order"
    assert repair_typos("stauts cehck", vocab) == "status check"
    assert repair_typos("zebra crossing", vocab) == "zebra crossing", "unknown words with no neighbour stay as they are"


def test_a_return_is_not_a_refund_even_when_the_llm_says_so():
    """The real chat: the LLM read "i want to return my order" as a refund; the customer wanted a return."""
    from ecommerce.language.parser import MessageParser
    from ecommerce.language.router_store import get_router

    class FakeLLM:
        last_error = None

        def chat_json(self, system, user, schema):
            return {"goals": ["refund"], "order_id": None, "target_payment_method": None, "in_scope": True}, 0.0

    p = MessageParser(get_router(), mode="llm", llm=FakeLLM())
    assert p.parse("i want to return my order").goals == ("return_item",)
    assert p.parse("i want a refund for my lost order").goals == ("refund",), "a real refund request is still a refund"
    rules_only = MessageParser(get_router(), mode="rules")
    assert rules_only.parse("i want to return it and get my money back").goals == ("return_item",), "return and refund in one message is one request"
    assert rules_only.parse("my order never arrived, i want my money back").goals == ("refund",)


def test_i_have_already_returned_it_is_a_refund_request_not_a_new_return():
    """Real chat (Arjun Khan): "have returned my order. i want my refund" started the return questions again."""
    from ecommerce.language.parser import MessageParser
    from ecommerce.language.router_store import get_router

    p = MessageParser(get_router(), mode="rules")
    for text in ("have returned my order. i want my refund", "i have already sent it back, where is my money", "i returned it last week, refund please"):
        assert p.parse(text).goals == ("refund",), text
    for text in ("i want to return my order", "i want to return it and get my money back", "the item arrived broken, i want to send it back"):
        assert p.parse(text).goals == ("return_item",), text


def test_damage_wording_with_a_request_for_money_back_is_a_damage_claim():
    from ecommerce.language.parser import MessageParser
    from ecommerce.language.router_store import get_router

    p = MessageParser(get_router(), mode="rules")
    for text in ("my coffee mug ORD-9004 arrived broken, i want my money back", "my parcel arrived broken, give me a refund"):
        r = p.parse(text)
        assert r.goals == ("return_item",) and r.return_kind == "refund" and r.return_reason == "damaged", text
    for text in ("my order never arrived, i want my money back", "i want a refund for my lost order ORD-4749"):
        assert p.parse(text).goals == ("refund",), text


def test_the_checker_rejects_wrong_currency_and_invented_delivery_claims():
    """Found while taking screenshots: the LLM wrote "$180.00" (the shop is in rupees) and "ready for dispatch"."""
    from ecommerce.language.reply import check_reply

    facts = {"customer": "Ishaan", "terminal": "finish", "outcomes": {"return_item": {"kind": "done", "mode": "instant_refund", "refund_id": "RF-9001",
             "amount": 180.0, "order_id": "ORD-9001", "return_id": "RT-9001"}}, "order_id": "ORD-9001", "tool_trouble": []}
    assert any("currency" in p for p in check_reply("Hi Ishaan, I've started a refund of $180.00 for ORD-9001.", facts))
    assert any("not in the facts" in p for p in check_reply("Hi Ishaan, your items are ready for dispatch. I've started a refund (RF-9001) of 180 rupees for ORD-9001.", facts))
    assert check_reply("Hi Ishaan, I've started a refund (RF-9001) of 180 rupees for ORD-9001; our team will review it afterwards.", facts) == []
