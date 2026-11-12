"""Policy questions: the policy texts match the rules, topics are found, and a policy question goes through the RL agent
and its `lookup_policy` tool (no order needed, nothing changed)."""

import pytest
from fastapi.testclient import TestClient

from ecommerce.backend.main import create_app
from ecommerce.backend.services import Services
from ecommerce.config import get_settings
from ecommerce.language.knowledge import TEST_QUESTIONS, KnowledgeBase, build_topics
from ecommerce.language.parser import MessageParser
from ecommerce.language.router_store import get_router


@pytest.fixture(scope="module")
def parser():
    return MessageParser(get_router(), mode="rules")


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return TestClient(create_app(Services(db_path=tmp_path_factory.mktemp("kb") / "t.sqlite"), warm_up=False))


def test_answers_use_the_numbers_from_the_configuration():
    w = get_settings().world
    topics = build_topics()
    assert str(w.delay_refund_days) in topics["refund_policy"][1] and str(int(w.high_value_threshold)) in topics["refund_policy"][1]
    assert str(w.return_window_days) in topics["returns_exchange"][1]
    damaged = topics["damaged_item"][1]
    assert str(w.damage_report_window_days * 24) in damaged and str(int(w.instant_refund_cap)) in damaged and str(w.damage_claims_limit) in damaged
    assert len(topics) == 7


def test_the_brief_example_question_is_a_policy_goal_with_the_right_topic(parser):
    p = parser.parse("so if order is lost what happens?")
    assert "policy_info" in p.goals and p.policy_topic == "lost_order"


@pytest.mark.parametrize("text", [
    "i want a refund for my order", "where is my order", "cancel my order please", "change my payment to UPI",
    "refund ORD-1234", "what is the status of ORD-1234?", "can you cancel ORD-1234 for me?",
])
def test_requests_about_an_order_are_not_taken_for_policy_questions(parser, text):
    assert "policy_info" not in parser.parse(text).goals


def test_held_out_policy_questions_get_the_right_topic():
    kb = KnowledgeBase(get_router().embedder)
    hits = [kb.best(q)[0] == topic for topic, qs in TEST_QUESTIONS.items() for q in qs]
    assert sum(hits) / len(hits) >= 0.85                 # measured 95.2% (results/language/knowledge_eval.json)


def test_a_policy_question_goes_through_the_agent_and_its_tool_and_changes_nothing(client):
    before = client.get("/api/customers").json()
    r = client.post("/api/chat/run", json={"message": "so if order is lost what happens?", "customer_id": "CUST-0007",
                                           "agent": "rule_based", "language_mode": "template"}).json()
    done = r["events"][-1]
    tools = [e["action"] for e in r["events"] if e["type"] == "tool"]
    assert "lookup_policy" in tools and "lookup_customer" not in tools, "no order is needed for a policy question"
    assert done["terminal"] == "finish" and done["success"] is True and done["database_changes"] == []
    reply = next(e for e in r["events"] if e["type"] == "reply")["text"]
    assert "refund" in reply and "5000" in reply
    assert client.get("/api/customers").json() == before


def test_a_policy_question_mixed_with_a_request_answers_both(client):
    c = next(c for c in client.get("/api/customers").json() if len(c["orders"]) == 1 and c["orders"][0]["status"] in ("shipped", "delayed", "processing"))
    oid = c["orders"][0]["order_id"]
    r = client.post("/api/chat/run", json={"message": f"what is the cancellation rule? and where is {oid}", "customer_id": c["customer_id"],
                                           "agent": "rule_based", "language_mode": "template"}).json()
    reply = next(e for e in r["events"] if e["type"] == "reply")["text"]
    assert "cancelled only while" in reply and oid in reply
