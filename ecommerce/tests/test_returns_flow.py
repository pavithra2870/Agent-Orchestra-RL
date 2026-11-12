"""Returns, exchanges and damage claims through the chat API (rule-based agent, template replies, demo showcase customers)."""

import pytest
from fastapi.testclient import TestClient

from ecommerce.backend.main import create_app
from ecommerce.backend.services import Services

DAMAGED = "It arrived damaged or faulty"
MIND = "I don't want it, I changed my mind"
EXCHANGE, RETURN_REFUND, YES, NO = "Exchange for a new one", "Return it and get a refund", "Yes, I confirm", "No"


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    return TestClient(create_app(Services(db_path=tmp_path_factory.mktemp("rf") / "t.sqlite"), warm_up=False))


@pytest.fixture(autouse=True)
def _original_data(client):
    client.post("/api/demo/restore")
    yield


def talk(client, cid, messages, agent="rule_based"):
    """Send messages one after another, like the dashboard: carry the context, send the last 3 messages as history."""
    ctx, hist, out = None, [], []
    for m in messages:
        body = {"message": m, "customer_id": cid, "agent": agent, "language_mode": "template"}
        if ctx:
            body["context"] = ctx
        if hist:
            body["history"] = hist[-3:]
        ev = client.post("/api/chat/run", json=body).json()["events"]
        reply = next(e for e in ev if e["type"] == "reply")
        out.append((ev, reply))
        hist += [{"role": "user", "text": m}, {"role": "agent", "text": reply["text"]}]
        ctx = ev[-1].get("context") if ev[-1]["terminal"] == "awaiting_customer" else None
    return out


def order(client, cid):
    return next(c for c in client.get("/api/customers").json() if c["customer_id"] == cid)["orders"][0]


def test_small_damaged_item_asks_the_questions_in_order_then_refunds_right_away(client):
    out = talk(client, "CUST-0061", ["the item in my order ORD-9001 arrived broken", RETURN_REFUND, YES, "yes"])
    kinds = [r["text"] for _, r in out]
    assert "exchange" in kinds[0] and [o["label"] for o in out[0][1]["options"]] == [EXCHANGE, RETURN_REFUND]    # reason was in the text
    assert "did not damage it" in kinds[1] and [o["label"] for o in out[1][1]["options"]] == [YES, NO]
    assert "right away" in kinds[2] and "180 rupees" in kinds[2] and out[2][0][-1]["database_changes"] == [], "asks before acting"
    done = out[3][0][-1]
    assert done["terminal"] == "finish"
    changes = {c["field"]: c["after"] for c in done["database_changes"]}
    assert changes["payment_state"] == "refunded" and changes["return_state"] == "instant_refund"
    assert "refund" in out[3][1]["text"] and "180 rupees" in out[3][1]["text"] and "review" in out[3][1]["text"]
    o = order(client, "CUST-0061")
    assert (o["payment_state"], o["return_state"]) == ("refunded", "instant_refund")
    assert any(c["customer_id"] == "CUST-0061" for c in client.get("/api/demo/claims").json()["claims"]), "the claim is on the audit list"


def test_exchange_opens_a_request_and_waits_for_inspection(client):
    out = talk(client, "CUST-0061", ["the item in my order ORD-9001 arrived broken", EXCHANGE, YES, "yes"])
    changes = {c["field"]: c["after"] for c in out[3][0][-1]["database_changes"]}
    assert changes["return_state"] == "requested" and "payment_state" not in changes, "nothing is refunded yet"
    assert "exchange" in out[3][1]["text"] and "inspect" in out[3][1]["text"]
    waiting = client.get("/api/demo/returns").json()["waiting"]
    assert [(w["order_id"], w["kind"], w["reason"]) for w in waiting] == [("ORD-9001", "exchange", "damaged")]
    r = client.post("/api/demo/inspect", json={"order_id": "ORD-9001", "result": "passed"}).json()
    assert r["return_state"] == "inspection_passed" and r["exchange_id"] == "EX-9001" and r["payment_state"] == "paid"


def test_big_damaged_order_goes_to_staff_and_nothing_changes(client):
    out = talk(client, "CUST-0062", ["my order ORD-9002 arrived damaged", EXCHANGE, YES])
    done = out[-1][0][-1]
    assert done["terminal"] == "escalate" and done["database_changes"] == []
    assert "support staff will contact you" in out[-1][1]["text"]
    assert order(client, "CUST-0062")["return_state"] is None


def test_repeat_claimer_and_late_report_go_to_staff(client):
    out = talk(client, "CUST-0064", ["my coffee mug ORD-9004 arrived broken", RETURN_REFUND, YES])
    assert out[-1][0][-1]["terminal"] == "escalate" and "several recent damage claims" in out[-1][1]["text"]
    out = talk(client, "CUST-0063", ["my t-shirt ORD-9003 arrived damaged", RETURN_REFUND, YES])         # delivered 4 days ago
    assert out[-1][0][-1]["terminal"] == "escalate" and "hours after delivery" in out[-1][1]["text"]


def test_change_of_mind_return_refunds_only_after_inspection_passes(client):
    out = talk(client, "CUST-0063", ["i want to return my order", MIND, RETURN_REFUND, YES, "yes"])
    assert "damaged or faulty, or do you just not want it" in out[0][1]["text"]
    assert "unused and in its original packaging" in out[2][1]["text"]
    assert "refund is released once it passes" in out[3][1]["text"]
    o = order(client, "CUST-0063")
    assert (o["return_state"], o["payment_state"]) == ("requested", "paid")
    # a plain refund request now is declined: the return is in progress
    plain = talk(client, "CUST-0063", ["i want a refund for ORD-9003"])[0]
    assert plain[0][-1]["database_changes"] == [] and "return" in plain[1]["text"] and "in progress" in plain[1]["text"]
    r = client.post("/api/demo/inspect", json={"order_id": "ORD-9003", "result": "passed"}).json()
    assert (r["return_state"], r["payment_state"], r["refund_id"]) == ("inspection_passed", "refunded", "RF-9003")


def test_failed_inspection_means_no_refund_and_the_customer_is_told(client):
    talk(client, "CUST-0063", ["i want to return my order", MIND, RETURN_REFUND, YES, "yes"])
    r = client.post("/api/demo/inspect", json={"order_id": "ORD-9003", "result": "failed"}).json()
    assert (r["return_state"], r["payment_state"], r["refund_id"]) == ("inspection_failed", "paid", None)
    again = talk(client, "CUST-0063", ["please refund ORD-9003"])[0]
    assert again[0][-1]["database_changes"] == [] and "did not pass our inspection" in again[1]["text"] and "sent back to you" in again[1]["text"]


def test_a_return_outside_the_window_and_a_refused_declaration_are_declined(client):
    out = talk(client, "CUST-0065", ["i want to return my order ORD-9005", MIND, RETURN_REFUND, YES])         # delivered 12 days ago
    assert out[-1][0][-1]["database_changes"] == [] and "return window" in out[-1][1]["text"]
    out = talk(client, "CUST-0061", ["the item in my order ORD-9001 arrived broken", RETURN_REFUND, NO])
    assert out[-1][0][-1]["database_changes"] == [] and "can't open a return" in out[-1][1]["text"]


def test_saying_no_at_the_confirmation_changes_nothing(client):
    out = talk(client, "CUST-0061", ["the item in my order ORD-9001 arrived broken", RETURN_REFUND, YES, "no"])
    assert out[-1][0][-1]["database_changes"] == [] and order(client, "CUST-0061")["return_state"] is None


def test_inspection_endpoint_rejects_orders_with_no_waiting_return(client):
    assert client.post("/api/demo/inspect", json={"order_id": "ORD-9001", "result": "passed"}).status_code == 409
    assert client.post("/api/demo/inspect", json={"order_id": "ORD-0000", "result": "passed"}).status_code == 404
    assert client.post("/api/demo/inspect", json={"order_id": "bad", "result": "passed"}).status_code == 422


def test_claims_and_returns_survive_a_restart_until_restored(tmp_path):
    svc = Services(db_path=tmp_path / "t.sqlite")
    c = TestClient(create_app(svc, warm_up=False))
    talk(c, "CUST-0061", ["the item in my order ORD-9001 arrived broken", RETURN_REFUND, YES, "yes"])
    again = Services(db_path=tmp_path / "t.sqlite")
    assert again.demo_world().orders["ORD-9001"].return_state == "instant_refund"
    assert len(again.demo_world().customers["CUST-0061"].claims) == 1
    again.restore_demo_world()
    assert Services(db_path=tmp_path / "t.sqlite").demo_world().customers["CUST-0061"].claims == []


def test_an_item_that_is_already_returned_gets_a_refund_offer_from_the_database_state(client):
    """Arjun Khan's only order is already marked returned and paid: the agent must not ask the return questions again."""
    c = next(c for c in client.get("/api/customers").json() if c["name"] == "Arjun Khan")
    assert c["orders"][0]["status"] == "returned"
    out = talk(client, c["customer_id"], ["have returned my order. i want my refund", "yes"])
    assert "start a refund of" in out[0][1]["text"] and "damaged or faulty" not in out[0][1]["text"]
    changes = {x["field"]: x["after"] for x in out[1][0][-1]["database_changes"]}
    assert changes["payment_state"] == "refunded"


def test_return_questions_are_skipped_when_the_order_cannot_be_returned_anyway(client):
    c = next(c for c in client.get("/api/customers").json() if len(c["orders"]) == 1 and c["orders"][0]["status"] == "shipped")
    out = talk(client, c["customer_id"], ["i want to return my order"])
    done = out[0][0][-1]
    assert done["terminal"] == "finish" and done["database_changes"] == []
    assert "has not reached you yet" in out[0][1]["text"] and "damaged or faulty" not in out[0][1]["text"]
