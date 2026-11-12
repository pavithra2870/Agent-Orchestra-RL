"""API tests with a real FastAPI client and a throw-away database."""

import json

import pytest
from fastapi.testclient import TestClient

from ecommerce.backend.main import create_app
from ecommerce.backend.services import Services


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    db = tmp_path_factory.mktemp("db") / "t.sqlite"
    svc = Services(db_path=db)
    return TestClient(create_app(svc, warm_up=False))


@pytest.fixture(autouse=True)
def _original_data(client):
    """Live chats now change the demo database for real, so every test starts from the original data."""
    client.post("/api/demo/restore")
    yield


def _customer_with(client, pred):
    for c in client.get("/api/customers").json():
        if pred(c):
            return c
    pytest.skip("no suitable customer in the demo world")


def test_health_and_static_info(client):
    h = client.get("/api/health").json()
    assert h["ok"] and "rule_based" in h["agents_available"]
    assert {a["key"] for a in client.get("/api/agents").json()} >= {"dqn", "ppo", "cql", "rule_based"}
    assert "success_rate" in client.get("/api/metrics/help").json()
    assert len(client.get("/api/customers").json()) > 10
    assert client.get("/api/examples").json()


def test_chat_run_returns_ordered_events_and_persists(client):
    c = _customer_with(client, lambda c: len(c["orders"]) == 1)
    oid = c["orders"][0]["order_id"]
    r = client.post("/api/chat/run", json={"message": f"where is {oid}?", "customer_id": c["customer_id"], "agent": "rule_based"})
    assert r.status_code == 200
    types = [e["type"] for e in r.json()["events"]]
    assert types[0] == "start" and types[1] == "parse" and types[-1] == "done"
    assert {"decision", "tool", "state", "reply", "evaluation"} <= set(types)
    done = r.json()["events"][-1]
    assert done["success"] is True and done["terminal"] == "finish"
    runs = client.get("/api/runs").json()
    assert runs["total"] >= 1
    detail = client.get(f"/api/runs/{r.json()['run_id']}").json()
    assert detail["success"] == 1 and len(detail["events"]) == len(r.json()["events"]) and detail["reply"]


def test_streaming_endpoint_emits_server_sent_events(client):
    c = _customer_with(client, lambda c: len(c["orders"]) == 1)
    oid = c["orders"][0]["order_id"]
    with client.stream("GET", "/api/chat/stream", params={"message": f"status of {oid}", "customer_id": c["customer_id"],
                                                          "agent": "rule_based"}) as resp:
        assert resp.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[6:]) for line in resp.iter_lines() if line.startswith("data: ")]
    assert events[0]["type"] == "start" and events[-1]["type"] == "done"


def test_out_of_scope_is_handed_to_a_human(client):
    c = client.get("/api/customers").json()[0]
    r = client.post("/api/chat/run", json={"message": "can you recommend a good smartphone?", "customer_id": c["customer_id"],
                                           "agent": "rule_based"}).json()
    done = r["events"][-1]
    assert done["terminal"] == "escalate" and done["success"] is True and done["database_changes"] == []


def test_missing_order_asks_then_follow_up_finishes(client):
    # two orders are on their way, so inference cannot pick one and the agent must ask (listing only those two)
    c = _customer_with(client, lambda c: sum(o["status"] in ("processing", "shipped", "delayed") for o in c["orders"]) > 1)
    r = client.post("/api/chat/run", json={"message": "where is my order?", "customer_id": c["customer_id"], "agent": "rule_based"}).json()
    done = r["events"][-1]
    assert done["terminal"] == "awaiting_customer" and "context" in done
    reply = next(e for e in r["events"] if e["type"] == "reply")["text"]
    candidates = [o["order_id"] for o in c["orders"] if o["status"] in ("processing", "shipped", "delayed")]
    assert all(i in reply for i in candidates)
    oid = candidates[0]
    r2 = client.post("/api/chat/run", json={"message": oid, "customer_id": c["customer_id"], "agent": "rule_based",
                                            "context": done["context"]}).json()
    assert r2["events"][-1]["terminal"] == "finish" and r2["events"][-1]["success"] is True


def test_live_run_actually_changes_the_database_only_when_allowed(client):
    w = client.get("/api/customers").json()
    target = next((c for c in w for o in c["orders"] if o["status"] == "processing" and o["payment_method"] == "COD"), None)
    if target is None:
        pytest.skip("no processing COD order in the demo world")
    o = next(o for o in target["orders"] if o["status"] == "processing" and o["payment_method"] == "COD")
    cid = target["customer_id"]
    r = client.post("/api/chat/run", json={"message": f"please change the payment of {o['order_id']} to UPI",
                                           "customer_id": cid, "agent": "rule_based"}).json()
    asked = r["events"][-1]
    assert asked["terminal"] == "awaiting_customer" and asked["database_changes"] == [] and asked["context"]["confirm"], "asks first"
    assert client.get("/api/demo/changes").json()["changed_orders"] == []
    r2 = client.post("/api/chat/run", json={"message": "yes", "customer_id": cid, "agent": "rule_based", "context": asked["context"]}).json()
    ch = r2["events"][-1]["database_changes"]
    assert any(c["field"] == "payment_method" and c["after"] == "UPI" for c in ch)
    again = client.get("/api/customers").json()
    shown = next(x for c in again if c["customer_id"] == cid for x in c["orders"] if x["order_id"] == o["order_id"])
    assert shown["payment_method"] == "UPI", "the change is kept in the demo database"
    assert [c["order_id"] for c in client.get("/api/demo/changes").json()["changed_orders"]] == [o["order_id"]]
    assert client.post("/api/demo/restore").json()["restored_orders"] == 1
    back = client.get("/api/customers").json()
    assert next(x for c in back if c["customer_id"] == cid for x in c["orders"] if x["order_id"] == o["order_id"])["payment_method"] == "COD"


def test_unknown_agent_and_bad_input_are_rejected(client):
    assert client.post("/api/chat/run", json={"message": "hi", "customer_id": "CUST-0001", "agent": "nope"}).status_code == 404
    assert client.post("/api/chat/run", json={"message": "", "customer_id": "CUST-0001"}).status_code == 422
    assert client.post("/api/chat/run", json={"message": "hi", "customer_id": "bad"}).status_code == 422


def test_job_validation_blocks_bad_requests(client):
    for body in ({"agent": "rm -rf", "seeds": [0]}, {"agent": "dqn", "seeds": [0], "episodes": 5},
                 {"agent": "dqn", "seeds": [0], "hp": {"__import__": 1}}, {"agent": "dqn", "seeds": list(range(9))},
                 {"agent": "dqn", "seeds": [0], "variant": "evil"}):
        assert client.post("/api/jobs/train", json=body).status_code == 400, body
    assert client.get("/api/jobs/nope").status_code == 404


def test_replay_endpoint_reruns_a_scenario_deterministically(client):
    a = client.post("/api/replay", json={"scenario_id": "iid-00003", "agent": "rule_based"}).json()
    b = client.post("/api/replay", json={"scenario_id": "iid-00003", "agent": "rule_based"}).json()
    key = lambda r: [(e["action"], e["ok"]) for e in r["events"] if e["type"] == "tool"]
    assert key(a) == key(b) and a["scenario"]["scenario_id"] == "iid-00003"
    assert client.post("/api/replay", json={"scenario_id": "bogus-00001", "agent": "rule_based"}).status_code == 400


def test_follow_up_answer_in_natural_words_is_understood(client):
    c = _customer_with(client, lambda c: len(c["orders"]) > 1 and len({o["status"] for o in c["orders"]}) == len(c["orders"]))
    first = client.post("/api/chat/run", json={"message": "want refund for my order", "customer_id": c["customer_id"], "agent": "rule_based",
                                               "language_mode": "template"}).json()
    done = first["events"][-1]
    assert done["terminal"] == "awaiting_customer"
    target = c["orders"][1]
    answer = f"for the {target['status']} order ({target['order_id'][-4:]})"
    second = client.post("/api/chat/run", json={"message": answer, "customer_id": c["customer_id"], "agent": "rule_based",
                                                "context": done["context"]}).json()
    last = second["events"][-1]
    assert last["terminal"] != "awaiting_customer", "the order was named, so the agent must not ask again"
    assert any(e["type"] == "tool" and e["args"].get("order_id") == target["order_id"] for e in second["events"] if e["type"] == "tool" and e.get("args"))


def test_follow_up_answer_in_natural_words_is_understood(client):
    c = _customer_with(client, lambda c: len(c["orders"]) > 1 and len({o["status"] for o in c["orders"]}) == len(c["orders"]))
    first = client.post("/api/chat/run", json={"message": "want refund for my order", "customer_id": c["customer_id"],
                                               "agent": "rule_based"}).json()
    done = first["events"][-1]
    assert done["terminal"] == "awaiting_customer"
    target = c["orders"][1]
    second = client.post("/api/chat/run", json={"message": f"for the {target['status']} order ({target['order_id'][-4:]})",
                                                "customer_id": c["customer_id"], "agent": "rule_based", "context": done["context"]}).json()
    assert second["events"][-1]["terminal"] != "awaiting_customer", "the order was named, so the agent must not ask again"
    used = {e["args"].get("order_id") for e in second["events"] if e["type"] == "tool" and e.get("args")}
    assert target["order_id"] in used


def test_bare_order_number_in_the_first_message_is_understood(client):
    c = _customer_with(client, lambda c: len(c["orders"]) > 1)
    oid = c["orders"][0]["order_id"]
    r = client.post("/api/chat/run", json={"message": f"where is my order {oid[-4:]}", "customer_id": c["customer_id"], "agent": "rule_based"}).json()
    assert r["events"][-1]["terminal"] == "finish"
    assert next(e for e in r["events"] if e["type"] == "parse")["order_id"] == oid


def test_quoting_an_order_that_is_not_theirs_gets_a_could_not_find_answer_and_their_real_orders(client):
    customers = client.get("/api/customers").json()
    mine = next(c for c in customers if len(c["orders"]) > 1)
    other = next(c for c in customers if c["customer_id"] != mine["customer_id"])
    foreign = other["orders"][0]
    r = client.post("/api/chat/run", json={"message": f"where is my order {foreign['order_id']}", "customer_id": mine["customer_id"],
                                           "agent": "rule_based"}).json()
    reply = next(e for e in r["events"] if e["type"] == "reply")["text"]
    assert r["events"][-1]["terminal"] == "awaiting_customer"
    assert f"couldn't find order {foreign['order_id']}" in reply
    assert all(o["order_id"] in reply and all(i in reply for i in o["items"]) for o in mine["orders"])
    assert not any(i in reply for i in foreign["items"] if all(i not in o["items"] for o in mine["orders"])), \
        "must never reveal what is in somebody else's order"


# ---------------- order inference, memory and confirmation through the API
def _chat(client, cust, msg, ctx=None, hist=None, agent="dqn"):
    body = {"message": msg, "customer_id": cust["customer_id"], "agent": agent, "language_mode": "template"}
    if ctx:
        body["context"] = ctx
    if hist:
        body["history"] = hist
    return client.post("/api/chat/run", json=body).json()["events"]


def _in_transit(o):
    return o["status"] in ("processing", "shipped", "delayed")


def test_where_is_my_order_uses_the_only_order_on_its_way(client):
    c = _customer_with(client, lambda c: len(c["orders"]) > 1 and sum(_in_transit(o) for o in c["orders"]) == 1)
    target = next(o["order_id"] for o in c["orders"] if _in_transit(o))
    ev = _chat(client, c, "where is my order")
    parse = next(e for e in ev if e["type"] == "parse")
    reply = next(e for e in ev if e["type"] == "reply")
    assert parse["resolution"]["how"] == "inferred" and parse["order_id"] == target
    assert target in reply["text"] and "took you to mean" in reply["text"]
    assert ev[-1]["terminal"] == "finish"


def test_a_refund_on_an_assumed_order_waits_for_yes_and_changes_nothing_before(client):
    def refundable(o):
        return o["status"] in ("lost", "returned") and o["payment_state"] == "paid" and o["total"] <= 5000
    c = _customer_with(client, lambda c: len(c["orders"]) > 1 and sum(refundable(o) for o in c["orders"]) == 1)
    target = next(o["order_id"] for o in c["orders"] if refundable(o))
    ev = _chat(client, c, "i want a refund")
    done = ev[-1]
    assert done["terminal"] == "awaiting_customer" and done["steps"] == 0 and done["database_changes"] == []
    assert done["context"]["confirm"] and done["context"]["order_id"] == target
    yes = _chat(client, c, "yes", done["context"])
    assert yes[-1]["terminal"] == "finish"
    assert yes[-1]["database_changes"] and target in next(e for e in yes if e["type"] == "reply")["text"]
    no = _chat(client, c, "no", done["context"])
    assert no[-1]["database_changes"] == [] and no[-1]["terminal"] == "awaiting_customer"
    assert target not in (no[-1]["context"].get("order_id") or "")


def test_nothing_qualifies_so_the_agent_asks_and_says_why(client):
    def paid_closed(o):
        return o["status"] in ("lost", "returned", "delivered") and o["payment_state"] == "paid"
    c = _customer_with(client, lambda c: len(c["orders"]) > 1 and not any(paid_closed(o) for o in c["orders"]))
    ev = _chat(client, c, "i want a refund")
    reply = next(e for e in ev if e["type"] == "reply")["text"]
    assert ev[-1]["terminal"] == "awaiting_customer" and "looks eligible" in reply
    assert all(o["order_id"] in reply for o in c["orders"]) and ev[-1]["database_changes"] == []


def test_a_short_followup_is_parsed_and_uses_the_conversation(client):
    c = _customer_with(client, lambda c: len(c["orders"]) > 1)
    oid = c["orders"][0]["order_id"]
    first = _chat(client, c, f"sorry my order {oid}")
    hist = [{"role": "user", "text": f"sorry my order {oid}"}, {"role": "agent", "text": next(e for e in first if e["type"] == "reply")["text"]}]
    ev = _chat(client, c, "where it is", first[-1].get("context"), hist)
    parse = next(e for e in ev if e["type"] == "parse")
    assert parse["goals"] == ["delivery_status"] and parse["order_id"] == oid


def test_auto_language_mode_is_accepted_and_history_is_limited(client):
    c = _customer_with(client, lambda c: len(c["orders"]) == 1)
    r = client.post("/api/chat/run", json={"message": "where is my order", "customer_id": c["customer_id"], "agent": "rule_based",
                                           "language_mode": "auto", "history": []})
    assert r.status_code == 200 and r.json()["events"][0]["language_mode"] in ("llm", "template")
    too_long = [{"role": "user", "text": "x"}] * 4
    assert client.post("/api/chat/run", json={"message": "hi", "customer_id": c["customer_id"], "history": too_long}).status_code == 422


def test_online_is_not_a_payment_method_so_the_agent_asks_and_changes_nothing(client):
    c = _customer_with(client, lambda c: len(c["orders"]) == 1 and c["orders"][0]["status"] in ("processing", "shipped", "delayed")
                       and c["orders"][0]["payment_method"] == "COD")
    for agent in ("dqn", "rule_based", "ppo", "cql"):
        ev = _chat(client, c, "wanna pay now online itself", agent=agent)
        done, reply = ev[-1], next(e for e in ev if e["type"] == "reply")
        assert done["terminal"] == "awaiting_customer" and done["database_changes"] == [], agent
        assert "UPI" in reply["text"] and "card" in reply["text"], reply["text"]
        assert {o["text"] for o in reply["options"]} == {"UPI", "card"}


def test_cancel_and_refund_on_a_named_or_only_order_also_wait_for_yes(client):
    c = _customer_with(client, lambda c: len(c["orders"]) == 1 and c["orders"][0]["status"] == "processing")
    oid = c["orders"][0]["order_id"]
    for msg in ("i wanna cancel the order", f"cancel {oid}"):
        ev = _chat(client, c, msg)
        assert ev[-1]["terminal"] == "awaiting_customer" and ev[-1]["database_changes"] == [] and ev[-1]["context"]["confirm"], msg
        reply = next(e for e in ev if e["type"] == "reply")
        assert oid in reply["text"] and [o["label"] for o in reply["options"]] == ["Yes", "No"]
    yes = _chat(client, c, "yes", ev[-1]["context"])
    assert yes[-1]["database_changes"] and yes[-1]["database_changes"][0]["after"] == "cancelled"
    # the cancel is remembered: a refund now is judged on the cancelled order, not on a fresh copy
    shown = next(x for cc in client.get("/api/customers").json() if cc["customer_id"] == c["customer_id"] for x in cc["orders"])
    assert shown["status"] == "cancelled"


def test_nothing_to_confirm_when_the_action_would_be_declined(client):
    c = _customer_with(client, lambda c: len(c["orders"]) == 1 and c["orders"][0]["status"] in ("shipped", "delayed"))
    oid = c["orders"][0]["order_id"]
    ev = _chat(client, c, f"cancel {oid}")
    assert ev[-1]["terminal"] != "awaiting_customer" or "confirm" not in (ev[-1].get("context") or {})
    assert "can't cancel" in next(e for e in ev if e["type"] == "reply")["text"]


def test_demo_changes_survive_a_restart_until_restored(tmp_path):
    from ecommerce.backend.services import Services

    svc = Services(db_path=tmp_path / "t.sqlite")
    order = next(iter(svc.demo_world().orders.values()))
    changed = type(order)(**{**order.__dict__, "status": "cancelled"})
    svc.commit_orders({order.order_id: changed})
    again = Services(db_path=tmp_path / "t.sqlite")          # a new process reading the same file
    assert again.demo_world().orders[order.order_id].status == "cancelled"
    assert [c["order_id"] for c in again.changed_orders()] == [order.order_id]
    assert again.restore_demo_world() == 1
    assert Services(db_path=tmp_path / "t.sqlite").demo_world().orders[order.order_id].status == order.status


def test_a_clearly_off_topic_message_is_handed_to_a_human_by_the_agents_that_matter(client):
    """Real behaviour found while taking screenshots: the real router is very confident that off-topic is off-topic,
    a state the agents never saw in training, and the learned ones wandered. Now the parse reaches them as 'unsure'."""
    c = _customer_with(client, lambda c: len(c["orders"]) > 1)
    for msg in ("can you recommend a good smartphone?", "tell me a joke", "what's the weather like today?"):
        for agent in ("dqn", "ppo", "cql", "rule_based"):
            ev = _chat(client, c, msg, agent=agent)
            assert ev[-1]["terminal"] == "escalate" and ev[-1]["database_changes"] == [], (msg, agent)
            reply = next(e for e in ev if e["type"] == "reply")["text"]
            assert "human colleague" in reply and "which one do you mean" not in reply


def test_an_unclear_message_gets_a_what_do_you_need_question_not_which_order(client):
    c = _customer_with(client, lambda c: len(c["orders"]) > 1)
    ev = _chat(client, c, "hmm", agent="rule_based")
    reply = next(e for e in ev if e["type"] == "reply")["text"]
    assert "which one do you mean" not in reply
