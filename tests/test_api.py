"""API tests: approvals are protected, and the endpoints keep the agent's guarantees."""

import sqlite3

import pytest
from fastapi.testclient import TestClient

import agent
import api
import store
import tools
from llm_client import ModelTurn, ToolCall

KEY = "test-key"
EMAIL = {"to": "hakan@example.com", "subject": "Test", "body": "Hello"}
client = TestClient(api.app)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "AGENT_DB", str(tmp_path / "test.db"))
    monkeypatch.setattr(tools, "EMAIL_ALLOWLIST", {"hakan@example.com"})
    monkeypatch.setattr(api, "ADMIN_API_KEY", KEY)
    # Fake model: proposes one email, then answers in text.
    replies = iter([
        ModelTurn("", [ToolCall("c1", "send_email", EMAIL)], {"role": "assistant", "content": ""}),
        ModelTurn("Waiting for approval.", [], {"role": "assistant", "content": "Waiting for approval."}),
    ])
    monkeypatch.setattr(agent, "chat_with_tools", lambda messages, tool_defs: next(replies))


def create_pending():
    response = client.post("/agent", json={"message": "Email hakan@example.com"}, headers={"x-admin-key": KEY})
    assert response.status_code == 200
    [action] = response.json()["actions"]
    assert action["status"] == "pending"
    return action["id"]


def outbox_count():
    with sqlite3.connect(store.AGENT_DB) as db:
        return db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]


def test_approve_without_a_valid_key_is_refused():
    action_id = create_pending()
    assert client.post(f"/actions/{action_id}/approve").status_code == 401
    assert client.post(f"/actions/{action_id}/approve", headers={"x-admin-key": "wrong"}).status_code == 401
    assert outbox_count() == 0


def test_approvals_are_disabled_when_no_key_is_configured(monkeypatch):
    action_id = create_pending()
    monkeypatch.setattr(api, "ADMIN_API_KEY", "")
    assert client.post(f"/actions/{action_id}/approve").status_code == 503


def test_review_and_approve_flow():
    action_id = create_pending()
    headers = {"x-admin-key": KEY}

    pending = client.get("/actions", params={"status": "pending"}, headers=headers).json()
    assert [a["id"] for a in pending] == [action_id]
    assert pending[0]["args"]["to"] == "hakan@example.com"  # the reviewer sees what they approve

    for _ in range(2):  # approving twice must send once
        response = client.post(f"/actions/{action_id}/approve", headers=headers)
        assert response.status_code == 200
        assert response.json()["status"] == "done"
    assert outbox_count() == 1

    events = [e["event"] for e in client.get(f"/actions/{action_id}/audit", headers=headers).json()]
    assert events == ["proposed -> pending", "pending -> running", "running -> done"]


def test_unknown_action_returns_404():
    assert client.post("/actions/999/approve", headers={"x-admin-key": KEY}).status_code == 404


def test_invalid_status_filter_returns_422():
    assert client.get("/actions", params={"status": "hacked"}, headers={"x-admin-key": KEY}).status_code == 422


def test_message_length_is_limited():
    response = client.post("/agent", json={"message": "x" * 501}, headers={"x-admin-key": KEY})
    assert response.status_code == 422

def test_agent_requires_a_valid_key():
    assert client.post("/agent", json={"message": "hi"}).status_code == 401
    assert client.post("/agent", json={"message": "hi"}, headers={"x-admin-key": "wrong"}).status_code == 401