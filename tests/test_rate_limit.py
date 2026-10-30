"""Rate limits for the public /ask endpoint (written before the code, 28 Oct 2026)."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

import api
from ratelimit import RateLimiter, RateLimitExceeded

client = TestClient(api.app)


class FakeTime:
    def __init__(self):
        self.seconds = 0.0
        self.utc = datetime(2026, 10, 28, 12, 0, tzinfo=timezone.utc)

    def clock(self):
        return self.seconds

    def now_utc(self):
        return self.utc

    def advance(self, seconds):
        self.seconds += seconds
        self.utc += timedelta(seconds=seconds)


def limiter(per_minute=3, per_day=100):
    fake = FakeTime()
    return RateLimiter(per_minute, per_day, clock=fake.clock, now_utc=fake.now_utc), fake


def test_per_minute_limit_blocks_the_next_request():
    rl, _ = limiter(per_minute=3)
    for _ in range(3):
        rl.check("1.1.1.1")
    with pytest.raises(RateLimitExceeded) as exceeded:
        rl.check("1.1.1.1")
    assert exceeded.value.scope == "per_minute"
    assert 1 <= exceeded.value.retry_after <= 60


def test_window_frees_up_after_a_minute():
    rl, fake = limiter(per_minute=2)
    rl.check("1.1.1.1")
    rl.check("1.1.1.1")
    fake.advance(61)
    rl.check("1.1.1.1")  # no exception


def test_clients_are_counted_separately_per_minute():
    rl, _ = limiter(per_minute=1)
    rl.check("1.1.1.1")
    rl.check("2.2.2.2")  # another client still has its own minute
    with pytest.raises(RateLimitExceeded):
        rl.check("1.1.1.1")


def test_daily_cap_counts_all_clients_and_resets_next_day():
    rl, fake = limiter(per_minute=100, per_day=3)
    rl.check("1.1.1.1")
    rl.check("2.2.2.2")
    rl.check("3.3.3.3")
    with pytest.raises(RateLimitExceeded) as exceeded:
        rl.check("4.4.4.4")
    assert exceeded.value.scope == "daily"
    fake.advance(12 * 3600 + 1)  # past midnight UTC
    rl.check("4.4.4.4")


def test_limits_must_be_positive():
    with pytest.raises(ValueError):
        RateLimiter(per_minute=0, per_day=10)


@pytest.fixture
def fake_ask(monkeypatch):
    """/ask without a real index or model; counts model calls."""
    calls = {"chat": 0}

    def fake_chat(system, prompt):
        calls["chat"] += 1
        return "Antwoord [1]"

    monkeypatch.setattr(api, "get_collection", lambda: object())
    monkeypatch.setattr(api, "retrieve", lambda *a, **k: [
        ("Een stukje tekst.", {"source": "a.pdf", "page": 1, "chunk_id": "a.pdf-p1-c1"}, 0.1)])
    monkeypatch.setattr(api, "chat", fake_chat)
    monkeypatch.setattr(api.provenance, "record", lambda *a, **k: None)
    return calls


def test_ask_returns_429_with_retry_after_and_does_not_call_the_model(fake_ask, monkeypatch):
    rl, _ = limiter(per_minute=2)
    monkeypatch.setattr(api, "ask_limiter", rl)
    statuses = [client.post("/ask", json={"question": "Wat is een breuk?"}).status_code for _ in range(3)]
    assert statuses == [200, 200, 429]
    assert fake_ask["chat"] == 2
    response = client.post("/ask", json={"question": "Wat is een breuk?"})
    assert response.status_code == 429
    assert int(response.headers["retry-after"]) >= 1


def test_client_address_is_the_last_forwarded_for_entry(fake_ask, monkeypatch):
    rl, _ = limiter(per_minute=1)
    monkeypatch.setattr(api, "ask_limiter", rl)
    first = {"X-Forwarded-For": "6.6.6.6, 9.9.9.9"}
    second = {"X-Forwarded-For": "6.6.6.6, 8.8.8.8"}  # same spoofed start, other real address
    assert client.post("/ask", json={"question": "Vraag"}, headers=first).status_code == 200
    assert client.post("/ask", json={"question": "Vraag"}, headers=second).status_code == 200
    assert client.post("/ask", json={"question": "Vraag"}, headers=first).status_code == 429


def test_health_is_not_rate_limited(monkeypatch):
    rl, _ = limiter(per_minute=1)
    monkeypatch.setattr(api, "ask_limiter", rl)
    assert all(client.get("/health").status_code == 200 for _ in range(5))
