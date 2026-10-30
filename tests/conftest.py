"""Shared test setup."""

import sys

import pytest


@pytest.fixture(autouse=True)
def generous_ask_limit():
    """Tests that call /ask must not hit the production rate limit; tests about
    the limit set their own limiter with monkeypatch, which runs after this."""
    api = sys.modules.get("api")
    if api is None or not hasattr(api, "ask_limiter"):
        yield
        return
    from ratelimit import RateLimiter

    original = api.ask_limiter
    api.ask_limiter = RateLimiter(per_minute=100_000, per_day=100_000)
    yield
    api.ask_limiter = original
