"""In-memory rate limits for the public /ask endpoint.

Two limits protect the Azure OpenAI budget:
- per client: at most ``per_minute`` requests in any 60 seconds;
- in total: at most ``per_day`` requests per UTC day, for all clients together.

The counters live in memory, so they are exact for one replica (Container Apps
max replicas = 1) and reset when the container restarts. The daily cap does not
depend on the client address, so it also holds when an address is spoofed.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from datetime import date, datetime, timedelta, timezone
from typing import Callable


class RateLimitExceeded(Exception):
    def __init__(self, scope: str, retry_after: float):
        super().__init__(scope)
        self.scope = scope  # "per_minute" or "daily"
        self.retry_after = max(1, math.ceil(retry_after))


class RateLimiter:
    WINDOW_SECONDS = 60

    def __init__(self, per_minute: int, per_day: int,
                 clock: Callable[[], float] = time.monotonic,
                 now_utc: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
                 max_clients: int = 10_000):
        if per_minute < 1 or per_day < 1:
            raise ValueError("Rate limits must be at least 1.")
        self.per_minute = per_minute
        self.per_day = per_day
        self._clock = clock
        self._now_utc = now_utc
        self._max_clients = max_clients
        self._hits: dict[str, deque[float]] = {}
        self._day: date | None = None
        self._day_count = 0
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls) -> "RateLimiter":
        return cls(per_minute=int(os.getenv("ASK_RATE_PER_MINUTE", "10")),
                   per_day=int(os.getenv("ASK_DAILY_LIMIT", "200")))

    def check(self, client: str) -> None:
        """Count one request for ``client`` or raise RateLimitExceeded."""
        with self._lock:
            now = self._clock()
            utc_now = self._now_utc()
            if utc_now.date() != self._day:
                self._day, self._day_count = utc_now.date(), 0
            if self._day_count >= self.per_day:
                midnight = datetime.combine(self._day + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
                raise RateLimitExceeded("daily", (midnight - utc_now).total_seconds())

            window = self._hits.setdefault(client, deque())
            while window and now - window[0] >= self.WINDOW_SECONDS:
                window.popleft()
            if len(window) >= self.per_minute:
                raise RateLimitExceeded("per_minute", self.WINDOW_SECONDS - (now - window[0]))

            window.append(now)
            self._day_count += 1
            if len(self._hits) > self._max_clients:
                self._prune(now)

    def _prune(self, now: float) -> None:
        """Forget clients without a request in the last minute, so memory stays bounded."""
        stale = [client for client, window in self._hits.items()
                 if not window or now - window[-1] >= self.WINDOW_SECONDS]
        for client in stale:
            del self._hits[client]


def client_ip(request) -> str:
    """The caller's address. Behind Azure's ingress the proxy appends the real
    client address as the right-most X-Forwarded-For entry; earlier entries can
    be written by the client itself and are not trusted."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        last = forwarded.split(",")[-1].strip()
        if last:
            return last
    return request.client.host if request.client else "unknown"
