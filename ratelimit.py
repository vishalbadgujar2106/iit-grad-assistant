"""
ratelimit.py
A small in-memory sliding-window rate limiter for /chat, so one person (or a
script) can't burn through the Anthropic API budget.

Limits are per client IP, plus one global hourly cap across all clients.
State lives in process memory: on a single server that's exact, but on a
serverless host each warm instance keeps its own counts, so treat it as a
best-effort guard. A shared store (e.g. Redis) would be needed for a hard
global limit.
"""

import threading
import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    def __init__(self, limits, clock=time.monotonic):
        """
        limits: list of (max_requests, window_seconds) applied per key,
        e.g. [(8, 60), (60, 86400)] = 8 per minute and 60 per day.
        """
        self.limits = sorted(limits, key=lambda lw: lw[1])
        self.longest_window = max(w for _, w in limits)
        self.clock = clock
        self.hits = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key):
        """
        Record a request for key if it's allowed. Returns 0 when allowed,
        otherwise the number of seconds until the caller can retry.
        """
        now = self.clock()
        with self.lock:
            hits = self.hits[key]
            while hits and now - hits[0] >= self.longest_window:
                hits.popleft()

            for max_requests, window in self.limits:
                in_window = [t for t in hits if now - t < window]
                if len(in_window) >= max_requests:
                    # Free once the oldest request in this window expires.
                    oldest = in_window[-max_requests]
                    return max(1, int(window - (now - oldest)) + 1)

            hits.append(now)
            if len(self.hits) > 10_000:
                self._prune(now)
            return 0

    def _prune(self, now):
        """Drop keys with no recent requests so memory stays bounded."""
        stale = [k for k, h in self.hits.items()
                 if not h or now - h[-1] >= self.longest_window]
        for k in stale:
            del self.hits[k]
