"""Unit tests for the sliding-window rate limiter, using a fake clock."""

from ratelimit import SlidingWindowLimiter


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_allows_up_to_the_limit_then_blocks_with_retry_time():
    clock = FakeClock()
    limiter = SlidingWindowLimiter([(3, 60)], clock=clock)
    assert [limiter.check("a") for _ in range(3)] == [0, 0, 0]
    wait = limiter.check("a")
    assert 0 < wait <= 61


def test_window_slides_so_old_requests_expire():
    clock = FakeClock()
    limiter = SlidingWindowLimiter([(2, 60)], clock=clock)
    limiter.check("a")
    clock.now += 30
    limiter.check("a")
    assert limiter.check("a") > 0
    clock.now += 31  # the first request is now over 60s old
    assert limiter.check("a") == 0


def test_blocked_requests_are_not_counted():
    clock = FakeClock()
    limiter = SlidingWindowLimiter([(1, 60)], clock=clock)
    limiter.check("a")
    for _ in range(5):
        limiter.check("a")
    clock.now += 61
    assert limiter.check("a") == 0


def test_keys_are_independent():
    limiter = SlidingWindowLimiter([(1, 60)], clock=FakeClock())
    assert limiter.check("a") == 0
    assert limiter.check("b") == 0
    assert limiter.check("a") > 0


def test_longer_window_applies_after_short_one_resets():
    clock = FakeClock()
    limiter = SlidingWindowLimiter([(2, 60), (3, 3600)], clock=clock)
    for _ in range(2):
        limiter.check("a")
    clock.now += 61
    assert limiter.check("a") == 0  # third request this hour
    clock.now += 61
    wait = limiter.check("a")      # fourth: the hourly limit kicks in
    assert wait > 60
