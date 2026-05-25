"""
Deterministic 5-minute market clock.

BTC UP/DOWN 5M slugs are 100% deterministic from Unix time:
  slug = f"btc-updown-5m-{window_start}"
  window_start = now - (now % 300)   # floor to 300s boundary
  window_end   = window_start + 300

No API search needed to compute slugs — only needed to get token IDs.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import List

CYCLE_SECONDS = 300   # 5 minutes exactly
SLUG_PREFIX   = "btc-updown-5m"


@dataclass(frozen=True)
class RoundWindow:
    """Represents one 5-minute market window."""
    window_ts:   int     # Unix timestamp (start, divisible by 300)
    slug:        str     # "btc-updown-5m-{window_ts}"
    start_time:  int     # same as window_ts
    end_time:    int     # window_ts + 300
    round_number: int    # sequential round index (window_ts // 300)

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.end_time - time.time())

    @property
    def time_elapsed(self) -> float:
        return max(0.0, time.time() - self.start_time)

    @property
    def pct_elapsed(self) -> float:
        elapsed = time.time() - self.start_time
        return min(1.0, max(0.0, elapsed / CYCLE_SECONDS))

    @property
    def is_active(self) -> bool:
        now = time.time()
        return self.start_time <= now < self.end_time

    @property
    def is_future(self) -> bool:
        return time.time() < self.start_time

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.end_time

    @property
    def is_near_expiry(self) -> bool:
        return 0 < self.time_remaining < 60

    def __str__(self) -> str:
        tr = self.time_remaining
        m, s = divmod(int(tr), 60)
        return f"{self.slug}  [{m:01d}m{s:02d}s left]"


def current_window(server_offset: float = 0.0) -> RoundWindow:
    """Compute the currently active 5-minute window."""
    now = int(time.time() + server_offset)
    wts = now - (now % CYCLE_SECONDS)
    return _make_window(wts)


def next_window(n: int = 1, server_offset: float = 0.0) -> RoundWindow:
    """Compute the Nth next window (n=1 = next round)."""
    now = int(time.time() + server_offset)
    wts = now - (now % CYCLE_SECONDS) + n * CYCLE_SECONDS
    return _make_window(wts)


def window_schedule(
    count: int = 4,
    server_offset: float = 0.0,
) -> List[RoundWindow]:
    """
    Return [current, +1, +2, ..., +count-1] windows.
    Used for multi-round pre-subscription.
    """
    now = int(time.time() + server_offset)
    base = now - (now % CYCLE_SECONDS)
    return [_make_window(base + i * CYCLE_SECONDS) for i in range(count)]


def window_from_slug(slug: str) -> RoundWindow:
    """Parse a RoundWindow from a known slug."""
    # slug format: "btc-updown-5m-1779673200"
    parts = slug.split("-")
    ts    = int(parts[-1])
    return _make_window(ts)


def _make_window(wts: int) -> RoundWindow:
    return RoundWindow(
        window_ts=wts,
        slug=f"{SLUG_PREFIX}-{wts}",
        start_time=wts,
        end_time=wts + CYCLE_SECONDS,
        round_number=wts // CYCLE_SECONDS,
    )
