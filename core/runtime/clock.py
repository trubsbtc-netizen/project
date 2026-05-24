from __future__ import annotations

import time


def mono_ns() -> int:
    return time.monotonic_ns()


def unix_ms() -> int:
    return time.time_ns() // 1_000_000


def unix_s() -> float:
    return time.time()


def bucket_5m(server_time_s: float | None = None) -> int:
    t = int(time.time() if server_time_s is None else server_time_s)
    return (t // 300) * 300


def slug_for_bucket(bucket: int) -> str:
    return f"btc-updown-5m-{bucket}"

