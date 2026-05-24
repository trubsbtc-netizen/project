from __future__ import annotations

from dataclasses import dataclass

from core.runtime.clock import mono_ns
from core.types import FeedKind, FeedStatus


@dataclass(slots=True)
class FeedHeartbeat:
    name: str
    kind: FeedKind
    stale_after_ns: int
    connected: bool = False
    last_msg_mono_ns: int = 0
    reconnects: int = 0
    errors: int = 0

    def mark_connected(self) -> None:
        self.connected = True
        self.last_msg_mono_ns = mono_ns()

    def mark_disconnected(self) -> None:
        self.connected = False

    def mark_message(self, ts_ns: int | None = None) -> None:
        self.connected = True
        self.last_msg_mono_ns = mono_ns() if ts_ns is None else ts_ns

    def mark_reconnect(self) -> None:
        self.reconnects += 1
        self.connected = False

    def mark_error(self) -> None:
        self.errors += 1

    def status(self) -> FeedStatus:
        return FeedStatus(
            name=self.name,
            kind=self.kind,
            connected=self.connected,
            last_msg_mono_ns=self.last_msg_mono_ns,
            stale_after_ns=self.stale_after_ns,
            reconnects=self.reconnects,
            errors=self.errors,
        )


class HealthMonitor:
    def __init__(self) -> None:
        self._feeds: dict[str, FeedHeartbeat] = {}

    def register(self, name: str, kind: FeedKind, stale_after_s: float) -> FeedHeartbeat:
        hb = FeedHeartbeat(name=name, kind=kind, stale_after_ns=int(stale_after_s * 1e9))
        self._feeds[name] = hb
        return hb

    def status(self) -> dict[str, FeedStatus]:
        return {name: hb.status() for name, hb in self._feeds.items()}

    def stale_penalty(self) -> float:
        now = mono_ns()
        if not self._feeds:
            return 1.0
        stale = sum(1 for hb in self._feeds.values() if hb.status().stale(now))
        return min(1.0, stale / max(1, len(self._feeds)))

