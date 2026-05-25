from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, Generic, List, Optional, Sequence, Tuple, TypeVar

T = TypeVar("T")


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw in (None, ""):
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return default


def _utc_timestamp(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str) and value:
        text = value.strip()
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
        except ValueError:
            try:
                return float(text)
            except ValueError:
                return None
    return None


class ServiceState(str, Enum):
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    CONNECTED = "CONNECTED"
    DEGRADED = "DEGRADED"
    STOPPING = "STOPPING"


class RoundPhase(str, Enum):
    DISCOVERY = "DISCOVERY"
    ACTIVE = "ACTIVE"
    AWAITING_SETTLEMENT = "AWAITING_SETTLEMENT"
    SETTLED = "SETTLED"


class Outcome(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class PriceSnapshot:
    source: str
    price: float
    timestamp: float
    bid: float = 0.0
    ask: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.timestamp)


@dataclass(frozen=True)
class ChainlinkData:
    price: float
    updated_at: float
    round_id: int
    lag_seconds: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderBookLevel:
    price: float
    size: float


@dataclass(frozen=True)
class OrderBookSnapshot:
    bids: Tuple[OrderBookLevel, ...]
    asks: Tuple[OrderBookLevel, ...]
    timestamp: float
    asset_id: str = ""
    book_hash: str = ""

    @property
    def best_bid(self) -> float:
        return self.bids[0].price if self.bids else 0.0

    @property
    def best_ask(self) -> float:
        return self.asks[0].price if self.asks else 0.0

    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.best_bid + self.best_ask) * 0.5
        return 0.0

    @property
    def spread(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        return max(0.0, self.best_ask - self.best_bid)

    @property
    def total_bid_size(self) -> float:
        return sum(level.size for level in self.bids)

    @property
    def total_ask_size(self) -> float:
        return sum(level.size for level in self.asks)

    @property
    def age_s(self) -> float:
        return max(0.0, time.time() - self.timestamp)


@dataclass(frozen=True)
class MarketInfo:
    slug: str
    condition_id: str
    question: str
    up_token_id: str
    down_token_id: str
    price_to_beat: float
    end_time: float
    event_start_time: float
    tick_size: str = "0.01"
    neg_risk: bool = False
    min_order_size: float = 5.0
    active: bool = True
    fees_enabled: bool = True
    fee_rate: float = 0.0
    fee_exponent: float = 1.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def time_remaining_s(self) -> float:
        return max(0.0, self.end_time - time.time())

    @property
    def elapsed_s(self) -> float:
        return max(0.0, time.time() - self.event_start_time)

    @property
    def is_expired(self) -> bool:
        return self.time_remaining_s <= 0.0

    def with_price_to_beat(self, value: float) -> "MarketInfo":
        return replace(self, price_to_beat=float(value))


@dataclass(frozen=True)
class PTBRecord:
    slug: str
    price: float
    captured_at: float
    source: str
    event_start_time: float
    source_timestamp: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return self.price > 0.0


@dataclass(frozen=True)
class MarketResolution:
    resolved: bool = False
    outcome: Outcome = Outcome.UNKNOWN
    winner_token_id: str = ""
    up_payout: Optional[float] = None
    down_payout: Optional[float] = None
    payout_numerators: Tuple[int, ...] = ()
    payout_denominator: Optional[int] = None
    source: str = ""
    confirmations: int = 0
    onchain_verified: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_known(self) -> bool:
        return self.resolved and self.outcome in (Outcome.UP, Outcome.DOWN)


@dataclass(frozen=True)
class WalletSnapshot:
    asset_type: str
    balance: float
    allowance: float
    token_id: str = ""
    synced_at: float = 0.0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_s(self) -> float:
        if self.synced_at <= 0:
            return float("inf")
        return max(0.0, time.time() - self.synced_at)


@dataclass(frozen=True)
class ExecutionReadiness:
    live: bool
    collateral: Optional[WalletSnapshot]
    conditionals: Tuple[WalletSnapshot, ...]
    approvals_ready: bool
    blockers: Tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return self.live and self.approvals_ready and not self.blockers


@dataclass(frozen=True)
class ComponentHealth:
    name: str
    state: ServiceState
    connected: bool
    last_rx: float = 0.0
    last_tx: float = 0.0
    reconnects: int = 0
    stale_after_s: float = 0.0
    last_error: str = ""

    @property
    def age_s(self) -> float:
        if self.last_rx <= 0:
            return float("inf")
        return max(0.0, time.time() - self.last_rx)

    @property
    def stale(self) -> bool:
        return self.stale_after_s > 0 and self.age_s > self.stale_after_s


class RingBuffer(Generic[T]):
    def __init__(self, maxlen: int):
        self._items: Deque[T] = deque(maxlen=max(1, int(maxlen)))

    def append(self, item: T) -> None:
        self._items.append(item)

    def clear(self) -> None:
        self._items.clear()

    @property
    def last(self) -> Optional[T]:
        return self._items[-1] if self._items else None

    def snapshot(self) -> Tuple[T, ...]:
        return tuple(self._items)

    def __len__(self) -> int:
        return len(self._items)


@dataclass(frozen=True)
class MarketTruthState:
    phase: RoundPhase
    current_market: Optional[MarketInfo] = None
    ptb: Optional[PTBRecord] = None
    settlement: Optional[MarketResolution] = None
    binance: Optional[PriceSnapshot] = None
    coinbase: Optional[PriceSnapshot] = None
    chainlink: Optional[ChainlinkData] = None
    books: Dict[str, OrderBookSnapshot] = field(default_factory=dict)
    wallet: Optional[ExecutionReadiness] = None
    updated_at: float = 0.0
