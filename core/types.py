from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Direction(str, Enum):
    UP = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"

    @property
    def sign(self) -> int:
        if self is Direction.UP:
            return 1
        if self is Direction.DOWN:
            return -1
        return 0

    @staticmethod
    def from_sign(value: float, eps: float = 0.0) -> "Direction":
        if value > eps:
            return Direction.UP
        if value < -eps:
            return Direction.DOWN
        return Direction.FLAT


class FeedKind(str, Enum):
    CHAINLINK = "chainlink"
    POLYMARKET = "polymarket"
    BINANCE = "binance"
    COINBASE = "coinbase"
    USER = "user"
    SETTLEMENT = "settlement"


class TruthSource(str, Enum):
    RTDS_CHAINLINK = "rtds_chainlink"
    PTB = "ptb"
    OFFICIAL_SETTLEMENT = "official_settlement"
    PAYOUT_NUMERATORS = "payout_numerators"


class SignalSource(str, Enum):
    BINANCE = "binance"
    COINBASE = "coinbase"
    POLYMARKET_BOOK = "polymarket_book"
    POLYMARKET_TRADES = "polymarket_trades"
    RTDS_BINANCE = "rtds_binance"


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Outcome(str, Enum):
    UP = "UP"
    DOWN = "DOWN"


@dataclass(slots=True, frozen=True)
class PriceTick:
    source: SignalSource | TruthSource
    symbol: str
    price: float
    exchange_ts_ms: int
    recv_mono_ns: int
    quality: float = 1.0
    raw: Mapping[str, Any] | None = None


@dataclass(slots=True, frozen=True)
class TradeTick:
    source: SignalSource
    symbol: str
    price: float
    size: float
    side: OrderSide
    exchange_ts_ms: int
    recv_mono_ns: int
    trade_id: str = ""

    @property
    def signed_size(self) -> float:
        return self.size if self.side is OrderSide.BUY else -self.size


@dataclass(slots=True, frozen=True)
class TopOfBook:
    source: SignalSource
    symbol: str
    bid: float
    ask: float
    bid_size: float
    ask_size: float
    exchange_ts_ms: int
    recv_mono_ns: int

    @property
    def mid(self) -> float:
        if self.bid > 0.0 and self.ask > 0.0:
            return 0.5 * (self.bid + self.ask)
        return max(self.bid, self.ask)

    @property
    def spread(self) -> float:
        return max(0.0, self.ask - self.bid)


@dataclass(slots=True, frozen=True)
class OutcomeToken:
    outcome: Outcome
    token_id: str


@dataclass(slots=True, frozen=True)
class PriceToBeat:
    source: TruthSource
    value: float
    timestamp_ms: int


@dataclass(slots=True, frozen=True)
class PTBMarket:
    bucket: int
    slug: str
    condition_id: str
    market_id: str
    up: OutcomeToken
    down: OutcomeToken
    price_to_beat: PriceToBeat
    open_ts: int
    close_ts: int
    fetched_mono_ns: int
    raw_hash: str
    immutable: bool = True

    @property
    def token_ids(self) -> tuple[str, str]:
        return (self.up.token_id, self.down.token_id)

    def token_for(self, outcome: Outcome) -> str:
        return self.up.token_id if outcome is Outcome.UP else self.down.token_id


@dataclass(slots=True, frozen=True)
class SettlementTruth:
    condition_id: str
    payout_numerators: tuple[int, int]
    payout_denominator: int
    verified_block: int
    source: TruthSource = TruthSource.PAYOUT_NUMERATORS

    @property
    def resolved(self) -> bool:
        return self.payout_denominator > 0 and any(v > 0 for v in self.payout_numerators)

    @property
    def winner(self) -> Outcome | None:
        if not self.resolved:
            return None
        if self.payout_numerators[0] > self.payout_numerators[1]:
            return Outcome.UP
        if self.payout_numerators[1] > self.payout_numerators[0]:
            return Outcome.DOWN
        return None


@dataclass(slots=True, frozen=True)
class FeedStatus:
    name: str
    kind: FeedKind
    connected: bool
    last_msg_mono_ns: int
    stale_after_ns: int
    reconnects: int = 0
    errors: int = 0

    def stale(self, now_mono_ns: int) -> bool:
        return (not self.connected) or (now_mono_ns - self.last_msg_mono_ns > self.stale_after_ns)


@dataclass(slots=True, frozen=True)
class MicrostructureSnapshot:
    ts_mono_ns: int
    seconds_to_expiry: float
    truth_price: float
    price_to_beat: float
    signed_distance: float
    signal_mid: float
    exchange_divergence: float
    realized_vol: float
    jump_intensity: float
    drift: float
    drift_uncertainty: float
    taker_aggression: float
    flow_acceleration: float
    book_pressure: float
    spoof_resistant_imbalance: float
    queue_survival: float
    liquidity_stability: float
    spread_stability: float
    absorption_up: float
    absorption_down: float
    exhaustion: float
    burst_failure: float
    liquidity_vacuum: float
    entropy: float
    regime_trend: float
    regime_volatility: float
    stale_penalty: float


@dataclass(slots=True, frozen=True)
class ProbabilityEstimate:
    ts_mono_ns: int
    p_up: float
    p_down: float
    confidence: float
    posterior_uncertainty: float
    continuation_probability: float
    reversal_probability: float
    settlement_probability: float
    entropy: float
    direction: Direction
    edge: float
    fair_price_up: float
    fair_price_down: float
    diagnostics: Mapping[str, float] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class OrderIntent:
    bucket: int
    market: str
    condition_id: str
    outcome: Outcome
    token_id: str
    side: OrderSide
    limit_price: float
    size: float
    max_slippage_bps: float
    ttl_ms: int
    reason: str
    estimate: ProbabilityEstimate


@dataclass(slots=True, frozen=True)
class OrderAck:
    accepted: bool
    order_id: str
    status: str
    message: str
    recv_mono_ns: int


@dataclass(slots=True, frozen=True)
class PositionState:
    bucket: int
    up_size: float = 0.0
    down_size: float = 0.0
    avg_up: float = 0.0
    avg_down: float = 0.0

    @property
    def net_direction(self) -> Direction:
        return Direction.from_sign(self.up_size - self.down_size)

    @property
    def gross_notional(self) -> float:
        return self.up_size * self.avg_up + self.down_size * self.avg_down


@dataclass(slots=True, frozen=True)
class RuntimeEvent:
    source: FeedKind
    payload: Any
    recv_mono_ns: int

