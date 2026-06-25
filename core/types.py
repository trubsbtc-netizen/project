"""
Core type definitions for Polymarket BTC UP/DOWN 5M trading bot.
All types are immutable dataclasses with strict field validation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple


# ─────────────────────────── Enums ───────────────────────────

class Side(Enum):
    BUY  = "BUY"
    SELL = "SELL"


class Direction(Enum):
    UP   = "UP"
    DOWN = "DOWN"
    FLAT = "FLAT"


class OrderType(Enum):
    GTC = "GTC"
    GTD = "GTD"
    FOK = "FOK"
    FAK = "FAK"


class OrderStatus(Enum):
    LIVE      = "live"
    MATCHED   = "matched"
    DELAYED   = "delayed"
    UNMATCHED = "unmatched"
    CANCELLED = "cancelled"
    FAILED    = "failed"


class TradeStatus(Enum):
    MATCHED   = "MATCHED"
    MINED     = "MINED"
    CONFIRMED = "CONFIRMED"
    RETRYING  = "RETRYING"
    FAILED    = "FAILED"


class SignalStrength(Enum):
    NONE    = 0
    WEAK    = 1
    MEDIUM  = 2
    STRONG  = 3
    EXTREME = 4


class CircuitState(Enum):
    CLOSED   = auto()   # Normal operating state
    OPEN     = auto()   # Circuit breaker tripped — no trading
    HALF_OPEN = auto()  # Recovery probe state


class MarketPhase(Enum):
    WAITING      = auto()   # Between cycles
    ACTIVE       = auto()   # Cycle running — trading allowed
    NEAR_EXPIRY  = auto()   # < 60s to resolution — restricted
    RESOLVED     = auto()   # Market resolved — transition
    TRANSITIONING = auto()  # Finding next market


class ConnectionState(Enum):
    DISCONNECTED  = auto()
    CONNECTING    = auto()
    CONNECTED     = auto()
    RECONNECTING  = auto()
    FAILED        = auto()


# ─────────────────────────── Time Window Mixin ───────────────────────────

class TimeWindowMixin:
    """Shared time-window properties for classes with start_time / end_time."""
    start_time: int
    end_time:   int

    @property
    def time_remaining(self) -> float:
        return max(0.0, self.end_time - time.time())

    @property
    def pct_elapsed(self) -> float:
        total = self.end_time - self.start_time
        if total <= 0:
            return 1.0
        elapsed = time.time() - self.start_time
        return min(1.0, max(0.0, elapsed / total))

    @property
    def is_near_expiry(self) -> bool:
        return 0 < self.time_remaining < 60.0

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.end_time


# ─────────────────────────── Price Levels ───────────────────────────

@dataclass(frozen=True)
class PriceLevel:
    price: Decimal
    size:  Decimal

    def __post_init__(self) -> None:
        if self.price < Decimal("0") or self.price > Decimal("1"):
            raise ValueError(f"Invalid price {self.price}: must be in [0,1]")
        if self.size < Decimal("0"):
            raise ValueError(f"Invalid size {self.size}: must be >= 0")


# ─────────────────────────── Orderbook State ───────────────────────────

@dataclass
class OrderbookState:
    asset_id:   str
    market:     str
    bids:       Dict[Decimal, Decimal]  # price -> size
    asks:       Dict[Decimal, Decimal]  # price -> size
    timestamp:  int                     # unix ms
    hash:       str                     # server-side hash for desync detection
    last_trade_price: Optional[Decimal] = None
    last_trade_side:  Optional[Side]    = None
    sequence:         int = 0           # local sequence counter for ordering

    @property
    def best_bid(self) -> Optional[Decimal]:
        return max(self.bids.keys()) if self.bids else None

    @property
    def best_ask(self) -> Optional[Decimal]:
        return min(self.asks.keys()) if self.asks else None

    @property
    def spread(self) -> Optional[Decimal]:
        bb = self.best_bid
        ba = self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    @property
    def midprice(self) -> Optional[Decimal]:
        bb = self.best_bid
        ba = self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / Decimal("2")

    def microprice(self) -> Optional[Decimal]:
        """
        Microprice = volume-weighted midprice.
        Formula: (ask_vol * best_bid + bid_vol * best_ask) / (bid_vol + ask_vol)
        This is statistically superior to simple midprice as it weights
        toward the side with less liquidity (where price is more likely to move).
        """
        bb = self.best_bid
        ba = self.best_ask
        if bb is None or ba is None:
            return None

        bid_vol = self.bids.get(bb, Decimal("0"))
        ask_vol = self.asks.get(ba, Decimal("0"))
        total = bid_vol + ask_vol
        if total == Decimal("0"):
            return self.midprice

        return (ask_vol * bb + bid_vol * ba) / total

    def bid_depth(self, levels: int = 5) -> Decimal:
        """Total size across top N bid levels."""
        sorted_bids = sorted(self.bids.keys(), reverse=True)
        return sum(self.bids[p] for p in sorted_bids[:levels])

    def ask_depth(self, levels: int = 5) -> Decimal:
        """Total size across top N ask levels."""
        sorted_asks = sorted(self.asks.keys())
        return sum(self.asks[p] for p in sorted_asks[:levels])

    def is_empty(self) -> bool:
        return len(self.bids) == 0 and len(self.asks) == 0

    def clone(self) -> "OrderbookState":
        return OrderbookState(
            asset_id=self.asset_id,
            market=self.market,
            bids=dict(self.bids),
            asks=dict(self.asks),
            timestamp=self.timestamp,
            hash=self.hash,
            last_trade_price=self.last_trade_price,
            last_trade_side=self.last_trade_side,
            sequence=self.sequence,
        )


# ─────────────────────────── Market Info ───────────────────────────

@dataclass
class MarketTokenPair(TimeWindowMixin):
    """Represents the UP/DOWN token pair for a BTC 5M market."""
    condition_id: str
    up_token_id:  str
    down_token_id: str
    tick_size:    str          # e.g. "0.01"
    neg_risk:     bool         # always True for UP/DOWN markets
    start_time:   int          # unix timestamp
    end_time:     int          # unix timestamp (start + 300 seconds)
    question:     str
    slug:         str


# ─────────────────────────── Signal / Probability ───────────────────────────

@dataclass
class OrderFlowMetrics:
    """Computed order flow metrics from a single orderbook."""
    asset_id:         str
    direction:        Direction
    microprice:       Optional[Decimal]
    midprice:         Optional[Decimal]
    spread:           Optional[Decimal]
    bid_depth_5:      Decimal            # Top 5 bid levels total
    ask_depth_5:      Decimal            # Top 5 ask levels total
    depth_imbalance:  Decimal            # (bid - ask) / (bid + ask), range [-1, 1]
    ofi:              Decimal            # Order flow imbalance (running)
    taker_buy_vol:    Decimal            # Aggressive buy volume (running)
    taker_sell_vol:   Decimal            # Aggressive sell volume (running)
    sweep_detected:   bool               # Recent aggressive sweep
    spoof_probability: Decimal           # Estimated probability of spoofed orders
    timestamp:        int


@dataclass
class UnifiedSignal:
    """
    Combined probabilistic signal from BOTH UP and DOWN orderbooks.
    This is the primary input to the entry decision logic.
    """
    # Token IDs
    up_asset_id:   str
    down_asset_id: str

    # Microprices (probability estimates from book structure)
    up_microprice:   Optional[Decimal]
    down_microprice: Optional[Decimal]

    # The unified probability of UP outcome (0.0 - 1.0)
    # Derived from: microprice, OFI, depth ratio, price acceleration
    up_probability:   Decimal
    down_probability: Decimal

    # Confidence in the directional call (0.0 - 1.0)
    confidence:       Decimal

    # Directional call
    direction:        Direction
    signal_strength:  SignalStrength

    # Component scores (each 0.0 - 1.0, can be negative for opposing direction)
    microprice_score:     Decimal   # Microprice deviation from 0.5
    ofi_score:            Decimal   # Order flow imbalance score
    depth_ratio_score:    Decimal   # Depth asymmetry score
    velocity_score:       Decimal   # Price change velocity score
    absorption_score:     Decimal   # Liquidity absorption score

    # Market conditions
    up_spread:     Optional[Decimal]
    down_spread:   Optional[Decimal]
    liquidity_ok:  bool
    spread_ok:     bool

    # Risk flags
    manipulation_flag:   bool
    liquidity_vacuum:    bool
    violent_reversal:    bool

    timestamp: int

    @property
    def is_tradeable(self) -> bool:
        """
        Returns True only when ALL conditions for a safe entry are met.
        This is the final gate before execution.
        """
        return (
            self.signal_strength.value >= SignalStrength.STRONG.value
            and self.confidence >= Decimal("0.65")
            and self.liquidity_ok
            and self.spread_ok
            and not self.manipulation_flag
            and not self.liquidity_vacuum
            and not self.violent_reversal
            and self.direction != Direction.FLAT
        )


# ─────────────────────────── Position / Execution ───────────────────────────

@dataclass
class Position:
    asset_id:   str
    direction:  Direction
    side:       Side
    entry_price: Decimal
    size:        Decimal          # shares
    cost_basis:  Decimal          # USDC spent
    fee_paid:    Decimal          # USDC fee at entry
    order_id:    str
    entered_at:  int              # unix ms
    condition_id: str
    market_end_time: int          # unix timestamp of market resolution

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.market_end_time

    def unrealized_pnl(self, current_price: Decimal) -> Decimal:
        """PnL if sold at current_price."""
        return (current_price - self.entry_price) * self.size - self.fee_paid

    def expected_pnl_at_resolution(self) -> Decimal:
        """PnL if market resolves in our favor (token → $1)."""
        return (Decimal("1") - self.entry_price) * self.size - self.fee_paid


@dataclass
class TradeRecord:
    order_id:    str
    asset_id:    str
    direction:   Direction
    side:        Side
    price:       Decimal
    size:        Decimal
    fee:         Decimal
    status:      TradeStatus
    timestamp:   int
    pnl:         Optional[Decimal] = None  # filled at resolution
    condition_id: str = ""


# ─────────────────────────── Risk State ───────────────────────────

@dataclass
class RiskState:
    total_exposure_usdc:    Decimal = Decimal("0")
    session_pnl_usdc:       Decimal = Decimal("0")
    session_trades:         int     = 0
    consecutive_losses:     int     = 0
    consecutive_wins:       int     = 0
    max_drawdown_usdc:      Decimal = Decimal("0")
    peak_equity_usdc:       Decimal = Decimal("0")
    current_equity_usdc:    Decimal = Decimal("0")
    circuit_state:          CircuitState = CircuitState.CLOSED
    kill_switch_active:     bool    = False
    last_trade_time:        Optional[int] = None  # unix ms
    open_positions:         int     = 0


# ─────────────────────────── Metrics Snapshot ───────────────────────────

@dataclass
class MetricsSnapshot:
    timestamp:          int
    market_phase:       MarketPhase
    risk_state:         RiskState
    up_book_depth:      Decimal
    down_book_depth:    Decimal
    up_microprice:      Optional[Decimal]
    down_microprice:    Optional[Decimal]
    signal_strength:    SignalStrength
    confidence:         Decimal
    ws_latency_ms:      float
    orders_per_minute:  float
    fill_rate:          float
