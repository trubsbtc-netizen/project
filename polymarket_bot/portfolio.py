"""
Position and PnL tracking for Polymarket binary outcome positions.

The tracker is deliberately local and conservative. It only records positions
after the execution layer reports a fill, marks open risk from executable bid
prices when available, and realizes PnL once the 5-minute market window has
expired.

Settlement states:
  - "active"            : Round is still live (now < market_end_timestamp).
  - "awaiting_settlement": Round has ended, background task is polling for
                           official settlement or a PTB-based outcome.
  - "settled"           : Final outcome has been recorded and position has been
                           moved from _positions to RealizedPosition.
"""

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from polymarket_bot.config import Direction, RegimeState
from polymarket_bot.bot_types import PolymarketOrderbook, TradeDecision


@dataclass
class OpenPosition:
    """A filled long position in one Polymarket outcome token."""

    position_id: str
    market_slug: str
    condition_id: str
    token_id: str
    direction: Direction
    shares: float
    cost: float
    entry_price: float
    entry_fee: float
    entry_timestamp: float
    price_to_beat: Optional[float]
    market_start_timestamp: Optional[float]
    market_end_timestamp: Optional[float]
    confidence_at_entry: float
    edge_at_entry_bps: float
    regime_at_entry: RegimeState
    p_up_at_entry: float
    order_id: str = ""
    is_dry_run: bool = True
    last_mark_price: float = 0.0
    last_mark_timestamp: float = 0.0
    unrealized_pnl: float = 0.0
    settlement_state: str = "active"

    @property
    def notional_value(self) -> float:
        return self.shares * self.last_mark_price

    @property
    def is_awaiting_settlement(self) -> bool:
        return self.settlement_state == "awaiting_settlement"

    @property
    def is_active_round(self) -> bool:
        """True while the round is still live and the position is trading."""
        if self.settlement_state == "awaiting_settlement":
            return False
        if self.settlement_state == "settled":
            return False
        if self.market_end_timestamp is None:
            return True
        return time.time() < self.market_end_timestamp


@dataclass
class PositionSnapshot:
    """Aggregate mark-to-market state for all open positions."""

    timestamp: float
    open_count: int
    total_cost: float
    total_shares: float
    market_value: float
    unrealized_pnl: float
    current_direction: Direction


@dataclass
class RealizedPosition:
    """A settled position result ready for risk/accounting updates."""

    position_id: str
    market_slug: str
    condition_id: str
    token_id: str
    direction: Direction
    shares: float
    cost: float
    entry_price: float
    entry_fee: float
    exit_price: float
    payout: float
    pnl: float
    final_btc_price: float
    price_to_beat: float
    actual_outcome: Direction
    confidence_at_entry: float
    edge_at_entry_bps: float
    regime_at_entry: RegimeState
    p_up_at_entry: float
    order_id: str
    entry_timestamp: float
    market_start_timestamp: Optional[float]
    market_end_timestamp: Optional[float]
    is_dry_run: bool
    is_win: bool


class PositionTracker:
    """Track filled positions, mark unrealized PnL, and settle expired rounds."""

    def __init__(self) -> None:
        self._positions: Dict[str, OpenPosition] = {}
        self._sequence = 0

    @staticmethod
    def window_from_slug(slug: str) -> Tuple[Optional[float], Optional[float]]:
        prefix = "btc-updown-5m-"
        if not slug.startswith(prefix):
            return None, None
        try:
            start = float(int(slug[len(prefix):]))
        except ValueError:
            return None, None
        return start, start + 300.0

    @staticmethod
    def _finite_positive(value: Optional[float]) -> bool:
        return value is not None and value > 0.0 and value < float("inf")

    @staticmethod
    def _is_active_position(
        position: OpenPosition,
        now: Optional[float] = None,
    ) -> bool:
        """True only when the round is live AND position is still trading.

        Positions in 'awaiting_settlement' or 'settled' state are excluded
        even if market_end_timestamp has not yet passed.
        """
        if position.settlement_state != "active":
            return False
        if position.market_end_timestamp is None:
            return True
        now = time.time() if now is None else now
        return now < position.market_end_timestamp

    @staticmethod
    def _mark_price(
        orderbook: Optional[PolymarketOrderbook],
        fallback_price: float,
    ) -> float:
        if orderbook is not None:
            if orderbook.best_bid is not None and orderbook.best_bid > 0.0:
                return float(orderbook.best_bid)
            if orderbook.mid_price is not None and orderbook.mid_price > 0.0:
                return float(orderbook.mid_price)
        return float(max(0.0, min(1.0, fallback_price)))

    def has_open_position_for_market(self, market_slug: str) -> bool:
        """True if there is an *active* (trading) position for this round.

        Positions that have transitioned to 'awaiting_settlement' are excluded
        so they do not block entry into a new round.
        """
        if not market_slug:
            return False
        return any(
            position.market_slug == market_slug
            and position.settlement_state == "active"
            for position in self._positions.values()
        )

    def get_position(self, position_id: str) -> Optional[OpenPosition]:
        return self._positions.get(position_id)

    def mark_position_awaiting_settlement(self, position_id: str) -> bool:
        """Transition a position from 'active' to 'awaiting_settlement'.

        Called when the round ends and settlement polling begins.
        Returns True if the transition was made.
        """
        position = self._positions.get(position_id)
        if position is None:
            return False
        if position.settlement_state == "active":
            position.settlement_state = "awaiting_settlement"
            return True
        return False

    def active_positions(self) -> List[OpenPosition]:
        """Return positions that are still in a live round (trading)."""
        return [
            p for p in self._positions.values()
            if p.settlement_state == "active"
        ]

    def awaiting_settlement_positions(self) -> List[OpenPosition]:
        """Return positions whose rounds have ended but settlement is pending."""
        return [
            p for p in self._positions.values()
            if p.settlement_state == "awaiting_settlement"
        ]

    def add_filled_buy(
        self,
        *,
        decision: TradeDecision,
        market_slug: str,
        price_to_beat: Optional[float],
        condition_id: str = "",
        order_id: str = "",
        filled_price: Optional[float] = None,
        filled_shares: Optional[float] = None,
        filled_notional: Optional[float] = None,
        filled_fee: Optional[float] = None,
        is_dry_run: bool = True,
    ) -> Optional[OpenPosition]:
        """Record a filled BUY as an open long outcome-token position."""
        if self.has_open_position_for_market(market_slug):
            return None

        entry_price = (
            float(filled_price)
            if self._finite_positive(filled_price)
            else float(decision.execution_estimate.limit_price or 0.0)
        )
        if entry_price <= 0.0:
            entry_price = float(decision.execution_estimate.effective_price or 0.0)
        if entry_price <= 0.0:
            entry_price = float(decision.market_price or 0.0)
        entry_price = float(max(0.01, min(0.99, entry_price)))

        trade_notional = (
            float(filled_notional)
            if self._finite_positive(filled_notional)
            else float(decision.position_size)
        )
        entry_fee = max(
            0.0,
            float(filled_fee) if filled_fee is not None and filled_fee < float("inf") else 0.0,
        )
        shares = (
            float(filled_shares)
            if self._finite_positive(filled_shares)
            else trade_notional / entry_price
        )
        cost = trade_notional + entry_fee
        if cost <= 0.0 or shares <= 0.0:
            return None

        self._sequence += 1
        market_start, market_end = self.window_from_slug(market_slug)
        position_id = f"{market_slug}:{decision.token_id}:{self._sequence}"
        position = OpenPosition(
            position_id=position_id,
            market_slug=market_slug,
            condition_id=condition_id,
            token_id=decision.token_id,
            direction=decision.direction,
            shares=shares,
            cost=cost,
            entry_price=entry_price,
            entry_fee=entry_fee,
            entry_timestamp=time.time(),
            price_to_beat=price_to_beat,
            market_start_timestamp=market_start,
            market_end_timestamp=market_end,
            confidence_at_entry=decision.confidence,
            edge_at_entry_bps=decision.edge_bps,
            regime_at_entry=decision.regime_state,
            p_up_at_entry=decision.settlement_forecast.p_up_settlement,
            order_id=order_id,
            is_dry_run=is_dry_run,
            last_mark_price=entry_price,
            last_mark_timestamp=time.time(),
            unrealized_pnl=0.0,
        )
        self._positions[position_id] = position
        return position

    def mark_to_market(
        self,
        orderbooks_by_token: Dict[str, Optional[PolymarketOrderbook]],
    ) -> PositionSnapshot:
        """Update and return aggregate unrealized PnL from current books."""
        now = time.time()
        total_cost = 0.0
        total_shares = 0.0
        market_value = 0.0
        active_count = 0
        directions = set()

        for position in self._positions.values():
            if not self._is_active_position(position, now):
                continue
            active_count += 1
            fallback = position.last_mark_price or position.entry_price
            orderbook = orderbooks_by_token.get(position.token_id)
            mark_price = self._mark_price(orderbook, fallback)
            position.last_mark_price = mark_price
            position.last_mark_timestamp = now
            position.unrealized_pnl = position.shares * mark_price - position.cost
            total_cost += position.cost
            total_shares += position.shares
            market_value += position.shares * mark_price
            directions.add(position.direction)

        current_direction = Direction.NEUTRAL
        if len(directions) == 1:
            current_direction = next(iter(directions))

        return PositionSnapshot(
            timestamp=now,
            open_count=active_count,
            total_cost=total_cost,
            total_shares=total_shares,
            market_value=market_value,
            unrealized_pnl=market_value - total_cost,
            current_direction=current_direction,
        )

    def positions_due_for_settlement(
        self,
        now: Optional[float] = None,
        grace_seconds: float = 1.0,
    ) -> List[OpenPosition]:
        now = time.time() if now is None else now
        due = []
        for position in self._positions.values():
            if position.market_end_timestamp is None:
                continue
            if now >= position.market_end_timestamp + grace_seconds:
                due.append(position)
        return due

    def settle_position(
        self,
        position_id: str,
        final_btc_price: float,
        actual_outcome: Optional[Direction] = None,
    ) -> Optional[RealizedPosition]:
        position = self._positions.get(position_id)
        if position is None:
            return None
        if position.price_to_beat is None or position.price_to_beat <= 0.0:
            return None

        if actual_outcome not in {Direction.UP, Direction.DOWN}:
            actual_outcome = (
                Direction.UP
                if final_btc_price >= position.price_to_beat
                else Direction.DOWN
            )
        exit_price = 1.0 if position.direction == actual_outcome else 0.0
        payout = position.shares * exit_price
        pnl = payout - position.cost

        self._positions.pop(position_id, None)
        return RealizedPosition(
            position_id=position.position_id,
            market_slug=position.market_slug,
            condition_id=position.condition_id,
            token_id=position.token_id,
            direction=position.direction,
            shares=position.shares,
            cost=position.cost,
            entry_price=position.entry_price,
            entry_fee=position.entry_fee,
            exit_price=exit_price,
            payout=payout,
            pnl=pnl,
            final_btc_price=final_btc_price,
            price_to_beat=position.price_to_beat,
            actual_outcome=actual_outcome,
            confidence_at_entry=position.confidence_at_entry,
            edge_at_entry_bps=position.edge_at_entry_bps,
            regime_at_entry=position.regime_at_entry,
            p_up_at_entry=position.p_up_at_entry,
            order_id=position.order_id,
            entry_timestamp=position.entry_timestamp,
            market_start_timestamp=position.market_start_timestamp,
            market_end_timestamp=position.market_end_timestamp,
            is_dry_run=position.is_dry_run,
            is_win=pnl > 0.0,
        )

    def snapshot(self) -> PositionSnapshot:
        """Return aggregate state without changing mark prices."""
        now = time.time()
        active_positions = [
            position
            for position in self._positions.values()
            if self._is_active_position(position, now)
        ]
        total_cost = sum(position.cost for position in active_positions)
        total_shares = sum(position.shares for position in active_positions)
        market_value = sum(position.notional_value for position in active_positions)
        directions = {position.direction for position in active_positions}
        current_direction = Direction.NEUTRAL
        if len(directions) == 1:
            current_direction = next(iter(directions))
        return PositionSnapshot(
            timestamp=time.time(),
            open_count=len(active_positions),
            total_cost=total_cost,
            total_shares=total_shares,
            market_value=market_value,
            unrealized_pnl=market_value - total_cost,
            current_direction=current_direction,
        )

    def open_positions(self) -> List[OpenPosition]:
        return list(self._positions.values())
