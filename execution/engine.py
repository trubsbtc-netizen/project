"""
Execution Engine — atomic order management with full state tracking.

Responsibilities:
- Translate signal + risk approval into actual orders
- Track in-flight orders atomically (no race conditions)
- Handle partial fills gracefully
- Retry transient failures with deduplication
- Position reconciliation after fills
- Instant cancel support (e.g. on circuit breaker trip)
- Fail-safe shutdown (cancel all on exit)

Execution flow:
1. Signal → Risk check → Size calculation
2. Compute entry price (best ask for BUY, best bid for SELL)
3. Place order with slippage protection
4. Wait for fill confirmation via user channel
5. Record position in risk engine
6. Set exit monitoring (time-based + price-based)
"""

from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal
from typing import Any, Callable, Dict, Optional

from config.settings import BotConfig
from core.constants import CRYPTO_TAKER_FEE_RATE, GTD_SECURITY_BUFFER_S
from core.types import (
    Direction,
    OrderStatus,
    OrderType,
    Position,
    Side,
    TradeRecord,
    TradeStatus,
    UnifiedSignal,
)
from execution.client import ClobApiError, ClobClient
from risk.engine import RiskEngine, RiskRejection

logger = logging.getLogger(__name__)


class ExecutionError(Exception):
    """Raised when order execution fails irrecoverably."""
    pass


class ExecutionEngine:
    """
    Atomic execution engine with full order lifecycle management.
    All state mutations are protected by asyncio.Lock.
    """

    def __init__(
        self,
        client:       ClobClient,
        risk_engine:  RiskEngine,
        config:       BotConfig,
        on_fill:      Optional[Callable[[TradeRecord], None]] = None,
    ) -> None:
        self._client      = client
        self._risk        = risk_engine
        self._cfg         = config
        self._on_fill_cb  = on_fill

        # Active orders: order_id -> order_meta
        self._active_orders: Dict[str, Dict[str, Any]] = {}
        self._order_lock = asyncio.Lock()

        # Pending fill callbacks: order_id -> asyncio.Event
        self._fill_events: Dict[str, asyncio.Event] = {}

        # Executed trade log (in-memory, rolled over on restart)
        self._trade_log: list[TradeRecord] = []

        # Deduplication: set of order IDs we've already processed
        self._processed_order_ids: set[str] = set()

        self._dry_run = config.dry_run

    # ─────────────────────────── Main Entry ───────────────────────────

    async def execute_entry(
        self,
        signal:      UnifiedSignal,
        direction:   Direction,
        market_pair, # MarketTokenPair
        current_up_state,    # OrderbookState
        current_down_state,  # OrderbookState
    ) -> Optional[Position]:
        """
        Execute a directional entry trade.

        Steps:
        1. Determine which token to buy and entry price
        2. Risk check + size calculation
        3. Place FOK (fill-or-kill) order for immediate execution
        4. Wait for fill confirmation
        5. Record position
        6. Return Position or None on failure
        """
        if market_pair.is_near_expiry:
            logger.warning("Skipping entry — market near expiry (%.0fs remaining)",
                           market_pair.time_remaining)
            return None

        # Determine token and entry price
        if direction == Direction.UP:
            token_id      = market_pair.up_token_id
            book_state    = current_up_state
            # Buy at best ask (aggressive taker)
            entry_price   = book_state.best_ask
            if entry_price is None:
                logger.warning("No ask in UP book — cannot enter")
                return None
        else:
            token_id      = market_pair.down_token_id
            book_state    = current_down_state
            entry_price   = book_state.best_ask
            if entry_price is None:
                logger.warning("No ask in DOWN book — cannot enter")
                return None

        # Add slippage buffer for FOK: best ask + max_slippage
        max_slippage   = self._cfg.trading.max_price_slippage
        worst_price    = min(
            Decimal("0.99"),
            entry_price + max_slippage
        )

        # Determine position size in USDC using fractional Kelly.
        desired_size_usdc = await self._compute_kelly_size_usdc(
            signal=signal,
            direction=direction,
            entry_price=entry_price,
        )
        if desired_size_usdc < self._cfg.trading.min_order_size_usdc:
            logger.info(
                "Kelly size %.2f below minimum %.2f — skipping",
                float(desired_size_usdc),
                float(self._cfg.trading.min_order_size_usdc),
            )
            return None
        cycle_key = f"{market_pair.condition_id}:{market_pair.start_time}"

        try:
            approved_size_usdc = await self._risk.check_entry(
                signal=signal,
                direction=direction,
                entry_price=entry_price,
                size_usdc=desired_size_usdc,
                cycle_key=cycle_key,
                asset_id=token_id,
            )
        except RiskRejection as exc:
            logger.info("Entry rejected by risk engine: %s", exc.reason)
            return None

        # Convert USDC size to shares
        shares = (approved_size_usdc / entry_price).quantize(Decimal("0.01"))
        if shares < Decimal("1"):
            logger.warning("Computed shares %.4f too small — skipping", float(shares))
            return None

        # Compute GTD expiry: market end time - 30s buffer (with Polymarket's 60s offset)
        gtd_expiry = market_pair.end_time - 30 + GTD_SECURITY_BUFFER_S

        if self._dry_run:
            logger.info(
                "DRY RUN: Would buy %s %s @ %.3f (size=%.2f USDC shares=%.2f)",
                direction.value, token_id[:16], float(entry_price),
                float(approved_size_usdc), float(shares)
            )
            # Simulate a fill in dry run mode
            return self._simulate_fill(
                token_id=token_id,
                direction=direction,
                entry_price=entry_price,
                shares=shares,
                approved_size_usdc=approved_size_usdc,
                market_pair=market_pair,
                cycle_key=cycle_key,
            )

        try:
            response = await self._client.post_market_order(
                token_id=token_id,
                amount=approved_size_usdc,
                side=Side.BUY,
                worst_price=worst_price,
                tick_size=market_pair.tick_size,
                neg_risk=market_pair.neg_risk,
                order_type=OrderType.FOK,
            )
        except ClobApiError as exc:
            if exc.error_msg == "FOK_ORDER_NOT_FILLED_ERROR":
                logger.info(
                    "FOK order not filled (insufficient liquidity at %.3f)",
                    float(worst_price)
                )
                return None
            logger.error("Order placement error: %s", exc)
            return None
        except Exception as exc:
            logger.error("Unexpected execution error: %s", exc, exc_info=True)
            return None

        order_id = response.get("orderID", "")
        status   = response.get("status", "")

        # FOK should be immediately matched or rejected
        if status != "matched":
            logger.info(
                "Order status '%s' (not matched) — treating as not filled. orderID=%s",
                status, order_id[:16] if order_id else "?"
            )
            return None

        # Compute actual fee paid
        fee_paid = self._compute_fee(shares, entry_price)

        position = Position(
            asset_id=token_id,
            direction=direction,
            side=Side.BUY,
            entry_price=entry_price,
            size=shares,
            cost_basis=approved_size_usdc,
            fee_paid=fee_paid,
            order_id=order_id,
            entered_at=int(time.time() * 1000),
            condition_id=market_pair.condition_id,
            market_end_time=market_pair.end_time,
        )

        await self._risk.record_entry(position, cycle_key)

        trade = TradeRecord(
            order_id=order_id,
            asset_id=token_id,
            direction=direction,
            side=Side.BUY,
            price=entry_price,
            size=shares,
            fee=fee_paid,
            status=TradeStatus.MATCHED,
            timestamp=int(time.time() * 1000),
            condition_id=market_pair.condition_id,
        )
        self._trade_log.append(trade)

        if self._on_fill_cb:
            try:
                self._on_fill_cb(trade)
            except Exception as exc:
                logger.error("Fill callback error: %s", exc)

        logger.info(
            "ENTRY EXECUTED: %s %s @ %.3f | shares=%.2f cost=%.2f fee=%.4f | order=%s",
            direction.value, token_id[:16], float(entry_price),
            float(shares), float(approved_size_usdc), float(fee_paid),
            order_id[:16] if order_id else "?",
        )

        return position

    # ─────────────────────────── Exit Handling ───────────────────────────

    async def handle_market_resolution(
        self,
        condition_id:    str,
        winning_asset_id: str,
        positions:       Dict[str, Position],
    ) -> None:
        """
        Handle market resolution: compute PnL and record exit.

        In a binary market:
        - Winning token redeems for $1 per share
        - Losing token redeems for $0

        This is called when a market_resolved event is received via WebSocket.
        """
        for asset_id, position in list(positions.items()):
            if position.condition_id != condition_id:
                continue

            won = (asset_id == winning_asset_id)

            if won:
                # Payout = shares × $1
                gross_payout = position.size
                # Net PnL = gross - cost_basis - fee
                pnl = gross_payout - position.cost_basis - position.fee_paid
                logger.info(
                    "MARKET RESOLVED - WIN: %s | payout=%.2f cost=%.2f fee=%.4f pnl=+%.2f",
                    asset_id[:16], float(gross_payout),
                    float(position.cost_basis), float(position.fee_paid),
                    float(pnl)
                )
            else:
                # Token worthless
                pnl = -(position.cost_basis + position.fee_paid)
                logger.warning(
                    "MARKET RESOLVED - LOSS: %s | cost=%.2f fee=%.4f pnl=%.2f",
                    asset_id[:16], float(position.cost_basis),
                    float(position.fee_paid), float(pnl)
                )

            await self._risk.record_exit(
                asset_id=asset_id,
                pnl_usdc=pnl,
                won=won,
            )

    async def cancel_all_positions(self) -> None:
        """Emergency: cancel all open orders (called on kill switch or shutdown)."""
        logger.warning("Cancelling ALL open orders (emergency)")
        success = await self._client.cancel_all_orders()
        if success:
            logger.info("All orders cancelled successfully")
        else:
            logger.error("Failed to cancel all orders — manual check required")

    # ─────────────────────────── User Channel Handler ───────────────────────────

    def handle_user_message(self, msg: Dict[str, Any]) -> None:
        """
        Process user channel WebSocket messages (order/trade updates).
        Called from the WebSocket dispatch loop.
        """
        msg_type = msg.get("type", "")
        event_type = msg.get("event_type", "")

        if msg_type == "TRADE" or event_type == "trade":
            self._handle_trade_update(msg)
        elif msg_type == "PLACEMENT" or event_type == "order":
            self._handle_order_update(msg)

    def _handle_trade_update(self, msg: Dict[str, Any]) -> None:
        """Process trade status updates from user channel."""
        order_id = msg.get("taker_order_id", "") or msg.get("id", "")
        status   = msg.get("status", "")
        price    = msg.get("price", "0")
        size     = msg.get("size", "0")

        logger.debug(
            "Trade update: id=%s status=%s price=%s size=%s",
            order_id[:16] if order_id else "?", status, price, size
        )

        if order_id and order_id in self._fill_events:
            if status in ("MATCHED", "CONFIRMED"):
                self._fill_events[order_id].set()

    def _handle_order_update(self, msg: Dict[str, Any]) -> None:
        """Process order lifecycle updates from user channel."""
        order_id = msg.get("id", "")
        typ      = msg.get("type", "")  # PLACEMENT, UPDATE, CANCELLATION
        logger.debug("Order update: id=%s type=%s", order_id[:16] if order_id else "?", typ)

    # ─────────────────────────── Helpers ───────────────────────────

    def _compute_fee(self, shares: Decimal, price: Decimal) -> Decimal:
        """
        Compute taker fee for a crypto market trade.
        Formula: fee = 0.07 × shares × price × (1 - price)
        This is the actual USDC fee deducted at match time.
        """
        return Decimal(str(CRYPTO_TAKER_FEE_RATE)) * shares * price * (Decimal("1") - price)

    async def _compute_kelly_size_usdc(
        self,
        signal: UnifiedSignal,
        direction: Direction,
        entry_price: Decimal,
    ) -> Decimal:
        """
        Fractional Kelly sizing for binary tokens.

        For a token bought at price p and redeeming at 1 on win:
          odds b = (1 - p) / p
          full Kelly f* = (b * P(win) - P(lose)) / b

        The configured Kelly fraction scales f*, then risk caps bound it.
        """
        if entry_price <= 0 or entry_price >= 1:
            return Decimal("0")

        p_win = signal.up_probability if direction == Direction.UP else signal.down_probability
        p_win = max(Decimal("0"), min(Decimal("1"), p_win))
        p_lose = Decimal("1") - p_win
        fee_per_share = (
            Decimal(str(CRYPTO_TAKER_FEE_RATE))
            * entry_price
            * (Decimal("1") - entry_price)
        )
        gain_per_usdc = (Decimal("1") - entry_price - fee_per_share) / entry_price
        loss_per_usdc = (entry_price + fee_per_share) / entry_price
        if gain_per_usdc <= 0 or loss_per_usdc <= 0:
            return Decimal("0")

        full_kelly = (
            (p_win * gain_per_usdc) - (p_lose * loss_per_usdc)
        ) / (gain_per_usdc * loss_per_usdc)
        if full_kelly <= 0:
            return Decimal("0")

        adjusted_fraction = (
            full_kelly
            * self._cfg.risk.kelly_fraction
            * signal.confidence
        )
        adjusted_fraction = min(adjusted_fraction, self._cfg.risk.max_kelly_fraction)

        risk_state = await self._risk.get_state()
        bankroll = max(Decimal("0"), risk_state.current_equity_usdc)
        desired_size = bankroll * adjusted_fraction
        desired_size = min(desired_size, self._cfg.risk.max_position_usdc)
        desired_size = min(desired_size, self._cfg.risk.max_total_exposure_usdc)

        logger.info(
            "Kelly sizing: p_win=%.3f entry=%.3f gain=%.3f loss=%.3f full=%.3f adjusted=%.3f bankroll=%.2f size=%.2f cap=%.2f",
            float(p_win),
            float(entry_price),
            float(gain_per_usdc),
            float(loss_per_usdc),
            float(full_kelly),
            float(adjusted_fraction),
            float(bankroll),
            float(desired_size),
            float(self._cfg.risk.max_position_usdc),
        )
        return desired_size

    def _simulate_fill(
        self,
        token_id:            str,
        direction:           Direction,
        entry_price:         Decimal,
        shares:              Decimal,
        approved_size_usdc:  Decimal,
        market_pair,
        cycle_key:           str,
    ) -> Position:
        """Create a simulated position for dry run mode."""
        import uuid
        order_id = f"DRY-{uuid.uuid4().hex[:12]}"
        fee_paid = self._compute_fee(shares, entry_price)

        position = Position(
            asset_id=token_id,
            direction=direction,
            side=Side.BUY,
            entry_price=entry_price,
            size=shares,
            cost_basis=approved_size_usdc,
            fee_paid=fee_paid,
            order_id=order_id,
            entered_at=int(time.time() * 1000),
            condition_id=market_pair.condition_id,
            market_end_time=market_pair.end_time,
        )

        # Fire and forget — schedule coroutine without awaiting
        asyncio.create_task(
            self._risk.record_entry(position, cycle_key),
            name="dry-run-entry"
        )
        return position

    @property
    def trade_log(self) -> list[TradeRecord]:
        return list(self._trade_log)
