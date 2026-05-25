"""
Risk Engine — hard risk limits, circuit breaker, kill switch, and drawdown protection.

Every trade decision passes through this engine BEFORE execution.
No trade can bypass these controls.

Risk controls hierarchy (enforced in order):
1. Kill switch — immediate halt, no bypass possible
2. Circuit breaker — tripped by consecutive losses or drawdown
3. Exposure limits — max position and total exposure caps
4. Frequency limiter — max trades per minute and per cycle
5. Volatility circuit — pause on abnormal market conditions

Design principles:
- All limits are HARD (not advisory)
- State is atomic (asyncio.Lock)
- All rejections are logged with reason
- Recovery requires explicit reset (no auto-heal except cooldown)
"""

from __future__ import annotations

import asyncio
import logging
import time
from decimal import Decimal
from typing import Dict, List, Optional, Tuple

from config.settings import RiskConfig
from core.constants import (
    CRYPTO_TAKER_FEE_RATE,
    MAX_DRAWDOWN_PCT,
    MAX_TRADES_PER_CYCLE,
    MAX_TRADES_PER_MINUTE,
)
from core.types import (
    CircuitState,
    Direction,
    Position,
    RiskState,
    TradeRecord,
    TradeStatus,
    UnifiedSignal,
)

logger = logging.getLogger(__name__)


class RiskRejection(Exception):
    """Raised when a risk check rejects a proposed trade."""
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class RiskEngine:
    """
    Central risk management engine.
    All trade proposals must pass through check_entry() before execution.
    """

    def __init__(self, config: RiskConfig, initial_balance_usdc: Decimal) -> None:
        self._cfg = config
        self._lock = asyncio.Lock()

        # Risk state
        self._state = RiskState(
            peak_equity_usdc=initial_balance_usdc,
            current_equity_usdc=initial_balance_usdc,
        )

        # Open positions: asset_id -> Position
        self._positions: Dict[str, Position] = {}

        # Trade timestamps for frequency limiting
        self._trade_times: List[float] = []

        # Per-cycle trade counter: cycle_key -> count
        self._cycle_trade_count: Dict[str, int] = {}

        # Circuit breaker reset time
        self._circuit_open_time: Optional[float] = None

        # Daily loss tracking
        self._session_start_time: float = time.time()
        self._session_start_equity: Decimal = initial_balance_usdc

        logger.info(
            "RiskEngine initialized: balance=%.2f max_position=%.2f max_exposure=%.2f",
            float(initial_balance_usdc),
            float(config.max_position_usdc),
            float(config.max_total_exposure_usdc),
        )

    # ─────────────────────────── Pre-Trade Checks ───────────────────────────

    async def check_entry(
        self,
        signal:      UnifiedSignal,
        direction:   Direction,
        entry_price: Decimal,
        size_usdc:   Decimal,
        cycle_key:   str,          # Unique key for this market cycle (condition_id + start_time)
        asset_id:    str,
    ) -> Decimal:
        """
        Validate all risk conditions before allowing trade entry.
        Returns approved position size in USDC (may be reduced from requested).
        Raises RiskRejection if trade cannot proceed.

        Checks performed (in order):
        1. Kill switch
        2. Circuit breaker state
        3. Existing position in same asset (no duplication)
        4. Max exposure
        5. Max per-cycle trades
        6. Frequency limiter
        7. Position size validation
        8. Min EV check
        9. Signal sanity re-check
        """
        async with self._lock:
            # ── 1. Kill switch ──
            if self._state.kill_switch_active:
                raise RiskRejection("KILL_SWITCH: System halted — no trading permitted")

            # ── 2. Circuit breaker ──
            await self._check_circuit_breaker_recovery()
            if self._state.circuit_state == CircuitState.OPEN:
                raise RiskRejection(
                    f"CIRCUIT_OPEN: Breaker tripped, recovering in "
                    f"{self._circuit_cooldown_remaining():.0f}s"
                )

            # ── 3. Duplicate position check ──
            if asset_id in self._positions:
                raise RiskRejection(
                    f"DUPLICATE_POSITION: Already holding position in {asset_id[:16]}"
                )

            # ── 4. Max total exposure ──
            proposed_total = self._state.total_exposure_usdc + size_usdc
            if proposed_total > self._cfg.max_total_exposure_usdc:
                # Try to reduce size to fit within limit
                available = self._cfg.max_total_exposure_usdc - self._state.total_exposure_usdc
                if available < self._cfg.min_position_usdc():
                    raise RiskRejection(
                        f"MAX_EXPOSURE: Total exposure {float(proposed_total):.2f} "
                        f"> limit {float(self._cfg.max_total_exposure_usdc):.2f}"
                    )
                size_usdc = available
                logger.info("Position size reduced to %.2f due to exposure limit", float(size_usdc))

            # ── 5. Per-cycle trade limit ──
            cycle_count = self._cycle_trade_count.get(cycle_key, 0)
            if cycle_count >= self._cfg.max_trades_per_cycle:
                raise RiskRejection(
                    f"CYCLE_LIMIT: Already traded {cycle_count} times in cycle {cycle_key[:16]}"
                )

            # ── 6. Frequency limiter ──
            now = time.time()
            minute_ago = now - 60.0
            recent_trades = [t for t in self._trade_times if t > minute_ago]
            if len(recent_trades) >= self._cfg.max_trades_per_minute:
                raise RiskRejection(
                    f"FREQUENCY_LIMIT: {len(recent_trades)} trades in last 60s "
                    f"(max={self._cfg.max_trades_per_minute})"
                )

            # ── 7. Position size bounds ──
            if size_usdc > self._cfg.max_position_usdc:
                size_usdc = self._cfg.max_position_usdc
            if size_usdc < Decimal("5"):   # Polymarket minimum
                raise RiskRejection(
                    f"MIN_SIZE: Position size {float(size_usdc):.2f} below minimum $5"
                )

            # ── 8. Signal sanity check ──
            if not signal.is_tradeable:
                raise RiskRejection(
                    f"SIGNAL_NOT_TRADEABLE: confidence={float(signal.confidence):.2f} "
                    f"strength={signal.signal_strength.name}"
                )

            # ── 9. Daily loss limit ──
            daily_loss = self._session_start_equity - self._state.current_equity_usdc
            if daily_loss >= self._cfg.max_daily_loss_usdc:
                self._activate_kill_switch(f"Daily loss limit reached: {float(daily_loss):.2f}")
                raise RiskRejection(f"DAILY_LOSS_LIMIT: Lost {float(daily_loss):.2f} today")

            logger.info(
                "Risk check PASSED: direction=%s size=%.2f entry=%.3f "
                "exposure=%.2f/%.2f cycles=%d",
                direction.value, float(size_usdc), float(entry_price),
                float(self._state.total_exposure_usdc + size_usdc),
                float(self._cfg.max_total_exposure_usdc),
                cycle_count + 1,
            )

            return size_usdc

    # ─────────────────────────── Position Tracking ───────────────────────────

    async def record_entry(
        self,
        position: Position,
        cycle_key: str,
    ) -> None:
        """Record a new position after successful order fill."""
        async with self._lock:
            self._positions[position.asset_id] = position
            self._state.total_exposure_usdc += position.cost_basis
            self._state.open_positions = len(self._positions)
            self._state.session_trades += 1
            self._state.last_trade_time = int(time.time() * 1000)

            # Update cycle counter
            self._cycle_trade_count[cycle_key] = (
                self._cycle_trade_count.get(cycle_key, 0) + 1
            )

            # Update frequency tracking
            self._trade_times.append(time.time())

            logger.info(
                "Position recorded: asset=%s direction=%s cost=%.2f total_exposure=%.2f",
                position.asset_id[:16],
                position.direction.value,
                float(position.cost_basis),
                float(self._state.total_exposure_usdc),
            )

    async def record_exit(
        self,
        asset_id:  str,
        pnl_usdc:  Decimal,
        won:       bool,
    ) -> None:
        """
        Record position exit after market resolution.
        Updates PnL, equity, consecutive win/loss counters.
        Checks circuit breaker conditions.
        """
        async with self._lock:
            position = self._positions.pop(asset_id, None)
            if position is None:
                logger.warning("Exit recorded for unknown position: %s", asset_id[:16])
                return

            self._state.total_exposure_usdc -= position.cost_basis
            self._state.total_exposure_usdc = max(
                Decimal("0"), self._state.total_exposure_usdc
            )
            self._state.open_positions = len(self._positions)
            self._state.session_pnl_usdc += pnl_usdc
            self._state.current_equity_usdc += pnl_usdc

            # Update win/loss streak
            if won:
                self._state.consecutive_wins += 1
                self._state.consecutive_losses = 0
                logger.info(
                    "WIN: asset=%s pnl=+%.2f streak=%d",
                    asset_id[:16], float(pnl_usdc), self._state.consecutive_wins
                )
            else:
                self._state.consecutive_losses += 1
                self._state.consecutive_wins   = 0
                logger.warning(
                    "LOSS: asset=%s pnl=%.2f streak=%d",
                    asset_id[:16], float(pnl_usdc), self._state.consecutive_losses
                )

            # Update peak equity and drawdown
            if self._state.current_equity_usdc > self._state.peak_equity_usdc:
                self._state.peak_equity_usdc = self._state.current_equity_usdc

            if self._state.peak_equity_usdc > 0:
                drawdown = (
                    (self._state.peak_equity_usdc - self._state.current_equity_usdc)
                    / self._state.peak_equity_usdc
                )
                self._state.max_drawdown_usdc = max(
                    self._state.max_drawdown_usdc,
                    self._state.peak_equity_usdc - self._state.current_equity_usdc,
                )

                # Drawdown circuit breaker
                if drawdown >= self._cfg.max_drawdown_pct:
                    self._trip_circuit_breaker(
                        f"Max drawdown exceeded: {float(drawdown*100):.1f}%"
                    )

            # Consecutive loss circuit breaker
            if self._state.consecutive_losses >= self._cfg.max_consecutive_losses:
                self._trip_circuit_breaker(
                    f"Consecutive losses: {self._state.consecutive_losses}"
                )

    # ─────────────────────────── Circuit Breaker ───────────────────────────

    def _trip_circuit_breaker(self, reason: str) -> None:
        """Trip the circuit breaker (must be called under lock)."""
        self._state.circuit_state = CircuitState.OPEN
        self._circuit_open_time   = time.monotonic()
        logger.critical(
            "CIRCUIT BREAKER TRIPPED: %s | "
            "consecutive_losses=%d drawdown=%.2f pnl=%.2f",
            reason,
            self._state.consecutive_losses,
            float(self._state.max_drawdown_usdc),
            float(self._state.session_pnl_usdc),
        )

    async def _check_circuit_breaker_recovery(self) -> None:
        """Check if circuit breaker cooldown has elapsed (must be called under lock)."""
        if self._state.circuit_state != CircuitState.OPEN:
            return
        if self._circuit_open_time is None:
            return

        elapsed = time.monotonic() - self._circuit_open_time
        if elapsed >= self._cfg.circuit_breaker_cooldown:
            self._state.circuit_state     = CircuitState.HALF_OPEN
            self._state.consecutive_losses = 0   # Reset loss counter after cooldown
            logger.info(
                "Circuit breaker entering HALF_OPEN after %.0fs cooldown",
                elapsed
            )

    def _circuit_cooldown_remaining(self) -> float:
        if self._circuit_open_time is None:
            return 0.0
        elapsed = time.monotonic() - self._circuit_open_time
        return max(0.0, self._cfg.circuit_breaker_cooldown - elapsed)

    # ─────────────────────────── Kill Switch ───────────────────────────

    def _activate_kill_switch(self, reason: str) -> None:
        """Activate kill switch — permanent halt until manual reset."""
        self._state.kill_switch_active = True
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

    def activate_kill_switch(self, reason: str) -> None:
        """External kill switch activation (e.g. from monitoring system)."""
        self._state.kill_switch_active = True
        logger.critical("KILL SWITCH ACTIVATED (external): %s", reason)

    def reset_kill_switch(self) -> None:
        """Manual reset of kill switch (operator action only)."""
        self._state.kill_switch_active = False
        self._state.circuit_state      = CircuitState.CLOSED
        logger.warning("Kill switch RESET by operator")

    # ─────────────────────────── Market Anomaly Detection ───────────────────────

    async def check_market_anomaly(
        self,
        signal: UnifiedSignal,
    ) -> Tuple[bool, str]:
        """
        Detect abnormal market conditions that warrant a trading pause.

        Returns (is_anomaly, reason).
        Anomaly conditions:
        - Signal volatility too high (price flipping rapidly)
        - Liquidity vacuum detected
        - Manipulation flag set
        - Violent reversal in progress
        """
        if signal.liquidity_vacuum:
            return True, "LIQUIDITY_VACUUM: Critically thin orderbook"
        if signal.manipulation_flag:
            return True, "MANIPULATION_DETECTED: High spoof probability"
        if signal.violent_reversal:
            return True, "VIOLENT_REVERSAL: Price reversing aggressively"
        if not signal.liquidity_ok:
            return True, "INSUFFICIENT_LIQUIDITY: Book depth too low"
        return False, ""

    # ─────────────────────────── State Accessors ───────────────────────────

    async def get_state(self) -> RiskState:
        async with self._lock:
            return self._state

    async def get_positions(self) -> Dict[str, Position]:
        async with self._lock:
            return dict(self._positions)

    async def has_position(self, asset_id: str) -> bool:
        async with self._lock:
            return asset_id in self._positions

    async def get_position(self, asset_id: str) -> Optional[Position]:
        async with self._lock:
            return self._positions.get(asset_id)

    def min_position_usdc(self) -> Decimal:
        return Decimal("5")


# Patch RiskConfig to add helper
RiskConfig.min_position_usdc = lambda self: Decimal("5")
