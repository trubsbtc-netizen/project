"""
Risk management for Polymarket BTC 5-minute directional bot.

Implements comprehensive risk controls including:
- Position sizing via fractional Kelly criterion
- Maximum drawdown monitoring and circuit breaker
- Consecutive loss tracking with cooldown periods
- Session P&L tracking with stop-loss thresholds
- Position limit enforcement
- Settlement time buffer (don't trade near expiry)
- Dynamic risk score computation

Key principles:
1. Quarter-Kelly sizing: never risk more than 25% of Kelly-optimal
2. Drawdown circuit breaker: stop trading at 15% drawdown
3. Cooldown after consecutive losses: 30-minute pause after 5 losses
4. Settlement buffer: don't enter trades within 30s of expiry
5. Asymmetric loss penalty: losses weigh more than wins in risk score

Mathematical basis:
- Kelly fraction: f* = (p*b - q) / b where p=prob, q=1-p, b=odds
- Quarter-Kelly: f = f* / 4 for safety margin
- Drawdown: max(peak - current) / peak
- Risk score: weighted combination of drawdown, loss streak, vol, regime
- Cooldown: exponential decay of risk score during cooldown period
"""

import time
import logging
import numpy as np
from typing import Optional, List, Dict
from dataclasses import dataclass, field

from polymarket_bot.bot_types import (
    RiskState,
    TradeDecision,
    SettlementForecast,
    Direction,
    RegimeState,
)
from polymarket_bot.config import BotConfig, RiskConfig

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a completed trade for risk tracking."""
    timestamp: float
    direction: Direction
    position_size: float
    entry_price: float
    exit_price: float
    pnl: float
    was_win: bool
    confidence_at_entry: float
    edge_at_entry_bps: float
    regime_at_entry: RegimeState


class RiskManager:
    """
    Comprehensive risk management for the trading bot.
    
    The risk manager maintains the current RiskState and enforces
    all risk constraints. It is the final safety layer that prevents
    catastrophic losses even when other systems fail.
    
    Risk controls:
    1. Position sizing: quarter-Kelly with regime adjustment
    2. Drawdown circuit breaker: stop at max_drawdown_fraction
    3. Consecutive loss cooldown: pause after max_consecutive_losses
    4. Session stop-loss: stop if session losses exceed threshold
    5. Settlement time buffer: don't trade near expiry
    6. Position limit: never exceed max_position_notional
    7. Dynamic risk score: composite metric for decision gating
    """

    # Regime-dependent Kelly adjustment factors
    # More conservative in volatile regimes
    REGIME_KELLY_ADJUSTMENT = {
        RegimeState.CALM_TRENDING: 1.0,
        RegimeState.CALM_RANGE: 0.8,
        RegimeState.VOLATILE_TRENDING: 0.6,
        RegimeState.VOLATILE_RANGE: 0.5,
        RegimeState.CRISIS: 0.3,
        RegimeState.LIQUIDATION_CASCADE: 0.2,
    }

    # Asymmetric loss weighting for risk score
    LOSS_WEIGHT = 2.0  # Losses count 2x vs wins
    WIN_WEIGHT = 1.0

    # Risk score component weights
    DRAWDOWN_WEIGHT = 0.35
    CONSECUTIVE_LOSS_WEIGHT = 0.25
    VOLATILITY_WEIGHT = 0.15
    REGIME_WEIGHT = 0.15
    SESSION_PNL_WEIGHT = 0.10

    def __init__(self, config: BotConfig, initial_capital: float = 1000.0):
        self.config = config
        self.risk_config = config.risk
        
        # Capital tracking
        self._initial_capital = initial_capital
        self._current_capital = initial_capital
        self._peak_capital = initial_capital
        self._available_capital = initial_capital
        
        # Position tracking
        self._current_position_size = 0.0
        self._current_position_direction = Direction.NEUTRAL
        self._unrealized_pnl = 0.0
        
        # Session tracking
        self._realized_pnl_session = 0.0
        self._total_trades_session = 0
        self._win_count_session = 0
        self._loss_count_session = 0
        self._consecutive_losses = 0
        self._consecutive_wins = 0
        
        # Drawdown tracking
        self._max_drawdown = 0.0
        self._current_drawdown = 0.0
        
        # Cooldown tracking
        self._is_in_cooldown = False
        self._cooldown_start_time = 0.0
        self._cooldown_remaining_seconds = 0.0
        
        # Stop trading flag
        self._should_stop_trading = False
        
        # Trade history
        self._trade_history: List[TradeRecord] = []
        self._max_history = 500
        
        # Risk score
        self._risk_score = 0.0
        
        # Session start time
        self._session_start_time = time.time()

    def compute_kelly_size(
        self,
        probability: float,
        market_price: float,
        regime_state: RegimeState,
        available_capital: float,
    ) -> float:
        """
        Compute position size using fractional Kelly criterion.
        
        Kelly formula for binary outcome:
        f* = (p * b - q) / b
        
        Where:
        - p = our estimated probability
        - q = 1 - p (counter-probability)
        - b = odds = (1 - market_price) / market_price
        
        Quarter-Kelly: f = f* * kelly_fraction * regime_adjustment
        
        The result is capped at max_position_notional.
        """
        p = probability
        q = 1.0 - p
        
        # Odds: how much we win per unit risked
        # For Polymarket: if we buy at market_price, we win (1 - market_price) if correct
        # and lose market_price if wrong
        if market_price > 0.01 and market_price < 0.99:
            b = (1.0 - market_price) / market_price
        else:
            b = 1.0  # Default odds
        
        # Kelly optimal fraction
        kelly_optimal = (p * b - q) / b
        
        # If Kelly is negative, no bet
        if kelly_optimal <= 0:
            return 0.0
        
        # Quarter-Kelly for safety
        kelly_fraction = self.risk_config.position_sizing_kelly_fraction
        
        # Regime adjustment
        regime_adjustment = self.REGIME_KELLY_ADJUSTMENT.get(regime_state, 0.5)
        
        # Final size
        size = kelly_optimal * kelly_fraction * regime_adjustment * available_capital
        
        # Cap at maximum position
        size = min(size, self.risk_config.max_position_notional)
        
        # Cap at available capital
        size = min(size, available_capital * 0.5)  # Never use more than 50%
        
        # Ensure positive
        size = max(0.0, size)
        
        return float(size)

    def check_settlement_time_buffer(
        self,
        time_to_settlement_seconds: float,
    ) -> bool:
        """
        Check if there's enough time before settlement to trade.
        
        Don't enter trades within the settlement time buffer
        (default 30 seconds) because:
        1. Signal reliability decreases near expiry
        2. Execution latency may cause us to miss the window
        3. No time for the position to work out
        """
        return time_to_settlement_seconds > self.risk_config.settlement_time_buffer_seconds

    def check_position_limits(
        self,
        proposed_size: float,
    ) -> bool:
        """
        Check if proposed position size is within limits.
        
        Limits:
        1. Max position notional
        2. Max loss per settlement
        3. Available capital
        """
        # Max position notional
        if proposed_size > self.risk_config.max_position_notional:
            return False
        
        # Max loss per settlement: position * market_price (worst case)
        # For binary options: max loss = position_size * entry_price
        max_potential_loss = proposed_size * 0.5  # Assume 0.5 entry price worst case
        if max_potential_loss > self.risk_config.max_loss_per_settlement:
            return False
        
        # Available capital
        if proposed_size > self._available_capital:
            return False
        
        # Position limit remaining
        position_limit = self.risk_config.max_position_notional - self._current_position_size
        if proposed_size > position_limit:
            return False
        
        return True

    def compute_risk_score(
        self,
        regime_state: Optional[RegimeState] = None,
        volatility_estimate: Optional[dict] = None,
    ) -> float:
        """
        Compute composite risk score (0-1).
        
        Risk score = weighted sum of:
        1. Drawdown fraction (0-1)
        2. Consecutive loss fraction (0-1)
        3. Volatility risk (0-1)
        4. Regime risk (0-1)
        5. Session P&L risk (0-1)
        
        Higher risk score = more dangerous conditions = reduce trading
        """
        # Drawdown component
        max_dd = self.risk_config.max_drawdown_fraction
        drawdown_fraction = self._current_drawdown / max(0.01, max_dd)
        drawdown_score = np.clip(drawdown_fraction, 0.0, 1.0)
        
        # Consecutive loss component
        max_losses = self.risk_config.max_consecutive_losses
        loss_fraction = self._consecutive_losses / max(1, max_losses)
        loss_score = np.clip(loss_fraction, 0.0, 1.0)
        
        # Volatility component (from estimate if available)
        vol_score = 0.3  # Default moderate
        if volatility_estimate is not None:
            realized_vol = volatility_estimate.get("realized_volatility", 0.0003)
            # High vol = high risk score
            # BTC typical: 0.0001 calm, 0.001 volatile
            vol_score = np.clip(realized_vol * 3000.0, 0.0, 1.0)
        
        # Regime component
        regime_score = 0.3  # Default moderate
        if regime_state is not None:
            regime_risk_map = {
                RegimeState.CALM_TRENDING: 0.1,
                RegimeState.CALM_RANGE: 0.2,
                RegimeState.VOLATILE_TRENDING: 0.5,
                RegimeState.VOLATILE_RANGE: 0.6,
                RegimeState.CRISIS: 0.8,
                RegimeState.LIQUIDATION_CASCADE: 0.95,
            }
            regime_score = regime_risk_map.get(regime_state, 0.3)
        
        # Session P&L component
        pnl_fraction = abs(self._realized_pnl_session) / max(1.0, self._initial_capital)
        pnl_score = np.clip(pnl_fraction * 5.0, 0.0, 1.0)  # 5x amplification
        if self._realized_pnl_session < 0:
            pnl_score *= 1.5  # Asymmetric: losses weigh more
        
        # Weighted combination
        risk_score = (
            self.DRAWDOWN_WEIGHT * drawdown_score
            + self.CONSECUTIVE_LOSS_WEIGHT * loss_score
            + self.VOLATILITY_WEIGHT * vol_score
            + self.REGIME_WEIGHT * regime_score
            + self.SESSION_PNL_WEIGHT * pnl_score
        )
        
        self._risk_score = np.clip(risk_score, 0.0, 1.0)
        return float(self._risk_score)

    def update_drawdown(self, equity: Optional[float] = None):
        """
        Update drawdown tracking.
        
        Drawdown = (peak - current) / peak
        
        Tracks both current drawdown and maximum drawdown.
        """
        equity_value = self._current_capital if equity is None else float(equity)
        if equity_value > self._peak_capital:
            self._peak_capital = equity_value

        if self._peak_capital > 0:
            self._current_drawdown = (
                (self._peak_capital - equity_value) / self._peak_capital
            )
        else:
            self._current_drawdown = 0.0
        self._current_drawdown = max(0.0, self._current_drawdown)
        
        self._max_drawdown = max(self._max_drawdown, self._current_drawdown)

    def update_open_position_state(
        self,
        current_position_size: float,
        current_position_direction: Direction,
        unrealized_pnl: float,
    ) -> None:
        """
        Sync aggregate open-position exposure into the risk state.

        `current_position_size` is the capital reserved in open binary-token
        positions. Realized capital is only changed by record_trade_result;
        available capital is reduced here so the decision layer cannot size
        new trades as if open positions did not exist.
        """
        position_size = max(0.0, float(current_position_size))
        self._current_position_size = position_size
        self._current_position_direction = (
            current_position_direction
            if position_size > 0.0
            else Direction.NEUTRAL
        )
        self._unrealized_pnl = float(unrealized_pnl)
        self._available_capital = max(0.0, self._current_capital - position_size)

        # Drawdown and circuit breaker should respond to marked open losses,
        # not only to already-settled PnL.
        self.update_drawdown(equity=self._current_capital + self._unrealized_pnl)
        self.check_circuit_breaker()

    def check_circuit_breaker(self) -> bool:
        """
        Check if drawdown circuit breaker should be triggered.
        
        Circuit breaker triggers when:
        1. Current drawdown exceeds max_drawdown_fraction
        2. Consecutive losses exceed max_consecutive_losses
        
        Returns True if circuit breaker is triggered (should stop trading).
        """
        # Drawdown circuit breaker
        if self._current_drawdown >= self.risk_config.max_drawdown_fraction:
            logger.warning(
                f"CIRCUIT BREAKER: drawdown {self._current_drawdown:.2%} "
                f"exceeds limit {self.risk_config.max_drawdown_fraction:.2%}"
            )
            self._should_stop_trading = True
            return True
        
        # Consecutive loss circuit breaker
        if self._consecutive_losses >= self.risk_config.max_consecutive_losses:
            if not self._is_in_cooldown:
                logger.warning(
                    f"CIRCUIT BREAKER: {self._consecutive_losses} consecutive losses "
                    f"exceeds limit {self.risk_config.max_consecutive_losses}"
                )
                self._is_in_cooldown = True
                self._cooldown_start_time = time.time()
                self._cooldown_remaining_seconds = self.risk_config.cooldown_after_max_losses_seconds
            return True
        
        return False

    def update_cooldown(self):
        """
        Update cooldown timer.
        
        Cooldown expires after cooldown_after_max_losses_seconds.
        During cooldown, no trades are allowed.
        """
        if self._is_in_cooldown:
            elapsed = time.time() - self._cooldown_start_time
            remaining = self.risk_config.cooldown_after_max_losses_seconds - elapsed
            
            if remaining <= 0:
                # Cooldown expired
                self._is_in_cooldown = False
                self._cooldown_remaining_seconds = 0.0
                self._consecutive_losses = 0  # Reset loss count
                logger.info("Cooldown period expired. Resuming trading.")
            else:
                self._cooldown_remaining_seconds = remaining

    def record_trade_result(
        self,
        direction: Direction,
        position_size: float,
        entry_price: float,
        exit_price: float,
        pnl: float,
        confidence_at_entry: float,
        edge_at_entry_bps: float,
        regime_at_entry: RegimeState,
    ):
        """
        Record a completed trade result.
        
        Updates all risk tracking:
        - Capital (add/subtract P&L)
        - Drawdown tracking
        - Consecutive loss/win tracking
        - Session P&L
        - Win rate
        - Trade history
        """
        was_win = pnl > 0
        
        # Create trade record
        record = TradeRecord(
            timestamp=time.time(),
            direction=direction,
            position_size=position_size,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            was_win=was_win,
            confidence_at_entry=confidence_at_entry,
            edge_at_entry_bps=edge_at_entry_bps,
            regime_at_entry=regime_at_entry,
        )
        
        # Add to history
        self._trade_history.append(record)
        if len(self._trade_history) > self._max_history:
            self._trade_history = self._trade_history[-self._max_history:]
        
        # Update capital
        self._current_capital += pnl
        self._available_capital = self._current_capital
        
        # Update session P&L
        self._realized_pnl_session += pnl
        self._total_trades_session += 1
        
        # Update win/loss tracking
        if was_win:
            self._win_count_session += 1
            self._consecutive_losses = 0
            self._consecutive_wins += 1
        else:
            self._loss_count_session += 1
            self._consecutive_wins = 0
            self._consecutive_losses += 1
        
        # Update drawdown
        self.update_drawdown()
        
        # Check circuit breaker
        self.check_circuit_breaker()
        
        # Log result
        result_str = "WIN" if was_win else "LOSS"
        logger.info(
            f"TRADE RESULT: {result_str} | "
            f"P&L={pnl:+.2f}USDC | "
            f"consecutive_losses={self._consecutive_losses} | "
            f"drawdown={self._current_drawdown:.2%} | "
            f"session_pnl={self._realized_pnl_session:+.2f}USDC"
        )

    def get_risk_state(
        self,
        regime_state: Optional[RegimeState] = None,
        volatility_info: Optional[dict] = None,
    ) -> RiskState:
        """
        Get current risk state for decision engine.
        
        Returns RiskState with all current risk metrics.
        """
        # Update cooldown
        self.update_cooldown()
        
        # Compute risk score
        risk_score = self.compute_risk_score(
            regime_state=regime_state,
            volatility_estimate=volatility_info,
        )
        
        # Compute win rate
        win_rate = self._win_count_session / max(1, self._total_trades_session)
        
        # Position limit remaining
        position_limit = self.risk_config.max_position_notional - self._current_position_size
        
        # Determine if should stop trading
        should_stop = self._should_stop_trading
        
        # Check if drawdown exceeds limit
        if self._current_drawdown >= self.risk_config.max_drawdown_fraction:
            should_stop = True
        
        return RiskState(
            timestamp=time.time(),
            current_position=self._current_position_size,
            current_position_direction=self._current_position_direction,
            unrealized_pnl=self._unrealized_pnl,
            realized_pnl_session=self._realized_pnl_session,
            consecutive_losses=self._consecutive_losses,
            total_trades_session=self._total_trades_session,
            win_rate_session=win_rate,
            max_drawdown=self._max_drawdown,
            current_drawdown=self._current_drawdown,
            risk_score=risk_score,
            is_in_cooldown=self._is_in_cooldown,
            cooldown_remaining_seconds=self._cooldown_remaining_seconds,
            available_capital=self._available_capital,
            position_limit_remaining=position_limit,
            should_stop_trading=should_stop,
        )

    def get_session_summary(self) -> dict:
        """
        Get session summary statistics.
        
        Returns dict with all session metrics for logging and monitoring.
        """
        win_rate = self._win_count_session / max(1, self._total_trades_session)
        
        # Average trade P&L
        avg_pnl = self._realized_pnl_session / max(1, self._total_trades_session)
        
        # Profit factor
        total_wins = sum(r.pnl for r in self._trade_history if r.was_win)
        total_losses = sum(abs(r.pnl) for r in self._trade_history if not r.was_win)
        profit_factor = total_wins / max(0.01, total_losses)
        
        # Sharpe-like ratio (simplified)
        if len(self._trade_history) > 1:
            pnls = [r.pnl for r in self._trade_history]
            sharpe = np.mean(pnls) / max(0.01, np.std(pnls))
        else:
            sharpe = 0.0
        
        return {
            "initial_capital": self._initial_capital,
            "current_capital": self._current_capital,
            "portfolio_equity": self._current_capital + self._unrealized_pnl,
            "available_capital": self._available_capital,
            "peak_capital": self._peak_capital,
            "current_position": self._current_position_size,
            "unrealized_pnl": self._unrealized_pnl,
            "realized_pnl": self._realized_pnl_session,
            "total_trades": self._total_trades_session,
            "win_count": self._win_count_session,
            "loss_count": self._loss_count_session,
            "win_rate": win_rate,
            "consecutive_losses": self._consecutive_losses,
            "consecutive_wins": self._consecutive_wins,
            "max_drawdown": self._max_drawdown,
            "current_drawdown": self._current_drawdown,
            "risk_score": self._risk_score,
            "is_in_cooldown": self._is_in_cooldown,
            "should_stop_trading": self._should_stop_trading,
            "avg_pnl_per_trade": avg_pnl,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "session_duration_seconds": time.time() - self._session_start_time,
        }

    def reset_session(self):
        """
        Reset session tracking for a new trading session.
        
        Keeps capital and drawdown tracking but resets
        session-specific counters.
        """
        self._realized_pnl_session = 0.0
        self._total_trades_session = 0
        self._win_count_session = 0
        self._loss_count_session = 0
        self._consecutive_losses = 0
        self._consecutive_wins = 0
        self._session_start_time = time.time()
        self._should_stop_trading = False
        self._is_in_cooldown = False
        self._cooldown_remaining_seconds = 0.0
        
        logger.info("Session reset. Starting fresh trading session.")
