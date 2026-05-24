"""
Spread stability and expansion analysis for microstructure inference.

Mathematical foundation:
- Quoted spread: S = P_ask - P_bid
- Effective spread: S_eff = 2 * |P_trade - P_mid| (trade-weighted average)
- Realized spread: S_realized = 2 * (P_trade - P_mid_at_trade_time) * sign(trade)
  measured after price movement to capture true market maker profit
- Spread stability: variance of spread over time, normalized by mean
  stability = 1 - (var(S) / mean(S)^2) = 1 - CV^2
- Spread expansion: deviation from recent mean in standard deviations
  expansion_score = max(0, (S_current - mean_S) / std_S)
- Spread asymmetry: (P_ask - P_mid) vs (P_mid - P_bid) difference
  asymmetry = (ask_distance - bid_distance) / S
- Time-weighted spread: EMA of spread with time-based decay

Spread dynamics are critical for:
1. Execution cost estimation (slippage modeling)
2. Market health assessment (stable spread = healthy market)
3. Regime transition detection (spread expansion = regime change)
4. Liquidity quality assessment (tight spread = good liquidity)
"""

import logging
import time
import numpy as np
from typing import Optional, List, Tuple
from collections import deque

from polymarket_bot.config import MicrostructureConfig
from polymarket_bot.bot_types import (
    OrderbookSnapshot, TickData, SpreadMetrics
)

logger = logging.getLogger(__name__)


class SpreadAnalyzer:
    """
    Analyzes spread dynamics for market health and execution cost estimation.
    
    Key innovations:
    1. Multi-model spread estimation (quoted, effective, realized)
    2. Spread stability scoring via coefficient of variation
    3. Spread expansion detection (regime transition signal)
    4. Spread asymmetry analysis (directional pressure indicator)
    5. Time-weighted spread with exponential decay
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        
        # Spread history for stability and expansion analysis
        self._spread_history: deque = deque(maxlen=config.spread_stability_window)
        self._spread_bps_history: deque = deque(maxlen=config.spread_stability_window)
        self._mid_price_history: deque = deque(maxlen=config.spread_stability_window)
        
        # Effective spread tracking (trade-weighted)
        self._effective_spread_ema = 0.0
        self._realized_spread_ema = 0.0
        
        # Spread asymmetry tracking
        self._spread_asymmetry_ema = 0.0
        
        # Time-weighted spread
        self._time_weighted_spread_ema = 0.0
        self._last_update_time = 0.0
        
        # Expansion detection
        self._spread_mean = 0.0
        self._spread_std = 0.0
        self._expansion_score_ema = 0.0
        
        # Stability score
        self._stability_score_ema = 0.5
        
        # Previous metrics
        self._prev_metrics: Optional[SpreadMetrics] = None
        self._initialized = False
    
    def analyze(self, snapshot: OrderbookSnapshot,
                last_tick: Optional[TickData] = None) -> SpreadMetrics:
        """
        Compute spread metrics from orderbook snapshot and optional last trade.
        
        Process:
        1. Compute quoted spread and spread in bps
        2. Compute effective spread from last trade (if available)
        3. Compute spread stability from history
        4. Detect spread expansion
        5. Compute spread asymmetry
        6. Compute time-weighted spread
        7. Smooth all metrics
        """
        timestamp = snapshot.timestamp
        
        if not snapshot.bids or not snapshot.asks:
            return self._empty_metrics(timestamp)
        
        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        mid_price = snapshot.mid_price
        
        if best_bid is None or best_ask is None or mid_price is None or mid_price <= 0:
            return self._empty_metrics(timestamp)
        
        # Step 1: Quoted spread
        quoted_spread = best_ask - best_bid
        spread_bps = (quoted_spread / mid_price) * 10000.0
        
        # Step 2: Effective spread from last trade
        effective_spread = self._compute_effective_spread(
            last_tick, mid_price, quoted_spread
        )
        
        # Step 3: Realized spread (simplified - needs trade + future mid)
        realized_spread = self._compute_realized_spread(
            last_tick, mid_price, quoted_spread
        )
        
        # Step 4: Update history and compute stability
        self._spread_history.append(quoted_spread)
        self._spread_bps_history.append(spread_bps)
        self._mid_price_history.append(mid_price)
        
        stability_score = self._compute_stability()
        
        # Step 5: Spread expansion detection
        expansion_score = self._detect_expansion(quoted_spread)
        
        # Step 6: Spread asymmetry
        asymmetry = self._compute_asymmetry(best_bid, best_ask, mid_price, quoted_spread)
        
        # Step 7: Time-weighted spread
        dt = timestamp - self._last_update_time if self._last_update_time > 0 else 1.0
        dt = max(dt, 0.001)
        decay_rate = 0.1  # Decay per second
        alpha = 1.0 - np.exp(-decay_rate * dt)
        self._time_weighted_spread_ema = (1 - alpha) * self._time_weighted_spread_ema + alpha * quoted_spread
        
        # Smooth all metrics
        if self._prev_metrics is not None and self._initialized:
            smoothing = 0.7
            effective_spread = smoothing * effective_spread + (1 - smoothing) * self._prev_metrics.effective_spread
            realized_spread = smoothing * realized_spread + (1 - smoothing) * self._prev_metrics.realized_spread
            asymmetry = smoothing * asymmetry + (1 - smoothing) * self._prev_metrics.spread_asymmetry
        
        result = SpreadMetrics(
            timestamp=timestamp,
            spread=quoted_spread,
            spread_bps=spread_bps,
            effective_spread=effective_spread,
            realized_spread=realized_spread,
            spread_stability=stability_score,
            spread_expansion_score=expansion_score,
            spread_asymmetry=asymmetry,
            quoted_spread=quoted_spread,
            time_weighted_spread=self._time_weighted_spread_ema,
        )
        
        self._prev_metrics = result
        self._last_update_time = timestamp
        self._initialized = True
        
        return result
    
    def _compute_effective_spread(self, last_tick: Optional[TickData],
                                   mid_price: float, quoted_spread: float) -> float:
        """
        Compute effective spread from last trade.
        
        Effective spread = 2 * |P_trade - P_mid|
        
        This measures the actual cost paid by takers, which is typically
        less than quoted spread due to price improvement.
        
        If no trade available, use quoted spread * effective_weight factor.
        """
        if last_tick is not None and last_tick.price > 0 and mid_price > 0:
            effective = 2.0 * abs(last_tick.price - mid_price)
            # Smooth with EMA
            self._effective_spread_ema = 0.6 * effective + 0.4 * self._effective_spread_ema
            return self._effective_spread_ema
        
        # No trade: estimate from quoted spread with weight factor
        # Effective spread is typically 70-90% of quoted spread
        estimated = quoted_spread * self.config.effective_spread_weight
        self._effective_spread_ema = 0.6 * estimated + 0.4 * self._effective_spread_ema
        return self._effective_spread_ema
    
    def _compute_realized_spread(self, last_tick: Optional[TickData],
                                  mid_price: float, quoted_spread: float) -> float:
        """
        Compute realized spread (simplified model).
        
        Realized spread = effective spread - adverse selection component
        
        In practice: S_realized = S_effective - (P_mid_future - P_mid_at_trade) * sign
        
        Since we can't observe future mid immediately, we use a simplified model:
        S_realized = S_effective * (1 - adverse_selection_fraction)
        
        Adverse selection fraction estimated from spread stability:
        - Stable spread: low adverse selection (0.2)
        - Expanding spread: high adverse selection (0.5)
        """
        adverse_fraction = 0.2 + 0.3 * self._expansion_score_ema
        realized = self._effective_spread_ema * (1.0 - adverse_fraction)
        
        self._realized_spread_ema = 0.6 * realized + 0.4 * self._realized_spread_ema
        return self._realized_spread_ema
    
    def _compute_stability(self) -> float:
        """
        Compute spread stability score from history.
        
        stability = 1 - CV^2 where CV = std(S) / mean(S)
        
        High stability (>0.8): spread has been consistent (healthy market)
        Low stability (<0.3): spread has been fluctuating (unhealthy/regime change)
        """
        if len(self._spread_history) < 10:
            return 0.5
        
        spreads = np.array(list(self._spread_history))
        mean_spread = np.mean(spreads)
        var_spread = np.var(spreads)
        
        if mean_spread > 0:
            cv_squared = var_spread / (mean_spread ** 2)
            stability = max(0.0, min(1.0, 1.0 - cv_squared))
        else:
            stability = 0.0
        
        # Smooth
        self._stability_score_ema = 0.7 * stability + 0.3 * self._stability_score_ema
        
        # Update running statistics
        self._spread_mean = mean_spread
        self._spread_std = np.std(spreads)
        
        return self._stability_score_ema
    
    def _detect_expansion(self, current_spread: float) -> float:
        """
        Detect spread expansion (regime transition signal).
        
        expansion_score = max(0, (S_current - mean_S) / std_S)
        
        Score > 2.0: significant expansion (likely regime change)
        Score > 1.0: moderate expansion (possible regime change)
        Score < 0.5: normal spread (stable regime)
        
        Spread expansion is one of the earliest signals of regime transition
        and often precedes significant price movement.
        """
        if self._spread_std <= 0 or self._spread_mean <= 0:
            return 0.0
        
        # Z-score of current spread relative to history
        z_score = (current_spread - self._spread_mean) / max(self._spread_std, 1e-10)
        
        # Only flag expansion (not contraction)
        if z_score > 0:
            # Normalize: z=2 -> score=1.0
            raw_score = min(1.0, z_score / self.config.spread_expansion_threshold)
        else:
            raw_score = 0.0
        
        # Smooth
        self._expansion_score_ema = 0.6 * raw_score + 0.4 * self._expansion_score_ema
        
        return self._expansion_score_ema
    
    def _compute_asymmetry(self, best_bid: float, best_ask: float,
                           mid_price: float, spread: float) -> float:
        """
        Compute spread asymmetry (directional pressure indicator).
        
        asymmetry = (ask_distance - bid_distance) / spread
        
        where ask_distance = P_ask - P_mid, bid_distance = P_mid - P_bid
        
        Positive asymmetry: ask side further from mid (sell pressure, wider ask)
        Negative asymmetry: bid side further from mid (buy pressure, wider bid)
        Near zero: symmetric spread (balanced market)
        
        Asymmetry is a subtle but powerful directional signal:
        - Market makers widen the side they expect pressure on
        - Asymmetric spread = directional risk premium
        """
        if spread <= 0:
            return 0.0
        
        ask_distance = best_ask - mid_price
        bid_distance = mid_price - best_bid
        
        asymmetry = (ask_distance - bid_distance) / spread
        
        # Smooth
        self._spread_asymmetry_ema = 0.7 * asymmetry + 0.3 * self._spread_asymmetry_ema
        
        return self._spread_asymmetry_ema
    
    def _empty_metrics(self, timestamp: float) -> SpreadMetrics:
        """Return empty metrics when no data available."""
        return SpreadMetrics(
            timestamp=timestamp,
            spread=0.0,
            spread_bps=0.0,
            effective_spread=0.0,
            realized_spread=0.0,
            spread_stability=0.5,
            spread_expansion_score=0.0,
            spread_asymmetry=0.0,
            quoted_spread=0.0,
            time_weighted_spread=0.0,
        )