"""
Liquidity stability, absorption, and vacuum detection analyzer.

Mathematical foundation:
- Liquidity stability: variance of depth over time, normalized by mean depth
- Absorption detection: volume executed at a level vs displayed volume at that level
  absorption_ratio = executed_volume / displayed_volume; if > threshold, absorption detected
- Liquidity vacuum: rapid removal of depth from one side of the book
  vacuum_score = depth_removed / initial_depth, with time-weighted urgency
- Real liquidity fraction: fraction of displayed liquidity that survives > T seconds
  P(survive) = exp(-cancel_rate * T), real_fraction = E[P(survive)] across levels
- Cancellation velocity: rate of order removal per unit time
- Add/cancel ratio: order addition rate vs cancellation rate (health indicator)
- Depth concentration: Herfindahl index of depth distribution across levels
  H = sum(s_i^2) / (sum(s_i))^2, where s_i is size at level i

All metrics are computed with regime-awareness and time-based exponential smoothing.
"""

import logging
import time
import numpy as np
from typing import Optional, List, Dict, Tuple
from collections import deque

from polymarket_bot.config import MicrostructureConfig
from polymarket_bot.bot_types import (
    OrderbookSnapshot, OrderbookLevel, OrderbookDelta, OrderbookBuffer,
    LiquidityMetrics
)

logger = logging.getLogger(__name__)


class LiquidityAnalyzer:
    """
    Analyzes liquidity stability, absorption, and vacuum conditions.
    
    Key innovations:
    1. Distinguishes real vs fake liquidity via survival probability
    2. Detects absorption walls (large hidden execution behind displayed levels)
    3. Detects liquidity vacuums (rapid depth removal creating price gaps)
    4. Depth concentration via Herfindahl index (liquidity clustering)
    5. Cancellation velocity tracking (spoof/real liquidity discriminator)
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        
        # Depth tracking history
        self._bid_depth_history: deque = deque(maxlen=config.liquidity_stability_window)
        self._ask_depth_history: deque = deque(maxlen=config.liquidity_stability_window)
        self._depth_ratio_history: deque = deque(maxlen=config.liquidity_stability_window)
        
        # Cancellation/addition tracking
        self._cancellation_events: deque = deque(maxlen=500)
        self._addition_events: deque = deque(maxlen=500)
        self._cancellation_velocity = 0.0
        self._addition_velocity = 0.0
        self._add_cancel_ratio = 1.0
        
        # Absorption tracking
        self._absorption_events: deque = deque(maxlen=100)
        self._absorption_score_ema = 0.0
        self._absorption_volume_ema = 0.0
        
        # Vacuum tracking
        self._prev_bid_depth = 0.0
        self._prev_ask_depth = 0.0
        self._vacuum_score_ema = 0.0
        self._vacuum_depth_ema = 0.0
        
        # Real liquidity fraction
        self._real_liquidity_fraction_ema = 0.5  # Start neutral
        
        # Stability score
        self._stability_score_ema = 0.5
        
        # Previous metrics for smoothing
        self._prev_metrics: Optional[LiquidityMetrics] = None
        
        # Time tracking
        self._last_update_time = 0.0
        self._initialized = False
    
    def analyze(self, snapshot: OrderbookSnapshot,
                buffer: Optional[OrderbookBuffer] = None) -> LiquidityMetrics:
        """
        Compute full liquidity metrics from orderbook snapshot and delta history.
        
        Process:
        1. Compute total and real depth at configured levels
        2. Track cancellation/addition velocity from buffer
        3. Detect absorption from executed volume vs displayed
        4. Detect vacuum from rapid depth changes
        5. Compute depth concentration (Herfindahl index)
        6. Compute stability score from depth variance
        7. Smooth all metrics with previous estimates
        """
        timestamp = snapshot.timestamp
        
        if not snapshot.bids or not snapshot.asks:
            return self._empty_metrics(timestamp)
        
        depth = self.config.orderbook_depth_levels
        
        # Step 1: Compute total depth
        total_bid_depth = sum(l.size for l in snapshot.bids[:depth])
        total_ask_depth = sum(l.size for l in snapshot.asks[:depth])
        
        # Compute real depth (using is_real flag from spoof filtering)
        real_bid_depth = sum(l.size for l in snapshot.bids[:depth] if l.is_real)
        real_ask_depth = sum(l.size for l in snapshot.asks[:depth] if l.is_real)
        
        # If no spoof filtering applied, use total as real
        if real_bid_depth == 0 and total_bid_depth > 0:
            real_bid_depth = total_bid_depth * self._real_liquidity_fraction_ema
        if real_ask_depth == 0 and total_ask_depth > 0:
            real_ask_depth = total_ask_depth * self._real_liquidity_fraction_ema
        
        # Depth ratio
        depth_ratio = total_bid_depth / max(total_ask_depth, 1e-10)
        
        # Step 2: Cancellation/addition velocity from buffer
        self._update_velocity_from_buffer(buffer, timestamp)
        
        # Step 3: Absorption detection
        absorption_score, absorption_volume = self._detect_absorption(snapshot, buffer)
        
        # Step 4: Vacuum detection
        vacuum_score, vacuum_depth = self._detect_vacuum(
            total_bid_depth, total_ask_depth, timestamp
        )
        
        # Step 5: Real liquidity fraction
        real_fraction = self._compute_real_liquidity_fraction(snapshot, buffer)
        
        # Step 6: Depth concentration (Herfindahl index)
        bid_concentration = self._compute_herfindahl(snapshot.bids[:depth])
        ask_concentration = self._compute_herfindahl(snapshot.asks[:depth])
        liquidity_concentration = (bid_concentration + ask_concentration) / 2.0
        
        # Step 7: Stability score
        stability_score = self._compute_stability_score(
            total_bid_depth, total_ask_depth
        )
        
        # Step 8: Depth skew coefficient (power-law)
        mid_price = snapshot.mid_price or 0
        depth_skew_coeff = self._compute_depth_skew_coefficient(snapshot, mid_price)
        
        # Smooth metrics with previous estimates
        if self._prev_metrics is not None and self._initialized:
            alpha = 0.7  # Smoothing factor
            absorption_score = alpha * absorption_score + (1 - alpha) * self._prev_metrics.absorption_score
            vacuum_score = alpha * vacuum_score + (1 - alpha) * self._prev_metrics.vacuum_score
            stability_score = alpha * stability_score + (1 - alpha) * self._prev_metrics.liquidity_stability_score
            real_fraction = alpha * real_fraction + (1 - alpha) * self._prev_metrics.real_liquidity_fraction
        
        result = LiquidityMetrics(
            timestamp=timestamp,
            total_bid_depth=total_bid_depth,
            total_ask_depth=total_ask_depth,
            real_bid_depth=real_bid_depth,
            real_ask_depth=real_ask_depth,
            depth_ratio=depth_ratio,
            liquidity_stability_score=stability_score,
            cancellation_velocity=self._cancellation_velocity,
            add_cancel_ratio=self._add_cancel_ratio,
            absorption_score=absorption_score,
            absorption_volume=absorption_volume,
            vacuum_score=vacuum_score,
            vacuum_depth=vacuum_depth,
            real_liquidity_fraction=real_fraction,
            liquidity_concentration=liquidity_concentration,
            depth_skew_coefficient=depth_skew_coeff,
        )
        
        self._prev_metrics = result
        self._prev_bid_depth = total_bid_depth
        self._prev_ask_depth = total_ask_depth
        self._last_update_time = timestamp
        self._initialized = True
        
        return result
    
    def _update_velocity_from_buffer(self, buffer: Optional[OrderbookBuffer],
                                       timestamp: float):
        """
        Update cancellation and addition velocity from orderbook delta buffer.
        
        Velocity = volume_rate per second of cancellations/additions.
        Uses recent window of deltas for time-based rate estimation.
        """
        if buffer is None:
            return
        
        recent_deltas = buffer.recent_deltas(300)
        if len(recent_deltas) < 5:
            return
        
        time_span = recent_deltas[-1].timestamp - recent_deltas[0].timestamp
        if time_span < 1.0:
            time_span = 1.0
        
        cancel_volume = 0.0
        add_volume = 0.0
        cancel_count = 0
        add_count = 0
        
        for delta in recent_deltas:
            if delta.is_trade:
                continue  # Trades are real flow
            if delta.size_delta < 0:
                cancel_volume += abs(delta.size_delta)
                cancel_count += 1
            elif delta.size_delta > 0:
                add_volume += delta.size_delta
                add_count += 1
        
        # Velocity = volume per second
        self._cancellation_velocity = cancel_volume / time_span
        self._addition_velocity = add_volume / time_span
        
        # Add/cancel ratio (health indicator)
        total_velocity = self._cancellation_velocity + self._addition_velocity
        if total_velocity > 0:
            self._add_cancel_ratio = self._addition_velocity / total_velocity
        else:
            self._add_cancel_ratio = 0.5  # Neutral
    
    def _detect_absorption(self, snapshot: OrderbookSnapshot,
                           buffer: Optional[OrderbookBuffer]) -> Tuple[float, float]:
        """
        Detect absorption: when large volume is executed behind displayed levels.
        
        Absorption occurs when:
        1. A large trade happens at a price level
        2. The displayed size at that level was much smaller than the trade
        3. The level still has remaining size after the trade (hidden liquidity)
        
        absorption_ratio = executed_volume / displayed_volume_at_level
        
        If ratio > threshold, absorption is detected.
        This indicates hidden liquidity (iceberg orders or dark pool flow).
        
        Returns: (absorption_score 0-1, absorption_volume)
        """
        if buffer is None:
            return self._absorption_score_ema, self._absorption_volume_ema
        
        recent_deltas = buffer.recent_deltas(100)
        
        total_absorbed_volume = 0.0
        absorption_count = 0
        
        # Build a map of displayed sizes at each level
        displayed_bid_sizes: Dict[float, float] = {}
        displayed_ask_sizes: Dict[float, float] = {}
        
        depth = self.config.orderbook_depth_levels
        for level in snapshot.bids[:depth]:
            displayed_bid_sizes[level.price] = level.size
        for level in snapshot.asks[:depth]:
            displayed_ask_sizes[level.price] = level.size
        
        # Check trade deltas for absorption
        for delta in recent_deltas:
            if not delta.is_trade:
                continue
            
            # Find displayed size at trade price
            if delta.side == "bid":
                displayed = displayed_bid_sizes.get(delta.price, 0)
            else:
                displayed = displayed_ask_sizes.get(delta.price, 0)
            
            # Absorption: trade volume much larger than displayed
            if displayed > 0:
                trade_volume = abs(delta.size_delta)
                ratio = trade_volume / displayed
                
                if ratio > self.config.absorption_detection_volume_ratio:
                    total_absorbed_volume += trade_volume
                    absorption_count += 1
        
        # Compute absorption score
        if absorption_count > 0:
            # Score based on count and volume
            count_factor = min(1.0, absorption_count / 5.0)
            volume_factor = min(1.0, total_absorbed_volume / 
                              max(sum(l.size for l in snapshot.bids[:5]), 1e-10))
            raw_score = 0.5 * count_factor + 0.5 * volume_factor
        else:
            raw_score = 0.0
        
        # Smooth
        self._absorption_score_ema = 0.7 * raw_score + 0.3 * self._absorption_score_ema
        self._absorption_volume_ema = 0.7 * total_absorbed_volume + 0.3 * self._absorption_volume_ema
        
        return self._absorption_score_ema, self._absorption_volume_ema
    
    def _detect_vacuum(self, bid_depth: float, ask_depth: float,
                       timestamp: float) -> Tuple[float, float]:
        """
        Detect liquidity vacuum: rapid removal of depth from one side.
        
        Vacuum occurs when:
        1. Depth on one side drops significantly in a short time
        2. The other side's depth remains stable or increases
        3. This creates a directional liquidity gap
        
        vacuum_score = max(depth_removed_bid, depth_removed_ask) / initial_depth
        
        Vacuum on bid side = sell pressure vacuum (price likely to drop)
        Vacuum on ask side = buy pressure vacuum (price likely to rise)
        
        Returns: (vacuum_score 0-1, vacuum_depth in absolute terms)
        """
        if not self._initialized:
            self._prev_bid_depth = bid_depth
            self._prev_ask_depth = ask_depth
            return 0.0, 0.0
        
        # Compute depth changes
        bid_change = bid_depth - self._prev_bid_depth
        ask_change = ask_depth - self._prev_ask_depth
        
        # Vacuum: significant depth removal on one side
        bid_removed = max(0, -bid_change)  # Positive if bid depth decreased
        ask_removed = max(0, -ask_change)  # Positive if ask depth decreased
        
        # Normalize by previous depth
        bid_vacuum = bid_removed / max(self._prev_bid_depth, 1e-10)
        ask_vacuum = ask_removed / max(self._prev_ask_depth, 1e-10)
        
        # Take the maximum vacuum (most significant side)
        max_vacuum = max(bid_vacuum, ask_vacuum)
        vacuum_depth = max(bid_removed, ask_removed)
        
        # Threshold detection
        if max_vacuum > self.config.vacuum_detection_threshold:
            raw_score = min(1.0, max_vacuum / 0.5)  # Scale: 0.5 removal = score 1.0
        else:
            raw_score = max_vacuum * 0.5  # Sub-threshold: linear scaling
        
        # Smooth
        self._vacuum_score_ema = 0.6 * raw_score + 0.4 * self._vacuum_score_ema
        self._vacuum_depth_ema = 0.6 * vacuum_depth + 0.4 * self._vacuum_depth_ema
        
        return self._vacuum_score_ema, self._vacuum_depth_ema
    
    def _compute_real_liquidity_fraction(self, snapshot: OrderbookSnapshot,
                                          buffer: Optional[OrderbookBuffer]) -> float:
        """
        Compute fraction of displayed liquidity that is real (will survive).
        
        Real liquidity estimation:
        1. From spoof filtering: count levels flagged as real vs total
        2. From cancellation velocity: P(survive T) = exp(-cancel_rate * T)
        3. From add/cancel ratio: healthy ratio > 0.5 means more real liquidity
        
        Combined estimate = weighted average of these methods.
        """
        depth = self.config.orderbook_depth_levels
        
        # Method 1: Spoof-filtered fraction
        bid_levels = snapshot.bids[:depth]
        ask_levels = snapshot.asks[:depth]
        
        total_levels = len(bid_levels) + len(ask_levels)
        real_levels = sum(1 for l in bid_levels if l.is_real) + \
                      sum(1 for l in ask_levels if l.is_real)
        
        if total_levels > 0:
            spoof_fraction = real_levels / total_levels
        else:
            spoof_fraction = 0.5
        
        # Method 2: Survival probability from cancellation rate
        # P(survive T seconds) = exp(-cancel_rate * T)
        # Use T = 5 seconds (typical time for meaningful survival)
        survival_time = 5.0
        cancel_rate = self._cancellation_velocity
        if cancel_rate > 0:
            # Normalize cancel_rate by total depth to get per-unit rate
            total_depth = sum(l.size for l in bid_levels) + sum(l.size for l in ask_levels)
            per_unit_cancel_rate = cancel_rate / max(total_depth, 1e-10)
            survival_prob = np.exp(-per_unit_cancel_rate * survival_time)
        else:
            survival_prob = 1.0
        
        # Method 3: Add/cancel ratio indicator
        # Healthy ratio (>0.5) suggests more real liquidity
        ratio_indicator = min(1.0, self._add_cancel_ratio * 2.0)
        
        # Combined estimate: weighted average
        # Weight: spoof_fraction (40%), survival_prob (30%), ratio (30%)
        real_fraction = 0.4 * spoof_fraction + 0.3 * survival_prob + 0.3 * ratio_indicator
        
        # Floor: at least 10% real liquidity assumption (never assume 100% fake)
        real_fraction = max(0.1, min(1.0, real_fraction))
        
        # Smooth
        self._real_liquidity_fraction_ema = 0.7 * real_fraction + \
                                             0.3 * self._real_liquidity_fraction_ema
        
        return self._real_liquidity_fraction_ema
    
    def _compute_herfindahl(self, levels: List[OrderbookLevel]) -> float:
        """
        Compute Herfindahl concentration index for depth distribution.
        
        H = sum(s_i^2) / (sum(s_i))^2
        
        H near 0: depth evenly distributed (healthy, competitive)
        H near 1: depth concentrated at few levels (unhealthy, manipulable)
        
        For orderbook: H typically 0.05-0.30
        High H = liquidity concentrated at top = easier to manipulate
        Low H = depth spread across levels = more stable
        """
        if not levels:
            return 0.0
        
        sizes = np.array([l.size for l in levels])
        total = np.sum(sizes)
        
        if total < 1e-10:
            return 0.0
        
        # Herfindahl index
        shares = sizes / total
        H = np.sum(shares ** 2)
        
        # Normalize: for N levels, H ranges from 1/N (even) to 1 (concentrated)
        n = len(levels)
        if n > 1:
            # Normalized H: (H - 1/N) / (1 - 1/N)
            H_normalized = (H - 1.0/n) / (1.0 - 1.0/n)
            H_normalized = max(0.0, min(1.0, H_normalized))
            return H_normalized
        return 1.0
    
    def _compute_stability_score(self, bid_depth: float, ask_depth: float) -> float:
        """
        Compute liquidity stability score from depth variance history.
        
        Stability = 1 - normalized_variance
        where variance is computed over recent depth history.
        
        High stability: depth has been consistent (reliable liquidity)
        Low stability: depth has been fluctuating (unreliable, possible spoof)
        """
        self._bid_depth_history.append(bid_depth)
        self._ask_depth_history.append(ask_depth)
        
        total_depth = bid_depth + ask_depth
        self._depth_ratio_history.append(bid_depth / max(ask_depth, 1e-10))
        
        if len(self._bid_depth_history) < 10:
            return 0.5  # Neutral until enough data
        
        # Compute variance of total depth
        depths = np.array(list(self._bid_depth_history) + 
                         list(self._ask_depth_history))
        
        mean_depth = np.mean(depths)
        var_depth = np.var(depths)
        
        # Normalized variance: var / mean^2 (coefficient of variation squared)
        if mean_depth > 0:
            cv_squared = var_depth / (mean_depth ** 2)
            # Stability = 1 - CV^2, clamped
            stability = max(0.0, min(1.0, 1.0 - cv_squared))
        else:
            stability = 0.0
        
        # Also consider depth ratio stability
        if len(self._depth_ratio_history) >= 10:
            ratios = np.array(list(self._depth_ratio_history))
            ratio_var = np.var(ratios)
            ratio_mean = np.mean(ratios)
            if ratio_mean > 0:
                ratio_cv_sq = ratio_var / (ratio_mean ** 2)
                ratio_stability = max(0.0, min(1.0, 1.0 - ratio_cv_sq))
                # Combined stability
                stability = 0.6 * stability + 0.4 * ratio_stability
        
        # Smooth
        self._stability_score_ema = 0.7 * stability + 0.3 * self._stability_score_ema
        
        return self._stability_score_ema
    
    def _compute_depth_skew_coefficient(self, snapshot: OrderbookSnapshot,
                                         mid_price: float) -> float:
        """
        Compute depth distribution skew coefficient.
        
        Measures asymmetry in how depth is distributed across price levels.
        Uses the ratio of weighted depth near mid vs far from mid.
        
        skew = (near_depth - far_depth) / (near_depth + far_depth)
        
        Positive skew: more depth near mid on bid side (buy support)
        Negative skew: more depth near mid on ask side (sell resistance)
        """
        if mid_price <= 0:
            return 0.0
        
        depth = min(10, len(snapshot.bids), len(snapshot.asks))
        if depth < 4:
            return 0.0
        
        # Split into near (first 3 levels) and far (rest)
        near_bid = sum(l.size for l in snapshot.bids[:3])
        far_bid = sum(l.size for l in snapshot.bids[3:depth])
        
        near_ask = sum(l.size for l in snapshot.asks[:3])
        far_ask = sum(l.size for l in snapshot.asks[3:depth])
        
        # Bid skew: how much bid depth is concentrated near mid
        bid_near_ratio = near_bid / max(near_bid + far_bid, 1e-10)
        # Ask skew: how much ask depth is concentrated near mid
        ask_near_ratio = near_ask / max(near_ask + far_ask, 1e-10)
        
        # Overall skew: bid concentration vs ask concentration
        skew = (bid_near_ratio - ask_near_ratio)
        
        return skew
    
    def _empty_metrics(self, timestamp: float) -> LiquidityMetrics:
        """Return empty metrics when no data available."""
        return LiquidityMetrics(
            timestamp=timestamp,
            total_bid_depth=0.0,
            total_ask_depth=0.0,
            real_bid_depth=0.0,
            real_ask_depth=0.0,
            depth_ratio=1.0,
            liquidity_stability_score=0.5,
            cancellation_velocity=0.0,
            add_cancel_ratio=0.5,
            absorption_score=0.0,
            absorption_volume=0.0,
            vacuum_score=0.0,
            vacuum_depth=0.0,
            real_liquidity_fraction=0.5,
            liquidity_concentration=0.0,
            depth_skew_coefficient=0.0,
        )