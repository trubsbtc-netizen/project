"""
Continuation exhaustion detection engine.

Mathematical foundation:
- Exhaustion probability: P(exhaustion | data) derived from:
  1. Taker aggression decay: exponential decay in taker aggression rate
     decay_rate = -d(aggression)/dt, measured via finite differences
  2. Flow deceleration: rate of change of flow imbalance approaching zero
     deceleration = |d(imbalance)/dt| when approaching zero
  3. Volume climax: volume spike followed by rapid decline
     climax_ratio = max(volume) / mean(volume_recent)
  4. Absorption behind: hidden liquidity absorbing continuation orders
     absorption_behind = absorption_score on continuation side
  5. Spread expansion: spread widening during directional move
     expansion = spread_expansion_score during continuation
  6. Depth imbalance shift: orderbook depth shifting against continuation
     shift = change in depth_ratio opposing direction

- Exhaustion severity: how complete the exhaustion is
  severity = weighted_sum(components) with higher weight on flow decay

Exhaustion detection is critical for:
1. Reducing continuation conviction before reversal
2. Early reversal detection (exhaustion precedes reversal)
3. Avoiding entry at the end of a move (late continuation)
"""

import logging
import time
import numpy as np
from typing import Optional
from collections import deque

from polymarket_bot.config import SignalConfig, Direction
from polymarket_bot.bot_types import (
    ExhaustionSignal, FlowMetrics, OrderbookPressure,
    VolatilityEstimate, LiquidityMetrics, SpreadMetrics, KalmanState
)

logger = logging.getLogger(__name__)


class ExhaustionDetector:
    """
    Detects continuation exhaustion before reversal occurs.
    
    Key innovations:
    1. Taker aggression decay rate estimation
    2. Flow deceleration detection (momentum slowing)
    3. Volume climax identification (volume spike then decline)
    4. Absorption behind the move (hidden counter-liquidity)
    5. Spread expansion during move (market stress)
    6. Depth imbalance shift (liquidity migrating against move)
    """
    
    def __init__(self, config: SignalConfig):
        self.config = config
        
        # Taker aggression tracking for decay estimation
        self._aggression_history: deque = deque(maxlen=50)
        self._aggression_decay_rate = 0.0
        
        # Flow imbalance tracking for deceleration
        self._imbalance_history: deque = deque(maxlen=50)
        self._flow_deceleration = 0.0
        
        # Volume tracking for climax detection
        self._volume_history: deque = deque(maxlen=100)
        self._volume_mean_ema = 0.0
        
        # Previous signal
        self._prev_signal: Optional[ExhaustionSignal] = None
    
    def detect(
        self,
        flow_metrics: FlowMetrics,
        orderbook_pressure: OrderbookPressure,
        volatility_estimate: VolatilityEstimate,
        kalman_state: KalmanState,
        liquidity_metrics: Optional[LiquidityMetrics] = None,
        spread_metrics: Optional[SpreadMetrics] = None,
    ) -> ExhaustionSignal:
        """
        Detect continuation exhaustion from microstructure signals.
        
        Process:
        1. Determine direction being exhausted
        2. Compute taker decay rate
        3. Compute flow deceleration
        4. Compute volume climax score
        5. Compute absorption behind score
        6. Compute spread expansion score
        7. Compute depth imbalance shift
        8. Combine into exhaustion probability
        9. Compute severity
        """
        timestamp = time.time()
        
        # Step 1: Direction being exhausted
        velocity = kalman_state.velocity_estimate
        if velocity > 0:
            direction = Direction.UP  # UP move being exhausted
        elif velocity < 0:
            direction = Direction.DOWN  # DOWN move being exhausted
        else:
            direction = Direction.NEUTRAL
        
        # Step 2: Taker aggression decay rate
        self._aggression_history.append(flow_metrics.taker_aggression_score)
        decay_rate = self._compute_decay_rate(self._aggression_history)
        self._aggression_decay_rate = decay_rate
        
        # Step 3: Flow deceleration
        self._imbalance_history.append(flow_metrics.flow_imbalance)
        deceleration = self._compute_deceleration()
        self._flow_deceleration = deceleration
        
        # Step 4: Volume climax score
        self._volume_history.append(flow_metrics.total_volume)
        volume_climax = self._compute_volume_climax()
        
        # Step 5: Absorption behind the move
        absorption_behind = 0.0
        if liquidity_metrics is not None:
            absorption_behind = liquidity_metrics.absorption_score
            # Absorption is more significant when it's on the opposing side
            if direction == Direction.UP and orderbook_pressure.real_imbalance < 0:
                absorption_behind *= 1.5  # Sell absorption behind UP move
            elif direction == Direction.DOWN and orderbook_pressure.real_imbalance > 0:
                absorption_behind *= 1.5  # Buy absorption behind DOWN move
        
        # Step 6: Spread expansion during move
        spread_expansion = 0.0
        if spread_metrics is not None:
            spread_expansion = spread_metrics.spread_expansion_score
        
        # Step 7: Depth imbalance shift
        depth_shift = 0.0
        if liquidity_metrics is not None:
            # Depth ratio shifting against continuation direction
            if direction == Direction.UP:
                # Shift: ask depth growing relative to bid depth
                depth_shift = max(0, 1.0 - liquidity_metrics.depth_ratio) * 0.5
            elif direction == Direction.DOWN:
                # Shift: bid depth growing relative to ask depth
                depth_shift = max(0, liquidity_metrics.depth_ratio - 1.0) * 0.5
        
        # Step 8: Combine into exhaustion probability
        # Weights: [decay, deceleration, climax, absorption, spread, depth_shift]
        weights = np.array([0.25, 0.20, 0.15, 0.20, 0.10, 0.10])
        
        components = np.array([
            decay_rate,
            deceleration,
            volume_climax,
            absorption_behind,
            spread_expansion,
            depth_shift,
        ])
        
        raw_exhaustion = np.sum(weights * components)
        
        # Sigmoid transformation
        exhaustion_probability = 1.0 / (1.0 + np.exp(-raw_exhaustion * 3.0))
        
        # Step 9: Severity
        severity = min(1.0, raw_exhaustion)
        
        # Validation
        is_valid = (
            direction != Direction.NEUTRAL and
            exhaustion_probability > 0.3 and
            (decay_rate > 0.1 or deceleration > 0.1 or absorption_behind > 0.2)
        )
        
        result = ExhaustionSignal(
            timestamp=timestamp,
            direction=direction,
            exhaustion_probability=exhaustion_probability,
            taker_decay_rate=decay_rate,
            flow_deceleration=deceleration,
            volume_climax_score=volume_climax,
            absorption_behind=absorption_behind,
            spread_expansion=spread_expansion,
            depth_imbalance_shift=depth_shift,
            exhaustion_severity=severity,
            is_valid=is_valid,
        )
        
        self._prev_signal = result
        return result
    
    def _compute_decay_rate(self, history: deque) -> float:
        """
        Compute exponential decay rate of a signal series.
        
        Method: fit exponential decay y(t) = A * exp(-lambda * t) + C
        to the recent values of the signal.
        
        Simplified: compute rate of change of last few values.
        decay_rate = max(0, -d(signal)/dt) normalized
        
        Returns the rate at which the signal is declining.
        """
        if len(history) < 5:
            return 0.0
        
        values = np.array(list(history)[-10:])
        n = len(values)
        
        if n < 3:
            return 0.0
        
        # Compute finite difference rate of change
        changes = np.diff(values)
        
        # Decay = negative changes (signal declining)
        negative_changes = changes[changes < 0]
        
        if len(negative_changes) == 0:
            return 0.0  # No decay
        
        # Average decay rate
        avg_decay = np.mean(np.abs(negative_changes))
        
        # Normalize by current value
        current_value = values[-1]
        if current_value > 0:
            normalized_decay = avg_decay / current_value
        else:
            normalized_decay = 0.0
        
        return min(1.0, normalized_decay * self.config.exhaustion_taker_decay_rate * 10)
    
    def _compute_deceleration(self) -> float:
        """
        Compute flow imbalance deceleration (momentum slowing).
        
        Deceleration = rate at which flow imbalance is approaching zero.
        
        If imbalance was 0.5 and now 0.2, deceleration = 0.3 (significant)
        If imbalance was 0.5 and now 0.4, deceleration = 0.1 (moderate)
        If imbalance was 0.5 and now 0.6, deceleration = 0 (accelerating, not decelerating)
        """
        if len(self._imbalance_history) < 5:
            return 0.0
        
        values = np.array(list(self._imbalance_history)[-10:])
        
        # Check if imbalance magnitude is decreasing
        magnitudes = np.abs(values)
        
        if len(magnitudes) < 3:
            return 0.0
        
        # Compute change in magnitude
        magnitude_changes = np.diff(magnitudes)
        
        # Deceleration = magnitude decreasing
        deceleration_changes = magnitude_changes[magnitude_changes < 0]
        
        if len(deceleration_changes) == 0:
            return 0.0
        
        avg_deceleration = np.mean(np.abs(deceleration_changes))
        
        # Normalize
        current_magnitude = magnitudes[-1]
        if current_magnitude > 0:
            normalized = avg_deceleration / current_magnitude
        else:
            normalized = avg_deceleration
        
        return min(1.0, normalized * self.config.exhaustion_flow_deceleration_threshold * 5)
    
    def _compute_volume_climax(self) -> float:
        """
        Detect volume climax (spike followed by decline).
        
        climax_ratio = recent_max_volume / recent_mean_volume
        
        A climax occurs when:
        1. Recent volume was significantly above average
        2. Volume is now declining from the peak
        
        climax_score = (climax_ratio - 1) * decline_factor
        """
        if len(self._volume_history) < 10:
            return 0.0
        
        volumes = np.array(list(self._volume_history)[-50:])
        
        if len(volumes) < 5:
            return 0.0
        
        # Compute mean and max
        mean_vol = np.mean(volumes)
        max_vol = np.max(volumes)
        
        if mean_vol <= 0:
            return 0.0
        
        # Climax ratio
        climax_ratio = max_vol / mean_vol
        
        # Check if volume is declining from peak
        # Find peak position
        peak_idx = np.argmax(volumes)
        current_idx = len(volumes) - 1
        
        if peak_idx < current_idx:
            # Volume declining from peak
            decline = (max_vol - volumes[-1]) / max_vol
            climax_score = (climax_ratio - 1.0) * decline
        else:
            # Volume still at peak (not yet declining)
            climax_score = 0.0
        
        # Threshold: only significant if ratio > config threshold
        if climax_ratio < self.config.exhaustion_volume_climax_ratio:
            climax_score *= 0.5  # Reduce significance
        
        return min(1.0, max(0.0, climax_score))