"""
Fast reversal intelligence engine.

Mathematical foundation:
- Reversal probability: P(reversal | data) derived from:
  1. Taker inversion: rapid reversal in taker direction (flow inversion score)
  2. Continuation collapse: momentum persistence suddenly drops
  3. Liquidity vacuum: depth removed from continuation side
  4. Absorption wall: large hidden liquidity opposing continuation
  5. Microburst reversal: rapid price reversal after directional burst
  6. Exhaustion: taker aggression declining + flow deceleration
  7. Failed breakout: burst that reverses back through origin level

- Reversal speed metric: how fast reversal indicators are accumulating
  speed = rate_of_change(taker_inversion + vacuum + absorption) / time_window

- Reversal depth estimate: expected magnitude of reversal
  depth = |kalman_velocity| * reversal_probability * volatility_factor

The reversal engine MUST be extremely fast - it cannot rely on lagging indicators.
It uses microstructure-level signals that precede visible price reversals:
- Taker flow inversion happens BEFORE price reversal
- Liquidity vacuum creates the SPACE for reversal
- Absorption walls STOP continuation and ENABLE reversal
- Microburst reversals are the FASTEST reversal signal

All computations use minimal smoothing for speed.
"""

import logging
import time
import numpy as np
from typing import Optional
from collections import deque

from polymarket_bot.config import SignalConfig, RegimeState, Direction
from polymarket_bot.bot_types import (
    ReversalSignal, KalmanState, FlowMetrics, OrderbookPressure,
    RegimeStateEstimate, VolatilityEstimate, LiquidityMetrics, QueueMetrics,
    SpreadMetrics, ExhaustionSignal, BurstFailureSignal
)

logger = logging.getLogger(__name__)


class ReversalEngine:
    """
    Fast reversal detection engine using microstructure precursors.
    
    Key innovations:
    1. Taker inversion detection (flow reversal BEFORE price reversal)
    2. Continuation collapse detection (momentum persistence crash)
    3. Liquidity vacuum detection (space for reversal)
    4. Absorption wall detection (hidden liquidity opposing trend)
    5. Microburst reversal detection (fastest reversal signal)
    6. Failed breakout detection (burst that fails and reverses)
    7. Minimal smoothing for maximum speed
    8. Regime-aware sensitivity (more sensitive in volatile regimes)
    """
    
    def __init__(self, config: SignalConfig):
        self.config = config
        
        # Reversal component tracking
        self._taker_inversion_history: deque = deque(maxlen=50)
        self._continuation_collapse_history: deque = deque(maxlen=50)
        self._vacuum_history: deque = deque(maxlen=50)
        self._absorption_history: deque = deque(maxlen=50)
        self._microburst_history: deque = deque(maxlen=50)
        
        # Reversal probability tracking
        self._reversal_prob_ema = 0.0
        self._reversal_speed_ema = 0.0
        
        # Previous signal
        self._prev_signal: Optional[ReversalSignal] = None
        
        # Regime sensitivity multipliers
        # In volatile/crisis regimes, reversal signals are more significant
        self._regime_sensitivity = {
            RegimeState.CALM_TRENDING: 1.0,
            RegimeState.CALM_RANGE: 0.5,  # Less reversal in calm range
            RegimeState.VOLATILE_TRENDING: 1.5,  # More reversal in volatile
            RegimeState.VOLATILE_RANGE: 1.2,
            RegimeState.CRISIS: 2.0,  # Very high reversal sensitivity
            RegimeState.LIQUIDATION_CASCADE: 2.5,
        }
    
    def generate_signal(
        self,
        kalman_state: KalmanState,
        flow_metrics: FlowMetrics,
        orderbook_pressure: OrderbookPressure,
        regime_estimate: RegimeStateEstimate,
        volatility_estimate: VolatilityEstimate,
        liquidity_metrics: Optional[LiquidityMetrics] = None,
        queue_metrics: Optional[QueueMetrics] = None,
        spread_metrics: Optional[SpreadMetrics] = None,
        exhaustion_signal: Optional[ExhaustionSignal] = None,
        burst_failure_signal: Optional[BurstFailureSignal] = None,
    ) -> ReversalSignal:
        """
        Generate reversal signal from microstructure precursors.
        
        Process:
        1. Determine reversal direction (opposite of current trend)
        2. Compute taker inversion score
        3. Compute continuation collapse score
        4. Compute vacuum score
        5. Compute absorption score
        6. Compute microburst score
        7. Compute exhaustion score
        8. Compute failed breakout score
        9. Combine with regime-adaptive weights
        10. Compute reversal speed and depth
        11. Validate signal
        """
        timestamp = time.time()
        
        # Step 1: Determine reversal direction
        # Reversal direction is opposite of current Kalman velocity
        velocity = kalman_state.velocity_estimate
        
        if velocity > 0:
            reversal_direction = Direction.DOWN  # Trend is UP, reversal would be DOWN
        elif velocity < 0:
            reversal_direction = Direction.UP  # Trend is DOWN, reversal would be UP
        else:
            reversal_direction = Direction.NEUTRAL
        
        # Step 2: Taker inversion score
        # High inversion = taker flow has reversed direction
        taker_inversion = flow_metrics.taker_inversion
        
        # Also check if taker ratio has flipped
        if reversal_direction == Direction.UP:
            # Need taker buy ratio to increase
            taker_flip = max(0, flow_metrics.taker_ratio - 0.5) * 2.0
        elif reversal_direction == Direction.DOWN:
            # Need taker sell ratio to increase
            taker_flip = max(0, 0.5 - flow_metrics.taker_ratio) * 2.0
        else:
            taker_flip = 0.0
        
        taker_inversion_score = max(taker_inversion, taker_flip)
        self._taker_inversion_history.append(taker_inversion_score)
        
        # Step 3: Continuation collapse score
        # Momentum persistence suddenly drops = continuation is collapsing
        persistence = flow_metrics.flow_persistence
        prev_persistence = self._prev_signal is not None
        
        # Collapse: persistence drops below threshold
        collapse_threshold = self.config.reversal_continuation_collapse_rate
        if persistence < collapse_threshold:
            continuation_collapse = min(1.0, (collapse_threshold - persistence) / collapse_threshold)
        else:
            continuation_collapse = 0.0
        
        # Also check acceleration reversal
        acceleration = kalman_state.acceleration_estimate
        if reversal_direction == Direction.UP and acceleration > 0:
            # Positive acceleration when we expect DOWN reversal = no collapse
            accel_collapse = 0.0
        elif reversal_direction == Direction.DOWN and acceleration < 0:
            accel_collapse = 0.0
        else:
            # Acceleration supports reversal direction
            accel_collapse = min(1.0, abs(acceleration) / max(abs(velocity), 1e-10))
        
        continuation_collapse_score = max(continuation_collapse, accel_collapse)
        self._continuation_collapse_history.append(continuation_collapse_score)
        
        # Step 4: Vacuum score
        # Liquidity vacuum on the continuation side creates space for reversal
        vacuum_score = 0.0
        if liquidity_metrics is not None:
            # Vacuum on bid side = space for DOWN reversal
            # Vacuum on ask side = space for UP reversal
            if reversal_direction == Direction.DOWN:
                # Need bid vacuum (buyers pulled)
                vacuum_score = liquidity_metrics.vacuum_score
                # Check if depth ratio shifted against continuation
                if liquidity_metrics.depth_ratio < 0.8:  # More ask depth than bid
                    vacuum_score = max(vacuum_score, 1.0 - liquidity_metrics.depth_ratio)
            elif reversal_direction == Direction.UP:
                # Need ask vacuum (sellers pulled)
                vacuum_score = liquidity_metrics.vacuum_score
                if liquidity_metrics.depth_ratio > 1.2:  # More bid depth than ask
                    vacuum_score = max(vacuum_score, liquidity_metrics.depth_ratio - 1.0)
        
        vacuum_score = min(1.0, vacuum_score * self.config.reversal_vacuum_sensitivity)
        self._vacuum_history.append(vacuum_score)
        
        # Step 5: Absorption score
        # Absorption wall opposing continuation = reversal catalyst
        absorption_score = 0.0
        if liquidity_metrics is not None:
            absorption_score = liquidity_metrics.absorption_score
            
            # Check if absorption is on the reversal side
            # (absorbing continuation orders = building reversal wall)
            if reversal_direction == Direction.UP:
                # Need absorption on bid side (absorbing sell orders)
                if orderbook_pressure.real_imbalance > 0:
                    absorption_score *= 1.5  # Absorption + bid pressure = strong reversal
            elif reversal_direction == Direction.DOWN:
                if orderbook_pressure.real_imbalance < 0:
                    absorption_score *= 1.5
        
        absorption_score = min(1.0, absorption_score)
        self._absorption_history.append(absorption_score)
        
        # Step 6: Microburst score
        # Rapid price reversal after directional burst
        microburst_score = 0.0
        
        # Detect microburst from acceleration reversal
        if abs(acceleration) > 0:
            # Microburst: acceleration is opposite to velocity
            if (velocity > 0 and acceleration < 0) or (velocity < 0 and acceleration > 0):
                # Acceleration opposing velocity = microburst reversal
                burst_ratio = abs(acceleration) / max(abs(velocity), 1e-10)
                microburst_score = min(1.0, burst_ratio * self.config.reversal_microburst_velocity)
        
        # Also check from burst failure signal
        if burst_failure_signal is not None and burst_failure_signal.is_valid:
            microburst_score = max(microburst_score, burst_failure_signal.burst_failure_probability)
        
        self._microburst_history.append(microburst_score)
        
        # Step 7: Exhaustion score
        exhaustion_score = 0.0
        if exhaustion_signal is not None and exhaustion_signal.is_valid:
            exhaustion_score = exhaustion_signal.exhaustion_probability
        else:
            # Quick exhaustion estimate from flow metrics
            if flow_metrics.taker_aggression_score < 0.3:
                exhaustion_score = 0.5 * (1.0 - flow_metrics.taker_aggression_score)
            if abs(flow_metrics.flow_acceleration) < 0.01:
                exhaustion_score = max(exhaustion_score, 0.3)
        
        # Step 8: Failed breakout score
        failed_breakout_score = 0.0
        if burst_failure_signal is not None and burst_failure_signal.is_valid:
            failed_breakout_score = burst_failure_signal.burst_failure_probability
        
        # Step 9: Combine with regime-adaptive weights
        regime = regime_estimate.current_regime
        sensitivity = self._regime_sensitivity.get(regime, 1.0)
        
        # Reversal component weights
        # [taker_inversion, continuation_collapse, vacuum, absorption, 
        #  microburst, exhaustion, failed_breakout]
        weights = np.array([0.25, 0.15, 0.15, 0.15, 0.15, 0.10, 0.05])
        
        components = np.array([
            taker_inversion_score,
            continuation_collapse_score,
            vacuum_score,
            absorption_score,
            microburst_score,
            exhaustion_score,
            failed_breakout_score,
        ])
        
        # Weighted combination with regime sensitivity
        raw_reversal = np.sum(weights * components) * sensitivity
        
        # Sigmoid transformation
        reversal_probability = 1.0 / (1.0 + np.exp(-raw_reversal * 4.0))
        
        # Step 10: Reversal speed
        # How fast reversal indicators are accumulating
        speed_components = [
            taker_inversion_score,
            continuation_collapse_score,
            microburst_score,
        ]
        reversal_speed = np.mean(speed_components) * sensitivity
        
        # Step 11: Reversal depth estimate
        # Expected magnitude of reversal
        vol_factor = volatility_estimate.regime_adjusted_volatility
        reversal_depth = abs(velocity) * reversal_probability * vol_factor * 5.0
        reversal_depth = min(1.0, reversal_depth)
        
        # Signal quality
        regime_confidence = regime_estimate.regime_confidence
        velocity_confidence = kalman_state.velocity_confidence
        
        # Quality requires multiple components to agree
        n_active_components = sum(1 for c in components if c > 0.2)
        component_agreement = n_active_components / len(components)
        
        signal_quality = 0.4 * component_agreement + \
                          0.3 * regime_confidence + \
                          0.2 * velocity_confidence + \
                          0.1 * reversal_speed
        
        signal_quality = min(1.0, signal_quality)
        
        # Validation
        is_valid = (
            reversal_direction != Direction.NEUTRAL and
            signal_quality > 0.3 and
            reversal_probability > 0.4 and
            n_active_components >= 2  # At least 2 components must agree
        )
        
        # Build result
        result = ReversalSignal(
            timestamp=timestamp,
            direction=reversal_direction,
            reversal_probability=reversal_probability,
            reversal_speed=reversal_speed,
            taker_inversion_score=taker_inversion_score,
            continuation_collapse_score=continuation_collapse_score,
            vacuum_score=vacuum_score,
            absorption_score=absorption_score,
            microburst_score=microburst_score,
            exhaustion_score=exhaustion_score,
            failed_breakout_score=failed_breakout_score,
            reversal_depth=reversal_depth,
            signal_quality=signal_quality,
            is_valid=is_valid,
        )
        
        self._prev_signal = result
        return result