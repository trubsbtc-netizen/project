"""
Adaptive momentum continuation engine.

Mathematical foundation:
- Continuation probability: P(continuation | data) derived from:
  1. Kalman velocity/acceleration (trend speed and change)
  2. Flow imbalance persistence (autocorrelation of directional flow)
  3. Regime support (regime that favors continuation)
  4. Orderbook pressure confirmation (real pressure supports direction)
  5. Momentum strength (magnitude of directional signal)
  6. Exhaustion risk (counter-signal that may terminate continuation)

- Momentum strength: |flow_imbalance| * |kalman_velocity| * regime_persistence
  Normalized to [0, 1] range via sigmoid transformation

- Continuation probability model:
  P(cont) = sigmoid(w1*velocity + w2*flow + w3*regime + w4*pressure - w5*exhaustion)
  
  where weights are regime-adaptive:
  - In trending regimes: higher weight on velocity and flow
  - In range regimes: lower weight on velocity (noise), higher on exhaustion
  - In crisis regimes: lower weight on all (high uncertainty)

- Entropy-aware suppression: high signal entropy reduces continuation conviction
  suppression_factor = 1 - entropy_score * suppression_threshold

The continuation engine must NOT over-suppress valid momentum.
It maintains aggressive conviction when market conditions are healthy.
"""

import logging
import time
import numpy as np
from typing import Optional
from collections import deque

from polymarket_bot.config import SignalConfig, RegimeState, Direction
from polymarket_bot.bot_types import (
    ContinuationSignal, KalmanState, FlowMetrics, OrderbookPressure,
    RegimeStateEstimate, VolatilityEstimate, LiquidityMetrics, EntropyFilterResult
)

logger = logging.getLogger(__name__)


class ContinuationEngine:
    """
    Adaptive momentum continuation signal generator.
    
    Key innovations:
    1. Regime-adaptive weight scheme for signal components
    2. Kalman velocity/acceleration for trend speed estimation
    3. Flow persistence confirmation (not just current flow)
    4. Spoof-filtered orderbook pressure confirmation
    5. Entropy-aware conviction scaling
    6. Exhaustion risk integration (early warning)
    7. Sigmoid probability model with regime-adaptive weights
    """
    
    def __init__(self, config: SignalConfig):
        self.config = config
        
        # Regime-adaptive weight schemes
        # [velocity, flow, regime, pressure, exhaustion]
        self._regime_weights = {
            RegimeState.CALM_TRENDING: np.array([0.30, 0.25, 0.20, 0.15, 0.10]),
            RegimeState.CALM_RANGE: np.array([0.15, 0.20, 0.10, 0.15, 0.40]),
            RegimeState.VOLATILE_TRENDING: np.array([0.25, 0.30, 0.15, 0.20, 0.10]),
            RegimeState.VOLATILE_RANGE: np.array([0.10, 0.15, 0.10, 0.15, 0.50]),
            RegimeState.CRISIS: np.array([0.10, 0.10, 0.05, 0.10, 0.65]),
            RegimeState.LIQUIDATION_CASCADE: np.array([0.05, 0.05, 0.05, 0.05, 0.80]),
        }
        
        # Momentum tracking
        self._momentum_history: deque = deque(maxlen=100)
        self._continuation_prob_history: deque = deque(maxlen=100)
        
        # Previous signal for smoothing
        self._prev_signal: Optional[ContinuationSignal] = None
        
        # Sigmoid parameters
        self._sigmoid_temperature = 2.0  # Controls sharpness of probability
    
    def generate_signal(
        self,
        kalman_state: KalmanState,
        flow_metrics: FlowMetrics,
        orderbook_pressure: OrderbookPressure,
        regime_estimate: RegimeStateEstimate,
        volatility_estimate: VolatilityEstimate,
        liquidity_metrics: Optional[LiquidityMetrics] = None,
        entropy_filter: Optional[EntropyFilterResult] = None,
    ) -> ContinuationSignal:
        """
        Generate continuation signal from all microstructure inputs.
        
        Process:
        1. Determine direction from Kalman velocity
        2. Compute momentum strength from flow and velocity
        3. Compute regime support score
        4. Compute flow confirmation score
        5. Compute orderbook pressure confirmation
        6. Compute exhaustion risk
        7. Apply regime-adaptive weights to compute probability
        8. Apply entropy suppression
        9. Validate signal quality
        """
        timestamp = time.time()
        
        # Step 1: Determine direction from Kalman velocity
        velocity = kalman_state.velocity_estimate
        acceleration = kalman_state.acceleration_estimate
        price_scale = float(kalman_state.state[0]) if len(kalman_state.state) else 0.0
        price_scale = max(abs(price_scale), 1.0e-9)
        log_velocity = velocity / price_scale
        
        if velocity > 0:
            direction = Direction.UP
        elif velocity < 0:
            direction = Direction.DOWN
        else:
            direction = Direction.NEUTRAL
        
        # Step 2: Compute momentum strength
        # momentum = |velocity_normalized| * |flow_imbalance| * persistence
        velocity_strength = min(abs(log_velocity) / max(volatility_estimate.kalman_volatility, 1e-10), 3.0)
        velocity_strength = velocity_strength / 3.0  # Normalize to [0, 1]
        
        flow_strength = abs(flow_metrics.flow_imbalance)
        persistence = max(flow_metrics.flow_persistence, 0.0)
        
        momentum_strength = velocity_strength * (0.5 + 0.5 * flow_strength) * \
                           (0.3 + 0.7 * max(persistence, 0.1))
        momentum_strength = min(1.0, momentum_strength)
        
        # Step 3: Regime support score
        regime_support = self._compute_regime_support(regime_estimate, direction)
        
        # Step 4: Flow confirmation score
        # Flow confirms continuation if imbalance aligns with direction
        if direction == Direction.UP:
            flow_confirmation = max(0, flow_metrics.flow_imbalance) * \
                               max(0, flow_metrics.taker_ratio - 0.5) * 2.0
        elif direction == Direction.DOWN:
            flow_confirmation = max(0, -flow_metrics.flow_imbalance) * \
                               max(0, 0.5 - flow_metrics.taker_ratio) * 2.0
        else:
            flow_confirmation = 0.0
        
        flow_confirmation = min(1.0, flow_confirmation)
        
        # Step 5: Orderbook pressure confirmation
        # Real pressure (after spoof filtering) confirms direction
        if direction == Direction.UP:
            pressure_confirmation = max(0, orderbook_pressure.real_imbalance)
        elif direction == Direction.DOWN:
            pressure_confirmation = max(0, -orderbook_pressure.real_imbalance)
        else:
            pressure_confirmation = 0.0
        
        pressure_confirmation = min(1.0, abs(pressure_confirmation))
        
        # Step 6: Exhaustion risk
        exhaustion_risk = self._compute_exhaustion_risk(
            flow_metrics, volatility_estimate, direction
        )
        
        # Step 7: Apply regime-adaptive weights
        regime = regime_estimate.current_regime
        weights = self._regime_weights.get(regime, self._regime_weights[RegimeState.CALM_TRENDING])
        
        # Signal components: [velocity, flow, regime, pressure, exhaustion]
        # Note: exhaustion is a negative signal (reduces continuation probability)
        components = np.array([
            velocity_strength,
            flow_confirmation,
            regime_support,
            pressure_confirmation,
            exhaustion_risk,  # This reduces probability
        ])
        
        # Weighted sum: positive components minus exhaustion
        positive_signal = weights[0] * components[0] + \
                         weights[1] * components[1] + \
                         weights[2] * components[2] + \
                         weights[3] * components[3]
        negative_signal = weights[4] * components[4]
        
        net_signal = positive_signal - negative_signal
        
        # Sigmoid transformation for probability
        continuation_probability = self._sigmoid(net_signal, self._sigmoid_temperature)
        
        # Step 8: Entropy suppression
        entropy_score = 0.0
        if entropy_filter is not None:
            entropy_score = entropy_filter.entropy_score
            # Suppress if entropy is high
            suppression = entropy_filter.suppression_factor
            continuation_probability *= suppression
        
        # Step 9: Signal quality validation
        # Quality = confidence in the signal components
        velocity_confidence = kalman_state.velocity_confidence
        regime_confidence = regime_estimate.regime_confidence
        
        signal_quality = 0.4 * velocity_confidence + \
                        0.3 * regime_confidence + \
                        0.2 * flow_metrics.flow_persistence + \
                        0.1 * (1.0 - entropy_score)
        
        signal_quality = min(1.0, signal_quality)
        
        # Validation: signal must have minimum quality and direction
        is_valid = (
            direction != Direction.NEUTRAL and
            signal_quality > 0.3 and
            continuation_probability > 0.4 and
            regime_support > self.config.continuation_min_regime_persistence
        )
        
        # Persistence score (momentum continuation likelihood)
        persistence_score = max(0, flow_metrics.flow_persistence) * \
                           max(0, regime_estimate.regime_persistence)
        persistence_score = min(1.0, persistence_score)
        
        # Build result
        result = ContinuationSignal(
            timestamp=timestamp,
            direction=direction,
            continuation_probability=continuation_probability,
            momentum_strength=momentum_strength,
            flow_confirmation=flow_confirmation,
            regime_support=regime_support,
            persistence_score=persistence_score,
            exhaustion_risk=exhaustion_risk,
            entropy_score=entropy_score,
            kalman_velocity=velocity,
            kalman_acceleration=acceleration,
            signal_quality=signal_quality,
            is_valid=is_valid,
        )
        
        # Track history
        self._momentum_history.append(momentum_strength)
        self._continuation_prob_history.append(continuation_probability)
        self._prev_signal = result
        
        return result
    
    def _compute_regime_support(self, regime_estimate: RegimeStateEstimate,
                                 direction: Direction) -> float:
        """
        Compute how much the current regime supports continuation in given direction.
        
        Support score based on:
        1. Regime type: trending regimes support continuation
        2. Regime persistence: high persistence = continuation likely
        3. Regime flow characteristic: directional flow = continuation
        4. Regime confidence: confident classification = reliable support
        """
        regime = regime_estimate.current_regime
        persistence = regime_estimate.regime_persistence
        flow_char = regime_estimate.regime_flow_characteristic
        confidence = regime_estimate.regime_confidence
        
        # Regime type support
        if regime in (RegimeState.CALM_TRENDING, RegimeState.VOLATILE_TRENDING):
            type_support = 0.8  # Trending regimes strongly support continuation
        elif regime in (RegimeState.CALM_RANGE, RegimeState.VOLATILE_RANGE):
            type_support = 0.2  # Range regimes oppose continuation
        elif regime == RegimeState.CRISIS:
            type_support = 0.3  # Crisis: uncertain, moderate support
        elif regime == RegimeState.LIQUIDATION_CASCADE:
            type_support = 0.6  # Cascade: strong directional but risky
        else:
            type_support = 0.3
        
        # Direction alignment with regime flow
        if direction == Direction.UP and flow_char > 0:
            direction_alignment = min(1.0, abs(flow_char))
        elif direction == Direction.DOWN and flow_char < 0:
            direction_alignment = min(1.0, abs(flow_char))
        else:
            direction_alignment = 0.0  # Direction opposes regime flow
        
        # Combined support
        support = 0.4 * type_support + \
                 0.3 * persistence + \
                 0.2 * direction_alignment + \
                 0.1 * confidence
        
        return min(1.0, support)
    
    def _compute_exhaustion_risk(self, flow_metrics: FlowMetrics,
                                  volatility_estimate: VolatilityEstimate,
                                  direction: Direction) -> float:
        """
        Compute risk that current momentum is exhausting.
        
        Exhaustion indicators:
        1. Taker aggression decay: aggression declining = exhaustion
        2. Flow deceleration: imbalance acceleration negative = slowing
        3. Volume climax: unusually high volume then decline = climax
        4. Volatility expansion: vol increasing against trend = exhaustion
        5. Taker inversion: taker direction starting to reverse
        
        The exhaustion risk is a counter-signal that reduces continuation conviction.
        """
        # Taker aggression decay
        aggression = flow_metrics.taker_aggression_score
        # If aggression is low, momentum may be exhausting
        aggression_exhaustion = 1.0 - aggression
        
        # Flow deceleration
        # Negative acceleration = flow imbalance is weakening
        if direction == Direction.UP and flow_metrics.flow_acceleration < 0:
            deceleration = min(1.0, abs(flow_metrics.flow_acceleration) * 5.0)
        elif direction == Direction.DOWN and flow_metrics.flow_acceleration > 0:
            deceleration = min(1.0, abs(flow_metrics.flow_acceleration) * 5.0)
        else:
            deceleration = 0.0
        
        # Taker inversion
        inversion = flow_metrics.taker_inversion
        
        # Volatility expansion against trend
        vol_skew = volatility_estimate.volatility_skew
        if direction == Direction.UP and vol_skew < 0:
            vol_exhaustion = min(1.0, abs(vol_skew))
        elif direction == Direction.DOWN and vol_skew > 0:
            vol_exhaustion = min(1.0, abs(vol_skew))
        else:
            vol_exhaustion = 0.0
        
        # Combined exhaustion risk
        exhaustion = 0.3 * aggression_exhaustion + \
                    0.25 * deceleration + \
                    0.25 * inversion + \
                    0.2 * vol_exhaustion
        
        return min(1.0, exhaustion)
    
    def _sigmoid(self, x: float, temperature: float = 2.0) -> float:
        """
        Sigmoid function with temperature parameter.
        
        P = 1 / (1 + exp(-x / temperature))
        
        Higher temperature: softer probability (less extreme)
        Lower temperature: sharper probability (more decisive)
        
        Temperature is regime-adaptive:
        - Calm regimes: lower temperature (more decisive)
        - Volatile regimes: higher temperature (more cautious)
        """
        x_scaled = x / temperature
        # Numerically stable sigmoid
        if x_scaled > 20:
            return 1.0
        elif x_scaled < -20:
            return 0.0
        return 1.0 / (1.0 + np.exp(-x_scaled))
