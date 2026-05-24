"""
Directional hazard rate modeling for continuation decay estimation.

Mathematical foundation:
- Hazard rate: instantaneous rate of event occurrence at time t
  h(t) = f(t) / S(t) where f(t) is probability density, S(t) = survival function
  For continuous-time Cox proportional hazard model:
  h(t) = lambda(t) * exp(beta * X(t))
  where lambda(t) is baseline hazard and X(t) = covariates vector [velocity, flow, regime]

  
- Directional hazard: h_up(t) = lambda_up * h(t) for P(UP | state)
  h_down(t) = lambda_down * h(t) in P(DOWN | state)
  
- Continuation probability decay: P(continuation | data) = exp(-decay_rate * t)
  where decay_rate = -log(max(flow_persistence, 0.01))
  Higher persistence = slower decay (continuation lasts longer)
  
- Reversal hazard rate: h_reversal(t) = reversal_probability * reversal_speed
  The reversal hazard rate combines reversal probability and reversal speed
  
- Settlement hazard: P(settlement direction | data) = posterior * settlement_hazard_rate
  h_settlement_up = posterior.p_up * settlement_hazard_rate
  h_settlement_down = posterior.p_down * settlement_hazard_rate
  
- Time-to-reversal estimate: E[T_reversal] = 1 / reversal_hazard_rate
  If reversal_hazard_rate = 0: E[T] = infinity (no reversal expected)
  
- Volatility adjustment: hazard rates scaled by regime-adjusted volatility
  h_adjusted = h * volatility_multiplier
"""

import logging
import time
import numpy as np
from typing import Optional
from collections import deque

from polymarket_bot.config import RegimeState, Direction
from polymarket_bot.bot_types import (
    HazardEstimate, KalmanState, FlowMetrics, OrderbookPressure,
    RegimeStateEstimate, VolatilityEstimate, BayesianPosterior
)

logger = logging.getLogger(__name__)


class HazardModel:
    """
    Directional hazard rate modeling for continuation decay estimation.
    
    Key innovations:
    1. Cox proportional hazard model for continuous-time hazard
    2. Continuation decay rate from flow persistence
    3. Reversal hazard rate from reversal probability and reversal speed
    4. Settlement hazard rate from settlement probability
    5. Volatility-adjusted hazard rates
    6. Time-to-reversal estimate
    """
    
    def __init__(self):
        self._up_hazard_rate = 0.0
        self._down_hazard_rate = 0.0
        self._continuation_decay_rate = 0.05  # Initial estimate
        self._reversal_hazard_rate = 0.0
        
        self._settlement_hazard_up = 0.0
        self._settlement_hazard_down = 0.0
        
        # History tracking
        self._hazard_history: deque = deque(maxlen=200)
        self._continuation_decay_history: deque = deque(maxlen=200)
        
        # Previous estimate
        self._prev_estimate: Optional[HazardEstimate] = None
        self._initialized = False
    
    def estimate(
        self,
        kalman_state: KalmanState,
        flow_metrics: FlowMetrics,
        orderbook_pressure: OrderbookPressure,
        regime_estimate: RegimeStateEstimate,
        volatility_estimate: VolatilityEstimate,
        posterior: BayesianPosterior,
        time_to_settlement: float = 300.0,  # 5 minutes default
    ) -> HazardEstimate:
        """
        Estimate directional hazard rates from all microstructure inputs.
        
        Process:
        1. Compute Cox proportional hazard rate from Kalman velocity
        2. Compute continuation decay rate from flow persistence
        3. Compute reversal hazard rate from reversal probability
        4. Compute settlement hazard rate from settlement probability
        5. Apply volatility adjustment
        6. Compute time-to-reversal estimate
        """
        timestamp = time.time()
        
        # Step 1: Cox proportional hazard rate from Kalman velocity
        # h(t) = lambda * |velocity| /sigma
        # lambda = regime-specific base hazard rate
        velocity = kalman_state.velocity_estimate
        price_scale = float(kalman_state.state[0]) if len(kalman_state.state) else 0.0
        price_scale = max(abs(price_scale), 1.0e-9)
        log_velocity = velocity / price_scale
        sigma = volatility_estimate.regime_adjusted_volatility
        
        regime = regime_estimate.current_regime
        
        # Regime-specific lambda (base hazard rate)
        regime_lambda = {
            RegimeState.CALM_TRENDING: 0.5,
            RegimeState.CALM_RANGE: 0.3,
            RegimeState.VOLATILE_TRENDING: 1.5,
            RegimeState.VOLATILE_RANGE: 1.0,
            RegimeState.CRISIS: 2.0,
            RegimeState.LIQUIDATION_CASCADE: 3.0,
        }
        
        base_lambda = regime_lambda.get(regime, 0.5)
        
        # Direction-specific hazard
        # UP hazard: proportional to upward velocity
        # DOWN hazard: proportional to downward velocity
        velocity_normalized = abs(log_velocity) / max(sigma, 1e-10)
        
        if velocity > 0:
            up_hazard = base_lambda * velocity_normalized
            down_hazard = base_lambda * velocity_normalized * 0.3  # Lower base for opposing direction
        elif velocity < 0:
            up_hazard = base_lambda * velocity_normalized * 0.3  # Lower base for opposing direction
            down_hazard = base_lambda * velocity_normalized
        else:
            up_hazard = base_lambda * 0.1  # Minimal baseline
            down_hazard = base_lambda * 0.1
        
        # Step 2: Continuation decay rate from flow persistence
        # P(continuation) = exp(-decay_rate * t)
        # decay_rate = -log(max(flow_persistence, 0.01))
        # Higher persistence = slower decay (continuation lasts longer)
        persistence = max(flow_metrics.flow_persistence, 0.01)
        if persistence > 0.99:
            continuation_decay = -np.log(persistence)
        else:
            continuation_decay = 0.5  # Fast decay (low persistence)
        
        continuation_decay = max(0.01, min(1.0, continuation_decay))
        
        # Step 3: Reversal hazard rate
        # h_reversal(t) = reversal_probability * reversal_speed
        # Use posterior probability as proxy for reversal hazard
        reversal_hazard = posterior.p_down * 0.5  # Moderate base reversal hazard
        
        # Step 4: Settlement hazard rate from posterior probability
        # P(settlement UP) = posterior.p_up * settlement_hazard_rate
        # P(settlement DOWN) = posterior.p_down * settlement_hazard_rate
        settlement_hazard_up = posterior.p_up * base_lambda
        settlement_hazard_down = posterior.p_down * base_lambda
        
        # Step 5: Volatility adjustment
        # Hazard rates increase with volatility
        vol_adjust = sigma * 2.0  # Scale factor
        
        # Apply volatility adjustment to all hazard rates
        up_hazard *= vol_adjust
        down_hazard *= vol_adjust
        continuation_decay *= vol_adjust
        reversal_hazard *= vol_adjust
        settlement_hazard_up *= vol_adjust
        settlement_hazard_down *= vol_adjust
        
        # Step 6: Time-to-reversal estimate
        # E[T_reversal] = 1 / reversal_hazard_rate
        if reversal_hazard > 0:
            time_to_reversal = 1.0 / reversal_hazard
        else:
            time_to_reversal = float('inf')  # No reversal expected
        
        # Smooth with previous estimate
        if self._prev_estimate is not None:
            alpha = 0.7
            up_hazard = alpha * up_hazard + (1 - alpha) * self._prev_estimate.up_hazard_rate
            down_hazard = alpha * down_hazard + (1 - alpha) * self._prev_estimate.down_hazard_rate
            continuation_decay = alpha * continuation_decay + (1 - alpha) * self._prev_estimate.continuation_decay_rate
            reversal_hazard = alpha * reversal_hazard + (1 - alpha) * self._prev_estimate.reversal_hazard_rate
            settlement_hazard_up = alpha * settlement_hazard_up + (1 - alpha) * self._prev_estimate.settlement_hazard_up
            settlement_hazard_down = alpha * settlement_hazard_down + (1 - alpha) * self._prev_estimate.settlement_hazard_down
        
        # Build result
        result = HazardEstimate(
            timestamp=timestamp,
            up_hazard_rate=up_hazard,
            down_hazard_rate=down_hazard,
            continuation_decay_rate=continuation_decay,
            reversal_hazard_rate=reversal_hazard,
            time_to_reversal_estimate=time_to_reversal,
            settlement_hazard_up=settlement_hazard_up,
            settlement_hazard_down=settlement_hazard_down,
            hazard_volatility_adjustment=vol_adjust,
        )
        
        # Track history
        self._hazard_history.append((up_hazard, down_hazard))
        self._continuation_decay_history.append(continuation_decay)
        
        self._prev_estimate = result
        self._initialized = True
        
        return result
