"""
Failed breakout burst detection engine.

Mathematical foundation:
- Burst: rapid directional price movement with high velocity
  burst_velocity = |price_change| / time_window, measured in volatility units
  burst_threshold: velocity > N * sigma (typically N = 2.5)

- Burst failure: burst that reverses back through its origin level
  failure = price reverses beyond burst start point within time window
  
- Failure detection metrics:
  1. Burst velocity: initial speed of the burst (sigma-normalized)
  2. Reversal depth: how far price reversed from burst peak/trough
     reversal_depth = |peak - current| / |peak - origin|
  3. Failure speed: how fast the reversal occurred relative to burst
     failure_speed = reversal_time / burst_time
  4. Volume profile: volume during burst vs volume during reversal
     volume_ratio = reversal_volume / burst_volume
  5. Liquidity absorption: volume absorbed at burst peak/trough
     absorption = absorbed_volume / burst_volume

- Burst failure probability model:
  P(failure) = sigmoid(w1*reversal_depth + w2*failure_speed + w3*volume_ratio + w4*absorption)

Failed breakout bursts are one of the FASTEST reversal signals because:
- The burst itself creates overextension (price too far from equilibrium)
- The failure confirms the overextension was rejected by the market
- The reversal from failure is typically swift and decisive
"""

import logging
import time
import numpy as np
from typing import Optional, Tuple
from collections import deque

from polymarket_bot.config import SignalConfig, Direction
from polymarket_bot.bot_types import (
    BurstFailureSignal, KalmanState, FlowMetrics,
    VolatilityEstimate, LiquidityMetrics
)

logger = logging.getLogger(__name__)


class BurstDetector:
    """
    Detects failed breakout bursts - the fastest reversal signal.
    
    Key innovations:
    1. Burst velocity detection (sigma-normalized speed)
    2. Burst origin tracking (price level where burst started)
    3. Reversal depth measurement (how far burst has reversed)
    4. Failure speed estimation (reversal speed relative to burst)
    5. Volume profile analysis (burst vs reversal volume)
    6. Liquidity absorption at burst extremes
    7. Time window-based burst/failure tracking
    """
    
    def __init__(self, config: SignalConfig):
        self.config = config
        
        # Burst tracking
        self._burst_active = False
        self._burst_direction = Direction.NEUTRAL
        self._burst_origin_price = 0.0
        self._burst_peak_price = 0.0
        self._burst_start_time = 0.0
        self._burst_peak_time = 0.0
        self._burst_velocity = 0.0
        self._burst_volume = 0.0
        
        # Price tracking for burst detection
        self._price_history: deque = deque(maxlen=300)  # ~5 minutes at 1s intervals
        self._velocity_history: deque = deque(maxlen=100)
        
        # Volume tracking
        self._volume_during_burst = 0.0
        self._volume_during_reversal = 0.0
        
        # Previous signal
        self._prev_signal: Optional[BurstFailureSignal] = None
    
    def update(
        self,
        current_price: float,
        kalman_state: KalmanState,
        flow_metrics: FlowMetrics,
        volatility_estimate: VolatilityEstimate,
        liquidity_metrics: Optional[LiquidityMetrics] = None,
        timestamp: float = 0.0,
    ) -> BurstFailureSignal:
        """
        Update burst detection with new price and microstructure data.
        
        Process:
        1. Check for new burst (velocity exceeds threshold)
        2. If burst active, track peak and check for reversal
        3. If reversal exceeds depth threshold, detect failure
        4. Compute failure metrics (depth, speed, volume, absorption)
        5. Compute failure probability
        6. Reset burst tracking if failure confirmed or burst expires
        """
        if timestamp == 0.0:
            timestamp = time.time()
        
        # Store price
        self._price_history.append(current_price)
        price_scale = max(abs(float(kalman_state.state[0])) if len(kalman_state.state) else current_price, 1.0e-9)
        log_velocity = kalman_state.velocity_estimate / price_scale
        self._velocity_history.append(log_velocity)
        
        # Step 1: Check for new burst
        velocity = log_velocity
        vol = volatility_estimate.regime_adjusted_volatility
        
        # Burst velocity in sigma units
        burst_velocity_sigma = abs(velocity) / max(vol, 1e-10)
        
        # Detect new burst if velocity exceeds threshold
        if not self._burst_active and burst_velocity_sigma > self.config.burst_velocity_threshold:
            self._start_burst(current_price, velocity, timestamp, flow_metrics.total_volume)
        
        # Step 2: If burst active, track and check for failure
        if self._burst_active:
            result = self._track_burst(
                current_price, velocity, timestamp,
                flow_metrics, volatility_estimate, liquidity_metrics
            )
            return result
        
        # No burst active - return empty signal
        return BurstFailureSignal(
            timestamp=timestamp,
            direction=Direction.NEUTRAL,
            burst_failure_probability=0.0,
            burst_velocity=0.0,
            reversal_depth=0.0,
            failure_speed=0.0,
            volume_profile=0.0,
            liquidity_absorption=0.0,
            is_valid=False,
        )
    
    def _start_burst(self, price: float, velocity: float,
                      timestamp: float, volume: float):
        """Start tracking a new burst."""
        self._burst_active = True
        self._burst_direction = Direction.UP if velocity > 0 else Direction.DOWN
        self._burst_origin_price = price
        self._burst_peak_price = price
        self._burst_start_time = timestamp
        self._burst_peak_time = timestamp
        self._burst_velocity = abs(velocity)
        self._burst_volume = volume
        self._volume_during_burst = volume
        self._volume_during_reversal = 0.0
        
        logger.debug(f"Burst detected: direction={self._burst_direction}, "
                     f"velocity={self._burst_velocity}")
    
    def _track_burst(
        self,
        current_price: float,
        velocity: float,
        timestamp: float,
        flow_metrics: FlowMetrics,
        volatility_estimate: VolatilityEstimate,
        liquidity_metrics: Optional[LiquidityMetrics],
    ) -> BurstFailureSignal:
        """
        Track active burst and check for failure.
        
        Failure criteria:
        1. Price reverses beyond burst origin (complete failure)
        2. Price reverses beyond depth threshold (partial failure)
        3. Burst expires without continuation (time failure)
        """
        # Update peak price
        if self._burst_direction == Direction.UP:
            if current_price > self._burst_peak_price:
                self._burst_peak_price = current_price
                self._burst_peak_time = timestamp
                self._volume_during_burst += flow_metrics.total_volume
            else:
                self._volume_during_reversal += flow_metrics.total_volume
        elif self._burst_direction == Direction.DOWN:
            if current_price < self._burst_peak_price:
                self._burst_peak_price = current_price
                self._burst_peak_time = timestamp
                self._volume_during_burst += flow_metrics.total_volume
            else:
                self._volume_during_reversal += flow_metrics.total_volume
        
        # Compute reversal depth
        burst_range = abs(self._burst_peak_price - self._burst_origin_price)
        if burst_range > 0:
            if self._burst_direction == Direction.UP:
                reversal_from_peak = self._burst_peak_price - current_price
            else:
                reversal_from_peak = current_price - self._burst_peak_price
            
            reversal_depth = reversal_from_peak / burst_range
        else:
            reversal_depth = 0.0
        
        reversal_depth = max(0.0, reversal_depth)
        
        # Compute failure speed
        burst_duration = timestamp - self._burst_start_time
        reversal_duration = timestamp - self._burst_peak_time
        
        if reversal_duration > 0 and burst_duration > 0:
            failure_speed = reversal_duration / burst_duration
        else:
            failure_speed = 0.0
        
        # Compute volume profile
        if self._volume_during_burst > 0:
            volume_profile = self._volume_during_reversal / self._volume_during_burst
        else:
            volume_profile = 0.0
        
        # Compute liquidity absorption
        absorption = 0.0
        if liquidity_metrics is not None:
            absorption = liquidity_metrics.absorption_score
        
        # Burst velocity (sigma-normalized)
        vol = volatility_estimate.regime_adjusted_volatility
        burst_velocity_sigma = self._burst_velocity / max(vol, 1e-10)
        
        # Compute failure probability
        # P(failure) = sigmoid(w1*reversal_depth + w2*failure_speed + w3*volume_profile + w4*absorption)
        weights = np.array([0.35, 0.20, 0.20, 0.25])
        components = np.array([
            reversal_depth / self.config.burst_failure_reversal_depth,
            min(1.0, failure_speed),
            min(1.0, volume_profile),
            absorption,
        ])
        
        raw_score = np.sum(weights * components)
        failure_probability = 1.0 / (1.0 + np.exp(-raw_score * 3.0))
        
        # Check for complete failure (price reverses beyond origin)
        complete_failure = False
        if self._burst_direction == Direction.UP and current_price < self._burst_origin_price:
            complete_failure = True
            failure_probability = 1.0
        elif self._burst_direction == Direction.DOWN and current_price > self._burst_origin_price:
            complete_failure = True
            failure_probability = 1.0
        
        # Check for burst expiry (time window exceeded)
        burst_time = timestamp - self._burst_start_time
        if burst_time > self.config.burst_time_window_seconds and reversal_depth < 0.2:
            # Burst expired without significant reversal - not a failure
            self._burst_active = False
            return BurstFailureSignal(
                timestamp=timestamp,
                direction=self._burst_direction,
                burst_failure_probability=0.0,
                burst_velocity=burst_velocity_sigma,
                reversal_depth=reversal_depth,
                failure_speed=failure_speed,
                volume_profile=volume_profile,
                liquidity_absorption=absorption,
                is_valid=False,
            )
        
        # Determine if failure is significant enough to report
        is_valid = (
            reversal_depth > 0.2 or  # At least 20% reversal
            complete_failure or
            failure_probability > 0.5
        )
        
        # Reset burst if failure confirmed
        if complete_failure or (failure_probability > 0.8 and reversal_depth > 0.5):
            self._burst_active = False
        
        result = BurstFailureSignal(
            timestamp=timestamp,
            direction=self._burst_direction,
            burst_failure_probability=failure_probability,
            burst_velocity=burst_velocity_sigma,
            reversal_depth=reversal_depth,
            failure_speed=failure_speed,
            volume_profile=volume_profile,
            liquidity_absorption=absorption,
            is_valid=is_valid,
        )
        
        self._prev_signal = result
        return result
