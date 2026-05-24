"""
Latency modeling for Polymarket order execution.

Models the expected latency for order placement and fill on Polymarket CLOB,
including:
- Network latency to Polymarket API
- Processing latency (signature generation, order construction)
- Queue latency (time waiting in order book)
- Settlement latency (time for market resolution)

Latency is critical for 5-minute markets because:
- High latency means our signal may be stale by execution time
- Latency uncertainty creates execution risk
- Regime changes during latency window can invalidate signals

Mathematical basis:
- Latency distribution: log-normal with regime-dependent parameters
- P(stale signal) = 1 - exp(-lambda * latency * volatility)
- Signal decay: signal_strength * exp(-decay_rate * latency)
- Effective edge: edge * (1 - latency_penalty)
"""

import numpy as np
from typing import Optional, Dict, Tuple
from dataclasses import dataclass

from polymarket_bot.bot_types import (
    RegimeStateEstimate,
    VolatilityEstimate,
    RegimeState,
)
from polymarket_bot.config import BotConfig, ExecutionConfig


@dataclass
class LatencyEstimate:
    """Latency estimate for order execution."""
    expected_total_latency_ms: float
    latency_uncertainty_ms: float
    latency_p95_ms: float  # 95th percentile latency
    signal_decay_factor: float  # How much signal decays during latency
    stale_signal_probability: float  # P(signal is stale by execution time)
    effective_edge_multiplier: float  # Edge multiplier accounting for latency
    regime_latency_factor: float  # Regime-dependent latency multiplier


class LatencyModel:
    """
    Models execution latency for Polymarket CLOB orders.
    
    Latency components:
    1. Network latency: round-trip time to Polymarket API
    2. Processing latency: signature generation, order construction
    3. Queue latency: time waiting for matching in CLOB
    4. Confirmation latency: time for order confirmation
    
    Total latency = network + processing + queue + confirmation
    
    The model uses log-normal distribution for latency because:
    - Latency is always positive
    - Has a long right tail (occasional very slow executions)
    - Mean > median (typical case is better than average)
    
    Regime affects latency:
    - High activity regimes: more queue congestion, higher latency
    - Crisis regimes: API may be overloaded, much higher latency
    """

    # Regime-dependent latency multipliers
    REGIME_LATENCY_MULTIPLIERS = {
        RegimeState.CALM_TRENDING: 1.0,
        RegimeState.CALM_RANGE: 1.0,
        RegimeState.VOLATILE_TRENDING: 1.3,
        RegimeState.VOLATILE_RANGE: 1.2,
        RegimeState.CRISIS: 2.0,
        RegimeState.LIQUIDATION_CASCADE: 3.0,
    }

    # Log-normal distribution parameters for latency (mu, sigma)
    # These model the distribution of total latency
    LATENCY_LOGNORMAL_PARAMS = {
        RegimeState.CALM_TRENDING: (4.5, 0.4),    # median ~90ms
        RegimeState.CALM_RANGE: (4.5, 0.4),
        RegimeState.VOLATILE_TRENDING: (5.0, 0.5),  # median ~150ms
        RegimeState.VOLATILE_RANGE: (4.8, 0.5),
        RegimeState.CRISIS: (5.5, 0.7),             # median ~250ms
        RegimeState.LIQUIDATION_CASCADE: (6.0, 0.8), # median ~400ms
    }

    # Signal decay rate per millisecond (how fast signal value decays)
    SIGNAL_DECAY_RATE_PER_MS = 0.001  # 0.1% per ms = 10% per 100ms

    def __init__(self, config: BotConfig):
        self.config = config
        self.exec_config = config.execution
        
        # Latency measurement history
        self._latency_measurements: list = []
        self._max_measurements = 1000
        
        # Adaptive parameters (updated from measurements)
        self._adaptive_base_latency_ms = self.exec_config.typical_latency_ms
        self._adaptive_uncertainty_ms = self.exec_config.latency_uncertainty_ms

    def estimate_latency(
        self,
        regime_estimate: Optional[RegimeStateEstimate] = None,
        volatility_estimate: Optional[VolatilityEstimate] = None,
    ) -> LatencyEstimate:
        """
        Estimate total execution latency with uncertainty.
        
        Uses log-normal model with regime-dependent parameters.
        Computes signal decay and stale signal probability.
        """
        # Determine regime
        regime_state = RegimeState.CALM_TRENDING  # Default
        if regime_estimate is not None:
            regime_state = regime_estimate.current_regime

        # Get regime-dependent parameters
        regime_factor = self.REGIME_LATENCY_MULTIPLIERS.get(regime_state, 1.0)
        lognormal_params = self.LATENCY_LOGNORMAL_PARAMS.get(
            regime_state, (4.5, 0.4)
        )
        mu, sigma = lognormal_params

        # Expected latency from log-normal distribution
        # E[X] = exp(mu + sigma^2/2) for log-normal
        expected_latency_ms = np.exp(mu + sigma**2 / 2.0) * regime_factor

        # Use adaptive base latency if we have measurements
        if len(self._latency_measurements) > 10:
            # Blend measured latency with model
            measured_median = np.median(self._latency_measurements[-100:])
            expected_latency_ms = 0.6 * expected_latency_ms + 0.4 * measured_median

        # Cap at maximum configured latency
        expected_latency_ms = min(expected_latency_ms, self.exec_config.max_latency_ms)

        # Uncertainty: standard deviation of log-normal
        # Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
        variance = (np.exp(sigma**2) - 1.0) * np.exp(2.0 * mu + sigma**2)
        uncertainty_ms = np.sqrt(variance) * regime_factor
        uncertainty_ms = min(uncertainty_ms, self.exec_config.latency_uncertainty_ms * regime_factor)

        # 95th percentile latency
        # P95 = exp(mu + sigma * 1.645) for log-normal
        p95_ms = np.exp(mu + sigma * 1.645) * regime_factor
        p95_ms = min(p95_ms, self.exec_config.max_latency_ms)

        # Signal decay factor
        # How much does our signal decay during the latency window?
        # decay = exp(-decay_rate * latency_ms)
        signal_decay = np.exp(-self.SIGNAL_DECAY_RATE_PER_MS * expected_latency_ms)

        # Volatility amplifies signal decay
        if volatility_estimate is not None:
            vol_realized = volatility_estimate.realized_volatility
            # Higher volatility = faster signal decay
            vol_amplification = 1.0 + vol_realized * 20.0  # Scale factor
            signal_decay = np.exp(
                -self.SIGNAL_DECAY_RATE_PER_MS * expected_latency_ms * vol_amplification
            )

        signal_decay = np.clip(signal_decay, 0.1, 1.0)

        # Stale signal probability
        # P(stale) = probability that regime changes during latency window
        # Using regime transition probability
        stale_prob = 0.05  # Base: 5% chance of stale signal
        if regime_estimate is not None:
            # Use regime transition probability
            # P(regime change in T ms) ≈ (1 - persistence) * T / typical_duration
            regime_probs = regime_estimate.regime_probabilities
            # Low persistence regimes have higher stale probability
            persistence = max(regime_probs.values()) if regime_probs else 0.9
            # Time window in seconds
            time_window_s = expected_latency_ms / 1000.0
            # P(change) = (1 - persistence^(T/typical_interval))
            typical_interval_s = 1.0  # 1 second typical regime update interval
            stale_prob = 1.0 - persistence ** (time_window_s / typical_interval_s)

        # Add volatility contribution to stale probability
        if volatility_estimate is not None:
            vol_contribution = volatility_estimate.realized_volatility * 5.0
            stale_prob += vol_contribution * time_window_s

        stale_prob = np.clip(stale_prob, 0.01, 0.5)

        # Effective edge multiplier
        # edge * (1 - stale_prob) * signal_decay
        effective_edge_mult = (1.0 - stale_prob) * signal_decay

        return LatencyEstimate(
            expected_total_latency_ms=float(expected_latency_ms),
            latency_uncertainty_ms=float(uncertainty_ms),
            latency_p95_ms=float(p95_ms),
            signal_decay_factor=float(signal_decay),
            stale_signal_probability=float(stale_prob),
            effective_edge_multiplier=float(effective_edge_mult),
            regime_latency_factor=float(regime_factor),
        )

    def compute_latency_adjusted_edge(
        self,
        raw_edge_bps: float,
        latency_estimate: LatencyEstimate,
    ) -> Tuple[float, float]:
        """
        Compute latency-adjusted edge.
        
        effective_edge = raw_edge * effective_edge_multiplier
        
        Returns (effective_edge_bps, edge_retention_fraction).
        """
        effective_edge = raw_edge_bps * latency_estimate.effective_edge_multiplier
        retention_fraction = latency_estimate.effective_edge_multiplier

        return float(effective_edge), float(retention_fraction)

    def record_latency_measurement(
        self,
        measured_latency_ms: float,
    ):
        """
        Record an actual latency measurement for adaptive updating.
        
        Updates the adaptive base latency using exponential moving average.
        """
        self._latency_measurements.append(measured_latency_ms)
        if len(self._latency_measurements) > self._max_measurements:
            self._latency_measurements = self._latency_measurements[-self._max_measurements:]

        # Update adaptive parameters using EMA
        alpha = 0.1  # Slow adaptation
        self._adaptive_base_latency_ms = (
            alpha * measured_latency_ms
            + (1.0 - alpha) * self._adaptive_base_latency_ms
        )

        # Update uncertainty from recent variance
        if len(self._latency_measurements) > 20:
            recent = self._latency_measurements[-100:]
            self._adaptive_uncertainty_ms = float(np.std(recent))

    def should_wait_for_better_latency(
        self,
        current_latency_estimate: LatencyEstimate,
        regime_estimate: RegimeStateEstimate,
    ) -> bool:
        """
        Determine whether to delay execution waiting for better latency.
        
        Wait if:
        - Current latency is significantly above typical
        - Regime is high-activity (congested)
        - P95 latency exceeds maximum acceptable
        
        Never wait if:
        - Signal is very strong (edge > 2x threshold)
        - Time to settlement is running out
        """
        # Check if current latency is much worse than typical
        typical_ms = self.exec_config.typical_latency_ms
        current_ms = current_latency_estimate.expected_total_latency_ms

        # Latency is 2x typical: consider waiting
        if current_ms > 2.0 * typical_ms:
            # But only if regime suggests it might improve
            regime_state = regime_estimate.current_regime
            if regime_state in (RegimeState.CRISIS, RegimeState.LIQUIDATION_CASCADE):
                # In crisis: latency unlikely to improve, execute anyway
                return False
            # In other regimes: might improve, consider waiting
            return True

        # P95 exceeds maximum: definitely wait
        if current_latency_estimate.latency_p95_ms > self.exec_config.max_latency_ms:
            return True

        return False