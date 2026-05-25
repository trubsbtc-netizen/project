"""
Signal decorrelation for preventing redundant trade decisions.

Ensures that consecutive trade decisions are not based on the same
or highly correlated signals. This prevents:
- Overtrading on persistent signals that have already been acted on
- Double-entry on correlated microstructure features
- Cascade trades during regime transitions

The decorrelation module tracks:
1. Recent signal fingerprints (what signals drove recent decisions)
2. Correlation between current and recent signals
3. Minimum time between trades (cooldown)
4. Direction consistency (don't flip direction too rapidly)

Mathematical basis:
- Signal fingerprint: vector of normalized signal components
- Correlation: Pearson r between current and recent fingerprints
- Decorrelation threshold: if r > threshold, suppress trade
- Direction flip penalty: exponential decay on flip frequency
- Time cooldown: minimum interval between trades
"""

import numpy as np
import time
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from polymarket_bot.bot_types import (
    SettlementForecast,
    ExecutionEstimate,
    ContinuationSignal,
    ReversalSignal,
    EntropyFilterResult,
    MultiTimescaleEstimate,
    BayesianPosterior,
    KalmanState,
    OrderbookPressure,
    FlowMetrics,
    Direction,
    SignalType,
)
from polymarket_bot.config import BotConfig


@dataclass
class SignalFingerprint:
    """Compact representation of signal state for decorrelation tracking."""
    timestamp: float
    direction: Direction
    p_up: float
    velocity: float
    flow_imbalance: float
    orderbook_pressure: float
    continuation_prob: float
    reversal_prob: float
    regime_state: int
    entropy_suppression: float
    edge_bps: float


class DecorrelationFilter:
    """
    Filters trade decisions to prevent correlated/redundant entries.
    
    The filter maintains a history of recent signal fingerprints and
    checks whether the current signal is too correlated with recent
    ones. If so, the trade is suppressed to prevent overtrading.
    
    Decorrelation checks:
    1. Signal correlation: if current fingerprint is highly correlated
       with a recent fingerprint, the signal is redundant
    2. Direction flip: rapid direction changes indicate noise, not signal
    3. Time cooldown: minimum interval between trades
    4. Edge consistency: if edge is similar to recent, it's the same opportunity
    """

    # Defaults are overridden by SignalConfig when available.
    MIN_TRADE_INTERVAL = 10.0
    LATE_WINDOW_INTERVAL = 4.0
    LATE_WINDOW_START_SECONDS = 90.0
    CORRELATION_THRESHOLD = 0.7
    MAX_FLIPS_IN_WINDOW = 3
    FLIP_WINDOW_SECONDS = 300.0
    EDGE_SIMILARITY_THRESHOLD = 50.0

    def __init__(self, config: BotConfig):
        self.config = config
        signal_config = getattr(config, "signal", config)
        self.min_trade_interval = float(
            getattr(
                signal_config,
                "decorrelation_min_trade_interval_seconds",
                self.MIN_TRADE_INTERVAL,
            )
        )
        self.late_window_interval = float(
            getattr(
                signal_config,
                "decorrelation_late_window_interval_seconds",
                self.LATE_WINDOW_INTERVAL,
            )
        )
        self.late_window_start_seconds = float(
            getattr(
                signal_config,
                "decorrelation_late_window_start_seconds",
                self.LATE_WINDOW_START_SECONDS,
            )
        )
        self.correlation_threshold = float(
            getattr(
                signal_config,
                "decorrelation_correlation_threshold",
                self.CORRELATION_THRESHOLD,
            )
        )
        self.edge_similarity_threshold = float(
            getattr(
                signal_config,
                "decorrelation_edge_similarity_bps",
                self.EDGE_SIMILARITY_THRESHOLD,
            )
        )
        self.max_flips_in_window = int(
            getattr(signal_config, "decorrelation_max_flips_in_window", self.MAX_FLIPS_IN_WINDOW)
        )
        self.flip_window_seconds = float(
            getattr(signal_config, "decorrelation_flip_window_seconds", self.FLIP_WINDOW_SECONDS)
        )
        self._fingerprint_history: List[SignalFingerprint] = []
        self._max_fingerprints = 50
        self._last_trade_timestamp = 0.0
        self._last_trade_direction: Optional[Direction] = None
        self._direction_flip_times: List[float] = []
        self._max_flip_history = 20

    def _compute_fingerprint(
        self,
        settlement_forecast: SettlementForecast,
        kalman_state: Optional[KalmanState] = None,
        orderbook_pressure: Optional[OrderbookPressure] = None,
        flow_metrics: Optional[FlowMetrics] = None,
        continuation_signal: Optional[ContinuationSignal] = None,
        reversal_signal: Optional[ReversalSignal] = None,
        entropy_filter: Optional[EntropyFilterResult] = None,
        regime_state: int = 0,
    ) -> SignalFingerprint:
        """
        Compute a signal fingerprint from current market state.
        
        The fingerprint captures the key signal components in a compact
        vector that can be compared with recent fingerprints for
        decorrelation checking.
        """
        p_up = settlement_forecast.p_up_settlement
        direction = settlement_forecast.expected_settlement_direction
        edge_bps = settlement_forecast.edge_estimate * 10000.0

        velocity = 0.0
        if kalman_state is not None:
            velocity = kalman_state.velocity

        flow_imbalance = 0.0
        if flow_metrics is not None:
            flow_imbalance = flow_metrics.flow_imbalance

        pressure = 0.0
        if orderbook_pressure is not None:
            pressure = orderbook_pressure.net_pressure

        continuation_prob = 0.0
        if continuation_signal is not None:
            continuation_prob = continuation_signal.continuation_probability

        reversal_prob = 0.0
        if reversal_signal is not None:
            reversal_prob = reversal_signal.reversal_probability

        entropy_suppression = 1.0
        if entropy_filter is not None:
            entropy_suppression = entropy_filter.suppression_factor

        return SignalFingerprint(
            timestamp=time.time(),
            direction=direction,
            p_up=p_up,
            velocity=velocity,
            flow_imbalance=flow_imbalance,
            orderbook_pressure=pressure,
            continuation_prob=continuation_prob,
            reversal_prob=reversal_prob,
            regime_state=regime_state,
            entropy_suppression=entropy_suppression,
            edge_bps=edge_bps,
        )

    def _fingerprint_to_vector(
        self,
        fingerprint: SignalFingerprint,
    ) -> np.ndarray:
        """
        Convert fingerprint to normalized vector for correlation computation.
        
        Normalization ensures each component contributes equally
        to the correlation measure, regardless of scale.
        """
        # Direction as numeric: UP=1, DOWN=-1, NEUTRAL=0
        direction_numeric = 1.0 if fingerprint.direction == Direction.UP else (
            -1.0 if fingerprint.direction == Direction.DOWN else 0.0
        )

        # Raw components
        raw = np.array([
            fingerprint.p_up,
            fingerprint.velocity * 100.0,  # Scale velocity to similar range
            fingerprint.flow_imbalance,
            fingerprint.orderbook_pressure,
            fingerprint.continuation_prob,
            fingerprint.reversal_prob,
            direction_numeric,
            fingerprint.edge_bps / 500.0,  # Scale edge to similar range
        ])

        # Normalize each component to [-1, 1] range
        # Using simple clipping since components are already bounded
        normalized = np.clip(raw, -1.0, 1.0)

        return normalized

    def _compute_correlation_with_recent(
        self,
        current_vector: np.ndarray,
    ) -> float:
        """
        Compute maximum correlation between current and recent fingerprints.
        
        Uses Pearson correlation coefficient:
        r = (sum(x*y)) / (sqrt(sum(x^2)) * sqrt(sum(y^2)))
        
        Returns the maximum correlation with any recent fingerprint.
        High correlation means the current signal is redundant.
        """
        if not self._fingerprint_history:
            return 0.0  # No history: no correlation

        max_correlation = 0.0

        # Check against recent fingerprints (last 10)
        recent_fingerprints = self._fingerprint_history[-10:]

        for fp in recent_fingerprints:
            recent_vector = self._fingerprint_to_vector(fp)

            # Pearson correlation
            dot_product = np.dot(current_vector, recent_vector)
            norm_current = np.linalg.norm(current_vector)
            norm_recent = np.linalg.norm(recent_vector)

            if norm_current > 1e-10 and norm_recent > 1e-10:
                correlation = dot_product / (norm_current * norm_recent)
                max_correlation = max(max_correlation, abs(correlation))

        return float(max_correlation)

    def _required_trade_interval(
        self,
        settlement_forecast: SettlementForecast,
    ) -> float:
        time_to_settlement = float(settlement_forecast.time_to_settlement_seconds)
        if time_to_settlement <= self.late_window_start_seconds:
            return max(0.0, self.late_window_interval)
        return max(0.0, self.min_trade_interval)

    def _check_time_cooldown(
        self,
        settlement_forecast: SettlementForecast,
    ) -> Tuple[bool, float]:
        """
        Check if enough time has passed since last trade.
        
        Returns (is_in_cooldown, remaining_seconds).
        """
        current_time = time.time()
        elapsed = current_time - self._last_trade_timestamp
        required_interval = self._required_trade_interval(settlement_forecast)
        remaining = max(0.0, required_interval - elapsed)

        return remaining > 0, remaining

    def _check_direction_flip_rate(
        self,
        proposed_direction: Direction,
    ) -> Tuple[bool, int]:
        """
        Check if direction flip rate is too high.
        
        Returns (is_flip, recent_flip_count).
        """
        current_time = time.time()

        # Clean old flip records
        self._direction_flip_times = [
            t for t in self._direction_flip_times
            if current_time - t < self.flip_window_seconds
        ]

        # Check if this is a flip
        is_flip = False
        if self._last_trade_direction is not None:
            if proposed_direction != self._last_trade_direction:
                is_flip = True

        flip_count = len(self._direction_flip_times)

        return is_flip, flip_count

    def _check_edge_similarity(
        self,
        current_edge_bps: float,
        proposed_direction: Direction,
    ) -> bool:
        """
        Check if current edge is too similar to recent edges.
        
        Similar edge means we're trading on the same opportunity.
        """
        if not self._fingerprint_history:
            return False  # No history: not similar

        # Check against last 5 fingerprints
        recent = self._fingerprint_history[-5:]
        for fp in recent:
            if fp.direction != proposed_direction:
                continue
            edge_diff = abs(current_edge_bps - fp.edge_bps)
            if edge_diff < self.edge_similarity_threshold:
                return True  # Too similar

        return False

    def check_decorrelation(
        self,
        settlement_forecast: SettlementForecast,
        kalman_state: Optional[KalmanState] = None,
        orderbook_pressure: Optional[OrderbookPressure] = None,
        flow_metrics: Optional[FlowMetrics] = None,
        continuation_signal: Optional[ContinuationSignal] = None,
        reversal_signal: Optional[ReversalSignal] = None,
        entropy_filter: Optional[EntropyFilterResult] = None,
        regime_state: int = 0,
    ) -> Tuple[bool, str, float]:
        """
        Check if the current signal is decorrelated enough to trade.
        
        Returns (should_trade, reason, decorrelation_score).
        
        decorrelation_score: 0 = fully correlated (suppress), 1 = fully decorrelated (trade)
        
        Checks:
        1. Time cooldown
        2. Signal correlation with recent
        3. Direction flip rate
        4. Edge similarity
        """
        # Compute current fingerprint
        fingerprint = self._compute_fingerprint(
            settlement_forecast=settlement_forecast,
            kalman_state=kalman_state,
            orderbook_pressure=orderbook_pressure,
            flow_metrics=flow_metrics,
            continuation_signal=continuation_signal,
            reversal_signal=reversal_signal,
            entropy_filter=entropy_filter,
            regime_state=regime_state,
        )

        current_vector = self._fingerprint_to_vector(fingerprint)

        # Check 1: Direction flip rate. A valid flip can bypass duplicate-entry
        # cooldown/correlation checks, but repeated flips are still suppressed.
        proposed_direction = fingerprint.direction
        is_flip, flip_count = self._check_direction_flip_rate(proposed_direction)
        if is_flip and flip_count >= self.max_flips_in_window:
            return False, f"too_many_flips_{flip_count}", 0.1

        # Check 2: Time cooldown. Directional flips represent new probability
        # information, so do not treat them as duplicate same-side entries.
        in_cooldown, remaining = self._check_time_cooldown(settlement_forecast)
        if in_cooldown and not is_flip:
            return False, f"cooldown_{remaining:.0f}s_remaining", 0.0

        # Check 3: Signal correlation
        max_correlation = self._compute_correlation_with_recent(current_vector)
        if max_correlation > self.correlation_threshold and not is_flip:
            decorrelation_score = 1.0 - max_correlation
            return False, f"high_correlation_{max_correlation:.2f}", decorrelation_score

        # Check 4: Edge similarity
        edge_similar = self._check_edge_similarity(
            fingerprint.edge_bps,
            proposed_direction,
        )
        if edge_similar:
            return False, "edge_similar_to_recent", 0.2

        # All checks passed
        decorrelation_score = 1.0 - max_correlation * 0.5  # Partial penalty for some correlation

        # Flip penalty: reduce score for flips even if below threshold
        if is_flip:
            flip_penalty = 0.1 * min(flip_count, 3)
            decorrelation_score -= flip_penalty

        return True, "decorrelated", float(np.clip(decorrelation_score, 0.1, 1.0))

    def record_trade(
        self,
        direction: Direction,
        settlement_forecast: SettlementForecast,
        kalman_state: Optional[KalmanState] = None,
        orderbook_pressure: Optional[OrderbookPressure] = None,
        flow_metrics: Optional[FlowMetrics] = None,
        continuation_signal: Optional[ContinuationSignal] = None,
        reversal_signal: Optional[ReversalSignal] = None,
        entropy_filter: Optional[EntropyFilterResult] = None,
        regime_state: int = 0,
    ):
        """
        Record a trade decision for future decorrelation checks.
        
        Updates fingerprint history, last trade timestamp, and
        direction flip tracking.
        """
        fingerprint = self._compute_fingerprint(
            settlement_forecast=settlement_forecast,
            kalman_state=kalman_state,
            orderbook_pressure=orderbook_pressure,
            flow_metrics=flow_metrics,
            continuation_signal=continuation_signal,
            reversal_signal=reversal_signal,
            entropy_filter=entropy_filter,
            regime_state=regime_state,
        )

        # Add to fingerprint history
        self._fingerprint_history.append(fingerprint)
        if len(self._fingerprint_history) > self._max_fingerprints:
            self._fingerprint_history = self._fingerprint_history[-self._max_fingerprints:]

        # Update last trade timestamp
        self._last_trade_timestamp = time.time()

        # Track direction flips
        if self._last_trade_direction is not None:
            if direction != self._last_trade_direction:
                self._direction_flip_times.append(time.time())
                if len(self._direction_flip_times) > self._max_flip_history:
                    self._direction_flip_times = self._direction_flip_times[-self._max_flip_history:]

        self._last_trade_direction = direction

    def reset_for_new_round(self) -> None:
        """Clear per-round decorrelation state after market rollover."""
        self._fingerprint_history.clear()
        self._last_trade_timestamp = 0.0
        self._last_trade_direction = None
        self._direction_flip_times.clear()

    def get_decorrelation_stats(self) -> dict:
        """Return decorrelation filter statistics."""
        current_time = time.time()
        recent_flips = [
            t for t in self._direction_flip_times
            if current_time - t < self.flip_window_seconds
        ]

        return {
            "fingerprint_count": len(self._fingerprint_history),
            "last_trade_age_seconds": current_time - self._last_trade_timestamp,
            "recent_flip_count": len(recent_flips),
            "last_direction": self._last_trade_direction.value if self._last_trade_direction else None,
            "min_trade_interval_seconds": self.min_trade_interval,
            "late_window_interval_seconds": self.late_window_interval,
        }
