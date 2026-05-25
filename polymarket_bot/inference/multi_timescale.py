"""
Multi-timescale inference aggregation.

Combines directional probability estimates across multiple time horizons
(1s, 5s, 30s, 1m, 5m) using regime-adaptive weighting. Short timescales
capture microstructure noise and fast reversals; long timescales capture
trend persistence. The aggregation uses exponential decay weighting with
regime-dependent concentration, and computes consistency/conflict metrics
to detect when timescales disagree (a key uncertainty signal).

Mathematical basis:
- Exponential decay weighting: w(t) = exp(-lambda * |t - t_dominant|)
- Regime-dependent concentration parameter kappa
- Consistency = 1 - variance(normalized_probs)
- Conflict score from pairwise sign disagreements
- Entropy-weighted aggregation for uncertainty-aware fusion
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from polymarket_bot.bot_types import (
    MultiTimescaleEstimate,
    KalmanState,
    BayesianPosterior,
    HazardEstimate,
    RegimeStateEstimate,
    VolatilityEstimate,
    ContinuationSignal,
    ReversalSignal,
    EntropyFilterResult,
    Direction,
    RegimeState,
)
from polymarket_bot.config import BotConfig


class MultiTimescaleAggregator:
    """
    Aggregates directional probability across multiple time horizons.
    
    Each timescale provides a different view:
    - 1s: Microstructure noise, immediate order flow
    - 5s: Fast reversal precursors, taker inversion
    - 30s: Short-term momentum, flow persistence
    - 1m: Medium-term trend, regime alignment
    - 5m: Long-term settlement probability
    
    The aggregation weights are regime-adaptive:
    - Trend regimes: heavier on long timescales
    - Reversal regimes: heavier on short timescales
    - Choppy regimes: equal weighting with high conflict tolerance
    """

    TIMESCALES = ["1s", "5s", "30s", "1m", "5m"]
    TIMESCALE_SECONDS = {"1s": 1.0, "5s": 5.0, "30s": 30.0, "1m": 60.0, "5m": 300.0}

    # Regime-adaptive concentration parameters (higher = more concentrated on dominant)
    REGIME_CONCENTRATION = {
        RegimeState.CALM_TRENDING: 2.5,       # Strong trend: trust dominant timescale
        RegimeState.CALM_RANGE: 0.8,           # Choppy: spread weight evenly
        RegimeState.VOLATILE_TRENDING: 1.5,    # Volatile trend: moderate concentration
        RegimeState.VOLATILE_RANGE: 0.7,       # Volatile range: spread weight
        RegimeState.CRISIS: 0.6,               # Crisis: very spread, high uncertainty
        RegimeState.LIQUIDATION_CASCADE: 0.5,  # Cascade: maximum spread
    }

    # Regime-adaptive dominant timescale bias
    REGIME_DOMINANT_BIAS = {
        RegimeState.CALM_TRENDING: "5m",       # Calm trends: trust long-term
        RegimeState.CALM_RANGE: "30s",          # Calm range: trust medium
        RegimeState.VOLATILE_TRENDING: "1m",    # Volatile trend: trust medium-long
        RegimeState.VOLATILE_RANGE: "30s",      # Volatile range: trust medium
        RegimeState.CRISIS: "5s",               # Crisis: trust short-term
        RegimeState.LIQUIDATION_CASCADE: "1s",  # Cascade: trust immediate
    }

    # Base weights for each timescale (before regime adjustment)
    BASE_WEIGHTS = {"1s": 0.05, "5s": 0.10, "30s": 0.25, "1m": 0.30, "5m": 0.30}

    def __init__(self, config: BotConfig):
        self.config = config
        # History for exponential smoothing
        self._prev_estimates: Dict[str, float] = {}
        self._smoothing_alpha = 0.3  # EMA smoothing for individual timescales
        self._estimate_count = 0

    def estimate_timescale_probabilities(
        self,
        kalman_state: KalmanState,
        posterior: BayesianPosterior,
        hazard_estimate: HazardEstimate,
        regime_estimate: RegimeStateEstimate,
        volatility_estimate: VolatilityEstimate,
        continuation_signal: Optional[ContinuationSignal] = None,
        reversal_signal: Optional[ReversalSignal] = None,
        entropy_filter: Optional[EntropyFilterResult] = None,
        time_to_settlement: float = 300.0,
    ) -> Dict[str, float]:
        """
        Compute directional probability at each timescale.
        
        Each timescale uses different signal combinations:
        - 1s: Kalman velocity + microstructure pressure (noisy, fast)
        - 5s: Kalman velocity + reversal precursors + flow
        - 30s: Bayesian posterior + continuation + flow persistence
        - 1m: Bayesian posterior + regime alignment + continuation
        - 5m: Settlement hazard + posterior + regime
        
        Returns dict of timescale -> P(UP).
        """
        estimates = {}

        # ---- 1-second timescale: raw microstructure + Kalman velocity ----
        # This is the fastest, most noisy signal
        velocity = kalman_state.velocity
        velocity_confidence = kalman_state.velocity_confidence
        price_scale = float(kalman_state.state[0]) if len(kalman_state.state) else 0.0
        price_scale = max(abs(price_scale), 1.0e-9)
        log_velocity = velocity / price_scale

        # Sigmoid mapping of log-velocity to probability. Kalman velocity is in
        # price units/second, while volatility is in log-return units.
        vol_realized = volatility_estimate.realized_volatility
        velocity_z = float(np.clip(log_velocity / max(vol_realized, 1.0e-10), -8.0, 8.0))
        p_velocity = 1.0 / (1.0 + np.exp(-velocity_z))

        # Weight by velocity confidence
        p_1s = 0.5 + (p_velocity - 0.5) * velocity_confidence
        estimates["1s"] = np.clip(p_1s, 0.01, 0.99)

        # ---- 5-second timescale: fast reversal + flow ----
        p_base_5s = p_1s  # Start from 1s estimate

        # Reversal signal modifies probability
        if reversal_signal is not None:
            reversal_strength = reversal_signal.reversal_probability
            reversal_direction = reversal_signal.reversal_direction

            # If reversal is likely UP, shift probability toward UP
            reversal_adjustment = reversal_strength * 0.3
            if reversal_direction == Direction.UP:
                p_base_5s += reversal_adjustment
            else:
                p_base_5s -= reversal_adjustment

        # Flow acceleration provides short-term directional bias
        if kalman_state.acceleration_confidence > 0.3:
            accel = kalman_state.acceleration
            log_accel = accel / price_scale
            accel_z = float(np.clip(log_accel / max(vol_realized, 1.0e-10), -8.0, 8.0))
            p_accel = 1.0 / (1.0 + np.exp(-accel_z))
            # Blend acceleration signal (less weight than velocity)
            p_base_5s = 0.7 * p_base_5s + 0.3 * p_accel

        estimates["5s"] = np.clip(p_base_5s, 0.01, 0.99)

        # ---- 30-second timescale: Bayesian posterior + continuation ----
        p_30s = posterior.p_up  # Start from Bayesian posterior

        # Continuation signal adjusts
        if continuation_signal is not None:
            cont_prob = continuation_signal.continuation_probability
            cont_direction = continuation_signal.continuation_direction

            # Continuation supports current direction
            if cont_direction == Direction.UP:
                # If continuation is strong UP, increase P(UP)
                p_30s += cont_prob * 0.15
            else:
                p_30s -= cont_prob * 0.15

        # Flow persistence provides medium-term bias
        # High persistence means current direction is likely to continue
        if hasattr(posterior, 'evidence_strength') and posterior.evidence_strength > 0:
            # Scale adjustment by evidence strength
            persistence_factor = min(1.0, posterior.evidence_strength / 5.0)
            # If posterior already says UP, persistence reinforces it
            direction_bias = (p_30s - 0.5) * persistence_factor * 0.1
            p_30s += direction_bias

        estimates["30s"] = np.clip(p_30s, 0.01, 0.99)

        # ---- 1-minute timescale: posterior + regime alignment ----
        p_1m = posterior.p_up

        # Regime alignment: how well does the current regime support the direction
        regime_probs = regime_estimate.regime_probabilities
        regime_state = regime_estimate.current_regime

        # Compute regime directional bias
        # Trend regimes have strong directional bias
        calm_trending_prob = regime_probs.get(RegimeState.CALM_TRENDING, 0.0)
        volatile_trending_prob = regime_probs.get(RegimeState.VOLATILE_TRENDING, 0.0)
        calm_range_prob = regime_probs.get(RegimeState.CALM_RANGE, 0.0)
        volatile_range_prob = regime_probs.get(RegimeState.VOLATILE_RANGE, 0.0)

        # Net regime directional score
        # Trending regimes support direction, range regimes oppose it
        regime_direction_score = (
            calm_trending_prob * 0.8 + volatile_trending_prob * 0.6
            - calm_range_prob * 0.3 - volatile_range_prob * 0.4
        )

        # Blend regime direction with posterior
        regime_weight = 0.2  # Regime contributes 20% at 1m timescale
        p_1m = (1.0 - regime_weight) * p_1m + regime_weight * (0.5 + regime_direction_score * 0.5)

        # Continuation at 1m scale
        if continuation_signal is not None:
            cont_strength = continuation_signal.continuation_probability
            cont_dir = continuation_signal.continuation_direction
            if cont_dir == Direction.UP:
                p_1m += cont_strength * 0.1
            else:
                p_1m -= cont_strength * 0.1

        estimates["1m"] = np.clip(p_1m, 0.01, 0.99)

        # ---- 5-minute timescale: settlement probability ----
        # This is the final settlement probability
        # Use hazard rates for settlement modeling
        hazard_up = hazard_estimate.settlement_hazard_up
        hazard_down = hazard_estimate.settlement_hazard_down

        # Survival analysis approach:
        # P(UP at settlement) = 1 - exp(-hazard_up * T) / (1 - exp(-(hazard_up + hazard_down) * T))
        # Simplified: ratio of hazard rates
        total_hazard = hazard_up + hazard_down
        if total_hazard > 1e-10:
            # Normalize hazard rates to get probability
            p_hazard = hazard_up / total_hazard
        else:
            p_hazard = 0.5

        # Blend hazard-based probability with Bayesian posterior
        # At 5m scale, posterior is the primary signal, hazard is secondary
        p_5m = 0.6 * posterior.p_up + 0.4 * p_hazard

        # Time decay: as settlement approaches, probability converges
        # to the current direction (less time for reversal)
        time_fraction = time_to_settlement / 300.0  # 0 to 1 (1 = full 5 min remaining)
        if time_fraction < 0.2:
            # Less than 1 minute remaining: current direction dominates
            current_direction_prob = p_velocity  # Use current velocity direction
            blend_weight = 0.3 * (1.0 - time_fraction / 0.2)  # Up to 30% weight
            p_5m = (1.0 - blend_weight) * p_5m + blend_weight * current_direction_prob

        estimates["5m"] = np.clip(p_5m, 0.01, 0.99)

        # Apply EMA smoothing to reduce timescale noise
        for ts in self.TIMESCALES:
            if ts in self._prev_estimates:
                prev = self._prev_estimates[ts]
                estimates[ts] = (
                    self._smoothing_alpha * estimates[ts]
                    + (1.0 - self._smoothing_alpha) * prev
                )
            self._prev_estimates[ts] = estimates[ts]

        # Entropy suppression: reduce extreme probabilities when entropy is high
        if entropy_filter is not None:
            suppression = entropy_filter.suppression_factor
            for ts in self.TIMESCALES:
                # Pull toward 0.5 by suppression amount
                estimates[ts] = 0.5 + (estimates[ts] - 0.5) * suppression

        self._estimate_count += 1
        return estimates

    def compute_regime_weights(
        self,
        regime_estimate: RegimeStateEstimate,
        volatility_estimate: VolatilityEstimate,
    ) -> Dict[str, float]:
        """
        Compute regime-adaptive weights for each timescale.
        
        Uses exponential decay around the dominant timescale with
        regime-dependent concentration parameter kappa:
        
        w(t) = exp(-kappa * |t_seconds - t_dominant_seconds| / 300)
        
        Then normalized to sum to 1.
        """
        regime_state = regime_estimate.current_regime
        concentration = self.REGIME_CONCENTRATION.get(regime_state, 1.0)
        dominant = self.REGIME_DOMINANT_BIAS.get(regime_state, "1m")
        dominant_seconds = self.TIMESCALE_SECONDS[dominant]

        # Exponential decay weights around dominant timescale
        raw_weights = {}
        for ts in self.TIMESCALES:
            ts_seconds = self.TIMESCALE_SECONDS[ts]
            distance = abs(ts_seconds - dominant_seconds) / 300.0  # Normalize to 5m
            raw_weights[ts] = np.exp(-concentration * distance)

        # Blend with base weights (never fully ignore any timescale)
        blend_factor = 0.3  # 30% base, 70% regime-adaptive
        total_raw = sum(raw_weights.values())
        normalized_regime = {ts: w / total_raw for ts, w in raw_weights.items()}

        weights = {}
        for ts in self.TIMESCALES:
            weights[ts] = (
                blend_factor * self.BASE_WEIGHTS[ts]
                + (1.0 - blend_factor) * normalized_regime[ts]
            )

        # Normalize to sum to 1
        total = sum(weights.values())
        weights = {ts: w / total for ts, w in weights.items()}

        return weights

    def compute_consistency_and_conflict(
        self,
        timescale_probs: Dict[str, float],
    ) -> Tuple[float, float, str]:
        """
        Compute timescale consistency and conflict metrics.
        
        Consistency: how aligned are the timescale signals?
        - Normalize probabilities to directional signals: s(t) = 2*(p(t) - 0.5)
        - Consistency = 1 - variance(s) / max_possible_variance
        
        Conflict: how many timescales disagree on direction?
        - Pairwise sign disagreements between adjacent timescales
        
        Returns (consistency, conflict_score, dominant_timescale).
        """
        # Directional signals: centered at 0, range [-1, 1]
        signals = {ts: 2.0 * (prob - 0.5) for ts, prob in timescale_probs.items()}

        # Consistency via variance of normalized signals
        signal_values = np.array(list(signals.values()))
        signal_variance = np.var(signal_values)
        max_variance = 1.0  # Maximum possible variance of [-1,1] uniform
        consistency = 1.0 - signal_variance / max_variance
        consistency = np.clip(consistency, 0.0, 1.0)

        # Conflict score: pairwise sign disagreements
        conflict_count = 0
        total_pairs = 0
        ts_list = self.TIMESCALES
        for i in range(len(ts_list)):
            for j in range(i + 1, len(ts_list)):
                total_pairs += 1
                if np.sign(signals[ts_list[i]]) != np.sign(signals[ts_list[j]]):
                    conflict_count += 1

        conflict_score = conflict_count / max(1, total_pairs)

        # Dominant timescale: highest absolute signal strength
        dominant = max(ts_list, key=lambda ts: abs(signals[ts]))

        return float(consistency), float(conflict_score), dominant

    def aggregate(
        self,
        kalman_state: KalmanState,
        posterior: BayesianPosterior,
        hazard_estimate: HazardEstimate,
        regime_estimate: RegimeStateEstimate,
        volatility_estimate: VolatilityEstimate,
        continuation_signal: Optional[ContinuationSignal] = None,
        reversal_signal: Optional[ReversalSignal] = None,
        entropy_filter: Optional[EntropyFilterResult] = None,
        time_to_settlement: float = 300.0,
    ) -> MultiTimescaleEstimate:
        """
        Full multi-timescale aggregation producing MultiTimescaleEstimate.
        
        Process:
        1. Compute per-timescale probabilities
        2. Compute regime-adaptive weights
        3. Compute consistency and conflict metrics
        4. Weighted aggregation with conflict penalty
        5. Produce final estimate with all metadata
        """
        timestamp = kalman_state.timestamp

        # Step 1: Per-timescale probabilities
        timescale_probs = self.estimate_timescale_probabilities(
            kalman_state=kalman_state,
            posterior=posterior,
            hazard_estimate=hazard_estimate,
            regime_estimate=regime_estimate,
            volatility_estimate=volatility_estimate,
            continuation_signal=continuation_signal,
            reversal_signal=reversal_signal,
            entropy_filter=entropy_filter,
            time_to_settlement=time_to_settlement,
        )

        # Step 2: Regime-adaptive weights
        weights = self.compute_regime_weights(
            regime_estimate=regime_estimate,
            volatility_estimate=volatility_estimate,
        )

        # Step 3: Consistency and conflict
        consistency, conflict_score, dominant_ts = self.compute_consistency_and_conflict(
            timescale_probs=timescale_probs,
        )

        # Step 4: Weighted aggregation with conflict penalty
        # When timescales conflict, pull toward 0.5 (uncertainty increases)
        conflict_penalty = conflict_score * 0.3  # Up to 30% pull toward 0.5

        weighted_prob = 0.0
        for ts in self.TIMESCALES:
            weighted_prob += weights[ts] * timescale_probs[ts]

        # Apply conflict penalty: pull toward 0.5
        weighted_prob = 0.5 + (weighted_prob - 0.5) * (1.0 - conflict_penalty)

        # Consistency boost: when highly consistent, allow more extreme probabilities
        if consistency > 0.8:
            # Expand away from 0.5 by up to 10%
            expansion = 1.0 + 0.1 * (consistency - 0.8) / 0.2
            weighted_prob = 0.5 + (weighted_prob - 0.5) * expansion

        weighted_prob = np.clip(weighted_prob, 0.01, 0.99)

        # Step 5: Categorize signals
        short_term = (timescale_probs["1s"] + timescale_probs["5s"]) / 2.0
        medium_term = (timescale_probs["30s"] + timescale_probs["1m"]) / 2.0
        long_term = timescale_probs["5m"]

        return MultiTimescaleEstimate(
            timestamp=timestamp,
            timescale_estimates=timescale_probs,
            timescale_weights=weights,
            weighted_direction_probability=float(weighted_prob),
            timescale_consistency=consistency,
            dominant_timescale=dominant_ts,
            short_term_signal=float(short_term),
            medium_term_signal=float(medium_term),
            long_term_signal=float(long_term),
            timescale_conflict_score=conflict_score,
        )
