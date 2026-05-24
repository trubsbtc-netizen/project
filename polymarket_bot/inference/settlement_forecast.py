"""
Settlement probability forecast for Polymarket BTC 5-minute markets.

This is the final inference output that produces the probability estimate
for whether BTC will resolve UP or DOWN at the 5-minute settlement time.
It combines all inference outputs (Bayesian posterior, multi-timescale,
hazard rates, Kalman state) into a single calibrated settlement forecast
with confidence intervals, edge estimates, and execution recommendations.

Mathematical basis:
- P(settlement UP) = calibrated fusion of posterior + multi-timescale + hazard
- Confidence = f(evidence_strength, consistency, calibration_score)
- Edge = directional probability distance from 50/50, not market mispricing
- Execution recommendation via threshold on edge * confidence
- Probability range via Beta distribution credible intervals
- Time-weighted adjustment: as settlement approaches, current state dominates
"""

import numpy as np
from math import erf, sqrt
from typing import Optional, Tuple
from scipy.special import stdtr

from polymarket_bot.bot_types import (
    SettlementForecast,
    BayesianPosterior,
    MultiTimescaleEstimate,
    HazardEstimate,
    KalmanState,
    RegimeStateEstimate,
    VolatilityEstimate,
    Direction,
    RegimeState,
    ContinuationSignal,
    ReversalSignal,
    EntropyFilterResult,
    ObservationSignal,
    TechnicalMomentumSignal,
)
from polymarket_bot.config import BotConfig


class SettlementForecaster:
    """
    Produces the final settlement probability forecast.
    
    The forecast is the key output that drives all trading decisions.
    It fuses multiple probability estimates with regime-aware weighting,
    applies calibration corrections, and computes edge vs. market prices.
    
    Fusion hierarchy (by information quality):
    1. Bayesian posterior (most rigorous, full evidence integration)
    2. Multi-timescale estimate (regime-adaptive, consistency-aware)
    3. Hazard rates (survival analysis perspective)
    4. Kalman velocity (fast but noisy)
    
    The fusion weights are regime-dependent:
    - Trend regimes: trust posterior + multi-timescale more
    - Reversal regimes: trust hazard + Kalman velocity more
    - Volatile regimes: trust nothing, widen uncertainty
    """

    # Regime-adaptive fusion weights for [posterior, multi_timescale, hazard, kalman]
    REGIME_FUSION_WEIGHTS = {
        RegimeState.CALM_TRENDING:      [0.40, 0.35, 0.15, 0.10],
        RegimeState.CALM_RANGE:         [0.30, 0.30, 0.25, 0.15],
        RegimeState.VOLATILE_TRENDING:  [0.35, 0.30, 0.20, 0.15],
        RegimeState.VOLATILE_RANGE:     [0.30, 0.25, 0.25, 0.20],
        RegimeState.CRISIS:             [0.25, 0.25, 0.25, 0.25],
        RegimeState.LIQUIDATION_CASCADE: [0.20, 0.20, 0.30, 0.30],
    }

    # Minimum edge thresholds (in bps) for execution, by regime. This is an
    # inference-only positive EV gate; execution costs and final confidence are
    # still checked in DecisionEngine.
    REGIME_EDGE_THRESHOLDS = {
        RegimeState.CALM_TRENDING:      100,
        RegimeState.CALM_RANGE:         150,
        RegimeState.VOLATILE_TRENDING:  150,
        RegimeState.VOLATILE_RANGE:     200,
        RegimeState.CRISIS:             300,
        RegimeState.LIQUIDATION_CASCADE: 400,
    }

    # Minimum confidence for execution. Final execution still goes through
    # DecisionEngine/risk gates; this layer should not veto positive edge too early.
    MIN_CONFIDENCE_EXECUTE = 0.50

    # Maximum position size fraction (Kelly-derived, capped)
    MAX_SIZE_FRACTION = 0.25

    def __init__(self, config: BotConfig):
        self.config = config
        self._forecast_history = []  # For calibration tracking
        self._max_history = 1000

    def _fuse_probabilities(
        self,
        posterior: BayesianPosterior,
        multi_timescale: MultiTimescaleEstimate,
        hazard: HazardEstimate,
        kalman: KalmanState,
        regime_estimate: RegimeStateEstimate,
    ) -> float:
        """
        Fuse multiple probability estimates using regime-adaptive weights.
        
        P_fused = sum(w_i * P_i) where weights are regime-dependent.
        
        Each source provides P(UP):
        - posterior: Bayesian posterior P(UP)
        - multi_timescale: weighted_direction_probability
        - hazard: settlement_hazard_up / (settlement_hazard_up + settlement_hazard_down)
        - kalman: sigmoid(velocity / temperature)
        """
        regime_state = regime_estimate.current_regime
        weights = self.REGIME_FUSION_WEIGHTS.get(
            regime_state, [0.25, 0.25, 0.25, 0.25]
        )

        # Source probabilities
        p_posterior = posterior.p_up
        p_multits = multi_timescale.weighted_direction_probability

        # Hazard-based probability
        total_hazard = hazard.settlement_hazard_up + hazard.settlement_hazard_down
        if total_hazard > 1e-10:
            p_hazard = hazard.settlement_hazard_up / total_hazard
        else:
            p_hazard = 0.5

        # Kalman velocity-based probability
        velocity = kalman.velocity
        # Temperature scales with velocity confidence (low confidence = high temperature = less extreme)
        temp = max(0.5, 1.0 / max(0.01, kalman.velocity_confidence))
        p_kalman = 1.0 / (1.0 + np.exp(-velocity / temp))

        # Weighted fusion
        p_fused = (
            weights[0] * p_posterior
            + weights[1] * p_multits
            + weights[2] * p_hazard
            + weights[3] * p_kalman
        )

        return np.clip(p_fused, 0.01, 0.99)

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return 0.5 * (1.0 + erf(value / sqrt(2.0)))

    @staticmethod
    def _logit(probability: float) -> float:
        p = float(np.clip(probability, 1e-6, 1.0 - 1e-6))
        return float(np.log(p / (1.0 - p)))

    @staticmethod
    def _sigmoid(log_odds: float) -> float:
        if log_odds >= 0:
            z = np.exp(-log_odds)
            return float(1.0 / (1.0 + z))
        z = np.exp(log_odds)
        return float(z / (1.0 + z))

    def _directional_evidence_probability(
        self,
        posterior: BayesianPosterior,
        multi_timescale: MultiTimescaleEstimate,
        hazard: HazardEstimate,
    ) -> float:
        total_hazard = hazard.settlement_hazard_up + hazard.settlement_hazard_down
        p_hazard = (
            hazard.settlement_hazard_up / total_hazard
            if total_hazard > 1e-12
            else 0.5
        )
        p = np.average(
            [
                posterior.p_up,
                multi_timescale.weighted_direction_probability,
                p_hazard,
            ],
            weights=[
                max(0.1, posterior.evidence_strength),
                max(0.1, multi_timescale.timescale_consistency),
                0.5,
            ],
        )
        return float(np.clip(p, 1e-6, 1.0 - 1e-6))

    def _forecast_volatility_per_sqrt_second(
        self,
        volatility_estimate: VolatilityEstimate,
        empirical_sigma_per_sqrt_second: Optional[float] = None,
    ) -> float:
        one_step_candidates = [
            volatility_estimate.realized_volatility,
            volatility_estimate.kalman_volatility,
            volatility_estimate.regime_adjusted_volatility,
        ]
        valid = [
            float(v)
            for v in one_step_candidates
            if v is not None and np.isfinite(v) and v > 0
        ]
        if not valid:
            return 1e-5

        sigma = float(np.median(valid))
        forecast_5min = getattr(volatility_estimate, "volatility_forecast_5min", None)
        if (
            forecast_5min is not None
            and np.isfinite(forecast_5min)
            and forecast_5min > 0
        ):
            forecast_per_sqrt_second = float(forecast_5min) / np.sqrt(300.0)
            if forecast_per_sqrt_second > 1.0e-7:
                sigma = 0.88 * sigma + 0.12 * forecast_per_sqrt_second
        if (
            empirical_sigma_per_sqrt_second is not None
            and np.isfinite(empirical_sigma_per_sqrt_second)
            and empirical_sigma_per_sqrt_second > 0
        ):
            sigma = 0.80 * float(empirical_sigma_per_sqrt_second) + 0.20 * sigma
        upper = volatility_estimate.volatility_confidence_interval[1]
        if upper and np.isfinite(upper) and upper > 0:
            sigma = 0.75 * sigma + 0.25 * float(upper)
        return float(np.clip(sigma, 1e-6, 0.005))

    def _regularized_drift_per_second(
        self,
        btc_price: float,
        time_to_settlement: float,
        kalman: KalmanState,
        posterior: BayesianPosterior,
        multi_timescale: MultiTimescaleEstimate,
        hazard: HazardEstimate,
        sigma_per_sqrt_second: float,
        empirical_drift_per_second: Optional[float] = None,
    ) -> float:
        if btc_price <= 0 or time_to_settlement <= 0:
            return 0.0

        velocity_log = kalman.velocity / btc_price
        velocity_weight = 0.25 * float(np.clip(kalman.velocity_confidence, 0.0, 1.0))
        raw_drift = velocity_weight * velocity_log
        if (
            empirical_drift_per_second is not None
            and np.isfinite(empirical_drift_per_second)
        ):
            raw_drift = 0.75 * float(empirical_drift_per_second) + 0.25 * raw_drift

        max_abs_drift = max(
            2e-7,
            0.25 * max(float(sigma_per_sqrt_second), 1e-8) / np.sqrt(max(time_to_settlement, 1.0)),
        )
        return float(np.clip(raw_drift, -max_abs_drift, max_abs_drift))

    def _terminal_probability_up(
        self,
        *,
        btc_price: float,
        price_to_beat: float,
        time_to_settlement: float,
        posterior: BayesianPosterior,
        multi_timescale: MultiTimescaleEstimate,
        hazard: HazardEstimate,
        kalman: KalmanState,
        volatility_estimate: VolatilityEstimate,
        empirical_sigma_per_sqrt_second: Optional[float] = None,
        empirical_drift_per_second: Optional[float] = None,
        empirical_sample_count: int = 0,
    ) -> Tuple[float, float, float, float, float]:
        tau = max(float(time_to_settlement), 0.25)
        if btc_price <= 0 or price_to_beat <= 0:
            fallback = self._directional_evidence_probability(
                posterior,
                multi_timescale,
                hazard,
            )
            return fallback, 0.0, 0.0, 0.0, 0.0

        sigma = self._forecast_volatility_per_sqrt_second(
            volatility_estimate,
            empirical_sigma_per_sqrt_second=empirical_sigma_per_sqrt_second,
        )
        mu = self._regularized_drift_per_second(
            btc_price=btc_price,
            time_to_settlement=tau,
            kalman=kalman,
            posterior=posterior,
            multi_timescale=multi_timescale,
            hazard=hazard,
            sigma_per_sqrt_second=sigma,
            empirical_drift_per_second=empirical_drift_per_second,
        )

        raw_distance_log = float(np.log(float(btc_price) / float(price_to_beat)))
        tie_break_bps = max(0.0, float(getattr(self.config, "settlement_tie_break_bps", 1.5)))
        tie_break_abs_usd = max(0.0, float(getattr(self.config, "settlement_tie_break_abs_usd", 1.0)))
        tie_log_buffer = max(
            tie_break_bps * 1.0e-4,
            float(np.log1p(tie_break_abs_usd / max(float(price_to_beat), 1.0e-9))),
        )
        # UP resolves on equality; DOWN only wins when the settlement print is
        # clearly below PTB. Assign a small no-trade/tie band to UP so DOWN is
        # not overestimated around an exact or near-exact close.
        distance_log = raw_distance_log + tie_log_buffer
        variance = max((sigma ** 2) * tau, 1e-12)
        # mu is estimated in log-price units, so no Ito correction is applied
        # here. The -0.5*sigma^2 term only belongs when mu is arithmetic
        # return drift, not when it already describes d log(S).
        mean_log_terminal = distance_log + mu * tau

        n_eff = float(max(0, int(empirical_sample_count)))
        if n_eff >= 4:
            prior_kappa = 8.0
            kappa_n = prior_kappa + n_eff
            df = max(3.0, n_eff + 4.0)
            predictive_scale = float(np.sqrt(max(
                variance + (sigma ** 2) * (tau * tau) / kappa_n,
                1e-12,
            )))
            z_score = mean_log_terminal / predictive_scale
            p_terminal = float(stdtr(df, z_score))
        else:
            z_score = mean_log_terminal / np.sqrt(variance)
            p_terminal = self._normal_cdf(float(z_score))
        return (
            float(np.clip(p_terminal, 0.001, 0.999)),
            float(z_score),
            float(sigma),
            float(mu),
            float(raw_distance_log),
        )

    def _compute_confidence(
        self,
        posterior: BayesianPosterior,
        multi_timescale: MultiTimescaleEstimate,
        regime_estimate: RegimeStateEstimate,
        volatility_estimate: VolatilityEstimate,
        entropy_filter: Optional[EntropyFilterResult] = None,
    ) -> float:
        """
        Compute forecast confidence from multiple quality indicators.
        
        Confidence = geometric mean of:
        1. Evidence strength (how much data supports the estimate)
        2. Timescale consistency (how aligned are different timescales)
        3. Calibration score (how well calibrated historically)
        4. Regime certainty (how confident in regime classification)
        5. Entropy suppression (how much noise is filtered)
        
        All components are in [0, 1], geometric mean ensures that
        any single low component significantly reduces confidence.
        """
        # Evidence strength: normalize to [0, 1]
        evidence = posterior.evidence_strength
        evidence_score = min(1.0, evidence / 10.0)  # 10 units = full confidence

        # Timescale consistency
        consistency_score = multi_timescale.timescale_consistency

        # Calibration score from posterior
        calibration_score = min(1.0, posterior.calibration_score)

        # Regime certainty: max probability of any regime
        regime_probs = regime_estimate.regime_probabilities
        regime_certainty = max(regime_probs.values()) if regime_probs else 0.5

        # Entropy suppression (higher = more confident after filtering)
        entropy_score = 0.5  # Default
        if entropy_filter is not None:
            entropy_score = entropy_filter.suppression_factor

        # Volatility penalty: high volatility reduces confidence
        vol_realized = volatility_estimate.realized_volatility
        vol_penalty = max(0.0, 1.0 - vol_realized * 10.0)  # High vol = low penalty score

        # Geometric mean of all components
        components = [
            max(0.01, evidence_score),
            max(0.01, consistency_score),
            max(0.01, calibration_score),
            max(0.01, regime_certainty),
            max(0.01, entropy_score),
            max(0.01, vol_penalty),
        ]

        confidence = np.exp(np.mean(np.log(components)))
        return np.clip(float(confidence), 0.01, 0.99)

    def _compute_uncertainty(
        self,
        p_fused: float,
        confidence: float,
        posterior: BayesianPosterior,
        volatility_estimate: VolatilityEstimate,
        time_to_settlement: float,
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Compute uncertainty and probability range.
        
        Uncertainty is derived from:
        1. Posterior variance (Bayesian uncertainty)
        2. Confidence (inverse relationship)
        3. Volatility (higher vol = wider range)
        4. Time to settlement (more time = more uncertainty)
        
        Probability range uses Beta distribution approximation:
        Given P_fused and confidence, approximate Beta parameters:
        alpha = P * N, beta = (1-P) * N where N = confidence * concentration
        
        Then credible interval = Beta(alpha, beta) quantiles.
        """
        # Base uncertainty from confidence (inverse)
        base_uncertainty = 1.0 - confidence

        # Volatility amplification
        vol_realized = volatility_estimate.realized_volatility
        vol_amplification = 1.0 + vol_realized * 5.0  # High vol widens uncertainty

        # Time factor: more time remaining = more uncertainty
        time_fraction = time_to_settlement / 300.0  # 0 to 1
        time_factor = 0.5 + 0.5 * time_fraction  # Range [0.5, 1.0]

        # Combined uncertainty
        uncertainty = base_uncertainty * vol_amplification * time_factor
        uncertainty = np.clip(uncertainty, 0.01, 0.5)

        # Probability range via Beta distribution
        # Concentration parameter: higher confidence = higher concentration
        concentration = confidence * 50.0  # Scale to reasonable Beta parameters
        concentration = max(2.0, concentration)  # Minimum for valid Beta

        alpha_param = p_fused * concentration
        beta_param = (1.0 - p_fused) * concentration

        # Ensure valid Beta parameters
        alpha_param = max(0.5, alpha_param)
        beta_param = max(0.5, beta_param)

        # 90% credible interval using Beta quantiles
        from scipy.stats import beta as beta_dist
        low = beta_dist.ppf(0.05, alpha_param, beta_param)
        high = beta_dist.ppf(0.95, alpha_param, beta_param)

        # Widen interval by uncertainty factor
        width = high - low
        widened_width = width * (1.0 + uncertainty)
        center = (low + high) / 2.0
        low = np.clip(center - widened_width / 2.0, 0.01, 0.99)
        high = np.clip(center + widened_width / 2.0, low + 0.02, 0.99)

        return float(uncertainty), (float(low), float(high))

    def _compute_directional_edge(
        self,
        p_up: float,
        expected_direction: Optional[Direction] = None,
    ) -> Tuple[float, float, float, Direction, float, float]:
        """
        Compute directional edge from the model probability.

        Direction is determined by the modeled probability, not by market price.
        """
        p_down = 1.0 - p_up
        if expected_direction == Direction.UP:
            direction = Direction.UP
            probability = p_up
        elif expected_direction == Direction.DOWN:
            direction = Direction.DOWN
            probability = p_down
        else:
            direction = Direction.UP if p_up >= p_down else Direction.DOWN
            probability = p_up if direction == Direction.UP else p_down

        directional_edge = float(probability - 0.5)
        return (
            float(directional_edge),
            float(directional_edge * 10000.0),
            float(directional_edge),
            direction,
            float(probability),
            0.5,
        )

    def _should_execute(
        self,
        edge_bps: float,
        confidence: float,
        regime_state: RegimeState,
        timescale_conflict: float,
        direction: Direction,
        selected_probability: float,
        selected_market_price: float,
        time_to_settlement: float,
    ) -> Tuple[bool, Direction, float, float, float, str]:
        """
        Determine whether the directional probability is strong enough.
        
        This gate does not evaluate Polymarket mispricing. The bot is a
        directional forecaster: orderbook price is handled later as an
        execution/risk constraint.
        
        Position size uses fractional Kelly:
        size = (edge * confidence) / (2 * max_loss) * kelly_fraction
        
        Capped at MAX_SIZE_FRACTION.
        """
        min_edge_bps = max(
            float(self.config.risk.min_edge_bps),
            0.0,
        )
        reject_reason = ""

        settlement_buffer_seconds = float(
            getattr(self.config.risk, "settlement_time_buffer_seconds", 30.0)
        )
        if float(time_to_settlement) <= settlement_buffer_seconds:
            return (
                False,
                direction,
                0.0,
                0.0,
                float(min_edge_bps),
                (
                    "settlement_time_buffer("
                    f"tau={float(time_to_settlement):.2f}s<="
                    f"{settlement_buffer_seconds:.2f}s)"
                ),
            )

        should_execute = direction != Direction.NEUTRAL
        if direction == Direction.NEUTRAL:
            reject_reason = "no_directional_side"

        probability = float(np.clip(selected_probability, 0.001, 0.999))
        min_directional_probability = float(np.clip(
            max(
                getattr(self.config, "directional_min_probability", 0.62),
                0.5 + min_edge_bps / 10000.0,
            ),
            0.5,
            0.99,
        ))
        min_directional_confidence = float(np.clip(
            getattr(self.config, "directional_min_confidence", 0.55),
            0.0,
            0.99,
        ))
        if should_execute and probability < min_directional_probability:
            should_execute = False
            reject_reason = (
                "directional_probability_below_threshold("
                f"{probability:.3f}<{min_directional_probability:.3f})"
            )
        if should_execute and confidence < min_directional_confidence:
            should_execute = False
            reject_reason = (
                "directional_confidence_below_threshold("
                f"{confidence:.3f}<{min_directional_confidence:.3f})"
            )

        # Directional sizing: stronger probability separation allows larger
        # size. This intentionally does not use market-price odds.
        directional_strength = max(0.0, probability - 0.5)
        confidence_factor = max(0.20, float(np.clip(confidence, 0.0, 0.99)))
        raw_size = directional_strength * confidence_factor
        size_fraction = np.clip(raw_size, 0.03, self.MAX_SIZE_FRACTION) if should_execute else 0.0

        # Execution confidence means confidence in direction, not market edge.
        exec_confidence = float(np.clip(confidence_factor * (0.5 + directional_strength), 0.0, 0.99)) if should_execute else 0.0

        return (
            should_execute,
            direction,
            float(size_fraction),
            float(exec_confidence),
            float(min_edge_bps),
            reject_reason,
        )

    def forecast(
        self,
        posterior: BayesianPosterior,
        multi_timescale: MultiTimescaleEstimate,
        hazard: HazardEstimate,
        kalman: KalmanState,
        regime_estimate: RegimeStateEstimate,
        volatility_estimate: VolatilityEstimate,
        continuation_signal: Optional[ContinuationSignal] = None,
        reversal_signal: Optional[ReversalSignal] = None,
        entropy_filter: Optional[EntropyFilterResult] = None,
        market_price_up: Optional[float] = None,
        market_price_down: Optional[float] = None,
        time_to_settlement: float = 300.0,
        btc_price: Optional[float] = None,
        price_to_beat: Optional[float] = None,
        empirical_sigma_per_sqrt_second: Optional[float] = None,
        empirical_drift_per_second: Optional[float] = None,
        empirical_sample_count: int = 0,
        observation_signal: Optional[ObservationSignal] = None,
        technical_signal: Optional[TechnicalMomentumSignal] = None,
        round_direction: Optional[Direction] = None,
        round_direction_p_up: Optional[float] = None,
        round_direction_confidence: float = 0.0,
        round_direction_reason: str = "",
    ) -> SettlementForecast:
        """
        Produce the final settlement probability forecast.
        
        This is the primary output of the inference layer. It combines
        all probability estimates, computes confidence and uncertainty,
        evaluates edge vs. market, and produces execution recommendations.
        
        Process:
        1. Fuse probabilities from all sources (regime-adaptive weights)
        2. Apply time-to-settlement adjustment
        3. Compute confidence from quality indicators
        4. Compute uncertainty and probability range
        5. Compute edge vs. market price
        6. Determine execution recommendation
        7. Produce SettlementForecast with all metadata
        """
        timestamp = kalman.timestamp

        # Step 1: compute the actual terminal barrier probability for this
        # market: P(S_T >= price_to_beat | S_t, tau, sigma, drift).
        if btc_price is not None and price_to_beat is not None:
            (
                p_fused,
                z_score,
                terminal_sigma,
                terminal_drift,
                terminal_distance_log,
            ) = self._terminal_probability_up(
                btc_price=btc_price,
                price_to_beat=price_to_beat,
                time_to_settlement=time_to_settlement,
                posterior=posterior,
                multi_timescale=multi_timescale,
                hazard=hazard,
                kalman=kalman,
                volatility_estimate=volatility_estimate,
                empirical_sigma_per_sqrt_second=empirical_sigma_per_sqrt_second,
                empirical_drift_per_second=empirical_drift_per_second,
                empirical_sample_count=empirical_sample_count,
            )
        else:
            p_fused = self._fuse_probabilities(
                posterior=posterior,
                multi_timescale=multi_timescale,
                hazard=hazard,
                kalman=kalman,
                regime_estimate=regime_estimate,
            )
            z_score = 0.0
            terminal_sigma = 0.0
            terminal_drift = 0.0
            terminal_distance_log = 0.0

        time_fraction = np.clip(time_to_settlement / 300.0, 0.0, 1.0)
        observation_ready = observation_signal.is_ready if observation_signal else False
        observation_valid = observation_signal.is_valid if observation_signal else False
        observation_seconds = observation_signal.observed_seconds if observation_signal else 0.0
        observation_p_up = observation_signal.p_up if observation_signal else None
        observation_confidence = observation_signal.confidence if observation_signal else 0.0
        observation_validation_score = observation_signal.validation_score if observation_signal else 0.0
        observation_reason = observation_signal.reason if observation_signal else "missing_observation_signal"
        observation_gate_reason = ""
        technical_valid = technical_signal.is_valid if technical_signal else False
        technical_p_up = technical_signal.p_up if technical_signal else None
        technical_confidence = technical_signal.confidence if technical_signal else 0.0
        technical_consensus_score = technical_signal.consensus_score if technical_signal else 0.0
        technical_log_odds = technical_signal.log_odds if technical_signal else 0.0
        technical_dominant_timeframe = technical_signal.dominant_timeframe if technical_signal else ""
        technical_reason = technical_signal.reason if technical_signal else "missing_technical_signal"
        round_direction_locked = round_direction in {Direction.UP, Direction.DOWN}
        round_locked_direction = round_direction if round_direction_locked else Direction.NEUTRAL
        round_locked_p_up = round_direction_p_up if round_direction_locked else None
        round_locked_confidence = float(round_direction_confidence if round_direction_locked else 0.0)
        round_locked_reason = round_direction_reason if round_direction_locked else ""

        if observation_signal is not None and observation_signal.is_valid:
            base_log_odds = self._logit(p_fused)
            obs_log_odds = self._logit(observation_signal.p_up)
            obs_sample_weight = float(np.clip(
                observation_signal.effective_sample_size
                / (observation_signal.effective_sample_size + 18.0),
                0.0,
                1.0,
            ))
            obs_uncertainty_weight = float(np.clip(
                1.0 - 1.65 * observation_signal.uncertainty,
                0.15,
                1.0,
            ))
            obs_weight = float(np.clip(
                observation_signal.confidence
                * observation_signal.validation_score
                * obs_sample_weight
                * obs_uncertainty_weight,
                0.10,
                0.64,
            ))
            conflict = abs(base_log_odds - obs_log_odds)
            pooled_log_odds = (1.0 - obs_weight) * base_log_odds + obs_weight * obs_log_odds
            p_fused = float(np.clip(self._sigmoid(float(pooled_log_odds)), 0.001, 0.999))
            observation_reason = (
                f"valid,w={obs_weight:.2f},conflict={conflict:.2f},"
                f"{observation_signal.reason}"
            )
        elif getattr(self.config, "require_observation_window", True):
            if observation_signal is None:
                observation_gate_reason = "observation_missing"
            elif not observation_signal.is_ready:
                observation_gate_reason = observation_signal.reason or "observation_pending"
            else:
                observation_gate_reason = (
                    f"observation_invalid({observation_signal.reason})"
                )

        if (
            bool(getattr(self.config, "technical_momentum_enabled", True))
            and technical_signal is not None
            and technical_signal.is_valid
            and not round_direction_locked
        ):
            base_log_odds = self._logit(p_fused)
            tech_weight = float(np.clip(
                getattr(self.config, "technical_momentum_fusion_weight", 0.55)
                * technical_signal.confidence
                * technical_signal.validation_score
                * (0.40 + 0.60 * technical_signal.consensus_score),
                0.0,
                0.45,
            ))
            tau_factor = float(np.clip(1.0 - time_fraction * 0.35, 0.55, 1.0))
            bounded_tech_llr = float(np.clip(
                technical_signal.log_odds,
                -getattr(self.config, "technical_momentum_max_log_odds", 2.2),
                getattr(self.config, "technical_momentum_max_log_odds", 2.2),
            ))
            fused_log_odds = base_log_odds + tech_weight * tau_factor * bounded_tech_llr
            p_fused = float(np.clip(self._sigmoid(fused_log_odds), 0.001, 0.999))
            technical_reason = (
                f"valid,w={tech_weight:.2f},tf={technical_signal.dominant_timeframe},"
                f"{technical_signal.reason}"
            )

        if round_direction_locked:
            locked_probability = None
            if round_locked_p_up is not None and np.isfinite(round_locked_p_up):
                locked_probability = float(np.clip(round_locked_p_up, 0.001, 0.999))
            elif round_locked_direction == Direction.UP:
                locked_probability = float(max(p_fused, 0.5))
            elif round_locked_direction == Direction.DOWN:
                locked_probability = float(min(p_fused, 0.5))

            if locked_probability is not None:
                base_log_odds = self._logit(p_fused)
                lock_log_odds = self._logit(locked_probability)
                lock_weight = float(np.clip(
                    0.68 + 0.22 * round_locked_confidence,
                    0.68,
                    0.90,
                ))
                p_fused = float(np.clip(
                    self._sigmoid((1.0 - lock_weight) * base_log_odds + lock_weight * lock_log_odds),
                    0.001,
                    0.999,
                ))

            lock_floor = float(np.clip(
                0.535 + 0.085 * round_locked_confidence,
                0.535,
                0.620,
            ))
            if round_locked_direction == Direction.UP:
                p_fused = float(np.clip(max(p_fused, lock_floor), 0.001, 0.999))
            elif round_locked_direction == Direction.DOWN:
                p_fused = float(np.clip(min(p_fused, 1.0 - lock_floor), 0.001, 0.999))

            round_locked_reason = (
                f"enforced,dir={round_locked_direction.value},"
                f"{round_locked_reason}"
            )

        # Step 3: Confidence
        confidence = self._compute_confidence(
            posterior=posterior,
            multi_timescale=multi_timescale,
            regime_estimate=regime_estimate,
            volatility_estimate=volatility_estimate,
            entropy_filter=entropy_filter,
        )
        if btc_price is not None and price_to_beat is not None:
            distance_confidence = 1.0 - np.exp(-abs(z_score))
            horizon_confidence = 1.0 - 0.35 * time_fraction
            barrier_confidence = np.clip(0.25 + 0.75 * distance_confidence, 0.25, 0.98)
            confidence = float(np.clip(
                0.25 * confidence + 0.55 * barrier_confidence + 0.20 * horizon_confidence,
                0.01,
                0.99,
            ))
        if observation_signal is not None and observation_signal.is_valid:
            obs_conflict_penalty = np.exp(
                -abs(self._logit(p_fused) - self._logit(observation_signal.p_up)) / 3.0
            )
            confidence = float(np.clip(
                0.55 * confidence
                + 0.45 * observation_signal.confidence * obs_conflict_penalty,
                0.01,
                0.99,
            ))
        if technical_signal is not None and technical_signal.is_valid:
            tech_conflict_penalty = np.exp(
                -abs(self._logit(p_fused) - technical_signal.log_odds) / 3.5
            )
            confidence = float(np.clip(
                0.72 * confidence
                + 0.28 * technical_signal.confidence * tech_conflict_penalty,
                0.01,
                0.99,
            ))

        # Step 4: Uncertainty and probability range
        uncertainty, prob_range = self._compute_uncertainty(
            p_fused=p_fused,
            confidence=confidence,
            posterior=posterior,
            volatility_estimate=volatility_estimate,
            time_to_settlement=time_to_settlement,
        )

        expected_direction = (
            round_locked_direction
            if round_direction_locked
            else Direction.UP if p_fused >= 0.5 else Direction.DOWN
        )

        # Step 5: Directional edge. Market price is deliberately not used to
        # choose direction; it is only attached for execution observability.
        (
            edge,
            edge_bps,
            edge_absolute,
            edge_direction,
            selected_probability,
            _directional_reference_price,
        ) = self._compute_directional_edge(
            p_up=p_fused,
            expected_direction=expected_direction,
        )
        selected_market_price = (
            market_price_up
            if edge_direction == Direction.UP
            else market_price_down
            if edge_direction == Direction.DOWN
            else None
        )

        # Step 6: Execution recommendation
        regime_state = regime_estimate.current_regime
        (
            should_execute,
            direction,
            size_fraction,
            exec_confidence,
            min_edge_bps_required,
            reject_reason,
        ) = self._should_execute(
            edge_bps=edge_bps,
            confidence=confidence,
            regime_state=regime_state,
            timescale_conflict=multi_timescale.timescale_conflict_score,
            direction=edge_direction,
            selected_probability=selected_probability,
            selected_market_price=selected_market_price,
            time_to_settlement=time_to_settlement,
        )
        if round_direction_locked:
            direction = round_locked_direction
            if should_execute:
                size_fraction = max(float(size_fraction), 0.05)
                exec_confidence = max(
                    float(exec_confidence),
                    float(np.clip(round_locked_confidence, 0.0, 0.99)),
                )
                reject_reason = f"round_direction_locked({round_locked_reason})"
        elif bool(getattr(self.config, "require_round_direction_lock_for_entry", True)):
            should_execute = False
            size_fraction = 0.0
            exec_confidence = 0.0
            reject_reason = "round_direction_lock_pending"
        if observation_gate_reason:
            should_execute = bool(
                should_execute
                and confidence >= float(getattr(self.config, "directional_min_confidence", 0.40))
            )
            if not should_execute:
                size_fraction = 0.0
                exec_confidence = 0.0
                reject_reason = observation_gate_reason
            else:
                size_fraction = max(0.03, size_fraction * 0.60)
                exec_confidence *= 0.75
                reject_reason = f"observation_soft_gate({observation_gate_reason})"

        # Edge confidence: how confident we are in the directional estimate
        edge_confidence = confidence * (1.0 - multi_timescale.timescale_conflict_score)

        # Forecast quality: composite metric
        # High quality = high confidence + low conflict + low uncertainty + good calibration
        forecast_quality = (
            confidence * 0.3
            + (1.0 - multi_timescale.timescale_conflict_score) * 0.2
            + (1.0 - uncertainty) * 0.2
            + posterior.calibration_score * 0.15
            + min(1.0, (edge_absolute * 10000.0) / max(1.0, min_edge_bps_required * 3.0)) * 0.15
        )
        forecast_quality = np.clip(forecast_quality, 0.01, 0.99)

        # Store for calibration tracking
        self._forecast_history.append({
            "timestamp": timestamp,
            "p_up": p_fused,
            "confidence": confidence,
            "direction": expected_direction,
            "btc_price": btc_price,
            "price_to_beat": price_to_beat,
            "z_score": z_score,
            "terminal_sigma": terminal_sigma,
            "terminal_drift": terminal_drift,
            "terminal_distance_log": terminal_distance_log,
            "market_price_up": market_price_up,
            "market_price_down": market_price_down,
            "selected_probability": selected_probability,
            "selected_market_price": selected_market_price,
            "reject_reason": reject_reason,
            "observation_ready": observation_ready,
            "observation_valid": observation_valid,
            "observation_p_up": observation_p_up,
            "observation_confidence": observation_confidence,
            "observation_validation_score": observation_validation_score,
            "observation_reason": observation_reason,
            "technical_valid": technical_valid,
            "technical_p_up": technical_p_up,
            "technical_confidence": technical_confidence,
            "technical_consensus_score": technical_consensus_score,
            "technical_log_odds": technical_log_odds,
            "technical_reason": technical_reason,
            "round_direction_locked": round_direction_locked,
            "round_direction": round_locked_direction,
            "round_direction_confidence": round_locked_confidence,
            "round_direction_p_up": round_locked_p_up,
            "round_direction_reason": round_locked_reason,
        })
        if len(self._forecast_history) > self._max_history:
            self._forecast_history = self._forecast_history[-self._max_history:]

        return SettlementForecast(
            timestamp=timestamp,
            p_up_settlement=float(p_fused),
            p_down_settlement=float(1.0 - p_fused),
            time_to_settlement_seconds=time_to_settlement,
            settlement_confidence=float(confidence),
            settlement_uncertainty=float(uncertainty),
            settlement_probability_range=prob_range,
            expected_settlement_direction=expected_direction,
            edge_estimate=float(edge_absolute),
            edge_confidence=float(edge_confidence),
            forecast_quality=float(forecast_quality),
            execution_recommended=should_execute,
            execution_direction=direction,
            execution_size_fraction=float(size_fraction),
            execution_confidence=float(exec_confidence),
            btc_price=btc_price,
            price_to_beat=price_to_beat,
            terminal_z_score=float(z_score),
            terminal_sigma_per_sqrt_second=float(terminal_sigma),
            terminal_drift_per_second=float(terminal_drift),
            terminal_distance_log=float(terminal_distance_log),
            market_price_up=market_price_up,
            market_price_down=market_price_down,
            selected_market_price=(
                float(selected_market_price)
                if selected_market_price is not None and np.isfinite(selected_market_price)
                else None
            ),
            selected_probability=float(selected_probability),
            min_edge_bps_required=float(min_edge_bps_required),
            reject_reason=reject_reason,
            observation_ready=observation_ready,
            observation_valid=observation_valid,
            observation_seconds=float(observation_seconds),
            observation_p_up=(
                float(observation_p_up)
                if observation_p_up is not None and np.isfinite(observation_p_up)
                else None
            ),
            observation_confidence=float(observation_confidence),
            observation_validation_score=float(observation_validation_score),
            observation_reason=observation_reason,
            technical_valid=technical_valid,
            technical_p_up=(
                float(technical_p_up)
                if technical_p_up is not None and np.isfinite(technical_p_up)
                else None
            ),
            technical_confidence=float(technical_confidence),
            technical_consensus_score=float(technical_consensus_score),
            technical_log_odds=float(technical_log_odds),
            technical_dominant_timeframe=technical_dominant_timeframe,
            technical_reason=technical_reason,
            round_direction_locked=round_direction_locked,
            round_direction=round_locked_direction,
            round_direction_confidence=float(round_locked_confidence),
            round_direction_p_up=(
                float(round_locked_p_up)
                if round_locked_p_up is not None and np.isfinite(round_locked_p_up)
                else None
            ),
            round_direction_reason=round_locked_reason,
        )
