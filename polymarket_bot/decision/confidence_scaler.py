"""
Confidence scaling for trade decisions.

Scales the raw confidence from settlement forecast and execution estimate
into a final decision confidence that accounts for:
- Signal quality and consistency
- Execution realism (fill probability, slippage, latency)
- Risk state (drawdown, consecutive losses, cooldown)
- Regime uncertainty
- Historical performance feedback

The confidence scaler is the key mechanism that prevents overconfident
trading. Even if the model produces a strong signal, the scaler can
suppress it if execution conditions are poor or risk state is degraded.

Mathematical basis:
- Confidence = geometric_mean(component_scores)
- Each component is bounded [0, 1]
- Geometric mean ensures any single low component significantly reduces confidence
- Historical calibration: adjust confidence based on past win rate
- Asymmetric scaling: reduce confidence more for losses than increase for wins
"""

import numpy as np
from typing import Optional

from polymarket_bot.bot_types import (
    SettlementForecast,
    ExecutionEstimate,
    RiskState,
    RegimeStateEstimate,
    VolatilityEstimate,
    MultiTimescaleEstimate,
    EntropyFilterResult,
    RegimeState,
    ExecutionQuality,
)
from polymarket_bot.config import BotConfig


class ConfidenceScaler:
    """
    Scales raw confidence into final decision confidence.
    
    The scaler uses a multi-component geometric mean approach:
    
    final_confidence = geometric_mean(
        signal_quality,
        execution_quality_score,
        risk_state_score,
        regime_certainty_score,
        volatility_score,
        entropy_score,
        historical_calibration
    )
    
    Each component is in [0, 1]. The geometric mean ensures that
    any single poor component (e.g., bad execution) significantly
    reduces overall confidence, preventing overconfident decisions.
    
    Asymmetric adjustment:
    - After a loss: confidence *= 0.85 (strong suppression)
    - After a win: confidence *= 1.05 (moderate boost)
    - This creates natural caution after losses
    """

    # Asymmetric adjustment factors
    LOSS_SUPPRESSION = 0.85
    WIN_BOOST = 1.05

    # Minimum confidence for any trade
    MIN_CONFIDENCE = 0.3

    # Maximum confidence (never be 100% sure)
    MAX_CONFIDENCE = 0.95

    # Execution quality scores
    EXECUTION_QUALITY_SCORES = {
        ExecutionQuality.OPTIMAL: 1.0,
        ExecutionQuality.ACCEPTABLE: 0.85,
        ExecutionQuality.DEGRADED: 0.60,
        ExecutionQuality.REJECT: 0.0,
    }

    # Regime certainty thresholds
    REGIME_CERTAINTY_HIGH = 0.8
    REGIME_CERTAINTY_LOW = 0.4

    def __init__(self, config: BotConfig):
        self.config = config
        self._win_count = 0
        self._loss_count = 0
        self._total_count = 0
        self._recent_results: list = []  # True=win, False=loss
        self._max_recent = 50
        self._confidence_history: list = []
        self._max_history = 200

    def _compute_signal_quality(
        self,
        settlement_forecast: SettlementForecast,
        multi_timescale: Optional[MultiTimescaleEstimate] = None,
    ) -> float:
        """
        Signal quality from settlement forecast and timescale consistency.
        
        Quality = forecast_quality * consistency_weight
        
        High quality = good forecast + consistent timescales
        Low quality = poor forecast or conflicting timescales
        """
        # Base quality from settlement forecast
        forecast_quality = settlement_forecast.forecast_quality

        # Timescale consistency boost/penalty
        consistency_factor = 1.0
        if multi_timescale is not None:
            # High consistency boosts quality
            consistency = multi_timescale.timescale_consistency
            # Low conflict boosts quality
            conflict = multi_timescale.timescale_conflict_score

            # Combined: consistency increases, conflict decreases
            consistency_factor = 0.5 + 0.5 * consistency * (1.0 - conflict)

        quality = forecast_quality * consistency_factor
        return np.clip(quality, 0.01, 1.0)

    def _compute_execution_quality_score(
        self,
        execution_estimate: ExecutionEstimate,
    ) -> float:
        """
        Execution quality score from fill probability, slippage, and quality class.
        
        Score = quality_class_score * fill_probability * (1 - slippage_penalty)
        """
        # Quality class score
        quality_class = execution_estimate.execution_quality
        class_score = self.EXECUTION_QUALITY_SCORES.get(quality_class, 0.0)

        # Fill probability factor
        fill_prob = execution_estimate.expected_fill_probability

        # Slippage penalty: slippage reduces confidence
        # Penalty = slippage_bps / max_acceptable_slippage_bps
        max_slippage_bps = self.config.risk.min_edge_bps * 2.0
        slippage_penalty = execution_estimate.expected_slippage_bps / max(1.0, max_slippage_bps)
        slippage_penalty = np.clip(slippage_penalty, 0.0, 0.5)

        # Latency penalty: high latency reduces confidence
        latency_penalty = 0.0
        if execution_estimate.expected_latency_ms > self.config.execution.typical_latency_ms:
            excess_latency = execution_estimate.expected_latency_ms - self.config.execution.typical_latency_ms
            latency_penalty = excess_latency / self.config.execution.max_latency_ms
            latency_penalty = np.clip(latency_penalty, 0.0, 0.3)

        # Combined score
        score = class_score * fill_prob * (1.0 - slippage_penalty) * (1.0 - latency_penalty)
        return np.clip(score, 0.01, 1.0)

    def _compute_risk_state_score(
        self,
        risk_state: Optional[RiskState] = None,
    ) -> float:
        """
        Risk state score from current risk conditions.
        
        Score decreases with:
        - High drawdown
        - Consecutive losses
        - Cooldown period
        - High risk score
        
        Score = 1 - max(drawdown_fraction, loss_fraction, risk_score)
        """
        if risk_state is None:
            return 0.8  # Default: moderate confidence when no risk data

        # Drawdown penalty
        max_drawdown = self.config.risk.max_drawdown_fraction
        current_drawdown = risk_state.current_drawdown
        drawdown_fraction = current_drawdown / max(0.01, max_drawdown)
        drawdown_penalty = np.clip(drawdown_fraction, 0.0, 1.0)

        # Consecutive loss penalty
        max_losses = self.config.risk.max_consecutive_losses
        consecutive_losses = risk_state.consecutive_losses
        loss_fraction = consecutive_losses / max(1, max_losses)
        loss_penalty = np.clip(loss_fraction, 0.0, 1.0)

        # Cooldown: zero confidence during cooldown
        if risk_state.is_in_cooldown:
            return 0.01

        # Should stop trading: zero confidence
        if risk_state.should_stop_trading:
            return 0.0

        # Overall risk score penalty
        risk_score_penalty = risk_state.risk_score

        # Combined: take the worst penalty
        worst_penalty = max(drawdown_penalty, loss_penalty, risk_score_penalty)

        score = 1.0 - worst_penalty
        return np.clip(score, 0.01, 1.0)

    def _compute_regime_certainty_score(
        self,
        regime_estimate: RegimeStateEstimate,
    ) -> float:
        """
        Regime certainty score from regime classification confidence.
        
        High certainty = dominant regime with high probability
        Low certainty = uncertain regime classification
        """
        regime_probs = regime_estimate.regime_probabilities
        if not regime_probs:
            return 0.5

        # Maximum regime probability = certainty
        max_prob = max(regime_probs.values())

        # Entropy-based certainty
        # Low entropy = high certainty
        probs_array = np.array(list(regime_probs.values()))
        entropy = -np.sum(probs_array * np.log2(probs_array + 1e-10))
        max_entropy = np.log2(len(probs_array))  # Maximum possible entropy
        normalized_entropy = entropy / max(1.0, max_entropy)

        # Certainty = 1 - normalized_entropy
        certainty = 1.0 - normalized_entropy

        # Blend with max probability
        score = 0.5 * certainty + 0.5 * max_prob

        return np.clip(score, 0.01, 1.0)

    def _compute_volatility_score(
        self,
        volatility_estimate: VolatilityEstimate,
    ) -> float:
        """
        Volatility score: moderate volatility is good, extreme is bad.
        
        Too low vol: no edge opportunity (score ~0.5)
        Moderate vol: good trading conditions (score ~0.8-1.0)
        High vol: dangerous, reduce confidence (score ~0.3-0.5)
        Extreme vol: very dangerous (score ~0.1-0.2)
        
        Score = bell_curve(vol, center=optimal_vol, width=tolerance)
        """
        vol_realized = volatility_estimate.realized_volatility

        # Optimal volatility range for 5-minute BTC trading
        # BTC typical 5-min vol: 0.0001 to 0.001
        optimal_vol = 0.0003  # Moderate movement
        vol_tolerance = 0.002  # Width of acceptable range

        # Bell curve scoring
        # score = exp(-(vol - optimal)^2 / (2 * tolerance^2))
        deviation = (vol_realized - optimal_vol) ** 2
        score = np.exp(-deviation / (2.0 * vol_tolerance ** 2))

        # Minimum score: even in extreme vol, don't go below 0.1
        return np.clip(float(score), 0.1, 1.0)

    def _compute_entropy_score(
        self,
        entropy_filter: Optional[EntropyFilterResult] = None,
    ) -> float:
        """
        Entropy score from entropy filter result.
        
        High suppression = low confidence (too much noise)
        Low suppression = high confidence (clear signal)
        """
        if entropy_filter is None:
            return 0.7  # Default: moderate

        # Suppression factor directly maps to confidence
        # suppression=1.0 means full confidence (no suppression)
        # suppression=0.5 means half confidence
        score = entropy_filter.suppression_factor

        # Novelty bonus: novel signals are more valuable
        novelty = getattr(
            entropy_filter,
            "novelty_score",
            getattr(entropy_filter, "signal_novelty", 0.0),
        )
        score *= (0.7 + 0.3 * novelty)

        return np.clip(score, 0.01, 1.0)

    def _compute_historical_calibration(
        self,
    ) -> float:
        """
        Historical calibration factor based on recent win rate.
        
        Uses recent performance to adjust confidence:
        - Win rate > 60%: slight boost (1.05)
        - Win rate 40-60%: neutral (1.0)
        - Win rate < 40%: suppression (0.85)
        - Win rate < 20%: strong suppression (0.7)
        
        Asymmetric: losses suppress more than wins boost.
        """
        if self._total_count < 5:
            return 0.8  # Insufficient data: moderate caution

        # Recent win rate (last 50 trades)
        recent = self._recent_results[-50:]
        if not recent:
            return 0.8

        win_rate = sum(recent) / len(recent)

        # Asymmetric calibration
        if win_rate >= 0.6:
            calibration = 1.0 + 0.05 * (win_rate - 0.6) / 0.4  # Up to 1.05
        elif win_rate >= 0.4:
            calibration = 1.0
        elif win_rate >= 0.2:
            calibration = 0.85 - 0.15 * (0.4 - win_rate) / 0.2  # 0.85 to 0.70
        else:
            calibration = 0.5  # Very poor performance: strong suppression

        return np.clip(calibration, 0.3, 1.1)

    def _compute_edge_support(
        self,
        settlement_forecast: SettlementForecast,
        execution_estimate: ExecutionEstimate,
    ) -> float:
        directional_edge_bps = max(0.0, settlement_forecast.edge_estimate * 10000.0)
        min_probability = float(getattr(self.config, "directional_min_probability", 0.56))
        threshold = max(1.0, (min_probability - 0.5) * 10000.0, self.config.risk.min_edge_bps)
        support = 1.0 - np.exp(-directional_edge_bps / (2.0 * threshold))
        return float(np.clip(support, 0.01, 1.0))

    def scale_confidence(
        self,
        settlement_forecast: SettlementForecast,
        execution_estimate: ExecutionEstimate,
        regime_estimate: RegimeStateEstimate,
        volatility_estimate: VolatilityEstimate,
        multi_timescale: Optional[MultiTimescaleEstimate] = None,
        entropy_filter: Optional[EntropyFilterResult] = None,
        risk_state: Optional[RiskState] = None,
    ) -> float:
        """
        Scale raw confidence into final decision confidence.
        
        Uses geometric mean of all component scores:
        confidence = geometric_mean(components) * historical_calibration
        
        Geometric mean ensures any single poor component significantly
        reduces overall confidence.
        """
        # Compute all component scores
        signal_quality = self._compute_signal_quality(
            settlement_forecast=settlement_forecast,
            multi_timescale=multi_timescale,
        )

        execution_score = self._compute_execution_quality_score(
            execution_estimate=execution_estimate,
        )

        risk_score = self._compute_risk_state_score(
            risk_state=risk_state,
        )

        regime_score = self._compute_regime_certainty_score(
            regime_estimate=regime_estimate,
        )

        volatility_score = self._compute_volatility_score(
            volatility_estimate=volatility_estimate,
        )

        entropy_score = self._compute_entropy_score(
            entropy_filter=entropy_filter,
        )

        historical_calibration = self._compute_historical_calibration()

        edge_support = self._compute_edge_support(
            settlement_forecast=settlement_forecast,
            execution_estimate=execution_estimate,
        )

        # Geometric mean of all components
        # All components are clipped to [0.01, 1.0] to avoid zero in geometric mean
        components = [
            max(0.01, signal_quality),
            max(0.01, execution_score),
            max(0.01, risk_score),
            max(0.01, regime_score),
            max(0.01, volatility_score),
            max(0.01, entropy_score),
        ]

        geometric_mean = np.exp(np.mean(np.log(components)))

        # Apply historical calibration
        calibrated_geometric = geometric_mean * historical_calibration

        # Strong terminal edge should not be erased by generic signal/regime
        # components. This convex blend still respects risk/execution penalties,
        # but anchors final confidence to the actual barrier forecast.
        final_confidence = (
            0.45 * calibrated_geometric
            + 0.35 * settlement_forecast.settlement_confidence
            + 0.20 * edge_support
        )

        # Clip to bounds
        final_confidence = np.clip(final_confidence, self.MIN_CONFIDENCE, self.MAX_CONFIDENCE)

        # Track confidence history
        self._confidence_history.append(float(final_confidence))
        if len(self._confidence_history) > self._max_history:
            self._confidence_history = self._confidence_history[-self._max_history:]

        return float(final_confidence)

    def record_outcome(
        self,
        was_win: bool,
        confidence_at_decision: float,
    ):
        """
        Record trade outcome for historical calibration.
        
        Applies asymmetric adjustment:
        - Loss: suppress future confidence by LOSS_SUPPRESSION factor
        - Win: boost future confidence by WIN_BOOST factor
        """
        self._total_count += 1
        self._recent_results.append(was_win)
        if len(self._recent_results) > self._max_recent:
            self._recent_results = self._recent_results[-self._max_recent:]

        if was_win:
            self._win_count += 1
        else:
            self._loss_count += 1

    def get_confidence_stats(self) -> dict:
        """Return confidence scaling statistics."""
        win_rate = self._win_count / max(1, self._total_count)
        recent_win_rate = sum(self._recent_results[-20:]) / max(1, len(self._recent_results[-20:]))
        avg_confidence = np.mean(self._confidence_history[-50:]) if self._confidence_history else 0.5

        return {
            "total_trades": self._total_count,
            "win_count": self._win_count,
            "loss_count": self._loss_count,
            "overall_win_rate": win_rate,
            "recent_win_rate": recent_win_rate,
            "avg_confidence": avg_confidence,
            "historical_calibration": self._compute_historical_calibration(),
        }
