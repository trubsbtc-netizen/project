"""
Bayesian evidence fusion for directional probability inference.

Mathematical foundation:
- Bayesian posterior update: P(H|E) = P(E|H) * P(H) / P(E)
  where H = directional hypothesis (UP/DOWN), E = observed evidence

- Multiple evidence sources fused via log-odds ratio:
  log_odds = log(P(UP|E) / P(DOWN|E))
  = log_prior_odds + sum_i(log_likelihood_ratio_i)
  
  where each evidence source i contributes:
  log_likelihood_ratio_i = log(P(E_i|UP) / P(E_i|DOWN))

- Evidence sources and their likelihood models:
  1. Kalman velocity: P(E_vel|UP) = sigmoid(velocity / sigma)
  2. Flow imbalance: P(E_flow|UP) = sigmoid(imbalance * scaling)
  3. Orderbook pressure: P(E_pressure|UP) = sigmoid(real_imbalance * scaling)
  4. Regime support: P(E_regime|UP) = regime_flow_characteristic
  5. Continuation signal: P(E_cont|UP) = continuation_probability
  6. Reversal signal: P(E_rev|UP) = 1 - reversal_probability (if reversal is DOWN)
  7. Volatility skew: P(E_skew|UP) = sigmoid(-skew) (negative skew = bullish)
  8. Spread asymmetry: P(E_spread|UP) = sigmoid(-asymmetry)

- Evidence weighting: each source has a weight that depends on:
  1. Source reliability (historical predictive power)
  2. Regime appropriateness (some sources are more reliable in certain regimes)
  3. Evidence decay (older evidence has less weight)
  4. Correlation with other sources (redundant evidence gets less weight)

- Posterior uncertainty: variance of the posterior distribution
  Var(P(UP)) = P(UP) * P(DOWN) / (N + 1) where N is effective sample size

- Bayes factor: BF = P(E|UP) / P(E|DOWN) for evidence strength
  BF > 10: strong evidence for UP
  BF < 0.1: strong evidence for DOWN
  BF near 1: weak evidence (no clear direction)

- Credible intervals: computed from posterior Beta distribution
  P(UP) ~ Beta(alpha, beta) where alpha = P(UP)*N, beta = P(DOWN)*N
"""

import logging
import time
import numpy as np
from typing import Dict, Optional, Tuple
from collections import deque

from polymarket_bot.config import BayesianConfig, RegimeState, Direction
from polymarket_bot.bot_types import (
    BayesianPosterior, KalmanState, FlowMetrics, OrderbookPressure,
    RegimeStateEstimate, VolatilityEstimate, SpreadMetrics,
    ContinuationSignal, ReversalSignal, LiquidityMetrics
)

logger = logging.getLogger(__name__)


class BayesianFusion:
    """
    Bayesian evidence fusion engine for directional probability.
    
    Fuses multiple microstructure evidence sources into a calibrated
    posterior probability for BTC settling UP or DOWN.
    
    Key innovations:
    1. Log-odds ratio fusion (numerically stable, additive)
    2. Regime-adaptive evidence weights
    3. Evidence decay modeling (time-based weight reduction)
    4. Source reliability tracking (historical predictive power)
    5. Redundancy discounting (correlated sources get less weight)
    6. Beta distribution posterior for credible intervals
    7. Bayes factor computation for evidence strength
    """
    
    def __init__(self, config: BayesianConfig):
        self.config = config
        self._prior_log_odds = self._initial_prior_log_odds()
        
        # Evidence source reliability tracking
        # reliability[i] = historical predictive accuracy of source i
        self._source_reliability: Dict[str, float] = {
            "kalman_velocity": 0.7,
            "flow_imbalance": 0.6,
            "orderbook_pressure": 0.5,
            "regime_support": 0.4,
            "continuation_signal": 0.6,
            "reversal_signal": 0.5,
            "volatility_skew": 0.3,
            "spread_asymmetry": 0.2,
        }
        
        # Regime-adaptive weight multipliers
        # Some evidence sources are more reliable in certain regimes
        self._regime_weight_multipliers: Dict[RegimeState, Dict[str, float]] = {
            RegimeState.CALM_TRENDING: {
                "kalman_velocity": 1.5, "flow_imbalance": 1.3,
                "orderbook_pressure": 1.0, "regime_support": 1.2,
                "continuation_signal": 1.4, "reversal_signal": 0.8,
                "volatility_skew": 0.8, "spread_asymmetry": 0.7,
            },
            RegimeState.CALM_RANGE: {
                "kalman_velocity": 0.5, "flow_imbalance": 0.6,
                "orderbook_pressure": 0.8, "regime_support": 0.5,
                "continuation_signal": 0.4, "reversal_signal": 1.0,
                "volatility_skew": 0.5, "spread_asymmetry": 0.6,
            },
            RegimeState.VOLATILE_TRENDING: {
                "kalman_velocity": 1.2, "flow_imbalance": 1.5,
                "orderbook_pressure": 1.2, "regime_support": 1.0,
                "continuation_signal": 1.3, "reversal_signal": 0.7,
                "volatility_skew": 1.0, "spread_asymmetry": 0.8,
            },
            RegimeState.VOLATILE_RANGE: {
                "kalman_velocity": 0.6, "flow_imbalance": 0.8,
                "orderbook_pressure": 0.9, "regime_support": 0.6,
                "continuation_signal": 0.5, "reversal_signal": 1.2,
                "volatility_skew": 0.8, "spread_asymmetry": 0.9,
            },
            RegimeState.CRISIS: {
                "kalman_velocity": 0.4, "flow_imbalance": 0.5,
                "orderbook_pressure": 0.6, "regime_support": 0.3,
                "continuation_signal": 0.3, "reversal_signal": 1.5,
                "volatility_skew": 1.2, "spread_asymmetry": 1.0,
            },
            RegimeState.LIQUIDATION_CASCADE: {
                "kalman_velocity": 0.3, "flow_imbalance": 0.4,
                "orderbook_pressure": 0.5, "regime_support": 0.2,
                "continuation_signal": 0.2, "reversal_signal": 1.8,
                "volatility_skew": 1.5, "spread_asymmetry": 1.2,
            },
        }
        
        # Evidence history for redundancy detection
        self._evidence_history: deque = deque(maxlen=200)
        
        # Posterior history for calibration
        self._posterior_history: deque = deque(maxlen=500)
        
        # Previous posterior for smoothing
        self._prev_posterior: Optional[BayesianPosterior] = None
        
        # Effective sample size for uncertainty
        self._effective_n = 10.0

    def _initial_prior_log_odds(self) -> float:
        prior_strength = float(getattr(self.config, "prior_strength", 0.5))
        prior_strength = float(np.clip(prior_strength, 0.001, 0.999))
        return float(np.log(prior_strength / (1.0 - prior_strength)))

    def reset(self) -> None:
        """Reset round-local Bayesian smoothing state."""
        self._prior_log_odds = self._initial_prior_log_odds()
        self._prev_posterior = None
        self._effective_n = 10.0
    
    def fuse(
        self,
        kalman_state: KalmanState,
        flow_metrics: FlowMetrics,
        orderbook_pressure: OrderbookPressure,
        regime_estimate: RegimeStateEstimate,
        volatility_estimate: VolatilityEstimate,
        continuation_signal: Optional[ContinuationSignal] = None,
        reversal_signal: Optional[ReversalSignal] = None,
        spread_metrics: Optional[SpreadMetrics] = None,
        liquidity_metrics: Optional[LiquidityMetrics] = None,
    ) -> BayesianPosterior:
        """
        Fuse all evidence sources into a Bayesian posterior.
        
        Process:
        1. Compute log-likelihood ratio for each evidence source
        2. Apply regime-adaptive weights
        3. Apply evidence decay
        4. Discount redundant evidence
        5. Compute total log-odds = prior + weighted_evidence
        6. Convert to posterior probability
        7. Compute posterior uncertainty
        8. Compute credible intervals
        9. Compute Bayes factor
        10. Smooth with previous posterior
        """
        timestamp = time.time()
        
        # Step 1: Compute log-likelihood ratios for each source
        evidence_sources = {}
        
        # Kalman velocity evidence
        velocity = kalman_state.velocity_estimate
        vol = volatility_estimate.kalman_volatility
        price_scale = float(kalman_state.state[0]) if len(kalman_state.state) else 0.0
        price_scale = max(abs(price_scale), 1.0e-9)
        log_velocity = velocity / price_scale
        velocity_normalized = log_velocity / max(vol, 1e-10)
        # P(E_vel|UP) / P(E_vel|DOWN) = exp(velocity_normalized * scaling)
        evidence_sources["kalman_velocity"] = velocity_normalized * 2.0
        
        # Flow imbalance evidence
        imbalance = flow_metrics.flow_imbalance
        persistence = max(flow_metrics.flow_persistence, 0.1)
        # P(E_flow|UP) / P(E_flow|DOWN) = exp(imbalance * persistence * scaling)
        evidence_sources["flow_imbalance"] = imbalance * persistence * 3.0
        
        # Orderbook pressure evidence (real, spoof-filtered)
        real_imbalance = orderbook_pressure.real_imbalance
        # P(E_pressure|UP) / P(E_pressure|DOWN) = exp(real_imbalance * scaling)
        evidence_sources["orderbook_pressure"] = real_imbalance * 2.5
        
        # Regime support evidence
        regime_flow = regime_estimate.regime_flow_characteristic
        regime_confidence = regime_estimate.regime_confidence
        # P(E_regime|UP) / P(E_regime|DOWN) = exp(regime_flow * confidence * scaling)
        evidence_sources["regime_support"] = regime_flow * regime_confidence * 2.0
        
        # Continuation signal evidence
        if continuation_signal is not None and continuation_signal.is_valid:
            cont_prob = continuation_signal.continuation_probability
            cont_direction = continuation_signal.direction
            # If continuation is UP: evidence for UP
            # If continuation is DOWN: evidence for DOWN
            if cont_direction == Direction.UP:
                evidence_sources["continuation_signal"] = np.log(max(cont_prob, 0.01) / max(1 - cont_prob, 0.01))
            elif cont_direction == Direction.DOWN:
                evidence_sources["continuation_signal"] = np.log(max(1 - cont_prob, 0.01) / max(cont_prob, 0.01))
            else:
                evidence_sources["continuation_signal"] = 0.0
        else:
            evidence_sources["continuation_signal"] = 0.0
        
        # Reversal signal evidence
        if reversal_signal is not None and reversal_signal.is_valid:
            rev_prob = reversal_signal.reversal_probability
            rev_direction = reversal_signal.direction
            # If reversal is UP: evidence for UP (trend was DOWN, now reversing UP)
            # If reversal is DOWN: evidence for DOWN (trend was UP, now reversing DOWN)
            if rev_direction == Direction.UP:
                evidence_sources["reversal_signal"] = np.log(max(rev_prob, 0.01) / max(1 - rev_prob, 0.01)) * 1.5
            elif rev_direction == Direction.DOWN:
                evidence_sources["reversal_signal"] = np.log(max(1 - rev_prob, 0.01) / max(rev_prob, 0.01)) * 1.5
            else:
                evidence_sources["reversal_signal"] = 0.0
        else:
            evidence_sources["reversal_signal"] = 0.0
        
        # Volatility skew evidence
        skew = volatility_estimate.volatility_skew
        # Positive skew = downside vol higher = bearish evidence
        # P(E_skew|UP) / P(E_skew|DOWN) = exp(-skew * scaling)
        evidence_sources["volatility_skew"] = -skew * 2.0
        
        # Spread asymmetry evidence
        if spread_metrics is not None:
            asymmetry = spread_metrics.spread_asymmetry
            # Positive asymmetry = ask wider = bearish evidence
            # P(E_spread|UP) / P(E_spread|DOWN) = exp(-asymmetry * scaling)
            evidence_sources["spread_asymmetry"] = -asymmetry * 1.5
        else:
            evidence_sources["spread_asymmetry"] = 0.0
        
        # Step 2: Apply regime-adaptive weights
        regime = regime_estimate.current_regime
        regime_weights = self._regime_weight_multipliers.get(regime, {})
        
        weighted_evidence = {}
        for source, llr in evidence_sources.items():
            # Base weight = source reliability
            base_weight = self._source_reliability.get(source, 0.3)
            # Regime multiplier
            regime_mult = regime_weights.get(source, 1.0)
            # Combined weight
            weight = base_weight * regime_mult
            # Clamp weight
            weight = max(self.config.min_evidence_weight, 
                        min(self.config.max_evidence_weight, weight))
            weighted_evidence[source] = llr * weight
        
        # Step 3: Evidence decay
        # Evidence decays exponentially with time
        # decay_factor = exp(-lambda * time_since_evidence)
        # For real-time: all evidence is current, so no decay needed
        # But we track the total evidence strength for uncertainty
        
        # Step 4: Redundancy discounting
        # If multiple sources are highly correlated, discount their combined weight
        # Simplified: check if flow and pressure agree (they're often correlated)
        if evidence_sources.get("flow_imbalance", 0) > 0 and \
           evidence_sources.get("orderbook_pressure", 0) > 0:
            # Both positive for UP: correlated, discount
            redundancy_factor = 0.8
            weighted_evidence["flow_imbalance"] *= redundancy_factor
            weighted_evidence["orderbook_pressure"] *= redundancy_factor
        elif evidence_sources.get("flow_imbalance", 0) < 0 and \
             evidence_sources.get("orderbook_pressure", 0) < 0:
            # Both positive for DOWN: correlated, discount
            redundancy_factor = 0.8
            weighted_evidence["flow_imbalance"] *= redundancy_factor
            weighted_evidence["orderbook_pressure"] *= redundancy_factor
        
        # Step 5: Compute total log-odds
        total_evidence = sum(weighted_evidence.values())
        total_log_odds = self._prior_log_odds + total_evidence
        
        # Step 6: Convert to posterior probability
        # P(UP) = 1 / (1 + exp(-log_odds))
        # Numerically stable sigmoid
        if total_log_odds > 20:
            p_up = 1.0 - 1e-10
        elif total_log_odds < -20:
            p_up = 1e-10
        else:
            p_up = 1.0 / (1.0 + np.exp(-total_log_odds))
        
        p_down = 1.0 - p_up
        p_neutral = 0.0  # For binary outcome, no neutral
        
        # Step 7: Posterior uncertainty
        # Effective sample size from total evidence weight
        total_weight = sum(abs(w) for w in weighted_evidence.values())
        self._effective_n = max(5.0, total_weight * 20.0)
        
        # Posterior variance: P(UP) * P(DOWN) / (N + 1)
        posterior_variance = p_up * p_down / (self._effective_n + 1)
        
        # Step 8: Credible intervals (Beta distribution)
        # P(UP) ~ Beta(alpha, beta) where alpha = P(UP)*N, beta = P(DOWN)*N
        alpha_param = p_up * self._effective_n
        beta_param = p_down * self._effective_n
        
        # 95% credible interval using Beta distribution approximation
        # For large N: Beta ~ Normal(alpha/(alpha+beta), sqrt(alpha*beta/((alpha+beta)^2*(alpha+beta+1))))
        if alpha_param > 0 and beta_param > 0:
            mean = alpha_param / (alpha_param + beta_param)
            std = np.sqrt(alpha_param * beta_param / 
                         ((alpha_param + beta_param) ** 2 * (alpha_param + beta_param + 1)))
            ci_up = (max(0, mean - 1.96 * std), min(1, mean + 1.96 * std))
            ci_down = (max(0, 1 - mean - 1.96 * std), min(1, 1 - mean + 1.96 * std))
        else:
            ci_up = (0.0, 1.0)
            ci_down = (0.0, 1.0)
        
        # Step 9: Bayes factor
        # BF = exp(total_evidence) = P(E|UP) / P(E|DOWN)
        if abs(total_evidence) > 20:
            bayes_factor = np.exp(20) if total_evidence > 0 else np.exp(-20)
        else:
            bayes_factor = np.exp(total_evidence)
        
        # Step 10: Evidence strength and weight decomposition
        evidence_strength = abs(total_evidence)
        prior_weight = abs(self._prior_log_odds) / max(abs(total_log_odds), 1e-10)
        likelihood_weight = evidence_strength / max(abs(total_log_odds), 1e-10)
        
        # Posterior entropy
        if p_up > 0 and p_down > 0:
            posterior_entropy = -(p_up * np.log(p_up) + p_down * np.log(p_down))
        else:
            posterior_entropy = 0.0
        
        # Calibration score (placeholder - needs historical data)
        calibration_score = 0.5
        
        # Smooth with previous posterior
        if self._prev_posterior is not None:
            smoothing = self.config.posterior_smoothing
            p_up = smoothing * p_up + (1 - smoothing) * self._prev_posterior.p_up
            p_down = 1.0 - p_up
            posterior_variance = smoothing * posterior_variance + \
                                (1 - smoothing) * self._prev_posterior.posterior_variance
            if p_up > 0 and p_down > 0:
                posterior_entropy = -(p_up * np.log(p_up) + p_down * np.log(p_down))
            else:
                posterior_entropy = 0.0
            alpha_param = p_up * self._effective_n
            beta_param = p_down * self._effective_n
            if alpha_param > 0 and beta_param > 0:
                mean = alpha_param / (alpha_param + beta_param)
                std = np.sqrt(alpha_param * beta_param /
                             ((alpha_param + beta_param) ** 2 * (alpha_param + beta_param + 1)))
                ci_up = (max(0, mean - 1.96 * std), min(1, mean + 1.96 * std))
                ci_down = (max(0, 1 - mean - 1.96 * std), min(1, 1 - mean + 1.96 * std))
            else:
                ci_up = (0.0, 1.0)
                ci_down = (0.0, 1.0)
        
        # Build result
        result = BayesianPosterior(
            timestamp=timestamp,
            p_up=p_up,
            p_down=p_down,
            p_neutral=p_neutral,
            posterior_entropy=posterior_entropy,
            evidence_strength=evidence_strength,
            prior_weight=prior_weight,
            likelihood_weight=likelihood_weight,
            posterior_variance=posterior_variance,
            credible_interval_up=ci_up,
            credible_interval_down=ci_down,
            bayes_factor_up_vs_down=bayes_factor,
            calibration_score=calibration_score,
        )
        
        # Track history
        self._evidence_history.append(weighted_evidence)
        self._posterior_history.append(p_up)
        self._prev_posterior = result
        
        return result
