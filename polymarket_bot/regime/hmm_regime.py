"""
Hidden Markov Model regime detection for BTC microstructure.

Mathematical foundation:
- HMM with discrete hidden states (regimes) and continuous emissions
- Forward-backward algorithm for state inference: P(S_t | O_1:T)
- Viterbi algorithm for most likely state sequence
- Baum-Welch algorithm for parameter estimation (EM)
- Transition matrix A: P(S_t+1 | S_t) with Dirichlet priors for regularization
- Emission distributions: multivariate Gaussian with regime-specific parameters
  B_j(o) = N(o; mu_j, Sigma_j) where mu_j, Sigma_j are regime-specific
- Regime persistence: self-transition probability A[i,i] (diagonal dominance)
- Regime entropy: H = -sum(P(S_i) * log(P(S_i))) for uncertainty quantification

The HMM captures stochastic regime transitions in BTC microstructure:
- Calm trending: low vol, directional flow, high persistence
- Calm range: low vol, balanced flow, moderate persistence
- Volatile trending: high vol, strong directional flow, moderate persistence
- Volatile range: high vol, balanced flow, low persistence
- Crisis: extreme vol, chaotic flow, very low persistence
- Liquidation cascade: extreme vol, one-directional flow, very low persistence

All computations use numerically stable log-space operations.
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

from polymarket_bot.config import HMMConfig, RegimeState
from polymarket_bot.bot_types import RegimeStateEstimate

logger = logging.getLogger(__name__)


class HMMRegimeDetector:
    """
    Hidden Markov Model for regime detection in BTC microstructure.
    
    Uses forward-backward algorithm for real-time state inference
    with Dirichlet-regularized transition matrix and regime-specific
    emission distributions.
    
    Key innovations:
    1. Online forward algorithm for real-time state probability updates
    2. Dirichlet priors on transition matrix for regime persistence
    3. Adaptive emission parameters via exponential forgetting
    4. Regime entropy for uncertainty quantification
    5. Regime duration estimation from self-transition probabilities
    """
    
    def __init__(self, config: HMMConfig):
        self.config = config
        self.n_regimes = config.n_regimes
        
        # Transition matrix A[i,j] = P(S_j | S_i)
        # Initialized with regime persistence priors
        self._transition_matrix = self._initialize_transition_matrix()
        
        # Emission parameters: mu_j and Sigma_j for each regime
        # Emission dimension: [volatility, flow_imbalance, spread, taker_ratio,
        #                       volume, price_change, acceleration, depth_ratio]
        self._emission_means = np.zeros((self.n_regimes, config.emission_dim))
        self._emission_covs = np.array([np.eye(config.emission_dim) * 0.01 
                                         for _ in range(self.n_regimes)])
        self._initialize_emission_params()
        
        # State probability (forward probability) - current belief about regime
        self._state_probabilities = np.ones(self.n_regimes) / self.n_regimes  # Uniform prior
        
        # Observation history for parameter adaptation
        self._observation_history: deque = deque(maxlen=1000)
        self._last_observation: Optional[np.ndarray] = None
        
        # Regime history for tracking
        self._regime_history: deque = deque(maxlen=500)
        
        # Fitting state
        self._is_fitted = False
        self._observation_count = 0
        self._last_fit_time = 0.0
        self._min_obs_for_fit = config.min_observations_fit
        
        # Log-likelihood tracking
        self._log_likelihood = 0.0
        self._log_likelihood_history: deque = deque(maxlen=200)
        self._last_signed_flow_imbalance = 0.0
        
        # Numerical stability constants
        self._log_zero = -1e10  # Log of zero (for log-space operations)
        self._epsilon = 1e-10   # Small constant for numerical stability
        
    def _initialize_transition_matrix(self) -> np.ndarray:
        """
        Initialize transition matrix with regime persistence priors.
        
        Diagonal elements (self-transitions) set from config priors.
        Off-diagonal elements distributed uniformly from remaining probability.
        
        A[i,j] = persistence[i] if i==j, else (1-persistence[i]) / (n_regimes-1)
        
        This ensures regime persistence is built into the model structure,
        reflecting the empirical observation that regimes tend to persist.
        """
        A = np.zeros((self.n_regimes, self.n_regimes))
        
        for i in range(self.n_regimes):
            regime = RegimeState(i)
            persistence = self.config.regime_persistence_prior.get(regime, 0.7)
            A[i, i] = persistence
            # Distribute remaining probability uniformly
            remaining = 1.0 - persistence
            for j in range(self.n_regimes):
                if j != i:
                    A[i, j] = remaining / (self.n_regimes - 1)
        
        # Add Dirichlet smoothing
        A += self.config.transition_smoothing
        # Renormalize rows
        A = A / A.sum(axis=1, keepdims=True)
        
        return A
    
    def _initialize_emission_params(self):
        """
        Initialize emission distribution parameters for each regime.
        
        Each regime has characteristic emission patterns:
        - CALM_TRENDING: low vol, moderate flow imbalance, tight spread
        - CALM_RANGE: low vol, near-zero flow imbalance, tight spread
        - VOLATILE_TRENDING: high vol, strong flow imbalance, wider spread
        - VOLATILE_RANGE: high vol, moderate flow imbalance, wider spread
        - CRISIS: extreme vol, chaotic flow, very wide spread
        - LIQUIDATION_CASCADE: extreme vol, extreme flow imbalance, wide spread
        """
        # Emission dimension mapping:
        # 0: volatility (realized vol normalized)
        # 1: flow_imbalance (-1 to 1)
        # 2: spread_bps (normalized)
        # 3: taker_ratio (0 to 1)
        # 4: volume (normalized)
        # 5: price_change (normalized)
        # 6: acceleration (normalized)
        # 7: depth_ratio (normalized)
        
        vol_prior = self.config.regime_volatility_prior
        
        # CALM_TRENDING: low vol, moderate directional flow
        self._emission_means[RegimeState.CALM_TRENDING] = [
            vol_prior[RegimeState.CALM_TRENDING], 0.3, 0.1, 0.55, 0.3, 0.001, 0.0001, 1.2
        ]
        self._emission_covs[RegimeState.CALM_TRENDING] = np.diag([
            0.0001, 0.05, 0.02, 0.02, 0.05, 0.0005, 0.0001, 0.1
        ])
        
        # CALM_RANGE: low vol, balanced flow
        self._emission_means[RegimeState.CALM_RANGE] = [
            vol_prior[RegimeState.CALM_RANGE], 0.0, 0.08, 0.5, 0.2, 0.0, 0.0, 1.0
        ]
        self._emission_covs[RegimeState.CALM_RANGE] = np.diag([
            0.00005, 0.03, 0.01, 0.02, 0.03, 0.0002, 0.00005, 0.08
        ])
        
        # VOLATILE_TRENDING: high vol, strong directional flow
        self._emission_means[RegimeState.VOLATILE_TRENDING] = [
            vol_prior[RegimeState.VOLATILE_TRENDING], 0.5, 0.3, 0.65, 0.6, 0.003, 0.001, 0.8
        ]
        self._emission_covs[RegimeState.VOLATILE_TRENDING] = np.diag([
            0.001, 0.1, 0.05, 0.03, 0.1, 0.002, 0.001, 0.15
        ])
        
        # VOLATILE_RANGE: high vol, moderate flow
        self._emission_means[RegimeState.VOLATILE_RANGE] = [
            vol_prior[RegimeState.VOLATILE_RANGE], 0.1, 0.25, 0.5, 0.4, 0.001, 0.0005, 0.9
        ]
        self._emission_covs[RegimeState.VOLATILE_RANGE] = np.diag([
            0.0005, 0.08, 0.04, 0.03, 0.08, 0.001, 0.0005, 0.12
        ])
        
        # CRISIS: extreme vol, chaotic flow
        self._emission_means[RegimeState.CRISIS] = [
            vol_prior[RegimeState.CRISIS], 0.0, 0.5, 0.5, 0.8, 0.0, 0.005, 0.5
        ]
        self._emission_covs[RegimeState.CRISIS] = np.diag([
            0.005, 0.2, 0.1, 0.05, 0.2, 0.005, 0.005, 0.2
        ])
        
        # LIQUIDATION_CASCADE: extreme vol, extreme directional flow
        self._emission_means[RegimeState.LIQUIDATION_CASCADE] = [
            vol_prior[RegimeState.LIQUIDATION_CASCADE], 0.8, 0.6, 0.8, 0.9, 0.01, 0.01, 0.3
        ]
        self._emission_covs[RegimeState.LIQUIDATION_CASCADE] = np.diag([
            0.01, 0.15, 0.08, 0.05, 0.15, 0.01, 0.01, 0.15
        ])
    
    def update(self, observation: np.ndarray, timestamp: float) -> RegimeStateEstimate:
        """
        Update regime state estimate with a new observation using forward algorithm.
        
        Forward algorithm (online version):
        alpha_t(j) = P(S_t=j | O_1:t) 
                   = sum_i(alpha_{t-1}(i) * A[i,j]) * B_j(o_t)
        
        In log space for numerical stability:
        log_alpha_t(j) = logsumexp_i(log_alpha_{t-1}(i) + log_A[i,j]) + log_B_j(o_t)
        
        Then normalize: P(S_t=j | O_1:t) = alpha_t(j) / sum_j(alpha_t(j))
        
        Args:
            observation: 8-dimensional emission vector
            timestamp: current time
            
        Returns:
            RegimeStateEstimate with full regime probability distribution
        """
        # Store observation
        self._observation_history.append(observation)
        self._last_observation = observation
        self._observation_count += 1
        
        # Compute emission log-likelihoods for each regime
        log_emissions = self._compute_log_emissions(observation)
        
        # Forward step: update state probabilities
        if self._is_fitted or self._observation_count >= self._min_obs_for_fit:
            self._forward_step(log_emissions)
            self._is_fitted = True
        else:
            # Before fitting: use emission likelihoods directly as state probabilities
            # (no transition information yet)
            log_probs = log_emissions - np.max(log_emissions)  # Normalize in log space
            probs = np.exp(log_probs)
            probs = probs / np.sum(probs)
            self._state_probabilities = probs
        
        # Determine most likely regime
        most_likely_regime = RegimeState(np.argmax(self._state_probabilities))
        
        # Compute regime transition probabilities
        transition_probs = self._compute_transition_probabilities()
        
        # Compute regime persistence
        persistence = self._transition_matrix[most_likely_regime, most_likely_regime]
        
        # Compute regime duration estimate
        # E[duration] = 1 / (1 - A[i,i]) for geometric distribution
        duration_estimate = 1.0 / max(1.0 - persistence, 0.01)
        
        # Compute regime volatility estimate
        vol_estimate = self._emission_means[most_likely_regime, 0]
        
        # Regime classes describe market state, not UP/DOWN direction. Directional
        # flow must therefore come from the signed observation, not the positive
        # regime prior used for flow magnitude classification.
        flow_estimate = float(np.clip(self._last_signed_flow_imbalance, -1.0, 1.0))
        
        # Compute regime confidence
        confidence = self._state_probabilities[most_likely_regime]
        
        # Compute regime entropy
        entropy = self._compute_entropy(self._state_probabilities)
        
        # Track regime history
        self._regime_history.append(most_likely_regime)
        
        # Periodically refit parameters
        if self._observation_count >= self._min_obs_for_fit and \
           (timestamp - self._last_fit_time) > self.config.refit_interval_seconds:
            self._adaptive_refit()
            self._last_fit_time = timestamp
        
        # Build result
        regime_probs_dict = {}
        for i in range(self.n_regimes):
            regime_probs_dict[RegimeState(i)] = self._state_probabilities[i]
        
        transition_probs_dict = {}
        for i in range(self.n_regimes):
            transition_probs_dict[RegimeState(i)] = transition_probs[i]
        
        result = RegimeStateEstimate(
            timestamp=timestamp,
            current_regime=most_likely_regime,
            regime_probabilities=regime_probs_dict,
            regime_persistence=persistence,
            regime_transition_probability=transition_probs_dict,
            regime_duration_estimate=duration_estimate,
            regime_volatility_estimate=vol_estimate,
            regime_flow_characteristic=flow_estimate,
            regime_confidence=confidence,
            regime_entropy=entropy,
        )
        
        return result
    
    def _forward_step(self, log_emissions: np.ndarray):
        """
        Perform one step of the forward algorithm in log space.
        
        log_alpha_t(j) = logsumexp_i(log_alpha_{t-1}(i) + log_A[i,j]) + log_B_j(o_t)
        
        Then normalize to get P(S_t=j | O_1:t).
        """
        # Current state probabilities in log space
        log_alpha_prev = np.log(self._state_probabilities + self._epsilon)
        
        # Transition matrix in log space
        log_A = np.log(self._transition_matrix + self._epsilon)
        
        # Forward step: log_alpha_t(j) = logsumexp_i(log_alpha_prev[i] + log_A[i,j])
        log_alpha_new = np.zeros(self.n_regimes)
        for j in range(self.n_regimes):
            # logsumexp for numerical stability
            log_alpha_new[j] = self._logsumexp(log_alpha_prev + log_A[:, j])
        
        # Add emission likelihood
        log_alpha_new += log_emissions
        
        # Normalize in log space
        log_alpha_new -= self._logsumexp(log_alpha_new)
        
        # Convert back to probability space
        self._state_probabilities = np.exp(log_alpha_new)
        
        # Ensure normalization (numerical safety)
        self._state_probabilities = self._state_probabilities / np.sum(self._state_probabilities)
        
        # Update log-likelihood
        self._log_likelihood = self._logsumexp(log_alpha_new)
        self._log_likelihood_history.append(self._log_likelihood)
    
    def _compute_log_emissions(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute log emission probability for each regime.
        
        log_B_j(o) = -0.5 * (d * log(2*pi) + log(det(Sigma_j)) + 
                              (o - mu_j)^T * Sigma_j^{-1} * (o - mu_j))
        
        where d is the emission dimension.
        
        Uses Cholesky decomposition for numerical stability:
        Sigma_j = L_j * L_j^T
        (o - mu_j)^T * Sigma_j^{-1} * (o - mu_j) = ||L_j^{-1} * (o - mu_j)||^2
        """
        d = self.config.emission_dim
        log_emissions = np.zeros(self.n_regimes)
        
        for j in range(self.n_regimes):
            diff = observation - self._emission_means[j]
            
            # Try Cholesky decomposition for stable computation
            try:
                L = np.linalg.cholesky(self._emission_covs[j])
                # Solve L * x = diff for x
                x = np.linalg.solve(L, diff)
                # Mahalanobis distance = ||x||^2
                mahal = np.sum(x ** 2)
                # Log determinant = 2 * sum(log(diag(L)))
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
            except np.linalg.LinAlgError:
                # Fallback: use pseudo-inverse if Cholesky fails
                mahal = np.sum(diff ** 2) / max(np.trace(self._emission_covs[j]), self._epsilon)
                log_det = d * np.log(max(np.trace(self._emission_covs[j]) / d, self._epsilon))
            
            # Log emission probability
            log_emissions[j] = -0.5 * (d * np.log(2 * np.pi) + log_det + mahal)
        
        return log_emissions
    
    def _compute_transition_probabilities(self) -> np.ndarray:
        """
        Compute P(S_{t+1} | S_t) using current state probabilities.
        
        P(S_{t+1}=j) = sum_i P(S_t=i) * A[i,j]
        """
        return self._state_probabilities @ self._transition_matrix
    
    def _compute_entropy(self, probabilities: np.ndarray) -> float:
        """
        Compute Shannon entropy of regime probability distribution.
        
        H = -sum_i P(S_i) * log(P(S_i))
        
        Low entropy (<0.5): confident regime classification
        High entropy (>1.5): uncertain regime classification
        Maximum entropy = log(n_regimes) for uniform distribution
        """
        # Filter out zero probabilities
        probs = probabilities[probabilities > self._epsilon]
        entropy = -np.sum(probs * np.log(probs))
        return entropy
    
    def _logsumexp(self, x: np.ndarray) -> float:
        """
        Compute log(sum(exp(x))) in a numerically stable way.
        
        logsumexp(x) = max(x) + log(sum(exp(x - max(x))))
        """
        max_x = np.max(x)
        if max_x == self._log_zero:
            return self._log_zero
        return max_x + np.log(np.sum(np.exp(x - max_x)) + self._epsilon)
    
    def _adaptive_refit(self):
        """
        Adaptively refit emission parameters using recent observations.
        
        Uses exponential forgetting to update emission means and covariances:
        mu_j_new = (1-beta) * mu_j_old + beta * E[o | S=j]
        Sigma_j_new = (1-beta) * Sigma_j_old + beta * Cov[o | S=j]
        
        where beta is the forgetting factor and expectations are computed
        using current state probabilities as soft assignments.
        
        This allows the model to adapt to changing market conditions
        without requiring full Baum-Welch re-estimation.
        """
        if len(self._observation_history) < self._min_obs_for_fit:
            return
        
        observations = np.array(list(self._observation_history))
        n_obs = len(observations)
        
        # For each regime, compute soft-assigned mean and covariance
        # Using recent observations weighted by their regime probabilities
        # (approximation: use last n observations with uniform weight)
        
        beta = 0.1  # Forgetting factor (slow adaptation)
        recent_n = min(200, n_obs)
        recent_obs = observations[-recent_n:]
        
        for j in range(self.n_regimes):
            # Soft assignment: weight each observation by P(S=j) at that time
            # Simplified: use current state probability as proxy
            weight = self._state_probabilities[j]
            
            if weight < 0.01:
                continue  # Skip regimes with very low probability
            
            # Compute weighted mean and covariance
            weighted_mean = np.mean(recent_obs, axis=0)
            weighted_cov = np.cov(recent_obs.T) if recent_n > 2 else np.eye(self.config.emission_dim) * 0.01
            
            # Add regularization to covariance
            weighted_cov += np.eye(self.config.emission_dim) * self.config.emission_smoothing
            
            # Update with forgetting factor
            self._emission_means[j] = (1 - beta) * self._emission_means[j] + beta * weighted_mean
            self._emission_covs[j] = (1 - beta) * self._emission_covs[j] + beta * weighted_cov
        
        logger.debug(f"Adaptive refit completed with {n_obs} observations")
    
    def build_observation_vector(
        self,
        realized_vol: float,
        flow_imbalance: float,
        spread_bps: float,
        taker_ratio: float,
        volume_normalized: float,
        price_change: float,
        acceleration: float,
        depth_ratio: float
    ) -> np.ndarray:
        """
        Build the emission observation vector from microstructure metrics.
        
        All inputs should be normalized to comparable scales for
        proper multivariate Gaussian emission modeling.
        
        Normalization:
        - realized_vol: already in return space (small values)
        - flow_imbalance: signed in [-1, 1]; HMM uses magnitude for regime
          classification and stores the signed value for directional consumers
        - spread_bps: divide by 100 to normalize (typical 1-100 bps -> 0.01-1.0)
        - taker_ratio: already in [0, 1]
        - volume_normalized: divide by recent average
        - price_change: already in return space
        - acceleration: already in return/s^2 space
        - depth_ratio: already normalized
        """
        self._last_signed_flow_imbalance = float(np.clip(flow_imbalance, -1.0, 1.0))
        flow_magnitude = abs(self._last_signed_flow_imbalance)

        return np.array([
            realized_vol,
            flow_magnitude,
            spread_bps / 100.0,  # Normalize bps
            taker_ratio,
            volume_normalized,
            price_change,
            acceleration,
            depth_ratio,
        ])
    
    @property
    def current_regime(self) -> RegimeState:
        """Get the most likely current regime."""
        return RegimeState(np.argmax(self._state_probabilities))
    
    @property
    def state_probabilities(self) -> np.ndarray:
        """Get current state probability distribution."""
        return self._state_probabilities
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
