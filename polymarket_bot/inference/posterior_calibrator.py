"""
Posterior probability calibration for settlement accuracy.

Mathematical foundation:
- Calibration: mapping from raw posterior probability to calibrated probability
  such that when we predict P(UP)=0.7, UP actually occurs 70% of the time
- Platt scaling: calibrated_p = sigmoid(a * raw_p + b) where a,b fitted on history
- Isotonic regression: monotonic mapping from raw to calibrated probabilities
- Beta calibration: calibrated_p = Beta(alpha * raw_p + beta, gamma * (1-raw_p) + delta)
  more flexible than Platt, captures asymmetric calibration errors
- Calibration bins: divide predictions into bins, compute actual frequency in each
  calibration_error = |predicted_avg - actual_avg| per bin
- Expected Calibration Error (ECE): weighted average of calibration errors across bins
- Reliability diagram: plot of predicted vs actual probability per bin

Calibration is critical for:
1. Settlement probability accuracy (our core objective)
2. Kelly criterion position sizing (requires calibrated probabilities)
3. Edge estimation (requires accurate probability comparison with market)
"""

import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

from polymarket_bot.config import BayesianConfig
from polymarket_bot.bot_types import BayesianPosterior

logger = logging.getLogger(__name__)


class PosteriorCalibrator:
    """
    Calibrates posterior probabilities for settlement accuracy.
    
    Key innovations:
    1. Beta calibration model (flexible, asymmetric)
    2. Online calibration with exponential forgetting
    3. Expected Calibration Error (ECE) monitoring
    4. Reliability diagram tracking
    5. Adaptive calibration parameters
    """
    
    def __init__(self, config: BayesianConfig):
        self.config = config
        
        # Calibration parameters (Beta calibration)
        # calibrated_p = Beta(alpha * raw_p + beta, gamma * (1-raw_p) + delta)
        # Simplified: calibrated_p = a * raw_p^b / (a * raw_p^b + c * (1-raw_p)^d)
        self._cal_a = 1.0  # Initial: identity mapping
        self._cal_b = 1.0
        self._cal_c = 1.0
        self._cal_d = 1.0
        
        # Calibration history: (predicted_prob, actual_outcome)
        # outcome: 1 for UP, 0 for DOWN
        self._calibration_history: deque = deque(maxlen=config.calibration_window)
        
        # Bin-based calibration tracking
        self._n_bins = config.calibration_bins
        self._bin_predictions: Dict[int, List[float]] = {
            i: [] for i in range(self._n_bins)
        }
        self._bin_outcomes: Dict[int, List[float]] = {
            i: [] for i in range(self._n_bins)
        }
        
        # ECE tracking
        self._ece_history: deque = deque(maxlen=100)
        self._current_ece = 0.0
        
        # Last calibration fit time
        self._last_fit_time = 0.0
        self._fit_interval = 60.0  # Refit every 60 seconds
        
        # Previous calibrated posterior
        self._prev_calibrated: Optional[float] = None

    def reset(self) -> None:
        """Reset only short-lived smoothing state; keep calibration history."""
        self._prev_calibrated = None
    
    def calibrate(self, posterior: BayesianPosterior, 
                  timestamp: float) -> BayesianPosterior:
        """
        Calibrate the posterior probability.
        
        Process:
        1. Apply Beta calibration transformation
        2. Ensure probability stays in valid range
        3. Smooth with previous calibrated estimate
        4. Update calibration tracking
        """
        raw_p_up = posterior.p_up
        
        # Step 1: Apply Beta calibration
        calibrated_p_up = self._beta_calibration(raw_p_up)
        
        # Step 2: Ensure valid range
        calibrated_p_up = max(0.01, min(0.99, calibrated_p_up))
        calibrated_p_down = 1.0 - calibrated_p_up
        
        # Step 3: Smooth with previous
        if self._prev_calibrated is not None:
            smoothing = 0.05  # Very light smoothing to preserve calibration
            calibrated_p_up = smoothing * self._prev_calibrated + \
                              (1 - smoothing) * calibrated_p_up
            calibrated_p_up = max(0.01, min(0.99, calibrated_p_up))
            calibrated_p_down = 1.0 - calibrated_p_up
        
        self._prev_calibrated = calibrated_p_up
        
        # Step 4: Update calibration tracking
        # Store prediction for later outcome comparison
        bin_idx = int(raw_p_up * self._n_bins)
        bin_idx = min(bin_idx, self._n_bins - 1)
        self._bin_predictions[bin_idx].append(raw_p_up)
        
        # Periodically refit calibration parameters
        if timestamp - self._last_fit_time > self._fit_interval:
            self._refit_calibration()
            self._last_fit_time = timestamp
        
        # Compute calibration score
        calibration_score = self._compute_calibration_score(raw_p_up)
        
        # Update posterior with calibrated values
        # Recompute credible intervals with calibrated probability
        alpha_param = calibrated_p_up * posterior.evidence_strength * 10
        beta_param = calibrated_p_down * posterior.evidence_strength * 10
        
        if alpha_param > 0 and beta_param > 0:
            mean = alpha_param / (alpha_param + beta_param)
            std = np.sqrt(alpha_param * beta_param / 
                         ((alpha_param + beta_param) ** 2 * (alpha_param + beta_param + 1)))
            ci_up = (max(0, mean - 1.96 * std), min(1, mean + 1.96 * std))
        else:
            ci_up = (0.0, 1.0)
        
        # Posterior entropy with calibrated values
        if calibrated_p_up > 0 and calibrated_p_down > 0:
            entropy = -(calibrated_p_up * np.log(calibrated_p_up) + 
                        calibrated_p_down * np.log(calibrated_p_down))
        else:
            entropy = 0.0
        
        # Return calibrated posterior
        return BayesianPosterior(
            timestamp=posterior.timestamp,
            p_up=calibrated_p_up,
            p_down=calibrated_p_down,
            p_neutral=posterior.p_neutral,
            posterior_entropy=entropy,
            evidence_strength=posterior.evidence_strength,
            prior_weight=posterior.prior_weight,
            likelihood_weight=posterior.likelihood_weight,
            posterior_variance=calibrated_p_up * calibrated_p_down / 
                              max(posterior.evidence_strength * 10 + 1, 1),
            credible_interval_up=ci_up,
            credible_interval_down=(max(0, 1 - ci_up[1]), min(1, 1 - ci_up[0])),
            bayes_factor_up_vs_down=posterior.bayes_factor_up_vs_down,
            calibration_score=calibration_score,
        )
    
    def record_outcome(self, predicted_prob: float, actual_outcome: float):
        """
        Record actual outcome for calibration fitting.
        
        Args:
            predicted_prob: the raw predicted P(UP) before calibration
            actual_outcome: 1.0 if UP occurred, 0.0 if DOWN occurred
        """
        self._calibration_history.append((predicted_prob, actual_outcome))
        
        # Update bin tracking
        bin_idx = int(predicted_prob * self._n_bins)
        bin_idx = min(bin_idx, self._n_bins - 1)
        self._bin_outcomes[bin_idx].append(actual_outcome)
    
    def _beta_calibration(self, raw_p: float) -> float:
        """
        Apply Beta calibration transformation.
        
        calibrated_p = a * raw_p^b / (a * raw_p^b + c * (1-raw_p)^d)
        
        When a=b=c=d=1: identity mapping (no calibration)
        When b>1: stretches high probabilities higher (overconfident correction)
        When d>1: stretches low probabilities lower (underconfident correction)
        """
        if raw_p <= 0:
            return 0.01
        if raw_p >= 1:
            return 0.99
        
        numerator = self._cal_a * (raw_p ** self._cal_b)
        denominator = numerator + self._cal_c * ((1 - raw_p) ** self._cal_d)
        
        if denominator <= 0:
            return 0.5  # Fallback to neutral
        
        calibrated = numerator / denominator
        return max(0.01, min(0.99, calibrated))
    
    def _refit_calibration(self):
        """
        Refit calibration parameters from recent history.
        
        Uses bin-based calibration to estimate the mapping function.
        For each bin, compute the actual frequency of UP outcomes.
        Then fit Beta calibration parameters to match the observed mapping.
        
        Fitting method: minimize ECE across bins using gradient-free optimization.
        Simplified: use bin averages to directly estimate calibration curve,
        then fit Beta parameters to this curve.
        """
        if len(self._calibration_history) < 20:
            return  # Not enough data to refit
        
        # Compute bin averages
        bin_predicted = []
        bin_actual = []
        
        for i in range(self._n_bins):
            if len(self._bin_outcomes[i]) >= 3:
                avg_predicted = np.mean(self._bin_predictions[i][-50:])
                avg_actual = np.mean(self._bin_outcomes[i][-50:])
                bin_predicted.append(avg_predicted)
                bin_actual.append(avg_actual)
        
        if len(bin_predicted) < 3:
            return  # Not enough bins with data
        
        bin_predicted = np.array(bin_predicted)
        bin_actual = np.array(bin_actual)
        
        # Fit Beta calibration parameters
        # Simple approach: find a,b,c,d that minimize MSE between
        # beta_calibration(predicted) and actual
        
        best_params = (self._cal_a, self._cal_b, self._cal_c, self._cal_d)
        best_mse = float('inf')
        
        # Grid search over parameter space
        for a in [0.5, 1.0, 1.5, 2.0]:
            for b in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
                for c in [0.5, 1.0, 1.5, 2.0]:
                    for d in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
                        mse = self._compute_calibration_mse(
                            bin_predicted, bin_actual, a, b, c, d
                        )
                        if mse < best_mse:
                            best_mse = mse
                            best_params = (a, b, c, d)
        
        # Update parameters with slow adaptation
        adaptation_rate = 0.1  # Slow adaptation for stability
        self._cal_a = adaptation_rate * best_params[0] + (1 - adaptation_rate) * self._cal_a
        self._cal_b = adaptation_rate * best_params[1] + (1 - adaptation_rate) * self._cal_b
        self._cal_c = adaptation_rate * best_params[2] + (1 - adaptation_rate) * self._cal_c
        self._cal_d = adaptation_rate * best_params[3] + (1 - adaptation_rate) * self._cal_d
        
        # Compute and track ECE
        ece = self._compute_ece()
        self._ece_history.append(ece)
        self._current_ece = ece
        
        logger.debug(f"Calibration refit: a={self._cal_a:.2f}, b={self._cal_b:.2f}, "
                     f"c={self._cal_c:.2f}, d={self._cal_d:.2f}, ECE={ece:.4f}")
    
    def _compute_calibration_mse(self, predicted: np.ndarray, actual: np.ndarray,
                                  a: float, b: float, c: float, d: float) -> float:
        """Compute MSE between Beta calibration and actual outcomes."""
        calibrated = np.array([
            self._beta_calibration_raw(p, a, b, c, d) for p in predicted
        ])
        mse = np.mean((calibrated - actual) ** 2)
        return mse
    
    def _beta_calibration_raw(self, raw_p: float, a: float, b: float,
                               c: float, d: float) -> float:
        """Raw Beta calibration with given parameters."""
        if raw_p <= 0:
            return 0.01
        if raw_p >= 1:
            return 0.99
        
        numerator = a * (raw_p ** b)
        denominator = numerator + c * ((1 - raw_p) ** d)
        
        if denominator <= 0:
            return 0.5
        
        return numerator / denominator
    
    def _compute_ece(self) -> float:
        """
        Compute Expected Calibration Error.
        
        ECE = sum_{b=1}^{B} (n_b / N) * |avg_predicted_b - avg_actual_b|
        
        where B is number of bins, n_b is count in bin b, N is total count.
        """
        total_count = 0
        total_error = 0.0
        
        for i in range(self._n_bins):
            n_b = len(self._bin_outcomes[i])
            if n_b < 1:
                continue
            
            avg_predicted = np.mean(self._bin_predictions[i][-50:]) if \
                           self._bin_predictions[i] else 0.5
            avg_actual = np.mean(self._bin_outcomes[i][-50:]) if \
                        self._bin_outcomes[i] else 0.5
            
            total_count += n_b
            total_error += n_b * abs(avg_predicted - avg_actual)
        
        if total_count > 0:
            ece = total_error / total_count
        else:
            ece = 0.0
        
        return ece
    
    def _compute_calibration_score(self, raw_p: float) -> float:
        """
        Compute calibration quality score for a given probability.
        
        Score = 1 - ECE, clamped to [0, 1]
        Higher score = better calibrated
        """
        score = 1.0 - min(1.0, self._current_ece * 2.0)  # Scale ECE for sensitivity
        return max(0.0, min(1.0, score))
    
    @property
    def current_ece(self) -> float:
        """Current Expected Calibration Error."""
        return self._current_ece
