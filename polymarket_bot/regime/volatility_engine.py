"""
Dynamic volatility regime engine with multi-model estimation.

Mathematical foundation:
- GARCH(1,1): sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
  where omega = long-run variance * (1 - alpha - beta)
  Persistence = alpha + beta (should be < 1 for stationarity)
- Stochastic volatility (Ornstein-Uhlenbeck process):
  d(ln sigma) = kappa * (mu - ln sigma) * dt + sigma_v * dW
  kappa = mean reversion speed, mu = long-run log-vol, sigma_v = vol-of-vol
- Realized volatility: RV = sum(r_i^2) over window (integrated variance estimator)
- Volatility clustering: autocorrelation of |r| or r^2 (GARCH effect)
- Vol-of-vol: variance of volatility changes (second-order clustering)
- Volatility skew: asymmetry in up vs down volatility (leverage effect)
  skew = (sigma_down - sigma_up) / (sigma_down + sigma_up)
- Kalman-filtered volatility: state-space model with adaptive noise
- Regime-adjusted volatility: vol scaled by regime-specific multiplier

All models use exponential forgetting for adaptive estimation.
"""

import logging
import time
import numpy as np
from typing import Optional, Tuple, Dict
from collections import deque

from polymarket_bot.config import VolatilityConfig, RegimeState
from polymarket_bot.bot_types import VolatilityEstimate

logger = logging.getLogger(__name__)


class VolatilityEngine:
    """
    Multi-model volatility estimation engine with regime adaptation.
    
    Key innovations:
    1. GARCH(1,1) with constrained parameters for stationarity
    2. Stochastic volatility via Ornstein-Uhlenbeck process
    3. Realized volatility with multiple window sizes
    4. Vol-of-vol estimation for regime transition detection
    5. Volatility skew (leverage effect) for directional inference
    6. Kalman-filtered volatility for smooth estimation
    7. Regime-adjusted volatility for context-aware estimation
    8. 5-minute forward volatility forecast
    """
    
    def __init__(self, config: VolatilityConfig):
        self.config = config
        self._default_second_vol = 6.0e-5
        
        # GARCH(1,1) parameters
        # sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
        # Persistence = alpha + beta < 1 for stationarity
        self._garch_alpha = 0.1     # Innovation weight
        self._garch_beta = 0.85     # Persistence weight
        long_run_variance = self._default_second_vol ** 2
        self._garch_omega = long_run_variance * (1.0 - self._garch_alpha - self._garch_beta)
        self._garch_variance = long_run_variance  # Current conditional variance estimate
        
        # Stochastic volatility (OU process) parameters
        # d(ln sigma) = kappa * (mu - ln sigma) * dt + sigma_v * dW
        self._sv_kappa = config.stochastic_volatility_mean_reversion  # Mean reversion speed
        self._sv_mu = np.log(self._default_second_vol)  # Long-run log-volatility
        self._sv_sigma_v = config.stochastic_volatility_vol_of_vol  # Vol-of-vol
        self._sv_log_vol = np.log(self._default_second_vol)  # Current log-volatility state
        
        # Realized volatility tracking
        self._returns_history: deque = deque(maxlen=config.volatility_clustering_window)
        self._squared_returns_history: deque = deque(maxlen=config.volatility_clustering_window)
        self._abs_returns_history: deque = deque(maxlen=config.volatility_clustering_window)
        
        # Up/down volatility tracking (for skew)
        self._up_returns: deque = deque(maxlen=200)
        self._down_returns: deque = deque(maxlen=200)
        
        # Kalman volatility filter
        self._kalman_vol_state = self._default_second_vol  # Current volatility estimate
        self._kalman_vol_variance = self._default_second_vol ** 2  # State variance
        self._kalman_process_noise = self._default_second_vol ** 2 * 0.1
        self._kalman_measurement_noise = self._default_second_vol ** 2
        
        # Vol-of-vol tracking
        self._vol_changes_history: deque = deque(maxlen=100)
        self._vol_of_vol_ema = 0.0
        
        # Regime multipliers for volatility adjustment
        self._regime_vol_multipliers: Dict[RegimeState, float] = {
            RegimeState.CALM_TRENDING: 1.0,
            RegimeState.CALM_RANGE: 0.8,
            RegimeState.VOLATILE_TRENDING: 2.5,
            RegimeState.VOLATILE_RANGE: 2.0,
            RegimeState.CRISIS: 5.0,
            RegimeState.LIQUIDATION_CASCADE: 8.0,
        }
        
        # Previous estimate for smoothing
        self._prev_estimate: Optional[VolatilityEstimate] = None
        self._last_update_time = 0.0
        self._initialized = False
        self._update_count = 0
    
    def update(
        self,
        price_return: float,
        timestamp: float,
        regime_state: Optional[RegimeState] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> VolatilityEstimate:
        """
        Update all volatility models with a new price return.
        
        Args:
            price_return: log return r_t = log(P_t / P_{t-1})
            timestamp: current time
            regime_state: current regime for regime-adjusted volatility
            
        Returns:
            VolatilityEstimate with all model outputs
        """
        self._update_count += 1
        
        if elapsed_seconds is None:
            elapsed_seconds = timestamp - self._last_update_time if self._last_update_time else 1.0
        elapsed_seconds = float(np.clip(elapsed_seconds, 1e-3, 30.0))

        # Normalize irregular tick returns into log-return per sqrt(second).
        normalized_return = price_return / np.sqrt(elapsed_seconds)

        # Store normalized return
        self._returns_history.append(normalized_return)
        self._squared_returns_history.append(normalized_return ** 2)
        self._abs_returns_history.append(abs(normalized_return))
        
        # Separate up/down returns for skew
        if normalized_return > 0:
            self._up_returns.append(normalized_return ** 2)
        elif normalized_return < 0:
            self._down_returns.append(normalized_return ** 2)
        
        # Step 1: GARCH(1,1) update
        garch_vol = self._update_garch(normalized_return)
        
        # Step 2: Stochastic volatility update
        sv_vol = self._update_stochastic_volatility(normalized_return, timestamp)
        
        # Step 3: Realized volatility
        realized_vol = self._compute_realized_volatility()
        
        # Step 4: Kalman volatility filter
        kalman_vol = self._update_kalman_volatility(realized_vol)
        
        # Step 5: Vol-of-vol
        vol_of_vol = self._compute_vol_of_vol()
        
        # Step 6: Volatility skew
        vol_skew = self._compute_volatility_skew()
        
        # Step 7: Regime-adjusted volatility
        regime_adjusted_vol = self._compute_regime_adjusted_volatility(
            kalman_vol, regime_state
        )
        
        # Step 8: 5-minute forward forecast
        forecast_5min = self._forecast_volatility_5min(
            garch_vol, kalman_vol, regime_adjusted_vol, regime_state
        )
        
        # Step 9: Confidence interval
        confidence_interval = self._compute_confidence_interval(
            regime_adjusted_vol, vol_of_vol
        )
        
        # Step 10: Regime alignment
        regime_alignment = self._compute_regime_alignment(
            realized_vol, regime_state
        )
        
        # Implied volatility placeholder (from Polymarket prices if available)
        implied_vol = 0.0  # Would need Polymarket price data
        
        # Build result
        result = VolatilityEstimate(
            timestamp=timestamp,
            realized_volatility=realized_vol,
            garch_volatility=garch_vol,
            stochastic_volatility=sv_vol,
            implied_volatility=implied_vol,
            kalman_volatility=kalman_vol,
            regime_adjusted_volatility=regime_adjusted_vol,
            volatility_of_volatility=vol_of_vol,
            volatility_skew=vol_skew,
            volatility_forecast_5min=forecast_5min,
            volatility_confidence_interval=confidence_interval,
            volatility_regime_alignment=regime_alignment,
        )
        
        self._prev_estimate = result
        self._last_update_time = timestamp
        self._initialized = True
        
        return result
    
    def _update_garch(self, return_value: float) -> float:
        """
        GARCH(1,1) update: sigma_t^2 = omega + alpha * r_{t-1}^2 + beta * sigma_{t-1}^2
        
        Parameters constrained for stationarity:
        alpha + beta < 1 (persistence < 1)
        omega > 0 (positive long-run variance)
        
        Long-run variance = omega / (1 - alpha - beta)
        """
        # Update conditional variance
        self._garch_variance = (
            self._garch_omega +
            self._garch_alpha * (return_value ** 2) +
            self._garch_beta * self._garch_variance
        )
        
        # Ensure positivity
        self._garch_variance = max(self._garch_variance, 1e-10)
        
        # Return volatility (standard deviation)
        return np.sqrt(self._garch_variance)
    
    def _update_stochastic_volatility(self, return_value: float,
                                        timestamp: float) -> float:
        """
        Stochastic volatility update via Ornstein-Uhlenbeck process.
        
        d(ln sigma) = kappa * (mu - ln sigma) * dt + sigma_v * dW
        
        Discrete approximation:
        ln sigma_t = ln sigma_{t-1} + kappa * (mu - ln sigma_{t-1}) * dt 
                     + sigma_v * sqrt(dt) * Z
        
        where Z ~ N(0,1) and dt is the time step.
        
        This model captures:
        - Mean reversion in volatility (kappa)
        - Stochastic volatility shocks (sigma_v)
        - Volatility clustering (via OU dynamics)
        """
        dt = 1.0  # Assume 1-second time step (normalized)
        
        # OU process update
        # Mean reversion component
        mean_reversion = self._sv_kappa * (self._sv_mu - self._sv_log_vol) * dt
        
        # Stochastic shock component
        # Use the return as a proxy for the volatility shock
        # Map return to a standard normal via inverse CDF approximation
        vol_shock = self._sv_sigma_v * np.sqrt(dt) * \
                    np.sign(return_value) * min(abs(return_value) * 100, 3.0)
        
        # Update log-volatility state
        self._sv_log_vol += mean_reversion + vol_shock
        
        # Convert to volatility
        sv_vol = np.exp(self._sv_log_vol)
        
        return sv_vol
    
    def _compute_realized_volatility(self) -> float:
        """
        Compute realized volatility from recent returns.
        
        RV = sqrt(sum(r_i^2) / n) for n recent returns
        
        This is the simplest and most robust volatility estimator.
        Uses multiple window sizes and takes the most recent estimate.
        """
        if len(self._squared_returns_history) < 5:
            return self._default_second_vol
        
        # Short window (20 returns) - responsive
        recent_sq = list(self._squared_returns_history)[-20:]
        rv_short = np.sqrt(np.mean(recent_sq))
        
        # Medium window (50 returns) - balanced
        recent_sq_med = list(self._squared_returns_history)[-50:]
        rv_medium = np.sqrt(np.mean(recent_sq_med))
        
        # Long window (100 returns) - stable
        recent_sq_long = list(self._squared_returns_history)[-100:]
        rv_long = np.sqrt(np.mean(recent_sq_long))
        
        # Weighted combination: more weight on recent for responsiveness
        realized = 0.5 * rv_short + 0.3 * rv_medium + 0.2 * rv_long
        
        return max(realized, 1e-10)
    
    def _update_kalman_volatility(self, realized_vol: float) -> float:
        """
        Kalman filter for volatility estimation.
        
        State model: sigma_t = sigma_{t-1} + process_noise
        Measurement model: realized_vol = sigma_t + measurement_noise
        
        Kalman update:
        K_t = P_t / (P_t + R)
        sigma_t = sigma_{t-1} + K_t * (realized_vol - sigma_{t-1})
        P_t = (1 - K_t) * P_t + Q
        
        This provides a smooth, adaptive volatility estimate that
        balances responsiveness (from realized vol) with stability
        (from the state model).
        """
        # Prediction step
        predicted_vol = self._kalman_vol_state
        predicted_variance = self._kalman_vol_variance + self._kalman_process_noise
        
        # Innovation (measurement residual)
        innovation = realized_vol - predicted_vol
        
        # Kalman gain
        total_variance = predicted_variance + self._kalman_measurement_noise
        kalman_gain = predicted_variance / max(total_variance, 1e-10)
        
        # Update step
        self._kalman_vol_state = predicted_vol + kalman_gain * innovation
        self._kalman_vol_variance = (1 - kalman_gain) * predicted_variance
        
        # Ensure positivity
        self._kalman_vol_state = max(self._kalman_vol_state, 1e-10)
        
        return self._kalman_vol_state
    
    def _compute_vol_of_vol(self) -> float:
        """
        Compute volatility of volatility (second-order clustering).
        
        Vol-of-vol = std(volatility_changes) / mean(volatility)
        
        High vol-of-vol: volatility is itself volatile (regime transitions)
        Low vol-of-vol: volatility is stable (steady regime)
        
        This is a key indicator for regime transition detection.
        """
        if self._prev_estimate is None:
            return 0.0
        
        # Track volatility changes
        vol_change = abs(self._kalman_vol_state - self._prev_estimate.kalman_volatility)
        self._vol_changes_history.append(vol_change)
        
        if len(self._vol_changes_history) < 10:
            return 0.0
        
        changes = np.array(list(self._vol_changes_history))
        vol_of_vol = np.std(changes) / max(np.mean(changes), 1e-10)
        
        # Smooth
        self._vol_of_vol_ema = 0.7 * vol_of_vol + 0.3 * self._vol_of_vol_ema
        
        return self._vol_of_vol_ema
    
    def _compute_volatility_skew(self) -> float:
        """
        Compute volatility skew (leverage effect).
        
        skew = (sigma_down - sigma_up) / (sigma_down + sigma_up)
        
        Positive skew: downside volatility higher (bearish pressure)
        Negative skew: upside volatility higher (bullish pressure)
        Near zero: symmetric volatility (neutral)
        
        The leverage effect is a well-documented phenomenon where
        negative returns lead to higher future volatility than
        positive returns of the same magnitude.
        """
        if len(self._up_returns) < 5 or len(self._down_returns) < 5:
            return 0.0
        
        up_vol = np.sqrt(np.mean(list(self._up_returns)[-50:]))
        down_vol = np.sqrt(np.mean(list(self._down_returns)[-50:]))
        
        total_vol = up_vol + down_vol
        if total_vol < 1e-10:
            return 0.0
        
        skew = (down_vol - up_vol) / total_vol
        
        return skew
    
    def _compute_regime_adjusted_volatility(self, base_vol: float,
                                              regime_state: Optional[RegimeState]) -> float:
        """
        Adjust volatility estimate based on current regime.
        
        regime_adjusted_vol = base_vol * regime_multiplier
        
        Regime multipliers reflect the empirical observation that
        volatility characteristics differ dramatically across regimes:
        - Calm regimes: vol tends to understate true risk (multiplier > 1)
        - Volatile regimes: vol tends to overstate near-term risk (multiplier < 1)
        - Crisis/cascade: vol dramatically understates tail risk (multiplier >> 1)
        """
        if regime_state is None:
            return base_vol
        
        multiplier = self._regime_vol_multipliers.get(regime_state, 1.0)
        adjusted = base_vol * multiplier
        
        return max(adjusted, 1e-10)
    
    def _forecast_volatility_5min(self, garch_vol: float, kalman_vol: float,
                                   regime_vol: float,
                                   regime_state: Optional[RegimeState]) -> float:
        """
        Forecast volatility 5 minutes into the future.
        
        GARCH forecast: sigma_{t+k}^2 -> long-run variance as k -> infinity
        sigma_{t+5min}^2 = omega/(1-alpha-beta) + (alpha+beta)^n * (sigma_t^2 - omega/(1-alpha-beta))
        
        For short horizons (5 min), GARCH forecast is close to current sigma.
        For longer horizons, it converges to unconditional variance.
        
        Combined forecast: weighted average of GARCH, Kalman, and regime-adjusted.
        """
        # GARCH 5-minute forecast
        persistence = self._garch_alpha + self._garch_beta
        long_run_var = self._garch_omega / max(1 - persistence, 0.01)
        
        # Number of steps in 5 minutes (assuming 1-second updates)
        n_steps = 300
        
        # GARCH multi-step forecast
        # sigma_{t+n}^2 = long_run_var + persistence^n * (sigma_t^2 - long_run_var)
        garch_forecast_var = long_run_var + (persistence ** n_steps) * \
                            (self._garch_variance - long_run_var)
        garch_forecast = np.sqrt(max(garch_forecast_var, 1e-10))
        
        # Kalman forecast: assume volatility persists (OU mean reversion)
        # sigma_{t+5min} = mu + (sigma_t - mu) * exp(-kappa * 5min)
        sv_forecast = np.exp(
            self._sv_mu + (self._sv_log_vol - self._sv_mu) * 
            np.exp(-self._sv_kappa * 300)
        )
        
        # Combined forecast: weighted average
        # Weight: GARCH (30%), Kalman (30%), regime-adjusted (40%)
        forecast = 0.3 * garch_forecast + 0.3 * sv_forecast + 0.4 * regime_vol
        
        return max(forecast, 1e-10)
    
    def _compute_confidence_interval(self, vol: float, vol_of_vol: float) -> Tuple[float, float]:
        """
        Compute confidence interval for volatility estimate.
        
        CI = vol * (1 +/- vol_of_vol * z_alpha)
        
        where z_alpha = 1.96 for 95% confidence
        
        Wider interval when vol-of-vol is high (uncertain regime)
        Narrower interval when vol-of-vol is low (stable regime)
        """
        z = 1.96  # 95% confidence
        margin = vol * vol_of_vol * z
        
        lower = max(0, vol - margin)
        upper = vol + margin
        
        return (lower, upper)
    
    def _compute_regime_alignment(self, realized_vol: float,
                                   regime_state: Optional[RegimeState]) -> float:
        """
        Compute how well current realized volatility aligns with regime expectation.
        
        alignment = 1 - |realized_vol - expected_vol| / max(realized_vol, expected_vol)
        
        High alignment (>0.8): vol matches regime (confident classification)
        Low alignment (<0.3): vol doesn't match regime (possible misclassification)
        """
        if regime_state is None:
            return 0.5
        
        expected_priors = getattr(self.config, "regime_volatility_prior", {})
        expected_vol = expected_priors.get(regime_state, self._default_second_vol)
        
        max_vol = max(realized_vol, expected_vol, 1e-10)
        alignment = 1.0 - abs(realized_vol - expected_vol) / max_vol
        
        return max(0.0, min(1.0, alignment))
