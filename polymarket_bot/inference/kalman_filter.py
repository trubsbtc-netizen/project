"""
Kalman filter for BTC price state estimation with adaptive noise.

Mathematical foundation:
- State-space model:
  x_t = F * x_{t-1} + w_t  (state transition, w ~ N(0, Q))
  z_t = H * x_t + v_t      (measurement, v ~ N(0, R))
  
  where:
  x = [price, velocity, acceleration] (3-dimensional state)
  F = [[1, dt, dt^2/2], [0, 1, dt], [0, 0, 1]] (constant acceleration model)
  H = [1, 0, 0] (observe price only)
  
- Kalman update equations:
  Prediction:
    x_pred = F * x_prev
    P_pred = F * P_prev * F^T + Q
  
  Update:
    y = z - H * x_pred                    (innovation)
    S = H * P_pred * H^T + R              (innovation covariance)
    K = P_pred * H^T * S^{-1}             (Kalman gain)
    x = x_pred + K * y                    (state update)
    P = (I - K * H) * P_pred              (covariance update)
  
- Adaptive noise estimation:
  Q and R are estimated from innovation sequence using:
  R_t = alpha * (y^2 - H * P_pred * H^T) + (1-alpha) * R_{t-1}
  Q_t = alpha * (K * y * y^T * K^T) + (1-alpha) * Q_{t-1}
  
- Outlier rejection via innovation test:
  y^2 / S > chi2_threshold => observation rejected as outlier
  
- Velocity and acceleration extraction:
  velocity = x[1] (price velocity = trend speed)
  acceleration = x[2] (price acceleration = trend change)
  
  Confidence in velocity/acceleration from covariance:
  velocity_confidence = 1 - P[1,1] / max(P[1,1], threshold)
  acceleration_confidence = 1 - P[2,2] / max(P[2,2], threshold)

The Kalman filter provides the foundational state estimate for all
higher-level inference: velocity drives continuation, acceleration
drives reversal, and covariance drives confidence.
"""

import logging
import time
import numpy as np
from typing import Optional, Tuple

from polymarket_bot.config import KalmanConfig
from polymarket_bot.bot_types import KalmanState

logger = logging.getLogger(__name__)


class KalmanFilter:
    """
    Adaptive Kalman filter for BTC price state estimation.
    
    State vector: [price, velocity, acceleration]
    Provides trend speed (velocity) and trend change (acceleration)
    with confidence estimates from state covariance.
    
    Key innovations:
    1. Constant acceleration state model (3D state)
    2. Adaptive process and measurement noise
    3. Outlier rejection via chi-squared innovation test
    4. Velocity and acceleration confidence from covariance
    5. Numerically stable Joseph form for covariance update
    6. Time-varying transition matrix (dt from timestamps)
    """
    
    def __init__(self, config: KalmanConfig):
        self.config = config
        
        # State vector: [price, velocity, acceleration]
        self._state = np.zeros(config.state_dim)
        self._state[0] = 0.0  # Price (will be initialized on first observation)
        
        # State covariance
        self._P = np.eye(config.state_dim) * config.initial_state_variance
        
        # Process noise covariance Q
        self._Q = np.eye(config.state_dim) * config.process_noise_scale
        
        # Measurement noise R
        self._R = config.measurement_noise_scale
        
        # Kalman gain (last computed)
        self._K = np.zeros((config.state_dim, 1))
        
        # Innovation tracking
        self._innovation = 0.0
        self._innovation_variance = 0.0
        self._log_likelihood = 0.0
        
        # Adaptive noise estimation
        self._adaptive_window = config.adaptive_noise_window
        self._innovation_history = []
        self._adaptive_alpha = 0.1  # Smoothing for adaptive noise
        
        # Outlier rejection
        self._is_outlier = False
        self._chi2_threshold = config.innovation_threshold
        
        # Measurement matrix H: observe price only
        self._H = np.zeros((1, config.state_dim))
        self._H[0, 0] = 1.0
        
        # Time tracking
        self._last_timestamp = 0.0
        self._initialized = False
        self._update_count = 0
        
        # Confidence thresholds
        self._velocity_conf_threshold = 0.001
        self._acceleration_conf_threshold = 0.0001
    
    def update(self, price: float, timestamp: float) -> KalmanState:
        """
        Update Kalman filter with a new price observation.
        
        Process:
        1. Compute time step dt from timestamps
        2. Build time-varying transition matrix F(dt)
        3. Predict state and covariance
        4. Compute innovation (measurement residual)
        5. Check for outlier via innovation test
        6. Compute Kalman gain
        7. Update state and covariance (Joseph form)
        8. Adapt noise estimates
        9. Extract velocity, acceleration, and confidence
        """
        # Step 1: Compute time step
        if not self._initialized:
            self._state[0] = price
            self._last_timestamp = timestamp
            self._initialized = True
            return self._build_state(timestamp)
        
        dt = timestamp - self._last_timestamp
        if dt <= 0:
            dt = 0.001  # Minimum time step
        dt = min(dt, 10.0)  # Maximum time step (prevent huge jumps)
        
        self._last_timestamp = timestamp
        self._update_count += 1
        
        # Step 2: Build transition matrix F(dt)
        # Constant acceleration model:
        # F = [[1, dt, dt^2/2],
        #      [0, 1, dt],
        #      [0, 0, 1]]
        F = np.eye(self.config.state_dim)
        F[0, 1] = dt
        F[0, 2] = dt ** 2 / 2.0
        F[1, 2] = dt
        
        # Step 3: Predict
        x_pred = F @ self._state
        P_pred = F @ self._P @ F.T + self._Q
        
        # Step 4: Innovation
        z = np.array([price])
        z_pred = self._H @ x_pred
        innovation = z - z_pred
        self._innovation = innovation[0]
        
        # Innovation covariance
        S = self._H @ P_pred @ self._H.T + self._R
        self._innovation_variance = S[0, 0]
        
        # Step 5: Outlier test
        # Chi-squared test: y^2 / S > threshold
        innovation_test = self._innovation ** 2 / max(self._innovation_variance, 1e-10)
        self._is_outlier = innovation_test > self._chi2_threshold
        
        # Step 6: Kalman gain
        # K = P_pred * H^T * S^{-1}
        S_inv = 1.0 / max(self._innovation_variance, 1e-10)
        self._K = P_pred @ self._H.T * S_inv
        
        # Step 7: Update state and covariance
        if self._is_outlier:
            # Outlier: reduce Kalman gain (don't fully trust this observation)
            # Use reduced gain instead of skipping entirely
            reduced_factor = 0.1  # Only 10% weight for outlier
            self._K *= reduced_factor
            logger.debug(f"Outlier detected: innovation_test={innovation_test:.2f}")
        
        # State update
        self._state = x_pred + self._K @ innovation
        
        # Covariance update (Joseph form for numerical stability)
        # P = (I - K*H) * P_pred * (I - K*H)^T + K * R * K^T
        I_KH = np.eye(self.config.state_dim) - self._K @ self._H
        self._P = I_KH @ P_pred @ I_KH.T + self._K @ np.array([[self._R]]) @ self._K.T
        
        # Ensure covariance is positive definite
        self._P = self._ensure_positive_definite(self._P)
        
        # Step 8: Adaptive noise estimation
        self._adapt_noise(innovation, S)
        
        # Step 9: Log-likelihood
        # log L = -0.5 * (log(det(S)) + y^T * S^{-1} * y + d*log(2*pi))
        if self._innovation_variance > 0:
            self._log_likelihood = -0.5 * (
                np.log(self._innovation_variance) +
                self._innovation ** 2 / self._innovation_variance +
                np.log(2 * np.pi)
            )
        
        return self._build_state(timestamp)
    
    def _build_state(self, timestamp: float) -> KalmanState:
        """Build KalmanState from current filter state."""
        velocity = self._state[1]
        acceleration = self._state[2]
        
        # Confidence from covariance diagonal
        velocity_var = self._P[1, 1]
        acceleration_var = self._P[2, 2]
        
        # Confidence = 1 - normalized variance
        # Higher variance = lower confidence
        velocity_confidence = 1.0 - min(1.0, velocity_var / self._velocity_conf_threshold)
        acceleration_confidence = 1.0 - min(1.0, acceleration_var / self._acceleration_conf_threshold)
        
        # Ensure minimum confidence
        velocity_confidence = max(0.0, velocity_confidence)
        acceleration_confidence = max(0.0, acceleration_confidence)
        
        return KalmanState(
            timestamp=timestamp,
            state=self._state.copy(),
            covariance=self._P.copy(),
            innovation=self._innovation,
            innovation_variance=self._innovation_variance,
            kalman_gain=self._K.copy(),
            log_likelihood=self._log_likelihood,
            is_outlier=self._is_outlier,
            velocity_estimate=velocity,
            acceleration_estimate=acceleration,
            velocity_confidence=velocity_confidence,
            acceleration_confidence=acceleration_confidence,
        )
    
    def _adapt_noise(self, innovation: np.ndarray, S: np.ndarray):
        """
        Adaptively estimate process and measurement noise.
        
        Method: exponential smoothing of innovation-based estimates.
        
        R adaptation: based on innovation magnitude
        R_t = alpha * max(0, y^2 - H*P*H^T) + (1-alpha) * R_{t-1}
        
        Q adaptation: based on Kalman gain and innovation
        Q_t = alpha * K*y*y^T*K^T + (1-alpha) * Q_{t-1}
        
        This allows the filter to adapt to changing noise conditions:
        - In volatile regimes: R increases (more measurement noise)
        - In calm regimes: R decreases (less measurement noise)
        - During regime transitions: Q increases (more process noise)
        """
        alpha = self._adaptive_alpha
        
        # Measurement noise adaptation
        # R_t = alpha * max(0, y^2 - H*P*H^T) + (1-alpha) * R_{t-1}
        HPHt = (self._H @ self._P @ self._H.T)[0, 0]
        innovation_sq = innovation[0] ** 2
        
        # Only increase R if innovation is larger than expected
        new_R = max(0, innovation_sq - HPHt)
        self._R = alpha * new_R + (1 - alpha) * self._R
        
        # Floor R to prevent it from going to zero
        self._R = max(self._R, 1e-6)
        
        # Process noise adaptation
        # Q_t = alpha * K*y*y^T*K^T + (1-alpha) * Q_{t-1}
        Ky = self._K @ innovation
        Q_update = np.outer(Ky, Ky)
        
        self._Q = alpha * Q_update + (1 - alpha) * self._Q
        
        # Floor Q diagonal to prevent it from going to zero
        for i in range(self.config.state_dim):
            self._Q[i, i] = max(self._Q[i, i], 1e-8)
    
    def _ensure_positive_definite(self, P: np.ndarray) -> np.ndarray:
        """
        Ensure covariance matrix is positive definite.
        
        Method: eigenvalue decomposition with floor.
        If any eigenvalue < 0, set it to minimum value.
        This prevents numerical instability in the filter.
        """
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(P)
            
            # Floor eigenvalues
            min_eigenvalue = 1e-8
            eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
            
            # Reconstruct
            P = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            
            # Ensure symmetry
            P = (P + P.T) / 2.0
            
        except np.linalg.LinAlgError:
            # Fallback: reset to diagonal
            P = np.eye(self.config.state_dim) * self.config.initial_state_variance
        
        return P
    
    @property
    def current_price(self) -> float:
        """Current price estimate from Kalman state."""
        return self._state[0]
    
    @property
    def current_velocity(self) -> float:
        """Current velocity estimate from Kalman state."""
        return self._state[1]
    
    @property
    def current_acceleration(self) -> float:
        """Current acceleration estimate from Kalman state."""
        return self._state[2]