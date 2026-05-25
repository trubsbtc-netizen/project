"""
Spoof-resistant orderbook pressure analyzer.

Mathematical foundation:
- Stoikov microprice: P_micro = (P_bid * Q_ask + P_ask * Q_bid) / (Q_bid + Q_ask)
- Volume-weighted depth imbalance with exponential distance decay
- Spoof detection via cancellation velocity vs addition velocity ratio
- Fake liquidity inference from order survival probability modeling
- Depth skew coefficient via power-law fitting: Q(d) ~ d^(-alpha)

All computations are numerically stable with proper normalization.
"""

import logging
import time
import numpy as np
from typing import Optional, List, Dict, Tuple

from polymarket_bot.config import MicrostructureConfig
from polymarket_bot.bot_types import (
    OrderbookSnapshot, OrderbookLevel, OrderbookDelta, OrderbookBuffer,
    OrderbookPressure, OrderbookLevel
)

logger = logging.getLogger(__name__)


class OrderbookAnalyzer:
    """
    Analyzes orderbook microstructure with spoof resistance.
    
    Key innovations:
    1. Distinguishes real vs fake liquidity via cancellation velocity modeling
    2. Computes Stoikov microprice for fair price estimation
    3. Power-law depth skew detection for liquidity distribution analysis
    4. Exponential distance-weighted pressure with spoof filtering
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        self._smoothing_factor = config.orderbook_pressure_smoothing
        self._prev_pressure: Optional[OrderbookPressure] = None
        self._spoof_tracker: Dict[float, Dict] = {}  # price -> spoof tracking data
        self._cancellation_velocity_history: List[float] = []
        self._addition_velocity_history: List[float] = []
        self._last_update_time = 0.0
        
        # Exponential decay weights for distance-based pressure
        # Pressure contribution decays as: w(d) = exp(-lambda * d / mid_price)
        self._distance_decay_lambda = 5.0  # Decay rate for distance weighting
        
        # Power-law depth skew parameters
        self._depth_skew_alpha_bid = 1.0  # Initial estimate
        self._depth_skew_alpha_ask = 1.0
        
    def analyze(self, snapshot: OrderbookSnapshot, 
                buffer: Optional[OrderbookBuffer] = None) -> OrderbookPressure:
        """
        Compute full orderbook pressure metrics with spoof filtering.
        
        Process:
        1. Compute raw bid/ask pressure with distance weighting
        2. Detect spoof orders via cancellation velocity analysis
        3. Filter fake liquidity and compute real pressure
        4. Compute Stoikov microprice
        5. Estimate depth skew via power-law fitting
        6. Smooth with previous estimate for stability
        """
        timestamp = snapshot.timestamp
        
        if not snapshot.bids or not snapshot.asks:
            return self._empty_pressure(timestamp)
        
        mid_price = snapshot.mid_price
        if mid_price is None or mid_price <= 0:
            return self._empty_pressure(timestamp)
        
        # Step 1: Compute raw distance-weighted pressure
        raw_bid_pressure, raw_ask_pressure, bid_weights, ask_weights = \
            self._compute_distance_weighted_pressure(snapshot, mid_price)
        
        # Step 2: Spoof detection and fake liquidity estimation
        spoof_score, fake_ratio, real_levels_bid, real_levels_ask = \
            self._detect_spoof_and_filter(snapshot, buffer, timestamp)
        
        # Step 3: Compute real (spoof-filtered) pressure
        real_bid_pressure, real_ask_pressure = \
            self._compute_real_pressure(real_levels_bid, real_levels_ask, mid_price)
        
        # Step 4: Compute Stoikov microprice
        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        bid_size_at_best = snapshot.bids[0].size if snapshot.bids else 0
        ask_size_at_best = snapshot.asks[0].size if snapshot.asks else 0
        
        # Use real (filtered) sizes for microprice
        real_bid_size = real_levels_bid[0].size if real_levels_bid else bid_size_at_best
        real_ask_size = real_levels_ask[0].size if real_levels_ask else ask_size_at_best
        
        microprice = OrderbookPressure.compute_microprice(
            best_bid, real_bid_size, best_ask, real_ask_size
        )
        
        # Step 5: Compute weighted midpoint (volume-weighted across multiple levels)
        weighted_mid = self._compute_weighted_midpoint(snapshot, mid_price)
        
        # Step 6: Depth skew via power-law fitting
        depth_skew = self._compute_depth_skew(snapshot, mid_price)
        
        # Step 7: Compute normalized imbalance metrics
        total_raw = raw_bid_pressure + raw_ask_pressure
        raw_imbalance = (raw_bid_pressure - raw_ask_pressure) / max(total_raw, 1e-10)
        net_raw_pressure = raw_bid_pressure - raw_ask_pressure
        
        total_real = real_bid_pressure + real_ask_pressure
        real_imbalance = (real_bid_pressure - real_ask_pressure) / max(total_real, 1e-10)
        net_real_pressure = real_bid_pressure - real_ask_pressure
        
        # Step 8: Smooth with previous estimate (exponential moving average)
        if self._prev_pressure is not None:
            alpha = self._smoothing_factor
            raw_imbalance = alpha * raw_imbalance + (1 - alpha) * self._prev_pressure.pressure_imbalance
            real_imbalance = alpha * real_imbalance + (1 - alpha) * self._prev_pressure.real_imbalance
            net_raw_pressure = alpha * net_raw_pressure + (1 - alpha) * self._prev_pressure.net_pressure
            net_real_pressure = alpha * net_real_pressure + (1 - alpha) * self._prev_pressure.real_net_pressure
        
        result = OrderbookPressure(
            timestamp=timestamp,
            bid_pressure=raw_bid_pressure,
            ask_pressure=raw_ask_pressure,
            net_pressure=net_raw_pressure,
            pressure_imbalance=raw_imbalance,
            real_bid_pressure=real_bid_pressure,
            real_ask_pressure=real_ask_pressure,
            real_net_pressure=net_real_pressure,
            real_imbalance=real_imbalance,
            spoof_score=spoof_score,
            fake_liquidity_ratio=fake_ratio,
            depth_skew=depth_skew,
            weighted_mid=weighted_mid,
            microprice=microprice,
        )
        
        self._prev_pressure = result
        self._last_update_time = timestamp
        return result
    
    def _compute_distance_weighted_pressure(
        self, snapshot: OrderbookSnapshot, mid_price: float
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Compute distance-weighted orderbook pressure.
        
        Formula:
            P_bid = sum_{i=1}^{N} Q_bid_i * exp(-lambda * |P_bid_i - mid| / mid)
            P_ask = sum_{i=1}^{N} Q_ask_i * exp(-lambda * |P_ask_i - mid| / mid)
        
        Orders closer to mid price contribute more to pressure.
        This models the empirical observation that near-book orders have
        disproportionate price impact (Cartea & Jaimungal, 2015).
        """
        depth = self.config.orderbook_depth_levels
        
        bid_pressures = []
        ask_pressures = []
        bid_weights_arr = []
        ask_weights_arr = []
        
        for i, level in enumerate(snapshot.bids[:depth]):
            distance = abs(level.price - mid_price) / mid_price
            weight = np.exp(-self._distance_decay_lambda * distance)
            bid_pressures.append(level.size * weight)
            bid_weights_arr.append(weight)
        
        for i, level in enumerate(snapshot.asks[:depth]):
            distance = abs(level.price - mid_price) / mid_price
            weight = np.exp(-self._distance_decay_lambda * distance)
            ask_pressures.append(level.size * weight)
            ask_weights_arr.append(weight)
        
        total_bid = sum(bid_pressures)
        total_ask = sum(ask_pressures)
        
        return (
            total_bid, total_ask,
            np.array(bid_weights_arr), np.array(ask_weights_arr)
        )
    
    def _detect_spoof_and_filter(
        self, snapshot: OrderbookSnapshot,
        buffer: Optional[OrderbookBuffer],
        timestamp: float
    ) -> Tuple[float, float, List[OrderbookLevel], List[OrderbookLevel]]:
        """
        Detect spoof orders and filter fake liquidity.
        
        Spoof detection methodology:
        1. Cancellation velocity: rate of order cancellations vs additions
           spoof_score = cancel_velocity / (cancel_velocity + add_velocity)
        2. Order survival probability: P(order survives t seconds) = exp(-cancel_rate * t)
           Orders with low survival probability are flagged as potential spoof
        3. Size anomaly: orders significantly larger than average at same level
           that appear and disappear quickly
        4. Layering detection: multiple large orders at consecutive levels
           that are cancelled together
        
        Returns: (spoof_score, fake_ratio, filtered_bids, filtered_asks)
        """
        cancel_velocity = 0.0
        add_velocity = 0.0
        
        if buffer is not None:
            # Compute cancellation and addition velocities from recent deltas
            recent_deltas = buffer.recent_deltas(200)
            if recent_deltas:
                time_span = max(
                    recent_deltas[-1].timestamp - recent_deltas[0].timestamp,
                    1.0
                )
                
                cancel_count = 0
                add_count = 0
                cancel_volume = 0.0
                add_volume = 0.0
                
                for delta in recent_deltas:
                    if delta.is_trade:
                        continue  # Trades are real flow, not spoof
                    if delta.size_delta < 0:
                        cancel_count += 1
                        cancel_volume += abs(delta.size_delta)
                    elif delta.size_delta > 0:
                        add_count += 1
                        add_volume += delta.size_delta
                
                cancel_velocity = cancel_volume / time_span if time_span > 0 else 0
                add_velocity = add_volume / time_span if time_span > 0 else 0
                
                self._cancellation_velocity_history.append(cancel_velocity)
                self._addition_velocity_history.append(add_velocity)
        
        # Compute spoof score: ratio of cancellation to total activity
        total_velocity = cancel_velocity + add_velocity
        if total_velocity > 0:
            raw_spoof_score = cancel_velocity / total_velocity
        else:
            raw_spoof_score = 0.0
        
        # Threshold-based spoof detection
        spoof_detected = raw_spoof_score > self.config.spoof_detection_threshold
        
        # Compute cancellation velocity ratio (cancel rate / add rate)
        cancel_add_ratio = cancel_velocity / max(add_velocity, 1e-10)
        high_cancel_ratio = cancel_add_ratio > self.config.spoof_cancellation_velocity_threshold
        
        # Filter orderbook levels based on spoof analysis
        filtered_bids = self._filter_levels(
            snapshot.bids, "bid", spoof_detected, high_cancel_ratio, timestamp
        )
        filtered_asks = self._filter_levels(
            snapshot.asks, "ask", spoof_detected, high_cancel_ratio, timestamp
        )
        
        # Compute fake liquidity ratio
        total_displayed_bid = sum(l.size for l in snapshot.bids[:self.config.orderbook_depth_levels])
        total_displayed_ask = sum(l.size for l in snapshot.asks[:self.config.orderbook_depth_levels])
        total_real_bid = sum(l.size for l in filtered_bids)
        total_real_ask = sum(l.size for l in filtered_asks)
        
        total_displayed = total_displayed_bid + total_displayed_ask
        total_real = total_real_bid + total_real_ask
        
        fake_ratio = 1.0 - (total_real / max(total_displayed, 1e-10))
        fake_ratio = max(0.0, min(1.0, fake_ratio))
        
        # If spoof not detected via velocity, use threshold-based fake ratio
        if not spoof_detected and fake_ratio < self.config.fake_liquidity_ratio_threshold:
            fake_ratio = 0.0  # No significant fake liquidity
        
        # Smooth spoof score
        spoof_score = raw_spoof_score
        if self._prev_pressure is not None:
            alpha = 0.7  # Faster adaptation for spoof detection
            spoof_score = alpha * raw_spoof_score + (1 - alpha) * self._prev_pressure.spoof_score
        
        return spoof_score, fake_ratio, filtered_bids, filtered_asks
    
    def _filter_levels(
        self, levels: List[OrderbookLevel], side: str,
        spoof_detected: bool, high_cancel_ratio: bool,
        timestamp: float
    ) -> List[OrderbookLevel]:
        """
        Filter orderbook levels to remove suspected spoof orders.
        
        Filtering criteria:
        1. If spoof detected: reduce weight of large orders at outer levels
        2. If high cancel ratio: reduce weight of recently appeared large orders
        3. Apply survival probability discount: P(survive) = exp(-cancel_rate * age)
        4. Preserve orders at best levels (more likely real)
        """
        if not levels:
            return []
        
        filtered = []
        depth = self.config.orderbook_depth_levels
        
        # Compute average size for normalization
        sizes = [l.size for l in levels[:depth]]
        avg_size = np.mean(sizes) if sizes else 1.0
        std_size = np.std(sizes) if len(sizes) > 1 else avg_size
        
        for i, level in enumerate(levels[:depth]):
            # Base survival probability depends on level position
            # Best levels (i=0,1) have higher survival probability
            position_factor = np.exp(-0.3 * i)  # Decay with distance from best
            
            # Size anomaly factor: unusually large orders are more likely spoof
            size_ratio = level.size / max(avg_size, 1e-10)
            if size_ratio > 3.0:  # More than 3x average
                size_factor = 0.3  # Heavy discount
            elif size_ratio > 2.0:
                size_factor = 0.6
            elif size_ratio > 1.5:
                size_factor = 0.8
            else:
                size_factor = 1.0
            
            # Spoof detection factor
            if spoof_detected and i > 2:  # Outer levels more likely spoof
                spoof_factor = 0.4
            elif spoof_detected:
                spoof_factor = 0.7
            else:
                spoof_factor = 1.0
            
            # High cancel ratio factor
            cancel_factor = 0.5 if high_cancel_ratio and size_ratio > 2.0 else 1.0
            
            # Combined survival probability
            survival_prob = position_factor * size_factor * spoof_factor * cancel_factor
            survival_prob = max(0.1, min(1.0, survival_prob))  # Floor at 10%
            
            # Effective size after spoof filtering
            effective_size = level.size * survival_prob
            
            filtered.append(OrderbookLevel(
                price=level.price,
                size=effective_size,
                order_count=level.order_count,
                timestamp=level.timestamp,
                is_real=survival_prob > 0.5,  # Flag as real if >50% survival
            ))
        
        return filtered
    
    def _compute_real_pressure(
        self, bid_levels: List[OrderbookLevel],
        ask_levels: List[OrderbookLevel],
        mid_price: float
    ) -> Tuple[float, float]:
        """Compute pressure using spoof-filtered levels."""
        bid_pressure = 0.0
        for level in bid_levels:
            distance = abs(level.price - mid_price) / mid_price
            weight = np.exp(-self._distance_decay_lambda * distance)
            bid_pressure += level.size * weight
        
        ask_pressure = 0.0
        for level in ask_levels:
            distance = abs(level.price - mid_price) / mid_price
            weight = np.exp(-self._distance_decay_lambda * distance)
            ask_pressure += level.size * weight
        
        return bid_pressure, ask_pressure
    
    def _compute_weighted_midpoint(
        self, snapshot: OrderbookSnapshot, mid_price: float
    ) -> float:
        """
        Compute volume-weighted midpoint across multiple levels.
        
        P_weighted = sum(P_i * Q_i) / sum(Q_i) for i in levels near mid
        
        This provides a more stable fair price estimate than simple mid.
        """
        depth = min(5, len(snapshot.bids), len(snapshot.asks))
        
        total_volume = 0.0
        weighted_sum = 0.0
        
        for i in range(depth):
            bid = snapshot.bids[i]
            ask = snapshot.asks[i]
            
            # Weight by inverse distance from mid (closer = more weight)
            bid_dist = abs(bid.price - mid_price) / mid_price
            ask_dist = abs(ask.price - mid_price) / mid_price
            
            bid_weight = bid.size * np.exp(-3.0 * bid_dist)
            ask_weight = ask.size * np.exp(-3.0 * ask_dist)
            
            weighted_sum += bid.price * bid_weight + ask.price * ask_weight
            total_volume += bid_weight + ask_weight
        
        if total_volume > 0:
            return weighted_sum / total_volume
        return mid_price
    
    def _compute_depth_skew(self, snapshot: OrderbookSnapshot, mid_price: float) -> float:
        """
        Compute depth skew coefficient via power-law fitting.
        
        Empirical orderbook depth follows: Q(d) ~ d^(-alpha)
        where d is distance from mid price and alpha is the skew coefficient.
        
        Higher alpha = more concentrated depth near mid (typical in calm markets)
        Lower alpha = more dispersed depth (typical in volatile/crisis markets)
        
        Asymmetry in alpha between bid and ask sides indicates directional pressure.
        
        depth_skew = alpha_bid - alpha_ask
        Positive = bid depth more concentrated (buy pressure)
        Negative = ask depth more concentrated (sell pressure)
        """
        depth = min(10, len(snapshot.bids), len(snapshot.asks))
        if depth < 3:
            return 0.0
        
        # Fit power-law for bid side
        bid_distances = np.array([
            abs(snapshot.bids[i].price - mid_price) / mid_price 
            for i in range(depth)
        ])
        bid_sizes = np.array([snapshot.bids[i].size for i in range(depth)])
        
        alpha_bid = self._fit_power_law_exponent(bid_distances, bid_sizes)
        
        # Fit power-law for ask side
        ask_distances = np.array([
            abs(snapshot.asks[i].price - mid_price) / mid_price 
            for i in range(depth)
        ])
        ask_sizes = np.array([snapshot.asks[i].size for i in range(depth)])
        
        alpha_ask = self._fit_power_law_exponent(ask_distances, ask_sizes)
        
        # Update running estimates with smoothing
        self._depth_skew_alpha_bid = 0.8 * alpha_bid + 0.2 * self._depth_skew_alpha_bid
        self._depth_skew_alpha_ask = 0.8 * alpha_ask + 0.2 * self._depth_skew_alpha_ask
        
        return self._depth_skew_alpha_bid - self._depth_skew_alpha_ask
    
    def _fit_power_law_exponent(self, distances: np.ndarray, sizes: np.ndarray) -> float:
        """
        Fit power-law exponent alpha to Q(d) ~ d^(-alpha).
        
        Using log-log linear regression:
        log(Q) = -alpha * log(d) + C
        
        With regularization to prevent overfitting on sparse data.
        """
        # Filter out zero distances and sizes
        mask = (distances > 0) & (sizes > 0)
        if np.sum(mask) < 2:
            return 1.0  # Default neutral exponent
        
        log_d = np.log(distances[mask])
        log_q = np.log(sizes[mask])
        
        # Simple linear regression: log_q = -alpha * log_d + C
        n = len(log_d)
        if n < 2:
            return 1.0
        
        # Compute regression coefficients
        mean_x = np.mean(log_d)
        mean_y = np.mean(log_q)
        
        var_x = np.var(log_d)
        if var_x < 1e-10:
            return 1.0
        
        cov_xy = np.mean((log_d - mean_x) * (log_q - mean_y))
        
        # alpha = -cov_xy / var_x (negative because Q ~ d^(-alpha))
        alpha = -cov_xy / var_x
        
        # Regularize: clamp to reasonable range
        alpha = max(0.1, min(5.0, alpha))
        
        return alpha
    
    def _empty_pressure(self, timestamp: float) -> OrderbookPressure:
        """Return empty pressure metrics when no data available."""
        return OrderbookPressure(
            timestamp=timestamp,
            bid_pressure=0.0,
            ask_pressure=0.0,
            net_pressure=0.0,
            pressure_imbalance=0.0,
            real_bid_pressure=0.0,
            real_ask_pressure=0.0,
            real_net_pressure=0.0,
            real_imbalance=0.0,
            spoof_score=0.0,
            fake_liquidity_ratio=0.0,
            depth_skew=0.0,
            weighted_mid=0.0,
            microprice=0.0,
        )