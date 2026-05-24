"""
Queue persistence and survival modeling for orderbook dynamics.

Mathematical foundation:
- Queue survival probability: P(survive t) = exp(-lambda * t)
  where lambda = cancel_rate + fill_rate is the total departure rate
- Queue position estimation: E[position] = integral of survival probability
- Queue imbalance: asymmetry in bid vs ask queue survival rates
- Effective queue depth: displayed depth * P(survive settlement_time)
- Queue decay rate: exponential decay parameter estimated from cancellation data
- Front-of-queue probability: P(front) = 1 / (1 + effective_queue_depth * fill_rate)

The queue model captures the dynamics of how orders in the book evolve over time,
which is critical for understanding execution probability and real vs fake liquidity.
"""

import logging
import time
import numpy as np
from typing import Optional, Tuple
from collections import deque

from polymarket_bot.config import MicrostructureConfig
from polymarket_bot.bot_types import (
    OrderbookSnapshot, OrderbookDelta, OrderbookBuffer, QueueMetrics
)

logger = logging.getLogger(__name__)


class QueueModel:
    """
    Models orderbook queue dynamics with survival probability estimation.
    
    Key innovations:
    1. Queue survival probability via exponential decay model
    2. Queue position estimation for execution probability
    3. Queue imbalance detection (bid vs ask queue health)
    4. Effective queue depth accounting for cancellations
    5. Front-of-queue probability for fill estimation
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        
        # Queue state tracking
        self._bid_cancel_rate = 0.0  # Orders/second cancelled on bid side
        self._ask_cancel_rate = 0.0  # Orders/second cancelled on ask side
        self._bid_fill_rate = 0.0    # Orders/second filled on bid side
        self._ask_fill_rate = 0.0    # Orders/second filled on ask side
        
        # Queue survival tracking
        self._bid_queue_survival = 0.5  # P(bid queue survives)
        self._ask_queue_survival = 0.5  # P(ask queue survives)
        
        # Queue depth tracking
        self._bid_depth_history: deque = deque(maxlen=config.queue_persistence_window)
        self._ask_depth_history: deque = deque(maxlen=config.queue_persistence_window)
        
        # Cancellation tracking for rate estimation
        self._bid_cancel_events: deque = deque(maxlen=200)
        self._ask_cancel_events: deque = deque(maxlen=200)
        self._bid_fill_events: deque = deque(maxlen=200)
        self._ask_fill_events: deque = deque(maxlen=200)
        
        # Queue age tracking
        self._avg_bid_order_age = 0.0
        self._avg_ask_order_age = 0.0
        
        # Previous metrics for smoothing
        self._prev_metrics: Optional[QueueMetrics] = None
        self._last_update_time = 0.0
        self._initialized = False
    
    def analyze(self, snapshot: OrderbookSnapshot,
                buffer: Optional[OrderbookBuffer] = None) -> QueueMetrics:
        """
        Compute queue dynamics metrics from orderbook and delta history.
        
        Process:
        1. Estimate cancellation and fill rates from delta history
        2. Compute queue survival probability for each side
        3. Estimate queue position and front-of-queue probability
        4. Compute queue imbalance (bid vs ask survival asymmetry)
        5. Compute effective queue depth (displayed * survival)
        6. Estimate queue age distribution
        """
        timestamp = snapshot.timestamp
        
        if not snapshot.bids or not snapshot.asks:
            return self._empty_metrics(timestamp)
        
        # Step 1: Update rates from buffer
        self._update_rates_from_buffer(buffer, timestamp)
        
        # Step 2: Compute queue survival probability
        # P(survive T) = exp(-(cancel_rate + fill_rate) * T)
        settlement_time = self.config.queue_survival_decay_rate * 100  # Time horizon
        
        bid_departure_rate = self._bid_cancel_rate + self._bid_fill_rate
        ask_departure_rate = self._ask_cancel_rate + self._ask_fill_rate
        
        bid_survival = np.exp(-bid_departure_rate * settlement_time)
        ask_survival = np.exp(-ask_departure_rate * settlement_time)
        
        # Smooth survival estimates
        self._bid_queue_survival = 0.7 * bid_survival + 0.3 * self._bid_queue_survival
        self._ask_queue_survival = 0.7 * ask_survival + 0.3 * self._ask_queue_survival
        
        # Step 3: Queue position estimation
        # Position in queue depends on order count and fill rate
        bid_order_count = sum(l.order_count for l in snapshot.bids[:5])
        ask_order_count = sum(l.order_count for l in snapshot.asks[:5])
        
        bid_queue_position = self._estimate_queue_position(
            bid_order_count, self._bid_fill_rate, self._bid_cancel_rate
        )
        ask_queue_position = self._estimate_queue_position(
            ask_order_count, self._ask_fill_rate, self._ask_cancel_rate
        )
        
        # Step 4: Front-of-queue probability
        bid_front_prob = self._compute_front_of_queue_probability(
            bid_order_count, self._bid_fill_rate
        )
        ask_front_prob = self._compute_front_of_queue_probability(
            ask_order_count, self._ask_fill_rate
        )
        
        # Step 5: Queue imbalance
        total_survival = self._bid_queue_survival + self._ask_queue_survival
        if total_survival > 0:
            queue_imbalance = (self._bid_queue_survival - self._ask_queue_survival) / total_survival
        else:
            queue_imbalance = 0.0
        
        # Step 6: Effective queue depth
        depth = self.config.orderbook_depth_levels
        bid_displayed_depth = sum(l.size for l in snapshot.bids[:depth])
        ask_displayed_depth = sum(l.size for l in snapshot.asks[:depth])
        
        effective_bid_depth = bid_displayed_depth * self._bid_queue_survival
        effective_ask_depth = ask_displayed_depth * self._ask_queue_survival
        
        # Step 7: Queue decay rate
        bid_decay_rate = bid_departure_rate
        ask_decay_rate = ask_departure_rate
        
        # Step 8: Queue age distribution estimate
        # Average age inversely proportional to departure rate
        if bid_departure_rate > 0:
            avg_bid_age = 1.0 / bid_departure_rate
        else:
            avg_bid_age = 60.0  # Default: 1 minute
        
        if ask_departure_rate > 0:
            avg_ask_age = 1.0 / ask_departure_rate
        else:
            avg_ask_age = 60.0
        
        self._avg_bid_order_age = 0.7 * avg_bid_age + 0.3 * self._avg_bid_order_age
        self._avg_ask_order_age = 0.7 * avg_ask_age + 0.3 * self._avg_ask_order_age
        
        # Combined queue survival probability (average of both sides)
        combined_survival = (self._bid_queue_survival + self._ask_queue_survival) / 2.0
        
        # Build result
        result = QueueMetrics(
            timestamp=timestamp,
            queue_survival_probability=combined_survival,
            queue_position_estimate=(bid_queue_position + ask_queue_position) / 2.0,
            cancellation_rate=(self._bid_cancel_rate + self._ask_cancel_rate) / 2.0,
            fill_rate=(self._bid_fill_rate + self._ask_fill_rate) / 2.0,
            queue_decay_rate=(bid_decay_rate + ask_decay_rate) / 2.0,
            queue_imbalance=queue_imbalance,
            effective_queue_depth=effective_bid_depth + effective_ask_depth,
            queue_age_distribution=(self._avg_bid_order_age + self._avg_ask_order_age) / 2.0,
            front_of_queue_probability=(bid_front_prob + ask_front_prob) / 2.0,
        )
        
        self._prev_metrics = result
        self._last_update_time = timestamp
        self._initialized = True
        
        return result
    
    def _update_rates_from_buffer(self, buffer: Optional[OrderbookBuffer],
                                   timestamp: float):
        """
        Update cancellation and fill rates from orderbook delta history.
        
        Rates are computed as volume per second over recent window.
        Separate tracking for bid and ask sides for queue imbalance.
        """
        if buffer is None:
            return
        
        recent_deltas = buffer.recent_deltas(300)
        if len(recent_deltas) < 5:
            return
        
        time_span = recent_deltas[-1].timestamp - recent_deltas[0].timestamp
        if time_span < 1.0:
            time_span = 1.0
        
        bid_cancel_vol = 0.0
        ask_cancel_vol = 0.0
        bid_fill_vol = 0.0
        ask_fill_vol = 0.0
        
        for delta in recent_deltas:
            if delta.is_trade:
                # Trade execution = fill
                if delta.side == "bid":
                    bid_fill_vol += abs(delta.size_delta)
                else:
                    ask_fill_vol += abs(delta.size_delta)
            elif delta.size_delta < 0:
                # Cancellation
                if delta.side == "bid":
                    bid_cancel_vol += abs(delta.size_delta)
                else:
                    ask_cancel_vol += abs(delta.size_delta)
        
        # Compute rates (volume per second)
        self._bid_cancel_rate = bid_cancel_vol / time_span
        self._ask_cancel_rate = ask_cancel_vol / time_span
        self._bid_fill_rate = bid_fill_vol / time_span
        self._ask_fill_rate = ask_fill_vol / time_span
    
    def _estimate_queue_position(self, order_count: int, 
                                  fill_rate: float, cancel_rate: float) -> float:
        """
        Estimate expected queue position for a new order.
        
        Queue position = expected number of orders ahead at time of arrival.
        
        E[position] = sum_{i=1}^{N} P(order_i survives until our order fills)
                     = N * P(survive) where P(survive) depends on fill and cancel rates
        
        Simplified model:
        position = order_count * (1 - fill_rate / (fill_rate + cancel_rate))
        
        This accounts for the fact that some orders ahead will be cancelled
        before they're filled, improving our effective position.
        """
        if order_count == 0:
            return 0.0
        
        total_rate = fill_rate + cancel_rate
        if total_rate > 0:
            # Fraction of orders that will be filled (not cancelled)
            fill_fraction = fill_rate / total_rate
            # Expected position: orders ahead that will actually need to be filled
            position = order_count * fill_fraction
        else:
            position = float(order_count)
        
        return max(0.0, position)
    
    def _compute_front_of_queue_probability(self, order_count: int,
                                             fill_rate: float) -> float:
        """
        Compute probability of being at or near the front of the queue.
        
        P(front) = 1 / (1 + effective_queue_length * fill_rate * T)
        
        where T is the expected time to fill and effective_queue_length
        accounts for cancellations ahead.
        
        Higher fill_rate and fewer orders = higher front probability.
        """
        if order_count <= 1:
            return 1.0  # Already at front
        
        if fill_rate <= 0:
            return 0.1 / order_count  # Very low probability without fills
        
        # Effective queue length (orders that will actually need to fill ahead)
        effective_length = max(1, order_count * 0.5)  # Rough estimate
        
        # Probability of being within top 3 positions
        p_front = 3.0 / max(effective_length, 3.0)
        
        return max(0.01, min(1.0, p_front))
    
    def _empty_metrics(self, timestamp: float) -> QueueMetrics:
        """Return empty metrics when no data available."""
        return QueueMetrics(
            timestamp=timestamp,
            queue_survival_probability=0.5,
            queue_position_estimate=0.0,
            cancellation_rate=0.0,
            fill_rate=0.0,
            queue_decay_rate=0.0,
            queue_imbalance=0.0,
            effective_queue_depth=0.0,
            queue_age_distribution=60.0,
            front_of_queue_probability=0.5,
        )