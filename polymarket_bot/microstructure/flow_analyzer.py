"""
Taker aggression and order flow analyzer.

Mathematical foundation:
- Flow imbalance: I = (V_buy - V_sell) / (V_buy + V_sell)
- Taker aggression score: normalized ratio of aggressive taker volume
- Flow acceleration: dI/dt measured via finite differences with smoothing
- Cumulative flow delta: running sum of signed volume for persistence detection
- Flow persistence: autocorrelation of imbalance series (momentum indicator)
- Taker inversion: rapid reversal in taker direction (reversal signal)
- Flow decay rate: exponential decay parameter of flow signal predictive power

All computations use exponential smoothing for stability and regime adaptation.
"""

import logging
import time
import numpy as np
from typing import Optional, List, Deque
from collections import deque

from polymarket_bot.config import MicrostructureConfig
from polymarket_bot.bot_types import TickData, TickBuffer, FlowMetrics

logger = logging.getLogger(__name__)


class FlowAnalyzer:
    """
    Analyzes executed order flow and taker aggression patterns.
    
    Key innovations:
    1. Taker aggression modeling with directional decomposition
    2. Flow acceleration detection (rate of change of imbalance)
    3. Flow persistence via autocorrelation (momentum continuation)
    4. Taker inversion detection (fast reversal signal)
    5. Flow decay rate estimation (signal half-life)
    """
    
    def __init__(self, config: MicrostructureConfig):
        self.config = config
        
        # Running accumulators with exponential decay
        self._taker_buy_volume_ema = 0.0
        self._taker_sell_volume_ema = 0.0
        self._maker_volume_ema = 0.0
        self._total_volume_ema = 0.0
        self._trade_count_window = 0
        self._large_trade_count_window = 0
        
        # Flow imbalance tracking
        self._flow_imbalance_ema = 0.0
        self._prev_flow_imbalance = 0.0
        self._flow_acceleration = 0.0
        self._cumulative_flow_delta = 0.0
        
        # Taker ratio tracking
        self._taker_ratio_ema = 0.0
        
        # Taker inversion detection
        self._recent_taker_directions: Deque[float] = deque(maxlen=config.taker_aggression_window)
        self._taker_inversion_score = 0.0
        
        # Flow persistence (autocorrelation)
        self._imbalance_history: Deque[float] = deque(maxlen=200)
        self._flow_persistence = 0.0
        
        # Flow decay rate estimation
        self._flow_decay_rate = 0.05  # Initial estimate
        self._imbalance_prediction_errors: Deque[float] = deque(maxlen=100)
        
        # Large trade detection threshold (adaptive)
        self._avg_trade_size_ema = 0.0
        self._large_trade_threshold_multiplier = 3.0
        
        # Aggression score
        self._taker_aggression_ema = 0.0
        
        # Time tracking
        self._last_timestamp = 0.0
        self._window_start_time = 0.0
        self._initialized = False
        
        # Smoothing parameters
        self._imbalance_smoothing = config.flow_imbalance_smoothing
        self._volume_smoothing = 0.90  # Volume EMA smoothing
    
    def update(self, tick: TickData) -> Optional[FlowMetrics]:
        """
        Update flow metrics with a new tick.
        
        Returns None if insufficient data, otherwise returns updated FlowMetrics.
        """
        timestamp = tick.timestamp
        
        # Initialize on first tick
        if not self._initialized:
            self._last_timestamp = timestamp
            self._window_start_time = timestamp
            self._avg_trade_size_ema = tick.volume
            self._initialized = True
            return None
        
        # Compute time delta
        dt = timestamp - self._last_timestamp
        if dt <= 0:
            dt = 0.001  # Prevent zero/negative time
        
        # Decay factor for time-based EMA
        # EMA with time-based decay: alpha = 1 - exp(-lambda * dt)
        decay_rate = 0.1  # Decay per second
        alpha_volume = 1.0 - np.exp(-decay_rate * dt)
        alpha_imbalance = 1.0 - np.exp(-self.config.flow_imbalance_smoothing * dt)
        
        # Classify trade: taker buy or taker sell
        # In Binance convention: m=True means maker sold (taker bought)
        is_taker_buy = tick.side == "buy"
        is_taker_sell = tick.side == "sell"
        
        # Update volume EMAs
        if is_taker_buy:
            self._taker_buy_volume_ema = (1 - alpha_volume) * self._taker_buy_volume_ema + alpha_volume * tick.volume
        elif is_taker_sell:
            self._taker_sell_volume_ema = (1 - alpha_volume) * self._taker_sell_volume_ema + alpha_volume * tick.volume
        
        self._total_volume_ema = self._taker_buy_volume_ema + self._taker_sell_volume_ema
        
        # Update trade count
        self._trade_count_window += 1
        
        # Large trade detection (adaptive threshold)
        self._avg_trade_size_ema = (1 - alpha_volume) * self._avg_trade_size_ema + alpha_volume * tick.volume
        large_threshold = self._avg_trade_size_ema * self._large_trade_threshold_multiplier
        if tick.volume > large_threshold:
            self._large_trade_count_window += 1
        
        # Compute instantaneous flow imbalance
        total_taker = self._taker_buy_volume_ema + self._taker_sell_volume_ema
        if total_taker > 0:
            instant_imbalance = (self._taker_buy_volume_ema - self._taker_sell_volume_ema) / total_taker
        else:
            instant_imbalance = 0.0
        
        # Update flow imbalance EMA
        self._flow_imbalance_ema = (1 - alpha_imbalance) * self._flow_imbalance_ema + alpha_imbalance * instant_imbalance
        
        # Compute flow acceleration: d(imbalance)/dt
        if dt > 0:
            raw_acceleration = (self._flow_imbalance_ema - self._prev_flow_imbalance) / dt
            # Smooth acceleration
            self._flow_acceleration = 0.7 * raw_acceleration + 0.3 * self._flow_acceleration
        
        # Update cumulative flow delta (signed volume accumulator with decay)
        signed_volume = tick.volume if is_taker_buy else -tick.volume
        # Apply exponential decay to cumulative delta
        decay_factor = np.exp(-self._flow_decay_rate * dt)
        self._cumulative_flow_delta = self._cumulative_flow_delta * decay_factor + signed_volume
        
        # Update taker ratio
        if total_taker > 0:
            self._taker_ratio_ema = (1 - alpha_volume) * self._taker_ratio_ema + alpha_volume * (self._taker_buy_volume_ema / total_taker)
        
        # Track taker direction for inversion detection
        direction = 1.0 if is_taker_buy else -1.0
        self._recent_taker_directions.append(direction * tick.volume)
        
        # Compute taker inversion score
        self._compute_taker_inversion()
        
        # Update imbalance history for persistence computation
        self._imbalance_history.append(self._flow_imbalance_ema)
        
        # Compute flow persistence (autocorrelation at lag 1)
        self._compute_flow_persistence()
        
        # Compute flow decay rate
        self._update_flow_decay_rate()
        
        # Compute taker aggression score
        self._compute_taker_aggression()
        
        # Store previous imbalance for acceleration
        self._prev_flow_imbalance = self._flow_imbalance_ema
        self._last_timestamp = timestamp
        
        # Build FlowMetrics
        large_trade_ratio = self._large_trade_count_window / max(self._trade_count_window, 1)
        avg_trade_size = self._avg_trade_size_ema
        
        return FlowMetrics(
            timestamp=timestamp,
            taker_buy_volume=self._taker_buy_volume_ema,
            taker_sell_volume=self._taker_sell_volume_ema,
            taker_ratio=self._taker_ratio_ema,
            taker_aggression_score=self._taker_aggression_ema,
            flow_imbalance=self._flow_imbalance_ema,
            flow_acceleration=self._flow_acceleration,
            cumulative_flow_delta=self._cumulative_flow_delta,
            maker_volume=self._maker_volume_ema,
            total_volume=self._total_volume_ema,
            trade_count=self._trade_count_window,
            avg_trade_size=avg_trade_size,
            large_trade_count=self._large_trade_count_window,
            large_trade_ratio=large_trade_ratio,
            taker_inversion=self._taker_inversion_score,
            flow_persistence=self._flow_persistence,
            flow_decay_rate=self._flow_decay_rate,
        )
    
    def compute_from_buffer(self, buffer: TickBuffer) -> Optional[FlowMetrics]:
        """
        Compute flow metrics from a tick buffer (batch processing).
        Useful for initialization or when processing historical data.
        """
        if len(buffer) < 10:
            return None
        
        timestamps, prices, volumes = buffer.recent_slice(200)
        if len(timestamps) < 10:
            return None
        
        # We need side information which isn't in the numpy arrays
        # This method is for when we have the full deque available
        total_buy = 0.0
        total_sell = 0.0
        total_volume = 0.0
        trade_count = 0
        large_trades = 0
        
        sides = list(buffer.sides)
        n = min(len(sides), len(volumes), 200)
        recent_sides = sides[-n:]
        recent_volumes = volumes[-n:]
        
        avg_size = float(np.mean(recent_volumes)) if n > 0 else 0.0
        large_threshold = avg_size * self._large_trade_threshold_multiplier
        
        for i in range(n):
            vol = float(recent_volumes[i])
            side = recent_sides[i]
            total_volume += vol
            trade_count += 1
            
            if side == "buy":
                total_buy += vol
            else:
                total_sell += vol
            
            if vol > large_threshold:
                large_trades += 1
        
        if total_volume > 0:
            imbalance = (total_buy - total_sell) / total_volume
            taker_ratio = total_buy / total_volume
        else:
            imbalance = 0.0
            taker_ratio = 0.5
        
        # Compute autocorrelation for persistence
        if n > 5:
            # Create signed volume series
            signed_vols = np.zeros(n)
            for i in range(n):
                sign = 1.0 if recent_sides[i] == "buy" else -1.0
                signed_vols[i] = sign * recent_volumes[i]
            
            # Autocorrelation at lag 1
            if np.std(signed_vols) > 0:
                persistence = np.corrcoef(signed_vols[:-1], signed_vols[1:])[0, 1]
                persistence = max(-1.0, min(1.0, persistence))
            else:
                persistence = 0.0
        else:
            persistence = 0.0
        
        return FlowMetrics(
            timestamp=timestamps[-1] if len(timestamps) > 0 else time.time(),
            taker_buy_volume=total_buy,
            taker_sell_volume=total_sell,
            taker_ratio=taker_ratio,
            taker_aggression_score=0.0,  # Needs real-time tracking
            flow_imbalance=imbalance,
            flow_acceleration=0.0,  # Needs sequential updates
            cumulative_flow_delta=total_buy - total_sell,
            maker_volume=0.0,
            total_volume=total_volume,
            trade_count=trade_count,
            avg_trade_size=avg_size,
            large_trade_count=large_trades,
            large_trade_ratio=large_trades / max(trade_count, 1),
            taker_inversion=0.0,  # Needs sequential tracking
            flow_persistence=persistence,
            flow_decay_rate=self._flow_decay_rate,
        )
    
    def _compute_taker_inversion(self):
        """
        Detect taker direction inversion - a key reversal signal.
        
        Methodology:
        1. Compute weighted average direction over recent window
        2. Compare current direction to weighted average
        3. If current direction is opposite and strong, flag as inversion
        
        Inversion score = |current_direction - recent_avg_direction|
        normalized to [0, 1] range.
        
        This detects when taker flow suddenly reverses direction,
        which is a strong microstructure reversal signal.
        """
        if len(self._recent_taker_directions) < 5:
            self._taker_inversion_score = 0.0
            return
        
        # Split into two halves: recent and older
        n = len(self._recent_taker_directions)
        mid = n // 2
        
        older = list(self._recent_taker_directions)[:mid]
        recent = list(self._recent_taker_directions)[mid:]
        
        # Weighted average direction for each half
        older_avg = np.mean(older) if older else 0.0
        recent_avg = np.mean(recent) if recent else 0.0
        
        # Normalize by total volume in each half
        older_total = np.sum(np.abs(older)) if older else 1.0
        recent_total = np.sum(np.abs(recent)) if recent else 1.0
        
        older_normalized = older_avg / max(older_total, 1e-10)
        recent_normalized = recent_avg / max(recent_total, 1e-10)
        
        # Inversion score: how much direction has changed
        # Scale by the strength of the recent direction
        direction_change = abs(recent_normalized - older_normalized)
        recent_strength = abs(recent_normalized)
        
        # Inversion is significant only if recent direction is strong AND opposite
        if older_normalized * recent_normalized < 0:  # Opposite directions
            self._taker_inversion_score = min(1.0, direction_change * recent_strength * 4.0)
        else:
            # Same direction but weakening
            self._taker_inversion_score = min(0.5, direction_change * 0.5)
    
    def _compute_flow_persistence(self):
        """
        Compute flow persistence via autocorrelation of imbalance series.
        
        Persistence = autocorrelation at lag 1 of flow_imbalance series.
        
        High persistence (>0.5): momentum continuation likely
        Low persistence (<0.2): flow is random/noisy, no directional edge
        Negative persistence: oscillating flow, potential reversal
        
        Uses exponentially weighted autocorrelation for stability.
        """
        if len(self._imbalance_history) < 10:
            self._flow_persistence = 0.0
            return
        
        imbalances = np.array(list(self._imbalance_history)[-50:])
        n = len(imbalances)
        
        if n < 5:
            self._flow_persistence = 0.0
            return
        
        # Compute lag-1 autocorrelation
        mean_imb = np.mean(imbalances)
        var_imb = np.var(imbalances)
        
        if var_imb < 1e-10:
            self._flow_persistence = 0.0
            return
        
        cov_lag1 = np.mean((imbalances[:-1] - mean_imb) * (imbalances[1:] - mean_imb))
        autocorr = cov_lag1 / var_imb
        
        # Clamp to [-1, 1]
        autocorr = max(-1.0, min(1.0, autocorr))
        
        # Smooth with previous estimate
        self._flow_persistence = 0.7 * autocorr + 0.3 * self._flow_persistence
    
    def _update_flow_decay_rate(self):
        """
        Estimate the decay rate of flow signal predictive power.
        
        Methodology:
        The flow imbalance signal has a finite half-life for predicting
        future price direction. We estimate this by measuring how quickly
        the imbalance prediction error grows.
        
        decay_rate = -log(persistence) / dt
        
        If persistence is high, decay is slow (signal lasts longer).
        If persistence is low, decay is fast (signal fades quickly).
        
        This is used to weight how much we trust current flow signals
        for future prediction.
        """
        if self._flow_persistence > 0.01:
            # Decay rate from persistence: lambda = -ln(persistence)
            implied_decay = -np.log(max(self._flow_persistence, 0.01))
            # Smooth update
            self._flow_decay_rate = 0.8 * implied_decay + 0.2 * self._flow_decay_rate
        else:
            # Very low persistence = fast decay
            self._flow_decay_rate = min(1.0, self._flow_decay_rate * 1.1)
        
        # Clamp to reasonable range
        self._flow_decay_rate = max(0.01, min(1.0, self._flow_decay_rate))
    
    def _compute_taker_aggression(self):
        """
        Compute taker aggression score.
        
        Aggression measures how urgently takers are executing:
        - High aggression: takers hitting bids/asks aggressively (strong direction)
        - Low aggression: takers picking limit orders patiently (weak direction)
        
        Score components:
        1. Taker ratio: fraction of volume from taker side
        2. Trade size: larger trades = more aggressive
        3. Trade frequency: more trades in short time = more aggressive
        4. Flow imbalance strength: strong imbalance = aggressive directional flow
        
        aggression = weighted_combination(taker_ratio, size_factor, frequency_factor, imbalance_strength)
        """
        # Taker ratio component (already computed)
        taker_ratio = self._taker_ratio_ema
        
        # Size factor: how much larger than average
        avg_size = max(self._avg_trade_size_ema, 1e-10)
        recent_large_ratio = self._large_trade_count_window / max(self._trade_count_window, 1)
        size_factor = min(1.0, recent_large_ratio * 5.0)  # Scale up
        
        # Imbalance strength
        imbalance_strength = abs(self._flow_imbalance_ema)
        
        # Combined aggression score
        # Weight: taker_ratio (30%), size (20%), imbalance (50%)
        aggression = 0.3 * abs(taker_ratio - 0.5) * 2.0 + \
                     0.2 * size_factor + \
                     0.5 * imbalance_strength
        
        # Normalize to [0, 1]
        aggression = max(0.0, min(1.0, aggression))
        
        # Smooth
        self._taker_aggression_ema = 0.7 * aggression + 0.3 * self._taker_aggression_ema
    
    def reset(self):
        """Reset all accumulators for a new settlement period."""
        self._taker_buy_volume_ema = 0.0
        self._taker_sell_volume_ema = 0.0
        self._maker_volume_ema = 0.0
        self._total_volume_ema = 0.0
        self._trade_count_window = 0
        self._large_trade_count_window = 0
        self._flow_imbalance_ema = 0.0
        self._prev_flow_imbalance = 0.0
        self._flow_acceleration = 0.0
        self._cumulative_flow_delta = 0.0
        self._taker_ratio_ema = 0.0
        self._recent_taker_directions.clear()
        self._taker_inversion_score = 0.0
        self._imbalance_history.clear()
        self._flow_persistence = 0.0
        self._taker_aggression_ema = 0.0
        self._initialized = False
