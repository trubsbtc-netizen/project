"""
Entropy-aware signal suppression and decorrelation engine.

Mathematical foundation:
- Shannon entropy: H = -sum(p_i * log(p_i)) for signal probability distribution
  Applied to recent directional probability series to measure signal randomness
- Signal decorrelation: correlation between consecutive signals
  High correlation = redundant signals (suppress)
  Low correlation = novel signals (allow)
- Suppression factor: 1 - entropy * threshold, applied to signal strength
  High entropy (>threshold): suppress signal (noisy/random market)
  Low entropy (<threshold): allow signal (clear directional information)
- Novelty score: how different current signal is from recent signal history
  novelty = 1 - correlation_with_history
  High novelty: new information (allow)
  Low novelty: redundant information (suppress)

The entropy filter prevents the bot from:
1. Trading on noisy/random signals (high entropy)
2. Trading on redundant/correlated signals (low novelty)
3. Over-trading in uncertain market conditions
4. Being fooled by repetitive microstructure patterns

This is critical for avoiding gambling behavior and ensuring
each trade is based on genuinely new, high-quality directional information.
"""

import logging
import time
import numpy as np
from typing import Optional, List
from collections import deque

from polymarket_bot.config import SignalConfig, Direction
from polymarket_bot.bot_types import (
    EntropyFilterResult, ContinuationSignal, ReversalSignal, SignalBuffer
)

logger = logging.getLogger(__name__)


class EntropyFilter:
    """
    Entropy-aware signal suppression and decorrelation filter.
    
    Key innovations:
    1. Shannon entropy of directional probability series
    2. Signal decorrelation via autocorrelation analysis
    3. Novelty scoring via correlation with signal history
    4. Adaptive suppression factor based on regime entropy
    5. Signal quality gating (suppress low-quality, redundant signals)
    """
    
    def __init__(self, config: SignalConfig):
        self.config = config
        
        # Probability history for entropy computation
        self._probability_history: deque = deque(maxlen=config.entropy_window)
        
        # Signal strength history for decorrelation
        self._signal_strength_history: deque = deque(maxlen=config.decorrelation_window)
        
        # Direction history for pattern detection
        self._direction_history: deque = deque(maxlen=config.decorrelation_window)
        
        # Entropy tracking
        self._entropy_ema = 0.0
        self._suppression_factor_ema = 1.0
        
        # Decorrelation tracking
        self._correlation_ema = 0.0
        self._novelty_ema = 1.0
        
        # Previous result
        self._prev_result: Optional[EntropyFilterResult] = None
    
    def filter_signal(
        self,
        raw_signal_strength: float,
        direction_probability: float,
        direction: Direction,
        continuation_signal: Optional[ContinuationSignal] = None,
        reversal_signal: Optional[ReversalSignal] = None,
    ) -> EntropyFilterResult:
        """
        Apply entropy-aware filtering to a signal.
        
        Process:
        1. Compute entropy of recent probability series
        2. Compute decorrelation score (signal redundancy)
        3. Compute novelty score (signal originality)
        4. Compute suppression factor
        5. Apply suppression to signal strength
        6. Determine if signal should be fully suppressed
        """
        timestamp = time.time()
        
        # Step 1: Update probability history
        self._probability_history.append(direction_probability)
        
        # Step 2: Update signal strength history
        self._signal_strength_history.append(raw_signal_strength)
        
        # Step 3: Update direction history
        self._direction_history.append(direction.value)
        
        # Step 4: Compute entropy
        entropy_score = self._compute_entropy()
        
        # Step 5: Compute decorrelation
        decorrelation_score, correlation_with_history = self._compute_decorrelation()
        
        # Step 6: Compute novelty
        novelty_score = self._compute_novelty(raw_signal_strength, direction_probability)
        
        # Step 7: Compute suppression factor
        # suppression = 1 - max(entropy_excess, correlation_excess)
        # where excess = max(0, score - threshold)
        entropy_excess = max(0, entropy_score - self.config.entropy_suppression_threshold)
        correlation_excess = max(0, correlation_with_history - self.config.decorrelation_correlation_threshold)
        
        # Combined suppression
        suppression_factor = 1.0 - max(entropy_excess, correlation_excess)
        suppression_factor = max(0.0, min(1.0, suppression_factor))
        
        # Step 8: Apply suppression
        effective_signal_strength = raw_signal_strength * suppression_factor
        
        # Step 9: Determine if fully suppressed
        is_suppressed = (
            suppression_factor < 0.2 or  # Very high suppression
            entropy_score > 0.9 or  # Very high entropy (random market)
            effective_signal_strength < 0.1  # Signal too weak after suppression
        )
        
        # Smooth suppression factor
        self._suppression_factor_ema = 0.7 * suppression_factor + 0.3 * self._suppression_factor_ema
        
        # Build result
        result = EntropyFilterResult(
            timestamp=timestamp,
            raw_signal_strength=raw_signal_strength,
            entropy_score=entropy_score,
            decorrelation_score=decorrelation_score,
            suppression_factor=self._suppression_factor_ema,
            is_suppressed=is_suppressed,
            effective_signal_strength=effective_signal_strength,
            correlation_with_history=correlation_with_history,
            novelty_score=novelty_score,
        )
        
        self._prev_result = result
        return result
    
    def _compute_entropy(self) -> float:
        """
        Compute Shannon entropy of recent directional probability series.
        
        H = -sum(p_i * log(p_i)) where p_i are discretized probability values
        
        Method:
        1. Discretize probabilities into bins
        2. Compute frequency distribution
        3. Compute Shannon entropy of the distribution
        
        High entropy: probabilities are spread across many bins (uncertain market)
        Low entropy: probabilities concentrated in few bins (clear direction)
        
        Maximum entropy = log(n_bins) for uniform distribution
        Minimum entropy = 0 for single-bin concentration
        """
        if len(self._probability_history) < 10:
            return 0.5  # Default moderate entropy
        
        probs = np.array(list(self._probability_history))
        
        # Discretize into bins
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        discretized = np.digitize(probs, bins)
        
        # Compute frequency distribution
        counts = np.bincount(discretized, minlength=n_bins + 1)
        counts = counts[1:n_bins + 1]  # Remove the 0 bin edge case
        
        # Normalize to probabilities
        total = np.sum(counts)
        if total == 0:
            return 0.5
        
        freq_probs = counts / total
        
        # Shannon entropy
        # Filter out zero probabilities
        freq_probs = freq_probs[freq_probs > 0]
        entropy = -np.sum(freq_probs * np.log(freq_probs))
        
        # Normalize by maximum entropy (log(n_bins))
        max_entropy = np.log(n_bins)
        normalized_entropy = entropy / max_entropy
        
        # Smooth
        self._entropy_ema = 0.7 * normalized_entropy + 0.3 * self._entropy_ema
        
        return self._entropy_ema
    
    def _compute_decorrelation(self) -> tuple:
        """
        Compute signal decorrelation score and correlation with history.
        
        Decorrelation measures how different consecutive signals are:
        - High decorrelation: signals are diverse (good, each is new information)
        - Low decorrelation: signals are similar (bad, redundant information)
        
        Correlation with history: how much current signal resembles recent signals.
        - High correlation: current signal is similar to recent (redundant)
        - Low correlation: current signal is different from recent (novel)
        
        Returns: (decorrelation_score, correlation_with_history)
        """
        if len(self._signal_strength_history) < 5:
            return (1.0, 0.0)  # Default: high decorrelation, low correlation
        
        strengths = np.array(list(self._signal_strength_history))
        
        if len(strengths) < 3:
            return (1.0, 0.0)
        
        # Compute autocorrelation at lag 1
        mean_s = np.mean(strengths)
        var_s = np.var(strengths)
        
        if var_s < 1e-10:
            # All signals are the same strength = maximum correlation
            return (0.0, 1.0)
        
        cov_lag1 = np.mean((strengths[:-1] - mean_s) * (strengths[1:] - mean_s))
        autocorr = cov_lag1 / var_s
        autocorr = max(-1.0, min(1.0, autocorr))
        
        # Decorrelation = 1 - |autocorrelation|
        decorrelation = 1.0 - abs(autocorr)
        
        # Correlation with history = |autocorrelation|
        correlation_with_history = abs(autocorr)
        
        # Smooth
        self._correlation_ema = 0.7 * correlation_with_history + 0.3 * self._correlation_ema
        self._novelty_ema = 0.7 * decorrelation + 0.3 * self._novelty_ema
        
        return (self._novelty_ema, self._correlation_ema)
    
    def _compute_novelty(self, current_strength: float,
                          current_probability: float) -> float:
        """
        Compute novelty score: how different current signal is from recent history.
        
        novelty = 1 - similarity(current, recent_average)
        
        similarity computed as:
        1. Direction change: different direction from recent = more novel
        2. Probability change: different probability from recent = more novel
        3. Strength change: different strength from recent = more novel
        
        High novelty: genuinely new directional information
        Low novelty: same signal repeated (redundant, should suppress)
        """
        if len(self._probability_history) < 5 or len(self._signal_strength_history) < 5:
            return 1.0  # Default: novel
        
        # Recent averages
        recent_probs = np.array(list(self._probability_history)[-10:])
        recent_strengths = np.array(list(self._signal_strength_history)[-10:])
        
        avg_prob = np.mean(recent_probs)
        avg_strength = np.mean(recent_strengths)
        
        # Probability novelty
        prob_novelty = abs(current_probability - avg_prob) / max(abs(avg_prob), 0.1)
        prob_novelty = min(1.0, prob_novelty)
        
        # Strength novelty
        strength_novelty = abs(current_strength - avg_strength) / max(abs(avg_strength), 0.1)
        strength_novelty = min(1.0, strength_novelty)
        
        # Direction novelty (check if direction changed)
        recent_directions = list(self._direction_history)[-5:]
        if len(recent_directions) > 0:
            # Count how many recent signals had the same direction
            same_direction_count = sum(
                1 for d in recent_directions 
                if d == self._direction_history[-1] if len(self._direction_history) > 0
            )
            direction_novelty = 1.0 - (same_direction_count / len(recent_directions))
        else:
            direction_novelty = 1.0
        
        # Combined novelty
        novelty = 0.4 * prob_novelty + 0.3 * strength_novelty + 0.3 * direction_novelty
        
        return min(1.0, novelty)