"""
Signal Engine for Polymarket Trading Bot

Computes directional trading signals from:
- Orderflow imbalance
- Spread dynamics
- Liquidity shifts
- Momentum
- External price confirmation
"""

import asyncio
import time
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Tuple
from decimal import Decimal
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """Result of a single signal computation."""
    name: str
    value: float  # Normalized to [-1, 1] or [0, 1] depending on signal
    confidence: float  # [0, 1] how reliable this signal is
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_bullish(self) -> bool:
        return self.value > 0.5 if self.name != 'orderflow' else self.value > 0
    
    def is_bearish(self) -> bool:
        return self.value < 0.5 if self.name != 'orderflow' else self.value < 0


@dataclass
class MarketData:
    """Aggregated market data for signal computation."""
    # Orderbook data
    best_bid: Decimal
    best_ask: Decimal
    mid_price: Decimal
    spread_pct: float
    bid_levels: List[Tuple[Decimal, Decimal]]  # (price, size)
    ask_levels: List[Tuple[Decimal, Decimal]]
    
    # Recent trades
    recent_trades: List[Dict[str, Any]]
    
    # External prices
    binance_price: Optional[Decimal]
    coinbase_price: Optional[Decimal]
    
    # RTDS data
    rtds_reference_price: Optional[Decimal]
    rtds_time_to_expiry: Optional[float]
    
    # Timestamps
    last_update_time: float


class PriceHistory:
    """Maintains rolling price history for momentum calculations."""
    
    def __init__(self, max_periods: int = 100):
        self._prices: deque = deque(maxlen=max_periods)
        self._timestamps: deque = deque(maxlen=max_periods)
    
    def add(self, price: float, timestamp: float):
        self._prices.append(price)
        self._timestamps.append(timestamp)
    
    def get_returns(self, periods: int = 5) -> List[float]:
        """Get log returns for specified periods."""
        if len(self._prices) < periods + 1:
            return []
        
        returns = []
        prices_list = list(self._prices)
        
        for i in range(periods, len(prices_list)):
            if prices_list[i - periods] > 0:
                log_return = math.log(prices_list[i] / prices_list[i - periods])
                returns.append(log_return)
        
        return returns
    
    def get_volatility(self, periods: int = 10) -> Optional[float]:
        """Calculate rolling volatility (std dev of returns)."""
        returns = self.get_returns(periods)
        
        if len(returns) < 2:
            return None
        
        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
        
        return math.sqrt(variance)
    
    def get_momentum(self, periods: int = 5) -> Optional[float]:
        """Calculate momentum as sum of recent returns."""
        returns = self.get_returns(periods)
        
        if not returns:
            return None
        
        return sum(returns)


class SignalEngine:
    """
    Computes directional trading signals with:
    - Multiple signal types
    - Confidence scoring
    - Signal decay
    - Noise filtering
    """
    
    def __init__(
        self,
        weights: Dict[str, float],
        momentum_lookback: int = 5,
        volume_ma_periods: int = 20,
        volatility_lookback: int = 10,
        signal_decay_lambda: float = 0.1,
        min_depth_threshold: Decimal = Decimal('100'),
    ):
        self.weights = weights
        self.momentum_lookback = momentum_lookback
        self.volume_ma_periods = volume_ma_periods
        self.volatility_lookback = volatility_lookback
        self.signal_decay_lambda = signal_decay_lambda
        self.min_depth_threshold = min_depth_threshold
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.normalized_weights = {k: v / total_weight for k, v in weights.items()}
        else:
            self.normalized_weights = weights
        
        # Price history per market
        self._price_history: Dict[str, PriceHistory] = {}
        
        # Volume history for MA
        self._volume_history: Dict[str, deque] = {}
        
        # Previous signal values for decay
        self._previous_signals: Dict[str, Dict[str, float]] = {}
        
        # Metrics
        self._signal_count = 0
        self._computation_times: deque = deque(maxlen=100)
    
    def compute_all_signals(
        self,
        market_id: str,
        data: MarketData,
    ) -> Dict[str, SignalResult]:
        """Compute all signals for a market."""
        start_time = time.perf_counter()
        
        # Initialize price history if needed
        if market_id not in self._price_history:
            self._price_history[market_id] = PriceHistory()
            self._volume_history[market_id] = deque(maxlen=self.volume_ma_periods)
            self._previous_signals[market_id] = {}
        
        # Update price history
        if data.mid_price > 0:
            self._price_history[market_id].add(float(data.mid_price), data.last_update_time)
        
        # Compute individual signals
        signals = {}
        
        signals['orderflow'] = self._compute_orderflow_imbalance(market_id, data)
        signals['spread'] = self._compute_spread_signal(market_id, data)
        signals['liquidity'] = self._compute_liquidity_imbalance(market_id, data)
        signals['momentum'] = self._compute_momentum_signal(market_id, data)
        signals['external'] = self._compute_external_confirmation(market_id, data)
        
        # Apply decay to smooth signals
        for name, signal in signals.items():
            signals[name] = self._apply_signal_decay(market_id, name, signal)
        
        self._signal_count += 1
        
        computation_time = (time.perf_counter() - start_time) * 1000
        self._computation_times.append(computation_time)
        
        return signals
    
    def _compute_orderflow_imbalance(
        self,
        market_id: str,
        data: MarketData,
    ) -> SignalResult:
        """
        Compute orderflow imbalance signal.
        
        OFI measures net aggressive buying/selling pressure.
        Positive = bullish (more aggressive buyers)
        Negative = bearish (more aggressive sellers)
        """
        # Analyze recent trades for aggressive side
        buy_volume = Decimal('0')
        sell_volume = Decimal('0')
        
        for trade in data.recent_trades[-50:]:  # Last 50 trades
            size = Decimal(str(trade.get('size', 0)))
            side = trade.get('side', '').upper()
            
            if side == 'BUY' or side == 'BID':
                buy_volume += size
            elif side == 'SELL' or side == 'ASK':
                sell_volume += size
        
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return SignalResult(
                name='orderflow',
                value=0.0,
                confidence=0.0,
                timestamp=time.time(),
                metadata={'buy_volume': 0, 'sell_volume': 0},
            )
        
        # Normalize to [-1, 1]
        ofi = float((buy_volume - sell_volume) / total_volume)
        
        # Confidence based on volume
        volume_confidence = min(1.0, float(total_volume / self.min_depth_threshold))
        
        return SignalResult(
            name='orderflow',
            value=ofi,
            confidence=volume_confidence,
            timestamp=time.time(),
            metadata={
                'buy_volume': float(buy_volume),
                'sell_volume': float(sell_volume),
                'total_volume': float(total_volume),
                'raw_ofi': ofi,
            },
        )
    
    def _compute_spread_signal(
        self,
        market_id: str,
        data: MarketData,
    ) -> SignalResult:
        """
        Compute spread-based signal.
        
        Tight spreads indicate conviction and lower adverse selection.
        Wide spreads indicate uncertainty.
        """
        # Expected max spread for binary options (typically 2-5 cents)
        max_expected_spread_pct = 5.0
        
        if data.spread_pct <= 0:
            return SignalResult(
                name='spread',
                value=0.5,  # Neutral
                confidence=0.0,
                timestamp=time.time(),
                metadata={'reason': 'invalid_spread'},
            )
        
        # Score: 1.0 for tight spreads, 0.0 for wide spreads
        spread_score = 1.0 - min(data.spread_pct / max_expected_spread_pct, 1.0)
        
        # Confidence higher when spread is stable (not computed here for simplicity)
        confidence = spread_score  # More confident when spread is tight
        
        return SignalResult(
            name='spread',
            value=spread_score,
            confidence=confidence,
            timestamp=time.time(),
            metadata={
                'spread_pct': data.spread_pct,
                'max_expected_spread_pct': max_expected_spread_pct,
            },
        )
    
    def _compute_liquidity_imbalance(
        self,
        market_id: str,
        data: MarketData,
    ) -> SignalResult:
        """
        Compute liquidity imbalance signal.
        
        Measures skew in passive interest between bids and asks.
        Positive = more bid support = bullish
        """
        # Calculate depth within 5 ticks of mid
        tick_size = Decimal('0.01')
        threshold = tick_size * 5
        
        bid_depth = Decimal('0')
        ask_depth = Decimal('0')
        
        for price, size in data.bid_levels:
            if abs(price - data.mid_price) <= threshold:
                bid_depth += size
        
        for price, size in data.ask_levels:
            if abs(price - data.mid_price) <= threshold:
                ask_depth += size
        
        total_depth = bid_depth + ask_depth
        
        if total_depth == 0:
            return SignalResult(
                name='liquidity',
                value=0.0,
                confidence=0.0,
                timestamp=time.time(),
                metadata={'reason': 'no_depth'},
            )
        
        # Normalize to [-1, 1]
        liq_imbalance = float((bid_depth - ask_depth) / total_depth)
        
        # Confidence based on total depth
        depth_confidence = min(1.0, float(total_depth / self.min_depth_threshold))
        
        return SignalResult(
            name='liquidity',
            value=liq_imbalance,
            confidence=depth_confidence,
            timestamp=time.time(),
            metadata={
                'bid_depth': float(bid_depth),
                'ask_depth': float(ask_depth),
                'total_depth': float(total_depth),
            },
        )
    
    def _compute_momentum_signal(
        self,
        market_id: str,
        data: MarketData,
    ) -> SignalResult:
        """
        Compute momentum signal.
        
        Risk-adjusted momentum based on recent price returns.
        """
        price_hist = self._price_history.get(market_id)
        
        if not price_hist:
            return SignalResult(
                name='momentum',
                value=0.0,
                confidence=0.0,
                timestamp=time.time(),
                metadata={'reason': 'no_history'},
            )
        
        momentum = price_hist.get_momentum(self.momentum_lookback)
        volatility = price_hist.get_volatility(self.volatility_lookback)
        
        if momentum is None or volatility is None or volatility == 0:
            return SignalResult(
                name='momentum',
                value=0.0,
                confidence=0.0,
                timestamp=time.time(),
                metadata={'reason': 'insufficient_data'},
            )
        
        # Risk-adjusted momentum (Sharpe-like)
        risk_adjusted_momentum = momentum / volatility
        
        # Bound to [-1, 1] using tanh
        bounded_momentum = math.tanh(risk_adjusted_momentum)
        
        # Convert to [0, 1] scale for consistency
        normalized_value = 0.5 + 0.5 * bounded_momentum
        
        # Confidence based on data quality
        returns_count = len(price_hist.get_returns(self.momentum_lookback))
        confidence = min(1.0, returns_count / self.momentum_lookback)
        
        return SignalResult(
            name='momentum',
            value=normalized_value,
            confidence=confidence,
            timestamp=time.time(),
            metadata={
                'raw_momentum': momentum,
                'volatility': volatility,
                'risk_adjusted_momentum': risk_adjusted_momentum,
            },
        )
    
    def _compute_external_confirmation(
        self,
        market_id: str,
        data: MarketData,
    ) -> SignalResult:
        """
        Compute external price confirmation signal.
        
        Compares Polymarket implied probability with spot market direction.
        """
        if data.binance_price is None and data.coinbase_price is None:
            return SignalResult(
                name='external',
                value=0.5,  # Neutral
                confidence=0.0,
                timestamp=time.time(),
                metadata={'reason': 'no_external_data'},
            )
        
        # Get average external price
        external_prices = []
        if data.binance_price:
            external_prices.append(float(data.binance_price))
        if data.coinbase_price:
            external_prices.append(float(data.coinbase_price))
        
        avg_external_price = sum(external_prices) / len(external_prices)
        
        # Check if external price is moving up or down
        # (In production, would compare to reference price from RTDS)
        if data.rtds_reference_price:
            ref_price = float(data.rtds_reference_price)
            price_change_pct = (avg_external_price - ref_price) / ref_price * 100
            
            # Map price change to directional signal
            if price_change_pct > 0.5:  # Up more than 0.5%
                signal_value = 1.0
            elif price_change_pct < -0.5:  # Down more than 0.5%
                signal_value = 0.0
            else:
                # Linear interpolation
                signal_value = 0.5 + price_change_pct  # Maps -0.5 to 0.5, +0.5 to 1.0
            
            confidence = min(1.0, abs(price_change_pct) / 2.0)
            
        else:
            # No reference price - use Polymarket mid vs 0.5
            poly_mid = float(data.mid_price) if data.mid_price else 0.5
            signal_value = poly_mid
            confidence = 0.5
        
        return SignalResult(
            name='external',
            value=signal_value,
            confidence=confidence,
            timestamp=time.time(),
            metadata={
                'binance_price': float(data.binance_price) if data.binance_price else None,
                'coinbase_price': float(data.coinbase_price) if data.coinbase_price else None,
                'avg_external_price': avg_external_price,
            },
        )
    
    def _apply_signal_decay(
        self,
        market_id: str,
        signal_name: str,
        signal: SignalResult,
    ) -> SignalResult:
        """
        Apply exponential decay to smooth signal transitions.
        
        Prevents whipsaw from noisy data.
        """
        prev_signals = self._previous_signals.get(market_id, {})
        prev_value = prev_signals.get(signal_name, signal.value)
        
        # Exponential smoothing
        smoothed_value = (
            (1 - self.signal_decay_lambda) * prev_value +
            self.signal_decay_lambda * signal.value
        )
        
        # Store for next iteration
        self._previous_signals[market_id][signal_name] = smoothed_value
        
        # Create updated signal
        return SignalResult(
            name=signal.name,
            value=smoothed_value,
            confidence=signal.confidence,
            timestamp=signal.timestamp,
            metadata=signal.metadata,
        )
    
    def compute_weighted_probability(
        self,
        signals: Dict[str, SignalResult],
    ) -> Tuple[float, float]:
        """
        Compute weighted directional probability and confidence.
        
        Returns: (probability_UP, overall_confidence)
        """
        weighted_sum = 0.0
        total_confidence_weight = 0.0
        
        for signal_name, signal in signals.items():
            weight = self.normalized_weights.get(signal_name, 0.0)
            
            if weight > 0:
                # Convert signal value to contribution
                # For orderflow/liquidity/momentum: value in [-1, 1], map to [0, 1]
                if signal_name in ['orderflow', 'liquidity']:
                    prob_contribution = 0.5 + 0.5 * signal.value
                else:
                    prob_contribution = signal.value
                
                weighted_sum += weight * prob_contribution
                total_confidence_weight += weight * signal.confidence
        
        if total_confidence_weight == 0:
            return 0.5, 0.0
        
        # Final probability
        raw_probability = weighted_sum / sum(self.normalized_weights.values())
        
        # Apply sigmoid-like transformation for better calibration
        # Center at 0.5, stretch extremes
        centered = raw_probability - 0.5
        transformed = 0.5 + 0.5 * math.tanh(centered * 4)  # Stretch factor
        
        # Overall confidence is weighted average of individual confidences
        overall_confidence = total_confidence_weight / sum(self.normalized_weights.values())
        
        return transformed, overall_confidence
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get signal engine metrics."""
        avg_computation_time = (
            sum(self._computation_times) / len(self._computation_times)
            if self._computation_times else 0
        )
        
        return {
            'signal_count': self._signal_count,
            'avg_computation_time_ms': avg_computation_time,
            'markets_tracked': len(self._price_history),
        }
