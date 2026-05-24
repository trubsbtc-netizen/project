"""
Fill probability and slippage modeling for Polymarket order execution.

Models the probability of order fill and expected slippage based on:
- Polymarket orderbook depth and liquidity
- Regime-dependent spread expansion
- Queue position and survival probability
- Size impact on fill probability
- Historical fill rate tracking

Mathematical basis:
- Fill probability: P(fill) = base_fill * queue_survival * size_decay * regime_factor
- Slippage model: s = sigma * sqrt(size / depth) * regime_multiplier
  (Almgren-Chriss square-root law adapted for CLOB markets)
- Effective price: p_eff = p_mid + slippage * direction_sign
- Size impact: impact(size) = (size/depth)^alpha where alpha ~ 0.5
"""

import time

import numpy as np
from typing import Optional, Dict

from polymarket_bot.bot_types import (
    ExecutionEstimate,
    OrderbookSnapshot,
    PolymarketOrderbook,
    QueueMetrics,
    SpreadMetrics,
    LiquidityMetrics,
    RegimeStateEstimate,
    VolatilityEstimate,
    Direction,
    RegimeState,
    ExecutionQuality,
)
from polymarket_bot.config import BotConfig, ExecutionConfig


class FillModel:
    """
    Models fill probability and slippage for Polymarket CLOB execution.
    
    Polymarket uses a CLOB (Central Limit Order Book) where orders are
    matched based on price-time priority. Fill probability depends on:
    
    1. Queue position: how many orders are ahead of ours
    2. Available liquidity: depth at our price level
    3. Regime conditions: volatile regimes have worse fills
    4. Size relative to available depth: large orders face more slippage
    5. Spread conditions: wider spreads mean more slippage
    
    Slippage follows the Almgren-Chriss square-root model adapted
    for discrete CLOB markets with regime-dependent multipliers.
    """

    # Regime-dependent fill probability multipliers
    REGIME_FILL_MULTIPLIERS = {
        RegimeState.CALM_TRENDING: 1.0,
        RegimeState.CALM_RANGE: 1.0,
        RegimeState.VOLATILE_TRENDING: 0.85,
        RegimeState.VOLATILE_RANGE: 0.80,
        RegimeState.CRISIS: 0.60,
        RegimeState.LIQUIDATION_CASCADE: 0.40,
    }

    # Regime-dependent slippage multipliers (from config, with defaults)
    REGIME_SLIPPAGE_MULTIPLIERS = {
        RegimeState.CALM_TRENDING: 1.0,
        RegimeState.CALM_RANGE: 1.0,
        RegimeState.VOLATILE_TRENDING: 1.5,
        RegimeState.VOLATILE_RANGE: 1.3,
        RegimeState.CRISIS: 2.5,
        RegimeState.LIQUIDATION_CASCADE: 4.0,
    }

    # Almgren-Chriss impact exponent (square-root law)
    IMPACT_ALPHA = 0.5

    # Temporary impact fraction (vs permanent impact)
    TEMP_IMPACT_FRACTION = 0.7

    def __init__(self, config: BotConfig):
        self.config = config
        self.exec_config = config.execution
        
        # Use config regime multipliers if available
        if self.exec_config.spread_expansion_regime_multiplier:
            self.REGIME_SLIPPAGE_MULTIPLIERS = {
                k: v for k, v in self.exec_config.spread_expansion_regime_multiplier.items()
            }
        
        # Historical fill tracking
        self._fill_history: Dict[str, list] = {"success": [], "failure": []}
        self._slippage_history: list = []
        self._max_history = 500

    @staticmethod
    def _ask_notional(level: OrderbookLevel) -> float:
        return max(0.0, float(level.price) * float(level.size))

    def _project_market_buy(
        self,
        size_notional: float,
        polymarket_orderbook: Optional[PolymarketOrderbook],
        max_levels: int = 10,
    ) -> Dict[str, float]:
        """
        Project a marketable BUY against the visible ask book.

        `size_notional` is USDC. CLOB level size is outcome-token shares,
        so the exact fill cost is price * shares at each ask level.
        """
        result = {
            "best_ask": 0.0,
            "avg_price": 0.99,
            "limit_price": 0.99,
            "slippage": 0.50,
            "visible_depth_notional": 0.0,
            "best_ask_notional": 0.0,
            "filled_fraction": 0.0,
        }

        if (
            polymarket_orderbook is None
            or not polymarket_orderbook.asks
            or polymarket_orderbook.best_ask is None
        ):
            return result

        asks = sorted(polymarket_orderbook.asks[:max_levels], key=lambda level: level.price)
        best_ask = float(asks[0].price)
        if best_ask <= 0:
            return result

        result["best_ask"] = best_ask
        result["avg_price"] = best_ask
        result["limit_price"] = best_ask
        result["slippage"] = 0.0
        result["visible_depth_notional"] = float(sum(self._ask_notional(level) for level in asks))
        result["best_ask_notional"] = self._ask_notional(asks[0])

        if size_notional <= 0:
            result["filled_fraction"] = 1.0
            return result

        target_shares = float(size_notional) / best_ask
        remaining_shares = target_shares
        filled_shares = 0.0
        total_cost = 0.0
        limit_price = best_ask

        for level in asks:
            level_price = float(level.price)
            level_shares = max(0.0, float(level.size))
            if level_price <= 0 or level_shares <= 0:
                continue

            take_shares = min(remaining_shares, level_shares)
            total_cost += take_shares * level_price
            filled_shares += take_shares
            remaining_shares -= take_shares
            limit_price = level_price

            if remaining_shares <= 1e-9:
                break

        if filled_shares <= 0:
            return result

        avg_price = total_cost / filled_shares
        filled_fraction = min(1.0, filled_shares / max(target_shares, 1e-9))

        if filled_fraction < 1.0:
            # If visible depth cannot fill the requested notional, force a
            # conservative effective price so the execution gate rejects or
            # downsizes rather than assuming hidden liquidity.
            missing_fraction = 1.0 - filled_fraction
            avg_price = min(0.99, avg_price + 0.10 * missing_fraction)
            limit_price = min(0.99, limit_price + 0.10 * missing_fraction)

        result["avg_price"] = float(np.clip(avg_price, 0.01, 0.99))
        result["limit_price"] = float(np.clip(limit_price, 0.01, 0.99))
        result["slippage"] = float(max(0.0, result["avg_price"] - best_ask))
        result["filled_fraction"] = float(np.clip(filled_fraction, 0.0, 1.0))
        return result

    def compute_fill_probability(
        self,
        direction: Direction,
        size: float,
        polymarket_orderbook: Optional[PolymarketOrderbook] = None,
        queue_metrics: Optional[QueueMetrics] = None,
        liquidity_metrics: Optional[LiquidityMetrics] = None,
        regime_estimate: Optional[RegimeStateEstimate] = None,
    ) -> float:
        """
        Compute probability of order fill.
        
        P(fill) = base_fill * queue_survival * size_decay * regime_factor
        
        Components:
        1. base_fill: baseline fill probability from config
        2. queue_survival: probability our order survives in queue
        3. size_decay: exponential decay for large sizes relative to depth
        4. regime_factor: regime-dependent fill quality multiplier
        """
        # Base fill probability
        base_fill = self.exec_config.fill_probability_base

        projection = self._project_market_buy(size, polymarket_orderbook)
        if projection["best_ask"] <= 0 or projection["visible_depth_notional"] <= 0:
            return 0.01

        # Marketable FOK fill is governed by visible ask depth, not by BTC
        # queue metrics. Coverage is exact against the current Polymarket book.
        coverage = projection["filled_fraction"]
        size_decay = coverage ** 2

        # Stale books are less reliable even if displayed depth is sufficient.
        freshness = 1.0
        if polymarket_orderbook is not None and polymarket_orderbook.timestamp:
            age_seconds = max(0.0, time.time() - polymarket_orderbook.timestamp)
            freshness = float(np.exp(-age_seconds / 10.0))

        # Regime factor
        regime_factor = 1.0
        if regime_estimate is not None:
            regime_state = regime_estimate.current_regime
            regime_factor = max(
                0.75,
                self.REGIME_FILL_MULTIPLIERS.get(regime_state, 0.8),
            )

        # Liquidity quality factor
        liquidity_factor = 1.0
        if liquidity_metrics is not None:
            # Real liquidity fraction: how much of displayed depth is real
            liquidity_factor = 0.5 + 0.5 * liquidity_metrics.real_liquidity_fraction

        # Combined fill probability
        fill_prob = base_fill * size_decay * freshness * regime_factor * liquidity_factor

        return np.clip(float(fill_prob), 0.01, 0.99)

    def compute_slippage(
        self,
        direction: Direction,
        size: float,
        polymarket_orderbook: Optional[PolymarketOrderbook] = None,
        spread_metrics: Optional[SpreadMetrics] = None,
        liquidity_metrics: Optional[LiquidityMetrics] = None,
        regime_estimate: Optional[RegimeStateEstimate] = None,
        volatility_estimate: Optional[VolatilityEstimate] = None,
    ) -> tuple:
        """
        Compute expected slippage using Almgren-Chriss square-root model.
        
        Slippage = sigma * sqrt(size / depth) * regime_multiplier
        
        Returns (slippage_absolute, slippage_bps, effective_price,
        spread_impact, liquidity_impact, limit_price, visible_depth_notional)
        """
        # Base volatility for slippage calculation
        sigma = self.exec_config.slippage_model_sigma
        if volatility_estimate is not None:
            # Use realized volatility as base, scaled to price level
            sigma = volatility_estimate.realized_volatility

        projection = self._project_market_buy(size, polymarket_orderbook)
        total_depth = projection["visible_depth_notional"]

        # Almgren-Chriss square-root impact
        # impact = sigma * (size / total_depth)^alpha
        size_ratio = size / max(1.0, total_depth)
        base_impact = sigma * np.power(max(0.0, size_ratio), self.IMPACT_ALPHA)

        # Temporary impact (most of slippage is temporary for small orders)
        temp_impact = base_impact * self.TEMP_IMPACT_FRACTION
        perm_impact = base_impact * (1.0 - self.TEMP_IMPACT_FRACTION)
        total_impact = temp_impact + perm_impact

        # Regime multiplier
        regime_multiplier = 1.0
        if regime_estimate is not None:
            regime_state = regime_estimate.current_regime
            regime_multiplier = self.REGIME_SLIPPAGE_MULTIPLIERS.get(regime_state, 1.5)

        # Spread impact component in Polymarket probability units. This is
        # reported separately; when edge is computed against best ask it must
        # not be subtracted again as slippage.
        spread_impact = 0.0
        if (
            polymarket_orderbook is not None
            and polymarket_orderbook.best_bid is not None
            and polymarket_orderbook.best_ask is not None
        ):
            spread_impact = max(
                0.0,
                (polymarket_orderbook.best_ask - polymarket_orderbook.best_bid) / 2.0,
            )
        elif spread_metrics is not None:
            # BTC spread is in dollars, so it cannot be added directly to a
            # Polymarket token price. Use it only as a bounded stress signal.
            spread_impact = min(0.01, max(0.0, spread_metrics.spread_bps) / 10000.0)
        if spread_metrics is not None and spread_metrics.is_expanding:
            spread_impact *= 1.3

        # Liquidity impact: poor liquidity increases slippage
        liquidity_impact = 0.0
        if liquidity_metrics is not None:
            # Low real liquidity fraction means more slippage
            real_frac = liquidity_metrics.real_liquidity_fraction
            # Impact multiplier: 1.0 when real_frac=1, up to 2.0 when real_frac=0
            liquidity_impact = base_impact * (1.0 - real_frac) * 0.5

        # Total slippage
        if projection["best_ask"] <= 0:
            total_slippage = 0.50
            effective_price = 0.99
            limit_price = 0.99
        else:
            # Visible-book VWAP is the primary cost model. The square-root
            # component remains only as a small adverse-selection reserve.
            adverse_selection_reserve = total_impact * regime_multiplier * 0.25
            total_slippage = (
                projection["slippage"]
                + adverse_selection_reserve
                + liquidity_impact
            )
            effective_price = projection["avg_price"] + adverse_selection_reserve + liquidity_impact
            limit_price = max(projection["limit_price"], effective_price)
        total_slippage = float(np.clip(total_slippage, 0.0, 0.50))
        effective_price = np.clip(effective_price, 0.01, 0.99)
        limit_price = np.clip(limit_price, 0.01, 0.99)

        # Slippage in basis points
        slippage_bps = total_slippage * 10000

        # Track slippage history
        self._slippage_history.append({
            "timestamp": time.time(),
            "slippage": total_slippage,
            "slippage_bps": slippage_bps,
            "size": size,
            "direction": direction,
        })
        if len(self._slippage_history) > self._max_history:
            self._slippage_history = self._slippage_history[-self._max_history:]

        return (
            float(total_slippage),
            float(slippage_bps),
            float(effective_price),
            float(spread_impact),
            float(liquidity_impact),
            float(limit_price),
            float(total_depth),
        )

    def compute_max_size(
        self,
        direction: Direction,
        polymarket_orderbook: Optional[PolymarketOrderbook] = None,
        liquidity_metrics: Optional[LiquidityMetrics] = None,
        regime_estimate: Optional[RegimeStateEstimate] = None,
        max_slippage_bps: float = 200.0,
    ) -> float:
        """
        Compute maximum recommended position size given current conditions.
        
        Uses inverse Almgren-Chriss: given max acceptable slippage,
        compute max size that stays within that slippage bound.
        
        max_size = depth * (max_slippage / (sigma * regime_mult))^2
        """
        # Get regime multiplier
        regime_multiplier = 1.0
        if regime_estimate is not None:
            regime_state = regime_estimate.current_regime
            regime_multiplier = self.REGIME_SLIPPAGE_MULTIPLIERS.get(regime_state, 1.5)

        # Get sigma
        sigma = self.exec_config.slippage_model_sigma

        if (
            polymarket_orderbook is None
            or not polymarket_orderbook.asks
            or polymarket_orderbook.best_ask is None
        ):
            return 0.0

        asks = sorted(polymarket_orderbook.asks[:10], key=lambda level: level.price)
        best_ask = float(asks[0].price)
        if best_ask <= 0:
            return 0.0

        # Liquidity quality adjustment
        real_frac = 1.0
        if liquidity_metrics is not None:
            real_frac = liquidity_metrics.real_liquidity_fraction

        max_slippage = max_slippage_bps / 10000.0
        cumulative_cost = 0.0
        cumulative_shares = 0.0
        max_size = 0.0

        for level in asks:
            price = float(level.price)
            shares = max(0.0, float(level.size))
            if price <= 0 or shares <= 0:
                continue

            candidate_cost = cumulative_cost + price * shares
            candidate_shares = cumulative_shares + shares
            candidate_vwap = candidate_cost / max(candidate_shares, 1e-9)
            candidate_slippage = max(0.0, candidate_vwap - best_ask)

            if candidate_slippage <= max_slippage:
                cumulative_cost = candidate_cost
                cumulative_shares = candidate_shares
                max_size = candidate_cost
                continue

            target_vwap = best_ask + max_slippage
            denominator = price - target_vwap
            if denominator > 0:
                partial_shares = (target_vwap * cumulative_shares - cumulative_cost) / denominator
                partial_shares = float(np.clip(partial_shares, 0.0, shares))
                max_size = cumulative_cost + price * partial_shares
            break

        if max_size <= 0.0:
            # At least allow the best ask level when it is inside the threshold.
            max_size = self._ask_notional(asks[0])

        # Adjust for real liquidity fraction
        max_size *= float(np.clip(real_frac, 0.0, 1.0))

        # Cap at config maximum
        max_size = min(max_size, self.config.risk.max_position_notional)

        return float(max_size)

    def estimate_fill_time(
        self,
        direction: Direction,
        size: float,
        polymarket_orderbook: Optional[PolymarketOrderbook] = None,
        queue_metrics: Optional[QueueMetrics] = None,
        regime_estimate: Optional[RegimeStateEstimate] = None,
    ) -> float:
        """
        Estimate time to fill in seconds.
        
        Based on:
        - Queue position: more orders ahead = longer wait
        - Fill rate: historical rate of order matching
        - Regime: volatile regimes fill faster (more activity)
        """
        # Base fill time: 5 seconds for a small order at front of queue
        base_fill_time = 5.0

        # Queue position adjustment
        if queue_metrics is not None:
            # Expected queue position
            if direction == Direction.UP:
                queue_position = queue_metrics.bid_queue_position
            else:
                queue_position = queue_metrics.ask_queue_position

            # Each position in queue adds ~2 seconds
            base_fill_time += queue_position * 2.0

        # Size adjustment: larger orders take longer
        available_depth = 0.0
        if (
            polymarket_orderbook is not None
            and polymarket_orderbook.best_ask is not None
            and polymarket_orderbook.best_ask_size is not None
        ):
            available_depth = polymarket_orderbook.best_ask * polymarket_orderbook.best_ask_size
        if available_depth <= 0:
            available_depth = 1.0

        size_ratio = size / max(1.0, available_depth)
        base_fill_time *= (1.0 + size_ratio)

        # Regime adjustment
        regime_time_factor = 1.0
        if regime_estimate is not None:
            regime_state = regime_estimate.current_regime
            # Volatile regimes: faster fills (more activity)
            # Calm regimes: slower fills (less activity)
            time_factors = {
                RegimeState.CALM_TRENDING: 1.5,
                RegimeState.CALM_RANGE: 2.0,
                RegimeState.VOLATILE_TRENDING: 0.7,
                RegimeState.VOLATILE_RANGE: 0.8,
                RegimeState.CRISIS: 0.5,
                RegimeState.LIQUIDATION_CASCADE: 0.3,
            }
            regime_time_factor = time_factors.get(regime_state, 1.0)

        fill_time = base_fill_time * regime_time_factor

        return float(np.clip(fill_time, 1.0, 60.0))

    def classify_execution_quality(
        self,
        fill_probability: float,
        slippage_bps: float,
        spread_bps: float,
        regime_state: RegimeState,
    ) -> ExecutionQuality:
        """
        Classify execution quality based on fill probability and slippage.
        
        OPTIMAL: high fill prob, low slippage, calm regime
        ACCEPTABLE: moderate fill prob, moderate slippage
        DEGRADED: low fill prob or high slippage
        REJECT: very low fill prob or extreme slippage
        """
        wide_spread = spread_bps > self.exec_config.max_spread_bps_for_execution

        if fill_probability < self.exec_config.min_fill_probability or slippage_bps > 200:
            return ExecutionQuality.REJECT

        # Fill probability thresholds
        if fill_probability >= 0.85 and slippage_bps <= 50:
            if regime_state in (RegimeState.CALM_TRENDING, RegimeState.CALM_RANGE):
                return ExecutionQuality.OPTIMAL
            else:
                return ExecutionQuality.ACCEPTABLE

        if fill_probability >= 0.7 and slippage_bps <= 100:
            return ExecutionQuality.ACCEPTABLE

        if wide_spread:
            return ExecutionQuality.DEGRADED

        if fill_probability >= self.exec_config.min_fill_probability and slippage_bps <= 200:
            return ExecutionQuality.DEGRADED

        return ExecutionQuality.REJECT
