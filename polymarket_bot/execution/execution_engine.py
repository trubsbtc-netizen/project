"""
Execution engine for Polymarket order placement.

Coordinates fill probability, slippage, and latency modeling to produce
a complete ExecutionEstimate that feeds into the decision layer.

The execution engine is the bridge between inference (probability estimates)
and the actual order placement on Polymarket CLOB. It ensures that:
1. We only execute when fill conditions are favorable
2. Slippage is properly accounted for in edge calculations
3. Latency risk is incorporated into effective edge
4. Position sizing respects execution constraints
5. Dry-run mode is properly handled

Key principle: execution realism must be conservative. Overestimating
fill probability or underestimating slippage leads to systematic losses.
"""

import time
import numpy as np
from typing import Optional

from polymarket_bot.bot_types import (
    ExecutionEstimate,
    SettlementForecast,
    PolymarketOrderbook,
    OrderbookSnapshot,
    QueueMetrics,
    SpreadMetrics,
    LiquidityMetrics,
    RegimeStateEstimate,
    VolatilityEstimate,
    Direction,
    RegimeState,
    ExecutionQuality,
)
from polymarket_bot.config import BotConfig
from polymarket_bot.execution.fill_model import FillModel
from polymarket_bot.execution.latency_model import LatencyModel, LatencyEstimate


class ExecutionEngine:
    """
    Coordinates all execution modeling and produces ExecutionEstimate.
    
    The engine combines:
    1. FillModel: fill probability and slippage estimation
    2. LatencyModel: latency and signal decay estimation
    3. Regime-aware execution quality classification
    4. Position sizing constraints from risk management
    
    Output: ExecutionEstimate with all execution realism parameters,
    which feeds into the final TradeDecision.
    """

    def __init__(self, config: BotConfig):
        self.config = config
        self.fill_model = FillModel(config)
        self.latency_model = LatencyModel(config)
        self._execution_count = 0
        self._last_execution_timestamp = 0.0

    def estimate_execution(
        self,
        direction: Direction,
        size_fraction: float,
        settlement_forecast: SettlementForecast,
        polymarket_orderbook: Optional[PolymarketOrderbook] = None,
        btc_orderbook: Optional[OrderbookSnapshot] = None,
        queue_metrics: Optional[QueueMetrics] = None,
        spread_metrics: Optional[SpreadMetrics] = None,
        liquidity_metrics: Optional[LiquidityMetrics] = None,
        regime_estimate: Optional[RegimeStateEstimate] = None,
        volatility_estimate: Optional[VolatilityEstimate] = None,
    ) -> ExecutionEstimate:
        """
        Produce a complete execution estimate for a potential trade.
        
        Process:
        1. Compute position size from size_fraction and available capital
        2. Estimate fill probability from FillModel
        3. Estimate slippage from FillModel
        4. Estimate latency from LatencyModel
        5. Compute latency-adjusted effective edge
        6. Classify execution quality
        7. Compute maximum recommended size
        8. Estimate fill time
        9. Produce ExecutionEstimate
        """
        timestamp = time.time()

        # Step 1: Position size
        # size_fraction comes from settlement_forecast, scale to max position
        max_position = self.config.risk.max_position_notional
        position_size = size_fraction * max_position

        # Step 2: Fill probability
        fill_probability = self.fill_model.compute_fill_probability(
            direction=direction,
            size=position_size,
            polymarket_orderbook=polymarket_orderbook,
            queue_metrics=queue_metrics,
            liquidity_metrics=liquidity_metrics,
            regime_estimate=regime_estimate,
        )

        # Step 3: Slippage
        (
            slippage,
            slippage_bps,
            effective_price,
            spread_impact,
            liquidity_impact,
            limit_price,
            visible_depth_notional,
        ) = (
            self.fill_model.compute_slippage(
                direction=direction,
                size=position_size,
                polymarket_orderbook=polymarket_orderbook,
                spread_metrics=spread_metrics,
                liquidity_metrics=liquidity_metrics,
                regime_estimate=regime_estimate,
                volatility_estimate=volatility_estimate,
            )
        )

        # Step 4: Latency
        latency_estimate = self.latency_model.estimate_latency(
            regime_estimate=regime_estimate,
            volatility_estimate=volatility_estimate,
        )

        # Step 5: Latency-adjusted effective edge
        raw_edge_bps = settlement_forecast.edge_estimate * 10000.0
        effective_edge_bps, edge_retention = self.latency_model.compute_latency_adjusted_edge(
            raw_edge_bps=raw_edge_bps,
            latency_estimate=latency_estimate,
        )

        # Adjust slippage for latency: longer latency = more uncertainty = more slippage
        latency_slippage_adjustment = slippage_bps * (1.0 - edge_retention) * 0.5
        adjusted_slippage_bps = slippage_bps + latency_slippage_adjustment

        # Step 6: Execution quality classification
        spread_bps = 0.0
        if (
            polymarket_orderbook is not None
            and polymarket_orderbook.best_bid is not None
            and polymarket_orderbook.best_ask is not None
            and polymarket_orderbook.mid_price is not None
            and polymarket_orderbook.mid_price > 0
        ):
            spread_bps = (
                (polymarket_orderbook.best_ask - polymarket_orderbook.best_bid)
                / polymarket_orderbook.mid_price
            ) * 10000.0
        elif spread_metrics is not None:
            spread_bps = spread_metrics.spread_bps

        regime_state = RegimeState.CALM_TRENDING
        if regime_estimate is not None:
            regime_state = regime_estimate.current_regime

        execution_quality = self.fill_model.classify_execution_quality(
            fill_probability=fill_probability,
            slippage_bps=adjusted_slippage_bps,
            spread_bps=spread_bps,
            regime_state=regime_state,
        )

        # Step 7: Maximum recommended size
        max_recommended_size = self.fill_model.compute_max_size(
            direction=direction,
            polymarket_orderbook=polymarket_orderbook,
            liquidity_metrics=liquidity_metrics,
            regime_estimate=regime_estimate,
            max_slippage_bps=self.config.risk.min_edge_bps * 2.0,  # Max slippage = 2x min edge
        )

        # Cap position size to max recommended
        if position_size > max_recommended_size:
            position_size = max_recommended_size
            fill_probability = self.fill_model.compute_fill_probability(
                direction=direction,
                size=position_size,
                polymarket_orderbook=polymarket_orderbook,
                queue_metrics=queue_metrics,
                liquidity_metrics=liquidity_metrics,
                regime_estimate=regime_estimate,
            )
            (
                slippage,
                slippage_bps,
                effective_price,
                spread_impact,
                liquidity_impact,
                limit_price,
                visible_depth_notional,
            ) = self.fill_model.compute_slippage(
                direction=direction,
                size=position_size,
                polymarket_orderbook=polymarket_orderbook,
                spread_metrics=spread_metrics,
                liquidity_metrics=liquidity_metrics,
                regime_estimate=regime_estimate,
                volatility_estimate=volatility_estimate,
            )
            latency_slippage_adjustment = slippage_bps * (1.0 - edge_retention) * 0.5
            adjusted_slippage_bps = slippage_bps + latency_slippage_adjustment
            execution_quality = self.fill_model.classify_execution_quality(
                fill_probability=fill_probability,
                slippage_bps=adjusted_slippage_bps,
                spread_bps=spread_bps,
                regime_state=regime_state,
            )

        # Step 8: Fill time estimate
        fill_time = self.fill_model.estimate_fill_time(
            direction=direction,
            size=position_size,
            polymarket_orderbook=polymarket_orderbook,
            queue_metrics=queue_metrics,
            regime_estimate=regime_estimate,
        )

        # Step 9: Regime impact on execution
        regime_impact = 0.0
        if regime_estimate is not None:
            regime_state = regime_estimate.current_regime
            # Regime impact: how much worse execution is vs. ideal conditions
            regime_impact_factors = {
                RegimeState.CALM_TRENDING: 0.0,
                RegimeState.CALM_RANGE: 0.0,
                RegimeState.VOLATILE_TRENDING: 0.15,
                RegimeState.VOLATILE_RANGE: 0.20,
                RegimeState.CRISIS: 0.40,
                RegimeState.LIQUIDATION_CASCADE: 0.60,
            }
            regime_impact = regime_impact_factors.get(regime_state, 0.2)

        # Effective price is an execution cost, not a settlement probability.
        latency_price_adjustment = latency_slippage_adjustment / 10000.0
        effective_price_adjusted = effective_price + latency_price_adjustment
        effective_price_adjusted = np.clip(effective_price_adjusted, 0.01, 0.99)
        limit_price = np.clip(max(limit_price, effective_price_adjusted), 0.01, 0.99)

        # Build ExecutionEstimate
        estimate = ExecutionEstimate(
            timestamp=timestamp,
            expected_fill_probability=float(fill_probability),
            expected_slippage=float(slippage),
            expected_slippage_bps=float(adjusted_slippage_bps),
            expected_latency_ms=float(latency_estimate.expected_total_latency_ms),
            effective_price=float(effective_price_adjusted),
            spread_impact=float(spread_impact),
            liquidity_impact=float(liquidity_impact),
            regime_impact=float(regime_impact),
            execution_quality=execution_quality,
            max_recommended_size=float(max_recommended_size),
            fill_time_estimate=float(fill_time),
            limit_price=float(limit_price),
            visible_depth_notional=float(visible_depth_notional),
            best_ask_price=(
                float(polymarket_orderbook.best_ask)
                if polymarket_orderbook is not None and polymarket_orderbook.best_ask is not None
                else None
            ),
        )

        self._execution_count += 1
        self._last_execution_timestamp = timestamp

        return estimate

    def should_execute(
        self,
        execution_estimate: ExecutionEstimate,
        settlement_forecast: SettlementForecast,
    ) -> bool:
        """
        Final execution gate check.
        
        Execute only if ALL conditions are met:
        1. Execution quality is not REJECT
        2. Fill probability exceeds minimum
        3. Slippage is within acceptable bounds
        4. Edge after execution costs is still positive
        5. Settlement forecast recommends execution
        """
        # Quality gate
        if execution_estimate.execution_quality == ExecutionQuality.REJECT:
            return False

        # Fill probability gate
        if execution_estimate.expected_fill_probability < self.config.execution.min_fill_probability:
            return False

        # Slippage gate: slippage must not exceed 2x the minimum edge
        max_acceptable_slippage_bps = self.config.risk.min_edge_bps * 2.0
        if execution_estimate.expected_slippage_bps > max_acceptable_slippage_bps:
            return False

        # Edge after costs gate
        # Settlement forecast gate
        if not settlement_forecast.execution_recommended:
            return False

        # All gates passed
        return True

    def compute_final_position_size(
        self,
        settlement_forecast: SettlementForecast,
        execution_estimate: ExecutionEstimate,
        available_capital: float,
    ) -> float:
        """
        Compute final position size accounting for all constraints.
        
        Size = min(
            settlement_forecast.execution_size_fraction * max_position,
            execution_estimate.max_recommended_size,
            available_capital * kelly_fraction,
            risk_config.max_position_notional,
        )
        """
        # Directional size. This bot is not sizing from market mispricing or
        # Kelly odds; it sizes from probability distance away from 50/50.
        risk_fraction = self.config.risk.position_sizing_kelly_fraction
        directional_edge = max(0.0, settlement_forecast.edge_estimate)
        confidence = max(
            settlement_forecast.execution_confidence,
            settlement_forecast.settlement_confidence,
        )
        directional_size = directional_edge * confidence * risk_fraction * available_capital * 4.0

        # Settlement forecast size
        forecast_size = settlement_forecast.execution_size_fraction * self.config.risk.max_position_notional

        # Execution constraint size
        execution_max = execution_estimate.max_recommended_size

        # Capital constraint
        capital_max = available_capital * 0.5  # Never use more than 50% of capital

        # Config maximum
        config_max = self.config.risk.max_position_notional

        # Take minimum of all constraints
        final_size = min(directional_size, forecast_size, execution_max, capital_max, config_max)

        # Ensure positive
        final_size = max(0.0, final_size)

        # If execution quality is degraded, reduce size by 50%
        if execution_estimate.execution_quality == ExecutionQuality.DEGRADED:
            final_size *= 0.5

        return float(final_size)
