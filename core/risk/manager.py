from __future__ import annotations

from dataclasses import dataclass

from core.config import ExecutionConfig, StrategyConfig
from core.orderbook.top import CompositeTopOfBook, TopOfBookState
from core.probability.math import clamp
from core.types import OrderIntent, OrderSide, Outcome, PTBMarket, PositionState, ProbabilityEstimate


@dataclass(slots=True, frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    fill_probability: float
    slippage_bps: float
    intent: OrderIntent | None = None


class ExecutionUncertaintyModel:
    def __init__(self, execution: ExecutionConfig) -> None:
        self._execution = execution

    def evaluate_buy(self, book: TopOfBookState, estimate: ProbabilityEstimate) -> tuple[float, float, float]:
        tob = book.last
        if tob is None or tob.ask <= 0.0 or tob.ask_size <= 0.0:
            return 0.0, 10_000.0, 1.0
        spread_bps = (tob.spread / max(1e-9, tob.mid)) * 10_000.0
        size_pressure = min(1.0, self._execution.order_size_usd / max(1e-9, tob.ask_size))
        latency_penalty = clamp(self._execution.latency_budget_ms / max(1.0, estimate.diagnostics.get("latency_ms", 120.0)), 0.25, 1.0)
        stability = 0.45 * book.spread_stability + 0.35 * book.queue_survival + 0.20 * book.liquidity_stability
        fill_probability = clamp((0.18 + 0.82 * stability) * (1.0 - 0.55 * size_pressure) * latency_penalty, 0.0, 1.0)
        slippage_bps = spread_bps * (0.5 + size_pressure) + 18.0 * (1.0 - stability)
        uncertainty = clamp(1.0 - fill_probability + min(1.0, slippage_bps / max(1.0, self._execution.max_slippage_bps)) * 0.35, 0.0, 1.0)
        return fill_probability, slippage_bps, uncertainty


class RiskManager:
    def __init__(self, strategy: StrategyConfig, execution: ExecutionConfig) -> None:
        self._strategy = strategy
        self._execution = execution
        self._uncertainty = ExecutionUncertaintyModel(execution)
        self._last_bucket: int | None = None
        self._traded_buckets: set[int] = set()

    def evaluate(
        self,
        market: PTBMarket,
        estimate: ProbabilityEstimate,
        books: CompositeTopOfBook,
        seconds_to_expiry: float,
        position: PositionState | None,
    ) -> RiskDecision:
        if market.bucket in self._traded_buckets:
            return RiskDecision(False, "bucket_already_traded", 0.0, 0.0)
        if seconds_to_expiry < self._strategy.min_seconds_to_expiry:
            return RiskDecision(False, "expiry_too_close", 0.0, 0.0)
        if seconds_to_expiry > self._strategy.max_seconds_after_open:
            return RiskDecision(False, "round_not_live", 0.0, 0.0)
        if estimate.confidence < self._strategy.min_confidence:
            return RiskDecision(False, "confidence", 0.0, 0.0)
        if estimate.posterior_uncertainty > self._strategy.max_posterior_uncertainty:
            return RiskDecision(False, "posterior_uncertainty", 0.0, 0.0)
        if estimate.entropy > self._strategy.max_entropy:
            return RiskDecision(False, "entropy", 0.0, 0.0)
        if estimate.reversal_probability > estimate.continuation_probability + 0.16:
            return RiskDecision(False, "reversal_hazard", 0.0, 0.0)
        if position is not None and position.gross_notional >= self._execution.max_notional_usd:
            return RiskDecision(False, "max_notional", 0.0, 0.0)

        outcome = Outcome.UP if estimate.p_up >= estimate.p_down else Outcome.DOWN
        book = books.polymarket_up if outcome is Outcome.UP else books.polymarket_down
        tob = book.last
        if tob is None or tob.ask <= 0.0:
            return RiskDecision(False, "no_polymarket_offer", 0.0, 0.0)
        if tob.ask_size < self._execution.min_quote_size:
            return RiskDecision(False, "quote_size", 0.0, 0.0)
        fair = estimate.fair_price_up if outcome is Outcome.UP else estimate.fair_price_down
        executable_edge = fair - tob.ask
        if executable_edge < self._strategy.min_edge:
            return RiskDecision(False, "edge", 0.0, 0.0)
        fill_probability, slippage_bps, execution_uncertainty = self._uncertainty.evaluate_buy(book, estimate)
        if fill_probability < self._execution.min_fill_probability:
            return RiskDecision(False, "fill_probability", fill_probability, slippage_bps)
        if slippage_bps > self._execution.max_slippage_bps:
            return RiskDecision(False, "slippage", fill_probability, slippage_bps)
        if execution_uncertainty + estimate.posterior_uncertainty > 1.10:
            return RiskDecision(False, "execution_uncertainty", fill_probability, slippage_bps)

        size = min(self._execution.order_size_usd / max(1e-9, tob.ask), tob.ask_size)
        limit_price = clamp(tob.ask * (1.0 + self._execution.max_slippage_bps / 10_000.0), 0.01, 0.99)
        intent = OrderIntent(
            bucket=market.bucket,
            market=market.slug,
            condition_id=market.condition_id,
            outcome=outcome,
            token_id=market.token_for(outcome),
            side=OrderSide.BUY,
            limit_price=limit_price,
            size=size,
            max_slippage_bps=self._execution.max_slippage_bps,
            ttl_ms=self._execution.order_ttl_ms,
            reason="probabilistic_directional_edge",
            estimate=estimate,
        )
        return RiskDecision(True, "allowed", fill_probability, slippage_bps, intent)

    def mark_submitted(self, bucket: int) -> None:
        self._traded_buckets.add(bucket)

