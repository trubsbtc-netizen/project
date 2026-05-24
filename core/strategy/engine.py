from __future__ import annotations

from dataclasses import dataclass

from core.config import ExecutionConfig, StrategyConfig
from core.microstructure.features import MicrostructureEngine
from core.probability.fusion import BayesianEvidenceFusion
from core.risk.manager import RiskDecision, RiskManager
from core.types import PTBMarket, PositionState, ProbabilityEstimate


@dataclass(slots=True, frozen=True)
class StrategyOutput:
    estimate: ProbabilityEstimate
    risk: RiskDecision


class ProbabilisticStrategyEngine:
    def __init__(
        self,
        strategy_config: StrategyConfig,
        execution_config: ExecutionConfig,
        microstructure: MicrostructureEngine,
    ) -> None:
        self._fusion = BayesianEvidenceFusion(strategy_config)
        self._risk = RiskManager(strategy_config, execution_config)
        self._microstructure = microstructure
        self._last_estimate: ProbabilityEstimate | None = None

    @property
    def last_estimate(self) -> ProbabilityEstimate | None:
        return self._last_estimate

    def update_settlement_label(self, predicted_p_up: float, outcome_up: bool) -> None:
        self._fusion.update_settlement_label(predicted_p_up, outcome_up)

    def evaluate(
        self,
        ts_ns: int,
        market: PTBMarket,
        seconds_to_expiry: float,
        position: PositionState | None,
    ) -> StrategyOutput | None:
        snap = self._microstructure.snapshot(
            ts_ns=ts_ns,
            seconds_to_expiry=seconds_to_expiry,
            price_to_beat=market.price_to_beat.value,
        )
        if snap is None:
            return None
        estimate = self._fusion.estimate(snap)
        self._last_estimate = estimate
        risk = self._risk.evaluate(
            market=market,
            estimate=estimate,
            books=self._microstructure.books,
            seconds_to_expiry=seconds_to_expiry,
            position=position,
        )
        return StrategyOutput(estimate=estimate, risk=risk)

    def mark_submitted(self, bucket: int) -> None:
        self._risk.mark_submitted(bucket)

