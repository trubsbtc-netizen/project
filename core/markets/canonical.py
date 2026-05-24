from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from core.runtime.clock import bucket_5m, mono_ns
from core.types import PTBMarket, PositionState, PriceTick, SettlementTruth, TruthSource


@dataclass(slots=True)
class CanonicalRoundState:
    market: PTBMarket
    truth_price: PriceTick | None = None
    settlement: SettlementTruth | None = None
    position: PositionState | None = None

    @property
    def bucket(self) -> int:
        return self.market.bucket

    @property
    def seconds_to_expiry(self) -> float:
        now_s = time.time()
        return max(0.0, self.market.close_ts - now_s)

    @property
    def price_to_beat(self) -> float:
        return self.market.price_to_beat.value


class CanonicalMarketState:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._round: CanonicalRoundState | None = None
        self._archive: dict[int, CanonicalRoundState] = {}

    async def install_round(self, market: PTBMarket) -> bool:
        async with self._lock:
            if self._round is not None and self._round.bucket == market.bucket:
                return False
            if self._round is not None:
                self._archive[self._round.bucket] = self._round
            self._round = CanonicalRoundState(market=market)
            return True

    async def update_truth(self, tick: PriceTick) -> None:
        if tick.source is not TruthSource.RTDS_CHAINLINK:
            raise ValueError("canonical truth price can only be RTDS Chainlink")
        async with self._lock:
            if self._round is not None:
                self._round.truth_price = tick

    async def update_settlement(self, settlement: SettlementTruth) -> None:
        async with self._lock:
            if self._round is not None and self._round.market.condition_id == settlement.condition_id:
                self._round.settlement = settlement
                return
            for state in self._archive.values():
                if state.market.condition_id == settlement.condition_id:
                    state.settlement = settlement
                    return

    async def update_position(self, position: PositionState) -> None:
        async with self._lock:
            if self._round is not None and self._round.bucket == position.bucket:
                self._round.position = position

    async def current(self) -> CanonicalRoundState | None:
        async with self._lock:
            return self._round

    async def current_bucket(self) -> int:
        state = await self.current()
        if state is not None:
            return state.bucket
        return bucket_5m()

    async def archive(self) -> dict[int, CanonicalRoundState]:
        async with self._lock:
            return dict(self._archive)
