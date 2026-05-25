from __future__ import annotations

import time
from typing import Dict, Optional, Tuple

from polymarket_bot.infrastructure.discovery import MarketDiscovery
from polymarket_bot.infrastructure.feeds import RTDSChainlinkFeed
from polymarket_bot.infrastructure.types import MarketInfo, PTBRecord


class PTBLifecycle:
    def __init__(self, discovery: MarketDiscovery, chainlink: RTDSChainlinkFeed):
        self.discovery = discovery
        self.chainlink = chainlink
        self._cache: Dict[str, PTBRecord] = {}

    def get(self, slug: str) -> Optional[PTBRecord]:
        record = self._cache.get(slug)
        return record

    def get_ptb(self, slug: str) -> Optional[PTBRecord]:
        return self.get(slug)

    def records(self) -> Tuple[PTBRecord, ...]:
        return tuple(self._cache.values())

    async def resolve(self, market: MarketInfo) -> Tuple[MarketInfo, Optional[PTBRecord]]:
        existing = self._cache.get(market.slug)
        if existing:
            return market.with_price_to_beat(existing.price), existing

        now = time.time()
        if market.price_to_beat > 1000:
            record = PTBRecord(
                slug=market.slug,
                price=market.price_to_beat,
                captured_at=now,
                source="market_metadata",
                event_start_time=market.event_start_time,
                raw=market.raw,
            )
            self._cache[market.slug] = record
            return market, record

        refreshed = await self.discovery.refresh_price_to_beat(market.slug)
        if refreshed and refreshed > 0:
            record = PTBRecord(
                slug=market.slug,
                price=refreshed,
                captured_at=now,
                source="polymarket_metadata",
                event_start_time=market.event_start_time,
            )
            self._cache[market.slug] = record
            return market.with_price_to_beat(refreshed), record

        chainlink_data = await self.chainlink.fetch_price_at_or_after(market.event_start_time)
        if chainlink_data and chainlink_data.price > 0:
            record = PTBRecord(
                slug=market.slug,
                price=chainlink_data.price,
                captured_at=now,
                source="rtds_crypto_prices_chainlink",
                event_start_time=market.event_start_time,
                source_timestamp=chainlink_data.updated_at,
                raw=chainlink_data.raw,
            )
            self._cache[market.slug] = record
            return market.with_price_to_beat(chainlink_data.price), record

        return market, None
