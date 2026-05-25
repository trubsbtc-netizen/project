from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from polymarket_bot.infrastructure.config import InfrastructureConfig
from polymarket_bot.infrastructure.network import ConnectionPool
from polymarket_bot.infrastructure.types import (
    ChainlinkData,
    PriceSnapshot,
    RingBuffer,
    _safe_float,
    _utc_timestamp,
)
from polymarket_bot.infrastructure.ws import ManagedWebSocketClient

logger = logging.getLogger(__name__)


class RTDSChainlinkFeed(ManagedWebSocketClient):
    RTDS_WS_URL = "wss://ws-live-data.polymarket.com"

    def __init__(self, config: InfrastructureConfig):
        super().__init__("rtds_chainlink", self.RTDS_WS_URL, config, config.chainlink_stale_s)
        self._last_data: Optional[ChainlinkData] = None
        self._buffer = RingBuffer[ChainlinkData](900)
        self._lock = asyncio.Lock()

    async def _on_connect(self) -> None:
        await self.send_json(
            {
                "action": "subscribe",
                "subscriptions": [
                    {
                        "topic": "crypto_prices_chainlink",
                        "type": "*",
                        "filters": '{"symbol":"btc/usd"}',
                    }
                ],
            }
        )

    async def _on_raw_message(self, raw: str) -> None:
        text = raw.strip()
        if not text or text in {"PING", "PONG"}:
            return
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return
        ticks = self._extract_ticks(payload)
        if not ticks:
            return
        now = time.time()
        async with self._lock:
            for source_ts, price, tick_raw in ticks:
                data = ChainlinkData(
                    price=price,
                    updated_at=source_ts,
                    round_id=int(source_ts * 1000),
                    lag_seconds=max(0.0, now - source_ts),
                    raw=tick_raw,
                )
                if self._last_data is None or data.updated_at >= self._last_data.updated_at:
                    self._last_data = data
                self._buffer.append(data)

    @property
    def last(self) -> Optional[ChainlinkData]:
        return self._last_data

    async def fetch(self) -> Optional[ChainlinkData]:
        async with self._lock:
            return self._last_data

    async def fetch_price_at_or_after(
        self,
        target_ts: float,
        *,
        tolerance_after_s: float = 300.0,
    ) -> Optional[ChainlinkData]:
        async with self._lock:
            candidates = [
                item
                for item in self._buffer.snapshot()
                if item.updated_at >= target_ts and item.updated_at - target_ts <= tolerance_after_s
            ]
            if not candidates:
                return None
            return min(candidates, key=lambda item: item.updated_at)

    def _extract_ticks(self, message: Dict[str, Any]) -> List[Tuple[float, float, Dict[str, Any]]]:
        if not isinstance(message, dict) or message.get("topic") != "crypto_prices_chainlink":
            return []
        payload = message.get("payload") or {}
        if not isinstance(payload, dict):
            return []

        rows = payload.get("data")
        if isinstance(rows, list):
            source_rows = [row for row in rows if isinstance(row, dict)]
        else:
            source_rows = [payload]

        ticks: List[Tuple[float, float, Dict[str, Any]]] = []
        for row in source_rows:
            symbol = str(row.get("symbol") or payload.get("symbol") or "").lower()
            if symbol and symbol not in {"btc/usd", "btcusd", "btc-usd"}:
                continue
            ts = (
                _utc_timestamp(row.get("timestamp"))
                or _utc_timestamp(row.get("ts"))
                or _utc_timestamp(row.get("updatedAt"))
                or _utc_timestamp(row.get("updated_at"))
                or _utc_timestamp(row.get("validAfterTs"))
                or _utc_timestamp(row.get("valid_after_ts"))
                or _utc_timestamp(row.get("time"))
            )
            if ts is None:
                ts = time.time()
            elif ts > 1_000_000_000_000:
                ts /= 1000.0
            price = None
            for key in ("price", "value", "benchmark", "benchmarkValue", "mid", "answer"):
                val = row.get(key)
                parsed = _safe_float(val, 0.0)
                if parsed > 0:
                    price = parsed
                    break
            if price and price > 0:
                ticks.append((float(ts), float(price), dict(row)))
        return ticks


import asyncio


class BinanceFeed(ManagedWebSocketClient):
    WS_URL = "wss://data-stream.binance.vision/ws/btcusdt@bookTicker"
    REST_URL = "https://data-api.binance.vision/api/v3/ticker/bookTicker"

    def __init__(self, config: InfrastructureConfig, pool: ConnectionPool):
        super().__init__("binance_btcusdt", self.WS_URL, config, config.price_feed_stale_s)
        self.pool = pool
        self._last: Optional[PriceSnapshot] = None

    @property
    def last(self) -> Optional[PriceSnapshot]:
        return self._last

    async def start(self) -> None:
        await self.seed()
        await super().start()

    async def seed(self) -> None:
        try:
            session = await self.pool.session()
            async with session.get(self.REST_URL, params={"symbol": "BTCUSDT"}) as resp:
                if resp.status != 200:
                    return
                payload = await resp.json(content_type=None)
            bid = _safe_float(payload.get("bidPrice"), 0.0)
            ask = _safe_float(payload.get("askPrice"), 0.0)
            price = (bid + ask) * 0.5 if bid > 0 and ask > 0 else max(bid, ask)
            if price > 0:
                self._last = PriceSnapshot("binance", price, time.time(), bid=bid, ask=ask, raw=payload)
                self.last_rx = self._last.timestamp
        except Exception as exc:
            self.last_error = repr(exc)

    async def _on_raw_message(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        bid = _safe_float(payload.get("b"), 0.0)
        ask = _safe_float(payload.get("a"), 0.0)
        event_ts = _safe_float(payload.get("E"), 0.0)
        ts = event_ts / 1000.0 if event_ts > 1_000_000_000_000 else time.time()
        price = (bid + ask) * 0.5 if bid > 0 and ask > 0 else max(bid, ask)
        if price > 0:
            self._last = PriceSnapshot("binance", price, ts, bid=bid, ask=ask, raw=payload)


class CoinbaseFeed(ManagedWebSocketClient):
    WS_URL = "wss://ws-feed.exchange.coinbase.com"

    def __init__(self, config: InfrastructureConfig):
        super().__init__("coinbase_btcusd", self.WS_URL, config, config.price_feed_stale_s)
        self._last: Optional[PriceSnapshot] = None
        self._doh_resolver = None

    @property
    def last(self) -> Optional[PriceSnapshot]:
        return self._last

    async def _on_connect(self) -> None:
        await self.send_json(
            {
                "type": "subscribe",
                "product_ids": ["BTC-USD"],
                "channels": ["ticker"],
            }
        )

    async def start(self) -> None:
        if self._doh_resolver is None:
            from polymarket_bot.infrastructure.network import DoHResolver

            self._doh_resolver = DoHResolver()
        await super().start()

    async def _on_raw_message(self, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return
        if payload.get("type") not in {"ticker", "snapshot"}:
            return
        price = _safe_float(payload.get("price"), 0.0)
        bid = _safe_float(payload.get("best_bid"), 0.0)
        ask = _safe_float(payload.get("best_ask"), 0.0)
        if price <= 0 and bid > 0 and ask > 0:
            price = (bid + ask) * 0.5
        ts = _utc_timestamp(payload.get("time")) or time.time()
        if price > 0:
            self._last = PriceSnapshot("coinbase", price, ts, bid=bid, ask=ask, raw=payload)
