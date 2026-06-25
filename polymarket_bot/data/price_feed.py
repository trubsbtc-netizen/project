"""Price feed adapter: wraps BTC_POLY BinanceFeed/CoinbaseFeed into the original API."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
import numpy as np

from polymarket_bot.config import BotConfig
from polymarket_bot.infrastructure.config import InfrastructureConfig
from polymarket_bot.infrastructure.engine import InfrastructureEngine
from polymarket_bot.infrastructure.feeds import BinanceFeed, CoinbaseFeed
from polymarket_bot.infrastructure.network import ConnectionPool
from polymarket_bot.infrastructure.types import PriceSnapshot as InfraPriceSnapshot
from polymarket_bot.infrastructure.types import OrderBookSnapshot as InfraOrderBookSnapshot
from polymarket_bot.bot_types import (
    OrderbookDelta,
    OrderbookLevel,
    OrderbookSnapshot,
    OrderbookBuffer,
    TickBuffer,
    TickData,
)

logger = logging.getLogger(__name__)


class MultiExchangePriceFeed:
    """Backward-compatible price feed backed by BTC_POLY BinanceFeed + CoinbaseFeed."""

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.symbol: str = config.btc_price_symbol.lower()
        self.coinbase_product_id: str = getattr(config, "coinbase_product_id", "BTC-USD")
        self.settlement_price_source: str = getattr(config, "settlement_price_source", "coinbase").lower()
        self.tick_buffer = TickBuffer(max_size=10000)
        self.orderbook_buffer = OrderbookBuffer(max_snapshots=100, max_deltas=5000)

        self._running: bool = False
        self._tasks: List[asyncio.Task] = []

        self._latest_price: Optional[float] = None
        self._latest_price_source: str = "none"
        self._latest_price_time: float = 0.0
        self._settlement_price: Optional[float] = None
        self._settlement_price_source: str = "none"
        self._settlement_price_time: float = 0.0

        self._price_binance: Optional[float] = None
        self._price_coinbase: Optional[float] = None
        self._source_prices: Dict[str, Tuple[float, float]] = {}
        self._source_message_counts: Dict[str, int] = {}
        self._pending_ticks = deque(maxlen=5000)

        self._infra_config = InfrastructureConfig.from_env()
        self._pool = ConnectionPool(self._infra_config)
        self._binance_feed = BinanceFeed(self._infra_config, self._pool)
        self._coinbase_feed = CoinbaseFeed(self._infra_config)
        self._external_engine: Optional[InfrastructureEngine] = None
        self._owns_feeds = True
        self._last_infra_binance: Optional[InfraPriceSnapshot] = None
        self._last_infra_coinbase: Optional[InfraPriceSnapshot] = None

        self._books: Dict[str, Dict[str, Dict[float, float]]] = {
            "binance": {"bids": {}, "asks": {}},
            "coinbase": {"bids": {}, "asks": {}},
        }
        self._book_update_ids: Dict[str, int] = {"binance": 0, "coinbase": 0}
        self._latest_orderbook_by_exchange: Dict[str, OrderbookSnapshot] = {}

        self._tick_count: int = 0
        self._book_count: int = 0
        self._delta_count: int = 0
        self._error_count: int = 0
        self._reconnect_counts: Dict[str, int] = {}
        self._max_price_age_seconds: float = float(getattr(config, "btc_price_max_age_seconds", 8.0))

    def attach_infrastructure_engine(self, engine: InfrastructureEngine) -> None:
        """Reuse the canonical infrastructure engine instead of opening duplicate exchange sockets."""
        self._external_engine = engine
        self._infra_config = engine.config
        self._pool = engine.pool
        self._binance_feed = engine.binance
        self._coinbase_feed = engine.coinbase
        self._owns_feeds = False

    async def connect(self) -> None:
        if self._running:
            return
        self._running = True
        if self._owns_feeds:
            await self._pool.start()
            await self._binance_feed.start()
            await self._coinbase_feed.start()
        elif self._external_engine is not None and not self._external_engine.running:
            await self._external_engine.start()
        await self._sync_from_infra()
        self._tasks = [
            asyncio.create_task(self._poll_infra()),
        ]
        logger.info("MultiExchangePriceFeed: using BTC_POLY BinanceFeed + CoinbaseFeed")

    async def disconnect(self) -> None:
        self._running = False
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        if self._owns_feeds:
            await self._binance_feed.stop()
            await self._coinbase_feed.stop()
            await self._pool.close()

    def get_current_price(self) -> Optional[float]:
        if self._latest_price is None or self._latest_price_time <= 0.0 or time.time() - self._latest_price_time > self._max_price_age_seconds:
            return None
        return self._latest_price

    def get_current_price_source(self) -> str:
        age = self.get_current_price_age()
        if age is not None and age > self._max_price_age_seconds:
            return f"stale:{self._latest_price_source}"
        return self._latest_price_source

    def get_current_price_age(self) -> Optional[float]:
        if self._latest_price_time <= 0.0:
            return None
        return max(0.0, time.time() - self._latest_price_time)

    def get_current_settlement_price(self) -> Optional[float]:
        now = time.time()
        if self._settlement_price is not None and self._settlement_price_time > 0.0 and now - self._settlement_price_time <= self._max_price_age_seconds:
            return self._settlement_price
        return self.get_current_price()

    def get_current_settlement_price_source(self) -> str:
        age = self.get_current_settlement_price_age()
        if age is not None and age > self._max_price_age_seconds:
            source = self._settlement_price_source or self._latest_price_source
            return f"stale:{source}"
        return self._settlement_price_source or self._latest_price_source

    def get_current_settlement_price_age(self) -> Optional[float]:
        timestamp = self._settlement_price_time or self._latest_price_time
        if timestamp <= 0.0:
            return None
        return max(0.0, time.time() - timestamp)

    def get_current_orderbook(self) -> Optional[OrderbookSnapshot]:
        now = time.time()
        preferred = self._latest_orderbook_by_exchange.get(self.settlement_price_source)
        if preferred and now - preferred.timestamp <= 10.0 and preferred.bids and preferred.asks:
            return preferred
        fresh = [
            snapshot
            for snapshot in self._latest_orderbook_by_exchange.values()
            if now - snapshot.timestamp <= 10.0 and snapshot.bids and snapshot.asks
        ]
        if fresh:
            return max(fresh, key=lambda snapshot: snapshot.timestamp)
        if self.orderbook_buffer.snapshots:
            return self.orderbook_buffer.snapshots[-1]
        return None

    def drain_recent_ticks(self, limit: int = 500) -> List[TickData]:
        ticks: List[TickData] = []
        while self._pending_ticks and len(ticks) < limit:
            ticks.append(self._pending_ticks.popleft())
        return ticks

    def recent_orderbook_deltas(self, limit: int = 1000) -> List[OrderbookDelta]:
        return self.orderbook_buffer.recent_deltas(limit)

    async def _poll_infra(self) -> None:
        while self._running:
            try:
                await self._sync_from_infra()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Infra price sync error: %s", exc)
            await asyncio.sleep(0.05)

    async def _sync_from_infra(self) -> None:
        binance = self._binance_feed.last
        coinbase = self._coinbase_feed.last
        changed = False
        if binance is not None and (self._last_infra_binance is None or binance.timestamp != self._last_infra_binance.timestamp):
            self._last_infra_binance = binance
            self._update_price(binance.price, "binance_bookTicker")
            ts = time.time()
            snapshot = OrderbookSnapshot(
                timestamp=ts,
                bids=[OrderbookLevel(price=binance.bid, size=1.0, timestamp=ts, is_real=True)] if binance.bid > 0 else [],
                asks=[OrderbookLevel(price=binance.ask, size=1.0, timestamp=ts, is_real=True)] if binance.ask > 0 else [],
            )
            self._store_orderbook_snapshot("binance", snapshot)
            changed = True
        if coinbase is not None and (self._last_infra_coinbase is None or coinbase.timestamp != self._last_infra_coinbase.timestamp):
            self._last_infra_coinbase = coinbase
            self._update_price(coinbase.price, "coinbase_ticker")
            ts = time.time()
            snapshot = OrderbookSnapshot(
                timestamp=ts,
                bids=[OrderbookLevel(price=coinbase.bid, size=1.0, timestamp=ts, is_real=True)] if coinbase.bid > 0 else [],
                asks=[OrderbookLevel(price=coinbase.ask, size=1.0, timestamp=ts, is_real=True)] if coinbase.ask > 0 else [],
            )
            self._store_orderbook_snapshot("coinbase", snapshot)
            changed = True

    def _exchange_name(self, source: str) -> str:
        if source.startswith("binance"):
            return "binance"
        if source.startswith("coinbase"):
            return "coinbase"
        return source

    def _fresh_exchange_prices(self, now: Optional[float] = None, max_age_seconds: float = 5.0) -> Dict[str, float]:
        now = time.time() if now is None else now
        grouped: Dict[str, List[float]] = {}
        for source, (price, timestamp) in self._source_prices.items():
            if now - timestamp > max_age_seconds:
                continue
            grouped.setdefault(self._exchange_name(source), []).append(price)
        return {
            exchange: float(np.median(prices))
            for exchange, prices in grouped.items()
            if prices
        }

    def _update_price(self, price: float, source: str) -> None:
        now = time.time()
        if not np.isfinite(price) or price <= 0:
            return
        self._source_prices[source] = (float(price), now)
        self._source_message_counts[source] = self._source_message_counts.get(source, 0) + 1
        if source.startswith("binance"):
            self._price_binance = price
        elif source.startswith("coinbase"):
            self._price_coinbase = price
        fresh_prices = self._fresh_exchange_prices(now)
        if fresh_prices:
            prices = list(fresh_prices.values())
            self._latest_price = float(np.median(prices))
            if len(fresh_prices) > 1:
                self._latest_price_source = "multi_exchange(" + ",".join(
                    f"{exchange}={value:.2f}" for exchange, value in sorted(fresh_prices.items())
                ) + ")"
            else:
                exchange = next(iter(fresh_prices))
                self._latest_price_source = f"{exchange}:{source}"
            self._latest_price_time = now
            preferred = self.settlement_price_source
            if preferred in fresh_prices:
                self._settlement_price = fresh_prices[preferred]
                self._settlement_price_source = f"{preferred}:settlement"
            else:
                self._settlement_price = self._latest_price
                self._settlement_price_source = f"fallback:{self._latest_price_source}"
            self._settlement_price_time = now

    @staticmethod
    def _parse_coinbase_timestamp(value: Any) -> float:
        if value is None:
            return time.time()
        if isinstance(value, (int, float)):
            timestamp = float(value)
            return timestamp / 1000.0 if timestamp > 10_000_000_000 else timestamp
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return time.time()
            try:
                timestamp = float(raw)
                return timestamp / 1000.0 if timestamp > 10_000_000_000 else timestamp
            except ValueError:
                pass
            try:
                from datetime import datetime, timezone
                if raw.endswith("Z"):
                    raw = raw[:-1] + "+00:00"
                parsed = datetime.fromisoformat(raw)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp()
            except ValueError:
                return time.time()
        return time.time()

    def _canonical_tick_price(self, fallback_price: float) -> float:
        price = self.get_current_settlement_price()
        if price is not None and np.isfinite(price) and price > 0:
            return float(price)
        if self._latest_price is not None and np.isfinite(self._latest_price) and self._latest_price > 0:
            return float(self._latest_price)
        return float(fallback_price)

    def _record_trade_tick(self, tick: TickData, price_override: Optional[float] = None) -> bool:
        if tick.price <= 0 or tick.volume <= 0 or tick.side not in {"buy", "sell"}:
            return False
        canonical_tick = TickData(
            timestamp=tick.timestamp,
            price=float(price_override) if price_override and price_override > 0 else tick.price,
            volume=tick.volume,
            side=tick.side,
            trade_id=tick.trade_id,
            is_snapshot=tick.is_snapshot,
        )
        self.tick_buffer.append(canonical_tick)
        self._pending_ticks.append(canonical_tick)
        return True

    def _store_orderbook_snapshot(self, exchange: str, snapshot: OrderbookSnapshot) -> None:
        self._latest_orderbook_by_exchange[exchange] = snapshot
        self.orderbook_buffer.add_snapshot(snapshot)
        self._book_count += 1

    def _build_snapshot(self, timestamp: float, book: Optional[Dict[str, Dict[float, float]]] = None, sequence_number: int = 0) -> OrderbookSnapshot:
        book = book or self._books.get(self.settlement_price_source) or self._books["binance"]
        depth = self.config.microstructure.orderbook_depth_levels
        bid_items = sorted(book["bids"].items(), key=lambda x: x[0], reverse=True)[:depth]
        ask_items = sorted(book["asks"].items(), key=lambda x: x[0])[:depth]
        return OrderbookSnapshot(
            timestamp=timestamp,
            bids=[OrderbookLevel(price=p, size=s, timestamp=timestamp, is_real=True) for p, s in bid_items if s > 0],
            asks=[OrderbookLevel(price=p, size=s, timestamp=timestamp, is_real=True) for p, s in ask_items if s > 0],
            sequence_number=sequence_number,
        )

    def get_price_history(self, n: int = 100) -> tuple:
        return self.tick_buffer.recent_slice(n)[:2]

    def get_return_history(self, n: int = 100) -> np.ndarray:
        prices = self.tick_buffer.get_price_array()
        if len(prices) < 2:
            return np.array([])
        return np.diff(np.log(prices[-n:]))

    @property
    def is_connected(self) -> bool:
        return self._running and (time.time() - self._latest_price_time < 30.0)

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "tick_count": self._tick_count,
            "book_count": self._book_count,
            "delta_count": self._delta_count,
            "error_count": self._error_count,
            "latest_price": self._latest_price,
            "price_source": self._latest_price_source,
            "settlement_price": self.get_current_settlement_price(),
            "settlement_price_source": self.get_current_settlement_price_source(),
            "price_age_s": time.time() - self._latest_price_time if self._latest_price_time else None,
            "is_connected": self.is_connected,
            "exchange_prices": self._fresh_exchange_prices(),
            "source_counts": dict(self._source_message_counts),
            "buffer_sizes": {
                "ticks": len(self.tick_buffer),
                "snapshots": len(self.orderbook_buffer.snapshots),
                "deltas": len(self.orderbook_buffer.deltas),
            },
        }


BinanceWebSocket = MultiExchangePriceFeed


class RESTPriceFallback:
    """REST API fallback for BTC price."""

    BASE_URL = "https://data-api.binance.vision"

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.symbol: str = config.btc_price_symbol
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        from polymarket_bot.data.network import create_client_session
        self._session = create_client_session(
            base_url=self.BASE_URL,
            timeout=aiohttp.ClientTimeout(total=10.0),
        )

    async def stop(self) -> None:
        if self._session:
            await self._session.close()
            self._session = None

    async def get_current_price(self) -> Optional[float]:
        if not self._session:
            return None
        try:
            params = {"symbol": self.symbol}
            async with self._session.get("/api/v3/ticker/price", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return float(data.get("price", 0))
        except Exception as e:
            logger.warning(f"REST price fetch failed: {e}")
        return None

    async def get_recent_trades(self, limit: int = 100) -> List[TickData]:
        if not self._session:
            return []
        try:
            params = {"symbol": self.symbol, "limit": limit}
            async with self._session.get("/api/v3/trades", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [
                        TickData(
                            timestamp=float(t.get("time", 0)) / 1000.0,
                            price=float(t.get("price", 0)),
                            volume=float(t.get("qty", 0)),
                            side="buy" if not t.get("isBuyerMaker", True) else "sell",
                            trade_id=str(t.get("id", "")),
                        )
                        for t in data
                    ]
        except Exception as e:
            logger.warning(f"REST trades fetch failed: {e}")
        return []

    async def get_klines(self, interval: str = "1m", limit: int = 60) -> Optional[List[Dict]]:
        if not self._session:
            return None
        try:
            params = {"symbol": self.symbol, "interval": interval, "limit": limit}
            async with self._session.get("/api/v3/klines", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return [
                        {
                            "open_time": float(k[0]) / 1000.0,
                            "open": float(k[1]),
                            "high": float(k[2]),
                            "low": float(k[3]),
                            "close": float(k[4]),
                            "volume": float(k[5]),
                            "close_time": float(k[6]) / 1000.0,
                            "quote_volume": float(k[7]),
                            "trades": int(k[8]),
                        }
                        for k in data
                    ]
        except Exception as e:
            logger.warning(f"REST klines fetch failed: {e}")
        return None

    async def get_orderbook_snapshot(self, limit: int = 20) -> Optional[OrderbookSnapshot]:
        if not self._session:
            return None
        try:
            params = {"symbol": self.symbol, "limit": limit}
            async with self._session.get("/api/v3/depth", params=params) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    ts = time.time()
                    bids = [OrderbookLevel(price=float(b[0]), size=float(b[1]), timestamp=ts, is_real=True) for b in data.get("bids", [])]
                    asks = [OrderbookLevel(price=float(a[0]), size=float(a[1]), timestamp=ts, is_real=True) for a in data.get("asks", [])]
                    bids.sort(key=lambda x: x.price, reverse=True)
                    asks.sort(key=lambda x: x.price)
                    return OrderbookSnapshot(timestamp=ts, bids=bids, asks=asks)
        except Exception as e:
            logger.warning(f"REST orderbook fetch failed: {e}")
        return None
