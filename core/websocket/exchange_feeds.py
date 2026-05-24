from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from core.config import WebsocketConfig
from core.types import OrderSide, SignalSource, TopOfBook, TradeTick
from core.websocket.client import ResilientWebSocketClient, WebSocketSpec


ExchangeHandler = Callable[[TopOfBook | TradeTick], Awaitable[None]]


class BinanceFeed:
    def __init__(self, config: WebsocketConfig, heartbeat, handler: ExchangeHandler) -> None:
        self._config = config
        self._heartbeat = heartbeat
        self._handler = handler

    def _subscriptions(self) -> list[dict[str, Any]]:
        return []

    async def _handle(self, payload: Any, recv_ns: int) -> None:
        event = parse_binance(payload, recv_ns)
        if event is not None:
            await self._handler(event)

    async def run(self) -> None:
        spec = WebSocketSpec(
            name="binance.btc",
            url=self._config.binance_ws,
            heartbeat_s=self._config.heartbeat_s,
            connect_timeout_s=self._config.connect_timeout_s,
            max_backoff_s=self._config.max_backoff_s,
        )
        await ResilientWebSocketClient(spec, self._heartbeat, self._subscriptions, self._handle).run()


class CoinbaseFeed:
    def __init__(self, config: WebsocketConfig, heartbeat, handler: ExchangeHandler) -> None:
        self._config = config
        self._heartbeat = heartbeat
        self._handler = handler

    def _subscriptions(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "subscribe",
                "product_ids": ["BTC-USD"],
                "channels": ["ticker", "matches", "heartbeats"],
            }
        ]

    async def _handle(self, payload: Any, recv_ns: int) -> None:
        event = parse_coinbase(payload, recv_ns)
        if event is not None:
            await self._handler(event)

    async def run(self) -> None:
        spec = WebSocketSpec(
            name="coinbase.btc",
            url=self._config.coinbase_ws,
            heartbeat_s=self._config.heartbeat_s,
            connect_timeout_s=self._config.connect_timeout_s,
            max_backoff_s=self._config.max_backoff_s,
            doh_resolve=self._config.coinbase_use_doh,
            doh_resolver_url=self._config.doh_resolver_url,
            doh_timeout_s=self._config.doh_timeout_s,
        )
        await ResilientWebSocketClient(spec, self._heartbeat, self._subscriptions, self._handle).run()


def parse_binance(payload: Any, recv_ns: int) -> TopOfBook | TradeTick | None:
    if not isinstance(payload, dict):
        return None
    data = payload.get("data", payload)
    if not isinstance(data, dict):
        return None
    event_type = data.get("e")
    if event_type == "trade" or "p" in data and "q" in data and ("m" in data or "T" in data):
        try:
            side = OrderSide.SELL if bool(data.get("m")) else OrderSide.BUY
            return TradeTick(
                source=SignalSource.BINANCE,
                symbol="BTCUSDT",
                price=float(data["p"]),
                size=float(data["q"]),
                side=side,
                exchange_ts_ms=int(data.get("T") or data.get("E") or 0),
                recv_mono_ns=recv_ns,
                trade_id=str(data.get("t", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None
    if event_type == "bookTicker" or all(k in data for k in ("b", "a", "B", "A")):
        try:
            return TopOfBook(
                source=SignalSource.BINANCE,
                symbol="BTCUSDT",
                bid=float(data["b"]),
                ask=float(data["a"]),
                bid_size=float(data["B"]),
                ask_size=float(data["A"]),
                exchange_ts_ms=int(data.get("E") or data.get("u") or 0),
                recv_mono_ns=recv_ns,
            )
        except (KeyError, TypeError, ValueError):
            return None
    if "bids" in data and "asks" in data:
        try:
            bid = data["bids"][0]
            ask = data["asks"][0]
            return TopOfBook(
                source=SignalSource.BINANCE,
                symbol="BTCUSDT",
                bid=float(bid[0]),
                ask=float(ask[0]),
                bid_size=float(bid[1]),
                ask_size=float(ask[1]),
                exchange_ts_ms=int(data.get("E") or 0),
                recv_mono_ns=recv_ns,
            )
        except (IndexError, KeyError, TypeError, ValueError):
            return None
    return None


def parse_coinbase(payload: Any, recv_ns: int) -> TopOfBook | TradeTick | None:
    if not isinstance(payload, dict):
        return None
    msg_type = payload.get("type")
    if msg_type == "ticker":
        try:
            bid = float(payload["best_bid"])
            ask = float(payload["best_ask"])
            bid_size = float(payload.get("best_bid_size") or payload.get("last_size") or 0.0)
            ask_size = float(payload.get("best_ask_size") or payload.get("last_size") or 0.0)
            return TopOfBook(
                source=SignalSource.COINBASE,
                symbol="BTC-USD",
                bid=bid,
                ask=ask,
                bid_size=bid_size,
                ask_size=ask_size,
                exchange_ts_ms=0,
                recv_mono_ns=recv_ns,
            )
        except (KeyError, TypeError, ValueError):
            return None
    if msg_type in {"match", "last_match"}:
        try:
            side = OrderSide.BUY if str(payload.get("side")).lower() == "buy" else OrderSide.SELL
            return TradeTick(
                source=SignalSource.COINBASE,
                symbol="BTC-USD",
                price=float(payload["price"]),
                size=float(payload["size"]),
                side=side,
                exchange_ts_ms=0,
                recv_mono_ns=recv_ns,
                trade_id=str(payload.get("trade_id", "")),
            )
        except (KeyError, TypeError, ValueError):
            return None
    return None
