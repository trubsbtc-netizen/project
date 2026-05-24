from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from core.config import PolymarketConfig, WebsocketConfig
from core.types import OrderSide, Outcome, PTBMarket, SignalSource, TopOfBook, TradeTick
from core.websocket.client import ResilientWebSocketClient, WebSocketSpec


PolyMarketHandler = Callable[[TopOfBook | TradeTick], Awaitable[None]]
PolyUserHandler = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(slots=True)
class MarketSubscriptionState:
    market: PTBMarket | None = None
    last_token_ids: tuple[str, str] | None = None

    def update(self, market: PTBMarket) -> None:
        self.last_token_ids = self.market.token_ids if self.market is not None else None
        self.market = market

    def token_outcome(self, token_id: str) -> Outcome | None:
        if self.market is None:
            return None
        if token_id == self.market.up.token_id:
            return Outcome.UP
        if token_id == self.market.down.token_id:
            return Outcome.DOWN
        return None


class PolymarketMarketFeed:
    def __init__(
        self,
        ws_config: WebsocketConfig,
        subscription_state: MarketSubscriptionState,
        heartbeat,
        handler: PolyMarketHandler,
    ) -> None:
        self._ws_config = ws_config
        self._subscription_state = subscription_state
        self._heartbeat = heartbeat
        self._handler = handler
        self.control_queue: asyncio.Queue[dict[str, Any] | str] = asyncio.Queue(maxsize=128)

    async def resubscribe(self) -> None:
        previous = self._subscription_state.last_token_ids
        if previous:
            await self.control_queue.put({"operation": "unsubscribe", "assets_ids": list(previous)})
        for message in self._subscriptions():
            await self.control_queue.put(message)

    def _subscriptions(self) -> Iterable[dict[str, Any]]:
        market = self._subscription_state.market
        if market is None:
            return []
        return [
            {
                "type": "market",
                "operation": "subscribe",
                "assets_ids": list(market.token_ids),
                "custom_feature_enabled": True,
            }
        ]

    async def _handle(self, payload: Any, recv_ns: int) -> None:
        for event in parse_polymarket_market_payload(payload, recv_ns, self._subscription_state):
            await self._handler(event)

    async def run(self) -> None:
        spec = WebSocketSpec(
            name="polymarket.market",
            url=self._ws_config.polymarket_market_ws,
            heartbeat_s=self._ws_config.heartbeat_s,
            connect_timeout_s=self._ws_config.connect_timeout_s,
            max_backoff_s=self._ws_config.max_backoff_s,
            ping_payload="PING",
        )
        await ResilientWebSocketClient(
            spec,
            self._heartbeat,
            self._subscriptions,
            self._handle,
            control_queue=self.control_queue,
        ).run()


class PolymarketUserFeed:
    def __init__(
        self,
        ws_config: WebsocketConfig,
        poly_config: PolymarketConfig,
        heartbeat,
        handler: PolyUserHandler,
    ) -> None:
        self._ws_config = ws_config
        self._poly_config = poly_config
        self._heartbeat = heartbeat
        self._handler = handler

    def _subscriptions(self) -> Iterable[dict[str, Any]]:
        if not (self._poly_config.builder_api_key and self._poly_config.builder_secret and self._poly_config.builder_passphrase):
            return []
        return [
            {
                "type": "user",
                "auth": {
                    "apiKey": self._poly_config.builder_api_key,
                    "secret": self._poly_config.builder_secret,
                    "passphrase": self._poly_config.builder_passphrase,
                },
            }
        ]

    async def _handle(self, payload: Any, recv_ns: int) -> None:
        if isinstance(payload, dict):
            payload["_recv_mono_ns"] = recv_ns
            await self._handler(payload)
        elif isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    item["_recv_mono_ns"] = recv_ns
                    await self._handler(item)

    async def run(self) -> None:
        spec = WebSocketSpec(
            name="polymarket.user",
            url=self._ws_config.polymarket_user_ws,
            heartbeat_s=self._ws_config.heartbeat_s,
            connect_timeout_s=self._ws_config.connect_timeout_s,
            max_backoff_s=self._ws_config.max_backoff_s,
            ping_payload="PING",
        )
        await ResilientWebSocketClient(spec, self._heartbeat, self._subscriptions, self._handle).run()


def parse_polymarket_market_payload(
    payload: Any,
    recv_ns: int,
    subscription_state: MarketSubscriptionState,
) -> list[TopOfBook | TradeTick]:
    events: list[TopOfBook | TradeTick] = []
    if isinstance(payload, list):
        for item in payload:
            events.extend(parse_polymarket_market_payload(item, recv_ns, subscription_state))
        return events
    if not isinstance(payload, dict):
        return events
    event_type = str(payload.get("event_type") or payload.get("type") or "").lower()
    if event_type in {"book", "orderbook", "snapshot"} or ("bids" in payload and "asks" in payload):
        tob = _parse_book(payload, recv_ns, subscription_state)
        if tob is not None:
            events.append(tob)
    elif event_type == "price_change":
        events.extend(_parse_price_change(payload, recv_ns, subscription_state))
    elif event_type in {"best_bid_ask", "bba"}:
        tob = _parse_best_bid_ask(payload, recv_ns, subscription_state)
        if tob is not None:
            events.append(tob)
    elif event_type in {"last_trade_price", "trade"} or "price" in payload:
        trade = _parse_trade(payload, recv_ns, subscription_state)
        if trade is not None:
            events.append(trade)
    return events


def _parse_book(payload: dict[str, Any], recv_ns: int, subscription_state: MarketSubscriptionState) -> TopOfBook | None:
    asset_id = str(payload.get("asset_id") or payload.get("assetId") or payload.get("token_id") or "")
    outcome = subscription_state.token_outcome(asset_id)
    if outcome is None and asset_id:
        symbol = asset_id
    else:
        symbol = outcome.value if outcome is not None else "UNKNOWN"
    try:
        bid_price, bid_size = _best_level(payload.get("bids"), is_bid=True)
        ask_price, ask_size = _best_level(payload.get("asks"), is_bid=False)
    except (TypeError, ValueError):
        return None
    if bid_price <= 0.0 or ask_price <= 0.0:
        return None
    return TopOfBook(
        source=SignalSource.POLYMARKET_BOOK,
        symbol=symbol,
        bid=bid_price,
        ask=ask_price,
        bid_size=bid_size,
        ask_size=ask_size,
        exchange_ts_ms=_event_ts_ms(payload),
        recv_mono_ns=recv_ns,
    )


def _parse_trade(payload: dict[str, Any], recv_ns: int, subscription_state: MarketSubscriptionState) -> TradeTick | None:
    asset_id = str(payload.get("asset_id") or payload.get("assetId") or payload.get("token_id") or "")
    outcome = subscription_state.token_outcome(asset_id)
    symbol = outcome.value if outcome is not None else (asset_id or "UNKNOWN")
    try:
        price = float(payload.get("price") or payload.get("last_price") or payload.get("last_trade_price"))
        size = float(payload.get("size") or payload.get("amount") or payload.get("last_size") or 0.0)
    except (TypeError, ValueError):
        return None
    side_raw = str(payload.get("side") or payload.get("taker_side") or payload.get("maker_side") or "").upper()
    side = OrderSide.BUY if side_raw != "SELL" else OrderSide.SELL
    if price <= 0.0:
        return None
    return TradeTick(
        source=SignalSource.POLYMARKET_TRADES,
        symbol=symbol,
        price=price,
        size=max(size, 0.0),
        side=side,
        exchange_ts_ms=_event_ts_ms(payload),
        recv_mono_ns=recv_ns,
        trade_id=str(payload.get("id") or payload.get("trade_id") or payload.get("hash") or ""),
    )


def _parse_price_change(
    payload: dict[str, Any],
    recv_ns: int,
    subscription_state: MarketSubscriptionState,
) -> list[TopOfBook | TradeTick]:
    events: list[TopOfBook | TradeTick] = []
    changes = payload.get("changes") or payload.get("price_changes") or []
    if not isinstance(changes, list):
        changes = [payload]
    by_asset: dict[str, dict[str, Any]] = {}
    for change in changes:
        if not isinstance(change, dict):
            continue
        asset_id = str(change.get("asset_id") or change.get("assetId") or change.get("token_id") or payload.get("asset_id") or "")
        by_asset[asset_id] = change
        side_raw = str(change.get("side") or change.get("taker_side") or "").upper()
        if "price" in change:
            trade = _parse_trade({**payload, **change, "side": side_raw or payload.get("side", "")}, recv_ns, subscription_state)
            if trade is not None:
                events.append(trade)
    for asset_id, change in by_asset.items():
        bid = change.get("best_bid") or payload.get("best_bid")
        ask = change.get("best_ask") or payload.get("best_ask")
        if bid is None or ask is None:
            continue
        synthetic = {
            "asset_id": asset_id,
            "bid": bid,
            "ask": ask,
            "bid_size": change.get("best_bid_size") or change.get("bid_size") or payload.get("best_bid_size") or 0.0,
            "ask_size": change.get("best_ask_size") or change.get("ask_size") or payload.get("best_ask_size") or 0.0,
            "timestamp": payload.get("timestamp") or change.get("timestamp"),
        }
        tob = _parse_best_bid_ask(synthetic, recv_ns, subscription_state)
        if tob is not None:
            events.append(tob)
    return events


def _parse_best_bid_ask(payload: dict[str, Any], recv_ns: int, subscription_state: MarketSubscriptionState) -> TopOfBook | None:
    asset_id = str(payload.get("asset_id") or payload.get("assetId") or payload.get("token_id") or "")
    outcome = subscription_state.token_outcome(asset_id)
    symbol = outcome.value if outcome is not None else (asset_id or "UNKNOWN")
    try:
        bid = float(payload.get("bid") or payload.get("best_bid"))
        ask = float(payload.get("ask") or payload.get("best_ask"))
        bid_size = float(payload.get("bid_size") or payload.get("best_bid_size") or 0.0)
        ask_size = float(payload.get("ask_size") or payload.get("best_ask_size") or 0.0)
    except (TypeError, ValueError):
        return None
    if bid <= 0.0 or ask <= 0.0:
        return None
    return TopOfBook(
        source=SignalSource.POLYMARKET_BOOK,
        symbol=symbol,
        bid=bid,
        ask=ask,
        bid_size=max(0.0, bid_size),
        ask_size=max(0.0, ask_size),
        exchange_ts_ms=_event_ts_ms(payload),
        recv_mono_ns=recv_ns,
    )


def _best_level(levels: Any, is_bid: bool) -> tuple[float, float]:
    if not isinstance(levels, list) or not levels:
        raise ValueError("empty book side")
    best_price = -1.0 if is_bid else 2.0
    best_size = 0.0
    for item in levels:
        if isinstance(item, dict):
            price = float(item.get("price"))
            size = float(item.get("size"))
        else:
            price = float(item[0])
            size = float(item[1])
        better = price > best_price if is_bid else price < best_price
        if better:
            best_price = price
            best_size = size
    return best_price, best_size


def _event_ts_ms(payload: dict[str, Any]) -> int:
    raw = payload.get("timestamp") or payload.get("ts") or payload.get("time")
    try:
        value = int(float(raw))
        return value * 1000 if value < 10_000_000_000 else value
    except (TypeError, ValueError):
        return 0
