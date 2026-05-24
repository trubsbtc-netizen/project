from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from core.config import WebsocketConfig
from core.runtime.health import FeedHeartbeat
from core.types import PriceTick, TruthSource
from core.websocket.client import ResilientWebSocketClient, WebSocketSpec


TruthHandler = Callable[[PriceTick], Awaitable[None]]


class ChainlinkRTDSClient:
    def __init__(self, config: WebsocketConfig, heartbeat: FeedHeartbeat, handler: TruthHandler) -> None:
        self._config = config
        self._heartbeat = heartbeat
        self._handler = handler

    def _subscriptions(self) -> list[dict[str, Any]]:
        return [
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
        ]

    async def _handle(self, payload: Any, recv_ns: int) -> None:
        tick = parse_chainlink_price(payload, recv_ns)
        if tick is not None:
            await self._handler(tick)

    async def run(self) -> None:
        spec = WebSocketSpec(
            name="polymarket.rtds.chainlink",
            url=self._config.polymarket_rtds_ws,
            heartbeat_s=self._config.heartbeat_s,
            connect_timeout_s=self._config.connect_timeout_s,
            max_backoff_s=self._config.max_backoff_s,
            ping_payload="PING",
        )
        await ResilientWebSocketClient(spec, self._heartbeat, self._subscriptions, self._handle).run()


def parse_chainlink_price(payload: Any, recv_ns: int) -> PriceTick | None:
    if isinstance(payload, list):
        for item in payload:
            tick = parse_chainlink_price(item, recv_ns)
            if tick is not None:
                return tick
        return None
    if not isinstance(payload, dict):
        return None
    lower = {str(k).lower(): v for k, v in payload.items()}
    asset = str(lower.get("asset") or lower.get("symbol") or lower.get("ticker") or "").lower()
    channel = str(lower.get("channel") or lower.get("topic") or lower.get("type") or "").lower()
    data = lower.get("payload") or lower.get("data")
    if isinstance(data, dict):
        nested = {str(k).lower(): v for k, v in data.items()}
        lower = {**lower, **nested}
        asset = str(lower.get("asset") or lower.get("symbol") or lower.get("ticker") or "").lower()
        channel = str(lower.get("channel") or lower.get("topic") or lower.get("type") or channel).lower()
        data = lower.get("data")
    if isinstance(data, list):
        for item in reversed(data):
            if isinstance(item, dict):
                merged = {
                    "topic": lower.get("topic") or "crypto_prices_chainlink",
                    "type": lower.get("type") or "update",
                    "payload": {
                        **item,
                        "symbol": asset or "btc/usd",
                    },
                }
                tick = parse_chainlink_price(merged, recv_ns)
                if tick is not None:
                    return tick
        return None
    if asset and "btc" not in asset and "bitcoin" not in asset:
        return None
    if channel and not any(key in channel for key in ("crypto", "price", "chainlink", "btc")):
        return None
    raw_price = (
        lower.get("price")
        or lower.get("value")
        or lower.get("answer")
        or lower.get("mid")
        or lower.get("p")
    )
    try:
        price = float(raw_price)
    except (TypeError, ValueError):
        nested = lower.get("payload") or lower.get("data")
        return parse_chainlink_price(nested, recv_ns) if nested is not None else None
    if price <= 0.0:
        return None
    ts = lower.get("timestamp") or lower.get("ts") or lower.get("updatedat") or lower.get("time")
    try:
        ts_ms = int(float(ts))
        if ts_ms < 10_000_000_000:
            ts_ms *= 1000
    except (TypeError, ValueError):
        ts_ms = 0
    return PriceTick(
        source=TruthSource.RTDS_CHAINLINK,
        symbol="BTC/USD",
        price=price,
        exchange_ts_ms=ts_ms,
        recv_mono_ns=recv_ns,
        quality=1.0,
        raw=payload,
    )
