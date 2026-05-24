from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from polymarket_bot.infrastructure.config import InfrastructureConfig
from polymarket_bot.infrastructure.network import ConnectionPool
from polymarket_bot.infrastructure.types import (
    MarketInfo,
    OrderBookLevel,
    OrderBookSnapshot,
    _safe_float,
)
from polymarket_bot.infrastructure.ws import ManagedWebSocketClient

logger = logging.getLogger(__name__)


class PolymarketOrderBook(ManagedWebSocketClient):
    def __init__(self, config: InfrastructureConfig, pool: ConnectionPool):
        super().__init__("polymarket_orderbook", config.polymarket_ws, config, config.orderbook_stale_s)
        self.pool = pool
        self._books: Dict[str, OrderBookSnapshot] = {}
        self._active_assets: Tuple[str, ...] = ()
        self._resolved_events: Dict[str, Dict[str, Any]] = {}
        self._on_resolved: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None

    def set_resolution_callback(self, callback: Callable[[Dict[str, Any]], Awaitable[None]]) -> None:
        self._on_resolved = callback

    def get_book(self, asset_id: str) -> Optional[OrderBookSnapshot]:
        return self._books.get(str(asset_id))

    def books(self) -> Dict[str, OrderBookSnapshot]:
        return dict(self._books)

    def get_resolved_event(self, market: MarketInfo) -> Optional[Dict[str, Any]]:
        keys = (
            market.condition_id.lower(),
            market.slug.lower(),
            market.up_token_id,
            market.down_token_id,
        )
        for key in keys:
            if key and key in self._resolved_events:
                return dict(self._resolved_events[key])
        return None

    async def subscribe(self, asset_ids: Sequence[str]) -> None:
        assets = tuple(str(asset_id) for asset_id in asset_ids if asset_id)
        if assets == self._active_assets:
            return
        previous = set(self._active_assets)
        current = set(assets)
        removed = [asset for asset in previous if asset not in current]
        self._active_assets = assets
        self._books = {asset: book for asset, book in self._books.items() if asset in current}
        if self.connected:
            if removed:
                await self.send_json({"operation": "unsubscribe", "assets_ids": removed})
            await self._send_subscription(initial=False)
        await self._seed_books(assets)

    async def _on_connect(self) -> None:
        if self._active_assets:
            await self._send_subscription(initial=True)

    async def _send_subscription(self, *, initial: bool) -> None:
        if not self._active_assets:
            return
        payload: Dict[str, Any] = {
            "assets_ids": list(self._active_assets),
            "custom_feature_enabled": True,
        }
        if initial:
            payload["type"] = "market"
        else:
            payload["operation"] = "subscribe"
        await self.send_json(payload)

    async def _on_raw_message(self, raw: str) -> None:
        text = raw.strip()
        if not text or text in {"PING", "PONG"}:
            return
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return
        await self._handle_payload(payload)

    async def _handle_payload(self, payload: Any) -> None:
        if isinstance(payload, list):
            for item in payload:
                await self._handle_payload(item)
            return
        if not isinstance(payload, dict):
            return
        event_type = str(payload.get("event_type") or payload.get("type") or "")
        if event_type == "book":
            snap = self._parse_book_message(payload)
            if snap:
                self._books[snap.asset_id] = snap
        elif event_type == "price_change":
            self._apply_price_change(payload)
        elif event_type == "market_resolved":
            self._record_resolved_event(payload)
            if self._on_resolved:
                await self._on_resolved(dict(payload))

    def _parse_book_message(self, payload: Dict[str, Any]) -> Optional[OrderBookSnapshot]:
        asset_id = str(payload.get("asset_id") or payload.get("assetId") or "")
        if not asset_id:
            return None
        bids = self._parse_levels(payload.get("bids") or [])
        asks = self._parse_levels(payload.get("asks") or [])
        if not bids and not asks:
            return None
        return OrderBookSnapshot(
            bids=tuple(sorted(bids, key=lambda level: level.price, reverse=True)),
            asks=tuple(sorted(asks, key=lambda level: level.price)),
            timestamp=time.time(),
            asset_id=asset_id,
            book_hash=str(payload.get("hash") or ""),
        )

    def _apply_price_change(self, payload: Dict[str, Any]) -> None:
        changes = payload.get("price_changes")
        rows = changes if isinstance(changes, list) else [payload]
        for row in rows:
            if not isinstance(row, dict):
                continue
            asset_id = str(row.get("asset_id") or row.get("assetId") or payload.get("asset_id") or "")
            if not asset_id:
                continue
            current = self._books.get(asset_id) or OrderBookSnapshot((), (), time.time(), asset_id=asset_id)
            bids = list(current.bids)
            asks = list(current.asks)
            price = _safe_float(row.get("price"), 0.0)
            size = _safe_float(row.get("size"), 0.0)
            side = str(row.get("side") or "").upper()
            if price <= 0:
                continue
            if side == "BUY":
                bids = self._update_level(bids, price, size, reverse=True)
            elif side == "SELL":
                asks = self._update_level(asks, price, size, reverse=False)
            else:
                continue
            self._books[asset_id] = OrderBookSnapshot(
                bids=tuple(bids),
                asks=tuple(asks),
                timestamp=time.time(),
                asset_id=asset_id,
                book_hash=str(row.get("hash") or current.book_hash),
            )

    def _record_resolved_event(self, payload: Dict[str, Any]) -> None:
        for key in self._resolution_keys(payload):
            self._resolved_events[key] = dict(payload)

    def _resolution_keys(self, payload: Dict[str, Any]) -> Tuple[str, ...]:
        keys: List[str] = []
        for name in (
            "condition_id",
            "conditionId",
            "market",
            "market_id",
            "marketId",
            "slug",
            "market_slug",
            "ticker",
            "asset_id",
            "assetId",
            "token_id",
            "tokenId",
            "winning_token_id",
            "winningTokenId",
            "winningOutcomeTokenId",
        ):
            value = payload.get(name)
            if value not in (None, ""):
                text = str(value).strip()
                keys.append(text.lower() if "token" not in name.lower() and "asset" not in name.lower() else text)
        return tuple(dict.fromkeys(key for key in keys if key))

    def _parse_levels(self, raw_levels: Any) -> List[OrderBookLevel]:
        levels: List[OrderBookLevel] = []
        if not isinstance(raw_levels, list):
            return levels
        for raw in raw_levels:
            if isinstance(raw, dict):
                price = _safe_float(raw.get("price"), 0.0)
                size = _safe_float(raw.get("size"), 0.0)
            elif isinstance(raw, (list, tuple)) and len(raw) >= 2:
                price = _safe_float(raw[0], 0.0)
                size = _safe_float(raw[1], 0.0)
            else:
                continue
            if price > 0 and size > 0:
                levels.append(OrderBookLevel(price=price, size=size))
        return levels

    def _update_level(
        self,
        levels: List[OrderBookLevel],
        price: float,
        size: float,
        *,
        reverse: bool,
    ) -> List[OrderBookLevel]:
        updated: List[OrderBookLevel] = []
        found = False
        for level in levels:
            if abs(level.price - price) < 1e-12:
                found = True
                if size > 0:
                    updated.append(OrderBookLevel(price, size))
            else:
                updated.append(level)
        if not found and size > 0:
            updated.append(OrderBookLevel(price, size))
        return sorted(updated, key=lambda level: level.price, reverse=reverse)

    async def _seed_books(self, asset_ids: Sequence[str]) -> None:
        await asyncio.gather(*(self._fetch_book_snapshot(asset_id) for asset_id in asset_ids), return_exceptions=True)

    async def _fetch_book_snapshot(self, asset_id: str) -> None:
        try:
            session = await self.pool.session()
            async with session.get(f"{self.config.polymarket_host}/book", params={"token_id": asset_id}) as resp:
                if resp.status != 200:
                    return
                payload = await resp.json(content_type=None)
            snap = self._parse_snapshot_payload(asset_id, payload)
            if snap:
                self._books[asset_id] = snap
        except Exception as exc:
            self.last_error = repr(exc)

    def _parse_snapshot_payload(self, asset_id: str, payload: Any) -> Optional[OrderBookSnapshot]:
        if isinstance(payload, dict):
            if isinstance(payload.get("book"), dict):
                payload = payload["book"]
            elif isinstance(payload.get("data"), dict):
                payload = payload["data"]
        if not isinstance(payload, dict):
            return None
        bids = self._parse_levels(payload.get("bids") or [])
        asks = self._parse_levels(payload.get("asks") or [])
        if not bids and not asks:
            return None
        return OrderBookSnapshot(
            bids=tuple(sorted(bids, key=lambda level: level.price, reverse=True)),
            asks=tuple(sorted(asks, key=lambda level: level.price)),
            timestamp=time.time(),
            asset_id=asset_id,
            book_hash=str(payload.get("hash") or ""),
        )
