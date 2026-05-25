"""Polymarket client adapter: wraps BTC_POLY infrastructure into the original API."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Iterable, List, Optional, Tuple

from polymarket_bot.config import Direction, MarketConfig, PolymarketConfig
from polymarket_bot.fees import default_taker_fee_rate_for_market, normalize_taker_fee_rate
from polymarket_bot.infrastructure.config import InfrastructureConfig
from polymarket_bot.infrastructure.discovery import MarketDiscovery
from polymarket_bot.infrastructure.engine import InfrastructureEngine
from polymarket_bot.infrastructure.feeds import RTDSChainlinkFeed
from polymarket_bot.infrastructure.network import ConnectionPool
from polymarket_bot.infrastructure.orderbook import PolymarketOrderBook
from polymarket_bot.infrastructure.ptb import PTBLifecycle
from polymarket_bot.infrastructure.settlement import SettlementEngine
from polymarket_bot.infrastructure.types import (
    MarketInfo as InfraMarketInfo,
    PTBRecord,
)
from polymarket_bot.infrastructure.wallet import WalletMaintenance
from polymarket_bot.bot_types import (
    OrderbookLevel,
    PolymarketMarketInfo,
    PolymarketOrderbook,
    PriceToBeatSnapshot,
)

logger = logging.getLogger(__name__)

ERC20_ABI = [
    {"constant": True, "inputs": [{"name": "owner", "type": "address"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
    {"constant": True, "inputs": [{"name": "owner", "type": "address"}, {"name": "spender", "type": "address"}], "name": "allowance", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
    {"constant": False, "inputs": [{"name": "spender", "type": "address"}, {"name": "amount", "type": "uint256"}], "name": "approve", "outputs": [{"name": "", "type": "bool"}], "type": "function"},
]

ERC1155_ABI = [
    {"constant": True, "inputs": [{"name": "account", "type": "address"}, {"name": "id", "type": "uint256"}], "name": "balanceOf", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
    {"constant": True, "inputs": [{"name": "account", "type": "address"}, {"name": "operator", "type": "address"}], "name": "isApprovedForAll", "outputs": [{"name": "", "type": "bool"}], "type": "function"},
    {"constant": False, "inputs": [{"name": "operator", "type": "address"}, {"name": "approved", "type": "bool"}], "name": "setApprovalForAll", "outputs": [], "type": "function"},
    {"constant": True, "inputs": [{"name": "conditionId", "type": "bytes32"}], "name": "payoutDenominator", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
    {"constant": True, "inputs": [{"name": "conditionId", "type": "bytes32"}, {"name": "index", "type": "uint256"}], "name": "payoutNumerators", "outputs": [{"name": "", "type": "uint256"}], "type": "function"},
]

CTF_COLLATERAL_ADAPTER_ABI = [
    {"inputs": [{"name": "collateralToken", "type": "address"}, {"name": "parentCollectionId", "type": "bytes32"}, {"name": "conditionId", "type": "bytes32"}, {"name": "indexSets", "type": "uint256[]"}], "name": "redeemPositions", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
]


class PolymarketCLOBClient:
    """Backward-compatible Polymarket client backed by BTC_POLY infrastructure."""

    POLYMARKET_WEB_URL = "https://polymarket.com"

    def __init__(self, config: PolymarketConfig, market_config: Optional[MarketConfig] = None):
        self.config = config
        self.market_config = market_config or MarketConfig()
        self._market_info: Optional[PolymarketMarketInfo] = None
        self._last_market_lookup = 0.0
        self._last_price_to_beat: Optional[PriceToBeatSnapshot] = None
        self._max_ptb_age_seconds = self.market_config.settlement_interval_seconds + 30.0
        self._max_ptb_clock_skew_seconds = 10.0
        self._max_ptb_transport_lag_seconds = 10.0
        self._orderbook_cache: Dict[str, PolymarketOrderbook] = {}
        self._orderbook_cache_max_age_seconds = 5.0
        self._rate_limit_remaining = 100
        self._rate_limit_reset = 0.0
        self._request_count = 0
        self._error_count = 0
        self._cached_valid_ptb: Optional[PriceToBeatSnapshot] = None
        self._cached_valid_ptb_slug = ""
        self._retry_config = {"max_retries": 3, "base_delay": 0.5, "max_delay": 10.0, "backoff_factor": 2.0}

        self._infra_config = InfrastructureConfig.from_env()
        self._pool = ConnectionPool(self._infra_config)
        self._discovery = MarketDiscovery(self._infra_config, self._pool)
        self._chainlink = RTDSChainlinkFeed(self._infra_config)
        self._ptb = PTBLifecycle(self._discovery, self._chainlink)
        self._orderbook = PolymarketOrderBook(self._infra_config, self._pool)
        self._settlement = SettlementEngine(self._infra_config, self._pool)
        self._wallet = WalletMaintenance(self._infra_config)
        self._engine: Optional[InfrastructureEngine] = None
        self._owns_infra = True
        self._clob_sdk_client: Any = None

    def attach_infrastructure_engine(self, engine: InfrastructureEngine) -> None:
        """Reuse a canonical infrastructure engine so discovery/PTB/orderbook stay single-sourced."""
        self._engine = engine
        self._infra_config = engine.config
        self._pool = engine.pool
        self._discovery = engine.discovery
        self._chainlink = engine.chainlink
        self._ptb = engine.ptb_lifecycle
        self._orderbook = engine.orderbook
        self._settlement = engine.settlement
        self._wallet = engine.wallet
        self._owns_infra = False

    async def start(self) -> None:
        try:
            if self._engine is not None:
                if not self._engine.running:
                    await self._engine.start()
                self._orderbook.set_resolution_callback(self._on_resolution_event)
            else:
                await self._pool.start()
                await self._chainlink.start()
                await self._orderbook.start()
                await self._wallet.initialize()
                self._orderbook.set_resolution_callback(self._on_resolution_event)

            if not self.config.dry_run and self.config.private_key:
                try:
                    from py_clob_client_v2 import ApiCreds, ClobClient
                    from py_clob_client_v2 import SignatureTypeV2
                    api_key = self.config.clob_api_key or self.config.builder_api_key
                    api_secret = self.config.clob_api_secret or self.config.builder_secret
                    api_passphrase = self.config.clob_api_passphrase or self.config.builder_passphrase
                    kwargs: Dict[str, Any] = {"host": self.config.clob_api_url, "chain_id": self.config.chain_id, "key": self.config.private_key}
                    sig_type = self._sdk_signature_type()
                    if sig_type is not None:
                        kwargs["signature_type"] = sig_type
                    funder = self.config.funder or self.config.deposit_wallet
                    if funder:
                        kwargs["funder"] = funder
                    if api_key and api_secret and api_passphrase:
                        kwargs["creds"] = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=api_passphrase)
                    client = ClobClient(**kwargs)
                    if not (api_key and api_secret and api_passphrase):
                        creds = client.create_or_derive_api_key()
                        kwargs["creds"] = creds
                        client = ClobClient(**kwargs)
                    self._clob_sdk_client = client
                except Exception as exc:
                    logger.error("CLOB SDK init failed: %s", exc)
            logger.info("PolymarketCLOBClient started (BTC_POLY infrastructure)")
        except Exception:
            await self.stop()
            raise

    async def stop(self) -> None:
        if self._owns_infra:
            await self._orderbook.stop()
            await self._chainlink.stop()
            await self._wallet.close()
            await self._pool.close()
        self._clob_sdk_client = None
        logger.info("PolymarketCLOBClient stopped")

    async def _on_resolution_event(self, payload: Dict[str, Any]) -> None:
        self._settlement.record_ws_event(payload)

    def _sdk_signature_type(self) -> Optional[Any]:
        if self.config.sig_type == 0:
            return None
        try:
            from py_clob_client_v2 import SignatureTypeV2
            mapping = {1: "POLY_PROXY", 2: "POLY_GNOSIS_SAFE", 3: "POLY_1271"}
            enum_name = mapping.get(self.config.sig_type)
            if enum_name and hasattr(SignatureTypeV2, enum_name):
                return getattr(SignatureTypeV2, enum_name)
        except ImportError:
            pass
        return self.config.sig_type

    def _infra_to_polymarket_info(self, infra: InfraMarketInfo, ptb: Optional[PTBRecord] = None) -> PolymarketMarketInfo:
        ptb_snapshot: Optional[PriceToBeatSnapshot] = None
        if ptb is not None and ptb.price > 0:
            ptb_snapshot = PriceToBeatSnapshot(
                price=ptb.price,
                source_timestamp=ptb.event_start_time,
                received_timestamp=ptb.captured_at,
                source=ptb.source,
                market_slug=ptb.slug,
                is_valid=True,
            )
        elif infra.price_to_beat > 0:
            ptb_snapshot = PriceToBeatSnapshot(
                price=infra.price_to_beat,
                source_timestamp=infra.event_start_time,
                received_timestamp=time.time(),
                source="market_metadata",
                market_slug=infra.slug,
                is_valid=True,
            )
        return PolymarketMarketInfo(
            condition_id=infra.condition_id,
            question=infra.question,
            tokens={"UP": infra.up_token_id, "DOWN": infra.down_token_id},
            end_date_iso=str(infra.end_time),
            slug=infra.slug,
            active=infra.active,
            closed=infra.is_expired,
            minimum_order_size=infra.min_order_size,
            minimum_tick_size=float(infra.tick_size),
            fee_rate=infra.fee_rate,
            fee_exponent=infra.fee_exponent,
            price_to_beat=infra.price_to_beat if infra.price_to_beat > 0 else None,
            price_to_beat_snapshot=ptb_snapshot,
        )

    async def discover_btc_5min_market(self) -> Optional[PolymarketMarketInfo]:
        now = time.time()
        if self._can_use_cached_market_info(now):
            return self._market_info
        infra_market = await self._discovery.find_active_market()
        if infra_market is None:
            logger.error("No active BTC Up or Down 5m market found")
            return None
        infra_market = await self._discovery.refresh_market(infra_market)
        infra_market, ptb = await self._ptb.resolve(infra_market)
        market_info = self._infra_to_polymarket_info(infra_market, ptb)
        self._market_info = market_info
        self._last_market_lookup = time.time()
        if ptb is not None and ptb.price > 0:
            ptb_snapshot = PriceToBeatSnapshot(
                price=ptb.price,
                source_timestamp=ptb.event_start_time,
                received_timestamp=ptb.captured_at,
                source=ptb.source,
                market_slug=ptb.slug,
                is_valid=True,
            )
            self._last_price_to_beat = ptb_snapshot
        ptb_status = "none"
        if market_info.price_to_beat_snapshot:
            ptb_status = "valid" if market_info.price_to_beat_snapshot.is_valid else f"invalid:{market_info.price_to_beat_snapshot.reason}"
        logger.debug("Found Polymarket market: slug=%s price_to_beat=%s ptb_status=%s", market_info.slug, market_info.price_to_beat, ptb_status)
        return market_info

    async def discover_btc_market(self) -> Optional[PolymarketMarketInfo]:
        return await self.discover_btc_5min_market()

    async def fetch_btc_5min_market_by_slug(self, slug: str, *, activate: bool = False, fast: bool = False) -> Optional[PolymarketMarketInfo]:
        if not slug or not slug.startswith("btc-updown-5m-"):
            return None
        infra_market = await self._discovery.find_active_market()
        if infra_market is None or infra_market.slug != slug:
            return None
        infra_market, ptb = await self._ptb.resolve(infra_market)
        market_info = self._infra_to_polymarket_info(infra_market, ptb)
        if activate:
            self.set_active_market_info(market_info)
        return market_info

    def set_active_market_info(self, market_info: PolymarketMarketInfo) -> None:
        self._market_info = market_info
        self._last_market_lookup = time.time()
        if market_info.price_to_beat_snapshot is not None:
            self._last_price_to_beat = market_info.price_to_beat_snapshot

    async def refresh_price_to_beat(self, market_slug: Optional[str] = None) -> Optional[PriceToBeatSnapshot]:
        slug = market_slug or (self._market_info.slug if self._market_info else "")
        if not slug:
            return None
        if self._cached_valid_ptb is not None and self._cached_valid_ptb_slug == slug and self._cached_valid_ptb.is_valid:
            return self._cached_valid_ptb
        infra_market = InfraMarketInfo(slug=slug, condition_id="", question="", up_token_id="", down_token_id="", price_to_beat=0.0, end_time=0.0, event_start_time=self._start_time_from_slug(slug))
        _, ptb = await self._ptb.resolve(infra_market)
        if ptb is None or ptb.price <= 0:
            return None
        snapshot = PriceToBeatSnapshot(
            price=ptb.price,
            source_timestamp=ptb.event_start_time,
            received_timestamp=ptb.captured_at,
            source=ptb.source,
            market_slug=ptb.slug,
            is_valid=True,
        )
        self._cached_valid_ptb = snapshot
        self._cached_valid_ptb_slug = slug
        self._last_price_to_beat = snapshot
        return snapshot

    def clear_ptb_cache_for_round(self, slug: str) -> None:
        if self._cached_valid_ptb_slug == slug:
            self._cached_valid_ptb = None
            self._cached_valid_ptb_slug = ""

    def invalidate_ptb_cache(self) -> None:
        self._cached_valid_ptb = None
        self._cached_valid_ptb_slug = ""
        self._last_price_to_beat = None

    def get_valid_price_to_beat(self, market_slug: Optional[str] = None) -> Optional[PriceToBeatSnapshot]:
        snapshot = self._last_price_to_beat
        if snapshot is None:
            return None
        if market_slug and snapshot.market_slug and snapshot.market_slug != market_slug:
            return None
        now = time.time()
        if now - snapshot.received_timestamp > self._max_ptb_age_seconds:
            return None
        return snapshot

    async def subscribe_orderbook_stream(self, market_info: PolymarketMarketInfo) -> None:
        await self.subscribe_orderbook_streams([market_info])

    async def subscribe_orderbook_streams(self, market_infos: List[PolymarketMarketInfo]) -> None:
        seen_assets = set()
        asset_ids: List[str] = []
        for market_info in market_infos:
            if not market_info:
                continue
            for token_id in (getattr(market_info, 'token_id_up', None) or market_info.tokens.get("UP", ""),
                             getattr(market_info, 'token_id_down', None) or market_info.tokens.get("DOWN", "")):
                if token_id and token_id not in seen_assets:
                    seen_assets.add(token_id)
                    asset_ids.append(token_id)
        if not asset_ids:
            logger.warning("Skipping orderbook WS subscription: no token ids")
            return
        await self._orderbook.subscribe(asset_ids)

    async def stop_orderbook_stream(self) -> None:
        await self._orderbook.subscribe([])

    def get_cached_orderbook(self, token_id: str) -> Optional[PolymarketOrderbook]:
        infra_book = self._orderbook.get_book(token_id)
        if infra_book is None:
            return self._orderbook_cache.get(token_id)
        ts = time.time()
        if ts - infra_book.timestamp > self._orderbook_cache_max_age_seconds:
            return self._orderbook_cache.get(token_id)
        ob = PolymarketOrderbook(
            timestamp=infra_book.timestamp,
            token_id=infra_book.asset_id,
            bids=[OrderbookLevel(price=level.price, size=level.size, timestamp=infra_book.timestamp) for level in infra_book.bids],
            asks=[OrderbookLevel(price=level.price, size=level.size, timestamp=infra_book.timestamp) for level in infra_book.asks],
            condition_id=self._market_info.condition_id if self._market_info else "",
        )
        self._orderbook_cache[token_id] = ob
        return ob

    async def get_orderbook(self, token_id: str) -> Optional[PolymarketOrderbook]:
        cached = self.get_cached_orderbook(token_id)
        if cached is not None:
            return cached
        data = await self._request("GET", "/book", params={"token_id": token_id})
        if not data:
            return None
        def parse_side(levels: Iterable[Dict[str, Any]]) -> List[OrderbookLevel]:
            return [OrderbookLevel(price=self._float(level.get("price")), size=self._float(level.get("size")), timestamp=time.time()) for level in levels or []]
        bids = parse_side(data.get("bids", []))
        asks = parse_side(data.get("asks", []))
        bids.sort(key=lambda x: x.price, reverse=True)
        asks.sort(key=lambda x: x.price)
        condition_id = self._market_info.condition_id if self._market_info else ""
        return PolymarketOrderbook(timestamp=time.time(), token_id=token_id, bids=bids, asks=asks, condition_id=condition_id)

    async def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        if self._clob_sdk_client is None:
            logger.warning("CLOB SDK client not available for get_order")
            return None
        try:
            from py_clob_client_v2.clob_types import GetOrderParams
            result = await asyncio.to_thread(lambda: self._clob_sdk_client.get_order(order_id))
            return result if isinstance(result, dict) else None
        except Exception as exc:
            logger.error("get_order failed: %s", exc)
            return None

    async def place_order(self, token_id: str, side: str, direction: Direction, size: float, price: float) -> Optional[Dict[str, Any]]:
        if self._clob_sdk_client is None:
            if self.config.dry_run:
                return {"status": "dry_run", "order_id": "dry_run", "success": True}
            logger.warning("CLOB SDK client not available for place_order")
            return {"success": False, "reason": "sdk-unavailable"}
        try:
            from py_clob_client_v2.clob_types import OrderArgs, Side as ClobSide
            clob_side = ClobSide.BUY if side.upper() == "BUY" else ClobSide.SELL
            args = OrderArgs(
                token_id=str(token_id),
                side=clob_side,
                price=str(price),
                size=str(size),
            )
            signed = self._clob_sdk_client.create_order(args)
            result = await asyncio.to_thread(lambda: self._clob_sdk_client.place_order(signed))
            return result if isinstance(result, dict) else {"success": bool(result)}
        except Exception as exc:
            logger.error("place_order failed: %s", exc)
            return {"success": False, "reason": str(exc)}

    async def get_btc_5m_settlement_price(self, market_slug: str) -> Optional[Dict[str, Any]]:
        start_ts = self._start_time_from_slug(market_slug)
        if start_ts <= 0:
            return None
        end_ts = start_ts + 300.0

        chainlink_data = await self._chainlink.fetch_price_at_or_after(end_ts)
        if chainlink_data is not None and chainlink_data.price > 0:
            return {
                "price": chainlink_data.price,
                "source": "rtds_crypto_prices_chainlink",
                "source_timestamp": chainlink_data.updated_at,
                "slug": market_slug,
                "is_valid": True,
            }

        infra_market = InfraMarketInfo(slug=market_slug, condition_id="", question="", up_token_id="", down_token_id="", price_to_beat=0.0, end_time=0.0, event_start_time=end_ts)
        infra_market, ptb = await self._ptb.resolve(infra_market)
        if ptb is not None and ptb.price > 0:
            return {"price": ptb.price, "source": ptb.source, "source_timestamp": ptb.source_timestamp, "slug": ptb.slug, "is_valid": True}

        return None

    @staticmethod
    def _start_time_from_slug(slug: str) -> float:
        try:
            parts = slug.split("-")
            return float(parts[-1]) if len(parts) > 2 else 0.0
        except (ValueError, IndexError):
            return 0.0

    async def get_wallet_maintenance_status(self, token_ids: List[str], condition_ids: List[str]) -> Dict[str, Any]:
        readiness = await self._wallet.execution_readiness(token_ids)
        return {
            "ready": readiness.ready,
            "live": readiness.live,
            "approvals_ready": readiness.approvals_ready,
            "blockers": list(readiness.blockers),
            "collateral_balance": readiness.collateral.balance if readiness.collateral else None,
            "collateral_allowance": readiness.collateral.allowance if readiness.collateral else None,
        }

    async def update_balance_allowance(self, asset_type: str) -> None:
        if asset_type == "COLLATERAL":
            await self._wallet.refresh_collateral_balance_allowance(update_cache=True)

    async def approve_ctf_adapter(self) -> Any:
        return await self._wallet.ensure_deposit_wallet_approvals()

    async def redeem_positions(self, condition_id: str) -> Dict[str, Any]:
        return await self._wallet.redeem_standard_market(condition_id)

    async def cancel_all_orders(self) -> bool:
        return await self._wallet.cancel_all_orders()

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        return await self._wallet.get_open_orders()

    def _can_use_cached_market_info(self, timestamp: float) -> bool:
        if self._market_info is None:
            return False
        if (timestamp - self._last_market_lookup) >= 20:
            return False
        return True

    async def _request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None, json_data: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        for attempt in range(self._retry_config["max_retries"]):
            try:
                session = await self._pool.session()
                async with session.request(method, f"{self._infra_config.polymarket_host}{path}", params=params, json=json_data) as resp:
                    self._request_count += 1
                    if resp.status == 200:
                        return await resp.json(content_type=None)
                    if resp.status in (429, 500) and attempt < self._retry_config["max_retries"] - 1:
                        delay = min(self._retry_config["base_delay"] * (self._retry_config["backoff_factor"] ** attempt), self._retry_config["max_delay"])
                        await asyncio.sleep(delay)
                        continue
                    body = await resp.text()
                    logger.warning("Polymarket request failed: status=%s path=%s body=%s", resp.status, path, body[:300])
                    return None
            except Exception as exc:
                self._error_count += 1
                if attempt < self._retry_config["max_retries"] - 1:
                    delay = min(self._retry_config["base_delay"] * (self._retry_config["backoff_factor"] ** attempt), self._retry_config["max_delay"])
                    await asyncio.sleep(delay)
                    continue
                logger.warning("Polymarket request failed after retries: %s path=%s", exc, path)
                return None
        return None

    @staticmethod
    def _float(value: Any, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _round_to_tick(price: float, tick_size: float) -> float:
        tick = Decimal(str(tick_size or 0.01))
        rounded = (Decimal(str(price)) / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return float(rounded * tick)
