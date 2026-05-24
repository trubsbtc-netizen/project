from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

from polymarket_bot.infrastructure.config import InfrastructureConfig
from polymarket_bot.infrastructure.discovery import MarketDiscovery
from polymarket_bot.infrastructure.feeds import BinanceFeed, CoinbaseFeed, RTDSChainlinkFeed
from polymarket_bot.infrastructure.network import ConnectionPool
from polymarket_bot.infrastructure.orderbook import PolymarketOrderBook
from polymarket_bot.infrastructure.ptb import PTBLifecycle
from polymarket_bot.infrastructure.settlement import SettlementEngine
from polymarket_bot.infrastructure.types import (
    ComponentHealth,
    MarketTruthState,
    RoundPhase,
    ServiceState,
)
from polymarket_bot.infrastructure.wallet import WalletMaintenance

logger = logging.getLogger(__name__)


class InfrastructureEngine:
    def __init__(self, config: InfrastructureConfig):
        self.config = config
        self.pool = ConnectionPool(config)
        self.chainlink = RTDSChainlinkFeed(config)
        self.binance = BinanceFeed(config, self.pool)
        self.coinbase = CoinbaseFeed(config)
        self.orderbook = PolymarketOrderBook(config, self.pool)
        self.discovery = MarketDiscovery(config, self.pool)
        self.ptb_lifecycle = PTBLifecycle(self.discovery, self.chainlink)
        self.settlement = SettlementEngine(config, self.pool)
        self.wallet = WalletMaintenance(config)
        self.state = MarketTruthState(phase=RoundPhase.DISCOVERY, updated_at=time.time())
        self._running = False
        self._tasks: List[asyncio.Task] = []

    @property
    def running(self) -> bool:
        return self._running

    async def start(self) -> None:
        if self._running:
            return
        await self.pool.start()
        await self.wallet.initialize()
        self.orderbook.set_resolution_callback(self._on_resolution_event)
        await self.chainlink.start()
        await self.binance.start()
        await self.coinbase.start()
        await self.orderbook.start()
        self._running = True
        self._tasks = [
            asyncio.create_task(self._market_loop(), name="market-sync"),
            asyncio.create_task(self._wallet_loop(), name="wallet-maintenance"),
            asyncio.create_task(self._health_loop(), name="health-monitor"),
        ]
        logger.info("BTC_POLY infrastructure engine started")

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()
        await self.wallet.cancel_all_orders()
        await self.orderbook.stop()
        await self.coinbase.stop()
        await self.binance.stop()
        await self.chainlink.stop()
        await self.wallet.close()
        await self.pool.close()
        logger.info("BTC_POLY infrastructure engine stopped")

    async def _on_resolution_event(self, payload: Dict[str, Any]) -> None:
        self.settlement.record_ws_event(payload)

    async def _market_loop(self) -> None:
        while self._running:
            try:
                await self._sync_market_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("market sync error: %s", exc)
            await asyncio.sleep(max(1.0, self.config.market_poll_interval_s))

    async def _sync_market_once(self) -> None:
        market = await self.discovery.find_active_market()
        if market is None:
            self.state = replace(self.state, phase=RoundPhase.DISCOVERY, updated_at=time.time())
            return
        market = await self.discovery.refresh_market(market)
        market, ptb = await self.ptb_lifecycle.resolve(market)

        warm_markets = await self.discovery.find_warm_markets()
        asset_ids: List[str] = [market.up_token_id, market.down_token_id]
        for warm in warm_markets:
            asset_ids.extend([warm.up_token_id, warm.down_token_id])
        await self.orderbook.subscribe(tuple(dict.fromkeys(asset for asset in asset_ids if asset)))

        settlement = self.settlement.cached_resolution(market)
        phase = RoundPhase.ACTIVE
        if market.is_expired:
            phase = RoundPhase.AWAITING_SETTLEMENT
            settlement = await self.settlement.fetch_resolution(market)
            if settlement and settlement.is_known:
                phase = RoundPhase.SETTLED

        self.state = MarketTruthState(
            phase=phase,
            current_market=market,
            ptb=ptb,
            settlement=settlement,
            binance=self.binance.last,
            coinbase=self.coinbase.last,
            chainlink=self.chainlink.last,
            books=self.orderbook.books(),
            wallet=self.state.wallet,
            updated_at=time.time(),
        )

    async def _wallet_loop(self) -> None:
        while self._running:
            try:
                token_ids: Tuple[str, ...] = ()
                if self.state.current_market:
                    token_ids = (self.state.current_market.up_token_id, self.state.current_market.down_token_id)
                readiness = await self.wallet.execution_readiness(token_ids)
                self.state = replace(self.state, wallet=readiness, updated_at=time.time())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("wallet maintenance error: %s", exc)
            await asyncio.sleep(15.0)

    async def _health_loop(self) -> None:
        while self._running:
            try:
                await asyncio.sleep(30.0)
                health = self.health()
                degraded = [item.name for item in health if item.stale or item.state == ServiceState.DEGRADED]
                if degraded:
                    logger.info("degraded infrastructure components: %s", ", ".join(degraded))
                else:
                    logger.info("infrastructure health ok")
            except asyncio.CancelledError:
                raise

    def health(self) -> Tuple[ComponentHealth, ...]:
        return (
            self.chainlink.health(),
            self.binance.health(),
            self.coinbase.health(),
            self.orderbook.health(),
        )

    def snapshot(self) -> Dict[str, Any]:
        market = self.state.current_market
        ptb = self.state.ptb
        settlement = self.state.settlement
        wallet = self.state.wallet
        return {
            "phase": self.state.phase.value,
            "updated_at": self.state.updated_at,
            "market": None
            if market is None
            else {
                "slug": market.slug,
                "condition_id": market.condition_id,
                "up_token_id": market.up_token_id,
                "down_token_id": market.down_token_id,
                "price_to_beat": market.price_to_beat,
                "event_start_time": market.event_start_time,
                "end_time": market.end_time,
                "time_remaining_s": market.time_remaining_s,
            },
            "ptb": None if ptb is None else {"price": ptb.price, "source": ptb.source, "source_timestamp": ptb.source_timestamp},
            "settlement": None
            if settlement is None
            else {
                "resolved": settlement.resolved,
                "outcome": settlement.outcome.value,
                "winner_token_id": settlement.winner_token_id,
                "source": settlement.source,
                "onchain_verified": settlement.onchain_verified,
                "payout_numerators": list(settlement.payout_numerators),
                "payout_denominator": settlement.payout_denominator,
            },
            "prices": {
                "binance": None if self.state.binance is None else self.state.binance.price,
                "coinbase": None if self.state.coinbase is None else self.state.coinbase.price,
                "chainlink": None if self.state.chainlink is None else self.state.chainlink.price,
            },
            "books": {
                asset_id: {
                    "best_bid": book.best_bid,
                    "best_ask": book.best_ask,
                    "spread": book.spread,
                    "bid_size": book.total_bid_size,
                    "ask_size": book.total_ask_size,
                    "age_s": book.age_s,
                }
                for asset_id, book in self.state.books.items()
            },
            "wallet": None
            if wallet is None
            else {
                "live": wallet.live,
                "ready": wallet.ready,
                "approvals_ready": wallet.approvals_ready,
                "blockers": list(wallet.blockers),
                "collateral_balance": None if wallet.collateral is None else wallet.collateral.balance,
                "collateral_allowance": None if wallet.collateral is None else wallet.collateral.allowance,
            },
            "health": [
                {
                    "name": item.name,
                    "state": item.state.value,
                    "connected": item.connected,
                    "stale": item.stale,
                    "age_s": item.age_s,
                    "reconnects": item.reconnects,
                    "last_error": item.last_error,
                }
                for item in self.health()
            ],
        }
