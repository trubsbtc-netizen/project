from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from dataclasses import dataclass

from core.approvals.service import ApprovalService
from core.config import BotConfig
from core.execution.gateway import DryRunGateway, ExecutionEngine, PyClobExecutionGateway
from core.markets.canonical import CanonicalMarketState
from core.microstructure.features import MicrostructureEngine
from core.ptb.lifecycle import PTBMetadataUnavailable, PTBProvider
from core.rtds.chainlink import ChainlinkRTDSClient
from core.runtime.clock import bucket_5m
from core.runtime.health import HealthMonitor
from core.runtime.supervisor import TaskSupervisor
from core.settlement.service import SettlementService
from core.strategy.engine import ProbabilisticStrategyEngine
from core.types import FeedKind, PriceTick, PTBMarket, RuntimeEvent, SignalSource, TopOfBook, TradeTick
from core.wallet.service import WalletService
from core.websocket.exchange_feeds import BinanceFeed, CoinbaseFeed
from core.websocket.polymarket import MarketSubscriptionState, PolymarketMarketFeed, PolymarketUserFeed


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RuntimeCounters:
    truth_ticks: int = 0
    exchange_events: int = 0
    poly_events: int = 0
    strategy_evaluations: int = 0
    order_intents: int = 0
    orders_accepted: int = 0
    rollovers: int = 0
    settlement_checks: int = 0


class InstitutionalBTCPolyBot:
    def __init__(self, config: BotConfig, enable_tui: bool = False) -> None:
        self.config = config
        self.enable_tui = enable_tui
        self.health = HealthMonitor()
        self.market_state = CanonicalMarketState()
        self.ptb = PTBProvider(config.polymarket)
        self.market_subscription = MarketSubscriptionState()
        self.microstructure = MicrostructureEngine(config.strategy, self.health)
        self.strategy = ProbabilisticStrategyEngine(config.strategy, config.execution, self.microstructure)
        self.wallet = WalletService(config.polymarket, config.contracts)
        self.approvals = ApprovalService(config.polymarket, config.contracts)
        self.settlement = SettlementService(config.contracts, config.settlement)
        gateway = DryRunGateway() if config.execution.dry_run else PyClobExecutionGateway(config.polymarket, config.execution)
        self.execution = ExecutionEngine(gateway)
        self.events: asyncio.Queue[RuntimeEvent] = asyncio.Queue(maxsize=config.websockets.inbound_queue_size)
        self.counters = RuntimeCounters()
        self._last_estimate_by_bucket: dict[int, float] = {}
        self._last_risk_reason_by_bucket: dict[int, str] = {}
        self._settlement_seen: set[str] = set()
        self._poly_market_feed: PolymarketMarketFeed | None = None

    async def start(self) -> None:
        supervisor = TaskSupervisor()
        await supervisor.run(self._start_tasks)

    async def _start_tasks(self, supervisor: TaskSupervisor) -> None:
        logger.info(
            "starting bot dry_run=%s live_trading=%s ptb_mode=%s",
            self.config.execution.dry_run,
            self.config.execution.enable_live_trading,
            self.config.polymarket.ptb_mode,
        )
        if self.config.execution.dry_run or not self.config.execution.enable_live_trading:
            logger.info("execution is simulation-only; set DRY_RUN=false and ENABLE_LIVE_TRADING=true for live orders")
        await self._cold_start_readiness()
        market = await self.ptb.bootstrap_startup_round(time.time())
        await self._install_ptb_market(market)

        ws_cfg = self.config.websockets
        stale = ws_cfg.stale_feed_s
        hb_chainlink = self.health.register("rtds.chainlink", FeedKind.CHAINLINK, stale)
        hb_binance = self.health.register("binance", FeedKind.BINANCE, stale)
        hb_coinbase = self.health.register("coinbase", FeedKind.COINBASE, stale)
        hb_poly_market = self.health.register("polymarket.market", FeedKind.POLYMARKET, stale)
        hb_poly_user = self.health.register("polymarket.user", FeedKind.USER, max(8.0, stale * 4.0))

        supervisor.create(
            "ingest.rtds.chainlink",
            ChainlinkRTDSClient(ws_cfg, hb_chainlink, self._on_truth_tick).run(),
            critical=True,
        )
        supervisor.create(
            "ingest.binance",
            BinanceFeed(ws_cfg, hb_binance, self._on_exchange_event).run(),
            critical=True,
        )
        supervisor.create(
            "ingest.coinbase",
            CoinbaseFeed(ws_cfg, hb_coinbase, self._on_exchange_event).run(),
            critical=True,
        )
        self._poly_market_feed = PolymarketMarketFeed(ws_cfg, self.market_subscription, hb_poly_market, self._on_poly_event)
        supervisor.create(
            "ingest.polymarket.market",
            self._poly_market_feed.run(),
            critical=True,
        )
        supervisor.create(
            "ingest.polymarket.user",
            PolymarketUserFeed(ws_cfg, self.config.polymarket, hb_poly_user, self._on_user_event).run(),
            critical=False,
        )
        supervisor.create("runtime.event_processor", self._event_processor(), critical=True)
        supervisor.create("runtime.rollover", self._rollover_loop(), critical=True)
        supervisor.create("runtime.settlement", self._settlement_loop(), critical=False)
        supervisor.create("runtime.health", self._health_loop(), critical=False)
        if self.enable_tui:
            try:
                from core.ui.tui import TerminalUI
                tui = TerminalUI(self)
                supervisor.create("runtime.ui", tui.run(), critical=False)
            except Exception:
                logger.info("terminal UI unavailable (install rich>=13.0 for TUI support)")

    async def _cold_start_readiness(self) -> None:
        wallet_snapshot = await self.wallet.snapshot()
        logger.info("wallet configured=%s ready=%s owner=%s", wallet_snapshot.configured, wallet_snapshot.ready, wallet_snapshot.address or "none")
        if not self.config.execution.dry_run and not wallet_snapshot.ready:
            result = await self.approvals.ensure_approvals()
            if not result.submitted:
                raise RuntimeError(f"wallet not ready: {result.message}")

    async def _install_ptb_market(self, market: PTBMarket) -> None:
        await self.market_state.install_round(market)
        self.market_subscription.update(market)
        logger.info(
            "active PTB market slug=%s condition=%s ptb=%.2f up=%s down=%s",
            market.slug,
            market.condition_id,
            market.price_to_beat.value,
            market.up.token_id[:10],
            market.down.token_id[:10],
        )
        if self._poly_market_feed is not None:
            await self._poly_market_feed.resubscribe()
        self.counters.rollovers += 1

    async def _rollover_once(self) -> bool:
        now = time.time()
        market = await self.ptb.get_cached_round(now)
        if market is None:
            market = await self.ptb.bootstrap_round(now)
        await self._install_ptb_market(market)
        return True

    async def _rollover_loop(self) -> None:
        active_bucket = await self.market_state.current_bucket()
        failed_bucket: int | None = None
        while True:
            await asyncio.sleep(0.2)
            current_bucket = bucket_5m()
            if current_bucket == active_bucket:
                continue
            if current_bucket == failed_bucket:
                await asyncio.sleep(2.0)
                continue
            try:
                rolled = await self._rollover_once()
            except PTBMetadataUnavailable as exc:
                logger.warning("rollover failed for bucket=%s: %s", current_bucket, exc)
                rolled = False
            if rolled:
                active_bucket = current_bucket
                failed_bucket = None
            else:
                failed_bucket = current_bucket

    async def _event_processor(self) -> None:
        while True:
            event = await self.events.get()
            try:
                await self._process_event(event)
            finally:
                self.events.task_done()

    async def _process_event(self, event: RuntimeEvent) -> None:
        payload = event.payload
        if isinstance(payload, PriceTick):
            self.microstructure.update_truth_price(payload)
            await self.market_state.update_truth(payload)
            self.counters.truth_ticks += 1
        elif isinstance(payload, TopOfBook):
            self.microstructure.books.update_exchange(payload)
            if payload.source.value.startswith("polymarket"):
                self.counters.poly_events += 1
            else:
                self.counters.exchange_events += 1
        elif isinstance(payload, TradeTick):
            self.microstructure.update_trade(payload)
            if payload.source.value.startswith("polymarket"):
                self.counters.poly_events += 1
            else:
                self.counters.exchange_events += 1
        await self._maybe_evaluate(event.recv_mono_ns)

    async def _maybe_evaluate(self, recv_ns: int) -> None:
        state = await self.market_state.current()
        if state is None:
            return
        output = self.strategy.evaluate(
            ts_ns=recv_ns,
            market=state.market,
            seconds_to_expiry=state.seconds_to_expiry,
            position=state.position,
        )
        if output is None:
            return
        self.counters.strategy_evaluations += 1
        self._last_estimate_by_bucket[state.bucket] = output.estimate.p_up
        if not output.risk.allowed or output.risk.intent is None:
            previous = self._last_risk_reason_by_bucket.get(state.bucket)
            if previous != output.risk.reason:
                self._last_risk_reason_by_bucket[state.bucket] = output.risk.reason
                logger.info(
                    "strategy blocked bucket=%s reason=%s p_up=%.3f confidence=%.3f entropy=%.3f",
                    state.bucket,
                    output.risk.reason,
                    output.estimate.p_up,
                    output.estimate.confidence,
                    output.estimate.entropy,
                )
            return
        self.counters.order_intents += 1
        logger.info(
            "strategy intent bucket=%s outcome=%s p_up=%.3f confidence=%.3f limit=%.4f",
            output.risk.intent.bucket,
            output.risk.intent.outcome.value,
            output.estimate.p_up,
            output.estimate.confidence,
            output.risk.intent.limit_price,
        )
        ack = await self.execution.submit_once(output.risk.intent)
        if ack.accepted:
            logger.info("order accepted status=%s id=%s message=%s", ack.status, ack.order_id, ack.message)
            self.counters.orders_accepted += 1
            self.strategy.mark_submitted(output.risk.intent.bucket)

    async def _settlement_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            archive = await self.market_state.archive()
            current = await self.market_state.current()
            states = list(archive.values())
            if current is not None:
                states.append(current)
            now = time.time()
            for state in states:
                market = state.market
                if market.condition_id in self._settlement_seen or now < market.close_ts + 2.0:
                    continue
                self.counters.settlement_checks += 1
                truth = await self.settlement.verify_condition(market.condition_id)
                if truth is None or not truth.resolved:
                    continue
                self._settlement_seen.add(market.condition_id)
                await self.market_state.update_settlement(truth)
                predicted = self._last_estimate_by_bucket.get(market.bucket)
                if predicted is not None and truth.winner is not None:
                    self.strategy.update_settlement_label(predicted, truth.winner.value == "UP")
                if self.config.settlement.auto_claim:
                    with contextlib.suppress(Exception):
                        await self.settlement.redeem_positions(
                            market.condition_id,
                            self.config.polymarket.private_key,
                            self.config.contracts.pusd,
                        )

    async def _health_loop(self) -> None:
        while True:
            await asyncio.sleep(5.0)
            statuses = self.health.status()
            state = await self.market_state.current()
            estimate = self.strategy.last_estimate
            logger.info(
                "status bucket=%s truth=%s exchange_events=%s poly_events=%s evals=%s intents=%s accepted=%s p_up=%s feeds=%s",
                state.bucket if state is not None else "none",
                self.counters.truth_ticks,
                self.counters.exchange_events,
                self.counters.poly_events,
                self.counters.strategy_evaluations,
                self.counters.order_intents,
                self.counters.orders_accepted,
                f"{estimate.p_up:.3f}" if estimate is not None else "n/a",
                ",".join(
                    f"{item.name}:{'ok' if not item.stale(time.monotonic_ns()) else 'stale'}"
                    for item in statuses.values()
                ),
            )

    async def _on_truth_tick(self, tick: PriceTick) -> None:
        await self._put_event(RuntimeEvent(FeedKind.CHAINLINK, tick, tick.recv_mono_ns))

    async def _on_exchange_event(self, event: TopOfBook | TradeTick) -> None:
        kind = FeedKind.BINANCE if event.source is SignalSource.BINANCE else FeedKind.COINBASE
        await self._put_event(RuntimeEvent(kind, event, event.recv_mono_ns))

    async def _on_poly_event(self, event: TopOfBook | TradeTick) -> None:
        await self._put_event(RuntimeEvent(FeedKind.POLYMARKET, event, event.recv_mono_ns))

    async def _on_user_event(self, payload: dict) -> None:
        return None

    async def _put_event(self, event: RuntimeEvent) -> None:
        try:
            self.events.put_nowait(event)
        except asyncio.QueueFull:
            _ = self.events.get_nowait()
            self.events.task_done()
            self.events.put_nowait(event)


async def run_bot(config: BotConfig, enable_tui: bool = False) -> None:
    await InstitutionalBTCPolyBot(config, enable_tui).start()
