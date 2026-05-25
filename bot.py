"""
Polymarket BTC UP/DOWN 5M Trading Bot — Main Orchestrator

Wiring summary:
  Clock (deterministic) → MarketManager (slug/token fetch)
  MarketManager         → OrderbookManager (register/activate books)
  WebSocket streams     → OrderbookManager (route price events)
  OrderbookManager      → ProbabilityEngine (signal per round)
  ProbabilityEngine     → RiskEngine → ExecutionEngine

Multi-round lifecycle:
  T-60s  : prewarm triggered for NEXT slug
             → subscribe WS to next UP+DOWN token IDs
             → fetch REST snapshot for next book
             → ProbabilityEngine warms up
  T=0    : rollover
             → activate NEXT BookSet as ACTIVE (zero-latency swap)
             → previous ACTIVE evicted after 30s
             → begin prewarm for ROUND+2
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import signal
import socket
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import aiohttp

from config.settings import BotConfig, load_config
from core.constants import (
    CLOB_HOST,
    CTF_CONTRACT_ADDRESS,
    WS_MARKET_ENDPOINT,
    WS_USER_ENDPOINT,
)
from core.types import (
    CircuitState,
    Direction,
    MarketPhase,
    MarketTokenPair,
    MetricsSnapshot,
    SignalStrength,
    UnifiedSignal,
)
from execution.client import ClobApiError, ClobClient
from execution.engine import ExecutionEngine
from market.clock import current_window, next_window, window_schedule
from market.manager import MarketManager, MarketNotFoundError
from monitoring.logger import (
    MetricsCollector,
    TradeLogger,
    setup_logging,
    start_metrics_server,
)
from network.dns_resolver import DoHResolver
from orderbook.book import L2Orderbook
from orderbook.manager import BookSet, OrderbookManager
from recovery.state_manager import RecoveryManager, StateStore
from risk.engine import RiskEngine, RiskRejection
from signals.probability_engine import ProbabilityEngine
from websocket_engine.stream import StreamType, WebSocketStream

logger = logging.getLogger(__name__)

_SIGNAL_INTERVAL_S  = 0.2    # 200 ms signal loop
_METRICS_INTERVAL_S = 10.0
_STATS_INTERVAL_S   = 60.0   # Print session stats every 60s
_LOCK_FILE          = Path("state/bot.lock")
_CTF_PAYOUT_DENOMINATOR_SELECTOR = "dd34de67"  # payoutDenominator(bytes32)
_CTF_PAYOUT_NUMERATOR_SELECTOR   = "0504c814"  # payoutNumerators(bytes32,uint256)
_CTF_SETTLEMENT_RETRIES          = 12
_CTF_SETTLEMENT_RETRY_DELAY_S    = 5.0
_CTF_SETTLEMENT_RECHECK_DELAY_S  = 60.0


def _build_tcp_connector(config: BotConfig) -> aiohttp.TCPConnector:
    connector_kwargs: Dict[str, Any] = {
        "ssl": True,
        "limit": 40,
        "limit_per_host": 10,
        "ttl_dns_cache": 0,
        "use_dns_cache": False,
        "keepalive_timeout": 60,
        "enable_cleanup_closed": True,
    }

    params = inspect.signature(aiohttp.TCPConnector.__init__).parameters
    if "socket_factory" in params:
        connector_kwargs["socket_factory"] = _make_socket_factory(
            tcp_nodelay=config.network.tcp_nodelay,
            tcp_keepalive=config.network.tcp_keepalive,
        )
    elif "tcp_nodelay" in params:
        connector_kwargs["tcp_nodelay"] = config.network.tcp_nodelay

    return aiohttp.TCPConnector(**connector_kwargs)


def _make_socket_factory(tcp_nodelay: bool, tcp_keepalive: bool):
    def socket_factory(addr_info):
        family, type_, proto, _, _ = addr_info
        sock = socket.socket(family=family, type=type_, proto=proto)

        if tcp_nodelay:
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

        if tcp_keepalive:
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            except OSError:
                pass

        return sock

    return socket_factory


def _uint256_word(value: int) -> str:
    return f"{value:064x}"


def _condition_word(condition_id: str) -> str:
    cleaned = condition_id[2:] if condition_id.startswith("0x") else condition_id
    if len(cleaned) != 64:
        raise ValueError(f"Invalid condition id length: {condition_id}")
    int(cleaned, 16)
    return cleaned.lower()


class SingleInstanceLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle = None

    def acquire(self) -> bool:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._handle = os.open(
                self._path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
            )
            os.write(self._handle, str(os.getpid()).encode("ascii"))
            return True
        except FileExistsError:
            try:
                pid = int(self._path.read_text(encoding="ascii").strip() or "0")
                os.kill(pid, 0)
                return False
            except OSError:
                self._path.unlink(missing_ok=True)
                return self.acquire()
            except Exception:
                return False

    def release(self) -> None:
        if self._handle is not None:
            os.close(self._handle)
            self._handle = None
        self._path.unlink(missing_ok=True)


class Bot:
    def __init__(self, config: BotConfig) -> None:
        self._cfg = config

        self._session:      Optional[aiohttp.ClientSession] = None
        self._doh:          Optional[DoHResolver]           = None
        self._clob:         Optional[ClobClient]            = None
        self._risk:         Optional[RiskEngine]            = None
        self._exec:         Optional[ExecutionEngine]       = None
        self._market_mgr:   Optional[MarketManager]         = None
        self._book_mgr:     Optional[OrderbookManager]      = None
        self._recovery_mgr: Optional[RecoveryManager]       = None

        # Per-active-book probability engine
        self._prob_engine:  Optional[ProbabilityEngine]     = None

        # WebSocket streams
        self._market_ws:    Optional[WebSocketStream]       = None
        self._user_ws:      Optional[WebSocketStream]       = None

        # Observability
        self._metrics    = MetricsCollector()
        self._trade_log  = TradeLogger()

        self._shutdown   = asyncio.Event()

        # Cycle guard: only one trade per 5-min slug
        self._last_traded_slug: Optional[str] = None

        # Session stats
        self._session_wins:   int = 0
        self._session_losses: int = 0
        self._session_pnl:    Decimal = Decimal("0")
        self._started_at:     float = 0.0
        self._last_signal_log_at: float = 0.0
        self._last_signal_log_slug: Optional[str] = None

        # Server clock offset (computed at startup)
        self._server_offset: float = 0.0

        # Market metadata retained after orderbook eviction for CTF settlement checks.
        self._markets_by_condition: Dict[str, MarketTokenPair] = {}
        self._settlement_inflight: set[str] = set()

    # ─────────────────────────── Run ───────────────────────────

    async def run(self) -> None:
        self._started_at = time.time()
        logger.info(
            "━" * 64 + "\n  Polymarket BTC UP/DOWN 5M Bot  |  dry_run=%s\n" + "━" * 64,
            self._cfg.dry_run,
        )
        try:
            await self._init()
            await self._recover()
            await self._trading_loop()
        except asyncio.CancelledError:
            logger.info("Shutdown signal received")
        except MarketNotFoundError as exc:
            logger.critical("Cannot find active market: %s", exc)
        except Exception as exc:
            logger.critical("Fatal bot error", exc_info=True)
            if self._risk:
                self._risk.activate_kill_switch(f"Fatal: {type(exc).__name__}: {exc}")
        finally:
            await self._shutdown_gracefully()

    # ─────────────────────────── Initialization ───────────────────────────

    async def _init(self) -> None:
        logger.info("Initializing components …")

        # ── HTTP Session ──
        connector = _build_tcp_connector(self._cfg)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=30, connect=10),
            headers={"User-Agent": "PolymarketBot/2.0", "Accept": "application/json"},
        )

        # ── DoH DNS pre-resolve ──
        self._doh = DoHResolver(session=self._session)
        await self._doh._prefetch()

        # ── Server time sync ──
        self._clob = ClobClient(
            session=self._session,
            private_key=self._cfg.auth.private_key,
            api_key=self._cfg.auth.api_key,
            api_secret=self._cfg.auth.api_secret,
            api_passphrase=self._cfg.auth.api_passphrase,
            funder_address=self._cfg.auth.funder_address,
            signature_type=self._cfg.auth.signature_type,
        )
        server_ts = await self._clob.get_server_time()
        self._server_offset = server_ts - time.time()
        if abs(self._server_offset) > 3:
            logger.warning(
                "Clock skew detected: local vs server = %+.1fs", self._server_offset
            )

        has_l2_auth = await self._clob.ensure_api_credentials(
            auto_create=self._cfg.auth.auto_create_api_key
        )
        old_auth = (
            self._cfg.auth.api_key,
            self._cfg.auth.api_secret,
            self._cfg.auth.api_passphrase,
        )
        self._cfg.auth.api_key = self._clob.api_key
        self._cfg.auth.api_secret = self._clob.api_secret
        self._cfg.auth.api_passphrase = self._clob.api_passphrase
        new_auth = (
            self._cfg.auth.api_key,
            self._cfg.auth.api_secret,
            self._cfg.auth.api_passphrase,
        )
        if has_l2_auth and old_auth != new_auth:
            self._persist_api_credentials()
        if not has_l2_auth:
            self._cfg.auth.enable_user_ws = False
            self._cfg.auth.enable_heartbeat = False
            if not self._cfg.dry_run:
                self._cfg.dry_run = True
            logger.warning(
                "L2 auth unavailable; user websocket/heartbeat disabled and DRY_RUN forced on"
            )

        # ── Risk engine ──
        self._risk = RiskEngine(
            config=self._cfg.risk,
            initial_balance_usdc=Decimal("100"),
        )

        # ── Execution engine ──
        self._exec = ExecutionEngine(
            client=self._clob,
            risk_engine=self._risk,
            config=self._cfg,
        )

        # ── Orderbook manager ──
        self._book_mgr = OrderbookManager()

        # ── Market manager (multi-round) ──
        self._market_mgr = MarketManager(
            session=self._session,
            on_new_market=self._on_new_market,
            on_market_resolved=self._on_market_resolved,
            on_prewarm_ready=self._on_prewarm_ready,
            server_time_offset=self._server_offset,
        )

        # ── Recovery ──
        self._recovery_mgr = RecoveryManager(
            store=StateStore(),
            risk_engine=self._risk,
        )

        # ── WebSocket streams ──
        self._market_ws = WebSocketStream(
            name="market",
            url=WS_MARKET_ENDPOINT,
            stream_type=StreamType.MARKET,
            on_message=self._on_market_ws_msg,
            on_reconnect=self._on_ws_reconnect,
        )
        if self._cfg.auth.enable_user_ws:
            auth_cfg = {
                "api_key":        self._cfg.auth.api_key,
                "api_secret":     self._cfg.auth.api_secret,
                "api_passphrase": self._cfg.auth.api_passphrase,
            }
            self._user_ws = WebSocketStream(
                name="user",
                url=WS_USER_ENDPOINT,
                stream_type=StreamType.USER,
                auth_config=auth_cfg,
                on_message=self._on_user_ws_msg,
            )
        else:
            self._user_ws = None

        # ── Heartbeat ──
        if self._cfg.auth.enable_heartbeat:
            await self._clob.start_heartbeat(self._cfg.trading.heartbeat_interval_s)

        # ── Start streams ──
        await self._market_ws.start(self._session)
        if self._user_ws:
            await self._user_ws.start(self._session)

        # ── Market bootstrap (fetches current + next N slugs) ──
        await self._market_mgr.start()

        # ── Recovery manager ──
        await self._recovery_mgr.start()

        # ── Metrics HTTP ──
        metrics_task = asyncio.create_task(
            start_metrics_server(self._metrics, self._cfg.metrics_port),
            name="metrics-http",
        )
        metrics_task.add_done_callback(self._log_background_task_result)

        logger.info("All components ready")

    @staticmethod
    def _log_background_task_result(task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            logger.warning("Background task %s stopped: %s", task.get_name(), exc)

    async def _recover(self) -> None:
        if self._recovery_mgr:
            await self._recovery_mgr.recover()

    # ─────────────────────────── Trading Loop ───────────────────────────

    async def _trading_loop(self) -> None:
        logger.info("Trading loop started")
        last_metrics = time.monotonic()
        last_stats   = time.monotonic()

        while not self._shutdown.is_set():
            try:
                t0    = time.monotonic()
                phase = self._market_mgr.update_phase() if self._market_mgr else MarketPhase.WAITING

                if phase == MarketPhase.ACTIVE:
                    await self._tick_active()
                elif phase in (MarketPhase.WAITING, MarketPhase.TRANSITIONING):
                    await asyncio.sleep(0.5)
                    continue
                elif phase == MarketPhase.NEAR_EXPIRY:
                    await asyncio.sleep(1.0)   # No new entries near expiry
                    continue
                elif phase == MarketPhase.RESOLVED:
                    await asyncio.sleep(0.2)
                    continue

                # Periodic tasks
                now = time.monotonic()
                if now - last_metrics >= _METRICS_INTERVAL_S:
                    await self._emit_metrics()
                    last_metrics = now
                if now - last_stats >= _STATS_INTERVAL_S:
                    self._print_session_stats()
                    last_stats = now

                elapsed    = time.monotonic() - t0
                sleep_time = max(0.0, _SIGNAL_INTERVAL_S - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Trading loop error: %s", exc, exc_info=True)
                self._metrics.inc("loop_errors")
                await asyncio.sleep(1.0)

    async def _tick_active(self) -> None:
        """Single signal-compute + entry-evaluate tick."""
        if self._book_mgr is None or self._prob_engine is None:
            return

        active_bs = self._book_mgr.active
        if active_bs is None or not active_bs.both_initialized:
            return

        signal = await self._prob_engine.compute()
        if signal is None:
            return

        self._metrics.inc("signals_computed")

        # Active market info
        market = self._market_mgr.active_market if self._market_mgr else None
        if market is None:
            return

        # Log non-trivial signals at debug level
        if self._should_log_signal(signal, market.slug):
            ev = self._prob_engine.compute_entry_ev(
                signal, signal.up_microprice or Decimal("0.5"), signal.direction
            )
            self._trade_log.signal(
                slug=market.slug,
                direction=signal.direction.value,
                strength=signal.signal_strength.name,
                confidence=float(signal.confidence),
                up_prob=float(signal.up_probability),
                down_prob=float(signal.down_probability),
                ev=float(ev),
                tradeable=signal.is_tradeable,
            )

        if signal.is_tradeable:
            await self._evaluate_entry(signal, market, active_bs)

    def _should_log_signal(self, signal: UnifiedSignal, slug: str) -> bool:
        if self._cfg.log_all_signals:
            return True

        now = time.monotonic()
        if slug != self._last_signal_log_slug:
            self._last_signal_log_slug = slug
            self._last_signal_log_at = now
            return True
        if now - self._last_signal_log_at >= self._cfg.signal_log_interval_s:
            self._last_signal_log_at = now
            return True
        return False

    def _persist_api_credentials(self) -> None:
        env_path = Path(".env")
        if not env_path.exists():
            return

        updates = {
            "POLY_API_KEY": self._cfg.auth.api_key,
            "POLY_API_SECRET": self._cfg.auth.api_secret,
            "POLY_API_PASSPHRASE": self._cfg.auth.api_passphrase,
        }
        lines = env_path.read_text(encoding="utf-8").splitlines()
        seen = set()
        next_lines = []
        for line in lines:
            key = line.split("=", 1)[0].strip() if "=" in line else ""
            if key in updates:
                next_lines.append(f"{key}={updates[key]}")
                seen.add(key)
            else:
                next_lines.append(line)
        for key, value in updates.items():
            if key not in seen:
                next_lines.append(f"{key}={value}")

        env_path.write_text("\n".join(next_lines) + "\n", encoding="utf-8")
        logger.info("Persisted derived L2 API credentials to .env")

    # ─────────────────────────── Entry Evaluation ───────────────────────────

    async def _evaluate_entry(
        self,
        signal:    UnifiedSignal,
        market:    MarketTokenPair,
        book_set:  BookSet,
    ) -> None:
        # One trade per cycle
        if market.slug == self._last_traded_slug:
            return

        # Risk state gate
        risk_state = await self._risk.get_state()
        if risk_state.kill_switch_active:
            return
        if risk_state.circuit_state == CircuitState.OPEN:
            return

        # Market anomaly gate
        is_anomaly, reason = await self._risk.check_market_anomaly(signal)
        if is_anomaly:
            self._trade_log.signal_blocked(
                slug=market.slug,
                reason=reason,
                strength=signal.signal_strength.name,
            )
            self._metrics.inc("entries_blocked_anomaly")
            return

        # Min time remaining (don't enter with < 60s left)
        if market.is_near_expiry:
            return

        # Compute EV with correct direction
        direction = signal.direction
        up_snap   = await book_set.up_snapshot()
        down_snap = await book_set.down_snapshot()
        if up_snap is None or down_snap is None:
            return

        entry_price = (up_snap.best_ask if direction == Direction.UP else down_snap.best_ask)
        if entry_price is None:
            return

        ev = self._prob_engine.compute_entry_ev(signal, entry_price, direction)
        min_ev = self._cfg.signal.min_ev_threshold

        if ev < min_ev:
            logger.debug(
                "EV %.4f below threshold %.4f — skip  (dir=%s price=%.3f)",
                float(ev), float(min_ev), direction.value, float(entry_price),
            )
            return

        logger.info(
            "Entry candidate: dir=%s  strength=%s  conf=%.2f  EV=%+.4f",
            direction.value, signal.signal_strength.name,
            float(signal.confidence), float(ev),
        )

        # Execute
        position = await self._exec.execute_entry(
            signal=signal,
            direction=direction,
            market_pair=market,
            current_up_state=up_snap,
            current_down_state=down_snap,
        )

        if position:
            self._last_traded_slug = market.slug
            self._metrics.inc("entries_executed")
            if self._recovery_mgr:
                await self._recovery_mgr.save_now()
            self._trade_log.entry_placed(
                slug=market.slug,
                direction=direction.value,
                price=float(position.entry_price),
                shares=float(position.size),
                cost_usdc=float(position.cost_basis),
                fee_usdc=float(position.fee_paid),
                order_id=position.order_id,
                dry_run=self._cfg.dry_run,
            )
        else:
            self._metrics.inc("entries_failed")

    # ─────────────────────────── Market Lifecycle Callbacks ───────────────────────────

    def _on_new_market(self, market: MarketTokenPair, was_prewarmed: bool) -> None:
        """Called on rollover — new active market."""
        asyncio.create_task(
            self._activate_market(market, was_prewarmed),
            name=f"activate-{market.slug[-14:]}",
        )

    async def _activate_market(self, market: MarketTokenPair, was_prewarmed: bool) -> None:
        if self._book_mgr is None:
            return
        self._remember_market(market)

        # Activate or register BookSet
        bs = await self._book_mgr.activate(market.slug)
        if bs is None:
            bs = await self._book_mgr.register(market, is_active=True)

        # Build probability engine for this BookSet
        self._prob_engine = ProbabilityEngine(
            up_book=bs.up_book,
            down_book=bs.down_book,
            signal_config=self._cfg.signal,
        )

        # Subscribe WS if not already
        await self._subscribe_books([market])

        # Fetch REST snapshot if cold start
        if not was_prewarmed or not bs.both_initialized:
            asyncio.create_task(
                self._fetch_snapshots(market),
                name=f"snap-{market.slug[-14:]}",
            )

        tr = market.time_remaining
        self._trade_log.market_active(market.slug, tr, was_prewarmed)
        self._metrics.inc("market_rollovers")

        # Evict stale books (keep active + 2 future)
        keep = {market.slug}
        nw1  = next_window(1, self._server_offset)
        nw2  = next_window(2, self._server_offset)
        keep.update([nw1.slug, nw2.slug])
        asyncio.create_task(
            self._book_mgr.evict_old(keep),
            name="evict-old-books",
        )

    def _on_prewarm_ready(self, market: MarketTokenPair) -> None:
        """Called when next-round pre-warm should start."""
        asyncio.create_task(
            self._prewarm_market(market),
            name=f"prewarm-{market.slug[-14:]}",
        )

    async def _prewarm_market(self, market: MarketTokenPair) -> None:
        if self._book_mgr is None:
            return
        self._remember_market(market)

        self._trade_log.prewarm_start(market.slug)

        bs = await self._book_mgr.register(market, is_active=False)
        bs.is_prewarm = True

        # Subscribe WS to next-round tokens immediately
        await self._subscribe_books([market])

        # Fetch REST snapshot to seed the book
        await self._fetch_snapshots(market)

        # Compute initial depths for log
        up_snap   = await bs.up_snapshot()
        down_snap = await bs.down_snapshot()
        up_d      = float(up_snap.bid_depth(5) + up_snap.ask_depth(5)) if up_snap else 0.0
        down_d    = float(down_snap.bid_depth(5) + down_snap.ask_depth(5)) if down_snap else 0.0

        self._trade_log.prewarm_ready(market.slug, up_d, down_d)
        if self._book_mgr:
            self._book_mgr.mark_book_ready(market.slug)

    def _remember_market(self, market: MarketTokenPair) -> None:
        if market.condition_id:
            self._markets_by_condition[market.condition_id] = market

    def _market_for_condition(self, condition_id: str) -> Optional[MarketTokenPair]:
        market = self._markets_by_condition.get(condition_id)
        if market is not None:
            return market
        if self._book_mgr:
            for bs in self._book_mgr._books.values():
                if bs.market.condition_id == condition_id:
                    self._remember_market(bs.market)
                    return bs.market
        return None

    def _on_market_resolved(self, condition_id: str, winning_asset_id: str) -> None:
        if condition_id in self._settlement_inflight:
            logger.debug("Settlement already in progress for %s", condition_id[:14])
            return
        self._settlement_inflight.add(condition_id)
        task = asyncio.create_task(
            self._handle_resolution(condition_id, winning_asset_id),
            name="handle-resolution",
        )
        task.add_done_callback(
            lambda done, cid=condition_id: self._on_settlement_task_done(cid, done)
        )

    def _on_settlement_task_done(self, condition_id: str, task: asyncio.Task) -> None:
        self._settlement_inflight.discard(condition_id)
        self._log_background_task_result(task)

    def _schedule_ctf_recheck(self, condition_id: str, winning_asset_id: str) -> None:
        task = asyncio.create_task(
            self._ctf_recheck_later(condition_id, winning_asset_id),
            name=f"ctf-recheck-{condition_id[:10]}",
        )
        task.add_done_callback(self._log_background_task_result)

    async def _ctf_recheck_later(self, condition_id: str, winning_asset_id: str) -> None:
        await asyncio.sleep(_CTF_SETTLEMENT_RECHECK_DELAY_S)
        self._on_market_resolved(condition_id, winning_asset_id)

    async def _handle_resolution(self, condition_id: str, winning_asset_id: str) -> None:
        positions = await self._risk.get_positions() if self._risk else {}
        condition_positions = {
            asset_id: pos
            for asset_id, pos in positions.items()
            if pos.condition_id == condition_id
        }
        if not condition_positions or self._exec is None:
            return

        market = self._market_for_condition(condition_id)
        slug = market.slug if market is not None else condition_id[:14]
        if market is None:
            logger.warning(
                "Settlement skipped for %s: missing market metadata needed to map CTF payouts",
                condition_id[:14],
            )
            return

        settlement_asset_id, winning_side = await self._resolve_ctf_winner(
            condition_id=condition_id,
            market=market,
        )
        if not settlement_asset_id:
            logger.warning(
                "Settlement pending for %s: CTF did not return a valid winner; positions left open",
                condition_id[:14],
            )
            self._schedule_ctf_recheck(condition_id, winning_asset_id)
            return

        if winning_asset_id and winning_asset_id != settlement_asset_id:
            ws_side = self._side_for_asset(condition_id, winning_asset_id)
            logger.warning(
                "WS settlement mismatch for %s: ws=%s/%s ctf=%s/%s; using CTF",
                condition_id[:14],
                (ws_side or "?"),
                winning_asset_id[:16],
                winning_side,
                settlement_asset_id[:16],
            )

        for asset_id, pos in list(condition_positions.items()):
            won        = (asset_id == settlement_asset_id)
            cost       = float(pos.cost_basis)
            fee        = float(pos.fee_paid)

            if won:
                pnl     = float(pos.size) - cost - fee
                roi_pct = (pnl / cost) * 100 if cost > 0 else 0.0
                self._trade_log.exit_win(
                    slug=slug,
                    direction=pos.direction.value,
                    pnl_usdc=pnl,
                    cost_usdc=cost,
                    roi_pct=roi_pct,
                )
                self._session_wins  += 1
                self._session_pnl   += Decimal(str(pnl))
            else:
                pnl     = -(cost + fee)
                roi_pct = (pnl / cost) * 100 if cost > 0 else 0.0
                self._trade_log.exit_loss(
                    slug=slug,
                    direction=pos.direction.value,
                    pnl_usdc=pnl,
                    cost_usdc=cost,
                    roi_pct=roi_pct,
                )
                self._session_losses += 1
                self._session_pnl    += Decimal(str(pnl))

            await self._risk.record_exit(
                asset_id=asset_id,
                pnl_usdc=Decimal(str(pnl)),
                won=won,
            )
            if self._recovery_mgr:
                await self._recovery_mgr.save_now()
            self._metrics.inc("exits_total")

        if winning_side is None:
            logger.warning(
                "Resolved %s but could not map winning asset %s to UP/DOWN",
                condition_id[:14],
                settlement_asset_id[:16],
            )
            return
        self._trade_log.market_resolved(
            slug=slug,
            winning_side=winning_side,
            up_price=1.0 if winning_side == "UP" else 0.0,
            down_price=0.0 if winning_side == "UP" else 1.0,
        )

    def _side_for_asset(self, condition_id: str, asset_id: str) -> Optional[str]:
        market = self._market_for_condition(condition_id)
        if market is None:
            return None
        if asset_id == market.up_token_id:
            return "UP"
        if asset_id == market.down_token_id:
            return "DOWN"
        return None

    async def _resolve_ctf_winner(
        self,
        condition_id: str,
        market: MarketTokenPair,
    ) -> Tuple[Optional[str], Optional[str]]:
        last_reason = "not attempted"
        for attempt in range(1, _CTF_SETTLEMENT_RETRIES + 1):
            try:
                payouts = await self._fetch_ctf_payouts(condition_id)
            except Exception as exc:
                last_reason = str(exc)
            else:
                if payouts is None:
                    last_reason = "payout denominator is still zero"
                else:
                    up_payout, down_payout = payouts
                    if up_payout == down_payout:
                        logger.warning(
                            "CTF settlement ambiguous for %s: payouts=UP:%s DOWN:%s",
                            condition_id[:14], up_payout, down_payout
                        )
                        return None, None
                    if up_payout > down_payout:
                        return market.up_token_id, "UP"
                    return market.down_token_id, "DOWN"

            if attempt < _CTF_SETTLEMENT_RETRIES:
                logger.info(
                    "CTF settlement not ready for %s (%s); retry %d/%d in %.0fs",
                    condition_id[:14],
                    last_reason,
                    attempt + 1,
                    _CTF_SETTLEMENT_RETRIES,
                    _CTF_SETTLEMENT_RETRY_DELAY_S,
                )
                await asyncio.sleep(_CTF_SETTLEMENT_RETRY_DELAY_S)

        logger.warning(
            "CTF settlement unavailable for %s after %d attempts: %s",
            condition_id[:14],
            _CTF_SETTLEMENT_RETRIES,
            last_reason,
        )
        return None, None

    async def _fetch_ctf_payouts(self, condition_id: str) -> Optional[Tuple[int, int]]:
        condition_word = _condition_word(condition_id)
        denominator = await self._ctf_call_uint(
            _CTF_PAYOUT_DENOMINATOR_SELECTOR + condition_word
        )
        if denominator <= 0:
            return None
        payouts = []
        for index in (0, 1):
            data = (
                _CTF_PAYOUT_NUMERATOR_SELECTOR
                + condition_word
                + _uint256_word(index)
            )
            payouts.append(await self._ctf_call_uint(data))
        logger.info(
            "CTF settlement: condition=%s payouts=UP:%d DOWN:%d denominator=%d",
            condition_id[:14], payouts[0], payouts[1], denominator,
        )
        return payouts[0], payouts[1]

    async def _ctf_call_uint(self, data: str) -> int:
        if self._session is None:
            raise RuntimeError("HTTP session not initialized")
        call_data = data[2:] if data.startswith("0x") else data
        payload = {
            "jsonrpc": "2.0",
            "id": int(time.time() * 1000) % 1_000_000,
            "method": "eth_call",
            "params": [
                {
                    "to": CTF_CONTRACT_ADDRESS,
                    "data": "0x" + call_data,
                },
                "latest",
            ],
        }
        async with self._session.post(
            self._cfg.network.polygon_rpc_url,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=8),
        ) as resp:
            body = await resp.json(content_type=None)
        if resp.status >= 400:
            raise RuntimeError(f"RPC HTTP {resp.status}: {body}")
        if "error" in body:
            raise RuntimeError(body["error"])
        result = body.get("result", "0x")
        return int(result, 16) if result and result != "0x" else 0

    # ─────────────────────────── WS Event Routing ───────────────────────────

    def _on_market_ws_msg(self, msg: Dict[str, Any]) -> None:
        evt = msg.get("event_type", "")
        if evt == "book":
            asyncio.create_task(self._handle_book_snapshot(msg), name="ws-book")
        elif evt == "price_change":
            asyncio.create_task(self._handle_price_change(msg), name="ws-price")
        elif evt == "last_trade_price":
            asyncio.create_task(self._handle_trade_event(msg), name="ws-trade")
        elif evt in ("new_market", "market_resolved"):
            if self._market_mgr:
                self._market_mgr.handle_ws_message(msg)

    def _on_user_ws_msg(self, msg: Dict[str, Any]) -> None:
        if self._exec:
            self._exec.handle_user_message(msg)

    def _on_ws_reconnect(self) -> None:
        """Re-fetch snapshots for all tracked books after WS reconnect."""
        logger.warning("Market WS reconnected — re-syncing all orderbooks")
        if self._book_mgr:
            for bs in list(self._book_mgr._books.values()):
                asyncio.create_task(
                    self._fetch_snapshots(bs.market),
                    name=f"resync-{bs.slug[-14:]}",
                )

    # ─────────────────────────── Orderbook Event Handlers ───────────────────────────

    async def _handle_book_snapshot(self, msg: Dict[str, Any]) -> None:
        asset_id = msg.get("asset_id", "")
        book     = self._book_mgr.book_for_asset(asset_id) if self._book_mgr else None
        if book is None:
            return
        await book.apply_snapshot(
            bids=msg.get("bids", []),
            asks=msg.get("asks", []),
            timestamp=msg.get("timestamp", "0"),
            hash_val=msg.get("hash", ""),
        )
        # Mark pre-warm book as ready after first snapshot
        if self._book_mgr:
            bs = self._book_mgr.bookset_for_asset(asset_id)
            if bs and bs.is_prewarm and bs.both_initialized:
                self._book_mgr.mark_book_ready(bs.slug)

    async def _handle_price_change(self, msg: Dict[str, Any]) -> None:
        from core.types import Side
        changes   = msg.get("price_changes", [])
        timestamp = msg.get("timestamp", "0")
        for ch in changes:
            asset_id = ch.get("asset_id", "")
            book     = self._book_mgr.book_for_asset(asset_id) if self._book_mgr else None
            if book is None:
                continue
            price    = Decimal(ch.get("price", "0"))
            size     = Decimal(ch.get("size",  "0"))
            side     = Side.BUY if ch.get("side", "BUY") == "BUY" else Side.SELL
            await book.apply_price_change(
                price=price, size=size, side=side,
                hash_val=ch.get("hash", ""), timestamp=timestamp,
            )

    async def _handle_trade_event(self, msg: Dict[str, Any]) -> None:
        from core.types import Side
        asset_id = msg.get("asset_id", "")
        book     = self._book_mgr.book_for_asset(asset_id) if self._book_mgr else None
        if book is None:
            return
        await book.apply_trade(
            price=Decimal(msg.get("price", "0")),
            size=Decimal(msg.get("size", "0")),
            side=Side.BUY if msg.get("side", "BUY") == "BUY" else Side.SELL,
            timestamp=msg.get("timestamp", "0"),
        )

    # ─────────────────────────── Subscriptions & Snapshots ───────────────────────────

    async def _subscribe_books(self, markets: list[MarketTokenPair]) -> None:
        """Subscribe WS + user channel to a list of markets."""
        if self._market_ws is None:
            return
        all_ids     = []
        condition_ids = []
        for m in markets:
            all_ids.extend([m.up_token_id, m.down_token_id])
            condition_ids.append(m.condition_id)
        await self._market_ws.subscribe_assets(all_ids)
        if self._user_ws:
            await self._user_ws.subscribe_user_markets(condition_ids)

    async def _fetch_snapshots(self, market: MarketTokenPair) -> None:
        """Fetch REST orderbook snapshots for both tokens of a market."""
        if self._clob is None or self._book_mgr is None:
            return
        for token_id in (market.up_token_id, market.down_token_id):
            book = self._book_mgr.book_for_asset(token_id)
            if book is None:
                continue
            try:
                data = await self._clob.get_orderbook(token_id)
                await book.apply_snapshot(
                    bids=data.get("bids", []),
                    asks=data.get("asks", []),
                    timestamp=str(int(time.time() * 1000)),
                    hash_val=data.get("hash", ""),
                )
                logger.debug(
                    "REST snapshot: token=%-14s  bids=%d  asks=%d",
                    token_id[:14], len(data.get("bids", [])), len(data.get("asks", [])),
                )
            except Exception as exc:
                logger.error("REST snapshot failed: token=%-14s  %s", token_id[:14], exc)

    # ─────────────────────────── Metrics / Stats ───────────────────────────

    async def _emit_metrics(self) -> None:
        try:
            rs = await self._risk.get_state() if self._risk else None
            if rs is None:
                return
            ab = self._book_mgr.active if self._book_mgr else None
            self._metrics.gauge("session_pnl_usdc",     float(rs.session_pnl_usdc))
            self._metrics.gauge("total_exposure_usdc",  float(rs.total_exposure_usdc))
            self._metrics.gauge("consecutive_losses",   rs.consecutive_losses)
            self._metrics.gauge("circuit_state",        rs.circuit_state.value)
            self._metrics.gauge("open_positions",       rs.open_positions)
            self._metrics.gauge("managed_book_count",
                                 self._book_mgr.slug_count() if self._book_mgr else 0)
            if ab:
                us = await ab.up_snapshot()
                ds = await ab.down_snapshot()
                if us:
                    self._metrics.gauge("up_book_depth", float(us.bid_depth(5) + us.ask_depth(5)))
                if ds:
                    self._metrics.gauge("down_book_depth", float(ds.bid_depth(5) + ds.ask_depth(5)))
            if self._market_ws:
                self._metrics.gauge("ws_latency_ms", self._market_ws.stats.latency_ms)
        except Exception:
            pass

    def _print_session_stats(self) -> None:
        total_closed = self._session_wins + self._session_losses
        wr     = (self._session_wins / total_closed * 100) if total_closed > 0 else 0.0
        uptime = time.time() - self._started_at
        session_trades = 0
        open_positions = 0
        pnl_usdc = float(self._session_pnl)
        if self._risk:
            rs = self._risk._state
            session_trades = rs.session_trades
            open_positions = rs.open_positions
            pnl_usdc = float(rs.session_pnl_usdc)
        self._trade_log.session_stats(
            trades=session_trades,
            wins=self._session_wins,
            losses=self._session_losses,
            pnl_usdc=pnl_usdc,
            win_rate=wr,
            uptime_s=uptime,
            open_positions=open_positions,
        )

    # ─────────────────────────── Graceful Shutdown ───────────────────────────

    async def _shutdown_gracefully(self) -> None:
        logger.info("Shutting down …")
        if self._exec:
            await self._exec.cancel_all_positions()
        if self._clob:
            await self._clob.stop_heartbeat()
        if self._market_ws:
            await self._market_ws.stop()
        if self._user_ws:
            await self._user_ws.stop()
        if self._market_mgr:
            await self._market_mgr.stop()
        if self._recovery_mgr:
            await self._recovery_mgr.stop()
        if self._session:
            await self._session.close()
        self._print_session_stats()
        logger.info("Shutdown complete")


# ─────────────────────────── Entry Point ───────────────────────────

def _install_signal_handlers(bot: Bot, loop: asyncio.AbstractEventLoop) -> None:
    def handle(sig_name: str) -> None:
        logger.warning("Signal %s received", sig_name)
        bot._shutdown.set()

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, handle, sig.name)
        except NotImplementedError:
            pass


async def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(".env")

    config = load_config()
    setup_logging(config.log_level, "logs")

    lock = SingleInstanceLock(_LOCK_FILE)
    if not lock.acquire():
        logger.error("Another bot instance is already running; exiting")
        return

    bot  = Bot(config)
    loop = asyncio.get_event_loop()
    _install_signal_handlers(bot, loop)
    try:
        await bot.run()
    finally:
        lock.release()


if __name__ == "__main__":
    asyncio.run(main())
