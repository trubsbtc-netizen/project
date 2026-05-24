"""
Main bot orchestrator for Polymarket BTC 5-minute directional trading.

This is the central coordination point that wires together all subsystems:
    - Data feeds (multi-exchange BTC WebSockets + Polymarket CLOB)
    - Microstructure analysis (orderbook, flow, liquidity, queue, spread)
    - Regime detection (HMM + volatility)
    - Signal generation (continuation, reversal, exhaustion, burst, entropy)
    - Inference (Kalman, Bayesian, hazard, multi-timescale, settlement)
    - Execution modeling (fill, slippage, latency)
    - Decision engine (confidence scaling, decorrelation, final gate)
    - Risk management (Kelly sizing, drawdown, cooldown, circuit breaker)
    - Polymarket order placement (with dry-run support)

The bot runs in a continuous loop:
    1. Collect market data
    2. Run microstructure analysis
    3. Detect regime
    4. Generate signals
    5. Run inference pipeline
    6. Model execution
    7. Make decision
    8. Check risk limitses
    9. Execute or suppress trade
    10. Record outcome for calibration

Each cycle produces a MarketState that flows through all subsystems
    and ultimately produces a TradeDecision.
"""

import os
import time
import logging
import asyncio
import json
import numpy as np
from dataclasses import replace
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timezone
from math import erf, sqrt

from polymarket_bot.config import BotConfig, PolymarketConfig
from polymarket_bot.fees import estimate_entry_fee_usdc
from polymarket_bot.bot_types import (
    MarketState,
    TickData,
    OrderbookSnapshot,
    OrderbookDelta,
    PolymarketOrderbook,
    PolymarketMarketInfo,
    OrderbookPressure,
    FlowMetrics,
    LiquidityMetrics,
    QueueMetrics,
    SpreadMetrics,
    RegimeStateEstimate,
    VolatilityEstimate,
    ContinuationSignal,
    ReversalSignal,
    ExhaustionSignal,
    BurstFailureSignal,
    EntropyFilterResult,
    ObservationSignal,
    TechnicalMomentumSignal,
    KalmanState,
    BayesianPosterior,
    HazardEstimate,
    MultiTimescaleEstimate,
    SettlementForecast,
    ExecutionEstimate,
    TradeDecision,
    RiskState,
    Direction,
    RegimeState,
    SignalType,
    TickBuffer,
    OrderbookBuffer,
    SignalBuffer,
    RegimeBuffer,
)
from polymarket_bot.data.polymarket_client import PolymarketCLOBClient
from polymarket_bot.data.price_feed import MultiExchangePriceFeed, RESTPriceFallback
from polymarket_bot.infrastructure.config import InfrastructureConfig as InfraConfig
from polymarket_bot.infrastructure.engine import InfrastructureEngine
from polymarket_bot.microstructure.orderbook_analyzer import OrderbookAnalyzer
from polymarket_bot.microstructure.flow_analyzer import FlowAnalyzer
from polymarket_bot.microstructure.liquidity_analyzer import LiquidityAnalyzer
from polymarket_bot.microstructure.queue_model import QueueModel
from polymarket_bot.microstructure.spread_analyzer import SpreadAnalyzer
from polymarket_bot.regime.hmm_regime import HMMRegimeDetector
from polymarket_bot.regime.volatility_engine import VolatilityEngine
from polymarket_bot.signal.continuation_engine import ContinuationEngine
from polymarket_bot.signal.reversal_engine import ReversalEngine
from polymarket_bot.signal.exhaustion_detector import ExhaustionDetector
from polymarket_bot.signal.burst_detector import BurstDetector
from polymarket_bot.signal.entropy_filter import EntropyFilter
from polymarket_bot.inference.kalman_filter import KalmanFilter
from polymarket_bot.inference.bayesian_fusion import BayesianFusion
from polymarket_bot.inference.posterior_calibrator import PosteriorCalibrator
from polymarket_bot.inference.hazard_model import HazardModel
from polymarket_bot.inference.multi_timescale import MultiTimescaleAggregator
from polymarket_bot.inference.settlement_forecast import SettlementForecaster
from polymarket_bot.strategy.observation_window import ObservationWindowStrategy
from polymarket_bot.strategy.technical_momentum import TechnicalMomentumStrategy
from polymarket_bot.execution.execution_engine import ExecutionEngine
from polymarket_bot.decision.decision_engine import DecisionEngine
from polymarket_bot.risk.risk_manager import RiskManager
from polymarket_bot.portfolio import OpenPosition, PositionSnapshot, PositionTracker

logger = logging.getLogger(__name__)


def _bucket_from_slug(slug: str) -> float:
    try:
        parts = slug.split("-")
        return float(parts[-1]) if len(parts) > 2 else 0.0
    except (ValueError, IndexError):
        return 0.0


class TradingBot:
    """
    Main orchestrator for the Polymarket BTC 5-minute directional bot.
    
    Coordinates all subsystems in a continuous processing loop:
    1. Data collection (multi-exchange BTC + Polymarket)
    2. Microstructure analysis
    3. Regime detection
    4. Signal generation
    5. Inference pipeline
    6. Execution modeling
    7. Decision making
    8. Risk management
    9. Order placement
    
    The bot is designed to be robust:
    - Handles missing data gracefully (uses defaults/estimates)
    - Recovers from WebSocket disconnections
    - Respects dry_run mode (no actual orders placed)
    - Circuit breaker stops trading on drawdown/loss limits
    - Cooldown periods after consecutive losses
    """

    def __init__(
        self,
        config: BotConfig,
        initial_capital: float = 1000.0,
        tui: Optional[Any] = None,
    ):
        self.config = config
        self.poly_config = config.polymarket
        self.tui = tui
        
        # ---- Data Layer ----
        self.polymarket_client = PolymarketCLOBClient(self.poly_config, config.market)
        self.price_feed = MultiExchangePriceFeed(config)
        self.rest_fallback = RESTPriceFallback(config)

        # ---- Infrastructure Engine (BTC_POLY) ----
        self._infra_config = InfraConfig.from_env()
        self._infra_engine = InfrastructureEngine(self._infra_config)
        self._infra_ptb = None
        self._infra_roller = None
        self._infra_settlement = None
        self._owns_infra_engine = True
        
        # ---- Buffers ----
        self.tick_buffer = TickBuffer(max_size=10000)
        self.orderbook_buffer = OrderbookBuffer(max_snapshots=100, max_deltas=5000)
        self.signal_buffer = SignalBuffer(max_size=200)
        self.regime_buffer = RegimeBuffer(max_size=500)
        
        # ---- Microstructure ----
        self.orderbook_analyzer = OrderbookAnalyzer(config.microstructure)
        self.flow_analyzer = FlowAnalyzer(config.microstructure)
        self.liquidity_analyzer = LiquidityAnalyzer(config.microstructure)
        self.queue_model = QueueModel(config.microstructure)
        self.spread_analyzer = SpreadAnalyzer(config.microstructure)
        
        # ---- Regime & Volatility ----
        self.hmm_regime = HMMRegimeDetector(config.hmm)
        self.volatility_engine = VolatilityEngine(config.volatility)
        
        # ---- Signals ----
        self.continuation_engine = ContinuationEngine(config.signal)
        self.reversal_engine = ReversalEngine(config.signal)
        self.exhaustion_detector = ExhaustionDetector(config.signal)
        self.burst_detector = BurstDetector(config.signal)
        self.entropy_filter = EntropyFilter(config.signal)
        self.observation_strategy = ObservationWindowStrategy(config)
        self.technical_momentum_strategy = TechnicalMomentumStrategy(config)
        
        # ---- Inference ----
        self.kalman_filter = KalmanFilter(config.kalman)
        self.bayesian_fusion = BayesianFusion(config.bayesian)
        self.posterior_calibrator = PosteriorCalibrator(config.bayesian)
        self.hazard_model = HazardModel()
        self.multi_timescale = MultiTimescaleAggregator(config)
        self.settlement_forecaster = SettlementForecaster(config)
        
        # ---- Execution ----
        self.execution_engine = ExecutionEngine(config)
        
        # ---- Decision ----
        self.decision_engine = DecisionEngine(config)
        
        # ---- Risk ----
        self.risk_manager = RiskManager(config, initial_capital=initial_capital)
        self.position_tracker = PositionTracker()
        
        # ---- State ----
        self._running = False
        self._cycle_count = 0
        self._last_cycle_time = 0.0
        self._cycle_interval = 1.0  # seconds between processing cycles
        self._market_info: Optional[PolymarketMarketInfo] = None
        self._prefetched_market_info: Optional[PolymarketMarketInfo] = None
        self._prefetched_market_slug = ""
        self._prefetched_markets: Dict[str, PolymarketMarketInfo] = {}
        self._rollover_prefetch_task: Optional[asyncio.Task] = None
        self._wallet_maintenance_task: Optional[asyncio.Task] = None
        self._last_rollover_prefetch = 0.0
        self._last_orderbook_warmup_slugs = ""
        self._rollover_prefetch_seconds = max(
            0.0,
            float(getattr(config, "rollover_prefetch_seconds", 90.0) or 90.0),
        )
        self._rollover_prefetch_rounds = max(
            1,
            int(getattr(config, "rollover_prefetch_rounds", 2) or 2),
        )
        self._rollover_orderbook_warmup_seconds = max(
            0.0,
            float(getattr(config, "rollover_orderbook_warmup_seconds", 8.0) or 8.0),
        )
        self._rollover_prefetch_poll_seconds = max(
            0.25,
            float(getattr(config, "rollover_prefetch_poll_seconds", 1.0) or 1.0),
        )
        self._rollover_fetch_timeout_seconds = max(
            0.5,
            float(getattr(config, "rollover_fetch_timeout_seconds", 2.5) or 2.5),
        )
        self._current_market_state: Optional[MarketState] = None
        self._tui_refresh_task: Optional[asyncio.Task] = None
        self._tui_refresh_interval = max(
            0.02,
            float(
                os.getenv(
                    "TUI_REFRESH_INTERVAL_SECONDS",
                    str(getattr(config, "tui_refresh_interval_seconds", 0.05) or 0.05),
                )
            ),
        )
        self._tui_min_render_interval = max(
            0.04,
            float(
                os.getenv(
                    "TUI_MIN_RENDER_INTERVAL_SECONDS",
                    str(getattr(config, "tui_min_render_interval_seconds", 0.08) or 0.08),
                )
            ),
        )
        self._last_tui_render_at = 0.0
        self._last_tui_render_signature: Optional[Tuple[Any, ...]] = None
        self._last_ptb_refresh = 0.0
        self._ptb_refresh_interval = 2.0
        self._ptb_refresh_task: Optional[asyncio.Task] = None
        self._seen_tick_keys = set()
        self._seen_tick_key_order = []
        self._max_seen_tick_keys = 20000
        self._seen_orderbook_delta_keys = set()
        self._seen_orderbook_delta_key_order = []
        self._max_seen_orderbook_delta_keys = 20000
        self._latest_flow_metrics: Optional[FlowMetrics] = None
        self._last_trade_tick: Optional[TickData] = None
        self._last_observation_log_slug = ""
        self._last_observation_log_ready = False
        self._logged_observation_signal_keys = set()
        self._last_market_event_key = ""
        self._logged_market_round_keys = set()
        self._logged_entry_signal_keys = set()
        self._last_heartbeat_log = 0.0
        self._last_tui_error_log = 0.0
        self._heartbeat_interval = max(
            5.0,
            float(getattr(config, "heartbeat_interval_seconds", 15.0) or 15.0),
        )
        self._run_cycle_timeout = max(
            5.0,
            float(getattr(config, "run_cycle_timeout_seconds", 15.0) or 15.0),
        )
        self._use_official_settlement_close = bool(
            getattr(config, "use_official_settlement_close", False)
        )
        self._official_settlement_poll_seconds = max(
            0.25,
            float(getattr(config, "official_settlement_poll_seconds", 1.0) or 1.0),
        )
        self._settlement_tasks: Dict[str, asyncio.Task] = {}
        self._local_settlement_grace_seconds = max(
            0.0,
            float(getattr(config, "local_settlement_grace_seconds", 1.0) or 1.0),
        )
        self._ptb_refresh_timeout = max(
            0.25,
            float(getattr(config, "price_to_beat_refresh_timeout_seconds", 1.5) or 1.5),
        )
        self._round_direction_locks: Dict[str, Dict[str, Any]] = {}
        self._last_position_snapshot: PositionSnapshot = self.position_tracker.snapshot()
        self._trade_history_file = getattr(config, "trade_history_file", "")
        self._trade_history_events: List[Dict[str, Any]] = []
        self._max_trade_history_events = 250
        self._wallet_maintenance_enabled = bool(
            getattr(config, "wallet_maintenance_enabled", True)
        )
        self._wallet_maintenance_interval = max(
            5.0,
            float(getattr(config, "wallet_maintenance_interval_seconds", 30.0) or 30.0),
        )
        self._auto_update_allowance_enabled = bool(
            getattr(config, "auto_update_allowance_enabled", False)
        )
        self._auto_claim_enabled = bool(getattr(config, "auto_claim_enabled", False))
        self._claim_candidates: Dict[str, Dict[str, Any]] = {}
        self._claimed_condition_ids = set()
        self._last_wallet_status: Dict[str, Any] = {}
        
        # ---- Performance tracking ----
        self._start_time = time.time()
        self._total_cycles = 0
        self._total_order_attempts = 0
        self._total_trades = 0
        self._total_suppressed = 0

    def attach_infrastructure_engine(
        self,
        engine: InfrastructureEngine,
        *,
        owns_engine: bool = False,
    ) -> None:
        """Attach the canonical infrastructure engine used by lifecycle orchestration."""
        self._infra_engine = engine
        self._infra_config = engine.config
        self._infra_ptb = engine.ptb_lifecycle
        self._infra_roller = engine.discovery
        self._infra_settlement = engine.settlement
        self._owns_infra_engine = owns_engine

        attach_client = getattr(self.polymarket_client, "attach_infrastructure_engine", None)
        if attach_client is not None:
            attach_client(engine)
        attach_price_feed = getattr(self.price_feed, "attach_infrastructure_engine", None)
        if attach_price_feed is not None:
            attach_price_feed(engine)

    async def initialize(self) -> None:
        """
        Initialize the bot: discover markets, connect to data feeds.
        
        Steps:
        1. Load environment variables
        2. Discover Polymarket BTC 5-minute markets
        3. Connect to multi-exchange BTC WebSockets
        4. Warm up processing pipeline (initial data collection)
        """
        logger.info("Initializing Polymarket BTC 5-minute directional bot...")

        # Start or reuse BTC_POLY infrastructure before adapter startup so
        # client/feed wrappers do not create duplicate WebSocket stacks.
        try:
            if self._infra_engine.running:
                logger.info("Reusing BTC_POLY infrastructure engine")
            else:
                await self._infra_engine.start()
                logger.info("BTC_POLY infrastructure engine started")
            self._infra_ptb = self._infra_engine.ptb_lifecycle
            self._infra_roller = self._infra_engine.discovery
            self._infra_settlement = self._infra_engine.settlement
            self.polymarket_client.attach_infrastructure_engine(self._infra_engine)
            self.price_feed.attach_infrastructure_engine(self._infra_engine)
        except Exception as e:
            logger.warning(f"Infrastructure engine start failed: {e}.")
        self._update_tui(force=True)

        # Start Polymarket public adapter and official SDK wrapper before any CLOB calls.
        try:
            await self.polymarket_client.start()
        except Exception as e:
            logger.warning(f"Polymarket client start failed: {e}. Public reads may be unavailable.")
        self._update_tui(force=True)
        
        # Discover Polymarket markets
        try:
            self._market_info = await self.polymarket_client.discover_btc_5min_market()
            if self._market_info:
                ptb_status = "none"
                if self._market_info.price_to_beat_snapshot:
                    ptb_status = (
                        "valid"
                        if self._market_info.price_to_beat_snapshot.is_valid
                        else f"invalid:{self._market_info.price_to_beat_snapshot.reason}"
                    )
                logger.info(
                    f"Discovered market: {self._market_info.question} "
                    f"slug={self._market_info.slug} "
                    f"price_to_beat={self._market_info.price_to_beat} "
                    f"ptb_status={ptb_status}"
                )
                try:
                    await self.polymarket_client.subscribe_orderbook_stream(self._market_info)
                except Exception as e:
                    logger.warning(f"Polymarket orderbook WS subscribe failed: {e}.")
            else:
                logger.warning("Could not discover BTC 5-minute market. Using defaults.")
        except Exception as e:
            logger.warning(f"Market discovery failed: {e}. Using defaults.")
        self._update_tui(force=True)
        
        # Connect to multi-exchange BTC WebSockets
        try:
            await self.price_feed.connect()
            logger.info("Connected to multi-exchange BTC WebSockets.")
        except Exception as e:
            logger.warning(f"Multi-exchange BTC WebSocket connection failed: {e}. Will use REST fallback.")
        self._update_tui(force=True)

        # REST fallback is intentionally not started in the hot path. The bot
        # should trade from WebSocket data and avoid blocking startup on REST.
        
        # Short non-blocking warm-up. WebSocket tasks continue filling buffers
        # in the background; no REST backfill is used here.
        logger.info("Warming up processing pipeline...")
        await asyncio.sleep(0.25)
        await self._collect_initial_data()
        self._update_tui(force=True)
        
        self._running = True
        self._start_tui_refresh_task()
        self._start_rollover_prefetch_task()
        self._start_wallet_maintenance_task()
        logger.info("Bot initialized and ready to run.")

    async def _refresh_price_to_beat(self) -> None:
        if not self._market_info or not self._market_info.slug:
            return
        if (
            self._market_info.price_to_beat_snapshot is not None
            and self._market_info.price_to_beat_snapshot.is_valid
            and self._market_info.price_to_beat is not None
        ):
            return

        now = time.time()
        if now - self._last_ptb_refresh < self._ptb_refresh_interval:
            return
        self._last_ptb_refresh = now

        try:
            snapshot = await asyncio.wait_for(
                self.polymarket_client.refresh_price_to_beat(self._market_info.slug),
                timeout=self._ptb_refresh_timeout,
            )
        except asyncio.TimeoutError:
            logger.debug(
                "Price-to-beat refresh timed out: slug=%s timeout=%.1fs",
                self._market_info.slug,
                self._ptb_refresh_timeout,
            )
            return
        if snapshot:
            self._market_info.price_to_beat = snapshot.price if snapshot.is_valid else None
            self._market_info.price_to_beat_snapshot = snapshot
            if snapshot.is_valid:
                self._log_market_round_event(
                    market_slug=self._market_info.slug,
                    price_to_beat=self._market_info.price_to_beat,
                    time_to_settlement=self._time_to_market_settlement(),
                    reason="ptb_ready",
                )
            if not snapshot.is_valid:
                logger.debug(
                    "Price-to-beat rejected: slug=%s reason=%s",
                    self._market_info.slug,
                    snapshot.reason,
                )

    def _start_price_to_beat_refresh_task_if_needed(self) -> None:
        if not self._market_info or not self._market_info.slug:
            return
        if (
            self._market_info.price_to_beat_snapshot is not None
            and self._market_info.price_to_beat_snapshot.is_valid
            and self._market_info.price_to_beat is not None
        ):
            return

        # ── Infrastructure PTB: O(1) immutable lookup ──────────────────
        # If the infrastructure PTB engine has the PTB cached, use it
        # directly — NO REST call, NO task spawn. This eliminates the
        # problematic PTB refresh REST polling from the hot path.
        infra_ptb = getattr(self, "_infra_ptb", None)
        if infra_ptb is not None:
            try:
                cached = infra_ptb.get_ptb(self._market_info.slug)
                if cached is not None and cached.is_valid:
                    self._market_info.price_to_beat = cached.price
                    self._market_info.price_to_beat_snapshot = PriceToBeatSnapshot(
                        price=cached.price,
                        source=f"infra:{cached.source.name}",
                        is_valid=True,
                        reason="infra_cache",
                    )
                    self._log_market_round_event(
                        market_slug=self._market_info.slug,
                        price_to_beat=self._market_info.price_to_beat,
                        time_to_settlement=self._time_to_market_settlement(),
                        reason="ptb_ready",
                    )
                    return
            except Exception:
                pass

        task = self._ptb_refresh_task
        if task is not None and not task.done():
            return
        self._ptb_refresh_task = asyncio.create_task(
            self._refresh_price_to_beat(),
            name=f"polymarket-ptb-refresh-{self._market_info.slug}",
        )

    async def _get_polymarket_orderbooks(
        self,
    ) -> Tuple[Optional[PolymarketOrderbook], Optional[PolymarketOrderbook]]:
        """Read UP/DOWN books from the WebSocket cache only."""
        if not self._market_info:
            return None, None

        return (
            self.polymarket_client.get_cached_orderbook(self._market_info.token_id_up),
            self.polymarket_client.get_cached_orderbook(self._market_info.token_id_down),
        )

    def _time_to_market_settlement(self) -> float:
        if not self._market_info or not self._market_info.slug:
            return self.config.market.settlement_interval_seconds

        window = self._market_window_from_slug(self._market_info.slug)
        if window is not None:
            _, window_end = window
            return float(max(0.0, window_end - time.time()))

        return self.config.market.settlement_interval_seconds

    def _market_interval_seconds(self) -> int:
        try:
            interval = int(float(self.config.market.settlement_interval_seconds or 300))
        except (TypeError, ValueError):
            interval = 300
        return max(1, interval)

    def _market_window_from_slug(self, slug: str) -> Optional[Tuple[int, int]]:
        prefix = "btc-updown-5m-"
        if not slug or not slug.startswith(prefix):
            return None
        try:
            window_start = int(slug[len(prefix):])
        except ValueError:
            return None
        if window_start <= 0:
            return None
        return window_start, window_start + self._market_interval_seconds()

    def _expected_active_btc_5m_slug(self, now: Optional[float] = None) -> str:
        interval = self._market_interval_seconds()
        timestamp = int(now if now is not None else time.time())
        window_start = (timestamp // interval) * interval
        return f"btc-updown-5m-{window_start}"

    def _next_btc_5m_slug(self, slug: Optional[str] = None) -> str:
        interval = self._market_interval_seconds()
        window = self._market_window_from_slug(slug or "")
        if window is not None:
            window_start, _ = window
            return f"btc-updown-5m-{window_start + interval}"

        timestamp = int(time.time())
        window_start = ((timestamp // interval) + 1) * interval
        return f"btc-updown-5m-{window_start}"

    def _future_btc_5m_slugs(
        self,
        *,
        count: Optional[int] = None,
        base_slug: Optional[str] = None,
    ) -> List[str]:
        rounds = max(1, int(count or self._rollover_prefetch_rounds))
        interval = self._market_interval_seconds()
        window = self._market_window_from_slug(base_slug or (self._market_info.slug if self._market_info else ""))
        if window is not None:
            start = window[0]
        else:
            timestamp = int(time.time())
            start = (timestamp // interval) * interval
        return [f"btc-updown-5m-{start + (index + 1) * interval}" for index in range(rounds)]

    def _infra_current_market(self) -> Any:
        state = getattr(getattr(self, "_infra_engine", None), "state", None)
        return getattr(state, "current_market", None)

    def _infra_market_to_polymarket_info(self, market: Any) -> Optional[PolymarketMarketInfo]:
        if market is None or not getattr(market, "slug", ""):
            return None
        if not getattr(market, "up_token_id", "") or not getattr(market, "down_token_id", ""):
            return None
        return PolymarketMarketInfo(
            condition_id=getattr(market, "condition_id", ""),
            question=getattr(market, "question", "") or "BTC Up/Down 5m",
            tokens={
                "UP": getattr(market, "up_token_id", ""),
                "DOWN": getattr(market, "down_token_id", ""),
            },
            end_date_iso=str(getattr(market, "end_time", "")),
            slug=getattr(market, "slug", ""),
            active=bool(getattr(market, "active", True)),
            closed=bool(getattr(market, "is_expired", False)),
            minimum_order_size=float(getattr(market, "min_order_size", 5.0) or 5.0),
            minimum_tick_size=float(getattr(market, "tick_size", 0.01) or 0.01),
            price_to_beat=(
                float(getattr(market, "price_to_beat", 0.0))
                if float(getattr(market, "price_to_beat", 0.0) or 0.0) > 0.0
                else None
            ),
        )

    def _orderbook_stream_market_infos(self) -> List[PolymarketMarketInfo]:
        market_infos: List[PolymarketMarketInfo] = []
        if self._market_info is not None:
            market_infos.append(self._market_info)
        now = time.time()
        for slug in self._future_btc_5m_slugs(count=self._rollover_prefetch_rounds):
            market_info = self._prefetched_markets.get(slug)
            if market_info is None:
                continue
            window = self._market_window_from_slug(slug)
            if window is not None and window[1] < now:
                continue
            market_infos.append(market_info)
        return market_infos

    def _next_orderbook_stream_market_infos(self) -> List[PolymarketMarketInfo]:
        market_infos: List[PolymarketMarketInfo] = []
        if self._market_info is not None:
            market_infos.append(self._market_info)
            next_slug = self._next_btc_5m_slug(self._market_info.slug)
            next_market = self._prefetched_markets.get(next_slug)
            if next_market is not None:
                market_infos.append(next_market)
        return market_infos

    def _reset_round_runtime_state(self) -> None:
        self._last_observation_log_slug = ""
        self._last_observation_log_ready = False
        self._logged_observation_signal_keys.clear()
        self._logged_entry_signal_keys.clear()
        self._last_ptb_refresh = 0.0
        try:
            self.bayesian_fusion.reset()
        except Exception:
            pass
        try:
            self.posterior_calibrator.reset()
        except Exception:
            pass

    def _invalidate_current_market_state_for_round(
        self,
        market_info: PolymarketMarketInfo,
    ) -> None:
        state = self._current_market_state
        if state is None:
            return

        state.timestamp = time.time()
        state.market_slug = market_info.slug
        state.market_question = market_info.question
        state.market_end_iso = market_info.end_date_iso
        state.price_to_beat = market_info.price_to_beat
        state.polymarket_orderbook_up = None
        state.polymarket_orderbook_down = None
        state.observation_signal = None
        state.technical_signal = None
        state.settlement_forecast = None
        state.execution_estimate = None
        state.trade_decision = None

    def _start_rollover_prefetch_task(self) -> None:
        task = self._rollover_prefetch_task
        if task is not None and not task.done():
            return
        self._rollover_prefetch_task = asyncio.create_task(
            self._rollover_prefetch_loop(),
            name="polymarket-rollover-prefetch",
        )

    def _start_wallet_maintenance_task(self) -> None:
        if not self._wallet_maintenance_enabled:
            return
        task = self._wallet_maintenance_task
        if task is not None and not task.done():
            return
        self._wallet_maintenance_task = asyncio.create_task(
            self._wallet_maintenance_loop(),
            name="polymarket-wallet-maintenance",
        )

    def _start_tui_refresh_task(self) -> None:
        if self.tui is None:
            return
        task = self._tui_refresh_task
        if task is not None and not task.done():
            return
        self._tui_refresh_task = asyncio.create_task(
            self._tui_refresh_loop(),
            name="polymarket-tui-refresh",
        )

    async def _tui_refresh_loop(self) -> None:
        while self._running:
            try:
                self._update_tui()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                now = time.time()
                if now - self._last_tui_error_log >= 15.0:
                    self._last_tui_error_log = now
                    logger.warning("TUI refresh loop failed: %s", exc, exc_info=True)
            await asyncio.sleep(self._tui_refresh_interval)

    def _book_signature(self, book: Optional[PolymarketOrderbook]) -> Tuple[Any, ...]:
        if book is None:
            return (None,)
        bid = book.bids[0] if book.bids else None
        ask = book.asks[0] if book.asks else None
        return (
            round(bid.price, 4) if bid else None,
            round(bid.size, 4) if bid else None,
            round(ask.price, 4) if ask else None,
            round(ask.size, 4) if ask else None,
            len(book.bids),
            len(book.asks),
        )

    def _tui_snapshot_signature(self, state: Optional[MarketState]) -> Tuple[Any, ...]:
        market = self._market_info
        live_price = self.price_feed.get_current_settlement_price()
        live_source = self.price_feed.get_current_settlement_price_source()
        live_up_book = None
        live_down_book = None
        display_tau = None
        if market is not None and market.slug:
            try:
                display_tau = self._time_to_market_settlement()
            except Exception:
                display_tau = None
        elif state is not None and state.market_slug:
            window = self._market_window_from_slug(state.market_slug)
            if window is not None:
                _, window_end = window
                display_tau = max(0.0, float(window_end) - time.time())
        tau_bucket = (
            int(display_tau)
            if display_tau is not None and np.isfinite(float(display_tau))
            else None
        )
        if market is not None:
            try:
                live_up_book = self.polymarket_client.get_cached_orderbook(market.token_id_up)
                live_down_book = self.polymarket_client.get_cached_orderbook(market.token_id_down)
            except Exception:
                live_up_book = None
                live_down_book = None
        if state is None:
            return (
                "empty",
                market.slug if market else "",
                tau_bucket,
                round(float(live_price), 2) if live_price is not None else None,
                live_source,
                self._book_signature(live_up_book),
                self._book_signature(live_down_book),
            )
        decision = state.trade_decision
        forecast = (
            decision.settlement_forecast
            if decision is not None
            else state.settlement_forecast
        )
        direction = None
        if decision is not None and decision.direction is not None:
            direction = decision.direction.value
        round_direction = None
        if forecast is not None and forecast.round_direction is not None:
            round_direction = forecast.round_direction.value
        return (
            state.market_slug,
            round(float(live_price), 2) if live_price is not None else round(float(state.btc_price), 2),
            live_source or state.price_source,
            tau_bucket,
            round(float(state.price_to_beat), 2) if state.price_to_beat else None,
            round(float((state.settlement_forecast.price_to_beat if state.settlement_forecast and state.settlement_forecast.price_to_beat else state.price_to_beat) or 0.0), 2)
            if (state.settlement_forecast and state.settlement_forecast.price_to_beat) or state.price_to_beat
            else None,
            self._book_signature(live_up_book or state.polymarket_orderbook_up),
            self._book_signature(live_down_book or state.polymarket_orderbook_down),
            (
                bool(decision.should_trade),
                direction,
                round(float(decision.probability), 4),
                round(float(decision.market_price), 4),
                round(float(decision.edge_bps), 1),
                str(decision.reason)[:96],
            ) if decision is not None else None,
            (
                round(float(forecast.p_up_settlement), 4),
                round(float(forecast.p_down_settlement), 4),
                bool(forecast.observation_ready),
                bool(forecast.observation_valid),
                bool(forecast.round_direction_locked),
                round_direction,
            ) if forecast is not None else None,
            self._last_position_snapshot.open_count,
            round(float(self._last_position_snapshot.unrealized_pnl), 4),
        )

    async def _wallet_maintenance_loop(self) -> None:
        while self._running:
            try:
                await self._run_wallet_maintenance_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Wallet maintenance failed: %s", exc)
            await asyncio.sleep(self._wallet_maintenance_interval)

    async def _run_wallet_maintenance_once(self) -> None:
        token_ids = []
        for position in self.position_tracker.open_positions():
            if position.token_id:
                token_ids.append(position.token_id)
        for candidate in self._claim_candidates.values():
            for token_id in candidate.get("token_ids", []):
                if token_id:
                    token_ids.append(str(token_id))

        condition_ids = [
            condition_id
            for condition_id in self._claim_candidates
            if condition_id and condition_id not in self._claimed_condition_ids
        ]
        status = await self.polymarket_client.get_wallet_maintenance_status(
            token_ids=sorted(set(token_ids)),
            condition_ids=condition_ids,
        )
        self._last_wallet_status = status
        if not status.get("ready"):
            logger.debug("WALLET STATUS: not_ready reason=%s", status.get("reason"))
            return

        logger.debug(
            "WALLET STATUS: pusd=%.2f allowance=%.2f adapter_approved=%s claims_pending=%s",
            float(status.get("pusd_balance_raw", 0)) / 1_000_000.0,
            float(status.get("ctf_exchange_allowance_raw", 0)) / 1_000_000.0,
            status.get("ctf_adapter_approved"),
            len(condition_ids),
        )

        if self._auto_update_allowance_enabled and not self.poly_config.dry_run:
            try:
                await self.polymarket_client.update_balance_allowance(asset_type="COLLATERAL")
            except Exception as exc:
                logger.debug("CLOB collateral allowance update failed: %s", exc)
            if not status.get("ctf_adapter_approved"):
                result = await self.polymarket_client.approve_ctf_adapter()
                logger.info("CTF ADAPTER APPROVAL: %s", result)

        if not self._auto_claim_enabled:
            return

        for condition_id in condition_ids:
            condition = (status.get("conditions") or {}).get(condition_id) or {}
            if not condition.get("resolved"):
                continue
            candidate = self._claim_candidates.get(condition_id) or {}
            balances = status.get("tokens") or {}
            has_balance = any(
                isinstance(balances.get(str(token_id)), int) and balances.get(str(token_id), 0) > 0
                for token_id in candidate.get("token_ids", [])
            )
            if not has_balance:
                self._claimed_condition_ids.add(condition_id)
                continue
            result = await self.polymarket_client.redeem_positions(condition_id)
            logger.info("AUTO CLAIM: condition_id=%s result=%s", condition_id, result)
            if result.get("status") in {"submitted", "dry_run"}:
                self._claimed_condition_ids.add(condition_id)

    async def _rollover_prefetch_loop(self) -> None:
        while self._running:
            try:
                await self._prefetch_next_market_if_needed()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("Rollover prefetch loop error: %s", exc)
            await asyncio.sleep(self._rollover_prefetch_poll_seconds)

    async def _activate_market_info(
        self,
        market_info: PolymarketMarketInfo,
        *,
        reason: str,
    ) -> bool:
        market_changed = (
            not self._market_info
            or market_info.slug != self._market_info.slug
            or market_info.token_id_up != self._market_info.token_id_up
            or market_info.token_id_down != self._market_info.token_id_down
        )
        self._market_info = market_info
        self.polymarket_client.set_active_market_info(market_info)

        if not market_changed:
            return False

        self._reset_round_runtime_state()
        self._invalidate_current_market_state_for_round(market_info)
        self.polymarket_client.clear_ptb_cache_for_round(market_info.slug)
        self._log_market_round_event(
            market_slug=market_info.slug,
            price_to_beat=market_info.price_to_beat,
            time_to_settlement=self._time_to_market_settlement(),
            reason=reason,
        )
        self._update_tui()
        try:
            await self.polymarket_client.subscribe_orderbook_stream(market_info)
            self._last_orderbook_warmup_slugs = market_info.slug
        except Exception as exc:
            logger.warning("Polymarket orderbook WS resubscribe failed: %s.", exc)
        return True

    async def _prefetch_next_market_if_needed(self) -> None:
        if not self._market_info or not self._market_info.slug:
            return

        tau = self._time_to_market_settlement()
        if tau > self._rollover_prefetch_seconds:
            return

        now = time.time()
        if now - self._last_rollover_prefetch < self._rollover_prefetch_poll_seconds:
            return
        self._last_rollover_prefetch = now

        for slug, market_info in list(self._prefetched_markets.items()):
            window = self._market_window_from_slug(slug)
            if window is not None and window[1] < now:
                self._prefetched_markets.pop(slug, None)
                if self._prefetched_market_slug == slug:
                    self._prefetched_market_info = None
                    self._prefetched_market_slug = ""

        fetched_any = False
        for slug in self._future_btc_5m_slugs(count=self._rollover_prefetch_rounds):
            if slug in self._prefetched_markets:
                continue
            try:
                market_info = await asyncio.wait_for(
                    self.polymarket_client.fetch_btc_5min_market_by_slug(
                        slug,
                        activate=False,
                        fast=True,
                    ),
                    timeout=self._rollover_fetch_timeout_seconds,
                )
            except asyncio.TimeoutError:
                logger.debug(
                    "Rollover prefetch timed out: slug=%s timeout=%.1fs",
                    slug,
                    self._rollover_fetch_timeout_seconds,
                )
                continue
            except Exception as exc:
                logger.debug("Rollover prefetch failed: slug=%s error=%s", slug, exc)
                continue

            if market_info is None:
                continue

            self._prefetched_markets[market_info.slug] = market_info
            fetched_any = True
            if not self._prefetched_market_info or market_info.slug < self._prefetched_market_info.slug:
                self._prefetched_market_info = market_info
                self._prefetched_market_slug = market_info.slug
            logger.debug(
                "Rollover prefetch ready: slug=%s tau=%.0fs",
                market_info.slug,
                tau,
            )

        # Keep the active orderbook WebSocket stable. Restarting it during a
        # live round clears the current book cache and shows up as UI lag.
        if fetched_any:
            logger.debug(
                "Prefetched future market metadata: slugs=%s",
                ",".join(sorted(self._prefetched_markets)),
            )

        if (
            tau <= self._rollover_orderbook_warmup_seconds
            and self._market_info is not None
        ):
            warmup_infos = self._next_orderbook_stream_market_infos()
            warmup_slugs = ",".join(market.slug for market in warmup_infos if market.slug)
            if len(warmup_infos) > 1 and warmup_slugs != self._last_orderbook_warmup_slugs:
                try:
                    await self.polymarket_client.subscribe_orderbook_streams(warmup_infos)
                    self._last_orderbook_warmup_slugs = warmup_slugs
                    logger.info("ORDERBOOK PREWARM: slugs=%s tau=%.0fs", warmup_slugs, tau)
                except Exception as exc:
                    logger.debug("Orderbook prewarm failed: slugs=%s error=%s", warmup_slugs, exc)

    async def _rollover_market_if_needed(self) -> Optional[PolymarketMarketInfo]:
        infra_market = self._infra_current_market()
        infra_market_info = self._infra_market_to_polymarket_info(infra_market)
        if not self._market_info or not self._market_info.slug:
            # ── Infrastructure roller: deterministic active slug ──────
            if infra_market_info is not None:
                if infra_market_info.slug in self._prefetched_markets:
                    market_info = self._prefetched_markets[infra_market_info.slug]
                else:
                    market_info = infra_market_info
                await self._activate_market_info(market_info, reason="infra_state")
                return market_info
            market_info = await self.polymarket_client.discover_btc_5min_market()
            if market_info:
                await self._activate_market_info(market_info, reason="discovery")
            return market_info

        current_slug = self._market_info.slug
        # ── Infrastructure roller: deterministic expected slug ───────
        # The infrastructure roller computes the expected active slug
        # deterministically (no REST). Prefer it over the local method
        # when available, but both produce identical results.
        expected_slug = infra_market_info.slug if infra_market_info is not None else self._expected_active_btc_5m_slug()

        if current_slug == expected_slug:
            return None

        current_window = self._market_window_from_slug(current_slug)
        expected_window = self._market_window_from_slug(expected_slug)
        if (
            current_window is not None
            and expected_window is not None
            and expected_window[0] < current_window[0]
        ):
            return None

        market_info = None
        reason = "rollover_slug"
        slug_fetch_timed_out = False

        # ── Infrastructure roller: preloaded round ────────────────────
        # The background market scanner pre-enriches rounds with
        # condition_id and token IDs. Check the roller's cache first
        # to avoid REST calls in the hot path.
        roller = getattr(self, "_infra_roller", None)
        if roller is not None:
            try:
                cached_round = roller.get_round(expected_slug)
                if cached_round is not None and cached_round.condition_id:
                    market_info = PolymarketMarketInfo(
                        condition_id=cached_round.condition_id,
                        question="BTC Up/Down 5m",
                        tokens={
                            "UP": cached_round.up_token_id,
                            "DOWN": cached_round.down_token_id,
                        },
                        end_date_iso="",
                        slug=cached_round.slug,
                        active=True,
                        closed=False,
                        minimum_order_size=cached_round.min_order_size,
                        minimum_tick_size=cached_round.tick_size,
                    )
                    reason = "roller_cache"
            except Exception:
                pass

        if market_info is None:
            market_info = self._prefetched_markets.get(expected_slug)
        if market_info is not None:
            reason = "rollover_prefetch"
        else:
            try:
                market_info = await asyncio.wait_for(
                    self.polymarket_client.fetch_btc_5min_market_by_slug(
                        expected_slug,
                        activate=False,
                        fast=True,
                    ),
                    timeout=self._rollover_fetch_timeout_seconds,
                )
            except asyncio.TimeoutError:
                slug_fetch_timed_out = True
                logger.debug(
                    "Rollover slug fetch timed out: slug=%s timeout=%.1fs",
                    expected_slug,
                    self._rollover_fetch_timeout_seconds,
                )
            except Exception as exc:
                logger.debug("Rollover slug fetch failed: slug=%s error=%s", expected_slug, exc)

        if market_info is None and not slug_fetch_timed_out:
            try:
                market_info = await asyncio.wait_for(
                    self.polymarket_client.discover_btc_5min_market(),
                    timeout=self._rollover_fetch_timeout_seconds,
                )
                reason = "rollover"
            except asyncio.TimeoutError:
                logger.debug(
                    "Rollover discovery timed out: expected=%s timeout=%.1fs",
                    expected_slug,
                    self._rollover_fetch_timeout_seconds,
                )
            except Exception as exc:
                logger.debug("Rollover discovery failed: expected=%s error=%s", expected_slug, exc)
                return None

        if market_info is None:
            return None

        changed = await self._activate_market_info(market_info, reason=reason)
        if self._prefetched_market_slug == market_info.slug:
            self._prefetched_market_info = None
            self._prefetched_market_slug = ""
        self._prefetched_markets.pop(market_info.slug, None)
        return market_info if changed else None

    @staticmethod
    def _truthy_order_value(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "ok", "success"}
        return bool(value)

    @staticmethod
    def _order_result_sources(order_result: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(order_result, dict):
            return []
        sources = [order_result]
        raw = order_result.get("raw")
        if isinstance(raw, dict):
            sources.append(raw)
        for source in list(sources):
            for nested_key in ("data", "order", "result", "raw"):
                nested = source.get(nested_key)
                if isinstance(nested, dict):
                    sources.append(nested)
        return sources

    @classmethod
    def _order_result_value(
        cls,
        order_result: Optional[Dict[str, Any]],
        keys: Tuple[str, ...],
    ) -> Any:
        for source in cls._order_result_sources(order_result):
            for key in keys:
                if key in source and source[key] not in (None, ""):
                    return source[key]
        return None

    @classmethod
    def _order_result_float(
        cls,
        order_result: Optional[Dict[str, Any]],
        keys: Tuple[str, ...],
    ) -> Optional[float]:
        value = cls._order_result_value(order_result, keys)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None
        return parsed if np.isfinite(parsed) else None

    @classmethod
    def _order_result_status(cls, order_result: Optional[Dict[str, Any]]) -> str:
        statuses = []
        for source in cls._order_result_sources(order_result):
            for key in ("status", "state", "orderStatus", "order_status"):
                value = source.get(key)
                if value not in (None, ""):
                    statuses.append(str(value).strip().lower())

        terminal_statuses = {
            "dry_run",
            "matched",
            "filled",
            "complete",
            "completed",
            "done",
            "executed",
            "cancelled",
            "canceled",
            "expired",
            "failed",
            "rejected",
            "unfilled",
            "unmatched",
        }
        for status in statuses:
            if status in terminal_statuses:
                return status
        return statuses[0] if statuses else ""

    @classmethod
    def _order_result_order_id(cls, order_result: Optional[Dict[str, Any]]) -> str:
        value = cls._order_result_value(
            order_result,
            ("order_id", "orderID", "orderId", "id"),
        )
        return str(value or "")

    @classmethod
    def _order_result_indicates_fill(cls, order_result: Optional[Dict[str, Any]]) -> bool:
        status = cls._order_result_status(order_result)
        if status == "dry_run":
            return True

        rejected_statuses = {
            "cancelled",
            "canceled",
            "expired",
            "failed",
            "rejected",
            "unfilled",
            "unmatched",
        }
        if status in rejected_statuses:
            return False

        filled_statuses = {
            "matched",
            "filled",
            "complete",
            "completed",
            "done",
            "executed",
        }
        if status in filled_statuses:
            return True

        filled_shares = cls._order_result_float(
            order_result,
            (
                "filled_size",
                "filledSize",
                "matched_size",
                "matchedSize",
                "size_matched",
                "sizeMatched",
            ),
        )
        if filled_shares is not None and filled_shares > 0.0:
            return True

        tx_hashes = cls._order_result_value(
            order_result,
            ("transactionHashes", "transactionsHashes", "transaction_hashes"),
        )
        success = cls._truthy_order_value(
            cls._order_result_value(order_result, ("success", "ok"))
        )
        return bool(success and tx_hashes)

    @classmethod
    def _filled_order_details(
        cls,
        order_result: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Optional[float]]]:
        if not cls._order_result_indicates_fill(order_result):
            return None
        return {
            "filled_price": cls._order_result_float(
                order_result,
                (
                    "filled_price",
                    "filledPrice",
                    "average_price",
                    "averagePrice",
                    "avg_price",
                    "avgPrice",
                    "price",
                ),
            ),
            "filled_shares": cls._order_result_float(
                order_result,
                (
                    "filled_shares",
                    "filledShares",
                    "filled_size",
                    "filledSize",
                    "matched_size",
                    "matchedSize",
                    "share_size",
                    "shareSize",
                ),
            ),
            "filled_notional": cls._order_result_float(
                order_result,
                (
                    "filled_notional",
                    "filledNotional",
                    "notional_size",
                    "notionalSize",
                    "cost",
                ),
            ),
            "filled_fee": cls._order_result_float(
                order_result,
                (
                    "fee",
                    "fees",
                    "platform_fee",
                    "platformFee",
                    "trading_fee",
                    "tradingFee",
                    "entry_fee",
                    "entryFee",
                ),
            ),
        }

    @staticmethod
    def _estimate_entry_fee(
        *,
        market_info: Optional[PolymarketMarketInfo],
        entry_price: Optional[float],
        shares: Optional[float],
        explicit_fee: Optional[float],
        notional: Optional[float],
    ) -> float:
        return estimate_entry_fee_usdc(
            market_info=market_info,
            entry_price=entry_price,
            shares=shares,
            explicit_fee=explicit_fee,
            notional=notional,
        )

    async def _confirm_order_fill_if_needed(
        self,
        order_result: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if self.poly_config.dry_run or self._order_result_indicates_fill(order_result):
            return order_result

        order_id = self._order_result_order_id(order_result)
        if not order_id:
            return order_result

        for _ in range(2):
            await asyncio.sleep(0.5)
            try:
                latest = await self.polymarket_client.get_order(order_id)
            except Exception as exc:
                logger.debug("Order status refresh failed: order_id=%s error=%s", order_id, exc)
                return order_result
            if not latest:
                continue
            merged = dict(order_result or {})
            merged["raw"] = latest
            for status_key in ("status", "state", "orderStatus", "order_status"):
                if status_key in latest and latest[status_key] not in (None, ""):
                    merged[status_key] = latest[status_key]
            if self._order_result_indicates_fill(merged):
                return merged
            order_result = merged
        return order_result

    def _apply_position_snapshot(self, snapshot: PositionSnapshot) -> None:
        self._last_position_snapshot = snapshot
        self.risk_manager.update_open_position_state(
            current_position_size=snapshot.total_cost,
            current_position_direction=snapshot.current_direction,
            unrealized_pnl=snapshot.unrealized_pnl,
        )

    @staticmethod
    def _iso_timestamp(timestamp: Optional[float]) -> Optional[str]:
        if timestamp is None or timestamp <= 0.0:
            return None
        return datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()

    @staticmethod
    def _enum_text(value: Any) -> Any:
        if hasattr(value, "name"):
            return value.name
        if hasattr(value, "value"):
            return value.value
        return value

    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): cls._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._json_safe(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if hasattr(value, "name") or hasattr(value, "value"):
            return cls._enum_text(value)
        return value

    def _append_trade_history_event(self, event: Dict[str, Any]) -> None:
        payload = self._json_safe(event)
        self._trade_history_events.append(payload)
        if len(self._trade_history_events) > self._max_trade_history_events:
            self._trade_history_events = self._trade_history_events[-self._max_trade_history_events:]

        line = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        logger.debug("TRADE_HISTORY %s", line)

        if not self._trade_history_file:
            return

        try:
            directory = os.path.dirname(self._trade_history_file)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(self._trade_history_file, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception as exc:
            logger.debug("Trade history file append failed: %s", exc)

    def _log_market_round_event(
        self,
        *,
        market_slug: str,
        price_to_beat: Optional[float],
        time_to_settlement: Optional[float] = None,
        reason: str = "active",
    ) -> None:
        if not market_slug:
            return
        if reason == "active":
            return
        has_ptb = price_to_beat is not None and price_to_beat > 0.0
        event_reason = reason if has_ptb else f"{reason}_pending_ptb"

        if reason == "ptb_ready":
            if not has_ptb:
                return
            event_key = f"{market_slug}:ptb_ready"
        else:
            event_key = f"{market_slug}:{event_reason}"

        if event_key in self._logged_market_round_keys:
            return
        self._logged_market_round_keys.add(event_key)
        self._last_market_event_key = event_key
        logger.info(
            "MARKET ROUND: ptb=%s tau=%s reason=%s",
            f"{price_to_beat:.2f}" if price_to_beat is not None else "N/A",
            f"{time_to_settlement:.0f}s" if time_to_settlement is not None else "N/A",
            event_reason,
        )

    @staticmethod
    def _primary_decision_gate(reason: str) -> str:
        for part in str(reason or "").split("|"):
            gate = part.strip()
            if gate:
                return gate
        return "OK"

    def _log_entry_signal_event(
        self,
        *,
        market_slug: str,
        trade_decision: Optional[TradeDecision],
    ) -> None:
        if not market_slug or trade_decision is None:
            return
        forecast = trade_decision.settlement_forecast
        if (
            forecast is None
            or not forecast.observation_ready
            or not forecast.observation_valid
        ):
            return

        action = "EXECUTE" if trade_decision.should_trade else "NO_TRADE"
        observation_state = "valid"
        direction = (
            trade_decision.direction.value
            if trade_decision.direction is not None
            else "N/A"
        )
        primary_gate = self._primary_decision_gate(trade_decision.reason)
        event_key = f"{market_slug}:entry:{action}"
        if event_key in self._logged_entry_signal_keys:
            return
        self._logged_entry_signal_keys.add(event_key)

        execution = trade_decision.execution_estimate
        logger.info(
            "SIGNAL ENTRY: action=%s direction=%s obs=%s "
            "p=%.3f p_up=%.3f p_down=%.3f confidence=%.2f "
            "price=%.4f limit=%.4f size=%.2fUSDC edge=%+.0fbps "
            "ptb=%s tau=%ss gate=%s",
            action,
            direction,
            observation_state,
            trade_decision.probability,
            forecast.p_up_settlement,
            forecast.p_down_settlement,
            trade_decision.confidence,
            trade_decision.market_price,
            execution.limit_price if execution else 0.0,
            trade_decision.position_size,
            trade_decision.edge_bps,
            f"{forecast.price_to_beat:.2f}" if forecast.price_to_beat is not None else "N/A",
            f"{forecast.time_to_settlement_seconds:.0f}",
            primary_gate,
        )

    def _log_heartbeat_event(self) -> None:
        now = time.time()
        if now - self._last_heartbeat_log < self._heartbeat_interval:
            return
        self._last_heartbeat_log = now

        market = self._market_info
        slug = market.slug if market else "unknown"
        ptb = market.price_to_beat if market else None
        tau = self._time_to_market_settlement()
        snapshot = self._last_position_snapshot
        state = self._current_market_state
        btc_price = state.btc_price if state else None
        decision = state.trade_decision if state else None
        forecast = decision.settlement_forecast if decision else (state.settlement_forecast if state else None)

        logger.info(
            "BOT STATUS: cycle=%s slug=%s tau=%ss btc=%s ptb=%s "
            "obs=%s/%s decision=%s direction=%s p=%s open=%s uPnL=%+.2f",
            self._cycle_count,
            slug,
            f"{tau:.0f}",
            f"{btc_price:.2f}" if btc_price is not None else "N/A",
            f"{ptb:.2f}" if ptb is not None else "N/A",
            (
                "ready" if forecast and forecast.observation_ready else
                "pending" if forecast else "N/A"
            ),
            (
                "valid" if forecast and forecast.observation_valid else
                "invalid" if forecast else "N/A"
            ),
            (
                "EXECUTE" if decision and decision.should_trade else
                "NO_TRADE" if decision else "N/A"
            ),
            (
                decision.direction.value
                if decision and decision.direction is not None
                else "N/A"
            ),
            f"{decision.probability:.3f}" if decision else "N/A",
            snapshot.open_count,
            snapshot.unrealized_pnl,
        )

    def _market_metadata(self, market_slug: str = "") -> Dict[str, Any]:
        market = self._market_info
        slug = market_slug or (market.slug if market else "")
        market_start = None
        market_end = None
        if slug.startswith("btc-updown-5m-"):
            try:
                market_start = float(int(slug[len("btc-updown-5m-"):]))
                market_end = market_start + 300.0
            except ValueError:
                market_start = None
                market_end = None

        return {
            "market_slug": slug,
            "market_question": market.question if market and market.slug == slug else "",
            "condition_id": market.condition_id if market and market.slug == slug else "",
            "market_end_iso": market.end_date_iso if market and market.slug == slug else "",
            "market_start_ts": market_start,
            "market_end_ts": market_end,
            "market_start_utc": self._iso_timestamp(market_start),
            "market_end_utc": self._iso_timestamp(market_end),
        }

    def _open_trade_history_event(
        self,
        *,
        position: OpenPosition,
        trade_decision: TradeDecision,
        order_result: Optional[Dict[str, Any]],
        fill_details: Dict[str, Optional[float]],
    ) -> Dict[str, Any]:
        forecast = trade_decision.settlement_forecast
        execution = trade_decision.execution_estimate
        return {
            "schema": "polymarket_bot.trade_history.v1",
            "event": "open",
            "timestamp": time.time(),
            "timestamp_utc": self._iso_timestamp(time.time()),
            **self._market_metadata(position.market_slug),
            "position_id": position.position_id,
            "order_id": position.order_id,
            "condition_id": position.condition_id,
            "order_status": self._order_result_status(order_result) or "unknown",
            "token_id": position.token_id,
            "direction": position.direction,
            "is_dry_run": position.is_dry_run,
            "entry_price": position.entry_price,
            "shares": position.shares,
            "cost_usdc": position.cost,
            "entry_fee_usdc": position.entry_fee,
            "price_to_beat": position.price_to_beat,
            "p_up_at_entry": position.p_up_at_entry,
            "p_down_at_entry": 1.0 - position.p_up_at_entry,
            "confidence_at_entry": position.confidence_at_entry,
            "edge_at_entry_bps": position.edge_at_entry_bps,
            "regime_at_entry": position.regime_at_entry,
            "signal_type": trade_decision.signal_type,
            "market_price": trade_decision.market_price,
            "limit_price": execution.limit_price,
            "effective_price": execution.effective_price,
            "fill_probability": trade_decision.fill_probability,
            "expected_slippage_bps": trade_decision.expected_slippage_bps,
            "visible_depth_notional": execution.visible_depth_notional,
            "forecast_p_up": forecast.p_up_settlement,
            "forecast_p_down": forecast.p_down_settlement,
            "forecast_z_score": forecast.terminal_z_score,
            "forecast_sigma_per_sqrt_second": forecast.terminal_sigma_per_sqrt_second,
            "forecast_drift_per_second": forecast.terminal_drift_per_second,
            "round_lock": forecast.round_direction_locked,
            "round_direction": forecast.round_direction,
            "round_confidence": forecast.round_direction_confidence,
            "observation_ready": forecast.observation_ready,
            "observation_valid": forecast.observation_valid,
            "observation_seconds": forecast.observation_seconds,
            "observation_p_up": forecast.observation_p_up,
            "technical_valid": forecast.technical_valid,
            "technical_p_up": forecast.technical_p_up,
            "technical_timeframe": forecast.technical_dominant_timeframe,
            "fill_details": fill_details,
            "reason": trade_decision.reason,
        }

    def _settled_trade_history_event(
        self,
        *,
        realized: Any,
        price_source: str,
    ) -> Dict[str, Any]:
        pnl_pct = realized.pnl / max(realized.cost, 0.01)
        btc_minus_ptb = realized.final_btc_price - realized.price_to_beat
        return {
            "schema": "polymarket_bot.trade_history.v1",
            "event": "settle",
            "timestamp": time.time(),
            "timestamp_utc": self._iso_timestamp(time.time()),
            **self._market_metadata(realized.market_slug),
            "position_id": realized.position_id,
            "order_id": realized.order_id,
            "condition_id": realized.condition_id,
            "token_id": realized.token_id,
            "direction": realized.direction,
            "actual_outcome": realized.actual_outcome,
            "is_win": realized.is_win,
            "is_dry_run": realized.is_dry_run,
            "entry_timestamp": realized.entry_timestamp,
            "entry_timestamp_utc": self._iso_timestamp(realized.entry_timestamp),
            "entry_price": realized.entry_price,
            "exit_price": realized.exit_price,
            "shares": realized.shares,
            "cost_usdc": realized.cost,
            "entry_fee_usdc": realized.entry_fee,
            "payout_usdc": realized.payout,
            "pnl_usdc": realized.pnl,
            "pnl_pct": pnl_pct,
            "final_btc_price": realized.final_btc_price,
            "price_to_beat": realized.price_to_beat,
            "btc_minus_ptb": btc_minus_ptb,
            "settlement_price_source": price_source,
            "confidence_at_entry": realized.confidence_at_entry,
            "edge_at_entry_bps": realized.edge_at_entry_bps,
            "regime_at_entry": realized.regime_at_entry,
            "p_up_at_entry": realized.p_up_at_entry,
            "p_down_at_entry": 1.0 - realized.p_up_at_entry,
        }

    def _settlement_btc_price_for_position(
        self,
        position: OpenPosition,
        fallback_btc_price: Optional[float],
    ) -> Tuple[Optional[float], str]:
        if position.market_end_timestamp is not None:
            price, source = self.tick_buffer.price_at_timestamp(
                position.market_end_timestamp,
                max_skew_seconds=4.0,
                max_gap_seconds=8.0,
            )
            if price is not None:
                return price, f"tick_buffer:{source}"

        if fallback_btc_price is not None and fallback_btc_price > 0.0:
            return fallback_btc_price, "current_settlement_fallback"
        return None, "unavailable"

    @staticmethod
    def _outcome_from_settlement_price(
        final_btc_price: Optional[float],
        price_to_beat: Optional[float],
    ) -> Direction:
        if final_btc_price is None or price_to_beat is None or price_to_beat <= 0.0:
            return Direction.NEUTRAL
        return Direction.UP if final_btc_price >= price_to_beat else Direction.DOWN

    async def _settlement_btc_price_for_position_async(
        self,
        position: OpenPosition,
        fallback_btc_price: Optional[float],
    ) -> Tuple[Optional[float], str, Direction]:
        if not self._use_official_settlement_close:
            # ── Infrastructure settlement: authoritative outcome check ──
            # If the infra settlement engine has resolved this market,
            # use the authoritative outcome but still get BTC price from
            # existing feeds (settlement engine provides winner, not price).
            infra_settlement = getattr(self, "_infra_settlement", None)
            if infra_settlement is not None:
                try:
                    if infra_settlement.is_settled(position.market_slug):
                        resolution = infra_settlement.get_resolution(position.market_slug)
                        if resolution is not None and resolution.is_known:
                            from polymarket_bot.infrastructure.settlement import ResolutionOutcome
                            local_price, local_source = self._settlement_btc_price_for_position(
                                position, fallback_btc_price
                            )
                            final_price = local_price or fallback_btc_price or 0.0
                            actual_outcome = (
                                Direction.UP if resolution.outcome == ResolutionOutcome.UP
                                else Direction.DOWN
                            )
                            return (
                                final_price,
                                f"infra:{resolution.source.value}",
                                actual_outcome,
                            )
                    else:
                        from polymarket_bot.infrastructure.types import MarketInfo as InfraMarketInfo
                        lookup = InfraMarketInfo(
                            slug=position.market_slug,
                            condition_id=position.condition_id or "",
                            question="",
                            up_token_id="",
                            down_token_id="",
                            price_to_beat=0.0,
                            end_time=0.0,
                            event_start_time=_bucket_from_slug(position.market_slug),
                        )
                        resolution = await infra_settlement.fetch_resolution(lookup)
                        if resolution is not None and resolution.is_known:
                            from polymarket_bot.infrastructure.settlement import ResolutionOutcome
                            local_price, local_source = self._settlement_btc_price_for_position(
                                position, fallback_btc_price
                            )
                            final_price = local_price or fallback_btc_price or 0.0
                            actual_outcome = (
                                Direction.UP if resolution.outcome == ResolutionOutcome.UP
                                else Direction.DOWN
                            )
                            return (
                                final_price,
                                f"infra:{resolution.source.value}",
                                actual_outcome,
                            )
                except Exception:
                    pass

            local_price, local_source = self._settlement_btc_price_for_position(
                position,
                fallback_btc_price,
            )
            return (
                local_price,
                local_source,
                self._outcome_from_settlement_price(
                    local_price,
                    position.price_to_beat,
                ),
            )

        # ── Infrastructure settlement: WebSocket-first resolution ──────
        # The infra settlement engine captures CLOB WS market_resolved
        # events in real-time. If settled, use the authoritative outcome
        # but get BTC price from existing feeds.
        infra_settlement = getattr(self, "_infra_settlement", None)
        if infra_settlement is not None:
            try:
                if infra_settlement.is_settled(position.market_slug):
                    resolution = infra_settlement.get_resolution(position.market_slug)
                    if resolution is not None and resolution.is_known:
                        from polymarket_bot.infrastructure.settlement import ResolutionOutcome
                        local_price, local_source = self._settlement_btc_price_for_position(
                            position, fallback_btc_price
                        )
                        final_price = local_price or fallback_btc_price or 0.0
                        actual_outcome = (
                            Direction.UP if resolution.outcome == ResolutionOutcome.UP
                            else Direction.DOWN
                        )
                        return (
                            final_price,
                            f"infra:{resolution.source.value}",
                            actual_outcome,
                        )
                else:
                    from polymarket_bot.infrastructure.types import MarketInfo as InfraMarketInfo
                    lookup = InfraMarketInfo(
                        slug=position.market_slug,
                        condition_id=position.condition_id or "",
                        question="",
                        up_token_id="",
                        down_token_id="",
                        price_to_beat=0.0,
                        end_time=0.0,
                        event_start_time=_bucket_from_slug(position.market_slug),
                    )
                    resolution = await infra_settlement.fetch_resolution(lookup)
                    if resolution is not None and resolution.is_known:
                        from polymarket_bot.infrastructure.settlement import ResolutionOutcome
                        local_price, local_source = self._settlement_btc_price_for_position(
                            position, fallback_btc_price
                        )
                        final_price = local_price or fallback_btc_price or 0.0
                        actual_outcome = (
                            Direction.UP if resolution.outcome == ResolutionOutcome.UP
                            else Direction.DOWN
                        )
                        return (
                            final_price,
                            f"infra:{resolution.source.value}",
                            actual_outcome,
                        )
            except Exception:
                pass

        try:
            snapshot = await self.polymarket_client.get_btc_5m_settlement_price(
                position.market_slug
            )
        except Exception as exc:
            logger.debug(
                "Official settlement close fetch failed: slug=%s error=%s",
                position.market_slug,
                exc,
            )
            snapshot = None
        if snapshot is not None and snapshot.get("is_valid") and snapshot.get("price", 0.0) > 0.0:
            return (
                snapshot.get("price"),
                snapshot.get("source", ""),
                self._outcome_from_settlement_price(
                    snapshot.get("price"),
                    position.price_to_beat,
                ),
            )

        return None, "official_settlement_pending", Direction.NEUTRAL

    def _finalize_position_settlement(
        self,
        *,
        position_id: str,
        final_price: float,
        actual_outcome: Direction,
        price_source: str,
    ) -> bool:
        realized = self.position_tracker.settle_position(
            position_id,
            final_btc_price=final_price,
            actual_outcome=actual_outcome,
        )
        if realized is None:
            return False

        self.risk_manager.record_trade_result(
            direction=realized.direction,
            position_size=realized.cost,
            entry_price=realized.entry_price,
            exit_price=realized.exit_price,
            pnl=realized.pnl,
            confidence_at_entry=realized.confidence_at_entry,
            edge_at_entry_bps=realized.edge_at_entry_bps,
            regime_at_entry=realized.regime_at_entry,
        )
        try:
            self.posterior_calibrator.record_outcome(
                predicted_prob=realized.p_up_at_entry,
                actual_outcome=1.0 if realized.actual_outcome == Direction.UP else 0.0,
            )
        except Exception as exc:
            logger.debug("Posterior outcome record failed: %s", exc)

        logger.info(
            "TRADE SETTLED: direction=%s outcome=%s "
            "entry=%.4f exit=%.1f shares=%.4f cost=%.2f fee=%.4f payout=%.2f "
            "pnl=%+.2fUSDC btc_final=%.2f ptb=%.2f price_src=%s",
            realized.direction.value,
            realized.actual_outcome.value,
            realized.entry_price,
            realized.exit_price,
            realized.shares,
            realized.cost,
            realized.entry_fee,
            realized.payout,
            realized.pnl,
            realized.final_btc_price,
            realized.price_to_beat,
            price_source,
        )
        self._append_trade_history_event(
            self._settled_trade_history_event(
                realized=realized,
                price_source=price_source,
            )
        )
        if realized.condition_id:
            self._claim_candidates.setdefault(
                realized.condition_id,
                {
                    "market_slug": realized.market_slug,
                    "condition_id": realized.condition_id,
                    "token_ids": set(),
                    "settled_at": time.time(),
                },
            )
            self._claim_candidates[realized.condition_id]["token_ids"].add(
                realized.token_id
            )

        self._apply_position_snapshot(self.position_tracker.snapshot())
        return True

    def _settlement_task_done(self, position_id: str, task: asyncio.Task) -> None:
        if self._settlement_tasks.get(position_id) is task:
            self._settlement_tasks.pop(position_id, None)
        if task.cancelled():
            return
        try:
            task.result()
        except Exception as exc:
            logger.warning("Settlement background task failed: position=%s error=%s", position_id, exc)

    async def _settle_position_background(
        self,
        position_id: str,
        fallback_btc_price: Optional[float],
    ) -> bool:
        # Immediately background this position so it no longer blocks
        # entry into the next round and does not pollute "active" metrics.
        self.position_tracker.mark_position_awaiting_settlement(position_id)

        logged_pending = False
        while True:
            position = self.position_tracker.get_position(position_id)
            if position is None:
                return False

            # ── Infrastructure settlement: authoritative resolution ────
            # Check the infrastructure settlement engine first for
            # official onchain/CLOB resolution. The settlement engine
            # tells us WHO WON (UP/DOWN). We still need BTC price from
            # existing feeds for logging/metrics.
            infra_settlement = getattr(self, "_infra_settlement", None)
            if infra_settlement is not None:
                try:
                    if infra_settlement.is_settled(position.market_slug):
                        resolution = infra_settlement.get_resolution(position.market_slug)
                        if resolution is not None and resolution.is_known:
                            from polymarket_bot.infrastructure.settlement import ResolutionOutcome
                            actual_outcome = (
                                Direction.UP
                                if resolution.outcome == ResolutionOutcome.UP
                                else Direction.DOWN
                            )
                            final_price = (
                                fallback_btc_price
                                or (self.price_feed.get_current_settlement_price() if self.price_feed else None)
                                or 0.0
                            )
                            return self._finalize_position_settlement(
                                position_id=position_id,
                                final_price=final_price,
                                actual_outcome=actual_outcome,
                                price_source=f"infra:{resolution.source.value}",
                            )
                    else:
                        from polymarket_bot.infrastructure.types import MarketInfo as InfraMarketInfo
                        lookup = InfraMarketInfo(
                            slug=position.market_slug,
                            condition_id=position.condition_id or "",
                            question="",
                            up_token_id="",
                            down_token_id="",
                            price_to_beat=0.0,
                            end_time=0.0,
                            event_start_time=_bucket_from_slug(position.market_slug),
                        )
                        resolution = await infra_settlement.fetch_resolution(lookup)
                        if resolution is not None and resolution.is_known:
                            from polymarket_bot.infrastructure.settlement import ResolutionOutcome
                            actual_outcome = (
                                Direction.UP
                                if resolution.outcome == ResolutionOutcome.UP
                                else Direction.DOWN
                            )
                            final_price = (
                                fallback_btc_price
                                or (self.price_feed.get_current_settlement_price() if self.price_feed else None)
                                or 0.0
                            )
                            return self._finalize_position_settlement(
                                position_id=position_id,
                                final_price=final_price,
                                actual_outcome=actual_outcome,
                                price_source=f"infra:{resolution.source.value}",
                            )
                except Exception:
                    pass

            # ── Infra PTB backfill (O(1) lookup, no REST) ──────────────
            if (
                position.price_to_beat is None
                or position.price_to_beat <= 0.0
            ):
                infra_ptb = getattr(self, "_infra_ptb", None)
                if infra_ptb is not None:
                    try:
                        cached = infra_ptb.get_ptb(position.market_slug)
                        if cached is not None and cached.is_valid:
                            position.price_to_beat = cached.price
                            logger.info(
                                "SETTLEMENT PTB BACKFILL (infra): position=%s ptb=%.2f",
                                position_id[:32],
                                cached.price,
                            )
                    except Exception:
                        pass

            fallback_price = fallback_btc_price
            if not self._use_official_settlement_close:
                try:
                    latest_price = self.price_feed.get_current_settlement_price()
                except Exception:
                    latest_price = None
                if latest_price is not None and latest_price > 0.0:
                    fallback_price = latest_price

            final_price, price_source, actual_outcome = await self._settlement_btc_price_for_position_async(
                position,
                fallback_price,
            )
            if final_price is not None:
                return self._finalize_position_settlement(
                    position_id=position_id,
                    final_price=final_price,
                    actual_outcome=actual_outcome,
                    price_source=price_source,
                )

            if not logged_pending:
                logger.info(
                    "SETTLEMENT PENDING: direction=%s source=%s ptb=%s slug=%s",
                    position.direction.value,
                    price_source,
                    f"{position.price_to_beat:.2f}" if position.price_to_beat is not None and position.price_to_beat > 0 else "N/A",
                    position.market_slug,
                )
                logged_pending = True
            await asyncio.sleep(self._official_settlement_poll_seconds)

    async def _settle_due_positions(self, btc_price: Optional[float]) -> int:
        scheduled_count = 0
        for position in self.position_tracker.positions_due_for_settlement(
            grace_seconds=self._local_settlement_grace_seconds,
        ):
            existing_task = self._settlement_tasks.get(position.position_id)
            if existing_task is not None and not existing_task.done():
                continue

            task = asyncio.create_task(
                self._settle_position_background(position.position_id, btc_price),
                name=f"settlement-{position.position_id[:24]}",
            )
            self._settlement_tasks[position.position_id] = task
            task.add_done_callback(
                lambda done_task, position_id=position.position_id: self._settlement_task_done(
                    position_id,
                    done_task,
                )
            )
            scheduled_count += 1
        return scheduled_count

    async def _update_position_monitoring(
        self,
        *,
        btc_price: Optional[float],
        poly_ob_up: Optional[PolymarketOrderbook],
        poly_ob_down: Optional[PolymarketOrderbook],
    ) -> PositionSnapshot:
        await self._settle_due_positions(btc_price)

        orderbooks_by_token: Dict[str, Optional[PolymarketOrderbook]] = {}
        if poly_ob_up is not None:
            orderbooks_by_token[poly_ob_up.token_id] = poly_ob_up
        if poly_ob_down is not None:
            orderbooks_by_token[poly_ob_down.token_id] = poly_ob_down

        snapshot = self.position_tracker.mark_to_market(orderbooks_by_token)
        self._apply_position_snapshot(snapshot)
        return snapshot

    def _record_filled_position(
        self,
        *,
        trade_decision: TradeDecision,
        order_result: Optional[Dict[str, Any]],
    ) -> bool:
        fill_details = self._filled_order_details(order_result)
        if fill_details is None:
            return False

        market_slug = self._market_info.slug if self._market_info else ""
        condition_id = self._market_info.condition_id if self._market_info else ""
        price_to_beat = (
            self._market_info.price_to_beat
            if self._market_info and self._market_info.price_to_beat is not None
            else trade_decision.settlement_forecast.price_to_beat
        )
        if not market_slug or price_to_beat is None:
            logger.warning(
                "Filled order could not be tracked: missing market_slug or price_to_beat"
            )
            return False

        fee_entry_price = (
            fill_details.get("filled_price")
            or trade_decision.execution_estimate.limit_price
            or trade_decision.execution_estimate.effective_price
            or trade_decision.market_price
            or 0.0
        )
        fee_notional = fill_details.get("filled_notional") or trade_decision.position_size
        fee_shares = fill_details.get("filled_shares")
        if (fee_shares is None or fee_shares <= 0.0) and fee_entry_price > 0.0:
            fee_shares = fee_notional / fee_entry_price
        entry_fee = self._estimate_entry_fee(
            market_info=self._market_info,
            entry_price=fee_entry_price,
            shares=fee_shares,
            explicit_fee=fill_details.get("filled_fee"),
            notional=fee_notional,
        )
        fill_details["entry_fee_usdc"] = entry_fee

        position = self.position_tracker.add_filled_buy(
            decision=trade_decision,
            market_slug=market_slug,
            price_to_beat=price_to_beat,
            condition_id=condition_id,
            order_id=self._order_result_order_id(order_result),
            filled_price=fill_details.get("filled_price"),
            filled_shares=fill_details.get("filled_shares"),
            filled_notional=fill_details.get("filled_notional"),
            filled_fee=entry_fee,
            is_dry_run=self.poly_config.dry_run,
        )
        if position is None:
            logger.warning("Filled order could not be tracked: invalid fill details")
            return False

        snapshot = self.position_tracker.snapshot()
        self._apply_position_snapshot(snapshot)
        self._total_trades += 1
        logger.info(
            "TRADE OPENED: direction=%s token=%s entry=%.4f "
            "shares=%.4f cost=%.2fUSDC fee=%.4fUSDC ptb=%.2f dry_run=%s",
            position.direction.value,
            position.token_id,
            position.entry_price,
            position.shares,
            position.cost,
            position.entry_fee,
            position.price_to_beat or 0.0,
            position.is_dry_run,
        )
        self.decision_engine.record_filled_entry(market_slug, position.direction)
        self._append_trade_history_event(
            self._open_trade_history_event(
                position=position,
                trade_decision=trade_decision,
                order_result=order_result,
                fill_details=fill_details,
            )
        )
        return True

    @staticmethod
    def _prob_logit(probability: float) -> float:
        p = float(np.clip(probability, 1.0e-8, 1.0 - 1.0e-8))
        return float(np.log(p / (1.0 - p)))

    @staticmethod
    def _prob_sigmoid(log_odds: float) -> float:
        if log_odds >= 0.0:
            z = np.exp(-log_odds)
            return float(1.0 / (1.0 + z))
        z = np.exp(log_odds)
        return float(z / (1.0 + z))

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return float(0.5 * (1.0 + erf(value / sqrt(2.0))))

    def _barrier_lock_probability(
        self,
        *,
        btc_price: Optional[float],
        price_to_beat: Optional[float],
        time_to_settlement: float,
        empirical_sigma: Optional[float],
        empirical_drift: Optional[float],
    ) -> Tuple[Optional[float], float, float]:
        if (
            btc_price is None
            or price_to_beat is None
            or btc_price <= 0.0
            or price_to_beat <= 0.0
        ):
            return None, 0.0, 0.0
        sigma = (
            float(empirical_sigma)
            if empirical_sigma is not None and np.isfinite(empirical_sigma) and empirical_sigma > 0.0
            else None
        )
        if sigma is None:
            return None, 0.0, 0.0
        tau = max(float(time_to_settlement), 1.0)
        drift = (
            float(empirical_drift)
            if empirical_drift is not None and np.isfinite(empirical_drift)
            else 0.0
        )
        distance = float(np.log(float(btc_price) / float(price_to_beat)))
        scale = max(sigma * np.sqrt(tau), 1.0e-8)
        z_score = float(np.clip((distance + drift * tau) / scale, -8.0, 8.0))
        p_up = float(np.clip(self._normal_cdf(z_score), 0.001, 0.999))
        confidence = float(np.clip(1.0 - np.exp(-abs(z_score)), 0.0, 1.0))
        return p_up, confidence, z_score

    def _round_lock_conflicts_live_evidence(
        self,
        lock: Dict[str, Any],
        *,
        btc_price: Optional[float],
        price_to_beat: Optional[float],
        time_to_settlement: float,
        observation_signal: Optional[ObservationSignal],
        technical_signal: Optional[TechnicalMomentumSignal],
        empirical_sigma: Optional[float],
        empirical_drift: Optional[float],
    ) -> Optional[str]:
        locked_direction = lock.get("direction")
        if locked_direction not in {Direction.UP, Direction.DOWN}:
            return "invalid_lock_direction"
        locked_sign = 1.0 if locked_direction == Direction.UP else -1.0
        min_abs_log_odds = float(getattr(self.config, "round_direction_min_abs_log_odds", 0.50))
        conflict_log_odds = float(getattr(
            self.config,
            "round_direction_lock_conflict_log_odds",
            0.75,
        ))
        conflict_confidence = float(getattr(
            self.config,
            "round_direction_lock_conflict_confidence",
            0.55,
        ))

        def conflicts(
            name: str,
            log_odds: float,
            confidence: float,
            log_odds_floor: float,
        ) -> Optional[str]:
            if not np.isfinite(log_odds):
                return None
            evidence_sign = float(np.sign(log_odds))
            if (
                evidence_sign != 0.0
                and evidence_sign != locked_sign
                and abs(log_odds) >= log_odds_floor
                and confidence >= conflict_confidence
            ):
                return f"{name}_conflict(log_odds={log_odds:+.2f},conf={confidence:.2f})"
            return None

        if observation_signal is not None and observation_signal.is_valid:
            reason = conflicts(
                "observation",
                self._prob_logit(observation_signal.p_up),
                float(observation_signal.confidence),
                max(conflict_log_odds, min_abs_log_odds),
            )
            if reason:
                return reason

        if technical_signal is not None and technical_signal.is_valid:
            reason = conflicts(
                "technical",
                float(technical_signal.log_odds),
                float(technical_signal.confidence),
                max(conflict_log_odds, min_abs_log_odds * 0.90),
            )
            if reason:
                return reason

        barrier_probability, barrier_confidence, barrier_z = self._barrier_lock_probability(
            btc_price=btc_price,
            price_to_beat=price_to_beat,
            time_to_settlement=time_to_settlement,
            empirical_sigma=empirical_sigma,
            empirical_drift=empirical_drift,
        )
        if barrier_probability is not None:
            barrier_reason = conflicts(
                "barrier",
                self._prob_logit(barrier_probability),
                float(barrier_confidence),
                max(conflict_log_odds, min_abs_log_odds * 0.90),
            )
            if barrier_reason:
                return f"{barrier_reason},z={barrier_z:.2f}"
        return None

    def _get_round_direction_lock(
        self,
        *,
        market_slug: str,
        btc_price: Optional[float],
        price_to_beat: Optional[float],
        time_to_settlement: float,
        observation_signal: Optional[ObservationSignal],
        technical_signal: Optional[TechnicalMomentumSignal],
        empirical_sigma: Optional[float],
        empirical_drift: Optional[float],
    ) -> Optional[Dict[str, Any]]:
        if not getattr(self.config, "round_direction_lock_enabled", True):
            return None
        if not market_slug:
            return None
        existing = self._round_direction_locks.get(market_slug)
        if existing:
            conflict_reason = self._round_lock_conflicts_live_evidence(
                existing,
                btc_price=btc_price,
                price_to_beat=price_to_beat,
                time_to_settlement=time_to_settlement,
                observation_signal=observation_signal,
                technical_signal=technical_signal,
                empirical_sigma=empirical_sigma,
                empirical_drift=empirical_drift,
            )
            if conflict_reason:
                self._round_direction_locks.pop(market_slug, None)
                logger.warning(
                    "ROUND LOCK INVALIDATED: market=%s direction=%s reason=%s",
                    market_slug,
                    getattr(existing.get("direction"), "value", existing.get("direction")),
                    conflict_reason,
                )
                return None
            return existing
        if observation_signal is None or not observation_signal.is_ready:
            return None
        if not observation_signal.is_valid:
            return None

        evidences: List[Tuple[str, float, float]] = []
        obs_log_odds = self._prob_logit(observation_signal.p_up)
        min_abs_log_odds = float(getattr(self.config, "round_direction_min_abs_log_odds", 0.50))
        min_obs_confidence = float(np.clip(
            getattr(self.config, "round_direction_min_confidence", 0.68) * 0.92,
            0.60,
            0.70,
        ))
        if abs(obs_log_odds) < min_abs_log_odds * 0.75:
            return None
        if observation_signal.confidence < min_obs_confidence:
            return None

        obs_weight = float(np.clip(
            0.25 + 0.75 * observation_signal.confidence * observation_signal.validation_score,
            0.10,
            1.00,
        ))
        evidences.append(("obs", obs_log_odds, obs_weight))

        obs_direction_sign = float(np.sign(obs_log_odds))
        if (
            technical_signal is not None
            and technical_signal.is_valid
        ):
            tech_log_odds = float(technical_signal.log_odds)
            tech_sign = float(np.sign(tech_log_odds))
            technical_opposes = (
                obs_direction_sign != 0.0
                and tech_sign != 0.0
                and tech_sign != obs_direction_sign
                and abs(tech_log_odds) >= min_abs_log_odds * 0.70
                and technical_signal.confidence >= 0.45
            )
            if technical_opposes:
                return None
            if technical_signal.conflict_score <= float(
                getattr(self.config, "round_direction_max_technical_conflict", 0.35)
            ):
                tech_weight = float(np.clip(
                    0.15
                    + 0.65
                    * technical_signal.confidence
                    * technical_signal.validation_score
                    * technical_signal.consensus_score,
                    0.05,
                    0.80,
                ))
                evidences.append(("tech", tech_log_odds, tech_weight))

        barrier_probability, barrier_confidence, barrier_z = self._barrier_lock_probability(
            btc_price=btc_price,
            price_to_beat=price_to_beat,
            time_to_settlement=time_to_settlement,
            empirical_sigma=empirical_sigma,
            empirical_drift=empirical_drift,
        )
        if barrier_probability is not None:
            barrier_log_odds = self._prob_logit(barrier_probability)
            barrier_sign = float(np.sign(barrier_log_odds))
            barrier_opposes = (
                obs_direction_sign != 0.0
                and barrier_sign != 0.0
                and barrier_sign != obs_direction_sign
                and abs(barrier_log_odds) >= min_abs_log_odds * 0.75
                and barrier_confidence >= 0.45
            )
            if barrier_opposes:
                return None
            barrier_weight = float(np.clip(0.08 + 0.32 * barrier_confidence, 0.05, 0.40))
            evidences.append(("barrier", barrier_log_odds, barrier_weight))

        total_weight = sum(weight for _, _, weight in evidences)
        if total_weight <= 1.0e-12:
            return None
        locked_log_odds = float(
            sum(log_odds * weight for _, log_odds, weight in evidences) / total_weight
        )
        abs_log_odds = abs(locked_log_odds)
        if abs_log_odds < min_abs_log_odds:
            return None

        direction = Direction.UP if locked_log_odds > 0.0 else Direction.DOWN
        direction_sign = 1.0 if direction == Direction.UP else -1.0
        same_side_weight = sum(
            weight
            for _, log_odds, weight in evidences
            if np.sign(log_odds) == direction_sign or abs(log_odds) < 1.0e-9
        )
        agreement = float(np.clip(same_side_weight / total_weight, 0.0, 1.0))
        magnitude = float(np.clip(abs_log_odds / 2.0, 0.0, 1.0))
        validation = float(observation_signal.validation_score)
        confidence = float(np.clip(
            0.30 * agreement
            + 0.25 * magnitude
            + 0.25 * validation
            + 0.20 * observation_signal.confidence,
            0.0,
            0.99,
        ))
        if confidence < float(getattr(self.config, "round_direction_min_confidence", 0.68)):
            return None

        locked_probability_up = float(np.clip(self._prob_sigmoid(locked_log_odds), 0.001, 0.999))
        source_text = ",".join(
            f"{name}:{log_odds:+.2f}x{weight:.2f}"
            for name, log_odds, weight in evidences
        )
        lock = {
            "market_slug": market_slug,
            "timestamp": time.time(),
            "direction": direction,
            "p_up": locked_probability_up,
            "confidence": confidence,
            "log_odds": locked_log_odds,
            "reason": (
                f"sources={source_text},agree={agreement:.2f},"
                f"barrier_z={barrier_z:.2f}"
            ),
        }
        self._round_direction_locks[market_slug] = lock
        if len(self._round_direction_locks) > 64:
            keys = list(self._round_direction_locks.keys())
            for stale_key in keys[:-32]:
                self._round_direction_locks.pop(stale_key, None)
        logger.info(
            "ROUND LOCK: direction=%s p=%.3f conf=%.2f status=valid",
            direction.value,
            locked_probability_up if direction == Direction.UP else 1.0 - locked_probability_up,
            confidence,
        )
        return lock

    def _append_tick_to_buffer(self, tick: TickData) -> bool:
        key = (
            tick.trade_id
            if tick.trade_id
            else f"{tick.timestamp:.6f}:{tick.price:.8f}:{tick.volume:.8f}:{tick.side}"
        )
        if key in self._seen_tick_keys:
            return False

        self._seen_tick_keys.add(key)
        self._seen_tick_key_order.append(key)
        if len(self._seen_tick_key_order) > self._max_seen_tick_keys:
            stale = self._seen_tick_key_order[: self._max_seen_tick_keys // 2]
            self._seen_tick_key_order = self._seen_tick_key_order[self._max_seen_tick_keys // 2 :]
            for stale_key in stale:
                self._seen_tick_keys.discard(stale_key)

        self.tick_buffer.append(tick)
        if tick.side in {"buy", "sell"} and tick.volume > 0.0:
            self._last_trade_tick = tick
            last_flow_timestamp = float(getattr(self.flow_analyzer, "_last_timestamp", 0.0) or 0.0)
            if tick.timestamp + 1.0e-6 >= last_flow_timestamp:
                flow_metrics = self.flow_analyzer.update(tick)
                if flow_metrics is not None:
                    self._latest_flow_metrics = flow_metrics
        return True

    def _append_orderbook_delta_to_buffer(self, delta: OrderbookDelta) -> bool:
        key = (
            f"{delta.sequence_number}:"
            f"{delta.timestamp:.6f}:"
            f"{delta.side}:"
            f"{delta.price:.8f}:"
            f"{delta.size_delta:.8f}:"
            f"{int(delta.is_trade)}"
        )
        if key in self._seen_orderbook_delta_keys:
            return False

        self._seen_orderbook_delta_keys.add(key)
        self._seen_orderbook_delta_key_order.append(key)
        if len(self._seen_orderbook_delta_key_order) > self._max_seen_orderbook_delta_keys:
            midpoint = self._max_seen_orderbook_delta_keys // 2
            stale = self._seen_orderbook_delta_key_order[:midpoint]
            self._seen_orderbook_delta_key_order = self._seen_orderbook_delta_key_order[midpoint:]
            for stale_key in stale:
                self._seen_orderbook_delta_keys.discard(stale_key)

        self.orderbook_buffer.add_delta(delta)
        return True

    def _sync_orderbook_deltas_from_feed(self, limit: int = 1500) -> int:
        synced = 0
        for delta in self.price_feed.recent_orderbook_deltas(limit=limit):
            if self._append_orderbook_delta_to_buffer(delta):
                synced += 1
        return synced

    def _align_ticks_to_price_basis(
        self,
        ticks: List[TickData],
        anchor_price: Optional[float],
    ) -> List[TickData]:
        """
        Align REST backfill ticks to the settlement price basis before they enter
        the statistical buffers. This preserves short-horizon returns while
        avoiding Binance/Coinbase level jumps in the same time series.
        """
        if not ticks or anchor_price is None or anchor_price <= 0.0:
            return ticks

        latest_tick_price = None
        for tick in reversed(ticks):
            if tick.price > 0.0 and np.isfinite(tick.price):
                latest_tick_price = float(tick.price)
                break
        if latest_tick_price is None:
            return ticks

        scale = float(anchor_price) / latest_tick_price
        if not np.isfinite(scale) or scale <= 0.0:
            return ticks
        if abs(scale - 1.0) <= 1.0e-8:
            return ticks

        return [
            TickData(
                timestamp=tick.timestamp,
                price=float(tick.price) * scale if tick.price > 0.0 else tick.price,
                volume=tick.volume,
                side=tick.side,
                trade_id=tick.trade_id,
                is_snapshot=tick.is_snapshot,
            )
            for tick in ticks
        ]

    def _recent_sorted_tick_arrays(self, max_points: int = 1500) -> Tuple[np.ndarray, np.ndarray]:
        n = min(max_points, self.tick_buffer.len())
        if n < 2:
            return np.array([]), np.array([])

        timestamps = np.array(list(self.tick_buffer.timestamps)[-n:], dtype=np.float64)
        prices = np.array(list(self.tick_buffer.prices)[-n:], dtype=np.float64)
        mask = np.isfinite(timestamps) & np.isfinite(prices) & (timestamps > 0) & (prices > 0)
        timestamps = timestamps[mask]
        prices = prices[mask]
        if len(timestamps) < 2:
            return np.array([]), np.array([])

        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        prices = prices[order]

        unique_timestamps = []
        unique_prices = []
        for ts, price in zip(timestamps, prices):
            if unique_timestamps and abs(ts - unique_timestamps[-1]) < 1e-6:
                unique_prices[-1] = price
            else:
                unique_timestamps.append(float(ts))
                unique_prices.append(float(price))

        return (
            np.array(unique_timestamps, dtype=np.float64),
            np.array(unique_prices, dtype=np.float64),
        )

    def _latest_log_return_sample(self) -> Optional[Tuple[float, float]]:
        timestamps, prices = self._recent_sorted_tick_arrays(max_points=100)
        if len(timestamps) < 2:
            return None

        for idx in range(len(timestamps) - 1, 0, -1):
            dt = float(timestamps[idx] - timestamps[idx - 1])
            if 1e-3 <= dt <= 30.0 and prices[idx - 1] > 0:
                return float(np.log(prices[idx] / prices[idx - 1])), dt
        return None

    def _estimate_terminal_log_moments(self) -> Tuple[Optional[float], Optional[float], int]:
        timestamps, prices = self._recent_sorted_tick_arrays(max_points=1500)
        if len(timestamps) < 4:
            return None, None, 0

        now = time.time()
        lookback = min(300.0, max(45.0, self._time_to_market_settlement() * 2.0))
        mask = timestamps >= max(timestamps[-1] - lookback, now - lookback - 5.0)
        timestamps = timestamps[mask]
        prices = prices[mask]
        if len(timestamps) < 4:
            return None, None, 0

        start_second = int(np.floor(timestamps[0]))
        end_second = int(np.floor(timestamps[-1]))
        if end_second - start_second < 3:
            return None, None, 0

        grid = np.arange(start_second, end_second + 1, dtype=np.float64)
        sampled_prices = np.interp(grid, timestamps, prices)
        log_prices = np.log(sampled_prices)
        returns = np.diff(log_prices)
        interval_end = grid[1:]
        valid = np.isfinite(returns) & (np.abs(returns) <= 0.02)
        if np.count_nonzero(valid) < 3:
            return None, None, 0

        returns = returns[valid]
        interval_end = interval_end[valid]
        half_life = max(10.0, min(60.0, lookback / 3.0))
        weights = np.exp(-(grid[-1] - interval_end) / half_life)
        weighted_time = float(np.sum(weights))
        if weighted_time <= 0:
            return None, None, 0

        drift_raw = float(np.sum(weights * returns) / weighted_time)
        # Intraday BTC drift is tiny relative to 5-minute diffusion. Use a
        # strong standard-error shrink so a short tick burst cannot invert the
        # PTB-distance probability.
        sigma_raw = float(np.sqrt(max(np.sum(weights * (returns - drift_raw) ** 2) / weighted_time, 1e-12)))
        effective_n = max(1.0, weighted_time)
        drift_se = sigma_raw / np.sqrt(effective_n)
        shrink = min(0.25, abs(drift_raw) / (abs(drift_raw) + 4.0 * drift_se + 1e-12))
        drift = drift_raw * shrink
        residual = returns - drift
        variance_rate = float(np.sum(weights * residual * residual) / weighted_time)
        sigma = float(np.sqrt(max(variance_rate, 1e-12)))
        return drift, sigma, int(len(returns))

    async def _collect_initial_data(self) -> None:
        """
        Collect initial market data for pipeline warm-up.
        """
        # Get BTC price on the settlement/PTB basis.
        btc_price = self.price_feed.get_current_settlement_price()
        
        if btc_price is None:
            logger.debug("No WebSocket BTC price data available during warm-up.")
            return
        
        # Get BTC orderbook
        btc_ob = self.price_feed.get_current_orderbook()
        
        # Get Polymarket orderbooks
        poly_ob_up = None
        poly_ob_down = None
        if self._market_info:
            try:
                poly_ob_up, poly_ob_down = await self._get_polymarket_orderbooks()
            except Exception as e:
                logger.warning(f"Polymarket orderbook fetch failed: {e}")
        
        # Store in buffers
        for tick in self.price_feed.drain_recent_ticks(limit=500):
            self._append_tick_to_buffer(tick)
        self._sync_orderbook_deltas_from_feed(limit=1500)
        self._append_tick_to_buffer(TickData(
            timestamp=time.time(),
            price=btc_price,
            volume=0.0,
            side="neutral",
        ))
        if btc_ob:
            self.orderbook_buffer.add_snapshot(btc_ob)

    async def run_cycle(self) -> None:
        """
        Run one complete processing cycle.
        
        Pipeline:
        1. Collect fresh data
        2. Run microstructure analysis
        3. Detect regime
        4. Generate signals
        5. Run inference
        6. Model execution
        7. Make decision
        8. Check risk
        9. Execute/suppress trade
        10. Update state
        """
        cycle_start = time.time()
        self._cycle_count += 1
        self._total_cycles += 1
        
        # ---- Step 1: Collect fresh data ----
        btc_price = self.price_feed.get_current_settlement_price()
        
        if btc_price is None:
            logger.debug("No WebSocket BTC price data. Skipping cycle.")
            return

        ws_ticks = self.price_feed.drain_recent_ticks(limit=1000)
        for tick in ws_ticks:
            self._append_tick_to_buffer(tick)
        self._sync_orderbook_deltas_from_feed(limit=1500)

        if self._running:
            self._start_rollover_prefetch_task()
        await self._rollover_market_if_needed()
        self._start_price_to_beat_refresh_task_if_needed()

        if not ws_ticks:
            self._append_tick_to_buffer(TickData(
                timestamp=time.time(),
                price=btc_price,
                volume=0.0,
                side="neutral",
            ))

        # BTC orderbook
        btc_ob = self.price_feed.get_current_orderbook()
        
        # Polymarket orderbooks
        poly_ob_up = None
        poly_ob_down = None
        if self._market_info:
            try:
                poly_ob_up, poly_ob_down = await self._get_polymarket_orderbooks()
            except Exception as e:
                logger.warning(f"Polymarket orderbook error: {e}")
        
        # Process orderbook deltas
        if btc_ob:
            self.orderbook_buffer.add_snapshot(btc_ob)

        # Keep open position exposure and PnL current before risk gates run.
        await self._update_position_monitoring(
            btc_price=btc_price,
            poly_ob_up=poly_ob_up,
            poly_ob_down=poly_ob_down,
        )
        
        # ---- Step 2: Microstructure analysis ----
        orderbook_pressure = None
        if btc_ob:
            orderbook_pressure = self.orderbook_analyzer.analyze(
                btc_ob,
                buffer=self.orderbook_buffer,
            )
        
        flow_metrics = None
        if self.tick_buffer.len() > 10:
            if (
                self._latest_flow_metrics is not None
                and time.time() - self._latest_flow_metrics.timestamp <= 60.0
            ):
                flow_metrics = self._latest_flow_metrics
            else:
                flow_metrics = self.flow_analyzer.compute_from_buffer(self.tick_buffer)
                if flow_metrics is not None:
                    self._latest_flow_metrics = flow_metrics
        
        liquidity_metrics = None
        if btc_ob and self.orderbook_buffer.deltas:
            liquidity_metrics = self.liquidity_analyzer.analyze(
                snapshot=btc_ob,
                buffer=self.orderbook_buffer,
            )
        
        queue_metrics = None
        if btc_ob:
            queue_metrics = self.queue_model.analyze(btc_ob, self.orderbook_buffer)
        
        spread_metrics = None
        if btc_ob:
            spread_metrics = self.spread_analyzer.analyze(
                btc_ob,
                last_tick=self._last_trade_tick,
            )
        
        # ---- Step 3: Regime detection ----
        # Volatility
        volatility_estimate = None
        log_return_sample = self._latest_log_return_sample()
        if log_return_sample is not None:
            price_return, elapsed_seconds = log_return_sample
            volatility_estimate = self.volatility_engine.update(
                price_return=float(price_return),
                timestamp=time.time(),
                elapsed_seconds=float(elapsed_seconds),
            )

        # Kalman filter
        kalman_state = None
        if self.tick_buffer.len() > 0:
            kalman_state = self.kalman_filter.update(
                price=btc_price,
                timestamp=time.time(),
            )

        regime_estimate = None
        if self.tick_buffer.len() > 50:
            prices = np.array(list(self.tick_buffer.prices)[-200:])
            volumes = np.array(list(self.tick_buffer.volumes)[-200:])
            
            # Build emission vector for HMM using the detector's declared schema:
            # [volatility, flow_imbalance, spread_bps, taker_ratio,
            #  volume_normalized, price_change, acceleration, depth_ratio].
            if orderbook_pressure and flow_metrics and volatility_estimate:
                positive_volumes = volumes[np.isfinite(volumes) & (volumes > 0.0)]
                baseline_volume = (
                    float(np.median(positive_volumes))
                    if len(positive_volumes)
                    else max(flow_metrics.avg_trade_size, 1.0e-12)
                )
                volume_normalized = float(np.clip(
                    flow_metrics.avg_trade_size / max(baseline_volume, 1.0e-12),
                    0.0,
                    8.0,
                ))
                latest_return = 0.0
                if len(prices) > 1 and prices[-2] > 0.0:
                    latest_return = float(np.log(prices[-1] / prices[-2]))
                depth_ratio = 1.0
                if liquidity_metrics is not None:
                    depth_ratio = float(np.clip(liquidity_metrics.depth_ratio, 0.05, 20.0))
                elif btc_ob and btc_ob.asks:
                    bid_depth = sum(level.size for level in btc_ob.bids[:5])
                    ask_depth = sum(level.size for level in btc_ob.asks[:5])
                    depth_ratio = float(np.clip(bid_depth / max(ask_depth, 1.0e-12), 0.05, 20.0))

                emission_vector = self.hmm_regime.build_observation_vector(
                    realized_vol=float(volatility_estimate.realized_volatility),
                    flow_imbalance=float(flow_metrics.flow_imbalance),
                    spread_bps=float(spread_metrics.spread_bps if spread_metrics else 100.0),
                    taker_ratio=float(flow_metrics.taker_ratio),
                    volume_normalized=volume_normalized,
                    price_change=latest_return,
                    acceleration=float(kalman_state.acceleration_estimate if kalman_state else 0.0),
                    depth_ratio=depth_ratio,
                )
                regime_estimate = self.hmm_regime.update(emission_vector, time.time())
            else:
                # Fallback: use simple regime classification
                regime_estimate = RegimeStateEstimate(
                    timestamp=time.time(),
                    current_regime=RegimeState.CALM_TRENDING,
                    regime_probabilities={RegimeState.CALM_TRENDING: 0.6, RegimeState.CALM_RANGE: 0.4},
                    regime_persistence=0.8,
                    regime_transition_probability={
                        RegimeState.CALM_TRENDING: 0.8,
                        RegimeState.CALM_RANGE: 0.2,
                    },
                    regime_duration_estimate=60.0,
                    regime_volatility_estimate=0.0001,
                    regime_flow_characteristic=0.0,
                    regime_confidence=0.6,
                    regime_entropy=0.5,
                )
            
            self.regime_buffer.add(regime_estimate)
        
        # ---- Step 4: Signal generation ----
        continuation_signal = None
        if kalman_state and regime_estimate and flow_metrics and orderbook_pressure and volatility_estimate:
            continuation_signal = self.continuation_engine.generate_signal(
                kalman_state=kalman_state,
                regime_estimate=regime_estimate,
                flow_metrics=flow_metrics,
                orderbook_pressure=orderbook_pressure,
                volatility_estimate=volatility_estimate,
                liquidity_metrics=liquidity_metrics,
            )
        
        reversal_signal = None
        if kalman_state and regime_estimate and flow_metrics and orderbook_pressure and volatility_estimate:
            reversal_signal = self.reversal_engine.generate_signal(
                kalman_state=kalman_state,
                regime_estimate=regime_estimate,
                flow_metrics=flow_metrics,
                orderbook_pressure=orderbook_pressure,
                volatility_estimate=volatility_estimate,
                liquidity_metrics=liquidity_metrics,
                queue_metrics=queue_metrics,
                spread_metrics=spread_metrics,
            )
        
        exhaustion_signal = None
        if kalman_state and flow_metrics and orderbook_pressure and volatility_estimate:
            exhaustion_signal = self.exhaustion_detector.detect(
                flow_metrics=flow_metrics,
                orderbook_pressure=orderbook_pressure,
                volatility_estimate=volatility_estimate,
                kalman_state=kalman_state,
                liquidity_metrics=liquidity_metrics,
                spread_metrics=spread_metrics,
            )
        
        burst_failure_signal = None
        if kalman_state and flow_metrics and volatility_estimate:
            burst_failure_signal = self.burst_detector.update(
                current_price=btc_price,
                kalman_state=kalman_state,
                flow_metrics=flow_metrics,
                volatility_estimate=volatility_estimate,
                liquidity_metrics=liquidity_metrics,
            )
        
        entropy_result = None
        if self.signal_buffer.len() > 10:
            recent_probs = self.signal_buffer.recent_probabilities(30)
            direction_prob = float(recent_probs[-1]) if len(recent_probs) else 0.5
            raw_signal_strength = float(np.clip(abs(direction_prob - 0.5) * 2.0, 0.0, 1.0))
            if continuation_signal is not None and continuation_signal.is_valid:
                raw_signal_strength = max(
                    raw_signal_strength,
                    float(continuation_signal.continuation_probability),
                )
            if reversal_signal is not None and reversal_signal.is_valid:
                raw_signal_strength = max(
                    raw_signal_strength,
                    float(reversal_signal.reversal_probability),
                )
            entropy_result = self.entropy_filter.filter_signal(
                raw_signal_strength=raw_signal_strength,
                direction_probability=direction_prob,
                direction=Direction.UP if direction_prob >= 0.5 else Direction.DOWN,
                continuation_signal=continuation_signal,
                reversal_signal=reversal_signal,
            )

        observation_signal = None
        if self._market_info and self._market_info.price_to_beat is not None:
            observation_signal = self.observation_strategy.evaluate(
                market_slug=self._market_info.slug,
                tick_buffer=self.tick_buffer,
                btc_price=btc_price,
                price_to_beat=self._market_info.price_to_beat,
                time_to_settlement=self._time_to_market_settlement(),
                flow_metrics=flow_metrics,
                orderbook_pressure=orderbook_pressure,
                spread_metrics=spread_metrics,
                liquidity_metrics=liquidity_metrics,
                volatility_estimate=volatility_estimate,
            )
            if observation_signal.is_ready:
                observation_state = "valid" if observation_signal.is_valid else "invalid"
                observation_event_key = (
                    f"{observation_signal.market_slug}:observation:{observation_state}"
                )
                if observation_event_key not in self._logged_observation_signal_keys:
                    self._logged_observation_signal_keys.add(observation_event_key)
                    logger.info(
                        "SIGNAL OBSERVATION: ready=%s valid=%s direction=%s "
                        "obs=%.1fs/%ss p_up=%.3f conf=%.2f val=%.2f reason=%s",
                        observation_signal.is_ready,
                        observation_signal.is_valid,
                        observation_signal.direction.value,
                        observation_signal.observed_seconds,
                        f"{observation_signal.required_seconds:.0f}",
                        observation_signal.p_up,
                        observation_signal.confidence,
                        observation_signal.validation_score,
                        observation_signal.reason,
                    )

        technical_signal = None
        if self._market_info and getattr(self.config, "technical_momentum_enabled", True):
            technical_signal = self.technical_momentum_strategy.evaluate(
                market_slug=self._market_info.slug,
                tick_buffer=self.tick_buffer,
                time_to_settlement=self._time_to_market_settlement(),
            )
        
        # ---- Step 5: Inference pipeline ----
        
        # Bayesian fusion
        posterior = None
        if kalman_state and regime_estimate and flow_metrics and orderbook_pressure and volatility_estimate:
            posterior = self.bayesian_fusion.fuse(
                kalman_state=kalman_state,
                regime_estimate=regime_estimate,
                flow_metrics=flow_metrics,
                orderbook_pressure=orderbook_pressure,
                volatility_estimate=volatility_estimate,
                continuation_signal=continuation_signal,
                reversal_signal=reversal_signal,
                spread_metrics=spread_metrics,
                liquidity_metrics=liquidity_metrics,
            )
        
        # Posterior calibration
        if posterior:
            posterior = self.posterior_calibrator.calibrate(posterior, time.time())
        
        # Hazard model
        hazard_estimate = None
        if kalman_state and flow_metrics and orderbook_pressure and regime_estimate and volatility_estimate and posterior:
            hazard_estimate = self.hazard_model.estimate(
                kalman_state=kalman_state,
                flow_metrics=flow_metrics,
                orderbook_pressure=orderbook_pressure,
                regime_estimate=regime_estimate,
                volatility_estimate=volatility_estimate,
                posterior=posterior,
            )
        
        # Multi-timescale aggregation
        multi_timescale_estimate = None
        time_to_settlement = self._time_to_market_settlement()
        if kalman_state and posterior and hazard_estimate and regime_estimate and volatility_estimate:
            multi_timescale_estimate = self.multi_timescale.aggregate(
                kalman_state=kalman_state,
                posterior=posterior,
                hazard_estimate=hazard_estimate,
                regime_estimate=regime_estimate,
                volatility_estimate=volatility_estimate,
                continuation_signal=continuation_signal,
                reversal_signal=reversal_signal,
                entropy_filter=entropy_result,
                time_to_settlement=time_to_settlement,
            )
        
        # Settlement forecast
        settlement_forecast = None
        if posterior and multi_timescale_estimate and hazard_estimate and kalman_state and regime_estimate and volatility_estimate:
            empirical_drift, empirical_sigma, empirical_sample_count = (
                self._estimate_terminal_log_moments()
            )
            round_direction_lock = None
            
            # Get market price for edge calculation
            market_price_up = None
            market_price_down = None
            if poly_ob_up:
                market_price_up = poly_ob_up.best_ask_price if poly_ob_up.asks else None
            if poly_ob_down:
                market_price_down = poly_ob_down.best_ask_price if poly_ob_down.asks else None

            price_to_beat = self._market_info.price_to_beat if self._market_info else None
            if price_to_beat is None:
                logger.debug("Skipping settlement forecast: missing price_to_beat")
            else:
                round_direction_lock = self._get_round_direction_lock(
                    market_slug=self._market_info.slug if self._market_info else "",
                    btc_price=btc_price,
                    price_to_beat=price_to_beat,
                    time_to_settlement=time_to_settlement,
                    observation_signal=observation_signal,
                    technical_signal=technical_signal,
                    empirical_sigma=empirical_sigma,
                    empirical_drift=empirical_drift,
                )
                settlement_forecast = self.settlement_forecaster.forecast(
                    posterior=posterior,
                    multi_timescale=multi_timescale_estimate,
                    hazard=hazard_estimate,
                    kalman=kalman_state,
                    regime_estimate=regime_estimate,
                    volatility_estimate=volatility_estimate,
                    continuation_signal=continuation_signal,
                    reversal_signal=reversal_signal,
                    entropy_filter=entropy_result,
                    market_price_up=market_price_up,
                    market_price_down=market_price_down,
                    time_to_settlement=time_to_settlement,
                    btc_price=btc_price,
                    price_to_beat=price_to_beat,
                    empirical_sigma_per_sqrt_second=empirical_sigma,
                    empirical_drift_per_second=empirical_drift,
                    empirical_sample_count=empirical_sample_count,
                    observation_signal=observation_signal,
                    technical_signal=technical_signal,
                    round_direction=(
                        round_direction_lock["direction"]
                        if round_direction_lock
                        else None
                    ),
                    round_direction_p_up=(
                        round_direction_lock["p_up"]
                        if round_direction_lock
                        else None
                    ),
                    round_direction_confidence=(
                        round_direction_lock["confidence"]
                        if round_direction_lock
                        else 0.0
                    ),
                    round_direction_reason=(
                        round_direction_lock["reason"]
                        if round_direction_lock
                        else ""
                    ),
                )
        
        # ---- Step 6: Execution modeling ----
        execution_estimate = None
        if settlement_forecast:
            execution_direction = settlement_forecast.execution_direction
            if execution_direction == Direction.NEUTRAL:
                execution_direction = settlement_forecast.expected_settlement_direction
            execution_estimate = self.execution_engine.estimate_execution(
                direction=execution_direction,
                size_fraction=settlement_forecast.execution_size_fraction,
                settlement_forecast=settlement_forecast,
                polymarket_orderbook=poly_ob_up if execution_direction == Direction.UP else poly_ob_down,
                btc_orderbook=btc_ob,
                queue_metrics=queue_metrics,
                spread_metrics=spread_metrics,
                liquidity_metrics=liquidity_metrics,
                regime_estimate=regime_estimate,
                volatility_estimate=volatility_estimate,
            )
        
        # ---- Step 7: Decision ----
        trade_decision = None
        risk_state = None
        if settlement_forecast and execution_estimate:
            # Get risk state
            risk_state = self.risk_manager.get_risk_state(
                regime_state=regime_estimate.current_regime if regime_estimate else None,
                volatility_info={"realized_volatility": volatility_estimate.realized_volatility if volatility_estimate else 0.0003},
            )
            
            trade_decision = self.decision_engine.make_decision(
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                risk_state=risk_state,
                regime_estimate=regime_estimate,
                volatility_estimate=volatility_estimate,
                multi_timescale=multi_timescale_estimate,
                kalman_state=kalman_state,
                orderbook_pressure=orderbook_pressure,
                flow_metrics=flow_metrics,
                continuation_signal=continuation_signal,
                reversal_signal=reversal_signal,
                exhaustion_signal=exhaustion_signal,
                burst_failure_signal=burst_failure_signal,
                entropy_filter=entropy_result,
                market_price_up=market_price_up if settlement_forecast else None,
                market_price_down=market_price_down if settlement_forecast else None,
                token_id_up=self._market_info.token_id_up if self._market_info else None,
                token_id_down=self._market_info.token_id_down if self._market_info else None,
                market_slug=self._market_info.slug if self._market_info else None,
                minimum_order_size=(
                    self._market_info.minimum_order_size
                    if self._market_info
                    else self.config.market.min_order_size
                ),
                minimum_tick_size=(
                    self._market_info.minimum_tick_size
                    if self._market_info
                    else self.config.market.price_tick
                ),
                is_dry_run=self.poly_config.dry_run,
            )

            self._log_entry_signal_event(
                market_slug=self._market_info.slug if self._market_info else "unknown",
                trade_decision=trade_decision,
            )
        
        # ---- Step 8: Execute trade ----
        if trade_decision and trade_decision.should_trade:
            self._total_order_attempts += 1
            order_result = None
            filled_position = False
            mode = "DRY_RUN" if self.poly_config.dry_run else "LIVE"
            order_slug = self._market_info.slug if self._market_info else "unknown"
            order_direction = trade_decision.direction.value
            if (
                bool(getattr(self.config, "one_entry_per_round", True))
                and self.position_tracker.has_open_position_for_market(order_slug)
            ):
                logger.info(
                    "ORDER SKIPPED: mode=%s direction=%s reason=round_position_already_open",
                    mode,
                    order_direction,
                )
                trade_decision.should_trade = False
                trade_decision.reason = (
                    f"{trade_decision.reason} | round_position_already_open({order_slug})"
                )
                self._total_suppressed += 1
            else:
                try:
                    order_result = await self.polymarket_client.place_order(
                        token_id=trade_decision.token_id,
                        side="BUY",
                        direction=trade_decision.direction,
                        size=trade_decision.position_size,
                        price=trade_decision.execution_estimate.limit_price,
                    )
                    order_result = await self._confirm_order_fill_if_needed(order_result)
                    status = self._order_result_status(order_result) or "submitted"
                    order_id = self._order_result_order_id(order_result) or "N/A"
                    logger.info(
                        "ORDER SUBMIT: mode=%s direction=%s size=%.2fUSDC "
                        "price=%.4f edge=%+.0fbps confidence=%.2f order_id=%s",
                        mode,
                        order_direction,
                        trade_decision.position_size,
                        trade_decision.execution_estimate.limit_price,
                        trade_decision.edge_bps,
                        trade_decision.confidence,
                        order_id,
                    )
                    logger.info(
                        "ORDER STATUS: mode=%s direction=%s status=%s order_id=%s",
                        mode,
                        order_direction,
                        status,
                        order_id,
                    )

                    filled_position = self._record_filled_position(
                        trade_decision=trade_decision,
                        order_result=order_result,
                    )
                    if not filled_position:
                        logger.warning(
                            "ORDER STATUS: mode=%s direction=%s status=not_opened "
                            "source_status=%s order_id=%s",
                            mode,
                            order_direction,
                            status,
                            self._order_result_order_id(order_result) or "N/A",
                        )
                except Exception as e:
                    logger.error(
                        "ORDER STATUS: mode=%s direction=%s status=failed error=%s",
                        mode,
                        order_direction,
                        e,
                    )

            if filled_position:
                risk_state = self.risk_manager.get_risk_state(
                    regime_state=regime_estimate.current_regime if regime_estimate else None,
                    volatility_info={
                        "realized_volatility": (
                            volatility_estimate.realized_volatility
                            if volatility_estimate
                            else 0.0003
                        )
                    },
                )
        else:
            self._total_suppressed += 1
        
        market_slug = self._market_info.slug if self._market_info else "unknown"
        market_question = self._market_info.question if self._market_info else ""
        market_end_iso = self._market_info.end_date_iso if self._market_info else ""
        price_to_beat = self._market_info.price_to_beat if self._market_info else None
        settlement_source = self.price_feed.get_current_settlement_price_source()
        aggregate_source = self.price_feed.get_current_price_source()
        price_source = (
            settlement_source
            if settlement_source == aggregate_source
            else f"{settlement_source}|agg:{aggregate_source}"
        )

        # ---- Step 9: Update state ----
        # Build market state for this cycle
        self._current_market_state = MarketState(
            timestamp=time.time(),
            btc_price=btc_price,
            market_slug=market_slug,
            market_question=market_question,
            market_end_iso=market_end_iso,
            price_to_beat=price_to_beat,
            price_source=price_source,
            btc_orderbook=btc_ob,
            polymarket_orderbook_up=poly_ob_up,
            polymarket_orderbook_down=poly_ob_down,
            orderbook_pressure=orderbook_pressure,
            flow_metrics=flow_metrics,
            liquidity_metrics=liquidity_metrics,
            queue_metrics=queue_metrics,
            spread_metrics=spread_metrics,
            regime_state=regime_estimate,
            volatility_estimate=volatility_estimate,
            continuation_signal=continuation_signal,
            reversal_signal=reversal_signal,
            exhaustion_signal=exhaustion_signal,
            burst_failure_signal=burst_failure_signal,
            entropy_filter=entropy_result,
            observation_signal=observation_signal,
            technical_signal=technical_signal,
            kalman_state=kalman_state,
            bayesian_posterior=posterior,
            hazard_estimate=hazard_estimate,
            multi_timescale=multi_timescale_estimate,
            settlement_forecast=settlement_forecast,
            execution_estimate=execution_estimate,
            risk_state=risk_state if risk_state else None,
            trade_decision=trade_decision,
        )
        self._last_tui_render_signature = None
        self._last_tui_render_at = 0.0
        
        # Update signal buffer
        if settlement_forecast:
            self.signal_buffer.add_signal(
                timestamp=time.time(),
                direction_prob=settlement_forecast.p_up_settlement,
                continuation=continuation_signal,
                reversal=reversal_signal,
            )
        
        cycle_time = time.time() - cycle_start
        self._last_cycle_time = cycle_time
        
        # Log cycle summary
        if trade_decision:
            price_to_beat_text = f"{price_to_beat:.2f}" if price_to_beat is not None else "N/A"
            forecast = trade_decision.settlement_forecast
            execution = trade_decision.execution_estimate
            btc_text = f"{btc_price:.2f}"
            up_ask_text = (
                f"{forecast.market_price_up:.3f}"
                if forecast.market_price_up is not None
                else "N/A"
            )
            down_ask_text = (
                f"{forecast.market_price_down:.3f}"
                if forecast.market_price_down is not None
                else "N/A"
            )
            obs_p_text = (
                f"{forecast.observation_p_up:.3f}"
                if forecast.observation_p_up is not None
                else "N/A"
            )
            tech_p_text = (
                f"{forecast.technical_p_up:.3f}"
                if forecast.technical_p_up is not None
                else "N/A"
            )
            round_direction_probability = None
            if forecast.round_direction_p_up is not None:
                round_direction_probability = (
                    forecast.round_direction_p_up
                    if forecast.round_direction == Direction.UP
                    else 1.0 - forecast.round_direction_p_up
                    if forecast.round_direction == Direction.DOWN
                    else None
                )
            round_p_text = (
                f"{round_direction_probability:.3f}"
                if round_direction_probability is not None
                else "N/A"
            )
            latest_position_snapshot = self._last_position_snapshot
            logger.debug(
                f"Cycle {self._cycle_count}: "
                f"market_slug={market_slug} "
                f"btc={btc_text} "
                f"btc_src={price_source} "
                f"ptb={price_to_beat_text} "
                f"tau={forecast.time_to_settlement_seconds:.0f}s "
                f"decision={trade_decision.should_trade} "
                f"direction={trade_decision.direction.value if trade_decision.direction else 'N/A'} "
                f"p_up={forecast.p_up_settlement:.3f} "
                f"p_down={forecast.p_down_settlement:.3f} "
                f"obs_ready={forecast.observation_ready} "
                f"obs_valid={forecast.observation_valid} "
                f"obs_s={forecast.observation_seconds:.1f} "
                f"obs_p={obs_p_text} "
                f"obs_conf={forecast.observation_confidence:.2f} "
                f"obs_val={forecast.observation_validation_score:.2f} "
                f"tech_valid={forecast.technical_valid} "
                f"tech_p={tech_p_text} "
                f"tech_conf={forecast.technical_confidence:.2f} "
                f"tech_cons={forecast.technical_consensus_score:.2f} "
                f"tech_tf={forecast.technical_dominant_timeframe or 'N/A'} "
                f"round_lock={forecast.round_direction_locked} "
                f"round_dir={forecast.round_direction.value} "
                f"round_p={round_p_text} "
                f"round_conf={forecast.round_direction_confidence:.2f} "
                f"p={trade_decision.probability:.3f} "
                f"market={trade_decision.market_price:.3f} "
                f"up_ask={up_ask_text} "
                f"down_ask={down_ask_text} "
                f"eff={execution.effective_price:.3f} "
                f"limit={execution.limit_price:.3f} "
                f"size={trade_decision.position_size:.2f} "
                f"open_pos={latest_position_snapshot.open_count} "
                f"uPnL={latest_position_snapshot.unrealized_pnl:+.2f} "
                f"depth={execution.visible_depth_notional:.2f} "
                f"edge={trade_decision.edge_bps:.0f}bps "
                f"z={forecast.terminal_z_score:.2f} "
                f"sigma={forecast.terminal_sigma_per_sqrt_second:.7f} "
                f"drift={forecast.terminal_drift_per_second:.8f} "
                f"confidence={trade_decision.confidence:.2f} "
                f"fill={trade_decision.fill_probability:.2f} "
                f"slip={trade_decision.expected_slippage_bps:.1f}bps "
                f"reason={trade_decision.reason} "
                f"time={cycle_time:.3f}s"
            )
        else:
            logger.debug(f"Cycle {self._cycle_count}: no decision (insufficient data)")

        self._update_tui()

    def _build_tui_state_snapshot(self) -> Optional[MarketState]:
        """Build a display-only snapshot with the freshest websocket cache values."""
        state = self._current_market_state
        now = time.time()
        market = self._market_info

        live_btc_price = self.price_feed.get_current_settlement_price()
        if state is None:
            if market is None and live_btc_price is None:
                return None
            state = MarketState(
                timestamp=now,
                btc_price=live_btc_price if live_btc_price is not None else 0.0,
            )

        btc_price = live_btc_price if live_btc_price is not None else state.btc_price

        settlement_source = self.price_feed.get_current_settlement_price_source()
        aggregate_source = self.price_feed.get_current_price_source()
        price_source = (
            settlement_source
            if settlement_source == aggregate_source
            else f"{settlement_source}|agg:{aggregate_source}"
        )
        if not price_source:
            price_source = state.price_source

        btc_orderbook = self.price_feed.get_current_orderbook() or state.btc_orderbook
        poly_ob_up = state.polymarket_orderbook_up
        poly_ob_down = state.polymarket_orderbook_down

        if market is not None:
            try:
                live_up = self.polymarket_client.get_cached_orderbook(market.token_id_up)
                live_down = self.polymarket_client.get_cached_orderbook(market.token_id_down)
                if live_up is not None:
                    poly_ob_up = live_up
                if live_down is not None:
                    poly_ob_down = live_down
            except Exception as exc:
                logger.debug("TUI orderbook snapshot failed: %s", exc)

        price_to_beat = (
            market.price_to_beat
            if market is not None and market.price_to_beat is not None
            else state.price_to_beat
        )
        market_slug = market.slug if market is not None and market.slug else state.market_slug
        market_question = (
            market.question
            if market is not None and market.question
            else state.market_question
        )
        market_end_iso = (
            market.end_date_iso
            if market is not None and market.end_date_iso
            else state.market_end_iso
        )

        up_ask = (
            poly_ob_up.best_ask
            if poly_ob_up is not None and poly_ob_up.best_ask is not None
            else None
        )
        down_ask = (
            poly_ob_down.best_ask
            if poly_ob_down is not None and poly_ob_down.best_ask is not None
            else None
        )

        forecast = state.settlement_forecast
        if forecast is not None:
            selected_market_price = forecast.selected_market_price
            if forecast.execution_direction == Direction.UP and up_ask is not None:
                selected_market_price = up_ask
            elif forecast.execution_direction == Direction.DOWN and down_ask is not None:
                selected_market_price = down_ask
            forecast = replace(
                forecast,
                timestamp=now,
                btc_price=btc_price,
                price_to_beat=price_to_beat,
                time_to_settlement_seconds=self._time_to_market_settlement(),
                market_price_up=up_ask if up_ask is not None else forecast.market_price_up,
                market_price_down=(
                    down_ask if down_ask is not None else forecast.market_price_down
                ),
                selected_market_price=selected_market_price,
            )

        execution_estimate = state.execution_estimate
        trade_decision = state.trade_decision
        if trade_decision is not None:
            decision_forecast = forecast or trade_decision.settlement_forecast
            selected_price = None
            if decision_forecast is not None:
                selected_price = decision_forecast.selected_market_price
            if selected_price is None:
                selected_price = trade_decision.market_price

            decision_kwargs: Dict[str, Any] = {
                "settlement_forecast": decision_forecast,
            }
            if selected_price is not None:
                edge = float(trade_decision.probability) - float(selected_price)
                decision_kwargs.update(
                    {
                        "market_price": float(selected_price),
                        "edge": edge,
                        "edge_bps": edge * 10000.0,
                    }
                )
                if trade_decision.execution_estimate is not None:
                    execution_estimate = replace(
                        trade_decision.execution_estimate,
                        timestamp=now,
                        best_ask_price=float(selected_price),
                    )
                    decision_kwargs["execution_estimate"] = execution_estimate
            trade_decision = replace(trade_decision, **decision_kwargs)

        return replace(
            state,
            timestamp=now,
            btc_price=btc_price,
            market_slug=market_slug,
            market_question=market_question,
            market_end_iso=market_end_iso,
            price_to_beat=price_to_beat,
            price_source=price_source,
            btc_orderbook=btc_orderbook,
            polymarket_orderbook_up=poly_ob_up,
            polymarket_orderbook_down=poly_ob_down,
            settlement_forecast=forecast,
            execution_estimate=execution_estimate,
            trade_decision=trade_decision,
        )

    def _update_tui(self, *, force: bool = False) -> None:
        """Refresh the optional Rich dashboard with the latest display snapshot."""
        if self.tui is None:
            return
        try:
            state = self._build_tui_state_snapshot()
            signature = self._tui_snapshot_signature(state)
            now = time.time()
            if not force:
                if signature == self._last_tui_render_signature:
                    return
                if now - self._last_tui_render_at < self._tui_min_render_interval:
                    return
            self.tui.update(
                state=state,
                cycle=self._cycle_count,
                last_cycle_time=self._last_cycle_time,
                session_stats=self.get_monitoring_stats(),
            )
            self._last_tui_render_signature = signature
            self._last_tui_render_at = now
        except Exception as exc:
            now = time.time()
            if now - self._last_tui_error_log >= 15.0:
                self._last_tui_error_log = now
                logger.warning("TUI update failed: %s", exc, exc_info=True)

    async def run(self, duration_seconds: Optional[float] = None) -> None:
        """
        Run the bot continuously.
        
        Args:
            duration_seconds: How long to run (None = run indefinitely)
        """
        logger.info("Starting bot run loop...")

        try:
            await self.initialize()
            start_time = time.time()

            while self._running:
                # Check duration limit
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    logger.info(f"Duration limit reached: {duration_seconds}s")
                    break
                
                # Run processing cycle. A stuck network call should not freeze the dashboard.
                try:
                    await asyncio.wait_for(
                        self.run_cycle(),
                        timeout=self._run_cycle_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "CYCLE TIMEOUT: cycle=%s timeout=%.1fs slug=%s tau=%.0fs",
                        self._cycle_count,
                        self._run_cycle_timeout,
                        self._market_info.slug if self._market_info else "unknown",
                        self._time_to_market_settlement(),
                    )
                    self._log_heartbeat_event()
                except Exception as exc:
                    logger.exception(
                        "CYCLE ERROR: cycle=%s slug=%s error=%s",
                        self._cycle_count,
                        self._market_info.slug if self._market_info else "unknown",
                        exc,
                    )
                    self._log_heartbeat_event()
                
                # Wait between cycles
                await asyncio.sleep(self._cycle_interval)
                
        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Bot stopped by user.")
        except Exception as e:
            logger.error(f"Bot error: {e}")
        finally:
            self._running = False
            await self.shutdown()

    
    async def shutdown(self) -> None:
        """
        Clean shutdown: disconnect feeds, log final stats.
        """
        logger.info("Shutting down bot...")
        self._running = False

        task = self._tui_refresh_task
        self._tui_refresh_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._last_tui_render_signature = None
        self._last_tui_render_at = 0.0

        task = self._rollover_prefetch_task
        self._rollover_prefetch_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        task = self._wallet_maintenance_task
        self._wallet_maintenance_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        task = self._ptb_refresh_task
        self._ptb_refresh_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        settlement_tasks = list(self._settlement_tasks.values())
        self._settlement_tasks.clear()
        for task in settlement_tasks:
            if task and not task.done():
                task.cancel()
        if settlement_tasks:
            await asyncio.gather(*settlement_tasks, return_exceptions=True)
        
        # Disconnect BTC price feed WebSockets
        try:
            await self.price_feed.disconnect()
        except Exception as e:
            logger.warning(f"Multi-exchange BTC feed disconnect error: {e}")

        try:
            await self.polymarket_client.stop()
        except Exception as e:
            logger.warning(f"Polymarket disconnect error: {e}")

        if self._owns_infra_engine:
            try:
                await self._infra_engine.stop()
            except Exception as e:
                logger.warning(f"Infrastructure engine stop error: {e}")

        try:
            await self.rest_fallback.stop()
        except Exception as e:
            logger.warning(f"REST fallback disconnect error: {e}")
        
        # Log final stats
        stats = self.get_stats()
        logger.info(
            f"Session summary: "
            f"cycles={stats['total_cycles']} "
            f"attempts={stats['total_order_attempts']} "
            f"trades={stats['total_trades']} "
            f"open_positions={stats['open_positions']} "
            f"suppressed={stats['total_suppressed']} "
            f"duration={stats['duration_seconds']:.0f}s"
        )
        
        # Log risk manager summary
        risk_summary = self.risk_manager.get_session_summary()
        logger.info(
            f"Risk summary: "
            f"capital={risk_summary['current_capital']:.2f}USDC "
            f"equity={risk_summary['portfolio_equity']:.2f}USDC "
            f"available={risk_summary['available_capital']:.2f}USDC "
            f"open_position={risk_summary['current_position']:.2f}USDC "
            f"unrealized={risk_summary['unrealized_pnl']:+.2f}USDC "
            f"pnl={risk_summary['realized_pnl']:+.2f}USDC "
            f"drawdown={risk_summary['max_drawdown']:.2%} "
            f"win_rate={risk_summary['win_rate']:.2%} "
            f"trades={risk_summary['total_trades']}"
        )

    def get_stats(self) -> dict:
        """Get bot performance statistics."""
        return {
            "total_cycles": self._total_cycles,
            "total_order_attempts": self._total_order_attempts,
            "total_trades": self._total_trades,
            "total_suppressed": self._total_suppressed,
            "open_positions": self._last_position_snapshot.open_count,
            "unrealized_pnl": self._last_position_snapshot.unrealized_pnl,
            "open_position_cost": self._last_position_snapshot.total_cost,
            "cycle_interval": self._cycle_interval,
            "duration_seconds": time.time() - self._start_time,
            "last_cycle_time": self._last_cycle_time,
        }

    def get_monitoring_stats(self) -> dict:
        """Return combined bot, risk, PnL, and round metrics for the TUI."""
        bot_stats = self.get_stats()
        risk_stats = self.risk_manager.get_session_summary()
        position_snapshot = self._last_position_snapshot
        market_slug = self._market_info.slug if self._market_info else ""
        price_to_beat = self._market_info.price_to_beat if self._market_info else None

        stats = dict(risk_stats)
        open_position_details = []
        awaiting_settlement_details = []
        for position in self.position_tracker.open_positions():
            detail = {
                "position_id": position.position_id,
                "market_slug": position.market_slug,
                "token_id": position.token_id,
                "direction": position.direction.value,
                "shares": position.shares,
                "cost_usdc": position.cost,
                "entry_fee_usdc": position.entry_fee,
                "entry_price": position.entry_price,
                "mark_price": position.last_mark_price,
                "market_value": position.notional_value,
                "unrealized_pnl": position.unrealized_pnl,
                "price_to_beat": position.price_to_beat,
                "entry_timestamp_utc": self._iso_timestamp(position.entry_timestamp),
                "market_end_utc": self._iso_timestamp(position.market_end_timestamp),
                "confidence_at_entry": position.confidence_at_entry,
                "edge_at_entry_bps": position.edge_at_entry_bps,
                "order_id": position.order_id,
                "is_dry_run": position.is_dry_run,
                "settlement_state": position.settlement_state,
            }
            if position.settlement_state == "awaiting_settlement":
                awaiting_settlement_details.append(detail)
            else:
                open_position_details.append(detail)

        stats.update(bot_stats)
        stats.update(
            {
                "market_slug": market_slug,
                "market_question": self._market_info.question if self._market_info else "",
                "market_end_iso": self._market_info.end_date_iso if self._market_info else "",
                "price_to_beat": price_to_beat,
                "open_positions": position_snapshot.open_count,
                "open_position_cost": position_snapshot.total_cost,
                "open_position_value": position_snapshot.market_value,
                "open_position_shares": position_snapshot.total_shares,
                "open_position_direction": position_snapshot.current_direction.value,
                "open_position_details": open_position_details,
                "awaiting_settlement_count": len(awaiting_settlement_details),
                "awaiting_settlement_details": awaiting_settlement_details,
                "unrealized_pnl": position_snapshot.unrealized_pnl,
                "bot_total_trades": self._total_trades,
                "settled_trades": risk_stats.get("total_trades", 0),
                "order_attempts": self._total_order_attempts,
                "suppressed_cycles": self._total_suppressed,
                "cycle_count": self._cycle_count,
                "trade_history_file": self._trade_history_file,
                "recent_trade_history": self._trade_history_events[-8:],
                "wallet_status": self._last_wallet_status,
                "claim_candidates": len(self._claim_candidates),
            }
        )

        realized = float(stats.get("realized_pnl", 0.0))
        unrealized = float(stats.get("unrealized_pnl", 0.0))
        initial = float(stats.get("initial_capital", 0.0))
        stats["total_pnl"] = realized + unrealized
        stats["portfolio_equity"] = float(stats.get("current_capital", initial)) + unrealized
        stats["total_return"] = (
            stats["total_pnl"] / initial
            if initial > 0.0
            else 0.0
        )
        return stats
