"""Lifecycle engine: wraps TradingBot with BTC_POLY infrastructure orchestration."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from typing import Any, Optional

from polymarket_bot.infrastructure.config import InfrastructureConfig
from polymarket_bot.infrastructure.engine import InfrastructureEngine

logger = logging.getLogger(__name__)


class LifecycleEngine:
    """Manages the bot lifecycle: init → run → shutdown.

    Integrates BTC_POLY InfrastructureEngine with TradingBot's strategy pipeline.
    The infrastructure engine provides real-time market data, deterministic PTB,
    authoritative settlement, and component health monitoring. TradingBot owns
    all strategy logic (alpha, signals, execution, risk).
    """

    def __init__(self, bot: Any) -> None:
        self.bot = bot
        self._infra_config = InfrastructureConfig.from_env()
        self._infra_engine = InfrastructureEngine(self._infra_config)
        self._running = False

    async def start(self, duration_seconds: Optional[float] = None) -> None:
        """Initialize, run the trading loop, then shutdown."""
        self._running = True
        start_time = time.time()

        try:
            self._wire_infra()
            # 1. Start infrastructure engine (feeds, orderbook, discovery, wallet)
            try:
                await self._infra_engine.start()
                logger.info("Infrastructure engine started")
            except Exception as exc:
                logger.warning("Infrastructure engine start failed: %s", exc)

            # 2. Initialize bot (discovers market, connects price feeds)
            try:
                await self.bot.initialize()
            except Exception as exc:
                logger.error("Bot initialization failed: %s", exc)
                return

            # 3. Main trading loop
            cycle_count = 0
            while self._running:
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    logger.info("Duration limit reached: %.0fs", duration_seconds)
                    break

                try:
                    timeout = getattr(self.bot, '_run_cycle_timeout', 60.0)
                    await asyncio.wait_for(self.bot.run_cycle(), timeout=timeout)
                    cycle_count += 1
                except asyncio.TimeoutError:
                    logger.warning("Cycle timeout: cycle=%d", cycle_count)
                except Exception as exc:
                    logger.exception("Cycle error: cycle=%d error=%s", cycle_count, exc)

                from polymarket_bot.infrastructure.types import ServiceState
                await asyncio.sleep(getattr(self.bot, '_cycle_interval', 1.0))

        except (KeyboardInterrupt, asyncio.CancelledError):
            logger.info("Bot stopped by user")
        except Exception as exc:
            logger.error("Fatal lifecycle error: %s", exc)
        finally:
            await self._stop()

    async def _stop(self) -> None:
        self._running = False
        try:
            await self.bot.shutdown()
        except Exception as exc:
            logger.warning("Bot shutdown error: %s", exc)
        try:
            await self._infra_engine.stop()
        except Exception as exc:
            logger.warning("Infrastructure engine stop error: %s", exc)

    def _wire_infra(self) -> None:
        """Set infrastructure references on the bot for legacy code paths."""
        bot = self.bot
        engine = self._infra_engine
        attach = getattr(bot, "attach_infrastructure_engine", None)
        if attach is not None:
            attach(engine, owns_engine=False)
            return
        try:
            bot._infra_ptb = engine.ptb_lifecycle
        except AttributeError:
            pass
        try:
            bot._infra_roller = engine.discovery
        except AttributeError:
            pass
        try:
            bot._infra_settlement = engine.settlement
        except AttributeError:
            pass
