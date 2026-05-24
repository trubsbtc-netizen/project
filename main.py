"""
Entry point for the Polymarket BTC 5-minute directional trading bot.

Usage:
	python -m polymarket_bot.main
	python main.py

Environment:
	Requires a .env file with Polymarket authentication credentials.
	See README for setup instructions.

Modes:
	Dry-run (default): Simulates trading, logs all decisions, no real orders.
	Live: Set DRY_RUN=false in .env to execute real Polymarket orders.

The bot runs continuously, processing BTC price data from Binance and Coinbase,
analyzing orderbook microstructure, detecting market regimes,
generating directional signals, and producing trade decisions
for Polymarket BTC 5-minute UP/DOWN markets.

Quick start:
	1. Copy .env.example to .env and fill in credentials
	2. pip install -r requirements.txt
	3. python main.py
"""

import os
import sys
import asyncio
import logging
import argparse
import warnings
from pathlib import Path
from dataclasses import replace

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from polymarket_bot.config import BotConfig
from polymarket_bot.bot import TradingBot
from polymarket_bot.infrastructure.lifecycle import LifecycleEngine


def setup_logging(config: BotConfig, console: bool = True) -> logging.Logger:
	"""Configure logging for the bot."""
	log_format = "%(asctime)s | %(levelname)-7s | %(name)-25s | %(message)s"
	date_format = "%Y-%m-%d %H:%M:%S"

	root_logger = logging.getLogger()
	root_logger.setLevel(getattr(logging, config.log_level, logging.INFO))

	for handler in list(root_logger.handlers):
		root_logger.removeHandler(handler)
		try:
			handler.close()
		except Exception:
			pass

	formatter = logging.Formatter(log_format, datefmt=date_format)

	if console:
		stream_handler = logging.StreamHandler(sys.stdout)
		stream_handler.setFormatter(formatter)
		root_logger.addHandler(stream_handler)

	# File handler if log file is configured
	if config.log_file:
		file_handler = logging.FileHandler(config.log_file)
		file_handler.setFormatter(formatter)
		root_logger.addHandler(file_handler)

	# Suppress noisy third-party loggers
	logging.getLogger("websockets").setLevel(logging.WARNING)
	logging.getLogger("asyncio").setLevel(logging.WARNING)
	logging.getLogger("aiohttp").setLevel(logging.WARNING)
	logging.getLogger("urllib3").setLevel(logging.WARNING)

	return logging.getLogger(__name__)


async def main():
	"""Main entry point for the trading bot."""
	parser = argparse.ArgumentParser(
		description="Polymarket BTC 5-Minute Directional Trading Bot",
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python main.py                        # Dry-run mode (default)
  python main.py --duration 3600        # Run for 1 hour
  python main.py --log-level DEBUG      # Verbose logging
  python main.py --no-tui               # Plain console logs
  python main.py --no-dry-run           # LIVE TRADING (use with caution!)
		""",
	)
	parser.add_argument(
		"--duration",
		type=float,
		default=None,
		help="Run duration in seconds (default: run indefinitely)",
	)
	parser.add_argument(
		"--log-level",
		type=str,
		default=None,
		choices=["DEBUG", "INFO", "WARNING", "ERROR"],
		help="Override log level from .env config",
	)
	parser.add_argument(
		"--no-dry-run",
		action="store_true",
		help="Execute real trades on Polymarket (overrides DRY_RUN env var)",
	)
	parser.add_argument(
		"--initial-capital",
		type=float,
		default=1000.0,
		help="Initial capital for risk management tracking (default: 1000 USDC)",
	)
	parser.add_argument(
		"--no-tui",
		action="store_true",
		help="Use plain console logs instead of the Rich live trading dashboard",
	)

	args = parser.parse_args()

	# Load configuration from environment
	config = BotConfig()

	# Override log level if specified
	if args.log_level:
		config.log_level = args.log_level

	# Override dry run if specified
	if args.no_dry_run:
		logger = logging.getLogger(__name__)
		logger.warning("=" * 60)
		logger.warning("LIVE TRADING MODE ACTIVATED")
		logger.warning("Real orders will be placed on Polymarket CLOB!")
		logger.warning("=" * 60)

		# Confirm intent to trade live
		print("\n⚠️  LIVE TRADING MODE: Real orders will be placed on Polymarket.")
		confirm = input("Type 'YES' to confirm: ")
		if confirm != "YES":
			print("Aborted.")
			sys.exit(0)
		config.polymarket = replace(config.polymarket, dry_run=False)

	tui = None
	tui_error = None
	if not args.no_tui:
		try:
			from polymarket_bot.tui import ConsoleTUI

			tui = ConsoleTUI(
				mode="DRY RUN" if config.polymarket.dry_run else "LIVE",
				strategy=f"{config.strategy_name} v{config.strategy_version}",
			)
		except Exception as exc:
			tui_error = exc

	# Setup logging. With TUI active, the old one-line cycle spam is kept in
	# the file log while the terminal is reserved for the live dashboard.
	logger = setup_logging(config, console=tui is None)
	if tui is not None:
		logging.getLogger().addHandler(tui.get_log_handler())
		tui.start()
	elif tui_error is not None:
		logger.warning("Rich TUI unavailable, falling back to plain logs: %s", tui_error)

	logger.info("=" * 60)
	logger.info(f"Polymarket BTC 5-Minute Directional Bot v{config.strategy_version}")
	logger.info(f"Strategy: {config.strategy_name}")
	logger.info(f"Mode: {'DRY RUN' if config.polymarket.dry_run else 'LIVE TRADING'}")
	logger.info(f"Log Level: {config.log_level}")
	logger.info("=" * 60)

	# Display configuration summary
	logger.info("Configuration Summary:")
	logger.info(
		f"  BTC Price Source: {config.btc_price_source} "
		f"({config.btc_price_symbol}, {config.coinbase_product_id})"
	)
	logger.info(f"  Polymarket API: {config.polymarket.clob_api_url}")
	logger.info(f"  Max Position: {config.risk.max_position_notional} USDC")
	logger.info(f"  Kelly Fraction: {config.risk.position_sizing_kelly_fraction}")
	logger.info(f"  Max Drawdown: {config.risk.max_drawdown_fraction:.0%}")
	logger.info(f"  Settlement Buffer: {config.risk.settlement_time_buffer_seconds}s")
	logger.info(f"  Min Edge: {config.risk.min_edge_bps} bps")
	logger.info(
		f"  Observation Window: {config.observation_window_seconds:.0f}s "
		f"(required={config.require_observation_window})"
	)
	logger.info(f"  HMM Regimes: {config.hmm.n_regimes}")
	logger.info(f"  Kalman State Dim: {config.kalman.state_dim}")
	logger.info("=" * 60)

	# Create and run bot with infrastructure LifecycleEngine
	# The LifecycleEngine manages RTDS Chainlink WS, immutable PTB,
	# deterministic market rotation, and authoritative settlement.
	# ALL trading/alpha/signal logic remains in TradingBot.run_cycle().
	bot = TradingBot(config, initial_capital=args.initial_capital, tui=tui)
	engine = LifecycleEngine(bot)

	try:
		if args.duration:
			logger.info(f"Running for {args.duration} seconds...")
			await engine.start(duration_seconds=args.duration)
		else:
			logger.info("Running indefinitely (Ctrl+C to stop)...")
			await engine.start()

	except KeyboardInterrupt:
		logger.info("Bot stopped by user.")
	except Exception as e:
		logger.error(f"Fatal error: {e}", exc_info=True)
	finally:
		logger.info("Bot shutdown complete.")
		if tui is not None:
			tui.stop()


if __name__ == "__main__":
	# Reconfigure stdout/stderr to use UTF-8 on Windows to prevent UnicodeEncodeError
	# when rendering box-drawing characters and other symbols in the TUI.
	if sys.platform.startswith("win"):
		try:
			sys.stdout.reconfigure(encoding="utf-8")
			sys.stderr.reconfigure(encoding="utf-8")
		except Exception:
			pass

	# On Windows, Proactor can emit noisy ConnectionResetError callbacks when
	# remote WebSockets close during shutdown. Selector loop handles aiohttp WS
	# cleanup more quietly for this bot.
	if sys.platform.startswith("win"):
		with warnings.catch_warnings():
			warnings.filterwarnings(
				"ignore",
				category=DeprecationWarning,
			)
			selector_policy = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
			if selector_policy is not None:
				asyncio.set_event_loop_policy(selector_policy())

	# Load .env file if python-dotenv is available
	try:
		from dotenv import load_dotenv
		load_dotenv()
	except ImportError:
		pass  # dotenv not installed, expect env vars to be set manually

	try:
		asyncio.run(main())
	except KeyboardInterrupt:
		pass
