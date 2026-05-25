"""
Configuration system with full validation and secure environment loading.
All sensitive values loaded exclusively from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration is invalid or incomplete."""
    pass


@dataclass
class NetworkConfig:
    clob_host:           str = "https://clob.polymarket.com"
    gamma_api_host:      str = "https://gamma-api.polymarket.com"
    data_api_host:       str = "https://data-api.polymarket.com"
    polygon_rpc_url:     str = "https://api.zan.top/node/v1/polygon/mainnet/c0100d1f93234184bf1aaeb86f7f4537"
    ws_market_endpoint:  str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    ws_user_endpoint:    str = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
    connect_timeout_s:   float = 10.0
    read_timeout_s:      float = 30.0
    tcp_keepalive:       bool = True
    tcp_nodelay:         bool = True
    max_connections:     int = 20


@dataclass
class AuthConfig:
    private_key:         str = ""     # 0x-prefixed hex private key
    api_key:             str = ""     # L2 API key (UUID)
    api_secret:          str = ""     # L2 API secret (base64)
    api_passphrase:      str = ""     # L2 API passphrase
    funder_address:      str = ""     # Deposit wallet funder address
    signature_type:      int = 3      # POLY_1271 for new API users
    chain_id:            int = 137    # Polygon mainnet
    auto_create_api_key: bool = True
    enable_user_ws:      bool = True
    enable_heartbeat:    bool = True

    def validate(self) -> None:
        if not self.private_key:
            raise ConfigError("PRIVATE_KEY not set")
        if not self.private_key.startswith("0x"):
            raise ConfigError("PRIVATE_KEY must start with 0x")
        if not self.auto_create_api_key:
            if not self.api_key:
                raise ConfigError("API_KEY not set")
            if not self.api_secret:
                raise ConfigError("API_SECRET not set")
            if not self.api_passphrase:
                raise ConfigError("API_PASSPHRASE not set")
        if not self.funder_address:
            raise ConfigError("FUNDER_ADDRESS not set")
        if not self.funder_address.startswith("0x"):
            raise ConfigError("FUNDER_ADDRESS must be a hex address starting with 0x")


@dataclass
class RiskConfig:
    max_position_usdc:        Decimal = Decimal("50")
    max_total_exposure_usdc:  Decimal = Decimal("150")
    max_daily_loss_usdc:      Decimal = Decimal("75")
    max_consecutive_losses:   int     = 4
    max_drawdown_pct:         Decimal = Decimal("0.20")
    max_trades_per_minute:    int     = 3
    max_trades_per_cycle:     int     = 1
    kelly_fraction:           Decimal = Decimal("0.25")
    max_kelly_fraction:       Decimal = Decimal("0.20")
    circuit_breaker_cooldown: int     = 300   # seconds
    volatility_pause_s:       int     = 60

    def validate(self) -> None:
        if self.max_position_usdc <= 0:
            raise ConfigError("max_position_usdc must be > 0")
        if self.max_total_exposure_usdc < self.max_position_usdc:
            raise ConfigError("max_total_exposure_usdc must be >= max_position_usdc")
        if self.max_daily_loss_usdc <= 0:
            raise ConfigError("max_daily_loss_usdc must be > 0")
        if self.max_consecutive_losses < 1:
            raise ConfigError("max_consecutive_losses must be >= 1")
        if self.kelly_fraction <= 0 or self.kelly_fraction > 1:
            raise ConfigError("kelly_fraction must be in (0, 1]")
        if self.max_kelly_fraction <= 0 or self.max_kelly_fraction > 1:
            raise ConfigError("max_kelly_fraction must be in (0, 1]")


@dataclass
class SignalConfig:
    min_ev_threshold:             Decimal = Decimal("0.025")
    min_confidence:               Decimal = Decimal("0.65")
    ofi_signal_threshold:         Decimal = Decimal("0.25")
    depth_ratio_threshold:        Decimal = Decimal("0.20")
    max_tradeable_spread:         Decimal = Decimal("0.06")
    microprice_signal_deviation:  Decimal = Decimal("0.04")
    min_book_depth_usdc:          Decimal = Decimal("200")
    min_price_velocity:           Decimal = Decimal("0.002")
    spoof_disappear_ms:           int     = 2000
    spoof_size_threshold:         Decimal = Decimal("0.15")
    absorption_window_s:          float   = 10.0
    sweep_confirm_count:          int     = 2


@dataclass
class TradingConfig:
    min_order_size_usdc:     Decimal = Decimal("5")
    max_price_slippage:      Decimal = Decimal("0.03")
    default_tick_size:       str     = "0.01"
    market_near_expiry_s:    int     = 60
    market_cooldown_s:       int     = 3
    heartbeat_interval_s:    int     = 5
    gtd_security_buffer_s:   int     = 60
    order_timeout_s:         int     = 10   # Cancel order if not filled in Ns
    # Market slug patterns to search for BTC UP/DOWN 5M markets
    market_slug_patterns:    list    = field(default_factory=lambda: [
        "btc-up-or-down",
        "btc-updown-5m",
        "will-btc-go-up-or-down",
    ])


@dataclass
class BotConfig:
    network:  NetworkConfig  = field(default_factory=NetworkConfig)
    auth:     AuthConfig     = field(default_factory=AuthConfig)
    risk:     RiskConfig     = field(default_factory=RiskConfig)
    signal:   SignalConfig   = field(default_factory=SignalConfig)
    trading:  TradingConfig  = field(default_factory=TradingConfig)
    dry_run:  bool           = False    # If True: log signals but do NOT execute trades
    log_level: str           = "INFO"
    metrics_port: int        = 9090
    signal_log_interval_s: float = 5.0
    log_all_signals: bool = False

    def validate(self) -> None:
        self.auth.validate()
        self.risk.validate()
        logger.info("Configuration validated successfully", extra={
            "dry_run": self.dry_run,
            "max_position_usdc": str(self.risk.max_position_usdc),
        })


def load_config() -> BotConfig:
    """
    Load configuration exclusively from environment variables.
    All sensitive credentials MUST be provided via environment.
    Raises ConfigError if any required value is missing or invalid.
    """
    config = BotConfig(
        auth=AuthConfig(
            private_key=_require_env("POLY_PRIVATE_KEY"),
            api_key=os.getenv("POLY_API_KEY", ""),
            api_secret=os.getenv("POLY_API_SECRET", ""),
            api_passphrase=os.getenv("POLY_API_PASSPHRASE", ""),
            funder_address=_require_env("POLY_FUNDER_ADDRESS"),
            signature_type=int(os.getenv("POLY_SIGNATURE_TYPE", "3")),
            chain_id=int(os.getenv("POLY_CHAIN_ID", "137")),
            auto_create_api_key=os.getenv("POLY_AUTO_CREATE_API_KEY", "true").lower() == "true",
            enable_user_ws=os.getenv("ENABLE_USER_WS", "true").lower() == "true",
            enable_heartbeat=os.getenv("ENABLE_HEARTBEAT", "true").lower() == "true",
        ),
        risk=RiskConfig(
            max_position_usdc=Decimal(os.getenv("MAX_POSITION_USDC", "50")),
            max_total_exposure_usdc=Decimal(os.getenv("MAX_TOTAL_EXPOSURE_USDC", "150")),
            max_daily_loss_usdc=Decimal(os.getenv("MAX_DAILY_LOSS_USDC", "75")),
            max_consecutive_losses=int(os.getenv("MAX_CONSECUTIVE_LOSSES", "4")),
            max_drawdown_pct=Decimal(os.getenv("MAX_DRAWDOWN_PCT", "0.20")),
            max_trades_per_minute=int(os.getenv("MAX_TRADES_PER_MINUTE", "3")),
            max_trades_per_cycle=int(os.getenv("MAX_TRADES_PER_CYCLE", "1")),
            kelly_fraction=Decimal(os.getenv("KELLY_FRACTION", "0.25")),
            max_kelly_fraction=Decimal(os.getenv("MAX_KELLY_FRACTION", "0.20")),
        ),
        signal=SignalConfig(
            min_ev_threshold=Decimal(os.getenv("MIN_EV_THRESHOLD", "0.025")),
            min_confidence=Decimal(os.getenv("MIN_CONFIDENCE", "0.65")),
            max_tradeable_spread=Decimal(os.getenv("MAX_TRADEABLE_SPREAD", "0.06")),
            min_book_depth_usdc=Decimal(os.getenv("MIN_BOOK_DEPTH_USDC", "200")),
        ),
        trading=TradingConfig(
            min_order_size_usdc=Decimal(os.getenv("MIN_ORDER_SIZE_USDC", "5")),
            max_price_slippage=Decimal(os.getenv("MAX_PRICE_SLIPPAGE", "0.03")),
        ),
        dry_run=os.getenv("DRY_RUN", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        metrics_port=int(os.getenv("METRICS_PORT", "9090")),
        signal_log_interval_s=float(os.getenv("SIGNAL_LOG_INTERVAL_S", "5")),
        log_all_signals=os.getenv("LOG_ALL_SIGNALS", "false").lower() == "true",
    )
    config.network.polygon_rpc_url = os.getenv(
        "POLYGON_RPC_URL",
        config.network.polygon_rpc_url,
    )

    config.validate()
    return config


def _require_env(key: str) -> str:
    """Load required environment variable or raise ConfigError."""
    value = os.getenv(key)
    if not value:
        raise ConfigError(
            f"Required environment variable '{key}' is not set. "
            f"Copy .env.example to .env and fill in all values."
        )
    return value
