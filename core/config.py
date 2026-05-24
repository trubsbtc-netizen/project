from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class Env:
    def __init__(self, path: str | Path = ".env") -> None:
        self._file_values = _parse_env_file(Path(path))

    def get(self, key: str, default: str = "") -> str:
        value = os.environ.get(key)
        if value is not None:
            return value
        return self._file_values.get(key, default)

    def bool(self, key: str, default: bool = False) -> bool:
        raw = self.get(key, str(default)).strip().lower()
        return raw in {"1", "true", "t", "yes", "y", "on"}

    def int(self, key: str, default: int) -> int:
        raw = self.get(key, str(default)).strip()
        return int(raw) if raw else default

    def float(self, key: str, default: float) -> float:
        raw = self.get(key, str(default)).strip()
        return float(raw) if raw else default


@dataclass(slots=True, frozen=True)
class WebsocketConfig:
    polymarket_market_ws: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    polymarket_user_ws: str = "wss://ws-subscriptions-clob.polymarket.com/ws/user"
    polymarket_rtds_ws: str = "wss://ws-live-data.polymarket.com"
    binance_ws: str = (
        "wss://data-stream.binance.vision/stream"
        "?streams=btcusdt@trade/btcusdt@bookTicker/btcusdt@depth5@100ms"
    )
    coinbase_ws: str = "wss://ws-feed.exchange.coinbase.com"
    coinbase_use_doh: bool = True
    doh_resolver_url: str = "https://cloudflare-dns.com/dns-query"
    doh_timeout_s: float = 3.0
    connect_timeout_s: float = 8.0
    heartbeat_s: float = 5.0
    stale_feed_s: float = 2.0
    max_backoff_s: float = 20.0
    inbound_queue_size: int = 8192


@dataclass(slots=True, frozen=True)
class PolymarketConfig:
    web_base_url: str = "https://polymarket.com"
    gamma_base_url: str = "https://gamma-api.polymarket.com"
    clob_host: str = "https://clob.polymarket.com"
    chain_id: int = 137
    signature_type: int = 0
    private_key: str = ""
    funder: str = ""
    builder_api_key: str = ""
    builder_secret: str = ""
    builder_passphrase: str = ""
    relayer_url: str = "https://relayer-v2.polymarket.com"
    deposit_wallet: str = ""
    static_condition_id: str = ""
    static_market_id: str = ""
    static_up_token_id: str = ""
    static_down_token_id: str = ""
    static_price_to_beat: float = 0.0
    ptb_mode: str = "gamma"
    allow_cold_http_metadata: bool = False
    startup_bootstrap_timeout_s: float = 3.0

    def __repr__(self) -> str:
        return (
            "PolymarketConfig("
            f"web_base_url={self.web_base_url!r}, gamma_base_url={self.gamma_base_url!r}, "
            f"clob_host={self.clob_host!r}, "
            f"chain_id={self.chain_id}, signature_type={self.signature_type}, "
            f"funder={'set' if self.funder else 'empty'}, "
            f"private_key={'set' if self.private_key else 'empty'}, "
            f"builder_api_key={'set' if self.builder_api_key else 'empty'}, "
            f"deposit_wallet={'set' if self.deposit_wallet else 'empty'}, "
            f"ptb_mode={self.ptb_mode!r}, allow_cold_http_metadata={self.allow_cold_http_metadata}, "
            f"startup_bootstrap_timeout_s={self.startup_bootstrap_timeout_s})"
        )


@dataclass(slots=True, frozen=True)
class ContractConfig:
    polygon_ws_rpc: str = ""
    polygon_http_rpc: str = ""
    allow_http_rpc: bool = False
    ctf_exchange: str = "0xE111180000d2663C0091e4f400237545B87B996B"
    neg_risk_ctf_exchange: str = "0xe2222d279d744050d28e00520010520000310F59"
    conditional_tokens: str = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    pusd: str = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
    ctf_collateral_adapter: str = "0xAdA100Db00Ca00073811820692005400218FcE1f"
    neg_risk_ctf_collateral_adapter: str = "0xadA2005600Dec949baf300f4C6120000bDB6eAab"
    uma_adapter: str = "0x6A9D222616C90FcA5754cd1333cFD9b7fb6a4F74"


@dataclass(slots=True, frozen=True)
class StrategyConfig:
    bucket_seconds: int = 300
    min_seconds_to_expiry: float = 4.0
    max_seconds_after_open: float = 294.0
    min_confidence: float = 0.58
    min_edge: float = 0.025
    max_posterior_uncertainty: float = 0.42
    max_entropy: float = 0.985
    stale_suppression_weight: float = 0.65
    vol_halflife_s: float = 24.0
    flow_halflife_s: float = 8.0
    regime_halflife_s: float = 35.0
    covariance_halflife_s: float = 60.0
    max_model_drift_bps_per_s: float = 18.0


@dataclass(slots=True, frozen=True)
class ExecutionConfig:
    dry_run: bool = True
    enable_live_trading: bool = False
    order_size_usd: float = 5.0
    max_notional_usd: float = 20.0
    max_slippage_bps: float = 120.0
    order_ttl_ms: int = 900
    min_fill_probability: float = 0.52
    latency_budget_ms: float = 350.0
    min_quote_size: float = 1.0


@dataclass(slots=True, frozen=True)
class SettlementConfig:
    auto_claim: bool = False
    poll_timeout_s: float = 120.0
    poll_interval_s: float = 4.0
    confirmations_required: int = 1


@dataclass(slots=True, frozen=True)
class BotConfig:
    websockets: WebsocketConfig
    polymarket: PolymarketConfig
    contracts: ContractConfig
    strategy: StrategyConfig
    execution: ExecutionConfig
    settlement: SettlementConfig


def load_config(env_path: str | Path = ".env") -> BotConfig:
    env = Env(env_path)
    ws_defaults = WebsocketConfig()
    contract_defaults = ContractConfig()
    websockets = WebsocketConfig(
        polymarket_market_ws=env.get("POLYMARKET_MARKET_WS", ws_defaults.polymarket_market_ws),
        polymarket_user_ws=env.get("POLYMARKET_USER_WS", ws_defaults.polymarket_user_ws),
        polymarket_rtds_ws=env.get("POLYMARKET_RTDS_WS", ws_defaults.polymarket_rtds_ws),
        binance_ws=env.get("BINANCE_WS", ws_defaults.binance_ws),
        coinbase_ws=env.get("COINBASE_WS", ws_defaults.coinbase_ws),
        coinbase_use_doh=env.bool("COINBASE_USE_DOH", ws_defaults.coinbase_use_doh),
        doh_resolver_url=env.get("DOH_RESOLVER_URL", ws_defaults.doh_resolver_url),
        doh_timeout_s=env.float("DOH_TIMEOUT_S", ws_defaults.doh_timeout_s),
        connect_timeout_s=env.float("WS_CONNECT_TIMEOUT_S", 8.0),
        heartbeat_s=env.float("WS_HEARTBEAT_S", 5.0),
        stale_feed_s=env.float("STALE_FEED_S", 2.0),
        max_backoff_s=env.float("WS_MAX_BACKOFF_S", 20.0),
        inbound_queue_size=env.int("INBOUND_QUEUE_SIZE", 8192),
    )
    polymarket = PolymarketConfig(
        web_base_url=env.get("POLYMARKET_WEB_URL", "https://polymarket.com"),
        gamma_base_url=env.get("GAMMA_BASE_URL", "https://gamma-api.polymarket.com"),
        clob_host=env.get("CLOB_HOST", "https://clob.polymarket.com"),
        chain_id=env.int("CHAIN_ID", 137),
        signature_type=env.int("POLYMARKET_SIG_TYPE", 0),
        private_key=env.get("POLYMARKET_PRIVATE_KEY", ""),
        funder=env.get("POLYMARKET_FUNDER", ""),
        builder_api_key=env.get("BUILDER_API_KEY", ""),
        builder_secret=env.get("BUILDER_SECRET", ""),
        builder_passphrase=env.get("BUILDER_PASS_PHRASE", ""),
        relayer_url=env.get("RELAYER_URL", "https://relayer-v2.polymarket.com"),
        deposit_wallet=env.get("DEPOSIT_WALLET", ""),
        static_condition_id=env.get("PTB_CONDITION_ID", ""),
        static_market_id=env.get("PTB_MARKET_ID", ""),
        static_up_token_id=env.get("PTB_UP_TOKEN_ID", ""),
        static_down_token_id=env.get("PTB_DOWN_TOKEN_ID", ""),
        static_price_to_beat=env.float("PTB_PRICE_TO_BEAT", 0.0),
        ptb_mode=env.get("PTB_MODE", "gamma").lower(),
        allow_cold_http_metadata=env.bool("ALLOW_COLD_HTTP_METADATA", False),
        startup_bootstrap_timeout_s=env.float("PTB_STARTUP_BOOTSTRAP_TIMEOUT_S", 3.0),
    )
    contracts = ContractConfig(
        polygon_ws_rpc=env.get("POLYGON_WS_RPC", ""),
        polygon_http_rpc=env.get("POLYGON_HTTP_RPC", ""),
        allow_http_rpc=env.bool("ALLOW_CONTROL_PLANE_HTTP", False),
        ctf_exchange=env.get("CTF_EXCHANGE_ADDRESS", contract_defaults.ctf_exchange),
        neg_risk_ctf_exchange=env.get("NEG_RISK_CTF_EXCHANGE_ADDRESS", contract_defaults.neg_risk_ctf_exchange),
        conditional_tokens=env.get("CONDITIONAL_TOKENS_ADDRESS", contract_defaults.conditional_tokens),
        pusd=env.get("PUSD_ADDRESS", contract_defaults.pusd),
        ctf_collateral_adapter=env.get("CTF_COLLATERAL_ADAPTER_ADDRESS", contract_defaults.ctf_collateral_adapter),
        neg_risk_ctf_collateral_adapter=env.get(
            "NEG_RISK_CTF_COLLATERAL_ADAPTER_ADDRESS",
            contract_defaults.neg_risk_ctf_collateral_adapter,
        ),
        uma_adapter=env.get("UMA_ADAPTER_ADDRESS", contract_defaults.uma_adapter),
    )
    strategy = StrategyConfig(
        min_seconds_to_expiry=env.float("MIN_SECONDS_TO_EXPIRY", 4.0),
        min_confidence=env.float("MIN_CONFIDENCE", 0.58),
        min_edge=env.float("MIN_EDGE", 0.025),
        max_posterior_uncertainty=env.float("MAX_POSTERIOR_UNCERTAINTY", 0.42),
        max_entropy=env.float("MAX_ENTROPY", 0.985),
    )
    execution = ExecutionConfig(
        dry_run=env.bool("DRY_RUN", True),
        enable_live_trading=env.bool("ENABLE_LIVE_TRADING", False),
        order_size_usd=env.float("ORDER_SIZE_USD", 5.0),
        max_notional_usd=env.float("MAX_NOTIONAL_USD", 20.0),
        max_slippage_bps=env.float("MAX_SLIPPAGE_BPS", 120.0),
        order_ttl_ms=env.int("ORDER_TTL_MS", 900),
        min_fill_probability=env.float("MIN_FILL_PROBABILITY", 0.52),
        latency_budget_ms=env.float("LATENCY_BUDGET_MS", 350.0),
        min_quote_size=env.float("MIN_QUOTE_SIZE", 1.0),
    )
    settlement = SettlementConfig(
        auto_claim=env.bool("AUTO_CLAIM", False),
        poll_timeout_s=env.float("SETTLEMENT_POLL_TIMEOUT_S", 120.0),
        poll_interval_s=env.float("SETTLEMENT_POLL_INTERVAL_S", 4.0),
        confirmations_required=env.int("SETTLEMENT_CONFIRMATIONS_REQUIRED", 1),
    )
    return BotConfig(websockets, polymarket, contracts, strategy, execution, settlement)
