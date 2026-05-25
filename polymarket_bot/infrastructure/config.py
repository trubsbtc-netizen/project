from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

from polymarket_bot.infrastructure.types import _env_bool, _env_float, _env_int


def default_polymarket_host() -> str:
    return os.environ.get("POLYMARKET_HOST", "https://clob.polymarket.com").rstrip("/")


def load_dotenv(path: str | None = None) -> None:
    env_path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class InfrastructureConfig:
    polymarket_host: str = field(default_factory=default_polymarket_host)
    polymarket_ws: str = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
    gamma_api: str = "https://gamma-api.polymarket.com"
    polygon_rpc_urls: Tuple[str, ...] = (
        "https://polygon-bor-rpc.publicnode.com",
        "https://polygon.drpc.org",
    )
    chain_id: int = 137

    private_key: str = ""
    funder_address: str = ""
    signature_type: int = 3
    dry_run: bool = False

    relayer_url: str = "https://relayer-v2.polymarket.com"
    builder_api_key: str = ""
    builder_secret: str = ""
    builder_passphrase: str = ""
    deposit_wallet_address: str = ""
    wallet_email_mode: bool = False

    pusd_address: str = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
    ctf_address: str = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    ctf_collateral_adapter_address: str = "0xAdA100Db00Ca00073811820692005400218FcE1f"
    neg_risk_ctf_collateral_adapter_address: str = "0xadA2005600Dec949baf300f4C6120000bDB6eAab"

    market_slug_prefix: str = "btc-updown-5m-"
    market_search_tag: str = "btc-5-min"
    market_poll_interval_s: float = 5.0
    market_prewarm_count: int = 2
    market_prewarm_horizon_s: float = 900.0

    http_timeout_s: float = 10.0
    http_limit: int = 64
    http_limit_per_host: int = 16
    ws_ping_interval_s: float = 10.0
    ws_ping_timeout_s: float = 10.0
    ws_reconnect_initial_s: float = 1.0
    ws_reconnect_max_s: float = 30.0
    price_feed_stale_s: float = 15.0
    orderbook_stale_s: float = 20.0
    chainlink_stale_s: float = 90.0

    settlement_poll_timeout_s: float = 600.0
    settlement_poll_interval_s: float = 4.0
    settlement_confirmations_required: int = 2
    auto_claim: bool = False

    @classmethod
    def from_env(cls) -> "InfrastructureConfig":
        rpc_raw = os.environ.get("POLYGON_RPC_URLS", os.environ.get("POLYGON_RPC_URL", ""))
        rpc_urls = tuple(
            part.strip().rstrip("/")
            for part in rpc_raw.split(",")
            if part.strip()
        ) or cls.polygon_rpc_urls
        return cls(
            polymarket_host=default_polymarket_host(),
            polymarket_ws=os.environ.get("POLYMARKET_WS", cls.polymarket_ws),
            gamma_api=os.environ.get("GAMMA_API", cls.gamma_api).rstrip("/"),
            polygon_rpc_urls=rpc_urls,
            chain_id=_env_int("CHAIN_ID", 137),
            private_key=os.environ.get("POLYMARKET_PRIVATE_KEY", ""),
            funder_address=os.environ.get("POLYMARKET_FUNDER", ""),
            signature_type=_env_int("POLYMARKET_SIG_TYPE", 3),
            dry_run=_env_bool("DRY_RUN", False),
            relayer_url=os.environ.get("RELAYER_URL", cls.relayer_url),
            builder_api_key=os.environ.get("BUILDER_API_KEY", ""),
            builder_secret=os.environ.get("BUILDER_SECRET", ""),
            builder_passphrase=os.environ.get("BUILDER_PASS_PHRASE", ""),
            deposit_wallet_address=os.environ.get("DEPOSIT_WALLET", ""),
            wallet_email_mode=_env_bool("POLYMARKET_WALLET_EMAIL_MODE", False),
            pusd_address=os.environ.get("PUSD_ADDRESS", cls.pusd_address),
            ctf_address=os.environ.get("CTF_ADDRESS", cls.ctf_address),
            ctf_collateral_adapter_address=os.environ.get(
                "CTF_COLLATERAL_ADAPTER_ADDRESS",
                cls.ctf_collateral_adapter_address,
            ),
            neg_risk_ctf_collateral_adapter_address=os.environ.get(
                "NEG_RISK_CTF_COLLATERAL_ADAPTER_ADDRESS",
                cls.neg_risk_ctf_collateral_adapter_address,
            ),
            market_slug_prefix=os.environ.get("MARKET_SLUG_PREFIX", cls.market_slug_prefix),
            market_search_tag=os.environ.get("MARKET_SEARCH_TAG", cls.market_search_tag),
            market_poll_interval_s=_env_float("MARKET_POLL_INTERVAL_S", cls.market_poll_interval_s),
            market_prewarm_count=max(1, _env_int("MARKET_PREWARM_COUNT", cls.market_prewarm_count)),
            market_prewarm_horizon_s=_env_float("MARKET_PREWARM_HORIZON_S", cls.market_prewarm_horizon_s),
            http_timeout_s=_env_float("HTTP_TIMEOUT_S", cls.http_timeout_s),
            http_limit=max(1, _env_int("HTTP_LIMIT", cls.http_limit)),
            http_limit_per_host=max(1, _env_int("HTTP_LIMIT_PER_HOST", cls.http_limit_per_host)),
            ws_ping_interval_s=_env_float("WS_PING_INTERVAL_S", cls.ws_ping_interval_s),
            ws_ping_timeout_s=_env_float("WS_PING_TIMEOUT_S", cls.ws_ping_timeout_s),
            ws_reconnect_initial_s=_env_float("WS_RECONNECT_INITIAL_S", cls.ws_reconnect_initial_s),
            ws_reconnect_max_s=_env_float("WS_RECONNECT_MAX_S", cls.ws_reconnect_max_s),
            price_feed_stale_s=_env_float("PRICE_FEED_STALE_S", cls.price_feed_stale_s),
            orderbook_stale_s=_env_float("ORDERBOOK_STALE_S", cls.orderbook_stale_s),
            chainlink_stale_s=_env_float("CHAINLINK_STALE_S", cls.chainlink_stale_s),
            settlement_poll_timeout_s=_env_float("SETTLEMENT_POLL_TIMEOUT_S", cls.settlement_poll_timeout_s),
            settlement_poll_interval_s=_env_float("SETTLEMENT_POLL_INTERVAL_S", cls.settlement_poll_interval_s),
            settlement_confirmations_required=max(1, _env_int("SETTLEMENT_CONFIRMATIONS_REQUIRED", 2)),
            auto_claim=_env_bool("AUTO_CLAIM", False),
        )
