from __future__ import annotations

import asyncio
import time

import pytest

from core.config import ExecutionConfig, PolymarketConfig, StrategyConfig, load_config
from core.microstructure.features import MicrostructureEngine
from core.probability.fusion import BayesianEvidenceFusion
from core.ptb.lifecycle import PTBMetadataUnavailable, PTBProvider, deterministic_slug
from core.rtds.chainlink import parse_chainlink_price
from core.runtime.health import HealthMonitor
from core.types import (
    MicrostructureSnapshot,
    OrderSide,
    Outcome,
    OutcomeToken,
    PTBMarket,
    PriceTick,
    PriceToBeat,
    SignalSource,
    TopOfBook,
    TradeTick,
    TruthSource,
)
from core.websocket.exchange_feeds import parse_binance, parse_coinbase
from core.websocket.polymarket import MarketSubscriptionState, parse_polymarket_market_payload


def test_deterministic_slug_formula() -> None:
    bucket = int(time.time() // 300) * 300
    assert deterministic_slug(bucket) == f"btc-updown-5m-{bucket}"


def test_config_defaults_are_values_not_slot_descriptors(tmp_path) -> None:
    config = load_config(tmp_path / "missing.env")
    assert isinstance(config.websockets.polymarket_market_ws, str)
    assert config.websockets.polymarket_market_ws.startswith("wss://")
    assert config.websockets.binance_ws.startswith("wss://data-stream.binance.vision/")
    assert config.websockets.coinbase_use_doh is True
    assert isinstance(config.contracts.ctf_exchange, str)
    assert config.contracts.ctf_exchange.startswith("0x")


def _ptb_market(bucket: int) -> PTBMarket:
    return PTBMarket(
        bucket=bucket,
        slug=deterministic_slug(bucket),
        condition_id="0xcondition",
        market_id="market",
        up=OutcomeToken(Outcome.UP, "up-token"),
        down=OutcomeToken(Outcome.DOWN, "down-token"),
        price_to_beat=PriceToBeat(TruthSource.PTB, 100.0, bucket * 1000),
        open_ts=bucket,
        close_ts=bucket + 300,
        fetched_mono_ns=time.monotonic_ns(),
        raw_hash="hash",
    )


def test_startup_bootstrap_fetches_once_when_runtime_http_disabled(monkeypatch) -> None:
    async def scenario() -> None:
        now = time.time()
        bucket = int(now // 300) * 300
        provider = PTBProvider(
            PolymarketConfig(
                allow_cold_http_metadata=False,
                startup_bootstrap_timeout_s=0.25,
            )
        )
        calls: list[tuple[int, float]] = []

        def fake_fetch(fetch_bucket: int, timeout_s: float) -> PTBMarket:
            calls.append((fetch_bucket, timeout_s))
            return _ptb_market(fetch_bucket)

        monkeypatch.setattr(provider, "_gamma_fetch", fake_fetch)
        first = await provider.bootstrap_startup_round(now)
        second = await provider.bootstrap_startup_round(now)
        cached = await provider.get_round(now)
        assert first is second is cached
        assert calls == [(bucket, 0.25)]

    asyncio.run(scenario())


def test_runtime_get_round_does_not_fetch_when_cold_http_disabled(monkeypatch) -> None:
    async def scenario() -> None:
        provider = PTBProvider(PolymarketConfig(allow_cold_http_metadata=False))

        def fake_fetch(fetch_bucket: int, timeout_s: float) -> PTBMarket:
            raise AssertionError("runtime path must not fetch HTTP")

        monkeypatch.setattr(provider, "_gamma_fetch", fake_fetch)
        with pytest.raises(PTBMetadataUnavailable):
            await provider.get_round(time.time())

    asyncio.run(scenario())


def test_gamma_parser_accepts_clob_token_ids_json_string(monkeypatch) -> None:
    provider = PTBProvider(PolymarketConfig())
    payload = {
        "id": "event",
        "startTime": "2026-05-23T16:00:00Z",
        "markets": [
            {
                "id": "market",
                "conditionId": "0xcondition",
                "question": "Bitcoin Up or Down",
                "outcomes": '["Up","Down"]',
                "clobTokenIds": '["up-token","down-token"]',
            }
        ],
    }
    metadata = provider._parse_gamma_metadata(1_779_552_000, "btc-updown-5m-1779552000", payload)
    assert metadata.token_ids == ("up-token", "down-token")
    assert metadata.event_start_time == "2026-05-23T16:00:00Z"


def test_gamma_fetch_uses_crypto_open_price_when_gamma_has_no_ptb(monkeypatch) -> None:
    provider = PTBProvider(PolymarketConfig())
    bucket = 1_779_552_000

    def fake_fetch_metadata(fetch_bucket: int, timeout_s: float):
        return provider._parse_gamma_metadata(
            fetch_bucket,
            deterministic_slug(fetch_bucket),
            {
                "id": "event",
                "startTime": "2026-05-23T16:00:00Z",
                "markets": [
                    {
                        "id": "market",
                        "conditionId": "0xcondition",
                        "question": "Bitcoin Up or Down",
                        "outcomes": '["Up","Down"]',
                        "clobTokenIds": '["up-token","down-token"]',
                    }
                ],
            },
        )

    def fake_open_price(metadata, timeout_s: float):
        return 75432.05, {"openPrice": 75432.05}

    monkeypatch.setattr(provider, "_fetch_gamma_metadata", fake_fetch_metadata)
    monkeypatch.setattr(provider, "_fetch_crypto_open_price", fake_open_price)
    market = provider._gamma_fetch(bucket, 0.25)
    assert market.token_ids == ("up-token", "down-token")
    assert market.price_to_beat.value == 75432.05


def test_exchange_parsers() -> None:
    recv = time.monotonic_ns()
    b = parse_binance({"data": {"e": "bookTicker", "b": "100", "a": "101", "B": "2", "A": "3", "E": 1}}, recv)
    c = parse_coinbase(
        {"type": "ticker", "best_bid": "100", "best_ask": "101", "best_bid_size": "2", "best_ask_size": "3"},
        recv,
    )
    assert isinstance(b, TopOfBook)
    assert isinstance(c, TopOfBook)
    assert b.mid == c.mid == 100.5


def test_chainlink_rtds_parser_update_and_snapshot() -> None:
    recv = time.monotonic_ns()
    update = parse_chainlink_price(
        {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "payload": {"symbol": "btc/usd", "timestamp": 1779554030000, "value": 75410.18},
        },
        recv,
    )
    snapshot = parse_chainlink_price(
        {
            "payload": {
                "data": [
                    {"timestamp": 1779554029000, "value": 75409.0},
                    {"timestamp": 1779554030000, "value": 75410.0},
                ]
            }
        },
        recv,
    )
    assert update is not None
    assert update.price == 75410.18
    assert snapshot is not None
    assert snapshot.price == 75410.0


def test_polymarket_book_parser() -> None:
    state = MarketSubscriptionState()
    recv = time.monotonic_ns()
    events = parse_polymarket_market_payload(
        {
            "event_type": "book",
            "asset_id": "up",
            "bids": [{"price": "0.45", "size": "10"}],
            "asks": [{"price": "0.46", "size": "11"}],
        },
        recv,
        state,
    )
    assert len(events) == 1
    assert isinstance(events[0], TopOfBook)


def test_probability_engine_bounds() -> None:
    fusion = BayesianEvidenceFusion(StrategyConfig())
    snap = MicrostructureSnapshot(
        ts_mono_ns=time.monotonic_ns(),
        seconds_to_expiry=120.0,
        truth_price=100.5,
        price_to_beat=100.0,
        signed_distance=0.5,
        signal_mid=100.5,
        exchange_divergence=0.0,
        realized_vol=0.0004,
        jump_intensity=0.1,
        drift=0.00002,
        drift_uncertainty=0.0004,
        taker_aggression=0.25,
        flow_acceleration=0.1,
        book_pressure=0.1,
        spoof_resistant_imbalance=0.15,
        queue_survival=0.8,
        liquidity_stability=0.7,
        spread_stability=0.8,
        absorption_up=0.0,
        absorption_down=0.1,
        exhaustion=0.1,
        burst_failure=0.0,
        liquidity_vacuum=0.1,
        entropy=0.6,
        regime_trend=0.1,
        regime_volatility=0.1,
        stale_penalty=0.0,
    )
    estimate = fusion.estimate(snap)
    assert 0.0 < estimate.p_up < 1.0
    assert 0.0 <= estimate.confidence <= 1.0
    assert 0.0 <= estimate.posterior_uncertainty <= 1.0


def test_microstructure_snapshot_generation() -> None:
    health = HealthMonitor()
    engine = MicrostructureEngine(StrategyConfig(), health)
    now = time.monotonic_ns()
    engine.update_truth_price(PriceTick(TruthSource.RTDS_CHAINLINK, "BTC/USD", 100.0, 0, now))
    engine.update_truth_price(PriceTick(TruthSource.RTDS_CHAINLINK, "BTC/USD", 100.1, 0, now + 1_000_000_000))
    engine.books.update_exchange(TopOfBook(SignalSource.BINANCE, "BTCUSDT", 100.0, 100.2, 5.0, 4.0, 0, now))
    engine.books.update_exchange(TopOfBook(SignalSource.COINBASE, "BTC-USD", 100.0, 100.2, 5.0, 4.0, 0, now))
    engine.update_trade(TradeTick(SignalSource.BINANCE, "BTCUSDT", 100.1, 0.25, OrderSide.BUY, 0, now))
    snap = engine.snapshot(now + 1_000_000_000, 120.0, 100.0)
    assert snap is not None
    assert snap.truth_price == 100.1
    assert snap.realized_vol > 0.0
