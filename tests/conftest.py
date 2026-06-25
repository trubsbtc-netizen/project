"""Shared test fixtures for the Polymarket BTC bot test suite."""

from __future__ import annotations

import asyncio
from decimal import Decimal
from unittest.mock import AsyncMock

import pytest

from config.settings import BotConfig, RiskConfig, SignalConfig, TradingConfig
from core.types import (
    CircuitState,
    Direction,
    OrderbookState,
    Position,
    RiskState,
    Side,
    SignalStrength,
    UnifiedSignal,
)
from risk.engine import RiskEngine


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        max_position_usdc=Decimal("50"),
        max_total_exposure_usdc=Decimal("150"),
        max_daily_loss_usdc=Decimal("75"),
        max_consecutive_losses=4,
        max_drawdown_pct=Decimal("0.20"),
        max_trades_per_minute=3,
        max_trades_per_cycle=1,
        kelly_fraction=Decimal("0.25"),
        max_kelly_fraction=Decimal("0.20"),
        circuit_breaker_cooldown=300,
    )


@pytest.fixture
def risk_engine(risk_config: RiskConfig) -> RiskEngine:
    return RiskEngine(config=risk_config, initial_balance_usdc=Decimal("1000"))


def make_signal(
    *,
    direction: Direction = Direction.UP,
    strength: SignalStrength = SignalStrength.STRONG,
    confidence: Decimal = Decimal("0.75"),
    up_probability: Decimal = Decimal("0.65"),
    liquidity_ok: bool = True,
    spread_ok: bool = True,
    manipulation_flag: bool = False,
    liquidity_vacuum: bool = False,
    violent_reversal: bool = False,
) -> UnifiedSignal:
    return UnifiedSignal(
        up_asset_id="up-token-abc",
        down_asset_id="down-token-xyz",
        up_microprice=Decimal("0.55"),
        down_microprice=Decimal("0.45"),
        up_probability=up_probability,
        down_probability=Decimal("1") - up_probability,
        confidence=confidence,
        direction=direction,
        signal_strength=strength,
        microprice_score=Decimal("0.3"),
        ofi_score=Decimal("0.2"),
        depth_ratio_score=Decimal("0.1"),
        velocity_score=Decimal("0.15"),
        absorption_score=Decimal("0.05"),
        up_spread=Decimal("0.02"),
        down_spread=Decimal("0.03"),
        liquidity_ok=liquidity_ok,
        spread_ok=spread_ok,
        manipulation_flag=manipulation_flag,
        liquidity_vacuum=liquidity_vacuum,
        violent_reversal=violent_reversal,
        timestamp=1700000000000,
    )


@pytest.fixture
def tradeable_signal() -> UnifiedSignal:
    return make_signal()


@pytest.fixture
def untradeable_signal() -> UnifiedSignal:
    return make_signal(strength=SignalStrength.WEAK, confidence=Decimal("0.40"))


def make_orderbook_state(
    *,
    bids: dict[Decimal, Decimal] | None = None,
    asks: dict[Decimal, Decimal] | None = None,
    asset_id: str = "asset-123",
) -> OrderbookState:
    if bids is None:
        bids = {
            Decimal("0.50"): Decimal("100"),
            Decimal("0.49"): Decimal("200"),
            Decimal("0.48"): Decimal("150"),
        }
    if asks is None:
        asks = {
            Decimal("0.52"): Decimal("100"),
            Decimal("0.53"): Decimal("200"),
            Decimal("0.54"): Decimal("150"),
        }
    return OrderbookState(
        asset_id=asset_id,
        market="test-market",
        bids=bids,
        asks=asks,
        timestamp=1700000000000,
        hash="abc123",
    )


def make_position(
    *,
    asset_id: str = "asset-123",
    direction: Direction = Direction.UP,
    entry_price: Decimal = Decimal("0.55"),
    size: Decimal = Decimal("50"),
    cost_basis: Decimal = Decimal("27.50"),
    fee_paid: Decimal = Decimal("0.86"),
) -> Position:
    return Position(
        asset_id=asset_id,
        direction=direction,
        side=Side.BUY,
        entry_price=entry_price,
        size=size,
        cost_basis=cost_basis,
        fee_paid=fee_paid,
        order_id="order-abc",
        entered_at=1700000000000,
        condition_id="condition-xyz",
        market_end_time=int(__import__("time").time()) + 300,
    )
