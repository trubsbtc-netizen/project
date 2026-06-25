"""Tests for execution/engine.py — fee computation, Kelly sizing, resolution handling."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from config.settings import BotConfig, RiskConfig, TradingConfig
from core.constants import CRYPTO_TAKER_FEE_RATE
from core.types import Direction, Position, RiskState, Side, TradeStatus
from execution.engine import ExecutionEngine
from risk.engine import RiskEngine


def _make_bot_config(**overrides) -> BotConfig:
    from config.settings import AuthConfig
    cfg = BotConfig(
        auth=AuthConfig(
            private_key="0xdeadbeef",
            funder_address="0x1234abcd",
            auto_create_api_key=True,
        ),
        dry_run=True,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def _make_engine(
    risk_state: RiskState | None = None,
    positions: dict | None = None,
    **config_overrides,
) -> ExecutionEngine:
    client = MagicMock()
    risk = MagicMock(spec=RiskEngine)
    risk.get_state = AsyncMock(
        return_value=risk_state or RiskState(current_equity_usdc=Decimal("1000"))
    )
    risk.get_positions = AsyncMock(return_value=positions or {})
    risk.check_entry = AsyncMock(return_value=Decimal("25"))
    risk.record_entry = AsyncMock()
    risk.record_exit = AsyncMock()

    cfg = _make_bot_config(**config_overrides)
    return ExecutionEngine(client=client, risk_engine=risk, config=cfg)


# ──────────────────────────── _compute_fee ────────────────────────────


class TestComputeFee:
    def test_standard_fee(self) -> None:
        eng = _make_engine()
        fee = eng._compute_fee(Decimal("100"), Decimal("0.55"))
        expected = Decimal(str(CRYPTO_TAKER_FEE_RATE)) * Decimal("100") * Decimal("0.55") * Decimal("0.45")
        assert fee == expected

    def test_fee_at_price_0_5(self) -> None:
        eng = _make_engine()
        fee = eng._compute_fee(Decimal("100"), Decimal("0.50"))
        # 0.07 * 100 * 0.50 * 0.50 = 1.75
        expected = Decimal(str(CRYPTO_TAKER_FEE_RATE)) * Decimal("100") * Decimal("0.25")
        assert fee == expected

    def test_fee_near_zero_price(self) -> None:
        eng = _make_engine()
        fee = eng._compute_fee(Decimal("100"), Decimal("0.01"))
        # Very small fee near 0
        assert fee > Decimal("0")
        assert fee < Decimal("1")

    def test_fee_near_one_price(self) -> None:
        eng = _make_engine()
        fee = eng._compute_fee(Decimal("100"), Decimal("0.99"))
        assert fee > Decimal("0")
        assert fee < Decimal("1")

    def test_zero_shares(self) -> None:
        eng = _make_engine()
        fee = eng._compute_fee(Decimal("0"), Decimal("0.50"))
        assert fee == Decimal("0")

    def test_fee_symmetry(self) -> None:
        eng = _make_engine()
        fee_at_30 = eng._compute_fee(Decimal("100"), Decimal("0.30"))
        fee_at_70 = eng._compute_fee(Decimal("100"), Decimal("0.70"))
        # p*(1-p) is symmetric: 0.3*0.7 == 0.7*0.3
        assert fee_at_30 == fee_at_70


# ──────────────────────────── _compute_kelly_size_usdc ────────────────────────────


class TestKellySizing:
    @pytest.mark.asyncio
    async def test_positive_edge(self) -> None:
        eng = _make_engine()
        signal = MagicMock()
        signal.up_probability = Decimal("0.70")
        signal.confidence = Decimal("0.80")

        size = await eng._compute_kelly_size_usdc(
            signal=signal,
            direction=Direction.UP,
            entry_price=Decimal("0.55"),
        )
        assert size > Decimal("0")

    @pytest.mark.asyncio
    async def test_no_edge(self) -> None:
        eng = _make_engine()
        signal = MagicMock()
        signal.up_probability = Decimal("0.40")
        signal.confidence = Decimal("0.50")

        size = await eng._compute_kelly_size_usdc(
            signal=signal,
            direction=Direction.UP,
            entry_price=Decimal("0.55"),
        )
        assert size == Decimal("0")

    @pytest.mark.asyncio
    async def test_boundary_price_zero(self) -> None:
        eng = _make_engine()
        signal = MagicMock()
        signal.up_probability = Decimal("0.70")
        signal.confidence = Decimal("0.80")

        size = await eng._compute_kelly_size_usdc(
            signal=signal,
            direction=Direction.UP,
            entry_price=Decimal("0"),
        )
        assert size == Decimal("0")

    @pytest.mark.asyncio
    async def test_boundary_price_one(self) -> None:
        eng = _make_engine()
        signal = MagicMock()
        signal.up_probability = Decimal("0.90")
        signal.confidence = Decimal("0.90")

        size = await eng._compute_kelly_size_usdc(
            signal=signal,
            direction=Direction.UP,
            entry_price=Decimal("1"),
        )
        assert size == Decimal("0")

    @pytest.mark.asyncio
    async def test_capped_by_max_position(self) -> None:
        eng = _make_engine()
        signal = MagicMock()
        signal.up_probability = Decimal("0.95")
        signal.confidence = Decimal("0.95")

        size = await eng._compute_kelly_size_usdc(
            signal=signal,
            direction=Direction.UP,
            entry_price=Decimal("0.10"),
        )
        # Should be capped at max_position_usdc (50)
        assert size <= Decimal("50")

    @pytest.mark.asyncio
    async def test_down_direction(self) -> None:
        eng = _make_engine()
        signal = MagicMock()
        signal.up_probability = Decimal("0.30")
        signal.down_probability = Decimal("0.70")
        signal.confidence = Decimal("0.80")

        size = await eng._compute_kelly_size_usdc(
            signal=signal,
            direction=Direction.DOWN,
            entry_price=Decimal("0.55"),
        )
        assert size > Decimal("0")


# ──────────────────────────── handle_market_resolution ────────────────────────────


class TestMarketResolution:
    @pytest.mark.asyncio
    async def test_winning_position(self) -> None:
        eng = _make_engine()
        pos = Position(
            asset_id="win-token",
            direction=Direction.UP,
            side=Side.BUY,
            entry_price=Decimal("0.55"),
            size=Decimal("50"),
            cost_basis=Decimal("27.50"),
            fee_paid=Decimal("0.86"),
            order_id="ord-1",
            entered_at=1700000000000,
            condition_id="cond-1",
            market_end_time=1700000300,
        )
        await eng.handle_market_resolution(
            condition_id="cond-1",
            winning_asset_id="win-token",
            positions={"win-token": pos},
        )
        eng._risk.record_exit.assert_called_once()
        call_kwargs = eng._risk.record_exit.call_args[1]
        assert call_kwargs["won"] is True
        pnl = call_kwargs["pnl_usdc"]
        # pnl = 50 - 27.50 - 0.86 = 21.64
        assert pnl == Decimal("21.64")

    @pytest.mark.asyncio
    async def test_losing_position(self) -> None:
        eng = _make_engine()
        pos = Position(
            asset_id="lose-token",
            direction=Direction.UP,
            side=Side.BUY,
            entry_price=Decimal("0.55"),
            size=Decimal("50"),
            cost_basis=Decimal("27.50"),
            fee_paid=Decimal("0.86"),
            order_id="ord-2",
            entered_at=1700000000000,
            condition_id="cond-1",
            market_end_time=1700000300,
        )
        await eng.handle_market_resolution(
            condition_id="cond-1",
            winning_asset_id="other-token",
            positions={"lose-token": pos},
        )
        call_kwargs = eng._risk.record_exit.call_args[1]
        assert call_kwargs["won"] is False
        pnl = call_kwargs["pnl_usdc"]
        # pnl = -(27.50 + 0.86) = -28.36
        assert pnl == Decimal("-28.36")

    @pytest.mark.asyncio
    async def test_skip_other_condition(self) -> None:
        eng = _make_engine()
        pos = Position(
            asset_id="other",
            direction=Direction.UP,
            side=Side.BUY,
            entry_price=Decimal("0.55"),
            size=Decimal("50"),
            cost_basis=Decimal("27.50"),
            fee_paid=Decimal("0.86"),
            order_id="ord-3",
            entered_at=1700000000000,
            condition_id="cond-OTHER",
            market_end_time=1700000300,
        )
        await eng.handle_market_resolution(
            condition_id="cond-1",
            winning_asset_id="other",
            positions={"other": pos},
        )
        eng._risk.record_exit.assert_not_called()


# ──────────────────────────── User Channel Handler ────────────────────────────


class TestUserMessageHandler:
    def test_trade_update_sets_event(self) -> None:
        import asyncio
        eng = _make_engine()
        evt = asyncio.Event()
        eng._fill_events["order-123"] = evt

        eng.handle_user_message({
            "type": "TRADE",
            "taker_order_id": "order-123",
            "status": "MATCHED",
            "price": "0.55",
            "size": "50",
        })
        assert evt.is_set()

    def test_trade_update_unknown_order_id(self) -> None:
        eng = _make_engine()
        # Should not raise
        eng.handle_user_message({
            "type": "TRADE",
            "taker_order_id": "unknown",
            "status": "MATCHED",
        })

    def test_order_update(self) -> None:
        eng = _make_engine()
        eng.handle_user_message({
            "type": "PLACEMENT",
            "id": "order-123",
        })


# ──────────────────────────── trade_log ────────────────────────────


class TestTradeLog:
    def test_empty_initially(self) -> None:
        eng = _make_engine()
        assert eng.trade_log == []
