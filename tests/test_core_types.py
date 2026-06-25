"""Tests for core/types.py — enums, dataclasses, validation, and computed properties."""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import patch

import pytest

from core.types import (
    CircuitState,
    ConnectionState,
    Direction,
    MarketPhase,
    MarketTokenPair,
    MetricsSnapshot,
    OrderbookState,
    OrderFlowMetrics,
    OrderStatus,
    OrderType,
    Position,
    PriceLevel,
    RiskState,
    Side,
    SignalStrength,
    TradeRecord,
    TradeStatus,
    UnifiedSignal,
)


# ──────────────────────────── Enums ────────────────────────────


class TestEnums:
    def test_side_values(self) -> None:
        assert Side.BUY.value == "BUY"
        assert Side.SELL.value == "SELL"

    def test_direction_values(self) -> None:
        assert Direction.UP.value == "UP"
        assert Direction.DOWN.value == "DOWN"
        assert Direction.FLAT.value == "FLAT"

    def test_order_type_values(self) -> None:
        assert OrderType.GTC.value == "GTC"
        assert OrderType.GTD.value == "GTD"
        assert OrderType.FOK.value == "FOK"
        assert OrderType.FAK.value == "FAK"

    def test_order_status_values(self) -> None:
        assert OrderStatus.LIVE.value == "live"
        assert OrderStatus.MATCHED.value == "matched"
        assert OrderStatus.CANCELLED.value == "cancelled"
        assert OrderStatus.FAILED.value == "failed"

    def test_trade_status_values(self) -> None:
        assert TradeStatus.MATCHED.value == "MATCHED"
        assert TradeStatus.CONFIRMED.value == "CONFIRMED"
        assert TradeStatus.FAILED.value == "FAILED"

    def test_signal_strength_ordering(self) -> None:
        assert SignalStrength.NONE.value < SignalStrength.WEAK.value
        assert SignalStrength.WEAK.value < SignalStrength.MEDIUM.value
        assert SignalStrength.MEDIUM.value < SignalStrength.STRONG.value
        assert SignalStrength.STRONG.value < SignalStrength.EXTREME.value

    def test_circuit_state_enum(self) -> None:
        assert CircuitState.CLOSED is not None
        assert CircuitState.OPEN is not None
        assert CircuitState.HALF_OPEN is not None

    def test_market_phase_enum(self) -> None:
        assert MarketPhase.WAITING is not None
        assert MarketPhase.ACTIVE is not None
        assert MarketPhase.RESOLVED is not None

    def test_connection_state_enum(self) -> None:
        assert ConnectionState.DISCONNECTED is not None
        assert ConnectionState.CONNECTED is not None
        assert ConnectionState.FAILED is not None


# ──────────────────────────── PriceLevel ────────────────────────────


class TestPriceLevel:
    def test_valid_price_level(self) -> None:
        pl = PriceLevel(price=Decimal("0.50"), size=Decimal("100"))
        assert pl.price == Decimal("0.50")
        assert pl.size == Decimal("100")

    def test_boundary_prices(self) -> None:
        PriceLevel(price=Decimal("0"), size=Decimal("10"))
        PriceLevel(price=Decimal("1"), size=Decimal("10"))
        PriceLevel(price=Decimal("0.01"), size=Decimal("0"))

    def test_price_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid price"):
            PriceLevel(price=Decimal("-0.01"), size=Decimal("100"))

    def test_price_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid price"):
            PriceLevel(price=Decimal("1.01"), size=Decimal("100"))

    def test_negative_size_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid size"):
            PriceLevel(price=Decimal("0.50"), size=Decimal("-1"))

    def test_frozen_immutable(self) -> None:
        pl = PriceLevel(price=Decimal("0.50"), size=Decimal("100"))
        with pytest.raises(AttributeError):
            pl.price = Decimal("0.60")  # type: ignore[misc]


# ──────────────────────────── OrderbookState ────────────────────────────


class TestOrderbookState:
    def _make(
        self,
        bids: dict | None = None,
        asks: dict | None = None,
    ) -> OrderbookState:
        return OrderbookState(
            asset_id="asset-1",
            market="mkt-1",
            bids=bids or {},
            asks=asks or {},
            timestamp=1700000000,
            hash="h1",
        )

    def test_best_bid_ask(self) -> None:
        ob = self._make(
            bids={Decimal("0.49"): Decimal("100"), Decimal("0.50"): Decimal("200")},
            asks={Decimal("0.52"): Decimal("100"), Decimal("0.53"): Decimal("150")},
        )
        assert ob.best_bid == Decimal("0.50")
        assert ob.best_ask == Decimal("0.52")

    def test_empty_book_properties(self) -> None:
        ob = self._make()
        assert ob.best_bid is None
        assert ob.best_ask is None
        assert ob.spread is None
        assert ob.midprice is None
        assert ob.microprice() is None
        assert ob.is_empty()

    def test_spread(self) -> None:
        ob = self._make(
            bids={Decimal("0.48"): Decimal("100")},
            asks={Decimal("0.52"): Decimal("100")},
        )
        assert ob.spread == Decimal("0.04")

    def test_midprice(self) -> None:
        ob = self._make(
            bids={Decimal("0.48"): Decimal("100")},
            asks={Decimal("0.52"): Decimal("100")},
        )
        assert ob.midprice == Decimal("0.50")

    def test_microprice_equal_volumes(self) -> None:
        ob = self._make(
            bids={Decimal("0.48"): Decimal("100")},
            asks={Decimal("0.52"): Decimal("100")},
        )
        mp = ob.microprice()
        assert mp == Decimal("0.50")

    def test_microprice_asymmetric_volumes(self) -> None:
        ob = self._make(
            bids={Decimal("0.48"): Decimal("100")},
            asks={Decimal("0.52"): Decimal("300")},
        )
        mp = ob.microprice()
        assert mp is not None
        # More ask volume → microprice closer to bid
        assert mp < Decimal("0.50")

    def test_microprice_zero_volumes(self) -> None:
        ob = self._make(
            bids={Decimal("0.48"): Decimal("0")},
            asks={Decimal("0.52"): Decimal("0")},
        )
        mp = ob.microprice()
        assert mp == ob.midprice

    def test_bid_depth(self) -> None:
        ob = self._make(
            bids={
                Decimal("0.50"): Decimal("100"),
                Decimal("0.49"): Decimal("200"),
                Decimal("0.48"): Decimal("150"),
            },
        )
        assert ob.bid_depth(2) == Decimal("300")
        assert ob.bid_depth(5) == Decimal("450")

    def test_ask_depth(self) -> None:
        ob = self._make(
            asks={
                Decimal("0.52"): Decimal("80"),
                Decimal("0.53"): Decimal("120"),
            },
        )
        assert ob.ask_depth(1) == Decimal("80")
        assert ob.ask_depth(5) == Decimal("200")

    def test_clone(self) -> None:
        ob = self._make(
            bids={Decimal("0.50"): Decimal("100")},
            asks={Decimal("0.52"): Decimal("200")},
        )
        clone = ob.clone()
        assert clone.bids == ob.bids
        assert clone.asks == ob.asks
        assert clone is not ob
        assert clone.bids is not ob.bids

    def test_is_empty_with_data(self) -> None:
        ob = self._make(bids={Decimal("0.50"): Decimal("100")})
        assert not ob.is_empty()


# ──────────────────────────── MarketTokenPair ────────────────────────────


class TestMarketTokenPair:
    def _make(self, start_offset: float = -100, duration: int = 300) -> MarketTokenPair:
        now = time.time()
        return MarketTokenPair(
            condition_id="cond-1",
            up_token_id="up-1",
            down_token_id="down-1",
            tick_size="0.01",
            neg_risk=True,
            start_time=int(now + start_offset),
            end_time=int(now + start_offset + duration),
            question="Will BTC go up?",
            slug="btc-updown-5m-1700000000",
        )

    def test_time_remaining_active(self) -> None:
        pair = self._make(start_offset=-100, duration=300)
        assert pair.time_remaining > 0
        assert pair.time_remaining <= 200

    def test_time_remaining_expired(self) -> None:
        pair = self._make(start_offset=-400, duration=300)
        assert pair.time_remaining == 0

    def test_pct_elapsed(self) -> None:
        pair = self._make(start_offset=-150, duration=300)
        assert 0.4 < pair.pct_elapsed < 0.6

    def test_is_near_expiry(self) -> None:
        pair = self._make(start_offset=-250, duration=300)
        assert pair.is_near_expiry

    def test_is_not_near_expiry(self) -> None:
        pair = self._make(start_offset=-100, duration=300)
        assert not pair.is_near_expiry

    def test_is_expired(self) -> None:
        pair = self._make(start_offset=-400, duration=300)
        assert pair.is_expired

    def test_expired_not_near_expiry(self) -> None:
        pair = self._make(start_offset=-400, duration=300)
        assert not pair.is_near_expiry

    def test_pct_elapsed_future(self) -> None:
        pair = self._make(start_offset=100, duration=300)
        assert pair.pct_elapsed == 0.0


# ──────────────────────────── Position ────────────────────────────


class TestPosition:
    def _make(self) -> Position:
        return Position(
            asset_id="asset-1",
            direction=Direction.UP,
            side=Side.BUY,
            entry_price=Decimal("0.55"),
            size=Decimal("50"),
            cost_basis=Decimal("27.50"),
            fee_paid=Decimal("0.86"),
            order_id="ord-1",
            entered_at=1700000000000,
            condition_id="cond-1",
            market_end_time=int(time.time()) + 300,
        )

    def test_unrealized_pnl_profit(self) -> None:
        pos = self._make()
        pnl = pos.unrealized_pnl(Decimal("0.65"))
        expected = (Decimal("0.65") - Decimal("0.55")) * Decimal("50") - Decimal("0.86")
        assert pnl == expected

    def test_unrealized_pnl_loss(self) -> None:
        pos = self._make()
        pnl = pos.unrealized_pnl(Decimal("0.40"))
        assert pnl < Decimal("0")

    def test_expected_pnl_at_resolution(self) -> None:
        pos = self._make()
        expected = (Decimal("1") - Decimal("0.55")) * Decimal("50") - Decimal("0.86")
        assert pos.expected_pnl_at_resolution() == expected

    def test_is_expired_false(self) -> None:
        pos = self._make()
        assert not pos.is_expired


# ──────────────────────────── UnifiedSignal ────────────────────────────


class TestUnifiedSignal:
    def _make(self, **kwargs) -> UnifiedSignal:
        defaults = dict(
            up_asset_id="up-1",
            down_asset_id="down-1",
            up_microprice=Decimal("0.55"),
            down_microprice=Decimal("0.45"),
            up_probability=Decimal("0.65"),
            down_probability=Decimal("0.35"),
            confidence=Decimal("0.75"),
            direction=Direction.UP,
            signal_strength=SignalStrength.STRONG,
            microprice_score=Decimal("0.3"),
            ofi_score=Decimal("0.2"),
            depth_ratio_score=Decimal("0.1"),
            velocity_score=Decimal("0.15"),
            absorption_score=Decimal("0.05"),
            up_spread=Decimal("0.02"),
            down_spread=Decimal("0.03"),
            liquidity_ok=True,
            spread_ok=True,
            manipulation_flag=False,
            liquidity_vacuum=False,
            violent_reversal=False,
            timestamp=1700000000000,
        )
        defaults.update(kwargs)
        return UnifiedSignal(**defaults)

    def test_is_tradeable_all_conditions_met(self) -> None:
        sig = self._make()
        assert sig.is_tradeable

    def test_not_tradeable_weak_signal(self) -> None:
        sig = self._make(signal_strength=SignalStrength.WEAK)
        assert not sig.is_tradeable

    def test_not_tradeable_low_confidence(self) -> None:
        sig = self._make(confidence=Decimal("0.50"))
        assert not sig.is_tradeable

    def test_not_tradeable_no_liquidity(self) -> None:
        sig = self._make(liquidity_ok=False)
        assert not sig.is_tradeable

    def test_not_tradeable_bad_spread(self) -> None:
        sig = self._make(spread_ok=False)
        assert not sig.is_tradeable

    def test_not_tradeable_manipulation(self) -> None:
        sig = self._make(manipulation_flag=True)
        assert not sig.is_tradeable

    def test_not_tradeable_liquidity_vacuum(self) -> None:
        sig = self._make(liquidity_vacuum=True)
        assert not sig.is_tradeable

    def test_not_tradeable_violent_reversal(self) -> None:
        sig = self._make(violent_reversal=True)
        assert not sig.is_tradeable

    def test_not_tradeable_flat_direction(self) -> None:
        sig = self._make(direction=Direction.FLAT)
        assert not sig.is_tradeable

    def test_tradeable_extreme_strength(self) -> None:
        sig = self._make(signal_strength=SignalStrength.EXTREME)
        assert sig.is_tradeable

    def test_tradeable_boundary_confidence(self) -> None:
        sig = self._make(confidence=Decimal("0.65"))
        assert sig.is_tradeable

    def test_not_tradeable_just_below_confidence(self) -> None:
        sig = self._make(confidence=Decimal("0.64"))
        assert not sig.is_tradeable


# ──────────────────────────── RiskState defaults ────────────────────────────


class TestRiskState:
    def test_defaults(self) -> None:
        rs = RiskState()
        assert rs.total_exposure_usdc == Decimal("0")
        assert rs.session_pnl_usdc == Decimal("0")
        assert rs.session_trades == 0
        assert rs.circuit_state == CircuitState.CLOSED
        assert rs.kill_switch_active is False
        assert rs.open_positions == 0
