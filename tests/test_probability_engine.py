"""Tests for signals/probability_engine.py — signal fusion, classification, validation."""

from __future__ import annotations

from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.types import Direction, OrderbookState, SignalStrength
from orderbook.book import L2Orderbook
from signals.probability_engine import ProbabilityEngine


def _make_book(
    asset_id: str = "up-token",
    market: str = "test",
    direction: Direction = Direction.UP,
) -> L2Orderbook:
    return L2Orderbook(asset_id=asset_id, market=market, direction=direction)


def _make_state(
    bids: dict | None = None,
    asks: dict | None = None,
    asset_id: str = "asset-1",
) -> OrderbookState:
    return OrderbookState(
        asset_id=asset_id,
        market="mkt-1",
        bids=bids or {Decimal("0.50"): Decimal("500"), Decimal("0.49"): Decimal("300")},
        asks=asks or {Decimal("0.52"): Decimal("500"), Decimal("0.53"): Decimal("300")},
        timestamp=1700000000,
        hash="h1",
    )


# ──────────────────────────── Signal Fusion ────────────────────────────


class TestFuseSignals:
    def _engine(self) -> ProbabilityEngine:
        up = _make_book("up-token", "test", Direction.UP)
        down = _make_book("down-token", "test", Direction.DOWN)
        return ProbabilityEngine(up, down)

    def test_no_signals_returns_neutral(self) -> None:
        eng = self._engine()
        scores = {
            "microprice": Decimal("0"),
            "ofi": Decimal("0"),
            "depth_ratio": Decimal("0"),
            "velocity": Decimal("0"),
            "absorption": Decimal("0"),
        }
        prob, conf = eng._fuse_signals(scores)
        assert prob == Decimal("0.5")
        assert conf == Decimal("0.0")

    def test_all_bullish(self) -> None:
        eng = self._engine()
        scores = {
            "microprice": Decimal("0.8"),
            "ofi": Decimal("0.6"),
            "depth_ratio": Decimal("0.5"),
            "velocity": Decimal("0.4"),
            "absorption": Decimal("0.3"),
        }
        prob, conf = eng._fuse_signals(scores)
        assert prob > Decimal("0.5")
        assert conf > Decimal("0.5")

    def test_all_bearish(self) -> None:
        eng = self._engine()
        scores = {
            "microprice": Decimal("-0.8"),
            "ofi": Decimal("-0.6"),
            "depth_ratio": Decimal("-0.5"),
            "velocity": Decimal("-0.4"),
            "absorption": Decimal("-0.3"),
        }
        prob, conf = eng._fuse_signals(scores)
        assert prob < Decimal("0.5")
        assert conf > Decimal("0.5")

    def test_mixed_signals_lower_confidence(self) -> None:
        eng = self._engine()
        all_agree = {
            "microprice": Decimal("0.8"),
            "ofi": Decimal("0.6"),
            "depth_ratio": Decimal("0.5"),
            "velocity": Decimal("0.4"),
            "absorption": Decimal("0.3"),
        }
        mixed = {
            "microprice": Decimal("0.8"),
            "ofi": Decimal("-0.6"),
            "depth_ratio": Decimal("0.5"),
            "velocity": Decimal("-0.4"),
            "absorption": Decimal("0.3"),
        }
        _, conf_agree = eng._fuse_signals(all_agree)
        _, conf_mixed = eng._fuse_signals(mixed)
        assert conf_agree > conf_mixed

    def test_single_signal(self) -> None:
        eng = self._engine()
        scores = {
            "microprice": Decimal("0.5"),
            "ofi": Decimal("0"),
            "depth_ratio": Decimal("0"),
            "velocity": Decimal("0"),
            "absorption": Decimal("0"),
        }
        prob, conf = eng._fuse_signals(scores)
        assert prob > Decimal("0.5")


# ──────────────────────────── Direction Classification ────────────────────────────


class TestClassifyDirection:
    def _engine(self) -> ProbabilityEngine:
        up = _make_book("up-token", "test", Direction.UP)
        down = _make_book("down-token", "test", Direction.DOWN)
        return ProbabilityEngine(up, down)

    def test_up(self) -> None:
        eng = self._engine()
        assert eng._classify_direction(Decimal("0.60"), Decimal("0.70")) == Direction.UP

    def test_down(self) -> None:
        eng = self._engine()
        assert eng._classify_direction(Decimal("0.40"), Decimal("0.70")) == Direction.DOWN

    def test_flat_neutral(self) -> None:
        eng = self._engine()
        assert eng._classify_direction(Decimal("0.50"), Decimal("0.70")) == Direction.FLAT

    def test_flat_low_confidence(self) -> None:
        eng = self._engine()
        assert eng._classify_direction(Decimal("0.60"), Decimal("0.30")) == Direction.FLAT


# ──────────────────────────── Strength Classification ────────────────────────────


class TestClassifyStrength:
    def _engine(self) -> ProbabilityEngine:
        up = _make_book("up-token", "test", Direction.UP)
        down = _make_book("down-token", "test", Direction.DOWN)
        return ProbabilityEngine(up, down)

    def test_extreme(self) -> None:
        eng = self._engine()
        scores = {
            "microprice": Decimal("0.8"),
            "ofi": Decimal("0.7"),
            "depth_ratio": Decimal("0.6"),
            "velocity": Decimal("0.5"),
            "absorption": Decimal("0.4"),
        }
        s = eng._classify_strength(Decimal("0.75"), Decimal("0.85"), scores)
        assert s == SignalStrength.EXTREME

    def test_strong(self) -> None:
        eng = self._engine()
        scores = {
            "microprice": Decimal("0.5"),
            "ofi": Decimal("0.4"),
            "depth_ratio": Decimal("0.3"),
            "velocity": Decimal("0"),
            "absorption": Decimal("0"),
        }
        s = eng._classify_strength(Decimal("0.62"), Decimal("0.70"), scores)
        assert s == SignalStrength.STRONG

    def test_none_when_neutral(self) -> None:
        eng = self._engine()
        scores = {
            "microprice": Decimal("0"),
            "ofi": Decimal("0"),
            "depth_ratio": Decimal("0"),
            "velocity": Decimal("0"),
            "absorption": Decimal("0"),
        }
        s = eng._classify_strength(Decimal("0.50"), Decimal("0.50"), scores)
        assert s == SignalStrength.NONE


# ──────────────────────────── Market Condition Checks ────────────────────────────


class TestMarketConditions:
    def _engine(self) -> ProbabilityEngine:
        up = _make_book("up-token", "test", Direction.UP)
        down = _make_book("down-token", "test", Direction.DOWN)
        return ProbabilityEngine(up, down)

    def test_good_conditions(self) -> None:
        eng = self._engine()
        up_state = _make_state(
            bids={Decimal("0.50"): Decimal("1000"), Decimal("0.49"): Decimal("500")},
            asks={Decimal("0.52"): Decimal("1000"), Decimal("0.51"): Decimal("500")},
        )
        down_state = _make_state(
            bids={Decimal("0.48"): Decimal("1000"), Decimal("0.47"): Decimal("500")},
            asks={Decimal("0.50"): Decimal("1000"), Decimal("0.49"): Decimal("500")},
        )
        liquidity_ok, spread_ok = eng._check_market_conditions(up_state, down_state)
        assert liquidity_ok
        assert spread_ok

    def test_wide_spread(self) -> None:
        eng = self._engine()
        up_state = _make_state(
            bids={Decimal("0.30"): Decimal("1000")},
            asks={Decimal("0.70"): Decimal("1000")},
        )
        down_state = _make_state(
            bids={Decimal("0.30"): Decimal("1000")},
            asks={Decimal("0.70"): Decimal("1000")},
        )
        _, spread_ok = eng._check_market_conditions(up_state, down_state)
        assert not spread_ok


# ──────────────────────────── Neg-Risk Consistency ────────────────────────────


class TestNegRiskConsistency:
    def _engine(self) -> ProbabilityEngine:
        up = _make_book("up-token", "test", Direction.UP)
        down = _make_book("down-token", "test", Direction.DOWN)
        return ProbabilityEngine(up, down)

    def test_consistent(self) -> None:
        eng = self._engine()
        up = _make_state(
            bids={Decimal("0.48"): Decimal("100")},
            asks={Decimal("0.52"): Decimal("100")},
        )
        down = _make_state(
            bids={Decimal("0.48"): Decimal("100")},
            asks={Decimal("0.52"): Decimal("100")},
        )
        assert eng._check_neg_risk_consistency(up, down)

    def test_inconsistent(self) -> None:
        eng = self._engine()
        up = _make_state(
            bids={Decimal("0.75"): Decimal("100")},
            asks={Decimal("0.80"): Decimal("100")},
        )
        down = _make_state(
            bids={Decimal("0.75"): Decimal("100")},
            asks={Decimal("0.80"): Decimal("100")},
        )
        # midprice both 0.775, sum = 1.55, deviation = 0.55 > 0.08
        assert not eng._check_neg_risk_consistency(up, down)


# ──────────────────────────── Liquidity Vacuum ────────────────────────────


class TestLiquidityVacuum:
    def _engine(self) -> ProbabilityEngine:
        up = _make_book("up-token", "test", Direction.UP)
        down = _make_book("down-token", "test", Direction.DOWN)
        return ProbabilityEngine(up, down)

    def test_no_bids(self) -> None:
        eng = self._engine()
        up = OrderbookState(
            asset_id="a", market="m", bids={}, asks={Decimal("0.52"): Decimal("100")},
            timestamp=1, hash="h",
        )
        down = _make_state()
        assert eng._detect_liquidity_vacuum(up, down)

    def test_wide_spread_vacuum(self) -> None:
        eng = self._engine()
        up = _make_state(
            bids={Decimal("0.30"): Decimal("100")},
            asks={Decimal("0.60"): Decimal("100")},
        )
        down = _make_state()
        assert eng._detect_liquidity_vacuum(up, down)

    def test_healthy_book(self) -> None:
        eng = self._engine()
        up = _make_state(
            bids={Decimal("0.50"): Decimal("100"), Decimal("0.49"): Decimal("100"), Decimal("0.48"): Decimal("100")},
            asks={Decimal("0.52"): Decimal("100"), Decimal("0.53"): Decimal("100"), Decimal("0.54"): Decimal("100")},
        )
        down = _make_state(
            bids={Decimal("0.48"): Decimal("100"), Decimal("0.47"): Decimal("100"), Decimal("0.46"): Decimal("100")},
            asks={Decimal("0.50"): Decimal("100"), Decimal("0.51"): Decimal("100"), Decimal("0.52"): Decimal("100")},
        )
        assert not eng._detect_liquidity_vacuum(up, down)


# ──────────────────────────── Violent Reversal ────────────────────────────


class TestViolentReversal:
    def _engine(self) -> ProbabilityEngine:
        up = _make_book("up-token", "test", Direction.UP)
        down = _make_book("down-token", "test", Direction.DOWN)
        return ProbabilityEngine(up, down)

    def test_no_history(self) -> None:
        eng = self._engine()
        assert not eng._detect_violent_reversal()

    def test_alternating_directions(self) -> None:
        eng = self._engine()
        from tests.conftest import make_signal

        for d in [Direction.UP, Direction.DOWN, Direction.UP, Direction.DOWN]:
            eng._signal_history.append(make_signal(direction=d))

        assert eng._detect_violent_reversal()


# ──────────────────────────── EV Computation ────────────────────────────


class TestEntryEV:
    def _engine(self) -> ProbabilityEngine:
        up = _make_book("up-token", "test", Direction.UP)
        down = _make_book("down-token", "test", Direction.DOWN)
        return ProbabilityEngine(up, down)

    def test_positive_ev(self) -> None:
        eng = self._engine()
        signal = MagicMock()
        signal.up_probability = Decimal("0.70")
        signal.down_probability = Decimal("0.30")
        ev = eng.compute_entry_ev(signal, Decimal("0.55"), Direction.UP)
        assert ev > Decimal("0")

    def test_negative_ev_bad_odds(self) -> None:
        eng = self._engine()
        signal = MagicMock()
        signal.up_probability = Decimal("0.45")
        signal.down_probability = Decimal("0.55")
        ev = eng.compute_entry_ev(signal, Decimal("0.55"), Direction.UP)
        assert ev < Decimal("0")

    def test_ev_includes_fee_drag(self) -> None:
        eng = self._engine()
        signal = MagicMock()
        signal.up_probability = Decimal("0.60")
        signal.down_probability = Decimal("0.40")

        ev_at_55 = eng.compute_entry_ev(signal, Decimal("0.55"), Direction.UP)
        # Fee = 0.07 * 0.55 * 0.45 ≈ 0.0173
        # EV_before_fee = 0.60*0.45 - 0.40*0.55 = 0.27 - 0.22 = 0.05
        # EV_after_fee ≈ 0.05 - 0.0173 ≈ 0.033
        assert ev_at_55 > Decimal("0")
        assert ev_at_55 < Decimal("0.05")

    def test_down_direction(self) -> None:
        eng = self._engine()
        signal = MagicMock()
        signal.up_probability = Decimal("0.30")
        signal.down_probability = Decimal("0.70")
        ev = eng.compute_entry_ev(signal, Decimal("0.55"), Direction.DOWN)
        assert ev > Decimal("0")
