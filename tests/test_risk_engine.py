"""Tests for risk/engine.py — risk checks, circuit breaker, kill switch, position tracking."""

from __future__ import annotations

import time
from decimal import Decimal
from unittest.mock import patch

import pytest
import pytest_asyncio

from config.settings import RiskConfig
from core.types import CircuitState, Direction, Position, Side, SignalStrength
from risk.engine import RiskEngine, RiskRejection
from tests.conftest import make_position, make_signal


@pytest.fixture
def config() -> RiskConfig:
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
def engine(config: RiskConfig) -> RiskEngine:
    return RiskEngine(config=config, initial_balance_usdc=Decimal("1000"))


# ──────────────────────────── check_entry ────────────────────────────


class TestCheckEntry:
    @pytest.mark.asyncio
    async def test_passes_valid_entry(self, engine: RiskEngine) -> None:
        signal = make_signal()
        size = await engine.check_entry(
            signal=signal,
            direction=Direction.UP,
            entry_price=Decimal("0.55"),
            size_usdc=Decimal("25"),
            cycle_key="cycle-1",
            asset_id="asset-1",
        )
        assert size == Decimal("25")

    @pytest.mark.asyncio
    async def test_caps_at_max_position(self, engine: RiskEngine) -> None:
        signal = make_signal()
        size = await engine.check_entry(
            signal=signal,
            direction=Direction.UP,
            entry_price=Decimal("0.55"),
            size_usdc=Decimal("200"),
            cycle_key="cycle-1",
            asset_id="asset-1",
        )
        assert size == Decimal("50")

    @pytest.mark.asyncio
    async def test_rejects_kill_switch(self, engine: RiskEngine) -> None:
        engine.activate_kill_switch("test")
        signal = make_signal()
        with pytest.raises(RiskRejection, match="KILL_SWITCH"):
            await engine.check_entry(
                signal=signal,
                direction=Direction.UP,
                entry_price=Decimal("0.55"),
                size_usdc=Decimal("25"),
                cycle_key="cycle-1",
                asset_id="asset-1",
            )

    @pytest.mark.asyncio
    async def test_rejects_duplicate_position(self, engine: RiskEngine) -> None:
        pos = make_position(asset_id="asset-dup")
        await engine.record_entry(pos, "cycle-1")

        signal = make_signal()
        with pytest.raises(RiskRejection, match="DUPLICATE_POSITION"):
            await engine.check_entry(
                signal=signal,
                direction=Direction.UP,
                entry_price=Decimal("0.55"),
                size_usdc=Decimal("25"),
                cycle_key="cycle-2",
                asset_id="asset-dup",
            )

    @pytest.mark.asyncio
    async def test_rejects_per_cycle_limit(self, engine: RiskEngine) -> None:
        signal = make_signal()
        # First entry passes
        await engine.check_entry(
            signal=signal,
            direction=Direction.UP,
            entry_price=Decimal("0.55"),
            size_usdc=Decimal("25"),
            cycle_key="cycle-1",
            asset_id="asset-1",
        )
        pos = make_position(asset_id="asset-1")
        await engine.record_entry(pos, "cycle-1")

        # Second entry in same cycle should be rejected
        with pytest.raises(RiskRejection, match="CYCLE_LIMIT"):
            await engine.check_entry(
                signal=signal,
                direction=Direction.UP,
                entry_price=Decimal("0.55"),
                size_usdc=Decimal("25"),
                cycle_key="cycle-1",
                asset_id="asset-2",
            )

    @pytest.mark.asyncio
    async def test_rejects_untradeable_signal(self, engine: RiskEngine) -> None:
        signal = make_signal(strength=SignalStrength.WEAK, confidence=Decimal("0.40"))
        with pytest.raises(RiskRejection, match="SIGNAL_NOT_TRADEABLE"):
            await engine.check_entry(
                signal=signal,
                direction=Direction.UP,
                entry_price=Decimal("0.55"),
                size_usdc=Decimal("25"),
                cycle_key="cycle-1",
                asset_id="asset-1",
            )

    @pytest.mark.asyncio
    async def test_rejects_min_size(self, engine: RiskEngine) -> None:
        signal = make_signal()
        with pytest.raises(RiskRejection, match="MIN_SIZE"):
            await engine.check_entry(
                signal=signal,
                direction=Direction.UP,
                entry_price=Decimal("0.55"),
                size_usdc=Decimal("3"),
                cycle_key="cycle-1",
                asset_id="asset-1",
            )

    @pytest.mark.asyncio
    async def test_rejects_circuit_open(self, engine: RiskEngine) -> None:
        engine._state.circuit_state = CircuitState.OPEN
        engine._circuit_open_time = time.monotonic()

        signal = make_signal()
        with pytest.raises(RiskRejection, match="CIRCUIT_OPEN"):
            await engine.check_entry(
                signal=signal,
                direction=Direction.UP,
                entry_price=Decimal("0.55"),
                size_usdc=Decimal("25"),
                cycle_key="cycle-1",
                asset_id="asset-1",
            )

    @pytest.mark.asyncio
    async def test_frequency_limiter(self, engine: RiskEngine) -> None:
        signal = make_signal()
        # Record 3 trades in last minute (max_trades_per_minute = 3)
        engine._trade_times = [time.time() - 10, time.time() - 5, time.time() - 1]

        with pytest.raises(RiskRejection, match="FREQUENCY_LIMIT"):
            await engine.check_entry(
                signal=signal,
                direction=Direction.UP,
                entry_price=Decimal("0.55"),
                size_usdc=Decimal("25"),
                cycle_key="cycle-2",
                asset_id="asset-new",
            )


# ──────────────────────────── record_entry / record_exit ────────────────────────────


class TestPositionTracking:
    @pytest.mark.asyncio
    async def test_record_entry(self, engine: RiskEngine) -> None:
        pos = make_position(asset_id="asset-rec", cost_basis=Decimal("30"))
        await engine.record_entry(pos, "cycle-1")

        state = await engine.get_state()
        assert state.total_exposure_usdc == Decimal("30")
        assert state.session_trades == 1
        assert state.open_positions == 1
        assert await engine.has_position("asset-rec")

    @pytest.mark.asyncio
    async def test_record_exit_win(self, engine: RiskEngine) -> None:
        pos = make_position(asset_id="asset-w", cost_basis=Decimal("30"))
        await engine.record_entry(pos, "cycle-1")

        await engine.record_exit("asset-w", pnl_usdc=Decimal("10"), won=True)

        state = await engine.get_state()
        assert state.total_exposure_usdc == Decimal("0")
        assert state.session_pnl_usdc == Decimal("10")
        assert state.consecutive_wins == 1
        assert state.consecutive_losses == 0

    @pytest.mark.asyncio
    async def test_record_exit_loss(self, engine: RiskEngine) -> None:
        pos = make_position(asset_id="asset-l", cost_basis=Decimal("30"))
        await engine.record_entry(pos, "cycle-1")

        await engine.record_exit("asset-l", pnl_usdc=Decimal("-30"), won=False)

        state = await engine.get_state()
        assert state.consecutive_losses == 1
        assert state.consecutive_wins == 0
        assert state.session_pnl_usdc == Decimal("-30")

    @pytest.mark.asyncio
    async def test_exit_unknown_position(self, engine: RiskEngine) -> None:
        # Should not raise, just log a warning
        await engine.record_exit("nonexistent", pnl_usdc=Decimal("0"), won=False)

    @pytest.mark.asyncio
    async def test_exposure_never_negative(self, engine: RiskEngine) -> None:
        pos = make_position(asset_id="a", cost_basis=Decimal("10"))
        await engine.record_entry(pos, "c1")
        # Exit with a big PnL shouldn't make exposure negative
        await engine.record_exit("a", pnl_usdc=Decimal("5"), won=True)
        state = await engine.get_state()
        assert state.total_exposure_usdc >= Decimal("0")


# ──────────────────────────── Circuit Breaker ────────────────────────────


class TestCircuitBreaker:
    @pytest.mark.asyncio
    async def test_consecutive_losses_trip(self, engine: RiskEngine) -> None:
        for i in range(4):
            pos = make_position(asset_id=f"asset-{i}", cost_basis=Decimal("10"))
            await engine.record_entry(pos, f"cycle-{i}")
            await engine.record_exit(f"asset-{i}", pnl_usdc=Decimal("-10"), won=False)

        state = await engine.get_state()
        assert state.circuit_state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_drawdown_trips_circuit(self, engine: RiskEngine) -> None:
        # Start with $1000. A $200 loss = 20% drawdown
        pos = make_position(asset_id="big", cost_basis=Decimal("250"))
        await engine.record_entry(pos, "c1")
        await engine.record_exit("big", pnl_usdc=Decimal("-200"), won=False)

        state = await engine.get_state()
        assert state.circuit_state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_circuit_recovery_after_cooldown(self, engine: RiskEngine) -> None:
        engine._state.circuit_state = CircuitState.OPEN
        # Set open time far in the past (> cooldown)
        engine._circuit_open_time = time.monotonic() - 400

        await engine._check_circuit_breaker_recovery()
        assert engine._state.circuit_state == CircuitState.HALF_OPEN
        assert engine._state.consecutive_losses == 0

    @pytest.mark.asyncio
    async def test_circuit_no_recovery_before_cooldown(self, engine: RiskEngine) -> None:
        engine._state.circuit_state = CircuitState.OPEN
        engine._circuit_open_time = time.monotonic() - 10

        await engine._check_circuit_breaker_recovery()
        assert engine._state.circuit_state == CircuitState.OPEN


# ──────────────────────────── Kill Switch ────────────────────────────


class TestKillSwitch:
    def test_activate_external(self, engine: RiskEngine) -> None:
        engine.activate_kill_switch("manual halt")
        assert engine._state.kill_switch_active is True

    def test_reset(self, engine: RiskEngine) -> None:
        engine.activate_kill_switch("halt")
        engine.reset_kill_switch()
        assert engine._state.kill_switch_active is False
        assert engine._state.circuit_state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_daily_loss_activates_kill_switch(self, engine: RiskEngine) -> None:
        # Simulate huge loss to trigger daily loss limit
        engine._state.current_equity_usdc = Decimal("900")
        # session_start_equity = 1000, loss = 100 > 75 (max_daily_loss)

        signal = make_signal()
        with pytest.raises(RiskRejection, match="DAILY_LOSS_LIMIT"):
            await engine.check_entry(
                signal=signal,
                direction=Direction.UP,
                entry_price=Decimal("0.55"),
                size_usdc=Decimal("25"),
                cycle_key="cycle-1",
                asset_id="new-asset",
            )
        assert engine._state.kill_switch_active is True


# ──────────────────────────── Market Anomaly Detection ────────────────────────────


class TestMarketAnomaly:
    @pytest.mark.asyncio
    async def test_liquidity_vacuum(self, engine: RiskEngine) -> None:
        signal = make_signal(liquidity_vacuum=True)
        is_anomaly, reason = await engine.check_market_anomaly(signal)
        assert is_anomaly
        assert "LIQUIDITY_VACUUM" in reason

    @pytest.mark.asyncio
    async def test_manipulation_detected(self, engine: RiskEngine) -> None:
        signal = make_signal(manipulation_flag=True)
        is_anomaly, reason = await engine.check_market_anomaly(signal)
        assert is_anomaly
        assert "MANIPULATION" in reason

    @pytest.mark.asyncio
    async def test_violent_reversal(self, engine: RiskEngine) -> None:
        signal = make_signal(violent_reversal=True)
        is_anomaly, reason = await engine.check_market_anomaly(signal)
        assert is_anomaly
        assert "VIOLENT_REVERSAL" in reason

    @pytest.mark.asyncio
    async def test_insufficient_liquidity(self, engine: RiskEngine) -> None:
        signal = make_signal(liquidity_ok=False)
        is_anomaly, reason = await engine.check_market_anomaly(signal)
        assert is_anomaly
        assert "INSUFFICIENT_LIQUIDITY" in reason

    @pytest.mark.asyncio
    async def test_no_anomaly(self, engine: RiskEngine) -> None:
        signal = make_signal()
        is_anomaly, reason = await engine.check_market_anomaly(signal)
        assert not is_anomaly
        assert reason == ""


# ──────────────────────────── State Accessors ────────────────────────────


class TestStateAccessors:
    @pytest.mark.asyncio
    async def test_get_state(self, engine: RiskEngine) -> None:
        state = await engine.get_state()
        assert state.current_equity_usdc == Decimal("1000")

    @pytest.mark.asyncio
    async def test_get_positions_empty(self, engine: RiskEngine) -> None:
        positions = await engine.get_positions()
        assert positions == {}

    @pytest.mark.asyncio
    async def test_get_position(self, engine: RiskEngine) -> None:
        pos = make_position(asset_id="a-1")
        await engine.record_entry(pos, "c-1")

        got = await engine.get_position("a-1")
        assert got is not None
        assert got.asset_id == "a-1"

    @pytest.mark.asyncio
    async def test_get_position_missing(self, engine: RiskEngine) -> None:
        got = await engine.get_position("nonexistent")
        assert got is None

    def test_min_position_usdc(self, engine: RiskEngine) -> None:
        assert engine.min_position_usdc() == Decimal("5")
