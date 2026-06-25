"""Tests for recovery/state_manager.py — StateStore save/load/clear and RecoveryManager."""

from __future__ import annotations

import json
import os
import time
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.types import CircuitState, RiskState
from recovery.state_manager import RecoveryManager, StateStore


@pytest.fixture
def tmp_state_path(tmp_path) -> str:
    return str(tmp_path / "test_state.json")


@pytest.fixture
def store(tmp_state_path: str) -> StateStore:
    return StateStore(path=tmp_state_path)


# ──────────────────────────── StateStore ────────────────────────────


class TestStateStoreSave:
    def test_save_creates_file(self, store: StateStore, tmp_state_path: str) -> None:
        result = store.save({"key": "value"})
        assert result is True
        assert os.path.exists(tmp_state_path)

    def test_save_valid_json(self, store: StateStore, tmp_state_path: str) -> None:
        store.save({"positions": {}, "risk": {"pnl": "10.5"}})
        with open(tmp_state_path) as f:
            data = json.load(f)
        assert data["risk"]["pnl"] == "10.5"
        assert "_saved_at" in data

    def test_save_decimal_serialization(self, store: StateStore, tmp_state_path: str) -> None:
        store.save({"amount": Decimal("123.456")})
        with open(tmp_state_path) as f:
            data = json.load(f)
        assert data["amount"] == "123.456"

    def test_save_atomic_no_tmp_file(self, store: StateStore, tmp_state_path: str) -> None:
        store.save({"data": 1})
        assert not os.path.exists(tmp_state_path + ".tmp")

    def test_save_overwrites(self, store: StateStore, tmp_state_path: str) -> None:
        store.save({"version": 1})
        store.save({"version": 2})
        with open(tmp_state_path) as f:
            data = json.load(f)
        assert data["version"] == 2


class TestStateStoreLoad:
    def test_load_returns_none_no_file(self, store: StateStore) -> None:
        assert store.load() is None

    def test_load_round_trip(self, store: StateStore) -> None:
        original = {"risk": {"pnl": "5.0"}, "positions": {"a": 1}}
        store.save(original)
        loaded = store.load()
        assert loaded is not None
        assert loaded["risk"]["pnl"] == "5.0"

    def test_load_corrupt_json(self, store: StateStore, tmp_state_path: str) -> None:
        with open(tmp_state_path, "w") as f:
            f.write("{invalid json!!}")
        result = store.load()
        assert result is None

    def test_load_stale_state_still_loads(self, store: StateStore, tmp_state_path: str) -> None:
        state = {"data": 1, "_saved_at": time.time() - 700}
        with open(tmp_state_path, "w") as f:
            json.dump(state, f)
        loaded = store.load()
        assert loaded is not None
        assert loaded["data"] == 1


class TestStateStoreClear:
    def test_clear_removes_file(self, store: StateStore, tmp_state_path: str) -> None:
        store.save({"data": 1})
        store.clear()
        assert not os.path.exists(tmp_state_path)

    def test_clear_no_file_ok(self, store: StateStore) -> None:
        store.clear()  # Should not raise


class TestJsonSerial:
    def test_decimal(self) -> None:
        assert StateStore._json_serial(Decimal("1.23")) == "1.23"

    def test_non_serializable(self) -> None:
        with pytest.raises(TypeError, match="Not serializable"):
            StateStore._json_serial(object())


# ──────────────────────────── RecoveryManager ────────────────────────────


class TestRecoveryManagerRecover:
    @pytest.mark.asyncio
    async def test_recover_no_state(self, store: StateStore) -> None:
        risk = MagicMock()
        mgr = RecoveryManager(store=store, risk_engine=risk, interval_s=60)
        assert await mgr.recover() is False

    @pytest.mark.asyncio
    async def test_recover_missing_fields(self, store: StateStore) -> None:
        store.save({"something_else": True})
        risk = MagicMock()
        mgr = RecoveryManager(store=store, risk_engine=risk, interval_s=60)
        assert await mgr.recover() is False

    @pytest.mark.asyncio
    async def test_recover_empty_positions(self, store: StateStore) -> None:
        store.save({"positions": {}, "risk": {"pnl": "0"}})
        risk = MagicMock()
        mgr = RecoveryManager(store=store, risk_engine=risk, interval_s=60)
        assert await mgr.recover() is False

    @pytest.mark.asyncio
    async def test_recover_with_positions(self, store: StateStore) -> None:
        store.save({
            "positions": {
                "asset-1": {
                    "direction": "UP",
                    "entry_price": "0.55",
                    "cost_basis": "27.50",
                    "size": "50",
                    "fee_paid": "0.86",
                    "order_id": "ord-1",
                    "entered_at": 1700000000,
                    "condition_id": "cond-1",
                    "market_end_time": int(time.time()) + 300,
                },
            },
            "risk": {"session_pnl_usdc": "0"},
        })
        risk = MagicMock()
        mgr = RecoveryManager(store=store, risk_engine=risk, interval_s=60)
        assert await mgr.recover() is True


class TestRecoveryManagerSaveNow:
    @pytest.mark.asyncio
    async def test_save_now(self, store: StateStore, tmp_state_path: str) -> None:
        risk = MagicMock()
        risk.get_state = AsyncMock(return_value=RiskState())
        risk.get_positions = AsyncMock(return_value={})

        mgr = RecoveryManager(store=store, risk_engine=risk, interval_s=60)
        await mgr.save_now()

        assert os.path.exists(tmp_state_path)
        with open(tmp_state_path) as f:
            data = json.load(f)
        assert "risk" in data
        assert "positions" in data


class TestRecoveryManagerStartStop:
    @pytest.mark.asyncio
    async def test_start_stop(self, store: StateStore) -> None:
        risk = MagicMock()
        risk.get_state = AsyncMock(return_value=RiskState())
        risk.get_positions = AsyncMock(return_value={})

        mgr = RecoveryManager(store=store, risk_engine=risk, interval_s=60)
        await mgr.start()
        assert mgr._running is True
        await mgr.stop()
        assert mgr._running is False
