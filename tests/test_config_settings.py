"""Tests for config/settings.py — validation, env loading, and config composition."""

from __future__ import annotations

import os
from decimal import Decimal
from unittest.mock import patch

import pytest

from config.settings import (
    AuthConfig,
    BotConfig,
    ConfigError,
    NetworkConfig,
    RiskConfig,
    SignalConfig,
    TradingConfig,
    _require_env,
    load_config,
)


# ──────────────────────────── AuthConfig.validate ────────────────────────────


class TestAuthConfigValidation:
    def _make(self, **kwargs) -> AuthConfig:
        defaults = dict(
            private_key="0xdeadbeef",
            funder_address="0x1234abcd",
            auto_create_api_key=True,
        )
        defaults.update(kwargs)
        return AuthConfig(**defaults)

    def test_valid_auto_create(self) -> None:
        auth = self._make()
        auth.validate()

    def test_missing_private_key(self) -> None:
        auth = self._make(private_key="")
        with pytest.raises(ConfigError, match="PRIVATE_KEY not set"):
            auth.validate()

    def test_private_key_no_0x_prefix(self) -> None:
        auth = self._make(private_key="deadbeef")
        with pytest.raises(ConfigError, match="must start with 0x"):
            auth.validate()

    def test_missing_api_key_when_no_auto_create(self) -> None:
        auth = self._make(auto_create_api_key=False, api_key="")
        with pytest.raises(ConfigError, match="API_KEY not set"):
            auth.validate()

    def test_missing_api_secret_when_no_auto_create(self) -> None:
        auth = self._make(
            auto_create_api_key=False,
            api_key="key-1",
            api_secret="",
        )
        with pytest.raises(ConfigError, match="API_SECRET not set"):
            auth.validate()

    def test_missing_api_passphrase_when_no_auto_create(self) -> None:
        auth = self._make(
            auto_create_api_key=False,
            api_key="key-1",
            api_secret="secret-1",
            api_passphrase="",
        )
        with pytest.raises(ConfigError, match="API_PASSPHRASE not set"):
            auth.validate()

    def test_valid_manual_api_key(self) -> None:
        auth = self._make(
            auto_create_api_key=False,
            api_key="key-1",
            api_secret="secret-1",
            api_passphrase="pass-1",
        )
        auth.validate()

    def test_missing_funder_address(self) -> None:
        auth = self._make(funder_address="")
        with pytest.raises(ConfigError, match="FUNDER_ADDRESS not set"):
            auth.validate()

    def test_funder_address_no_0x_prefix(self) -> None:
        auth = self._make(funder_address="1234abcd")
        with pytest.raises(ConfigError, match="hex address starting with 0x"):
            auth.validate()


# ──────────────────────────── RiskConfig.validate ────────────────────────────


class TestRiskConfigValidation:
    def test_valid_defaults(self) -> None:
        rc = RiskConfig()
        rc.validate()

    def test_max_position_usdc_zero(self) -> None:
        rc = RiskConfig(max_position_usdc=Decimal("0"))
        with pytest.raises(ConfigError, match="max_position_usdc must be > 0"):
            rc.validate()

    def test_max_position_negative(self) -> None:
        rc = RiskConfig(max_position_usdc=Decimal("-10"))
        with pytest.raises(ConfigError, match="max_position_usdc must be > 0"):
            rc.validate()

    def test_exposure_less_than_position(self) -> None:
        rc = RiskConfig(
            max_position_usdc=Decimal("100"),
            max_total_exposure_usdc=Decimal("50"),
        )
        with pytest.raises(ConfigError, match="max_total_exposure_usdc must be >= max_position_usdc"):
            rc.validate()

    def test_daily_loss_zero(self) -> None:
        rc = RiskConfig(max_daily_loss_usdc=Decimal("0"))
        with pytest.raises(ConfigError, match="max_daily_loss_usdc must be > 0"):
            rc.validate()

    def test_consecutive_losses_zero(self) -> None:
        rc = RiskConfig(max_consecutive_losses=0)
        with pytest.raises(ConfigError, match="max_consecutive_losses must be >= 1"):
            rc.validate()

    def test_kelly_fraction_zero(self) -> None:
        rc = RiskConfig(kelly_fraction=Decimal("0"))
        with pytest.raises(ConfigError, match="kelly_fraction must be in"):
            rc.validate()

    def test_kelly_fraction_above_one(self) -> None:
        rc = RiskConfig(kelly_fraction=Decimal("1.1"))
        with pytest.raises(ConfigError, match="kelly_fraction must be in"):
            rc.validate()

    def test_kelly_fraction_exactly_one(self) -> None:
        rc = RiskConfig(kelly_fraction=Decimal("1"))
        rc.validate()

    def test_max_kelly_fraction_invalid(self) -> None:
        rc = RiskConfig(max_kelly_fraction=Decimal("0"))
        with pytest.raises(ConfigError, match="max_kelly_fraction must be in"):
            rc.validate()


# ──────────────────────────── BotConfig.validate ────────────────────────────


class TestBotConfigValidation:
    def test_valid(self) -> None:
        cfg = BotConfig(
            auth=AuthConfig(
                private_key="0xdeadbeef",
                funder_address="0x1234abcd",
                auto_create_api_key=True,
            ),
        )
        cfg.validate()

    def test_auth_failure_propagates(self) -> None:
        cfg = BotConfig(
            auth=AuthConfig(private_key="", funder_address="0x1234"),
        )
        with pytest.raises(ConfigError):
            cfg.validate()


# ──────────────────────────── _require_env ────────────────────────────


class TestRequireEnv:
    def test_present(self) -> None:
        with patch.dict(os.environ, {"TEST_KEY": "value123"}):
            assert _require_env("TEST_KEY") == "value123"

    def test_missing(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigError, match="TEST_MISSING"):
                _require_env("TEST_MISSING")

    def test_empty_string(self) -> None:
        with patch.dict(os.environ, {"TEST_EMPTY": ""}):
            with pytest.raises(ConfigError):
                _require_env("TEST_EMPTY")


# ──────────────────────────── load_config ────────────────────────────


class TestLoadConfig:
    def test_load_with_env_vars(self) -> None:
        env = {
            "POLY_PRIVATE_KEY": "0xdeadbeef",
            "POLY_FUNDER_ADDRESS": "0x1234abcd",
            "POLY_AUTO_CREATE_API_KEY": "true",
            "DRY_RUN": "true",
            "LOG_LEVEL": "DEBUG",
            "MAX_POSITION_USDC": "100",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_config()
            assert cfg.auth.private_key == "0xdeadbeef"
            assert cfg.auth.funder_address == "0x1234abcd"
            assert cfg.dry_run is True
            assert cfg.log_level == "DEBUG"
            assert cfg.risk.max_position_usdc == Decimal("100")

    def test_load_missing_required_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigError, match="POLY_PRIVATE_KEY"):
                load_config()

    def test_default_values(self) -> None:
        env = {
            "POLY_PRIVATE_KEY": "0xdeadbeef",
            "POLY_FUNDER_ADDRESS": "0x1234abcd",
        }
        with patch.dict(os.environ, env, clear=True):
            cfg = load_config()
            assert cfg.dry_run is False
            assert cfg.log_level == "INFO"
            assert cfg.risk.max_position_usdc == Decimal("50")
            assert cfg.signal.min_confidence == Decimal("0.65")


# ──────────────────────────── NetworkConfig defaults ────────────────────────────


class TestNetworkConfig:
    def test_default_hosts(self) -> None:
        nc = NetworkConfig()
        assert "polymarket.com" in nc.clob_host
        assert nc.connect_timeout_s == 10.0
        assert nc.max_connections == 20


# ──────────────────────────── SignalConfig defaults ────────────────────────────


class TestSignalConfig:
    def test_defaults(self) -> None:
        sc = SignalConfig()
        assert sc.min_ev_threshold == Decimal("0.025")
        assert sc.min_confidence == Decimal("0.65")
        assert sc.max_tradeable_spread == Decimal("0.06")


# ──────────────────────────── TradingConfig defaults ────────────────────────────


class TestTradingConfig:
    def test_defaults(self) -> None:
        tc = TradingConfig()
        assert tc.min_order_size_usdc == Decimal("5")
        assert tc.default_tick_size == "0.01"
        assert len(tc.market_slug_patterns) > 0
