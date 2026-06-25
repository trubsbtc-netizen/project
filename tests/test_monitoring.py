"""Tests for monitoring/logger.py — MetricsCollector and visual helpers."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from monitoring.logger import (
    MetricsCollector,
    _hms,
    _ms,
    _slug_tag,
    _dir_tag,
    _pnl_tag,
    _roi_tag,
)


# ──────────────────────────── MetricsCollector ────────────────────────────


class TestMetricsCollector:
    def test_inc_default(self) -> None:
        mc = MetricsCollector()
        mc.inc("trades")
        mc.inc("trades")
        assert mc._counters["trades"] == 2

    def test_inc_n(self) -> None:
        mc = MetricsCollector()
        mc.inc("orders", 5)
        assert mc._counters["orders"] == 5

    def test_gauge(self) -> None:
        mc = MetricsCollector()
        mc.gauge("spread", 0.04)
        assert mc._gauges["spread"] == 0.04
        mc.gauge("spread", 0.02)
        assert mc._gauges["spread"] == 0.02

    def test_uptime(self) -> None:
        mc = MetricsCollector()
        time.sleep(0.05)
        assert mc.uptime() >= 0.05

    def test_prometheus_text_contains_uptime(self) -> None:
        mc = MetricsCollector()
        text = mc.to_prometheus_text()
        assert "bot_uptime_seconds" in text

    def test_prometheus_text_counters(self) -> None:
        mc = MetricsCollector()
        mc.inc("entries", 3)
        text = mc.to_prometheus_text()
        assert "bot_entries_total 3" in text

    def test_prometheus_text_gauges(self) -> None:
        mc = MetricsCollector()
        mc.gauge("pnl_usdc", 12.5)
        text = mc.to_prometheus_text()
        assert "bot_pnl_usdc 12.5" in text

    def test_dash_replaced_with_underscore(self) -> None:
        mc = MetricsCollector()
        mc.inc("ws-reconnect", 2)
        text = mc.to_prometheus_text()
        assert "bot_ws_reconnect_total 2" in text
        assert "ws-reconnect" not in text


# ──────────────────────────── Visual Helpers ────────────────────────────


class TestHms:
    def test_zero(self) -> None:
        assert _hms(0) == "00:00:00"

    def test_one_hour(self) -> None:
        assert _hms(3661) == "01:01:01"

    def test_just_seconds(self) -> None:
        assert _hms(45) == "00:00:45"


class TestMs:
    def test_zero(self) -> None:
        assert _ms(0) == "0m00s"

    def test_five_minutes(self) -> None:
        assert _ms(300) == "5m00s"

    def test_mixed(self) -> None:
        assert _ms(125) == "2m05s"


class TestSlugTag:
    def test_long_slug(self) -> None:
        tag = _slug_tag("btc-updown-5m-1700000000")
        # Should contain last 14 chars
        assert "m-1700000000" in tag

    def test_short_slug(self) -> None:
        tag = _slug_tag("short")
        assert "short" in tag


class TestDirTag:
    def test_up(self) -> None:
        tag = _dir_tag("up")
        assert "UP" in tag

    def test_down(self) -> None:
        tag = _dir_tag("down")
        assert "DOWN" in tag


class TestPnlTag:
    def test_positive(self) -> None:
        tag = _pnl_tag(10.5)
        assert "+$10.50" in tag

    def test_negative(self) -> None:
        tag = _pnl_tag(-5.25)
        assert "-$5.25" in tag

    def test_zero(self) -> None:
        tag = _pnl_tag(0)
        assert "+$0.00" in tag


class TestRoiTag:
    def test_positive(self) -> None:
        tag = _roi_tag(15.5)
        assert "+15.5%" in tag

    def test_negative(self) -> None:
        tag = _roi_tag(-3.2)
        assert "-3.2%" in tag
