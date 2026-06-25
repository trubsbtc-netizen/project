"""Tests for market/clock.py — deterministic 5-minute window computation."""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

from market.clock import (
    CYCLE_SECONDS,
    SLUG_PREFIX,
    RoundWindow,
    _make_window,
    current_window,
    next_window,
    window_from_slug,
    window_schedule,
)


class TestMakeWindow:
    def test_fields(self) -> None:
        wts = 1700000100
        w = _make_window(wts)
        assert w.window_ts == wts
        assert w.slug == f"{SLUG_PREFIX}-{wts}"
        assert w.start_time == wts
        assert w.end_time == wts + CYCLE_SECONDS
        assert w.round_number == wts // CYCLE_SECONDS

    def test_frozen(self) -> None:
        w = _make_window(1700000100)
        with pytest.raises(AttributeError):
            w.window_ts = 0  # type: ignore[misc]


class TestCurrentWindow:
    def test_aligned_to_300(self) -> None:
        w = current_window()
        assert w.window_ts % CYCLE_SECONDS == 0

    def test_current_time_in_window(self) -> None:
        w = current_window()
        now = time.time()
        assert w.start_time <= now < w.end_time

    def test_server_offset(self) -> None:
        w_no_offset = current_window(server_offset=0.0)
        w_forward = current_window(server_offset=300.0)
        assert w_forward.window_ts >= w_no_offset.window_ts

    def test_slug_format(self) -> None:
        w = current_window()
        assert w.slug.startswith(SLUG_PREFIX)
        ts_part = w.slug.split("-")[-1]
        assert ts_part.isdigit()


class TestNextWindow:
    def test_next_is_ahead_of_current(self) -> None:
        cur = current_window()
        nxt = next_window(n=1)
        assert nxt.window_ts == cur.window_ts + CYCLE_SECONDS

    def test_next_n(self) -> None:
        cur = current_window()
        nxt3 = next_window(n=3)
        assert nxt3.window_ts == cur.window_ts + 3 * CYCLE_SECONDS

    def test_next_zero_is_current(self) -> None:
        cur = current_window()
        nxt0 = next_window(n=0)
        assert nxt0.window_ts == cur.window_ts


class TestWindowSchedule:
    def test_default_count(self) -> None:
        schedule = window_schedule(count=4)
        assert len(schedule) == 4

    def test_first_is_current(self) -> None:
        schedule = window_schedule()
        cur = current_window()
        assert schedule[0].window_ts == cur.window_ts

    def test_sequential_windows(self) -> None:
        schedule = window_schedule(count=5)
        for i in range(1, len(schedule)):
            assert schedule[i].window_ts == schedule[i - 1].window_ts + CYCLE_SECONDS

    def test_count_one(self) -> None:
        schedule = window_schedule(count=1)
        assert len(schedule) == 1


class TestWindowFromSlug:
    def test_parse_valid_slug(self) -> None:
        ts = 1779673200
        slug = f"btc-updown-5m-{ts}"
        w = window_from_slug(slug)
        assert w.window_ts == ts
        assert w.slug == slug
        assert w.end_time == ts + CYCLE_SECONDS

    def test_round_trip(self) -> None:
        original = current_window()
        parsed = window_from_slug(original.slug)
        assert parsed.window_ts == original.window_ts
        assert parsed.slug == original.slug


class TestRoundWindowProperties:
    @patch("market.clock.time.time")
    def test_time_remaining(self, mock_time) -> None:
        mock_time.return_value = 1700000150.0
        w = _make_window(1700000100)
        # end = 1700000400, remaining = 250
        assert w.time_remaining == 250.0

    @patch("market.clock.time.time")
    def test_time_elapsed(self, mock_time) -> None:
        mock_time.return_value = 1700000200.0
        w = _make_window(1700000100)
        assert w.time_elapsed == 100.0

    @patch("market.clock.time.time")
    def test_pct_elapsed(self, mock_time) -> None:
        mock_time.return_value = 1700000250.0
        w = _make_window(1700000100)
        # elapsed = 150/300 = 0.5
        assert w.pct_elapsed == 0.5

    @patch("market.clock.time.time")
    def test_is_active(self, mock_time) -> None:
        mock_time.return_value = 1700000200.0
        w = _make_window(1700000100)
        assert w.is_active

    @patch("market.clock.time.time")
    def test_is_future(self, mock_time) -> None:
        mock_time.return_value = 1700000050.0
        w = _make_window(1700000100)
        assert w.is_future

    @patch("market.clock.time.time")
    def test_is_expired(self, mock_time) -> None:
        mock_time.return_value = 1700000500.0
        w = _make_window(1700000100)
        assert w.is_expired

    @patch("market.clock.time.time")
    def test_is_near_expiry_true(self, mock_time) -> None:
        mock_time.return_value = 1700000350.0
        w = _make_window(1700000100)
        # remaining = 400 - 350 = 50s → near expiry
        assert w.is_near_expiry

    @patch("market.clock.time.time")
    def test_is_near_expiry_false(self, mock_time) -> None:
        mock_time.return_value = 1700000200.0
        w = _make_window(1700000100)
        # remaining = 200s → not near
        assert not w.is_near_expiry

    @patch("market.clock.time.time")
    def test_str(self, mock_time) -> None:
        mock_time.return_value = 1700000200.0
        w = _make_window(1700000100)
        s = str(w)
        assert SLUG_PREFIX in s
        assert "left" in s
