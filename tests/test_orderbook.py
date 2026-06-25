"""Tests for orderbook/book.py — L2Orderbook state management and signal computation."""

from __future__ import annotations

from decimal import Decimal

import pytest

from core.types import Direction, Side
from orderbook.book import L2Orderbook, OrderEntry, TradeEvent


# ──────────────────────────── TradeEvent ────────────────────────────


class TestTradeEvent:
    def test_aggressive_buy(self) -> None:
        t = TradeEvent(
            asset_id="a", price=Decimal("0.5"), size=Decimal("10"),
            side=Side.BUY, timestamp=100,
        )
        assert t.is_aggressive_buy
        assert not t.is_aggressive_sell

    def test_aggressive_sell(self) -> None:
        t = TradeEvent(
            asset_id="a", price=Decimal("0.5"), size=Decimal("10"),
            side=Side.SELL, timestamp=100,
        )
        assert t.is_aggressive_sell
        assert not t.is_aggressive_buy


# ──────────────────────────── OrderEntry ────────────────────────────


class TestOrderEntry:
    def test_post_init_max_size(self) -> None:
        e = OrderEntry(
            price=Decimal("0.50"), size=Decimal("100"),
            side=Side.BUY, first_seen=1.0, last_seen=1.0,
        )
        assert e.max_size == Decimal("100")


# ──────────────────────────── L2Orderbook ────────────────────────────


@pytest.fixture
def book() -> L2Orderbook:
    return L2Orderbook(
        asset_id="up-token-abc",
        market="btc-updown-5m-123",
        direction=Direction.UP,
    )


class TestL2OrderbookInit:
    def test_not_initialized(self, book: L2Orderbook) -> None:
        assert not book.is_initialized

    def test_attributes(self, book: L2Orderbook) -> None:
        assert book.asset_id == "up-token-abc"
        assert book.direction == Direction.UP


class TestApplySnapshot:
    @pytest.mark.asyncio
    async def test_basic_snapshot(self, book: L2Orderbook) -> None:
        bids = [
            {"price": "0.48", "size": "100"},
            {"price": "0.49", "size": "200"},
            {"price": "0.50", "size": "150"},
        ]
        asks = [
            {"price": "0.52", "size": "80"},
            {"price": "0.53", "size": "120"},
        ]
        state = await book.apply_snapshot(bids, asks, "1700000000", "hash-1")

        assert book.is_initialized
        assert state.best_bid == Decimal("0.50")
        assert state.best_ask == Decimal("0.52")
        assert len(state.bids) == 3
        assert len(state.asks) == 2

    @pytest.mark.asyncio
    async def test_snapshot_replaces_state(self, book: L2Orderbook) -> None:
        bids1 = [{"price": "0.48", "size": "100"}]
        asks1 = [{"price": "0.52", "size": "100"}]
        await book.apply_snapshot(bids1, asks1, "1000", "h1")

        bids2 = [{"price": "0.45", "size": "200"}]
        asks2 = [{"price": "0.55", "size": "200"}]
        state = await book.apply_snapshot(bids2, asks2, "2000", "h2")

        assert state.best_bid == Decimal("0.45")
        assert state.best_ask == Decimal("0.55")
        assert len(state.bids) == 1

    @pytest.mark.asyncio
    async def test_zero_size_filtered(self, book: L2Orderbook) -> None:
        bids = [{"price": "0.50", "size": "0"}, {"price": "0.49", "size": "100"}]
        asks = [{"price": "0.52", "size": "50"}]
        state = await book.apply_snapshot(bids, asks, "1000", "h1")
        assert Decimal("0.50") not in state.bids
        assert len(state.bids) == 1

    @pytest.mark.asyncio
    async def test_sequence_increments(self, book: L2Orderbook) -> None:
        bids = [{"price": "0.50", "size": "100"}]
        asks = [{"price": "0.52", "size": "100"}]
        s1 = await book.apply_snapshot(bids, asks, "1000", "h1")
        s2 = await book.apply_snapshot(bids, asks, "2000", "h2")
        assert s2.sequence > s1.sequence


class TestApplyPriceChange:
    @pytest.mark.asyncio
    async def test_add_bid_level(self, book: L2Orderbook) -> None:
        bids = [{"price": "0.50", "size": "100"}]
        asks = [{"price": "0.52", "size": "100"}]
        await book.apply_snapshot(bids, asks, "1000", "h1")

        state, delta = await book.apply_price_change(
            price=Decimal("0.49"), size=Decimal("50"),
            side=Side.BUY, hash_val="h2", timestamp="2000",
        )
        assert Decimal("0.49") in state.bids
        assert state.bids[Decimal("0.49")] == Decimal("50")

    @pytest.mark.asyncio
    async def test_remove_bid_level(self, book: L2Orderbook) -> None:
        bids = [{"price": "0.50", "size": "100"}]
        asks = [{"price": "0.52", "size": "100"}]
        await book.apply_snapshot(bids, asks, "1000", "h1")

        state, delta = await book.apply_price_change(
            price=Decimal("0.50"), size=Decimal("0"),
            side=Side.BUY, hash_val="h2", timestamp="2000",
        )
        assert Decimal("0.50") not in state.bids

    @pytest.mark.asyncio
    async def test_add_ask_level(self, book: L2Orderbook) -> None:
        bids = [{"price": "0.50", "size": "100"}]
        asks = [{"price": "0.52", "size": "100"}]
        await book.apply_snapshot(bids, asks, "1000", "h1")

        state, delta = await book.apply_price_change(
            price=Decimal("0.54"), size=Decimal("200"),
            side=Side.SELL, hash_val="h2", timestamp="2000",
        )
        assert Decimal("0.54") in state.asks

    @pytest.mark.asyncio
    async def test_ofi_updates(self, book: L2Orderbook) -> None:
        bids = [{"price": "0.50", "size": "100"}]
        asks = [{"price": "0.52", "size": "100"}]
        await book.apply_snapshot(bids, asks, "1000", "h1")

        _, delta = await book.apply_price_change(
            price=Decimal("0.49"), size=Decimal("50"),
            side=Side.BUY, hash_val="h2", timestamp="2000",
        )
        assert delta is not None
        assert delta == Decimal("50")  # New bid = positive OFI


class TestApplyTrade:
    @pytest.mark.asyncio
    async def test_trade_recorded(self, book: L2Orderbook) -> None:
        await book.apply_trade(
            price=Decimal("0.51"), size=Decimal("10"),
            side=Side.BUY, timestamp="1000",
        )
        assert len(book._trades) == 1
        assert book._trades[0].price == Decimal("0.51")


class TestSnapshot:
    @pytest.mark.asyncio
    async def test_none_if_not_initialized(self, book: L2Orderbook) -> None:
        state = await book.snapshot()
        assert state is None

    @pytest.mark.asyncio
    async def test_returns_state(self, book: L2Orderbook) -> None:
        bids = [{"price": "0.50", "size": "100"}]
        asks = [{"price": "0.52", "size": "100"}]
        await book.apply_snapshot(bids, asks, "1000", "h1")

        state = await book.snapshot()
        assert state is not None
        assert state.best_bid == Decimal("0.50")


class TestComputeOfi:
    @pytest.mark.asyncio
    async def test_zero_when_empty(self, book: L2Orderbook) -> None:
        bids = [{"price": "0.50", "size": "100"}]
        asks = [{"price": "0.52", "size": "100"}]
        await book.apply_snapshot(bids, asks, "1000", "h1")
        ofi = await book.compute_ofi()
        assert ofi == Decimal("0")

    @pytest.mark.asyncio
    async def test_positive_after_bid_add(self, book: L2Orderbook) -> None:
        bids = [{"price": "0.50", "size": "100"}]
        asks = [{"price": "0.52", "size": "100"}]
        await book.apply_snapshot(bids, asks, "1000", "h1")

        await book.apply_price_change(
            Decimal("0.49"), Decimal("200"), Side.BUY, "h2", "2000",
        )
        ofi = await book.compute_ofi()
        # Should be positive (bid added)
        assert ofi > Decimal("0")


class TestComputeTakerVolumes:
    @pytest.mark.asyncio
    async def test_empty(self, book: L2Orderbook) -> None:
        buy_vol, sell_vol = await book.compute_taker_volumes()
        assert buy_vol == Decimal("0")
        assert sell_vol == Decimal("0")

    @pytest.mark.asyncio
    async def test_with_trades(self, book: L2Orderbook) -> None:
        import time as _time
        ts = str(int(_time.time() * 1000))
        await book.apply_trade(Decimal("0.50"), Decimal("10"), Side.BUY, ts)
        await book.apply_trade(Decimal("0.50"), Decimal("5"), Side.SELL, ts)
        buy_vol, sell_vol = await book.compute_taker_volumes(window_s=60)
        assert buy_vol == Decimal("10")
        assert sell_vol == Decimal("5")


class TestDetectSweep:
    @pytest.mark.asyncio
    async def test_no_sweep_initially(self, book: L2Orderbook) -> None:
        bids = [{"price": "0.50", "size": "100"}]
        asks = [{"price": "0.52", "size": "100"}]
        await book.apply_snapshot(bids, asks, "1000", "h1")
        assert not await book.detect_sweep()


class TestEstimateSpoof:
    @pytest.mark.asyncio
    async def test_zero_without_orders(self, book: L2Orderbook) -> None:
        prob = await book.estimate_spoof_probability()
        assert prob == Decimal("0")


class TestGetDepthAtLevels:
    @pytest.mark.asyncio
    async def test_depth(self, book: L2Orderbook) -> None:
        bids = [
            {"price": "0.50", "size": "100"},
            {"price": "0.49", "size": "200"},
        ]
        asks = [
            {"price": "0.52", "size": "80"},
            {"price": "0.53", "size": "120"},
        ]
        await book.apply_snapshot(bids, asks, "1000", "h1")

        bid_depth, ask_depth = book.get_depth_at_levels(1)
        assert bid_depth == Decimal("100")
        assert ask_depth == Decimal("80")

        bid_depth, ask_depth = book.get_depth_at_levels(5)
        assert bid_depth == Decimal("300")
        assert ask_depth == Decimal("200")
