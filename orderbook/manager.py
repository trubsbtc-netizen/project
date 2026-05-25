"""
Multi-round orderbook manager.

Manages L2 orderbooks for multiple market rounds simultaneously:
  - ACTIVE round  : live orderbook, used for signal computation
  - NEXT round    : pre-warmed orderbook, ready at rollover T=0
  - ROUND+2       : token IDs fetched, WS subscribe pending

On rollover:
  1. NEXT becomes ACTIVE (instant swap, zero latency)
  2. ROUND+2 becomes NEXT (already subscribed, building)
  3. ROUND+3 is initialized (token IDs fetched)

This means at every rollover we already have a warm orderbook
instead of starting blind from zero.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Dict, Optional, Tuple

from core.types import Direction, MarketTokenPair, OrderbookState
from orderbook.book import L2Orderbook

logger = logging.getLogger(__name__)


class BookSet:
    """UP + DOWN orderbooks for a single market round."""

    def __init__(self, market: MarketTokenPair) -> None:
        self.market    = market
        self.up_book   = L2Orderbook(
            asset_id=market.up_token_id,
            market=market.condition_id,
            direction=Direction.UP,
        )
        self.down_book = L2Orderbook(
            asset_id=market.down_token_id,
            market=market.condition_id,
            direction=Direction.DOWN,
        )
        self.slug       = market.slug
        self.is_active  = False        # True while this is the trading round
        self.is_prewarm = False        # True while this is the next-round prewarm

    @property
    def both_initialized(self) -> bool:
        return self.up_book.is_initialized and self.down_book.is_initialized

    @property
    def asset_ids(self) -> Tuple[str, str]:
        return self.market.up_token_id, self.market.down_token_id

    async def up_snapshot(self) -> Optional[OrderbookState]:
        return await self.up_book.snapshot()

    async def down_snapshot(self) -> Optional[OrderbookState]:
        return await self.down_book.snapshot()

    def book_for_asset(self, asset_id: str) -> Optional[L2Orderbook]:
        if self.up_book.asset_id   == asset_id:
            return self.up_book
        if self.down_book.asset_id == asset_id:
            return self.down_book
        return None


class OrderbookManager:
    """
    Registry of BookSets keyed by slug.
    Handles routing of WS messages to the correct book,
    and clean handoff between rounds.
    """

    def __init__(self) -> None:
        self._books: Dict[str, BookSet]  = {}      # slug → BookSet
        self._active_slug: Optional[str] = None
        self._lock = asyncio.Lock()

    # ─────────────────────────── Registration ───────────────────────────

    async def register(self, market: MarketTokenPair, is_active: bool = False) -> BookSet:
        """Register a new BookSet. Returns existing if already registered."""
        async with self._lock:
            if market.slug in self._books:
                bs = self._books[market.slug]
                if is_active:
                    bs.is_active  = True
                    bs.is_prewarm = False
                    self._active_slug = market.slug
                return bs

            bs = BookSet(market)
            bs.is_active  = is_active
            bs.is_prewarm = not is_active
            self._books[market.slug] = bs

            if is_active:
                self._active_slug = market.slug

            logger.debug(
                "Registered book: slug=%-14s  UP=%-14s  DOWN=%-14s  active=%s",
                market.slug[-14:],
                market.up_token_id[:14],
                market.down_token_id[:14],
                is_active,
            )
            return bs

    async def activate(self, slug: str) -> Optional[BookSet]:
        """Switch active book to a pre-warmed book."""
        async with self._lock:
            bs = self._books.get(slug)
            if bs is None:
                return None

            # Deactivate previous active
            if self._active_slug and self._active_slug != slug:
                prev = self._books.get(self._active_slug)
                if prev:
                    prev.is_active  = False
                    prev.is_prewarm = False

            bs.is_active   = True
            bs.is_prewarm  = False
            self._active_slug = slug
            return bs

    async def evict_old(self, keep_slugs: set) -> None:
        """Remove books for slugs no longer needed."""
        async with self._lock:
            stale = [s for s in self._books if s not in keep_slugs]
            for s in stale:
                del self._books[s]
                logger.debug("Evicted book: %s", s[-14:])

    def mark_book_ready(self, slug: str) -> None:
        """Mark a prewarmed book as ready once both sides have snapshots."""
        bs = self._books.get(slug)
        if bs is None:
            return

        if bs.both_initialized:
            bs.is_prewarm = not bs.is_active

    # ─────────────────────────── Accessors ───────────────────────────

    @property
    def active(self) -> Optional[BookSet]:
        if self._active_slug:
            return self._books.get(self._active_slug)
        return None

    def prewarm_books(self) -> list[BookSet]:
        return [b for b in self._books.values() if b.is_prewarm]

    def book_for_asset(self, asset_id: str) -> Optional[L2Orderbook]:
        """Route an asset_id to the correct L2Orderbook across all rounds."""
        for bs in self._books.values():
            b = bs.book_for_asset(asset_id)
            if b is not None:
                return b
        return None

    def bookset_for_asset(self, asset_id: str) -> Optional[BookSet]:
        """Find the BookSet owning a given asset_id."""
        for bs in self._books.values():
            if bs.up_book.asset_id == asset_id or bs.down_book.asset_id == asset_id:
                return bs
        return None

    def all_asset_ids(self) -> list[str]:
        """All asset IDs currently tracked (for WS subscription)."""
        ids = []
        for bs in self._books.values():
            ids.extend(bs.asset_ids)
        return list(dict.fromkeys(ids))   # deduplicated, order preserved

    def slug_count(self) -> int:
        return len(self._books)
