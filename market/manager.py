"""
Market Manager — multi-round subscription with pre-warmed orderbooks.

Slug format confirmed: btc-updown-5m-{window_ts}
  window_ts = now - (now % 300)   always divisible by 300

Multi-round strategy:
  - Track CURRENT round (active trading)
  - Track NEXT round (pre-warm orderbook + token IDs)
  - Track ROUND+2 (pre-fetch token IDs only)

Pre-warm sequence (triggered at T-60s before rollover):
  1. Compute next slug from clock
  2. Fetch token IDs from Gamma API
  3. Subscribe WS to next token IDs (orderbook begins building)
  4. Fetch REST snapshot for next tokens
  5. At T=0: swap active → next, next → next+1

This gives the bot a full pre-warmed orderbook at round start
instead of waiting for first WS book event (latency ~300ms+).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from core.constants import GAMMA_API_HOST
from core.types import MarketPhase, MarketTokenPair
from market.clock import (
    CYCLE_SECONDS,
    RoundWindow,
    current_window,
    next_window,
    window_from_slug,
    window_schedule,
)

logger = logging.getLogger(__name__)

# How many seconds before end to start pre-warming next round
PREWARM_LEAD_S  = 60
# How many future rounds to keep token IDs pre-fetched
PREFETCH_ROUNDS = 3


class RoundInfo:
    """All data for a single market round."""

    def __init__(self, window: RoundWindow) -> None:
        self.window:     RoundWindow               = window
        self.market:     Optional[MarketTokenPair] = None   # set after API fetch
        self.book_ready: bool                      = False  # True after first WS snapshot
        self.subscribed: bool                      = False  # True after WS subscribe

    @property
    def slug(self) -> str:
        return self.window.slug

    @property
    def is_ready(self) -> bool:
        return self.market is not None and self.book_ready


class MarketNotFoundError(Exception):
    pass


class MarketManager:
    """
    Multi-round market manager with deterministic slug computation
    and pre-warmed orderbook for next rounds.
    """

    def __init__(
        self,
        session:              aiohttp.ClientSession,
        on_new_market:        Optional[Callable[[MarketTokenPair, bool], None]] = None,
        on_market_resolved:   Optional[Callable[[str, str], None]] = None,
        on_prewarm_ready:     Optional[Callable[[MarketTokenPair], None]] = None,
        gamma_host:           str = GAMMA_API_HOST,
        server_time_offset:   float = 0.0,
    ) -> None:
        self._session           = session
        self._on_new_market_cb  = on_new_market
        self._on_resolved_cb    = on_market_resolved
        self._on_prewarm_cb     = on_prewarm_ready
        self._gamma_host        = gamma_host
        self._offset            = server_time_offset

        # Round registry: slug -> RoundInfo
        self._rounds: Dict[str, RoundInfo] = {}

        self._active_slug: Optional[str] = None
        self._lock = asyncio.Lock()
        self._phase = MarketPhase.WAITING

        self._monitor_task:  Optional[asyncio.Task] = None
        self._started_at:    float = 0.0

    # ─────────────────────────── Public API ───────────────────────────

    @property
    def active_market(self) -> Optional[MarketTokenPair]:
        if self._active_slug and self._active_slug in self._rounds:
            return self._rounds[self._active_slug].market
        return None

    @property
    def active_window(self) -> Optional[RoundWindow]:
        if self._active_slug and self._active_slug in self._rounds:
            return self._rounds[self._active_slug].window
        return None

    @property
    def next_market(self) -> Optional[MarketTokenPair]:
        nw = next_window(1, self._offset)
        info = self._rounds.get(nw.slug)
        return info.market if info else None

    @property
    def phase(self) -> MarketPhase:
        return self._phase

    def update_phase(self) -> MarketPhase:
        w = self.active_window
        if w is None:
            self._phase = MarketPhase.WAITING
        elif w.is_expired:
            self._phase = MarketPhase.RESOLVED
        elif w.is_near_expiry:
            self._phase = MarketPhase.NEAR_EXPIRY
        else:
            self._phase = MarketPhase.ACTIVE
        return self._phase

    def mark_book_ready(self, slug: str) -> None:
        info = self._rounds.get(slug)
        if info:
            info.book_ready = True

    async def start(self) -> None:
        self._started_at = time.time()
        await self._bootstrap()
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="market-monitor"
        )

    async def stop(self) -> None:
        if self._monitor_task:
            self._monitor_task.cancel()
            await asyncio.gather(self._monitor_task, return_exceptions=True)

    # ─────────────────────────── Bootstrap ───────────────────────────

    async def _bootstrap(self) -> None:
        """
        On startup: fetch current round + next PREFETCH_ROUNDS rounds in parallel.
        Sets active market to the current round.
        """
        schedule = window_schedule(count=PREFETCH_ROUNDS + 1, server_offset=self._offset)
        logger.info("Bootstrapping %d rounds: %s … %s",
                    len(schedule), schedule[0].slug, schedule[-1].slug)

        # Initialize RoundInfo objects
        for w in schedule:
            self._rounds[w.slug] = RoundInfo(w)

        # Fetch all in parallel
        tasks = [
            asyncio.create_task(self._fetch_round(w.slug), name=f"fetch-{w.slug}")
            for w in schedule
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for w, result in zip(schedule, results):
            if isinstance(result, Exception):
                logger.warning("Bootstrap fetch failed for %s: %s", w.slug, result)

        # Set active to current
        cur = current_window(self._offset)
        async with self._lock:
            self._active_slug = cur.slug

        active = self.active_market
        if active is None:
            raise MarketNotFoundError(
                f"Cannot find active market: {cur.slug}\n"
                f"Ensure the market exists on Polymarket and the clock is correct."
            )

        logger.info("Active market confirmed: %s", cur.slug)

        # Fire callback for active market
        if self._on_new_market_cb:
            self._on_new_market_cb(active, False)   # False = not pre-warm

        # Fire pre-warm callbacks for already-fetched future rounds
        for w in schedule[1:]:
            info = self._rounds.get(w.slug)
            if info and info.market and self._on_prewarm_cb:
                self._on_prewarm_cb(info.market)

    # ─────────────────────────── Monitor Loop ───────────────────────────

    async def _monitor_loop(self) -> None:
        """
        Tick every 500ms:
        - Update phase
        - At T-60s: ensure next round is pre-fetched and WS-subscribed
        - At T=0: perform rollover
        - Keep round registry clean (remove stale rounds)
        """
        last_active_slug = self._active_slug

        while True:
            try:
                await asyncio.sleep(0.5)

                cur_window = current_window(self._offset)
                phase      = self.update_phase()

                # ── Rollover detection ──
                if cur_window.slug != last_active_slug:
                    await self._perform_rollover(cur_window)
                    last_active_slug = cur_window.slug

                # ── Pre-warm trigger (T-60s before end) ──
                aw = self.active_window
                if aw and aw.time_remaining < PREWARM_LEAD_S:
                    nw = next_window(1, self._offset)
                    info = self._rounds.get(nw.slug)
                    if info is None:
                        self._rounds[nw.slug] = RoundInfo(nw)
                    if not self._rounds[nw.slug].subscribed:
                        asyncio.create_task(
                            self._prewarm_round(nw.slug),
                            name=f"prewarm-{nw.slug}"
                        )

                # ── Prefetch round+2 token IDs quietly ──
                r2 = next_window(2, self._offset)
                if r2.slug not in self._rounds:
                    self._rounds[r2.slug] = RoundInfo(r2)
                    asyncio.create_task(
                        self._fetch_round(r2.slug),
                        name=f"prefetch-{r2.slug}"
                    )

                # ── Cleanup old rounds ──
                await self._cleanup_old_rounds()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Market monitor error: %s", exc)
                await asyncio.sleep(1.0)

    async def _perform_rollover(self, new_window: RoundWindow) -> None:
        """Switch active market to new_window."""
        old_slug = self._active_slug

        # Ensure new round is in registry
        if new_window.slug not in self._rounds:
            self._rounds[new_window.slug] = RoundInfo(new_window)

        info = self._rounds[new_window.slug]

        # Fetch if not yet fetched
        if info.market is None:
            logger.info("Rollover: fetching new round %s", new_window.slug)
            await self._fetch_round(new_window.slug)

        async with self._lock:
            self._active_slug = new_window.slug

        new_market = self._rounds[new_window.slug].market
        if new_market and self._on_new_market_cb:
            was_prewarmed = self._rounds[new_window.slug].subscribed
            self._on_new_market_cb(new_market, was_prewarmed)

        logger.info(
            "Rollover complete  %s → %s  (pre-warmed: %s)",
            (old_slug or "—")[-14:],
            new_window.slug[-14:],
            "✓" if self._rounds[new_window.slug].subscribed else "✗",
        )

    async def _prewarm_round(self, slug: str) -> None:
        """
        Pre-warm a future round:
        1. Fetch token IDs if not already fetched
        2. Mark as subscribed (triggers WS subscribe in bot.py via callback)
        3. Fire on_prewarm_cb so bot subscribes WS and fetches REST snapshot
        """
        info = self._rounds.get(slug)
        if info is None or info.subscribed:
            return

        if info.market is None:
            await self._fetch_round(slug)
            info = self._rounds.get(slug)

        if info and info.market:
            info.subscribed = True
            logger.info("Pre-warming next round: %s", slug[-14:])
            if self._on_prewarm_cb:
                self._on_prewarm_cb(info.market)

    # ─────────────────────────── Gamma API Fetch ───────────────────────────

    async def _fetch_round(self, slug: str) -> None:
        """Fetch token IDs and market metadata for a given slug from Gamma API."""
        info = self._rounds.get(slug)
        if info is None:
            return

        try:
            market = await self._query_gamma(slug)
            if market:
                info.market = market
                logger.debug("Fetched round metadata: %s", slug[-14:])
            else:
                logger.debug("Round not yet live on Gamma: %s", slug[-14:])
        except Exception as exc:
            logger.warning("Gamma fetch error for %s: %s", slug[-14:], exc)

    async def _query_gamma(self, slug: str) -> Optional[MarketTokenPair]:
        """Query Gamma API for a market by exact slug."""
        url    = f"{self._gamma_host}/markets"
        params = {"slug": slug}

        async with self._session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=8)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()

        markets = data if isinstance(data, list) else data.get("data", [])
        if not markets:
            return None

        m = markets[0]
        return self._parse_market(m, slug)

    def _parse_market(self, m: Dict[str, Any], slug: str) -> Optional[MarketTokenPair]:
        try:
            ctoken_ids = self._as_list(m.get("clobTokenIds") or m.get("clob_token_ids") or [])
            if len(ctoken_ids) < 2:
                return None

            outcomes = self._as_list(m.get("outcomes") or ["Up", "Down"])

            # Identify UP/DOWN token positions from outcomes list
            up_idx, down_idx = 0, 1
            for i, o in enumerate(outcomes):
                ol = str(o).lower()
                if ol in ("up", "yes"):
                    up_idx = i
                elif ol in ("down", "no"):
                    down_idx = i

            window  = window_from_slug(slug)
            end_ts  = window.end_time
            start_ts = window.start_time

            return MarketTokenPair(
                condition_id=m.get("conditionId") or m.get("condition_id", ""),
                up_token_id=ctoken_ids[up_idx],
                down_token_id=ctoken_ids[down_idx],
                tick_size=str(m.get("minimumTickSize") or m.get("minimum_tick_size") or "0.01"),
                neg_risk=bool(m.get("negRisk", True)),
                start_time=start_ts,
                end_time=end_ts,
                question=m.get("question", ""),
                slug=slug,
            )
        except Exception as exc:
            logger.debug("Parse failed for %s: %s", slug, exc)
            return None

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return [value]
            return parsed if isinstance(parsed, list) else [parsed]
        return list(value) if isinstance(value, tuple) else []

    async def _cleanup_old_rounds(self) -> None:
        """Remove rounds that ended more than 60s ago."""
        now = time.time()
        stale = [
            slug for slug, info in self._rounds.items()
            if info.window.end_time < now - 60
        ]
        for slug in stale:
            del self._rounds[slug]

    # ─────────────────────────── WS Event Handlers ───────────────────────────

    def handle_ws_message(self, msg: Dict[str, Any]) -> None:
        event_type = msg.get("event_type", "")
        if event_type == "new_market":
            asyncio.create_task(self._handle_new_market_ws(msg), name="ws-new-market")
        elif event_type == "market_resolved":
            asyncio.create_task(self._handle_resolved_ws(msg), name="ws-resolved")

    async def _handle_new_market_ws(self, msg: Dict[str, Any]) -> None:
        slug = msg.get("slug", "")
        if not slug.startswith("btc-updown-5m-"):
            return
        if slug not in self._rounds:
            self._rounds[slug] = RoundInfo(window_from_slug(slug))
        info = self._rounds[slug]
        if info.market is None:
            await self._fetch_round(slug)

    async def _handle_resolved_ws(self, msg: Dict[str, Any]) -> None:
        cond_id       = msg.get("market", "")
        winning_asset = msg.get("winning_asset_id", "")
        if self._on_resolved_cb:
            self._on_resolved_cb(cond_id, winning_asset)

    # ─────────────────────────── Accessors ───────────────────────────

    def get_round_info(self, slug: str) -> Optional[RoundInfo]:
        return self._rounds.get(slug)

    def all_subscribed_markets(self) -> List[MarketTokenPair]:
        """All markets currently subscribed (active + pre-warmed)."""
        return [
            info.market for info in self._rounds.values()
            if info.market is not None and info.subscribed
        ]

    def round_count(self) -> int:
        return len(self._rounds)
