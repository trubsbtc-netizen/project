from __future__ import annotations

import asyncio
import logging
import time
from typing import TYPE_CHECKING

from rich.align import Align
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

if TYPE_CHECKING:
    from core.runtime.bot import InstitutionalBTCPolyBot
    from core.markets.canonical import CanonicalRoundState

logger = logging.getLogger(__name__)


def _p(v: float) -> str:
    if v >= 100:
        return f"{v:.2f}"
    return f"{v:.4f}"


def _s(v: float) -> str:
    if v >= 0.01:
        return f"{v:.4f}"
    return f"{v:.6f}"


class TerminalUI:
    def __init__(self, bot: InstitutionalBTCPolyBot) -> None:
        self._bot = bot
        self._start_ts = time.monotonic()
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        layout = Layout()
        self._build_layout(layout)
        try:
            with Live(layout, refresh_per_second=10, screen=True) as live:
                while not self._stop.is_set():
                    await self._render(layout)
                    live.update(layout)
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    def _build_layout(self, layout: Layout) -> None:
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_column(
            Layout(name="body_top", ratio=1),
            Layout(name="body_bot", ratio=1),
        )
        layout["body"]["body_top"].split_row(
            Layout(name="feeds", ratio=2, minimum_size=30),
            Layout(name="orderbook", ratio=5, minimum_size=42),
        )
        layout["body"]["body_bot"].split_row(
            Layout(name="events", ratio=1),
            Layout(name="strategy", ratio=1),
        )

    async def _render(self, layout: Layout) -> None:
        state = await self._bot.market_state.current()
        layout["header"].update(self._header(state))
        layout["body"]["body_top"]["feeds"].update(self._feeds())
        layout["body"]["body_top"]["orderbook"].update(self._orderbook())
        layout["body"]["body_bot"]["events"].update(self._events())
        layout["body"]["body_bot"]["strategy"].update(self._strategy(state))
        layout["footer"].update(self._footer())

    def _header(self, state: CanonicalRoundState | None) -> Panel:
        uptime = time.monotonic() - self._start_ts
        h, r = divmod(int(uptime), 3600)
        m, s = divmod(r, 60)
        slug = state.market.slug if state else "awaiting market..."

        text = Text()
        text.append(" BTC POLY ", style="bold cyan")
        text.append(f"v{self._bot.config.polymarket.ptb_mode}", style="cyan")
        text.append("  \u2502  ", style="dim")
        text.append(slug, style="bold white")
        text.append("  \u2502  ", style="dim")
        text.append(f"uptime {h:02d}:{m:02d}:{s:02d}", style="bold white")
        text.append("  \u2502  ", style="dim")
        text.append("[Ctrl+C]", style="dim white")

        return Panel(Align.center(text), box=box.ROUNDED, border_style="cyan")

    def _feeds(self) -> Panel:
        now_ns = time.monotonic_ns()
        statuses = self._bot.health.status()
        table = Table.grid(padding=(0, 1))
        table.add_column()

        for fs in statuses.values():
            age_s = (now_ns - fs.last_msg_mono_ns) / 1e9 if fs.last_msg_mono_ns else 0
            is_stale = fs.stale(now_ns)

            if is_stale:
                indicator = "\u25d1"
                ind_style = "yellow"
                state = "STALE"
                state_style = "bold yellow"
            elif fs.connected:
                indicator = "\u25cf"
                ind_style = "green"
                state = "LIVE"
                state_style = "green"
            else:
                indicator = "\u25cb"
                ind_style = "red"
                state = "DOWN"
                state_style = "bold red"

            line = Text()
            line.append(f" {indicator} ", style=ind_style)
            line.append(f"{fs.name:<18}", style="bold white")
            line.append(f"{state:>6}", style=state_style)
            line.append(f"  {age_s:>5.1f}s  ", style="dim")
            line.append(f"rec:{fs.reconnects:>2}  err:{fs.errors}", style="dim")
            table.add_row(line)

        return Panel(
            Align.left(table),
            title="FEEDS",
            border_style="bright_blue",
            box=box.ROUNDED,
        )

    def _orderbook(self) -> Panel:
        books = self._bot.microstructure.books
        rows = [
            ("binance  ", books.binance.last),
            ("coinbase ", books.coinbase.last),
            ("poly UP  ", books.polymarket_up.last),
            ("poly DOWN", books.polymarket_down.last),
        ]
        text = Text()
        text.append(f"           {'BID':>10}  {'ASK':>10}  {'MID':>10}  {'SPREAD':>10}\n", style="bold underline")
        for label, tob in rows:
            if tob and tob.bid > 0:
                text.append(f"  {label}", style="bold white")
                text.append(f"  {_p(tob.bid):>10}", style="white")
                text.append(f"  {_p(tob.ask):>10}", style="white")
                text.append(f"  {_p(tob.mid):>10}", style="cyan")
                text.append(f"  {_s(tob.spread):>10}\n", style="yellow")
            else:
                text.append(f"  {label}  {'--':>10}  {'--':>10}  {'--':>10}  {'--':>10}\n", style="dim")

        sm = books.signal_mid
        text.append(f"\n  signal mid: {_p(sm)}", style="bold")
        text.append(f"  divergence: {books.exchange_divergence:.4f}", style="dim")

        return Panel(
            Align.left(text),
            title="ORDER BOOK",
            border_style="magenta",
            box=box.ROUNDED,
        )

    def _events(self) -> Panel:
        c = self._bot.counters
        text = Text()
        text.append(f"  Truth Ticks:   {c.truth_ticks:>10,}\n")
        text.append(f"  Exchange:       {c.exchange_events:>10,}\n")
        text.append(f"  Polymarket:     {c.poly_events:>10,}\n")
        text.append(f"  Evaluations:    {c.strategy_evaluations:>10,}\n")
        text.append(f"  Order Intents:  {c.order_intents:>10,}\n")
        text.append(f"  Orders Accepted:{c.orders_accepted:>10,}\n")
        text.append(f"  Rollovers:      {c.rollovers:>10,}\n")
        text.append(f"  Settlement Chk: {c.settlement_checks:>10,}")
        return Panel(
            Align.left(text),
            title="EVENTS",
            border_style="bright_green",
            box=box.ROUNDED,
        )

    def _strategy(self, state: CanonicalRoundState | None) -> Panel:
        estimate = self._bot.strategy.last_estimate
        sp = self._bot.health.stale_penalty()

        text = Text()
        if state is not None:
            m = state.market
            text.append(f"  bucket:     {m.bucket}\n", style="bold")
            text.append(f"  PTB:        {m.price_to_beat.value:.4f}\n", style="dim")
            if state.truth_price:
                text.append(f"  truth:      {state.truth_price.price:.4f}\n", style="dim")

        if estimate is not None:
            p_up = estimate.p_up
            direction = estimate.direction.name
            dir_color = "green" if direction == "UP" else "red"
            text.append(f"\n  P(UP):      ")
            text.append(f"{p_up:.4f}\n", style="bold cyan")
            text.append(f"  Conf:       ")
            text.append(f"{estimate.confidence:.4f}\n", style="bold")
            text.append(f"  Edge:       ")
            text.append(f"{estimate.edge:+.4f}\n", style="bold")
            text.append(f"  Entropy:    ")
            text.append(f"{estimate.entropy:.4f}\n", style="bold")
            text.append(f"  Signal:     ")
            text.append(f"{direction:>4}", style=f"bold {dir_color}")
        else:
            text.append("\n  awaiting data...", style="dim")

        text.append(f"\n\n  stale penalty: {sp:.2f}", style="bold yellow")
        return Panel(
            Align.left(text),
            title="STRATEGY",
            border_style="bright_yellow",
            box=box.ROUNDED,
        )

    def _footer(self) -> Panel:
        bot = self._bot
        now_ns = time.monotonic_ns()
        statuses = bot.health.status()
        bad_feeds = [name for name, fs in statuses.items() if fs.stale(now_ns)]

        text = Text()
        text.append("  mode: ", style="dim")
        mode_style = "bold yellow" if bot.config.execution.dry_run else "bold green"
        text.append(f"{'DRY RUN' if bot.config.execution.dry_run else 'LIVE'}", style=mode_style)
        text.append("  \u2502  ", style="dim")
        text.append("trading: ")
        trade_style = "green" if bot.config.execution.enable_live_trading else "red"
        text.append(f"{'ON' if bot.config.execution.enable_live_trading else 'OFF'}", style=trade_style)
        text.append("  \u2502  ", style="dim")
        if bad_feeds:
            text.append(f"stale: {', '.join(bad_feeds)}", style="bold yellow")
        else:
            text.append("all feeds nominal", style="green")

        return Panel(
            Align.left(text),
            box=box.ROUNDED,
            border_style="bright_black",
        )
