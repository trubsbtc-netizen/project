"""
Professional Terminal UI (TUI) for the Polymarket BTC 5-minute Trading Bot.

Uses the `rich` library to render a live, multi-panel dashboard directly in
the terminal. The dashboard auto-refreshes every cycle with live market data,
order books, decision engine output, risk metrics, and a scrolling event log.

Layout (top → bottom):
  ┌─ HEADER ──────────────────────────────────────────────────────────────┐
  │  Bot name · mode · uptime · cycle · last cycle time                  │
  ├─ ROW 1 ────────────────────────────────────────────────────────────────┤
  │  Market & Signals │  Inference & Regime  │  Polymarket Order Books    │
  ├─ ROW 2 ────────────────────────────────────────────────────────────────┤
  │  Decision Engine  │  Risk & PnL (MtM, trades, win-rate, drawdown)     │
  ├─ LOG ──────────────────────────────────────────────────────────────────┤
  │  Scrolling event log (last N lines, colour-coded by level)            │
  └────────────────────────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import logging
import json
import math
import os
import re
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Deque, Dict, List, Optional, Tuple

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.rule import Rule
from rich.style import Style
from rich.table import Table
from rich.text import Text

if TYPE_CHECKING:
    from polymarket_bot.bot_types import (
        MarketState,
        PolymarketOrderbook,
        Direction,
    )

# ── Colour palette ────────────────────────────────────────────────────────────
C_UP            = "#10B981"      # Emerald 500
C_DOWN          = "#EF4444"      # Coral/Red 500
C_NEUTRAL       = "#F59E0B"      # Amber 500
C_DIM           = "#64748B"      # Slate 500
C_GOOD          = "#10B981"
C_WARN          = "#F59E0B"
C_BAD           = "#EF4444"
C_ACCENT        = "#0EA5E9"      # Sky 500
C_HEADER        = "bold #38BDF8" # Sky 400
C_LABEL         = "#94A3B8"      # Slate 400
C_VALUE         = "bold #F1F5F9" # Slate 100
C_PANEL_BORDER  = "#334155"      # Slate 700
C_PANEL_BG      = ""

MAX_LOG_LINES = 200
REFRESH_RATE  = 4   # Rich live refresh per second


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}

# ─────────────────────────────────────────────────────────────────────────────
# Custom logging handler → feeds the scrolling log panel
# ─────────────────────────────────────────────────────────────────────────────

class TUILogHandler(logging.Handler):
    """Captures log records and stores them for the TUI log panel."""

    LEVEL_STYLES: Dict[int, str] = {
        logging.DEBUG:    "grey50",
        logging.INFO:     "bright_white",
        logging.WARNING:  "bold yellow",
        logging.ERROR:    "bold red",
        logging.CRITICAL: "bold red on white",
    }

    def __init__(self, maxlines: int = MAX_LOG_LINES) -> None:
        super().__init__()
        self._lines: Deque[Tuple[int, str]] = deque(maxlen=maxlines)
        self._active_round_key: str = ""
        self._version: int = 0
        self.setFormatter(logging.Formatter("%(message)s"))

    def set_active_round(self, round_key: str) -> None:
        if round_key and round_key != self._active_round_key:
            self._lines.clear()
            self._active_round_key = round_key
            self._version += 1

    @property
    def version(self) -> int:
        return self._version

    @staticmethod
    def _kv(message: str, key: str, default: str = "N/A") -> str:
        match = re.search(rf"(?:^|\s){re.escape(key)}=([^\s]+)", message)
        return match.group(1) if match else default

    @classmethod
    def _compact_message(cls, message: str) -> str:
        if message.startswith("TRADE_HISTORY "):
            try:
                item = json.loads(message[len("TRADE_HISTORY "):])
            except Exception:
                return message[:180]
            event = str(item.get("event", "")).upper()
            direction = item.get("direction", "N/A")
            if event == "OPEN":
                return (
                    f"OPEN {direction} "
                    f"entry={float(item.get('entry_price', 0.0) or 0.0):.3f} "
                    f"cost={float(item.get('cost_usdc', 0.0) or 0.0):.2f} "
                    f"edge={float(item.get('edge_at_entry_bps', 0.0) or 0.0):+.0f}bps"
                )
            if event == "SETTLE":
                pnl = float(item.get("pnl_usdc", 0.0) or 0.0)
                return (
                    "SETTLE "
                    f"{direction}->{item.get('actual_outcome', 'N/A')} "
                    f"pnl={pnl:+.2f}USDC "
                    f"btc={float(item.get('final_btc_price', 0.0) or 0.0):.2f} "
                    f"ptb={float(item.get('price_to_beat', 0.0) or 0.0):.2f}"
                )
            return f"TRADE {event}"

        if message.startswith(("SIGNAL OBSERVATION:", "OBSERVATION SIGNAL:")):
            valid = cls._kv(message, "valid")
            direction = cls._kv(message, "direction")
            obs = cls._kv(message, "obs")
            p_up = cls._kv(message, "p_up")
            conf = cls._kv(message, "conf")
            return f"OBS {direction} valid={valid} obs={obs} p_up={p_up} conf={conf}"

        if message.startswith(("SIGNAL ENTRY:", "ENTRY SIGNAL:")):
            action = cls._kv(message, "action")
            direction = cls._kv(message, "direction")
            obs = cls._kv(message, "obs")
            prob = cls._kv(message, "p")
            edge = cls._kv(message, "edge")
            price = cls._kv(message, "price")
            limit = cls._kv(message, "limit")
            size = cls._kv(message, "size")
            ptb = cls._kv(message, "ptb")
            tau = cls._kv(message, "tau")
            gate = cls._kv(message, "gate")
            return (
                f"ENTRY {action} {direction} p={prob} edge={edge} "
                f"price={price} limit={limit} size={size} PTB={ptb} tau={tau} gate={gate}"
            )

        if message.startswith("ORDER SUBMIT:"):
            mode = cls._kv(message, "mode")
            direction = cls._kv(message, "direction")
            size = cls._kv(message, "size")
            price = cls._kv(message, "price")
            edge = cls._kv(message, "edge")
            conf = cls._kv(message, "confidence")
            order_id = cls._kv(message, "order_id")
            return f"ORDER {mode} {direction} size={size} price={price} edge={edge} conf={conf}"

        if message.startswith("ORDER STATUS:"):
            mode = cls._kv(message, "mode")
            direction = cls._kv(message, "direction")
            status = cls._kv(message, "status")
            if status in {"submitted", "dry_run"}:
                return ""
            order_id = cls._kv(message, "order_id")
            return f"ORDER {mode} {direction} status={status} order={order_id}"

        if message.startswith("BOT STATUS:"):
            cycle = cls._kv(message, "cycle")
            tau = cls._kv(message, "tau")
            btc = cls._kv(message, "btc")
            ptb = cls._kv(message, "ptb")
            obs = cls._kv(message, "obs")
            decision = cls._kv(message, "decision")
            direction = cls._kv(message, "direction")
            prob = cls._kv(message, "p")
            open_pos = cls._kv(message, "open")
            upnl = cls._kv(message, "uPnL")
            return f"STATUS #{cycle} {decision} {direction} p={prob} BTC={btc} PTB={ptb} tau={tau} obs={obs} open={open_pos} uPnL={upnl}"

        if message.startswith("MARKET ROUND:"):
            ptb = cls._kv(message, "ptb")
            tau = cls._kv(message, "tau")
            reason = cls._kv(message, "reason")
            if "pending_ptb" in reason and ptb == "N/A":
                ptb = "pending"
            return f"ROUND {reason} PTB={ptb} tau={tau}"

        if message.startswith("ROUND LOCK:"):
            direction = cls._kv(message, "direction")
            probability = cls._kv(message, "p")
            conf = cls._kv(message, "conf")
            status = cls._kv(message, "status")
            return f"LOCK {direction} p={probability} conf={conf} {status}"

        if message.startswith(("DRY RUN:", "ORDER:")):
            mode = "DRY_RUN" if message.startswith("DRY RUN:") else "LIVE"
            body = message.split(":", 1)[1].strip()
            direction = body.split(" ", 1)[0] if body else "N/A"
            size = cls._kv(message, "size")
            price = cls._kv(message, "price")
            edge = cls._kv(message, "edge")
            conf = cls._kv(message, "confidence")
            status = cls._kv(message, "status")
            return f"ORDER {mode} {direction} status={status} size={size} price={price} edge={edge} conf={conf}"

        if message.startswith(("TRADE OPENED:", "POSITION OPENED:")):
            direction = cls._kv(message, "direction")
            entry = cls._kv(message, "entry")
            shares = cls._kv(message, "shares")
            cost = cls._kv(message, "cost")
            ptb = cls._kv(message, "ptb")
            return f"OPEN {direction} entry={entry} shares={shares} cost={cost} PTB={ptb}"

        if message.startswith(("TRADE SETTLED:", "POSITION SETTLED:")):
            direction = cls._kv(message, "direction")
            outcome = cls._kv(message, "outcome")
            pnl = cls._kv(message, "pnl")
            btc = cls._kv(message, "btc_final")
            ptb = cls._kv(message, "ptb")
            return f"SETTLE {direction}->{outcome} pnl={pnl} btc={btc} PTB={ptb}"

        if not message.startswith("Cycle "):
            return message

        cycle = message.split(":", 1)[0].replace("Cycle ", "#")
        if "no decision" in message:
            return f"{cycle} warming up - insufficient data"

        decision = cls._kv(message, "decision", "False")
        verdict = "EXECUTE" if decision == "True" else "NO_TRADE"
        direction = cls._kv(message, "direction")
        prob = cls._kv(message, "p")
        market = cls._kv(message, "market")
        edge = cls._kv(message, "edge")
        ptb = cls._kv(message, "ptb")
        tau = cls._kv(message, "tau")
        up_ask = cls._kv(message, "up_ask")
        down_ask = cls._kv(message, "down_ask")
        pnl = cls._kv(message, "uPnL")
        reason = cls._kv(message, "reason", "")
        return (
            f"{cycle} {verdict} {direction} "
            f"p={prob} mkt={market} edge={edge} "
            f"UP/DOWN={up_ask}/{down_ask} PTB={ptb} tau={tau} "
            f"uPnL={pnl} reason={reason}"
        )

    def emit(self, record: logging.LogRecord) -> None:
        try:
            raw = self.format(record)
            if not self._should_show(record, raw):
                return
            round_key = self._extract_round_key(raw)
            self.set_active_round(round_key)
            msg = self._compact_message(raw)
            if not msg:
                return
            timestamp = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
            name = record.name.replace("polymarket_bot.", "")
            line = f"{timestamp} {record.levelname:<7} {name:<18} {msg}"
            self._lines.append((record.levelno, line))
            self._version += 1
        except Exception:
            self.handleError(record)

    @classmethod
    def _should_show(cls, record: logging.LogRecord, message: str) -> bool:
        if message.startswith("BOT STATUS:"):
            return False
        if record.levelno >= logging.WARNING:
            return True
        prefixes = (
            "SIGNAL OBSERVATION:",
            "OBSERVATION SIGNAL:",
            "SIGNAL ENTRY:",
            "ENTRY SIGNAL:",
            "ORDER SUBMIT:",
            "ORDER STATUS:",
            "[DRY RUN] Order:",
            "DRY RUN:",
            "ORDER:",
            "TRADE OPENED:",
            "TRADE SETTLED:",
            "POSITION OPENED:",
            "POSITION SETTLED:",
            "MARKET ROUND:",
            "ROUND LOCK:",
            "CLAIM",
        )
        if message.startswith(prefixes):
            return True
        if "Order not counted as an open position" in message:
            return True
        return False

    @classmethod
    def _extract_round_key(cls, message: str) -> str:
        for key in ("slug", "market_slug"):
            match = re.search(rf"(?:^|\s){re.escape(key)}=([^\s]+)", message)
            if match:
                value = match.group(1)
                if value.startswith("btc-updown-5m-"):
                    return value
        return ""

    def render(self, height: int = 12) -> Text:
        """Return a rich Text block with the last `height` log lines."""
        lines = list(self._lines)[-height:]
        text = Text()
        if len(lines) >= 6:
            lines = lines[-6:]
        for level, msg in lines:
            style = self.LEVEL_STYLES.get(level, "white")
            text.append(msg + "\n", style=style)
        return text


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(value: Optional[float], decimals: int = 4, suffix: str = "") -> str:
    if value is None:
        return "[grey50]N/A[/grey50]"
    return f"{value:.{decimals}f}{suffix}"

def _fmt_pct(value: Optional[float]) -> str:
    if value is None:
        return "[grey50]N/A[/grey50]"
    return f"{value * 100:.1f}%"

def _fmt_pnl(value: Optional[float]) -> str:
    if value is None:
        return "[grey50]N/A[/grey50]"
    colour = C_GOOD if value >= 0 else C_BAD
    sign   = "+" if value >= 0 else ""
    return f"[{colour}]{sign}{value:.2f} USDC[/{colour}]"

def _dir_text(direction: Any) -> Text:
    try:
        val = direction.value if hasattr(direction, "value") else str(direction)
    except Exception:
        val = "N/A"
    if val == "UP":
        return Text(f"▲ {val}", style=f"bold {C_UP}")
    if val == "DOWN":
        return Text(f"▼ {val}", style=f"bold {C_DOWN}")
    return Text(val, style=C_NEUTRAL)

def _bool_icon(val: bool, true_style: str = C_GOOD, false_style: str = C_BAD) -> str:
    return f"[{true_style}]✔[/{true_style}]" if val else f"[{false_style}]✘[/{false_style}]"

def _uptime(start: float) -> str:
    delta = int(time.time() - start)
    h, rem = divmod(delta, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _short(text: Optional[str], max_len: int = 48) -> str:
    if not text:
        return "N/A"
    clean = str(text).replace("\n", " ").strip()
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3].rstrip() + "..."

def _primary_gate(reason: str) -> str:
    if not reason:
        return "N/A"
    parts = [part.strip() for part in reason.split("|") if part.strip()]
    preferred = (
        "OBS_WAIT",
        "OBS_INVALID",
        "CONFIRM_WAIT",
        "ROUND_SKIP",
        "entry_price_above_max",
        "below_min",
        "directional_probability_below_threshold",
        "confidence_below_threshold",
        "execution_rejected",
        "risk_",
        "RISK_",
        "DECOR_REJECT",
    )
    for token in preferred:
        for part in parts:
            if token in part:
                return part
    return parts[-2] if len(parts) >= 2 else parts[-1]

def _market_window_from_slug(slug: str) -> Tuple[Optional[float], Optional[float]]:
    prefix = "btc-updown-5m-"
    if not slug or not slug.startswith(prefix):
        return None, None
    try:
        start = float(int(slug[len(prefix):]))
    except ValueError:
        return None, None
    return start, start + 300.0

def _parse_market_end_timestamp(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return float(value.timestamp())

    text = str(value).strip()
    if not text:
        return None

    try:
        timestamp = float(text)
        if timestamp > 1.0e12:
            timestamp /= 1000.0
        if math.isfinite(timestamp) and timestamp > 0.0:
            return timestamp
    except (TypeError, ValueError):
        pass

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        timestamp = float(parsed.timestamp())
        return timestamp if math.isfinite(timestamp) and timestamp > 0.0 else None
    except (TypeError, ValueError):
        return None

def _time_to_settlement_from_state(
    state: Optional[Any],
    settlement_forecast: Optional[Any] = None,
    now: Optional[float] = None,
) -> Optional[float]:
    now_ts = time.time() if now is None else float(now)
    slug = getattr(state, "market_slug", "") if state is not None else ""
    _, round_end = _market_window_from_slug(slug)
    if round_end is None and state is not None:
        round_end = _parse_market_end_timestamp(getattr(state, "market_end_iso", ""))
    if round_end is not None:
        return float(max(0.0, round_end - now_ts))

    if settlement_forecast is not None:
        try:
            tau = float(getattr(settlement_forecast, "time_to_settlement_seconds"))
            if math.isfinite(tau) and tau >= 0.0:
                return tau
        except (TypeError, ValueError):
            pass

    return None

def _clock(timestamp: Optional[float]) -> str:
    if timestamp is None:
        return "N/A"
    return datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")

def _bar(value: float, width: int = 12, colour: str = "green") -> str:
    """ASCII-style bar chart for a 0-1 probability."""
    filled = max(0, min(width, round(value * width)))
    empty  = width - filled
    return f"[{colour}]{'█' * filled}[/{colour}][#1E293B]{'█' * empty}[/#1E293B]"

def _prob_bar(p_up: float, p_down: float, width: int = 16) -> str:
    total = p_up + p_down
    if total <= 0:
        p_up_norm = 0.5
    else:
        p_up_norm = p_up / total
    
    up_cells   = max(0, min(width, round(p_up_norm  * width)))
    down_cells = width - up_cells
    return (
        f"[{C_DOWN}]{'█' * down_cells}[/{C_DOWN}]"
        f"[{C_UP}]{'█' * up_cells}[/{C_UP}]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Panel builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_header(
    state: "MarketState",
    cycle: int,
    uptime_start: float,
    mode: str,
    strategy: str,
    last_cycle_ms: float,
) -> Panel:
    slug = _short(getattr(state, "market_slug", "") if state else "", 34)
    source = _short(getattr(state, "price_source", "") if state else "", 38)
    ptb = getattr(state, "price_to_beat", None) if state else None
    if state and state.settlement_forecast and state.settlement_forecast.price_to_beat:
        ptb = state.settlement_forecast.price_to_beat
    ptb_text = f"{ptb:,.2f}" if ptb else "N/A"

    mode_style = f"bold {C_BAD}" if mode == "LIVE" else f"bold {C_WARN}"

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style=f"bold {C_ACCENT}", justify="left")
    grid.add_column(style=C_VALUE,            justify="left")
    grid.add_column(style=f"bold {C_ACCENT}", justify="left")
    grid.add_column(style=C_VALUE,            justify="left")
    grid.add_column(style=f"bold {C_ACCENT}", justify="left")
    grid.add_column(style=C_VALUE,            justify="left")
    grid.add_column(style=f"bold {C_ACCENT}", justify="left")
    grid.add_column(style=C_VALUE,            justify="left")

    grid.add_row(
        "STRATEGY",  strategy,
        "MODE",      f"[{mode_style}]{mode}[/{mode_style}]",
        "MARKET",    f"[white]{slug}[/white]",
        "PTB",       f"[{C_ACCENT}]{ptb_text}[/{C_ACCENT}]",
    )
    grid.add_row(
        "SOURCE",    f"[grey70]{source}[/grey70]",
        "", "",
    )

    return Panel(
        grid,
        title=f"[{C_HEADER}] ◈  POLYMARKET BTC 5-MIN DIRECTIONAL BOT  ◈ [/{C_HEADER}]",
        border_style=C_PANEL_BORDER,
        padding=(0, 1),
    )


def _build_market_panel(state: "MarketState") -> Panel:
    """BTC price, PTB, tau, observation window, technical signal, round lock."""
    g = Table.grid(padding=(0, 1))
    g.add_column(style=C_LABEL,    min_width=18)
    g.add_column(style=C_VALUE, min_width=14)

    sf = state.settlement_forecast if state else None
    slug = getattr(state, "market_slug", "") if state else ""
    question = getattr(state, "market_question", "") if state else ""
    price_source = getattr(state, "price_source", "") if state else ""
    round_start, round_end = _market_window_from_slug(slug)
    ptb        = None
    if sf:
        ptb = sf.price_to_beat
    elif state:
        ptb = getattr(state, "price_to_beat", None)
    tau = _time_to_settlement_from_state(state, sf)

    btc_str = f"[bold {C_NEUTRAL}]{state.btc_price:,.2f} $[/bold {C_NEUTRAL}]" if (state and state.btc_price) else "[grey50]N/A[/grey50]"
    ptb_str = f"[{C_ACCENT}]{ptb:,.2f} $[/{C_ACCENT}]" if ptb else "[grey50]N/A[/grey50]"
    distance_str = "[grey50]N/A[/grey50]"
    if state and state.btc_price and ptb:
        distance = state.btc_price - ptb
        distance_colour = C_UP if distance >= 0 else C_DOWN
        distance_str = f"[{distance_colour}]{distance:+,.2f} $[/{distance_colour}]"

    if tau is not None:
        tau_colour = C_GOOD if tau > 60 else (C_WARN if tau > 20 else C_BAD)
        tau_str    = f"[{tau_colour}]{tau:.0f} s[/{tau_colour}]"
    else:
        tau_str = "[grey50]N/A[/grey50]"

    g.add_row(f"[{C_ACCENT}]── Current Round ──[/{C_ACCENT}]", "")
    g.add_row("  Market",           f"[white]{_short(slug, 38)}[/white]")
    if question:
        g.add_row("  Question",     f"[grey70]{_short(question, 38)}[/grey70]")
    if round_start is not None:
        g.add_row("  Window",       f"[white]{_clock(round_start)} - {_clock(round_end)}[/white]")
    if price_source:
        g.add_row("  BTC Source",   f"[grey70]{_short(price_source, 38)}[/grey70]")
    g.add_row("", "")

    g.add_row(f"[{C_ACCENT}]── BTC vs PTB ──[/{C_ACCENT}]", "")
    g.add_row("BTC Price",         btc_str)
    g.add_row("PTB",               ptb_str)
    g.add_row("BTC - PTB",         distance_str)
    g.add_row("Settlement Timer",  tau_str)
    g.add_row("", "")

    # Observation
    if sf:
        obs_ready = _bool_icon(sf.observation_ready)
        obs_valid = _bool_icon(sf.observation_valid)
        obs_p     = _fmt(sf.observation_p_up, 3)
        obs_conf  = _fmt(sf.observation_confidence, 2)
        obs_val   = _fmt(sf.observation_validation_score, 2)
        obs_s     = f"{sf.observation_seconds:.1f}s"
        g.add_row(f"[{C_ACCENT}]── Observation Window ──[/{C_ACCENT}]", "")
        g.add_row("  Ready / Valid",    f"{obs_ready}  {obs_valid}")
        g.add_row("  Observed",         f"[white]{obs_s}[/white]")
        g.add_row("  p(UP)",            f"[bold {C_UP}]{obs_p}[/bold {C_UP}]")
        g.add_row("  Confidence",       f"[white]{obs_conf}[/white]")
        g.add_row("  Validation Score", f"[white]{obs_val}[/white]")
        g.add_row("", "")

    # Technical
    ts_sig = state.technical_signal if state else None
    if ts_sig:
        t_valid = _bool_icon(ts_sig.is_valid)
        t_p     = _fmt(ts_sig.p_up, 3)
        t_conf  = _fmt(ts_sig.confidence, 2)
        t_cons  = _fmt(ts_sig.consensus_score, 2)
        t_tf    = ts_sig.dominant_timeframe or "N/A"
        g.add_row(f"[{C_ACCENT}]── Technical Momentum ──[/{C_ACCENT}]", "")
        g.add_row("  Valid",            t_valid)
        g.add_row("  p(UP)",            f"[bold {C_UP}]{t_p}[/bold {C_UP}]")
        g.add_row("  Confidence",       f"[white]{t_conf}[/white]")
        g.add_row("  Consensus",        f"[white]{t_cons}[/white]")
        g.add_row("  Dominant TF",      f"[{C_ACCENT}]{t_tf}[/{C_ACCENT}]")
        g.add_row("", "")

    # Round lock
    if sf:
        rl_locked = sf.round_direction_locked
        rl_dir    = _dir_text(sf.round_direction)
        rl_conf   = _fmt(sf.round_direction_confidence, 2)
        g.add_row(f"[{C_ACCENT}]── Round Direction Lock ──[/{C_ACCENT}]", "")
        g.add_row("  Locked",   _bool_icon(rl_locked, C_ACCENT, C_DIM))
        g.add_row("  Direction",str(rl_dir))
        g.add_row("  Confidence", f"[white]{rl_conf}[/white]")

    return Panel(g, title=f"[{C_HEADER}] ◈ MARKET [/{C_HEADER}]",
                 border_style=C_PANEL_BORDER, padding=(0, 1))


def _build_inference_panel(state: "MarketState") -> Panel:
    g = Table.grid(padding=(0, 1))
    g.add_column(style=C_LABEL,    min_width=18)
    g.add_column(style=C_VALUE, min_width=14)

    sf  = state.settlement_forecast if state else None
    bp  = state.bayesian_posterior  if state else None
    ks  = state.kalman_state        if state else None
    re  = state.regime_state        if state else None
    ve  = state.volatility_estimate if state else None
    mt  = state.multi_timescale     if state else None
    fp  = state.flow_metrics        if state else None
    op  = state.orderbook_pressure  if state else None

    # Settlement probability
    if sf:
        p_up   = sf.p_up_settlement
        p_down = sf.p_down_settlement
        bar    = _prob_bar(p_up, p_down, 22)
        g.add_row(f"[{C_ACCENT}]── Settlement Forecast ──[/{C_ACCENT}]", "")
        g.add_row("  p(UP) / p(DOWN)", f"[bold {C_UP}]{p_up:.3f}[/bold {C_UP}] / [bold {C_DOWN}]{p_down:.3f}[/bold {C_DOWN}]")
        g.add_row("  Direction bar",   bar)
        g.add_row("  Z-Score",         f"[white]{_fmt(sf.terminal_z_score, 2)}[/white]")
        g.add_row("  σ/√s",            f"[grey70]{sf.terminal_sigma_per_sqrt_second:.6f}[/grey70]")
        g.add_row("  Drift/s",         f"[grey70]{sf.terminal_drift_per_second:.8f}[/grey70]")
        g.add_row("", "")

    # Bayesian
    if bp:
        g.add_row(f"[{C_ACCENT}]── Bayesian Posterior ──[/{C_ACCENT}]", "")
        g.add_row("  p(UP)",    f"[bold {C_UP}]{bp.p_up:.4f}[/bold {C_UP}]")
        g.add_row("  p(DOWN)",  f"[bold {C_DOWN}]{bp.p_down:.4f}[/bold {C_DOWN}]")
        g.add_row("  Entropy",  f"[white]{bp.posterior_entropy:.3f}[/white]")
        g.add_row("  BF Up/Dn",f"[white]{_fmt(bp.bayes_factor_up_vs_down, 2)}[/white]")
        g.add_row("", "")

    # Kalman
    if ks:
        vel_c = C_UP if ks.velocity_estimate > 0 else C_DOWN
        acc_c = C_UP if ks.acceleration_estimate > 0 else C_DOWN
        g.add_row(f"[{C_ACCENT}]── Kalman Filter ──[/{C_ACCENT}]", "")
        g.add_row("  Velocity",     f"[{vel_c}]{ks.velocity_estimate:+.5f}[/{vel_c}]")
        g.add_row("  Acceleration", f"[{acc_c}]{ks.acceleration_estimate:+.5f}[/{acc_c}]")
        g.add_row("  Innovation",   f"[white]{ks.innovation:.4f}[/white]")
        g.add_row("", "")

    # Regime
    if re:
        regime_colour_map = {
            "CALM_TRENDING":       C_GOOD,
            "CALM_RANGE":          C_ACCENT,
            "VOLATILE_TRENDING":   C_WARN,
            "VOLATILE_RANGE":      C_WARN,
            "CRISIS":              C_BAD,
            "LIQUIDATION_CASCADE": f"bold {C_BAD}",
        }
        rname   = re.current_regime.name if hasattr(re.current_regime, "name") else str(re.current_regime)
        rcol    = regime_colour_map.get(rname, "white")
        rconf   = re.regime_confidence
        rbar    = _bar(rconf, 12, rcol)
        g.add_row(f"[{C_ACCENT}]── Market Regime ──[/{C_ACCENT}]", "")
        g.add_row("  Regime",     f"[{rcol}]{rname}[/{rcol}]")
        g.add_row("  Confidence", f"{rbar} [white]{rconf:.2f}[/white]")
        g.add_row("  Entropy",    f"[white]{re.regime_entropy:.3f}[/white]")
        g.add_row("", "")

    # Volatility
    if ve:
        g.add_row(f"[{C_ACCENT}]── Volatility ──[/{C_ACCENT}]", "")
        g.add_row("  Realized",   f"[white]{ve.realized_volatility:.5f}[/white]")
        g.add_row("  GARCH",      f"[white]{ve.garch_volatility:.5f}[/white]")
        g.add_row("  5-min Fcast",f"[white]{ve.volatility_forecast_5min:.5f}[/white]")
        g.add_row("", "")

    # Flow
    if fp:
        fi_c = C_UP if fp.flow_imbalance > 0 else C_DOWN
        g.add_row(f"[{C_ACCENT}]── Order Flow ──[/{C_ACCENT}]", "")
        g.add_row("  Flow Imbalance", f"[{fi_c}]{fp.flow_imbalance:+.3f}[/{fi_c}]")
        g.add_row("  Taker Ratio",    f"[white]{fp.taker_ratio:.3f}[/white]")
        g.add_row("  Aggression",     f"[white]{fp.taker_aggression_score:.3f}[/white]")

    # Multi-timescale
    if mt:
        g.add_row("", "")
        g.add_row(f"[{C_ACCENT}]── Multi-Timescale ──[/{C_ACCENT}]", "")
        g.add_row("  Consistency", f"[white]{mt.timescale_consistency:.3f}[/white]")
        g.add_row("  Dominant TF", f"[{C_ACCENT}]{mt.dominant_timescale}[/{C_ACCENT}]")
        g.add_row("  Short-term",  f"[white]{mt.short_term_signal:.4f}[/white]")
        g.add_row("  Medium-term", f"[white]{mt.medium_term_signal:.4f}[/white]")
        g.add_row("  Long-term",   f"[white]{mt.long_term_signal:.4f}[/white]")

    return Panel(g, title=f"[{C_HEADER}] ◈ INFERENCE [/{C_HEADER}]",
                 border_style=C_PANEL_BORDER, padding=(0, 1))


def _build_orderbook_panel(state: "MarketState") -> Panel:
    """Render Polymarket UP and DOWN order books side by side or stacked vertically."""
    ob_up   = state.polymarket_orderbook_up   if state else None
    ob_down = state.polymarket_orderbook_down if state else None

    def _ob_table(ob: Optional["PolymarketOrderbook"], label: str, colour: str) -> Table:
        t = Table(
            title=f"[bold {colour}]{label}[/bold {colour}]",
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style=f"bold {colour}",
            border_style=colour,
            padding=(0, 1),
            expand=True,
        )
        t.add_column("Side",     justify="left",  min_width=3)
        t.add_column("Price",    justify="right", min_width=5)
        t.add_column("Size",     justify="right", min_width=6)
        t.add_column("Notional", justify="right", min_width=6)
        t.add_column("Depth",    justify="left",  min_width=6)

        if ob is None:
            t.add_row("[grey50]No data[/grey50]", "", "", "", "")
            return t

        best_ask  = ob.best_ask  or 0.0
        best_bid  = ob.best_bid  or 0.0
        spread = max(0.0, best_ask - best_bid)
        mid = ob.mid_price
        bid_depth = sum(l.size for l in ob.bids[:10])
        ask_depth = sum(l.size for l in ob.asks[:10])
        imbalance = (bid_depth - ask_depth) / max(bid_depth + ask_depth, 1.0e-9)
        imb_colour = C_UP if imbalance >= 0 else C_DOWN
        level_count = f"{len(ob.bids)}/{len(ob.asks)}"

        t.add_row(
            "[grey50]BEST[/grey50]",
            f"[{colour}]{best_bid:.3f}/{best_ask:.3f}[/{colour}]",
            f"[white]spr {spread:.3f}[/white]",
            f"[grey70]mid {mid:.3f}[/grey70]" if mid else "[grey50]mid N/A[/grey50]",
            f"[grey70]{level_count}[/grey70]",
        )
        t.add_row(
            "[grey50]D10[/grey50]",
            f"[{C_UP}]{bid_depth:,.0f}[/{C_UP}]",
            f"[{C_DOWN}]{ask_depth:,.0f}[/{C_DOWN}]",
            f"[{imb_colour}]{imbalance:+.2f}[/{imb_colour}]",
            "",
        )

        visible_levels = ob.asks[:5] + ob.bids[:5]
        max_notional = max(
            (max(0.0, lvl.price) * max(0.0, lvl.size) for lvl in visible_levels),
            default=1.0,
        )

        asks = ob.asks[:5]
        for lvl in reversed(asks):
            notional = max(0.0, lvl.price) * max(0.0, lvl.size)
            bar_len = max(1, round(notional / max_notional * 6))
            bar = "█" * bar_len
            t.add_row(
                f"[{C_DOWN}]ASK[/{C_DOWN}]",
                f"[{C_DOWN}]{lvl.price:.3f}[/{C_DOWN}]",
                f"{lvl.size:,.1f}",
                f"{notional:,.0f}",
                f"[{C_DOWN}]{bar}[/{C_DOWN}]",
            )

        t.add_row("[grey50]----[/grey50]", "[grey50]spread[/grey50]", "", "", "")

        bids = ob.bids[:5]
        for lvl in bids:
            notional = max(0.0, lvl.price) * max(0.0, lvl.size)
            bar_len = max(1, round(notional / max_notional * 6))
            bar = "█" * bar_len
            t.add_row(
                f"[{C_UP}]BID[/{C_UP}]",
                f"[{C_UP}]{lvl.price:.3f}[/{C_UP}]",
                f"{lvl.size:,.1f}",
                f"{notional:,.0f}",
                f"[{C_UP}]{bar}[/{C_UP}]",
            )
        return t

    t_up   = _ob_table(ob_up,   "▲ UP CONTRACT", C_UP)
    t_down = _ob_table(ob_down, "▼ DOWN CONTRACT", C_DOWN)

    terminal_width = Console().width
    if terminal_width < 130:
        # Vertical stacking
        grid = Table.grid(padding=(0, 0), expand=True)
        grid.add_column()
        grid.add_row(t_up)
        grid.add_row(Rule(style=C_PANEL_BORDER))
        grid.add_row(t_down)
        content = grid
    else:
        # Side by side
        content = Columns([t_up, t_down], equal=True, expand=True)

    return Panel(content, title=f"[{C_HEADER}] ◈ ORDERBOOKS [/{C_HEADER}]",
                 border_style=C_PANEL_BORDER, padding=(0, 1))


def _build_decision_panel(state: "MarketState") -> Panel:
    g = Table.grid(padding=(0, 1))
    g.add_column(style=C_LABEL,    min_width=16)
    g.add_column(style=C_VALUE, min_width=18)

    td = state.trade_decision if state else None
    ee = td.execution_estimate if td else None
    sf = td.settlement_forecast if td else (state.settlement_forecast if state else None)
    rs = state.risk_state if state else None

    if td:
        should_trade = td.should_trade
        selected_price = sf.selected_market_price if sf and sf.selected_market_price is not None else td.market_price
        edge_colour = C_GOOD if td.edge_bps > 0 else C_BAD
        signal_name = td.signal_type.value if hasattr(td.signal_type, "value") else str(td.signal_type)
        live_position = float(getattr(rs, "current_position", 0.0) or 0.0) if rs else 0.0
        has_live_position = live_position > 0.0
        live_dir = _dir_text(getattr(rs, "current_position_direction", None)) if rs else Text("N/A")
        live_dir_plain = live_dir.plain if hasattr(live_dir, "plain") else str(live_dir)
        live_dir_label = "UP" if "UP" in live_dir_plain else ("DOWN" if "DOWN" in live_dir_plain else "N/A")
        live_dir_colour = C_UP if live_dir_label == "UP" else C_DOWN if live_dir_label == "DOWN" else C_DIM

        g.add_row(f"[{C_ACCENT}]── Bot Decision ──[/{C_ACCENT}]", "")
        if has_live_position:
            g.add_row("  TRADE?", f"[bold {C_WARN}]✔  HOLDING[/bold {C_WARN}]")
        else:
            g.add_row("  TRADE?",
                      f"[bold {C_UP if should_trade else C_BAD}]"
                      f"{'✔  YES  EXECUTE' if should_trade else '✘  NO TRADE'}"
                      f"[/bold {C_UP if should_trade else C_BAD}]")
        g.add_row("  Direction",    str(_dir_text(td.direction)))
        if sf:
            g.add_row("  Expected Settle", str(_dir_text(sf.expected_settlement_direction)))
            g.add_row("  Execution Side",  str(_dir_text(sf.execution_direction)))
        g.add_row("  Probability",  f"[bold white]{td.probability:.4f}[/bold white]")
        if sf:
            g.add_row("  p(UP/DOWN)", f"[{C_UP}]{sf.p_up_settlement:.3f}[/{C_UP}] / [{C_DOWN}]{sf.p_down_settlement:.3f}[/{C_DOWN}]")
            if sf.price_to_beat is not None:
                tau = _time_to_settlement_from_state(state, sf)
                tau_text = f"{tau:.0f}s" if tau is not None else "N/A"
                g.add_row("  PTB / Tau", f"[{C_ACCENT}]{sf.price_to_beat:,.2f}[/{C_ACCENT}] / [white]{tau_text}[/white]")
        g.add_row("  Market Price", f"[white]{selected_price:.4f}[/white]")
        g.add_row("  Edge",         f"[{edge_colour}]{td.edge_bps:+.0f} bps[/{edge_colour}]")
        g.add_row("  Confidence",   f"{_bar(td.confidence, 12, C_ACCENT)} [white]{td.confidence:.2f}[/white]")
        g.add_row("  Signal Size",  f"[bold white]{td.position_size:.2f} USDC[/bold white]")
        if has_live_position:
            g.add_row(
                "  Trading State",
                f"[{C_WARN}]HOLDING[/{C_WARN}] "
                f"[{live_dir_colour}]{live_dir_label}[/{live_dir_colour}] "
                f"{live_position:,.2f} USDC",
            )
        else:
            g.add_row("  Trading State", f"[{C_DIM}]FLAT[/{C_DIM}]")
        if has_live_position:
            g.add_row(
                "  Active Pos",
                f"[white]{live_position:,.2f} USDC[/white] "
                f"[{live_dir_colour}]{live_dir_label}[/{live_dir_colour}] "
                f"uPnL {_fmt_pnl(float(getattr(rs, 'unrealized_pnl', 0.0) or 0.0))}",
            )
        g.add_row("  Signal Type",  f"[{C_ACCENT}]{signal_name}[/{C_ACCENT}]")
        
        gate_status_str = _primary_gate(td.reason)
        if has_live_position and ("round_entry_already_taken" in gate_status_str or "ROUND_SKIP" in gate_status_str):
            gate_status_str = "HOLDING_POSITION"
        g.add_row("  Gate Status",  f"[white]{_short(gate_status_str, 54)}[/white]")
        g.add_row("  Fill Prob",    f"[white]{td.fill_probability:.2f}[/white]")
        g.add_row("  Slippage",     f"[white]{td.expected_slippage_bps:.1f} bps[/white]")
        g.add_row("", "")

        # Execution estimate
        if ee:
            g.add_row(f"[{C_ACCENT}]── Execution Estimate ──[/{C_ACCENT}]", "")
            g.add_row("  Limit Price",    f"[bold white]{ee.limit_price:.4f}[/bold white]")
            g.add_row("  Effective Price",f"[white]{ee.effective_price:.4f}[/white]")
            g.add_row("  Depth Notional", f"[white]{ee.visible_depth_notional:,.2f} USDC[/white]")
            g.add_row("  Exec Quality",
                      f"[{C_ACCENT}]{ee.execution_quality.value if hasattr(ee.execution_quality, 'value') else str(ee.execution_quality)}[/{C_ACCENT}]")
            g.add_row("", "")

        # Rejection reason
        reason = td.reason
        if reason:
            g.add_row(f"[{C_ACCENT}]── Gate Breakdown ──[/{C_ACCENT}]", "")
            parts = [part.strip() for part in reason.split("|") if part.strip()]
            for part in parts[:7]:
                colour = C_BAD if any(word in part.upper() for word in ("REJECT", "NO_TRADE", "SKIP", "SUPPRESS")) else C_DIM
                g.add_row("  •", f"[{colour}]{_short(part, 54)}[/{colour}]")
            if len(parts) > 7:
                g.add_row("  •", f"[grey50]+{len(parts) - 7} more gates in file log[/grey50]")

    else:
        g.add_row("[grey50]Awaiting first decision…[/grey50]", "")

    return Panel(g, title=f"[{C_HEADER}] ◈ DECISION [/{C_HEADER}]",
                 border_style=C_PANEL_BORDER, padding=(0, 1))


def _build_risk_panel(state: "MarketState", session_stats: Dict) -> Panel:
    """Risk management + PnL mark-to-market + win-rate + trade stats."""
    g = Table.grid(padding=(0, 1))
    g.add_column(style=C_LABEL,     min_width=18)
    g.add_column(style=C_VALUE, min_width=16)

    rs = state.risk_state if state else None
    ss = session_stats

    # ── PnL ──────────────────────────────────────────────────────────────
    realized   = ss.get("realized_pnl", 0.0)
    unrealized = ss.get("unrealized_pnl", 0.0)
    total_pnl  = ss.get("total_pnl", realized + unrealized)
    equity     = ss.get("portfolio_equity", ss.get("current_capital", 0.0))
    initial    = ss.get("initial_capital", 1000.0)
    available  = ss.get("available_capital", 0.0)
    total_ret  = (equity - initial) / max(0.01, initial) * 100.0
    open_count = ss.get("open_positions", 0)
    open_cost  = ss.get("open_position_cost", ss.get("current_position", 0.0))
    open_value = ss.get("open_position_value", 0.0)

    ret_c = C_GOOD if total_ret >= 0 else C_BAD

    g.add_row(f"[{C_ACCENT}]── P&L (Mark-to-Market) ──[/{C_ACCENT}]", "")
    g.add_row("  Equity (MtM)",    f"[bold white]{equity:,.2f} USDC[/bold white]")
    g.add_row("  Total P&L",       _fmt_pnl(total_pnl))
    g.add_row("  Realized P&L",    _fmt_pnl(realized))
    g.add_row("  Unrealized P&L",  _fmt_pnl(unrealized))
    g.add_row("  Total Return",    f"[{ret_c}]{'+' if total_ret >= 0 else ''}{total_ret:.2f}%[/{ret_c}]")
    g.add_row("  Available Cap",   f"[white]{available:,.2f} USDC[/white]")
    g.add_row("  Open Position",   f"[white]{open_count} pos / {open_cost:,.2f} cost / {open_value:,.2f} value[/white]")
    g.add_row("", "")

    open_details = list(ss.get("open_position_details", []) or [])[-3:]
    if open_details:
        g.add_row(f"[{C_ACCENT}]── Active Positions ──[/{C_ACCENT}]", "")
        active_forecast_dir = None
        if state and state.settlement_forecast:
            active_forecast_dir = state.settlement_forecast.expected_settlement_direction.value

        for idx, item in enumerate(open_details, start=1):
            direction = str(item.get("direction", "N/A"))
            direction_colour = C_UP if direction == "UP" else C_DOWN if direction == "DOWN" else C_DIM
            cost = float(item.get("cost_usdc", 0.0) or 0.0)
            entry = float(item.get("entry_price", 0.0) or 0.0)
            mark = float(item.get("mark_price", 0.0) or 0.0)
            pnl = float(item.get("unrealized_pnl", 0.0) or 0.0)

            alignment = "N/A"
            alignment_colour = C_DIM
            if active_forecast_dir in {"UP", "DOWN"} and direction in {"UP", "DOWN"}:
                if direction == active_forecast_dir:
                    alignment = "searah"
                    alignment_colour = C_GOOD
                else:
                    alignment = f"berlawanan signal {active_forecast_dir}"
                    alignment_colour = C_WARN

            label = f"  {direction}" if len(open_details) == 1 else f"  {direction} #{idx}"
            g.add_row(
                f"[{direction_colour}]{label}[/{direction_colour}]",
                f"[white]{cost:.2f} USDC[/white] | entry [white]{entry:.3f}[/white] -> mark [white]{mark:.3f}[/white]",
            )
            g.add_row(
                "  uPnL / Status",
                f"{_fmt_pnl(pnl)} | [{alignment_colour}]{alignment}[/{alignment_colour}]",
            )
        g.add_row("", "")

    # ── Awaiting Settlement ──────────────────────────────────────────────
    awaiting_details = list(ss.get("awaiting_settlement_details", []) or [])
    if awaiting_details:
        g.add_row(f"[{C_WARN}]── Awaiting Settlement ──[/{C_WARN}]", "")
        for item in awaiting_details[-5:]:
            direction = str(item.get("direction", "N/A"))
            direction_colour = C_UP if direction == "UP" else C_DOWN if direction == "DOWN" else C_DIM
            cost = float(item.get("cost_usdc", 0.0) or 0.0)
            entry = float(item.get("entry_price", 0.0) or 0.0)
            ptb = item.get("price_to_beat")
            ptb_text = f"{ptb:.2f}" if ptb is not None and ptb > 0 else "N/A"
            slug = str(item.get("market_slug", "") or "")
            slug_short = slug[-20:] if len(slug) > 20 else slug

            g.add_row(
                f"  [{direction_colour}]{direction}[/{direction_colour}] [{C_DIM}]{slug_short}[/{C_DIM}]",
                f"[white]{cost:.2f} USDC[/white] @ [white]{entry:.3f}[/white] PTB={ptb_text}",
            )
            g.add_row(
                "  Status",
                f"[{C_WARN}]menunggu settlement resmi Polymarket[/{C_WARN}]",
            )
        g.add_row("", "")

    # ── Trade Stats ───────────────────────────────────────────────────────
    total_trades  = ss.get("bot_total_trades", ss.get("total_trades", 0))
    settled_trades = ss.get("settled_trades", ss.get("total_trades", 0))
    attempts      = ss.get("order_attempts", ss.get("total_order_attempts", 0))
    suppressed    = ss.get("suppressed_cycles", ss.get("total_suppressed", 0))
    win_count     = ss.get("win_count", 0)
    loss_count    = ss.get("loss_count", 0)
    win_rate      = ss.get("win_rate", 0.0)
    profit_factor = ss.get("profit_factor", 0.0)
    avg_pnl       = ss.get("avg_pnl_per_trade", 0.0)
    sharpe        = ss.get("sharpe_ratio", 0.0)
    consec_losses = ss.get("consecutive_losses", 0)
    consec_wins   = ss.get("consecutive_wins", 0)

    wr_c   = C_GOOD if win_rate >= 0.55 else (C_WARN if win_rate >= 0.45 else C_BAD)
    pf_c   = C_GOOD if profit_factor >= 1.0 else C_BAD
    wr_bar = _bar(win_rate, 14, wr_c)

    g.add_row(f"[{C_ACCENT}]── Trade Statistics ──[/{C_ACCENT}]", "")
    g.add_row("  Total Trades",    f"[bold white]{total_trades} opened / {settled_trades} settled[/bold white]")
    g.add_row("  Attempts / Skips",f"[white]{attempts} / {suppressed}[/white]")
    g.add_row("  Wins / Losses",   f"[{C_GOOD}]{win_count}[/{C_GOOD}] / [{C_BAD}]{loss_count}[/{C_BAD}]")
    g.add_row("  Win Rate",        f"{wr_bar} [{wr_c}]{win_rate*100:.1f}%[/{wr_c}]")
    g.add_row("  Profit Factor",   f"[{pf_c}]{profit_factor:.2f}[/{pf_c}]")
    g.add_row("  Avg PnL/Trade",   _fmt_pnl(avg_pnl))
    g.add_row("  Sharpe (session)",f"[white]{sharpe:.2f}[/white]")
    g.add_row("  Consec Wins",     f"[{C_GOOD}]{consec_wins}[/{C_GOOD}]")
    g.add_row("  Consec Losses",   f"[{C_BAD}]{consec_losses}[/{C_BAD}]")
    g.add_row("", "")

    recent_trades = list(ss.get("recent_trade_history", []) or [])[-4:]
    if recent_trades:
        g.add_row(f"[{C_ACCENT}]── Trade History ──[/{C_ACCENT}]", "")
        hist_table = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style=f"bold {C_ACCENT}",
            border_style=C_PANEL_BORDER,
            padding=(0, 1),
            expand=True,
        )
        hist_table.add_column("Time",  style=C_DIM)
        hist_table.add_column("Event")
        hist_table.add_column("Dir")
        hist_table.add_column("PnL/Cost", justify="right")

        for item in reversed(recent_trades):
            event = str(item.get("event", "")).upper()
            direction = str(item.get("direction", "N/A"))
            dir_colour = C_UP if direction == "UP" else C_DOWN if direction == "DOWN" else C_DIM
            timestamp = str(item.get("timestamp_utc", ""))[11:19] or "N/A"
            if event == "OPEN":
                cost = float(item.get("cost_usdc", 0.0) or 0.0)
                hist_table.add_row(
                    timestamp,
                    f"[{C_ACCENT}]OPEN[/{C_ACCENT}]",
                    f"[{dir_colour}]{direction}[/{dir_colour}]",
                    f"{cost:.1f}",
                )
            elif event == "SETTLE":
                pnl = float(item.get("pnl_usdc", 0.0) or 0.0)
                win = bool(item.get("is_win", False))
                colour = C_GOOD if win else C_BAD
                hist_table.add_row(
                    timestamp,
                    f"[{colour}]SETTLE[/{colour}]",
                    f"[{dir_colour}]{direction}[/{dir_colour}]",
                    _fmt_pnl(pnl),
                )
        g.add_row(hist_table, "")
        g.add_row("", "")

    # ── Risk State ────────────────────────────────────────────────────────
    if rs:
        dd_current = rs.current_drawdown
        dd_max     = rs.max_drawdown
        risk_score = rs.risk_score
        cooldown   = rs.is_in_cooldown
        stop       = rs.should_stop_trading

        dd_c  = C_GOOD if dd_current < 0.05 else (C_WARN if dd_current < 0.10 else C_BAD)
        rs_c  = C_GOOD if risk_score < 0.35  else (C_WARN if risk_score < 0.65 else C_BAD)
        dd_bar = _bar(min(dd_current / 0.20, 1.0), 12, dd_c)
        rs_bar = _bar(risk_score, 12, rs_c)

        g.add_row(f"[{C_ACCENT}]── Risk Management ──[/{C_ACCENT}]", "")
        g.add_row("  Risk Score",     f"{rs_bar} [{rs_c}]{risk_score:.2f}[/{rs_c}]")
        g.add_row("  Drawdown",       f"{dd_bar} [{dd_c}]{dd_current:.2%}[/{dd_c}]")
        g.add_row("  Max Drawdown",   f"[{C_BAD}]{dd_max:.2%}[/{C_BAD}]")
        g.add_row("  Cooldown",       f"[{C_WARN if cooldown else C_DIM}]{'ACTIVE' if cooldown else 'OFF'}[/{C_WARN if cooldown else C_DIM}]")
        if cooldown and rs.cooldown_remaining_seconds > 0:
            g.add_row("  Cooldown Rem",  f"[{C_WARN}]{rs.cooldown_remaining_seconds:.0f}s[/{C_WARN}]")
        g.add_row("  Trading Halted", f"[{C_BAD if stop else C_DIM}]{'⛔ YES' if stop else 'OK'}[/{C_BAD if stop else C_DIM}]")

    return Panel(g, title=f"[{C_HEADER}] ◈ RISK & PNL [/{C_HEADER}]",
                 border_style=C_PANEL_BORDER, padding=(0, 1))


def _build_log_panel(handler: TUILogHandler, height: int = 10) -> Panel:
    log_text = handler.render(height=height)
    return Panel(log_text,
                 title=f"[{C_HEADER}] ◈ EVENT LOG [/{C_HEADER}]",
                 border_style=C_PANEL_BORDER,
                 padding=(0, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Main TUI class
# ─────────────────────────────────────────────────────────────────────────────

class ConsoleTUI:
    """
    Live Rich dashboard for the Polymarket bot.

    Usage
    -----
    tui = ConsoleTUI(mode="DRY RUN", strategy="v3.5-settlement")
    tui.start()
    ...
    tui.update(state, cycle_count, last_cycle_time, session_stats)
    ...
    tui.stop()
    """

    def __init__(self, mode: str = "DRY RUN", strategy: str = "BTC-5m") -> None:
        self.mode          = mode
        self.strategy      = strategy
        self._start_time   = time.time()
        self._cycle        = 0
        self._last_cycle_t = 0.0
        self._state: Optional["MarketState"] = None
        self._session_stats: Dict = {}

        self._console  = Console(highlight=False)
        self._handler  = TUILogHandler(maxlines=MAX_LOG_LINES)
        self._handler.setLevel(logging.DEBUG)

        self._live: Optional[Live] = None
        self._screen = _env_flag("TUI_ALT_SCREEN", False)

    # ── public interface ──────────────────────────────────────────────────

    def get_log_handler(self) -> TUILogHandler:
        """Return the log handler that should be attached to the root logger."""
        return self._handler

    def start(self) -> None:
        """Start the live display context."""
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=REFRESH_RATE,
            screen=self._screen,
            auto_refresh=False,
            transient=False,
            redirect_stdout=False,
            redirect_stderr=False,
        )
        self._live.start()
        self._live.refresh()

    def stop(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    def update(
        self,
        state: Optional["MarketState"],
        cycle: int,
        last_cycle_time: float,
        session_stats: Dict,
    ) -> None:
        """Called at the end of each bot cycle to refresh the dashboard."""
        self._state         = state
        self._cycle         = cycle
        self._last_cycle_t  = last_cycle_time
        self._session_stats = session_stats
        if state is not None:
            self._handler.set_active_round(getattr(state, "market_slug", ""))
        if self._live:
            try:
                self._live.update(self._render(), refresh=True)
            except Exception as exc:
                try:
                    self._live.update(self._render_error(exc), refresh=True)
                except Exception:
                    pass
                raise

    # ── rendering ─────────────────────────────────────────────────────────

    def _render_error(self, exc: Exception) -> Panel:
        message = Text()
        message.append("TUI render failed\n", style="bold red")
        message.append(f"{type(exc).__name__}: {exc}\n", style="red")
        message.append("Bot loop masih berjalan; detail ada di polymarket_bot.log.", style="grey70")
        return Panel(
            message,
            title="TUI Error",
            border_style="red",
            padding=(1, 2),
        )

    def _render_compact(self, state: Optional["MarketState"], stats: Dict) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="summary", size=10),
            Layout(name="log"),
        )
        
        g = Table.grid(padding=(0, 2))
        g.add_column(style=C_LABEL)
        g.add_column(style=C_VALUE)
        g.add_column(style=C_LABEL)
        g.add_column(style=C_VALUE)
        
        btc_str = f"[bold {C_NEUTRAL}]{state.btc_price:,.2f} $[/bold {C_NEUTRAL}]" if (state and state.btc_price) else "N/A"
        sf = state.settlement_forecast if state else None
        ptb = sf.price_to_beat if (sf and sf.price_to_beat) else (state.price_to_beat if state else None)
        ptb_str = f"{ptb:,.2f} $" if ptb else "N/A"
        tau = _time_to_settlement_from_state(state, sf)
        tau_str = f"{tau:.0f} s" if tau is not None else "N/A"
        
        realized = stats.get("realized_pnl", 0.0)
        unrealized = stats.get("unrealized_pnl", 0.0)
        total_pnl = stats.get("total_pnl", realized + unrealized)
        win_rate = stats.get("win_rate", 0.0)
        open_count = stats.get("open_positions", 0)
        
        halted = "OK"
        if state and state.risk_state:
            halted = "⛔ HALTED" if state.risk_state.should_stop_trading else "OK"
            
        g.add_row(
            "BTC Price:", btc_str,
            "Total PnL:", _fmt_pnl(total_pnl)
        )
        g.add_row(
            "PTB:", ptb_str,
            "Tau:", tau_str
        )
        g.add_row(
            "Active Pos:", f"{open_count} pos",
            "Realized PnL:", _fmt_pnl(realized)
        )
        g.add_row(
            "Risk Status:", halted,
            "Win Rate:", f"{win_rate * 100:.1f}%"
        )
        g.add_row(
            "Mode:", self.mode,
            "", ""
        )
        
        summary_panel = Panel(
            g,
            title=f"[{C_HEADER}] ◈ COMPACT SUMMARY [/{C_HEADER}]",
            border_style=C_PANEL_BORDER,
            padding=(1, 2)
        )
        
        layout["header"].update(
            _build_header(state, self._cycle, self._start_time,
                          self.mode, self.strategy, self._last_cycle_t)
        )
        layout["summary"].update(summary_panel)
        layout["log"].update(
            _build_log_panel(self._handler, height=max(5, self._console.height - 18))
        )
        return layout

    def _render(self) -> Layout:
        state = self._state
        stats = self._session_stats
        
        width = self._console.width
        height = self._console.height
        
        if width < 96 or height < 28:
            return self._render_compact(state, stats)

        layout = Layout()
        layout.split_column(
            Layout(name="header", size=5),
            Layout(name="body"),
            Layout(name="log",    size=14),
        )
        layout["body"].split_row(
            Layout(name="left",   ratio=3),
            Layout(name="center", ratio=3),
            Layout(name="right",  ratio=4),
        )
        layout["left"].split_column(
            Layout(name="market", ratio=2),
            Layout(name="inference", ratio=2),
        )
        layout["right"].split_column(
            Layout(name="orderbook"),
        )
        layout["center"].split_column(
            Layout(name="decision", ratio=1),
            Layout(name="risk",     ratio=1),
        )

        layout["header"].update(
            _build_header(state, self._cycle, self._start_time,
                          self.mode, self.strategy, self._last_cycle_t)
        )
        layout["market"].update(
            _build_market_panel(state) if state else
            Panel("[grey50]Initializing…[/grey50]",
                  title=f"[{C_HEADER}] ◈ MARKET [/{C_HEADER}]",
                  border_style=C_PANEL_BORDER)
        )
        layout["inference"].update(
            _build_inference_panel(state) if state else
            Panel("[grey50]Initializing…[/grey50]",
                  title=f"[{C_HEADER}] ◈ INFERENCE [/{C_HEADER}]",
                  border_style=C_PANEL_BORDER)
        )
        layout["center"]["decision"].update(
            _build_decision_panel(state) if state else
            Panel("[grey50]Initializing…[/grey50]",
                  title=f"[{C_HEADER}] ◈ DECISION [/{C_HEADER}]",
                  border_style=C_PANEL_BORDER)
        )
        layout["center"]["risk"].update(
            _build_risk_panel(state, stats) if stats else
            Panel("[grey50]Initializing…[/grey50]",
                  title=f"[{C_HEADER}] ◈ RISK & PNL [/{C_HEADER}]",
                  border_style=C_PANEL_BORDER)
        )
        layout["right"]["orderbook"].update(
            _build_orderbook_panel(state) if state else
            Panel("[grey50]Initializing…[/grey50]",
                  title=f"[{C_HEADER}] ◈ ORDERBOOKS [/{C_HEADER}]",
                  border_style=C_PANEL_BORDER)
        )
        layout["log"].update(
            _build_log_panel(self._handler, height=11)
        )

        return layout
