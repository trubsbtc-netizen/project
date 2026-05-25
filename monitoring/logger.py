"""
Professional logging for Polymarket BTC UP/DOWN 5M bot.

Two completely separate output channels:

  CONSOLE  → clean, human-readable, colour-coded
             timestamp  │  level badge  │  message only
             NO raw field dumps. ever.

  FILE     → newline-delimited JSON (logs/bot.jsonl, logs/trades.jsonl)
             every field explicit, machine-parseable

TradeLogger methods format their own message strings.
The console formatter prints ONLY record.getMessage() — no extras.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
import time
from collections import defaultdict
from decimal import Decimal
from typing import Any, Dict, Optional


# ─────────────────────────────────────────────────────────────────
#  Terminal colour helpers
# ─────────────────────────────────────────────────────────────────

_TTY = sys.stdout.isatty() if hasattr(sys.stdout, "isatty") else False

def _c(code: str) -> str:
    return f"\033[{code}m" if _TTY else ""

RST   = _c("0")
BOLD  = _c("1")
DIM   = _c("2")

# Named colours (256-colour)
SKY   = _c("38;5;117")   # info
AMB   = _c("38;5;220")   # warning
CRL   = _c("38;5;203")   # error
RED   = _c("38;5;196")   # critical
GRY   = _c("38;5;244")   # debug
GRN   = _c("38;5;84")    # win / up
MGT   = _c("38;5;213")   # down / loss
CYN   = _c("38;5;87")    # slug / accent
GLD   = _c("38;5;226")   # price
WHT   = _c("38;5;255")   # message text

_LEVEL_CLR = {
    "DEBUG":    GRY,
    "INFO":     SKY,
    "WARNING":  AMB,
    "ERROR":    CRL,
    "CRITICAL": RED,
}

_LEVEL_LBL = {
    "DEBUG":    "DBG",
    "INFO":     "INF",
    "WARNING":  "WRN",
    "ERROR":    "ERR",
    "CRITICAL": "CRT",
}


# ─────────────────────────────────────────────────────────────────
#  Console Formatter  — message only, no field dumps
# ─────────────────────────────────────────────────────────────────

class _ConsoleFormatter(logging.Formatter):
    """
    Format:  HH:MM:SS.mmm  LVL  message

    Rule: prints record.getMessage() verbatim.
    Extra fields are NEVER appended — each TradeLogger method
    is responsible for embedding what it wants in the message string.
    """

    def format(self, record: logging.LogRecord) -> str:
        t    = time.localtime(record.created)
        ms   = int(record.msecs)
        ts   = f"{DIM}{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ms:03d}{RST}"

        lvl  = record.levelname
        clr  = _LEVEL_CLR.get(lvl, "")
        lbl  = f"{clr}{BOLD}{_LEVEL_LBL.get(lvl, lvl)}{RST}"

        msg  = record.getMessage()

        # One-line exception summary only (no traceback on console)
        if record.exc_info and record.exc_info[1]:
            exc  = record.exc_info[1]
            msg += f"  {CRL}({type(exc).__name__}: {exc}){RST}"

        return f"{ts}  {lbl}  {msg}"


# ─────────────────────────────────────────────────────────────────
#  JSON File Formatter  — explicit fields, no omissions
# ─────────────────────────────────────────────────────────────────

_JSON_SKIP = frozenset({
    "name","msg","args","levelname","levelno","pathname","filename",
    "module","exc_info","exc_text","stack_info","lineno","funcName",
    "created","msecs","relativeCreated","thread","threadName",
    "processName","process","message","taskName",
})

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        doc: Dict[str, Any] = {
            "ts":     int(record.created * 1000),
            "level":  record.levelname,
            "logger": record.name,
            "msg":    record.getMessage(),
        }
        for k, v in record.__dict__.items():
            if k not in _JSON_SKIP and not k.startswith("_"):
                try:
                    json.dumps(v)
                    doc[k] = v
                except (TypeError, ValueError):
                    doc[k] = str(v)
        if record.exc_info and record.exc_info[1]:
            e = record.exc_info[1]
            doc["exception"] = {
                "type":  type(e).__name__,
                "msg":   str(e),
                "trace": self.formatException(record.exc_info),
            }
        return json.dumps(doc, default=str, separators=(",",":"))


# ─────────────────────────────────────────────────────────────────
#  Setup
# ─────────────────────────────────────────────────────────────────

def setup_logging(log_level: str = "INFO", log_dir: str = "logs") -> None:
    os.makedirs(log_dir, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)

    numeric = getattr(logging, log_level.upper(), logging.INFO)

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(numeric)
    ch.setFormatter(_ConsoleFormatter())
    root.addHandler(ch)

    # Rotating JSON file
    fh = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "bot.jsonl"),
        maxBytes=20 * 1024 * 1024, backupCount=5, encoding="utf-8",
    )
    fh.setLevel(numeric)
    fh.setFormatter(_JsonFormatter())
    root.addHandler(fh)

    for noisy in ("aiohttp", "asyncio", "websockets", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ─────────────────────────────────────────────────────────────────
#  Visual helpers used by TradeLogger
# ─────────────────────────────────────────────────────────────────

def _slug_tag(slug: str) -> str:
    """Last 14 chars of slug, coloured."""
    tag = slug[-14:] if len(slug) >= 14 else slug
    return f"{CYN}{tag}{RST}"

def _dir_tag(direction: str) -> str:
    clr = GRN if direction.upper() == "UP" else MGT
    return f"{clr}{BOLD}{direction.upper():<4}{RST}"

def _pnl_tag(pnl: float) -> str:
    if pnl >= 0:
        return f"{GRN}+${pnl:.2f}{RST}"
    return f"{CRL}-${abs(pnl):.2f}{RST}"

def _roi_tag(roi: float) -> str:
    if roi >= 0:
        return f"{GRN}+{roi:.1f}%{RST}"
    return f"{CRL}{roi:.1f}%{RST}"

def _hms(seconds: float) -> str:
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def _ms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:01d}m{s:02d}s"


# ─────────────────────────────────────────────────────────────────
#  TradeLogger
# ─────────────────────────────────────────────────────────────────

class TradeLogger:
    """
    Dedicated structured logger for all trade-lifecycle events.

    Each method builds its own complete, self-contained message string
    that reads clearly on the console WITHOUT any field dumps.
    JSON payloads go to trades.jsonl via the file handler only.
    """

    def __init__(self, log_dir: str = "logs") -> None:
        os.makedirs(log_dir, exist_ok=True)

        self._log = logging.getLogger("trade")
        self._log.propagate = False
        self._log.setLevel(logging.DEBUG)
        trade_file_level = getattr(
            logging,
            os.getenv("TRADE_LOG_LEVEL", "INFO").upper(),
            logging.INFO,
        )
        trade_console_level = getattr(
            logging,
            os.getenv("TRADE_CONSOLE_LEVEL", os.getenv("LOG_LEVEL", "INFO")).upper(),
            logging.INFO,
        )

        # Console — message only
        if not self._log.handlers:
            ch = logging.StreamHandler(sys.stdout)
            ch.setLevel(trade_console_level)
            ch.setFormatter(_ConsoleFormatter())
            self._log.addHandler(ch)

            # Dedicated JSON file for trade events
            fh = logging.handlers.RotatingFileHandler(
                os.path.join(log_dir, "trades.jsonl"),
                maxBytes=20 * 1024 * 1024, backupCount=10, encoding="utf-8",
            )
            fh.setLevel(trade_file_level)
            fh.setFormatter(_JsonFormatter())
            self._log.addHandler(fh)

    # ── divider ──────────────────────────────────────────────────

    def _sep(self, char: str = "─", width: int = 64) -> None:
        self._log.info(f"{DIM}{char * width}{RST}")

    # ── Market lifecycle ─────────────────────────────────────────

    def market_active(self, slug: str, time_remaining: float, prewarmed: bool) -> None:
        warm = f"{GRN}pre-warmed ✓{RST}" if prewarmed else f"{DIM}cold start{RST}"
        self._sep()
        self._log.info(
            f"  {GRN}{BOLD}▶ ROUND ACTIVE{RST}   "
            f"{_slug_tag(slug)}   "
            f"{BOLD}{_ms(time_remaining)} remaining{RST}   {warm}",
            extra={"event":"market_active","slug":slug,
                   "time_remaining":time_remaining,"prewarmed":prewarmed},
        )

    def market_near_expiry(self, slug: str, seconds_left: float) -> None:
        self._log.warning(
            f"  {AMB}⏱ NEAR EXPIRY{RST}    "
            f"{_slug_tag(slug)}   "
            f"{AMB}{BOLD}{seconds_left:.0f}s left — no new entries{RST}",
            extra={"event":"near_expiry","slug":slug,"seconds_left":seconds_left},
        )

    def market_resolved(
        self, slug: str, winning_side: str,
        up_price: float, down_price: float,
    ) -> None:
        icon = f"{GRN}⬆{RST}" if winning_side.upper() == "UP" else f"{MGT}⬇{RST}"
        self._log.info(
            f"  {icon} RESOLVED        "
            f"{_slug_tag(slug)}   "
            f"winner {_dir_tag(winning_side)}   "
            f"UP {GLD}{up_price:.2f}{RST}  /  DOWN {GLD}{down_price:.2f}{RST}",
            extra={"event":"market_resolved","slug":slug,"winner":winning_side,
                   "up_price":up_price,"down_price":down_price},
        )

    def market_rollover(self, from_slug: str, to_slug: str) -> None:
        self._log.info(
            f"  {SKY}↻ ROLLOVER{RST}       "
            f"{DIM}{from_slug[-14:]}{RST}  →  {_slug_tag(to_slug)}",
            extra={"event":"rollover","from_slug":from_slug,"to_slug":to_slug},
        )

    def prewarm_start(self, slug: str) -> None:
        self._log.info(
            f"  {SKY}⟳ PRE-WARM{RST}       "
            f"{_slug_tag(slug)}   subscribing next-round orderbook",
            extra={"event":"prewarm_start","slug":slug},
        )

    def prewarm_ready(
        self, slug: str, up_depth: float, down_depth: float,
    ) -> None:
        self._log.info(
            f"  {GRN}✓ PREWARM READY{RST}  "
            f"{_slug_tag(slug)}   "
            f"UP depth {GLD}{up_depth:,.0f}{RST}   DOWN depth {GLD}{down_depth:,.0f}{RST}",
            extra={"event":"prewarm_ready","slug":slug,
                   "up_depth":up_depth,"down_depth":down_depth},
        )

    # ── Signal events ────────────────────────────────────────────

    def signal(
        self, slug: str, direction: str, strength: str,
        confidence: float, up_prob: float, down_prob: float,
        ev: float, tradeable: bool,
    ) -> None:
        status = f"{GRN}● TRADEABLE{RST}" if tradeable else f"{DIM}○ below threshold{RST}"
        log_fn = self._log.info if tradeable else self._log.debug
        log_fn(
            f"  {DIM}◈ SIGNAL{RST}         "
            f"{_slug_tag(slug)}   "
            f"{_dir_tag(direction)}  "
            f"{BOLD}{strength:<8}{RST}  "
            f"conf {BOLD}{confidence:.0%}{RST}   "
            f"P(↑) {up_prob:.3f}  P(↓) {down_prob:.3f}   "
            f"EV {GLD}{ev:+.4f}{RST}   {status}",
            extra={"event":"signal","slug":slug,"direction":direction,
                   "strength":strength,"confidence":confidence,
                   "up_prob":up_prob,"down_prob":down_prob,
                   "ev":ev,"tradeable":tradeable},
        )

    def signal_blocked(self, slug: str, reason: str, strength: str) -> None:
        self._log.info(
            f"  {DIM}⊘ BLOCKED{RST}        "
            f"{_slug_tag(slug)}   "
            f"{AMB}{reason}{RST}   "
            f"{DIM}({strength}){RST}",
            extra={"event":"signal_blocked","slug":slug,
                   "reason":reason,"strength":strength},
        )

    # ── Trade entry / exit ────────────────────────────────────────

    def entry_placed(
        self, slug: str, direction: str, price: float,
        shares: float, cost_usdc: float, fee_usdc: float,
        order_id: str, dry_run: bool = False,
    ) -> None:
        tag = f"  {AMB}[DRY-RUN]{RST}" if dry_run else ""
        self._log.info(
            f"  {GRN}{BOLD}▲ ENTRY{RST}{tag}        "
            f"{_slug_tag(slug)}   "
            f"{_dir_tag(direction)}  "
            f"@ {GLD}{BOLD}{price:.3f}{RST}   "
            f"{shares:.2f} shares   "
            f"cost {GLD}${cost_usdc:.2f}{RST}   "
            f"fee ${fee_usdc:.4f}   "
            f"#{order_id[:12]}",
            extra={"event":"entry","slug":slug,"direction":direction,
                   "price":price,"shares":shares,"cost_usdc":cost_usdc,
                   "fee_usdc":fee_usdc,"order_id":order_id,"dry_run":dry_run},
        )

    def entry_rejected(self, slug: str, direction: str, reason: str) -> None:
        self._log.warning(
            f"  {CRL}✗ REJECTED{RST}       "
            f"{_slug_tag(slug)}   "
            f"{_dir_tag(direction)}   "
            f"{AMB}{reason}{RST}",
            extra={"event":"entry_rejected","slug":slug,
                   "direction":direction,"reason":reason},
        )

    def exit_win(
        self, slug: str, direction: str,
        pnl_usdc: float, cost_usdc: float, roi_pct: float,
    ) -> None:
        self._log.info(
            f"  {GRN}{BOLD}▼ EXIT  WIN{RST}      "
            f"{_slug_tag(slug)}   "
            f"{_dir_tag(direction)}  "
            f"{_pnl_tag(pnl_usdc)}   "
            f"ROI {_roi_tag(roi_pct)}   "
            f"cost ${cost_usdc:.2f}",
            extra={"event":"exit_win","slug":slug,"direction":direction,
                   "pnl_usdc":pnl_usdc,"cost_usdc":cost_usdc,"roi_pct":roi_pct},
        )

    def exit_loss(
        self, slug: str, direction: str,
        pnl_usdc: float, cost_usdc: float, roi_pct: float,
    ) -> None:
        self._log.warning(
            f"  {CRL}{BOLD}▼ EXIT  LOSS{RST}     "
            f"{_slug_tag(slug)}   "
            f"{_dir_tag(direction)}  "
            f"{_pnl_tag(pnl_usdc)}   "
            f"ROI {_roi_tag(roi_pct)}   "
            f"cost ${cost_usdc:.2f}",
            extra={"event":"exit_loss","slug":slug,"direction":direction,
                   "pnl_usdc":pnl_usdc,"cost_usdc":cost_usdc,"roi_pct":roi_pct},
        )

    # ── Risk events ───────────────────────────────────────────────

    def circuit_open(
        self, reason: str, consecutive_losses: int, drawdown_pct: float,
    ) -> None:
        self._sep("━")
        self._log.critical(
            f"  {RED}{BOLD}⚡ CIRCUIT OPEN{RST}   "
            f"{CRL}{reason}{RST}   "
            f"losses {BOLD}{consecutive_losses}{RST}   "
            f"drawdown {BOLD}{drawdown_pct:.1f}%{RST}",
            extra={"event":"circuit_open","reason":reason,
                   "consecutive_losses":consecutive_losses,
                   "drawdown_pct":drawdown_pct},
        )
        self._sep("━")

    def circuit_half_open(self, cooldown_elapsed: float) -> None:
        self._log.warning(
            f"  {AMB}⚡ CIRCUIT HALF-OPEN{RST}   "
            f"cooldown {cooldown_elapsed:.0f}s elapsed — probe mode",
            extra={"event":"circuit_half_open","cooldown_elapsed":cooldown_elapsed},
        )

    def circuit_closed(self) -> None:
        self._log.info(
            f"  {GRN}⚡ CIRCUIT CLOSED{RST}   trading resumed",
            extra={"event":"circuit_closed"},
        )

    def kill_switch(self, reason: str) -> None:
        self._sep("█")
        self._log.critical(
            f"  {RED}{BOLD}🛑  KILL SWITCH ACTIVATED{RST}   {CRL}{reason}{RST}",
            extra={"event":"kill_switch","reason":reason},
        )
        self._sep("█")

    # ── System events ─────────────────────────────────────────────

    def ws_reconnect(self, stream: str, attempt: int, delay_s: float) -> None:
        self._log.warning(
            f"  {AMB}↺ WS RECONNECT{RST}   "
            f"{DIM}{stream}{RST}   "
            f"attempt {attempt}   "
            f"back-off {delay_s:.1f}s",
            extra={"event":"ws_reconnect","stream":stream,
                   "attempt":attempt,"delay_s":delay_s},
        )

    def ws_connected(self, stream: str, latency_ms: float) -> None:
        self._log.info(
            f"  {GRN}✓ WS CONNECTED{RST}   "
            f"{DIM}{stream}{RST}   "
            f"latency {GLD}{latency_ms:.0f}ms{RST}",
            extra={"event":"ws_connected","stream":stream,"latency_ms":latency_ms},
        )

    def book_synced(self, slug: str, token: str, bids: int, asks: int) -> None:
        side = "UP  " if "up" in token.lower() else "DOWN"
        self._log.debug(
            f"  {DIM}≡ BOOK SYNC{RST}      "
            f"{_slug_tag(slug)}   "
            f"{side}   "
            f"bids {bids:3d}  asks {asks:3d}",
            extra={"event":"book_synced","slug":slug,"token":token,
                   "bids":bids,"asks":asks},
        )

    # ── Session summary ───────────────────────────────────────────

    def session_stats(
        self, trades: int, wins: int, losses: int,
        pnl_usdc: float, win_rate: float, uptime_s: float,
        open_positions: int = 0,
    ) -> None:
        closed = wins + losses
        self._sep("═")
        self._log.info(
            f"  {BOLD}📊 SESSION SUMMARY{RST}",
            extra={"event":"session_stats_header"},
        )
        self._log.info(
            f"     Entries  {BOLD}{trades:4d}{RST}   "
            f"Open {AMB}{open_positions:3d}{RST}   "
            f"Closed {BOLD}{closed:3d}{RST}   "
            f"Wins {GRN}{wins:3d}{RST}   "
            f"Losses {CRL}{losses:3d}{RST}   "
            f"Win rate {BOLD}{win_rate:.0f}%{RST}",
            extra={"event":"session_stats","trades":trades,"wins":wins,
                   "losses":losses,"open_positions":open_positions,
                   "closed":closed,"win_rate":win_rate},
        )
        self._log.info(
            f"     PnL      {_pnl_tag(pnl_usdc)}   "
            f"Uptime {BOLD}{_hms(uptime_s)}{RST}",
            extra={"event":"session_pnl","pnl_usdc":pnl_usdc,"uptime_s":uptime_s},
        )
        self._sep("═")


# ─────────────────────────────────────────────────────────────────
#  Metrics collector
# ─────────────────────────────────────────────────────────────────

class MetricsCollector:
    def __init__(self) -> None:
        self._counters: Dict[str, int]   = defaultdict(int)
        self._gauges:   Dict[str, float] = defaultdict(float)
        self._start     = time.time()

    def inc(self, key: str, n: int = 1) -> None:
        self._counters[key] += n

    def gauge(self, key: str, val: float) -> None:
        self._gauges[key] = val

    def uptime(self) -> float:
        return time.time() - self._start

    def to_prometheus_text(self) -> str:
        lines = [
            "# HELP bot_uptime_seconds Uptime",
            "# TYPE bot_uptime_seconds gauge",
            f"bot_uptime_seconds {self.uptime():.1f}", "",
        ]
        for k, v in sorted(self._gauges.items()):
            n = k.replace("-","_")
            lines += [f"# TYPE bot_{n} gauge", f"bot_{n} {v}", ""]
        for k, v in sorted(self._counters.items()):
            n = k.replace("-","_")
            lines += [f"# TYPE bot_{n}_total counter", f"bot_{n}_total {v}", ""]
        return "\n".join(lines)


async def start_metrics_server(
    collector: MetricsCollector, port: int = 9090,
) -> None:
    from aiohttp import web

    async def metrics(r: web.Request) -> web.Response:
        return web.Response(text=collector.to_prometheus_text(),
                            content_type="text/plain")

    async def health(r: web.Request) -> web.Response:
        return web.Response(text='{"status":"ok"}',
                            content_type="application/json")

    app = web.Application()
    app.router.add_get("/metrics", metrics)
    app.router.add_get("/health",  health)
    runner = web.AppRunner(app)
    try:
        await runner.setup()
        await web.TCPSite(runner, "0.0.0.0", port).start()
    except OSError as exc:
        await runner.cleanup()
        logging.getLogger(__name__).warning(
            "Metrics server disabled: port %d is unavailable (%s)",
            port, exc
        )
        return
    logging.getLogger(__name__).info(
        f"  {GRN}✓ METRICS SERVER{RST}  listening on :{port}"
    )
