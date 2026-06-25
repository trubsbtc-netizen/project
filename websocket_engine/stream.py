"""
WebSocket stream manager for Polymarket market and user channels.

Key production requirements addressed:
1. Jittered exponential reconnect with bounded retries
2. Silent WebSocket death detection (pong watchdog)
3. Async message queue with backpressure protection
4. State synchronization after reconnect (re-subscribe + fetch snapshot)
5. Zero blocking operations — fully async
6. Connection health metrics
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp

from core.constants import (
    WS_DEAD_TIMEOUT_S,
    WS_MAX_MESSAGE_SIZE,
    WS_PING_INTERVAL_S,
    WS_PONG_TIMEOUT_S,
    WS_QUEUE_MAXSIZE,
    WS_RECONNECT_BASE_S,
    WS_RECONNECT_FACTOR,
    WS_RECONNECT_JITTER,
    WS_RECONNECT_MAX_S,
)

logger = logging.getLogger(__name__)


class StreamType(Enum):
    MARKET = "market"
    USER   = "user"


@dataclass
class ConnectionStats:
    connects:           int = 0
    disconnects:        int = 0
    messages_received:  int = 0
    messages_dropped:   int = 0   # Due to full queue
    ping_sent:          int = 0
    pong_received:      int = 0
    total_reconnect_s:  float = 0.0
    last_message_time:  float = 0.0
    latency_ms:         float = 0.0   # Last ping-pong round-trip


class WebSocketStream:
    """
    Manages a single WebSocket connection with:
    - Automatic reconnect with jittered exponential backoff
    - Ping/pong watchdog to detect silent connection death
    - Async message queue with configurable backpressure
    - Subscription state tracking for re-subscribe on reconnect
    - Connection health monitoring
    """

    def __init__(
        self,
        name:        str,
        url:         str,
        stream_type: StreamType,
        auth_config: Optional[Dict] = None,
        on_message:  Optional[Callable[[Dict[str, Any]], None]] = None,
        on_reconnect: Optional[Callable[[], None]] = None,
    ) -> None:
        self.name        = name
        self.url         = url
        self.stream_type = stream_type
        self._auth_config = auth_config
        self._on_message_cb  = on_message
        self._on_reconnect_cb = on_reconnect

        # State
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = asyncio.Event()
        self._shutdown  = asyncio.Event()
        self._subscribed_assets: Set[str]    = set()
        self._subscribed_markets: Set[str]   = set()

        # Message queue with backpressure
        self._queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(
            maxsize=WS_QUEUE_MAXSIZE
        )

        # Metrics
        self.stats = ConnectionStats()

        # Watchdog state
        self._last_pong_time:    float = 0.0
        self._last_message_time: float = 0.0
        self._ping_send_time:    float = 0.0

        # Reconnect state
        self._reconnect_count:   int   = 0
        self._reconnect_delay:   float = WS_RECONNECT_BASE_S

        # Background tasks
        self._tasks: List[asyncio.Task] = []
        self._auth_failed = False
        self._initial_subscription_sent = False

    async def start(self, session: aiohttp.ClientSession) -> None:
        """Start the WebSocket stream manager."""
        self._session = session
        self._tasks = [
            asyncio.create_task(self._connection_loop(), name=f"{self.name}-conn"),
            asyncio.create_task(self._watchdog_loop(),   name=f"{self.name}-watchdog"),
            asyncio.create_task(self._dispatch_loop(),   name=f"{self.name}-dispatch"),
        ]
        logger.info("WebSocket stream started: %s -> %s", self.name, self.url)

    async def stop(self) -> None:
        """Gracefully stop the stream."""
        self._shutdown.set()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self._ws and not self._ws.closed:
            await self._ws.close()
        logger.info("WebSocket stream stopped: %s", self.name)

    async def subscribe_assets(self, asset_ids: List[str]) -> None:
        """Subscribe to market data for a list of asset IDs."""
        new_asset_ids = [aid for aid in asset_ids if aid not in self._subscribed_assets]
        self._subscribed_assets.update(asset_ids)
        if self._connected.is_set() and self._ws and not self._ws.closed:
            ids_to_send = new_asset_ids if self._initial_subscription_sent else list(self._subscribed_assets)
            if ids_to_send:
                await self._send_market_subscription(ids_to_send)

    async def subscribe_user_markets(self, condition_ids: List[str]) -> None:
        """Subscribe to user channel for given market condition IDs."""
        new_condition_ids = [
            cid for cid in condition_ids if cid not in self._subscribed_markets
        ]
        self._subscribed_markets.update(condition_ids)
        if self._connected.is_set() and self._ws and not self._ws.closed:
            ids_to_send = (
                new_condition_ids
                if self._initial_subscription_sent
                else list(self._subscribed_markets)
            )
            if ids_to_send:
                await self._send_user_subscription(ids_to_send)

    async def unsubscribe_assets(self, asset_ids: List[str]) -> None:
        """Remove assets from subscription set."""
        for aid in asset_ids:
            self._subscribed_assets.discard(aid)
        # Note: Polymarket WS doesn't support explicit unsubscribe on market channel
        # We just stop processing messages for that asset_id

    @property
    def is_connected(self) -> bool:
        return self._connected.is_set()

    @property
    def queue_size(self) -> int:
        return self._queue.qsize()

    async def wait_connected(self, timeout: float = 30.0) -> bool:
        """Block until connected or timeout."""
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    # ─────────────────────────── Connection Loop ───────────────────────────

    async def _connection_loop(self) -> None:
        """Main connection loop with reconnect logic."""
        while not self._shutdown.is_set():
            try:
                await self._connect_and_run()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                if self.stream_type == StreamType.USER and self._auth_failed:
                    logger.warning("User websocket disabled after auth failure")
                    break
                logger.error("WebSocket connection error on %s: %s", self.name, exc)

            if self._shutdown.is_set():
                break

            self._connected.clear()
            self.stats.disconnects += 1

            # Jittered exponential backoff
            delay = self._reconnect_delay * (
                1 + random.uniform(-WS_RECONNECT_JITTER, WS_RECONNECT_JITTER)
            )
            delay = min(delay, WS_RECONNECT_MAX_S)
            log_fn = logger.info if self._reconnect_count == 0 else logger.debug
            log_fn("Reconnecting %s in %.2fs (attempt #%d)",
                   self.name, delay, self._reconnect_count + 1)
            self.stats.total_reconnect_s += delay

            await asyncio.sleep(delay)
            self._reconnect_delay = min(
                self._reconnect_delay * WS_RECONNECT_FACTOR,
                WS_RECONNECT_MAX_S
            )
            self._reconnect_count += 1

    async def _connect_and_run(self) -> None:
        """Establish connection, subscribe, and read messages."""
        if not self._session:
            raise RuntimeError("Session not set")

        ws_kwargs = {
            "max_msg_size": WS_MAX_MESSAGE_SIZE,
            "heartbeat":    WS_PING_INTERVAL_S,   # aiohttp built-in keepalive
            "compress":     0,                    # Disable compression for latency
        }

        logger.info("Connecting to %s", self.url)
        t0 = time.monotonic()

        async with self._session.ws_connect(self.url, **ws_kwargs) as ws:
            self._ws = ws
            self._connected.set()
            self._initial_subscription_sent = False
            self._last_message_time = time.monotonic()
            self._last_pong_time    = time.monotonic()
            self.stats.connects += 1
            self._reconnect_delay = WS_RECONNECT_BASE_S  # Reset backoff on success

            connect_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "Connected to %s in %.1fms", self.name, connect_ms
            )

            # Send subscriptions immediately after connecting
            await self._resubscribe_all()

            # Notify reconnect callback (triggers orderbook re-sync)
            if self._on_reconnect_cb:
                try:
                    self._on_reconnect_cb()
                except Exception as exc:
                    logger.error("Reconnect callback error: %s", exc)

            # Read loop
            async for msg in ws:
                if self._shutdown.is_set():
                    break

                self._last_message_time = time.monotonic()
                self.stats.last_message_time = self._last_message_time

                if msg.type == aiohttp.WSMsgType.TEXT:
                    await self._handle_text(msg.data)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    await self._handle_text(msg.data.decode("utf-8"))
                elif msg.type == aiohttp.WSMsgType.PING:
                    # Respond to server pings
                    await ws.pong()
                elif msg.type == aiohttp.WSMsgType.PONG:
                    self._last_pong_time = time.monotonic()
                    self.stats.pong_received += 1
                    if self._ping_send_time > 0:
                        self.stats.latency_ms = (
                            (time.monotonic() - self._ping_send_time) * 1000
                        )
                elif msg.type in (
                    aiohttp.WSMsgType.CLOSE,
                    aiohttp.WSMsgType.CLOSING,
                    aiohttp.WSMsgType.CLOSED,
                    aiohttp.WSMsgType.ERROR,
                ):
                    logger.warning(
                        "WebSocket %s closed: type=%s data=%s",
                        self.name, msg.type, msg.data
                    )
                    break

        self._connected.clear()

    # ─────────────────────────── Message Handling ───────────────────────────

    async def _handle_text(self, text: str) -> None:
        """Parse and enqueue a WebSocket text message."""
        try:
            # Polymarket sometimes sends arrays of messages
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    await self._enqueue(item)
            else:
                await self._enqueue(data)
        except json.JSONDecodeError as exc:
            raw = text.strip()
            raw_upper = raw.upper()
            if raw_upper == "PONG":
                self._last_pong_time = time.monotonic()
                self.stats.pong_received += 1
                if self._ping_send_time > 0:
                    self.stats.latency_ms = (
                        (time.monotonic() - self._ping_send_time) * 1000
                    )
                return
            if raw_upper == "PING":
                if self._ws and not self._ws.closed:
                    await self._ws.send_str("PONG")
                return
            if self.stream_type == StreamType.USER and raw_upper == "INVALID OPERATION":
                logger.warning("User websocket subscription rejected: %s", raw)
                await self._force_reconnect()
                return
            if self.stream_type == StreamType.USER and raw_upper in {
                "UNAUTHORIZED",
                "AUTHENTICATION FAILED",
            }:
                self._auth_failed = True
                logger.warning("User websocket auth rejected: %s", raw)
                await self._force_reconnect()
                return
            logger.debug("Non-JSON websocket message on %s: %s", self.name, raw[:200])
        except Exception as exc:
            logger.error("Message handling error on %s: %s", self.name, exc)

    async def _enqueue(self, msg: Dict[str, Any]) -> None:
        """
        Enqueue message with backpressure protection.
        If queue is full, drop the OLDEST message (not the newest).
        Dropping old messages is preferred because fresh data is more valuable.
        """
        self.stats.messages_received += 1
        try:
            self._queue.put_nowait(msg)
        except asyncio.QueueFull:
            # Drop oldest message to make room
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            self._queue.put_nowait(msg)
            self.stats.messages_dropped += 1
            logger.warning(
                "Queue full on %s — dropped oldest message (size=%d)",
                self.name, WS_QUEUE_MAXSIZE
            )

    async def _dispatch_loop(self) -> None:
        """Continuously dispatch queued messages to callback."""
        while not self._shutdown.is_set():
            try:
                msg = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                if self._on_message_cb:
                    try:
                        self._on_message_cb(msg)
                    except Exception as exc:
                        logger.error("Message callback error on %s: %s", self.name, exc)
                self._queue.task_done()
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

    # ─────────────────────────── Watchdog ───────────────────────────

    async def _watchdog_loop(self) -> None:
        """
        Detect silent WebSocket death.
        Sends periodic pings and forces reconnect if:
        1. No pong received within WS_PONG_TIMEOUT_S after a ping
        2. No message received within WS_DEAD_TIMEOUT_S

        This protects against:
        - Silent TCP disconnections (carrier lost but no FIN/RST)
        - Load balancer idle timeouts
        - NAT table expiry
        """
        await asyncio.sleep(5.0)   # Initial grace period

        while not self._shutdown.is_set():
            try:
                await asyncio.sleep(WS_PING_INTERVAL_S)

                now = time.monotonic()

                if (
                    self._connected.is_set()
                    and self._ping_send_time > 0
                    and self._last_pong_time < self._ping_send_time
                    and (now - self._ping_send_time) > WS_PONG_TIMEOUT_S
                ):
                    logger.warning(
                        "WebSocket %s missed heartbeat pong in %.1fs, forcing reconnect",
                        self.name, now - self._ping_send_time
                    )
                    await self._force_reconnect()
                    continue

                # Check for dead connection (no messages)
                if (
                    self._connected.is_set()
                    and self._last_message_time > 0
                    and (now - self._last_message_time) > WS_DEAD_TIMEOUT_S
                ):
                    logger.warning(
                        "WebSocket %s appears dead — no message in %.1fs, forcing reconnect",
                        self.name, now - self._last_message_time
                    )
                    await self._force_reconnect()
                    continue

                # Send ping if connected
                if self._connected.is_set() and self._ws and not self._ws.closed:
                    try:
                        self._ping_send_time = time.monotonic()
                        await self._ws.send_str("PING")
                        self.stats.ping_sent += 1
                    except Exception as exc:
                        logger.warning("Ping failed on %s: %s", self.name, exc)
                        await self._force_reconnect()

            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Watchdog error on %s: %s", self.name, exc)

    async def _force_reconnect(self) -> None:
        """Force close current connection to trigger reconnect."""
        self._connected.clear()
        if self._ws and not self._ws.closed:
            try:
                await self._ws.close()
            except Exception as exc:
                logger.debug("WebSocket close error during force reconnect on %s: %s", self.name, exc)

    # ─────────────────────────── Subscriptions ───────────────────────────

    async def _resubscribe_all(self) -> None:
        """Re-send all subscriptions after reconnect."""
        if self.stream_type == StreamType.MARKET:
            if self._subscribed_assets:
                await self._send_market_subscription(list(self._subscribed_assets))
        elif self.stream_type == StreamType.USER:
            if self._auth_config and self._subscribed_markets:
                await self._send_user_subscription(list(self._subscribed_markets))

    async def _send_market_subscription(self, asset_ids: List[str]) -> None:
        """Send market channel subscription message."""
        if not self._ws or self._ws.closed:
            return
        if self._initial_subscription_sent:
            msg = {
                "assets_ids": asset_ids,
                "operation": "subscribe",
                "custom_feature_enabled": True,
            }
        else:
            msg = {
                "assets_ids": asset_ids,
                "type": "market",
                "custom_feature_enabled": True,
            }
        await self._send_json(msg)
        self._initial_subscription_sent = True
        logger.info(
            "Subscribed to %d assets on market channel", len(asset_ids)
        )

    async def _send_user_subscription(self, condition_ids: List[str]) -> None:
        """Send authenticated user channel subscription."""
        if not self._ws or self._ws.closed or not self._auth_config:
            return
        if not self._initial_subscription_sent:
            msg = {
                "auth": {
                    "apiKey":     self._auth_config["api_key"],
                    "secret":     self._auth_config["api_secret"],
                    "passphrase": self._auth_config["api_passphrase"],
                },
                "markets": condition_ids,
                "type": "user",
            }
            await self._send_json(msg)
            self._initial_subscription_sent = True
            logger.info("Subscribed to user channel for %d markets", len(condition_ids))
            return

        if not condition_ids:
            return
        msg = {
            "markets": condition_ids,
            "operation": "subscribe",
        }
        await self._send_json(msg)
        logger.info("Subscribed to user channel for %d markets", len(condition_ids))

    async def _send_json(self, data: Dict[str, Any]) -> None:
        """Send JSON message on WebSocket."""
        if not self._ws or self._ws.closed:
            return
        try:
            await self._ws.send_str(json.dumps(data))
        except Exception as exc:
            logger.error("Send error on %s: %s", self.name, exc)
