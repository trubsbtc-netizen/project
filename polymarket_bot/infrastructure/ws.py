from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
import urllib.parse
from typing import Any, Dict, Optional

import websockets

from polymarket_bot.infrastructure.config import InfrastructureConfig
from polymarket_bot.infrastructure.network import DoHResolver
from polymarket_bot.infrastructure.types import ComponentHealth, ServiceState

logger = logging.getLogger(__name__)


class ManagedWebSocketClient:
    def __init__(self, name: str, url: str, config: InfrastructureConfig, stale_after_s: float):
        self.name = name
        self.url = url
        self.config = config
        self.stale_after_s = stale_after_s
        self.state = ServiceState.STOPPED
        self.connected = False
        self.last_rx = 0.0
        self.last_tx = 0.0
        self.reconnects = 0
        self.last_error = ""
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._ws: Any = None
        self._send_lock = asyncio.Lock()
        self._doh_resolver = DoHResolver()

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._running = True
        self.state = ServiceState.STARTING
        self._task = asyncio.create_task(self._run(), name=f"{self.name}-ws")

    async def stop(self) -> None:
        self._running = False
        self.state = ServiceState.STOPPING
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        self.connected = False
        self.state = ServiceState.STOPPED

    def health(self) -> ComponentHealth:
        state = ServiceState.DEGRADED if self.connected and self.is_stale else self.state
        return ComponentHealth(
            name=self.name,
            state=state,
            connected=self.connected,
            last_rx=self.last_rx,
            last_tx=self.last_tx,
            reconnects=self.reconnects,
            stale_after_s=self.stale_after_s,
            last_error=self.last_error,
        )

    @property
    def is_stale(self) -> bool:
        if self.last_rx <= 0:
            return True
        return (time.time() - self.last_rx) > self.stale_after_s

    async def _run(self) -> None:
        delay = max(0.1, self.config.ws_reconnect_initial_s)
        while self._running:
            try:
                connect_kwargs: Dict[str, Any] = {}
                parsed_url = urllib.parse.urlparse(self.url)
                if parsed_url.hostname:
                    try:
                        resolver = self._doh_resolver
                        if resolver is None:
                            resolver = self._doh_resolver = DoHResolver()
                        resolved = await resolver.resolve(parsed_url.hostname)
                        if resolved:
                            connect_kwargs["host"] = resolved[0]["host"]
                    except Exception:
                        pass
                if parsed_url.scheme == "wss":
                    connect_kwargs["ssl"] = ssl.create_default_context()
                async with websockets.connect(
                    self.url,
                    ping_interval=None,
                    ping_timeout=self.config.ws_ping_timeout_s,
                    close_timeout=5,
                    max_size=2**22,
                    **connect_kwargs,
                ) as ws:
                    self._ws = ws
                    self.connected = True
                    self.state = ServiceState.CONNECTED
                    self.last_error = ""
                    delay = max(0.1, self.config.ws_reconnect_initial_s)
                    await self._on_connect()
                    self._heartbeat_task = asyncio.create_task(self._heartbeat(), name=f"{self.name}-heartbeat")
                    async for raw in ws:
                        if not self._running:
                            break
                        self.last_rx = time.time()
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8", errors="replace")
                        await self._on_raw_message(raw)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.last_error = repr(exc)
                if self._running:
                    logger.debug("%s websocket disconnected: %s", self.name, exc)
            finally:
                self.connected = False
                self._ws = None
                if self._heartbeat_task:
                    self._heartbeat_task.cancel()
                    try:
                        await self._heartbeat_task
                    except asyncio.CancelledError:
                        pass
                    self._heartbeat_task = None
                if self._running:
                    self.reconnects += 1
                    self.state = ServiceState.DEGRADED
                    await asyncio.sleep(delay)
                    delay = min(self.config.ws_reconnect_max_s, delay * 1.5)
        self.connected = False
        self.state = ServiceState.STOPPED

    async def _heartbeat(self) -> None:
        interval = max(1.0, self.config.ws_ping_interval_s)
        while self._running and self.connected:
            try:
                await asyncio.sleep(interval)
                ws = self._ws
                if ws is not None and hasattr(ws, "ping"):
                    await ws.ping()
                    self.last_tx = time.time()
                else:
                    await self.send_text("PING")
            except asyncio.CancelledError:
                break
            except Exception:
                break

    async def send_json(self, payload: Dict[str, Any]) -> None:
        await self.send_text(json.dumps(payload, separators=(",", ":")))

    async def send_text(self, payload: str) -> None:
        ws = self._ws
        if ws is None:
            return
        async with self._send_lock:
            await ws.send(payload)
            self.last_tx = time.time()

    async def _on_connect(self) -> None:
        return None

    async def _on_raw_message(self, raw: str) -> None:
        return None
