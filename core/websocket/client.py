from __future__ import annotations

import asyncio
import inspect
import ipaddress
import logging
import random
import urllib.parse
import urllib.request
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from core.runtime.clock import mono_ns
from core.runtime.health import FeedHeartbeat
from core.transport.json import dumps, loads


logger = logging.getLogger(__name__)

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None


MessageHandler = Callable[[Any, int], Awaitable[None]]
SubscriptionFactory = Callable[[], Iterable[dict[str, Any] | str]]


@dataclass(slots=True, frozen=True)
class WebSocketSpec:
    name: str
    url: str
    heartbeat_s: float
    connect_timeout_s: float
    max_backoff_s: float
    ping_payload: str | bytes | None = None
    parse_json: bool = True
    close_timeout_s: float = 2.0
    doh_resolve: bool = False
    doh_resolver_url: str = "https://cloudflare-dns.com/dns-query"
    doh_timeout_s: float = 3.0


class ResilientWebSocketClient:
    def __init__(
        self,
        spec: WebSocketSpec,
        heartbeat: FeedHeartbeat,
        subscription_factory: SubscriptionFactory,
        handler: MessageHandler,
        control_queue: asyncio.Queue[dict[str, Any] | str] | None = None,
    ) -> None:
        self._spec = spec
        self._heartbeat = heartbeat
        self._subscription_factory = subscription_factory
        self._handler = handler
        self._control_queue = control_queue
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        if websockets is None:
            raise RuntimeError("websockets package is required")
        attempt = 0
        while not self._stop.is_set():
            try:
                await self._connect_once()
                attempt = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._heartbeat.mark_error()
                self._heartbeat.mark_reconnect()
                attempt += 1
                delay = min(self._spec.max_backoff_s, 0.25 * (2 ** min(8, attempt)))
                delay *= 0.75 + random.random() * 0.5
                exc_msg = str(exc)
                if "no close frame" in exc_msg:
                    logger.debug("%s websocket disconnected (no close frame); reconnecting in %.2fs", self._spec.name, delay)
                else:
                    logger.warning("%s websocket error: %s; reconnecting in %.2fs", self._spec.name, exc, delay)
                await asyncio.sleep(delay)

    async def _connect_once(self) -> None:
        assert websockets is not None
        connect_kwargs = await self._connect_kwargs()
        async with websockets.connect(
            self._spec.url,
            open_timeout=self._spec.connect_timeout_s,
            close_timeout=self._spec.close_timeout_s,
            ping_interval=None,
            max_queue=1024,
            **connect_kwargs,
        ) as ws:
            self._heartbeat.mark_connected()
            logger.info("%s websocket connected", self._spec.name)
            for message in self._subscription_factory():
                await ws.send(message if isinstance(message, str) else dumps(message))
            ping_task = asyncio.create_task(self._ping_loop(ws), name=f"{self._spec.name}.ping")
            control_task = None
            if self._control_queue is not None:
                control_task = asyncio.create_task(self._control_loop(ws), name=f"{self._spec.name}.control")
            try:
                async for raw in ws:
                    ts = mono_ns()
                    self._heartbeat.mark_message(ts)
                    if raw == "PONG":
                        continue
                    payload = loads(raw) if self._spec.parse_json and raw else raw
                    await self._handler(payload, ts)
                    if self._stop.is_set():
                        break
            finally:
                ping_task.cancel()
                if control_task is not None:
                    control_task.cancel()
                self._heartbeat.mark_disconnected()
                logger.info("%s websocket disconnected", self._spec.name)
                await asyncio.gather(
                    ping_task,
                    *([control_task] if control_task is not None else []),
                    return_exceptions=True,
                )

    async def _connect_kwargs(self) -> dict[str, Any]:
        if not self._spec.doh_resolve:
            return {}
        parsed = urllib.parse.urlparse(self._spec.url)
        if not parsed.hostname:
            return {}
        resolved = await asyncio.to_thread(
            _resolve_a_record_doh,
            parsed.hostname,
            self._spec.doh_resolver_url,
            self._spec.doh_timeout_s,
        )
        port = parsed.port or (443 if parsed.scheme == "wss" else 80)
        logger.info("%s resolved %s via DoH to %s", self._spec.name, parsed.hostname, resolved)
        kwargs: dict[str, Any] = {"host": resolved, "port": port}
        if _websockets_supports_proxy_kw():
            kwargs["proxy"] = None
        return kwargs

    async def _ping_loop(self, ws: Any) -> None:
        while True:
            await asyncio.sleep(self._spec.heartbeat_s)
            if self._spec.ping_payload is None:
                await ws.ping()
            else:
                await ws.send(self._spec.ping_payload)

    async def _control_loop(self, ws: Any) -> None:
        assert self._control_queue is not None
        while True:
            message = await self._control_queue.get()
            try:
                await ws.send(message if isinstance(message, str) else dumps(message))
            finally:
                self._control_queue.task_done()


def _websockets_supports_proxy_kw() -> bool:
    if websockets is None:
        return False
    try:
        return "proxy" in inspect.signature(websockets.connect).parameters
    except (TypeError, ValueError):
        return False


def _resolve_a_record_doh(hostname: str, resolver_url: str, timeout_s: float) -> str:
    query = urllib.parse.urlencode({"name": hostname, "type": "A"})
    separator = "&" if "?" in resolver_url else "?"
    url = f"{resolver_url}{separator}{query}"
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/dns-json",
            "User-Agent": "btc-poly-institutional/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = loads(resp.read())
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid DoH response for {hostname}")
    for answer in payload.get("Answer") or []:
        if not isinstance(answer, dict):
            continue
        if int(answer.get("type") or 0) != 1:
            continue
        value = str(answer.get("data") or "")
        try:
            ipaddress.ip_address(value)
        except ValueError:
            continue
        return value
    status = payload.get("Status")
    raise RuntimeError(f"DoH resolver returned no A record for {hostname} (status={status})")
