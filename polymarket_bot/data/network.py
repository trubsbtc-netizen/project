"""Network adapter: wraps BTC_POLY infrastructure into the original polymarket_bot API."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import random
import socket
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp

from polymarket_bot.infrastructure.config import InfrastructureConfig
from polymarket_bot.infrastructure.network import ConnectionPool as InfraConnectionPool
from polymarket_bot.infrastructure.network import DoHResolver as InfraDoHResolver

logger = logging.getLogger(__name__)

PROXY_URL: Optional[str] = None

_CLIENT_SESSION_SUPPORTS_PROXY = "proxy" in inspect.signature(
    aiohttp.ClientSession
).parameters


class RobustDoHResolver(aiohttp.abc.AbstractResolver):
    """Backward-compatible wrapper over BTC_POLY DoHResolver."""

    DOH_PROVIDERS: Tuple[str, ...] = (
        "https://1.1.1.1/dns-query",
        "https://1.0.0.1/dns-query",
        "https://8.8.8.8/resolve",
        "https://8.8.4.4/resolve",
        "https://dns.quad9.net:5053/dns-query",
        "https://dns.google/dns-query",
        "https://cloudflare-dns.com/dns-query",
    )

    MAX_TTL: float = 300.0
    NEGATIVE_TTL: float = 30.0
    QUERY_TIMEOUT: float = 3.0
    SYSTEM_FALLBACK_TTL: float = 60.0

    def __init__(self) -> None:
        self._inner = InfraDoHResolver()

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET) -> List[dict]:
        return await self._inner.resolve(host, port, family)

    async def close(self) -> None:
        await self._inner.close()

    def clear_cache(self) -> None:
        self._inner._cache.clear()

    def _doh_query(self, host: str, port: int) -> List[dict]:
        # Backward compat stub
        return []

    def _try_provider(self, provider: str, host: str, port: int) -> Optional[List[dict]]:
        return None

    def _extract_ttl(self, answers: List[dict], host: str, port: int, now: float) -> float:
        return self.MAX_TTL

    async def _system_resolve(self, host: str, port: int) -> List[dict]:
        loop = asyncio.get_running_loop()
        addrs = await loop.getaddrinfo(host, port, family=socket.AF_INET, type=socket.SOCK_STREAM)
        return [
            {
                "hostname": host,
                "host": addr[4][0],
                "port": port,
                "family": socket.AF_INET,
                "proto": socket.IPPROTO_TCP,
                "flags": socket.AI_NUMERICHOST,
            }
            for addr in addrs
        ]


_resolver_singleton: Optional[RobustDoHResolver] = None
_resolver_lock = asyncio.Lock()


async def get_resolver() -> RobustDoHResolver:
    global _resolver_singleton
    if _resolver_singleton is None:
        async with _resolver_lock:
            if _resolver_singleton is None:
                _resolver_singleton = RobustDoHResolver()
    return _resolver_singleton


def get_resolver_sync() -> RobustDoHResolver:
    global _resolver_singleton
    if _resolver_singleton is None:
        _resolver_singleton = RobustDoHResolver()
    return _resolver_singleton


def create_tcp_connector(
    resolver: Optional[RobustDoHResolver] = None,
    limit: int = 150,
    limit_per_host: int = 30,
    force_close: bool = True,
    keepalive_timeout: float = 30.0,
) -> aiohttp.TCPConnector:
    if resolver is None:
        resolver = get_resolver_sync()
    connector_kwargs: Dict[str, Any] = {
        "resolver": resolver,
        "family": socket.AF_INET,
        "use_dns_cache": False,
        "force_close": force_close,
        "limit": limit,
        "limit_per_host": limit_per_host,
        "ttl_dns_cache": 300,
        "enable_cleanup_closed": True,
    }
    if not force_close:
        connector_kwargs["keepalive_timeout"] = keepalive_timeout
    return aiohttp.TCPConnector(**connector_kwargs)


def create_client_session(
    *,
    connector: Optional[aiohttp.TCPConnector] = None,
    base_url: Optional[str] = None,
    timeout: Optional[aiohttp.ClientTimeout] = None,
    headers: Optional[Dict[str, str]] = None,
) -> aiohttp.ClientSession:
    if connector is None:
        connector = create_tcp_connector()
    if timeout is None:
        timeout = aiohttp.ClientTimeout(total=30.0, connect=10.0)
    merged_headers = {
        "User-Agent": "polymarket-bot/2.0",
        "Accept": "application/json",
    }
    if headers:
        merged_headers.update(headers)
    session_kwargs: Dict[str, Any] = {
        "connector": connector,
        "timeout": timeout,
        "base_url": base_url,
        "headers": merged_headers,
    }
    if PROXY_URL and _CLIENT_SESSION_SUPPORTS_PROXY:
        session_kwargs["proxy"] = PROXY_URL
    return aiohttp.ClientSession(**session_kwargs)


def create_web3_http_provider(endpoint_uri: str, *, timeout: float = 3.0) -> Any:
    from web3 import Web3 as _Web3
    request_kwargs: Dict[str, Any] = {"timeout": timeout}
    if PROXY_URL:
        request_kwargs["proxies"] = {"http": PROXY_URL, "https": PROXY_URL}
    return _Web3.HTTPProvider(endpoint_uri, request_kwargs=request_kwargs)


def jittered_delay(attempt: int, base: float = 1.0, max_delay: float = 60.0) -> float:
    exponential = min(base * (2.0 ** attempt), max_delay)
    return random.uniform(0.0, exponential)


async def ws_connect_robust(
    url: str,
    tag: str,
    handler: Callable[[Any], Any],
    *,
    subscribe_msg: Optional[Any] = None,
    heartbeat_message: Optional[Any] = None,
    heartbeat_interval: float = 15.0,
    heartbeat_timeout: float = 10.0,
    stale_timeout: float = 60.0,
    receive_timeout: float = 30.0,
    connect_timeout: float = 15.0,
    max_backoff: float = 60.0,
    base_backoff: float = 1.0,
    running_check: Optional[Callable[[], bool]] = None,
    on_connected: Optional[Callable[[], None]] = None,
    on_disconnected: Optional[Callable[[], None]] = None,
) -> None:
    """Backward-compatible ws_connect_robust that delegates to infrastructure."""
    attempt = 0
    while running_check is None or running_check():
        ws: Optional[aiohttp.ClientWebSocketResponse] = None
        session: Optional[aiohttp.ClientSession] = None
        heartbeat_task: Optional[asyncio.Task] = None
        last_data = time.monotonic()

        try:
            connector = create_tcp_connector()
            session = create_client_session(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=connect_timeout),
            )
            connect_kw: Dict[str, Any] = {
                "timeout": aiohttp.ClientTimeout(total=connect_timeout),
                "receive_timeout": receive_timeout,
                "autoping": True,
                "heartbeat": None if heartbeat_message is not None else heartbeat_interval,
            }
            if PROXY_URL and not _CLIENT_SESSION_SUPPORTS_PROXY:
                connect_kw["proxy"] = PROXY_URL

            ws = await session.ws_connect(url, **connect_kw)
            logger.info("[%s] connected (attempt %d)", tag, attempt + 1)
            last_data = time.monotonic()
            attempt = 0

            if subscribe_msg is not None and not ws.closed:
                await ws.send_json(subscribe_msg)

            if on_connected:
                on_connected()

            if heartbeat_message is not None:
                heartbeat_task = asyncio.create_task(
                    _ws_send_heartbeat_loop(ws, tag, heartbeat_interval, heartbeat_message),
                    name=f"hb-{tag}",
                )

            async for msg in ws:
                if running_check and not running_check():
                    break
                if msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                    last_data = time.monotonic()
                    text: str = msg.data if isinstance(msg.data, str) else msg.data.decode()
                    if not text:
                        continue
                    try:
                        data = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    result = handler(data)
                    if inspect.isawaitable(result):
                        await result
                elif msg.type == aiohttp.WSMsgType.PING:
                    last_data = time.monotonic()
                    try:
                        await ws.pong()
                    except Exception:
                        pass
                elif msg.type == aiohttp.WSMsgType.PONG:
                    last_data = time.monotonic()
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSING, aiohttp.WSMsgType.CLOSE):
                    logger.warning("[%s] server closed", tag)
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    err = ws.exception()
                    logger.warning("[%s] WS error: %s", tag, err)
                    break
                if time.monotonic() - last_data > stale_timeout:
                    logger.warning("[%s] stale (no data for %.0fs), reconnecting", tag, stale_timeout)
                    break

        except asyncio.CancelledError:
            raise
        except (aiohttp.ClientError, asyncio.TimeoutError, Exception) as exc:
            attempt += 1
            delay = jittered_delay(attempt, base_backoff, max_backoff)
            logger.warning("[%s] %s: %s — retry in %.1fs (attempt %d)", tag, type(exc).__name__, exc, delay, attempt)
            await asyncio.sleep(delay)
        finally:
            if on_disconnected:
                on_disconnected()
            if heartbeat_task and not heartbeat_task.done():
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except (asyncio.CancelledError, Exception):
                    pass
            if ws and not ws.closed:
                try:
                    await ws.close()
                except Exception:
                    pass
            if session and not session.closed:
                try:
                    await session.close()
                except Exception:
                    pass


async def _ws_send_heartbeat_loop(ws: aiohttp.ClientWebSocketResponse, tag: str, interval: float, message: Any) -> None:
    while not ws.closed:
        await asyncio.sleep(interval)
        if ws.closed:
            break
        try:
            if isinstance(message, (dict, list)):
                await ws.send_json(message)
            else:
                await ws.send_str(str(message))
        except Exception as exc:
            logger.debug("[%s] heartbeat error: %s", tag, exc)
            break
