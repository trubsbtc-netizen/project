from __future__ import annotations

import asyncio
import json
import logging
import socket
import ssl
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

import aiohttp
from aiohttp.abc import AbstractResolver

from core.constants import DOH_PROVIDERS
from polymarket_bot.infrastructure.config import InfrastructureConfig
from shared.http import create_tcp_connector

logger = logging.getLogger(__name__)

_DOH_QUERY_ENDPOINTS = tuple(
    f"{url}?name={{host}}&type=A" for url in DOH_PROVIDERS
)


class DoHResolver(AbstractResolver):
    ENDPOINTS = _DOH_QUERY_ENDPOINTS

    def __init__(self):
        self._cache: Dict[str, List[str]] = {}

    async def resolve(self, host: str, port: int = 0, family: int = socket.AF_INET) -> List[dict]:
        ips = self._cache.get(host)
        if not ips:
            ips = await asyncio.to_thread(self._resolve_sync, host)
            self._cache[host] = ips
        return [
            {
                "hostname": host,
                "host": ip,
                "port": port,
                "family": family,
                "proto": 0,
                "flags": socket.AI_NUMERICHOST,
            }
            for ip in ips
        ]

    async def close(self) -> None:
        self._cache.clear()

    def _resolve_sync(self, host: str) -> List[str]:
        headers = {"accept": "application/dns-json", "user-agent": "Mozilla/5.0"}
        last_error: Optional[Exception] = None
        for template in self.ENDPOINTS:
            try:
                request = urllib.request.Request(template.format(host=host), headers=headers)
                with urllib.request.urlopen(request, timeout=10) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                answers = payload.get("Answer") or []
                ips = [str(item.get("data")) for item in answers if item.get("type") == 1 and item.get("data")]
                if ips:
                    return ips
            except Exception as exc:
                last_error = exc
        raise OSError(f"DoH resolution failed for {host}: {last_error}")


class ConnectionPool:
    def __init__(self, config: InfrastructureConfig):
        self.config = config
        self._session: Optional[aiohttp.ClientSession] = None

    async def start(self) -> None:
        if self._session is not None and not self._session.closed:
            return
        timeout = aiohttp.ClientTimeout(total=self.config.http_timeout_s)
        connector = create_tcp_connector(
            resolver=DoHResolver(),
            limit=self.config.http_limit,
            limit_per_host=self.config.http_limit_per_host,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"User-Agent": "BTC_POLY_INFRA/1.0"},
        )

    async def session(self) -> aiohttp.ClientSession:
        await self.start()
        assert self._session is not None
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None
