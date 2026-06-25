"""
DNS over HTTPS (DoH) resolver with intelligent failover, caching, and prefetch.

Standard system DNS is unreliable for production trading systems because:
1. It may be intercepted or rate-limited
2. No encryption of DNS queries
3. No TTL control
4. Single point of failure

This implementation uses multiple DoH providers with automatic failover,
local caching, and pre-resolution of known trading infrastructure hostnames.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import aiohttp

from core.constants import (
    DOH_PROVIDERS,
    DOH_CACHE_TTL_S,
    DOH_TIMEOUT_S,
    DNS_PREFETCH_HOSTS,
)

logger = logging.getLogger(__name__)

_DOH_PROVIDERS = [
    (f"provider_{i}", url) for i, url in enumerate(DOH_PROVIDERS)
]

_PREFETCH_HOSTS = DNS_PREFETCH_HOSTS

_DNS_CACHE_TTL_S  = DOH_CACHE_TTL_S
_DOH_TIMEOUT_S    = DOH_TIMEOUT_S
_RECORD_TYPE_A    = 1
_RECORD_TYPE_AAAA = 28


class DnsCache:
    """Thread-safe DNS cache with TTL."""

    def __init__(self) -> None:
        self._cache: Dict[str, Tuple[List[str], float]] = {}
        self._lock = asyncio.Lock()

    async def get(self, host: str) -> Optional[List[str]]:
        async with self._lock:
            entry = self._cache.get(host)
            if entry is None:
                return None
            ips, expires_at = entry
            if time.monotonic() > expires_at:
                del self._cache[host]
                return None
            return ips

    async def set(self, host: str, ips: List[str], ttl: int) -> None:
        async with self._lock:
            effective_ttl = min(ttl, _DNS_CACHE_TTL_S)
            expires_at = time.monotonic() + effective_ttl
            self._cache[host] = (ips, expires_at)

    async def invalidate(self, host: str) -> None:
        async with self._lock:
            self._cache.pop(host, None)

    async def size(self) -> int:
        async with self._lock:
            return len(self._cache)


class DoHResolver:
    """
    DNS over HTTPS resolver with:
    - Multiple provider failover
    - Local TTL cache
    - Startup pre-resolution
    - Automatic fallback to system resolver on complete failure
    """

    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        providers: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        self._session = session
        self._providers = providers or _DOH_PROVIDERS
        self._cache = DnsCache()
        self._provider_stats: Dict[str, Dict] = {
            name: {"success": 0, "failure": 0, "latency_ms": 0.0}
            for name, _ in self._providers
        }
        self._owns_session = session is None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "DoHResolver":
        if self._owns_session:
            from shared.http import create_tcp_connector
            connector = create_tcp_connector(
                ssl=True,
                limit=10,
                ttl_dns_cache=0,
                use_dns_cache=False,
            )
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=_DOH_TIMEOUT_S),
                headers={"Accept": "application/dns-json"},
            )
        await self._prefetch()
        return self

    async def __aexit__(self, *_) -> None:
        if self._owns_session and self._session:
            await self._session.close()

    async def resolve(self, host: str) -> List[str]:
        """
        Resolve hostname to IP addresses.
        Returns cached result if available and fresh.
        Falls back to system resolver if all DoH providers fail.
        """
        # Try cache first
        cached = await self._cache.get(host)
        if cached:
            return cached

        # Try each DoH provider in order
        ips, ttl = await self._resolve_with_failover(host)
        if ips:
            await self._cache.set(host, ips, ttl)
            logger.debug("DoH resolved %s -> %s (TTL=%ds)", host, ips, ttl)
            return ips

        # All DoH providers failed — fallback to system resolver
        logger.warning(
            "All DoH providers failed for %s, falling back to system resolver", host
        )
        try:
            loop = asyncio.get_event_loop()
            infos = await loop.getaddrinfo(
                host, None, family=socket.AF_INET, type=socket.SOCK_STREAM
            )
            ips = list({info[4][0] for info in infos})
            if ips:
                await self._cache.set(host, ips, 60)
                return ips
        except Exception as exc:
            logger.error("System resolver also failed for %s: %s", host, exc)

        raise ResolutionError(f"Cannot resolve {host}: all resolvers failed")

    async def _resolve_with_failover(
        self, host: str
    ) -> Tuple[List[str], int]:
        """Try each provider until one succeeds. Returns (ips, ttl)."""
        # Sort providers by success rate (best first)
        sorted_providers = sorted(
            self._providers,
            key=lambda p: self._provider_stats[p[0]]["failure"]
            / max(1, self._provider_stats[p[0]]["success"] + self._provider_stats[p[0]]["failure"]),
        )

        for name, url in sorted_providers:
            try:
                t0 = time.monotonic()
                ips, ttl = await self._query_provider(url, host)
                latency_ms = (time.monotonic() - t0) * 1000
                self._provider_stats[name]["success"] += 1
                self._provider_stats[name]["latency_ms"] = (
                    0.8 * self._provider_stats[name]["latency_ms"] + 0.2 * latency_ms
                )
                if ips:
                    return ips, ttl
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._provider_stats[name]["failure"] += 1
                logger.debug("DoH provider %s failed for %s: %s", name, host, exc)

        return [], 60

    async def _query_provider(
        self, base_url: str, host: str
    ) -> Tuple[List[str], int]:
        """
        Query a single DoH provider using application/dns-json format.
        Both Cloudflare and Google support this format.
        """
        if not self._session:
            raise RuntimeError("Session not initialized")

        url = f"{base_url}?name={quote(host)}&type=A"

        async with self._session.get(
            url,
            headers={"Accept": "application/dns-json"},
            timeout=aiohttp.ClientTimeout(total=_DOH_TIMEOUT_S),
            ssl=True,
        ) as resp:
            if resp.status != 200:
                raise ResolutionError(f"HTTP {resp.status} from DoH provider")
            data = await resp.json(content_type=None)

        return self._parse_dns_json(data)

    def _parse_dns_json(self, data: dict) -> Tuple[List[str], int]:
        """
        Parse application/dns-json response.
        Returns (list_of_ips, minimum_ttl).
        """
        status = data.get("Status", -1)
        if status != 0:  # 0 = NOERROR
            return [], 60

        answers = data.get("Answer", [])
        ips = []
        min_ttl = _DNS_CACHE_TTL_S

        for answer in answers:
            if answer.get("type") == _RECORD_TYPE_A:
                ip = answer.get("data", "").strip()
                if ip and _is_valid_ipv4(ip):
                    ips.append(ip)
                    ttl = int(answer.get("TTL", 60))
                    min_ttl = min(min_ttl, ttl)

        return ips, min_ttl

    async def _prefetch(self) -> None:
        """Pre-resolve known hostnames in parallel at startup."""
        tasks = [self._prefetch_host(h) for h in _PREFETCH_HOSTS]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for host, result in zip(_PREFETCH_HOSTS, results):
            if isinstance(result, Exception):
                logger.warning("Prefetch failed for %s: %s", host, result)
            else:
                logger.info("Prefetched DNS: %s -> %s", host, result)

    async def _prefetch_host(self, host: str) -> List[str]:
        return await self.resolve(host)

    def provider_health(self) -> Dict[str, Dict]:
        """Return current health stats for all providers."""
        return dict(self._provider_stats)


class ResolutionError(Exception):
    """Raised when all DNS resolution methods fail."""
    pass


def _is_valid_ipv4(ip: str) -> bool:
    """Validate IPv4 address format."""
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        return all(0 <= int(p) <= 255 for p in parts)
    except ValueError:
        return False
