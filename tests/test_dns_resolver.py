"""Tests for network/dns_resolver.py — DnsCache, IPv4 validation, DNS JSON parsing."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from network.dns_resolver import DnsCache, DoHResolver, ResolutionError, _is_valid_ipv4


# ──────────────────────────── _is_valid_ipv4 ────────────────────────────


class TestIsValidIpv4:
    def test_valid(self) -> None:
        assert _is_valid_ipv4("1.2.3.4")
        assert _is_valid_ipv4("0.0.0.0")
        assert _is_valid_ipv4("255.255.255.255")
        assert _is_valid_ipv4("192.168.1.1")

    def test_too_few_octets(self) -> None:
        assert not _is_valid_ipv4("1.2.3")

    def test_too_many_octets(self) -> None:
        assert not _is_valid_ipv4("1.2.3.4.5")

    def test_out_of_range(self) -> None:
        assert not _is_valid_ipv4("256.0.0.1")
        assert not _is_valid_ipv4("1.2.3.-1")

    def test_non_numeric(self) -> None:
        assert not _is_valid_ipv4("abc.def.ghi.jkl")

    def test_empty(self) -> None:
        assert not _is_valid_ipv4("")

    def test_ipv6(self) -> None:
        assert not _is_valid_ipv4("::1")


# ──────────────────────────── DnsCache ────────────────────────────


class TestDnsCache:
    @pytest.mark.asyncio
    async def test_get_set(self) -> None:
        cache = DnsCache()
        await cache.set("example.com", ["1.2.3.4"], ttl=60)
        result = await cache.get("example.com")
        assert result == ["1.2.3.4"]

    @pytest.mark.asyncio
    async def test_get_missing(self) -> None:
        cache = DnsCache()
        assert await cache.get("missing.com") is None

    @pytest.mark.asyncio
    async def test_invalidate(self) -> None:
        cache = DnsCache()
        await cache.set("test.com", ["1.1.1.1"], ttl=60)
        await cache.invalidate("test.com")
        assert await cache.get("test.com") is None

    @pytest.mark.asyncio
    async def test_invalidate_missing_no_error(self) -> None:
        cache = DnsCache()
        await cache.invalidate("nonexistent.com")

    @pytest.mark.asyncio
    async def test_size(self) -> None:
        cache = DnsCache()
        assert await cache.size() == 0
        await cache.set("a.com", ["1.1.1.1"], ttl=60)
        await cache.set("b.com", ["2.2.2.2"], ttl=60)
        assert await cache.size() == 2

    @pytest.mark.asyncio
    async def test_expired_entry(self) -> None:
        import time
        cache = DnsCache()
        # Set with TTL of 0 so it expires immediately
        await cache.set("expire.com", ["1.1.1.1"], ttl=0)
        # Monkeypatch time to be past expiry
        with patch("network.dns_resolver.time.monotonic", return_value=time.monotonic() + 400):
            result = await cache.get("expire.com")
            assert result is None

    @pytest.mark.asyncio
    async def test_ttl_capped(self) -> None:
        cache = DnsCache()
        # TTL larger than _DNS_CACHE_TTL_S should be capped
        await cache.set("ttl.com", ["1.1.1.1"], ttl=999999)
        result = await cache.get("ttl.com")
        assert result == ["1.1.1.1"]

    @pytest.mark.asyncio
    async def test_multiple_ips(self) -> None:
        cache = DnsCache()
        await cache.set("multi.com", ["1.1.1.1", "2.2.2.2", "3.3.3.3"], ttl=60)
        result = await cache.get("multi.com")
        assert result is not None
        assert len(result) == 3


# ──────────────────────────── DoHResolver._parse_dns_json ────────────────────────────


class TestParseDnsJson:
    def _resolver(self) -> DoHResolver:
        return DoHResolver(session=MagicMock(), providers=[])

    def test_noerror_with_a_record(self) -> None:
        r = self._resolver()
        data = {
            "Status": 0,
            "Answer": [
                {"type": 1, "data": "93.184.216.34", "TTL": 120},
            ],
        }
        ips, ttl = r._parse_dns_json(data)
        assert ips == ["93.184.216.34"]
        assert ttl == 120

    def test_noerror_multiple_a_records(self) -> None:
        r = self._resolver()
        data = {
            "Status": 0,
            "Answer": [
                {"type": 1, "data": "1.2.3.4", "TTL": 60},
                {"type": 1, "data": "5.6.7.8", "TTL": 120},
            ],
        }
        ips, ttl = r._parse_dns_json(data)
        assert len(ips) == 2
        assert "1.2.3.4" in ips
        assert ttl == 60

    def test_nxdomain(self) -> None:
        r = self._resolver()
        data = {"Status": 3, "Answer": []}
        ips, ttl = r._parse_dns_json(data)
        assert ips == []

    def test_cname_filtered(self) -> None:
        r = self._resolver()
        data = {
            "Status": 0,
            "Answer": [
                {"type": 5, "data": "alias.example.com", "TTL": 300},
                {"type": 1, "data": "1.2.3.4", "TTL": 60},
            ],
        }
        ips, ttl = r._parse_dns_json(data)
        assert ips == ["1.2.3.4"]

    def test_invalid_ip_filtered(self) -> None:
        r = self._resolver()
        data = {
            "Status": 0,
            "Answer": [
                {"type": 1, "data": "999.999.999.999", "TTL": 60},
                {"type": 1, "data": "1.2.3.4", "TTL": 60},
            ],
        }
        ips, _ = r._parse_dns_json(data)
        assert ips == ["1.2.3.4"]

    def test_empty_answer(self) -> None:
        r = self._resolver()
        data = {"Status": 0}
        ips, ttl = r._parse_dns_json(data)
        assert ips == []

    def test_missing_status(self) -> None:
        r = self._resolver()
        data = {}
        ips, ttl = r._parse_dns_json(data)
        assert ips == []


# ──────────────────────────── DoHResolver.provider_health ────────────────────────────


class TestProviderHealth:
    def test_initial_stats(self) -> None:
        resolver = DoHResolver(session=MagicMock())
        health = resolver.provider_health()
        for name, stats in health.items():
            assert stats["success"] == 0
            assert stats["failure"] == 0
