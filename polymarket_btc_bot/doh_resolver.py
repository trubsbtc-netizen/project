"""
DNS-over-HTTPS Resolver for Polymarket Trading Bot

Provides low-latency DNS resolution with caching, failover, and automatic refresh.
Uses DoH to prevent DNS poisoning and ensure consistent resolution.
"""

import asyncio
import logging
import time
import hashlib
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
from collections import OrderedDict
import aiohttp
import struct

logger = logging.getLogger(__name__)


@dataclass
class DNSCacheEntry:
    """Cached DNS response with TTL tracking."""
    ip_addresses: List[str]
    expires_at: float
    created_at: float
    provider: str
    query_count: int = 0
    
    def is_valid(self) -> bool:
        return time.time() < self.expires_at
    
    def time_to_expiry(self) -> float:
        return max(0, self.expires_at - time.time())


@dataclass 
class DoHProvider:
    """DNS-over-HTTPS provider configuration."""
    url: str
    name: str
    healthy: bool = True
    last_failure: Optional[float] = None
    failure_count: int = 0
    avg_latency_ms: float = 0.0
    
    def mark_unhealthy(self):
        self.healthy = False
        self.last_failure = time.time()
        self.failure_count += 1
    
    def mark_healthy(self, latency_ms: float):
        self.healthy = True
        # Exponential moving average for latency
        alpha = 0.3
        self.avg_latency_ms = alpha * latency_ms + (1 - alpha) * self.avg_latency_ms


class DNSOverHTTPSResolver:
    """
    Production-grade DNS-over-HTTPS resolver with:
    - Multiple provider failover
    - LRU cache with TTL
    - Background refresh
    - Health monitoring
    - Low-latency design
    """
    
    # DNS record type for A records
    DNS_TYPE_A = 1
    
    def __init__(
        self,
        providers: List[Dict[str, str]],
        cache_ttl_seconds: int = 300,
        cache_max_size: int = 1000,
        timeout_seconds: float = 5.0,
        refresh_threshold_seconds: float = 60.0,
    ):
        self.providers = [DoHProvider(**p) for p in providers]
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_max_size = cache_max_size
        self.timeout_seconds = timeout_seconds
        self.refresh_threshold_seconds = refresh_threshold_seconds
        
        # LRU cache: hostname -> DNSCacheEntry
        self._cache: OrderedDict[str, DNSCacheEntry] = OrderedDict()
        self._cache_lock = asyncio.Lock()
        
        # HTTP session for DoH queries
        self._session: Optional[aiohttp.ClientSession] = None
        
        # Background tasks
        self._refresh_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Metrics
        self._query_count = 0
        self._cache_hit_count = 0
        self._failover_count = 0
    
    async def start(self):
        """Initialize HTTP session and start background tasks."""
        connector = aiohttp.TCPConnector(
            ttl_dns_cache=300,
            use_dns_cache=True,
            limit=10,
        )
        
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=self.timeout_seconds),
        )
        
        # Start background tasks
        self._refresh_task = asyncio.create_task(self._background_refresh_loop())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop(self):
        """Shutdown resolver and cleanup resources."""
        if self._refresh_task:
            self._refresh_task.cancel()
            try:
                await self._refresh_task
            except asyncio.CancelledError:
                pass
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self._session:
            await self._session.close()
    
    async def resolve(self, hostname: str) -> List[str]:
        """
        Resolve hostname to IP addresses using DoH.
        
        Returns list of IP addresses, preferring cached valid entries.
        Implements failover across providers.
        """
        self._query_count += 1
        
        # Check cache first
        async with self._cache_lock:
            if hostname in self._cache:
                entry = self._cache[hostname]
                
                # Move to end (most recently used)
                self._cache.move_to_end(hostname)
                entry.query_count += 1
                
                if entry.is_valid():
                    self._cache_hit_count += 1
                    
                    # Check if we should refresh in background
                    if entry.time_to_expiry() < self.refresh_threshold_seconds:
                        asyncio.create_task(self._refresh_entry(hostname, entry))
                    
                    return entry.ip_addresses
        
        # Cache miss or expired - perform lookup
        return await self._lookup_with_failover(hostname)
    
    async def _lookup_with_failover(self, hostname: str) -> List[str]:
        """Try providers in order until one succeeds."""
        # Sort providers by health and latency
        healthy_providers = sorted(
            [p for p in self.providers if p.healthy],
            key=lambda p: (p.failure_count, p.avg_latency_ms)
        )
        
        if not healthy_providers:
            # All providers unhealthy - try all anyway
            healthy_providers = self.providers
        
        last_error = None
        
        for provider in healthy_providers:
            try:
                start_time = time.perf_counter()
                ips = await self._doh_query(hostname, provider.url)
                latency_ms = (time.perf_counter() - start_time) * 1000
                
                provider.mark_healthy(latency_ms)
                
                # Cache the result
                await self._cache_result(hostname, ips, provider.name)
                
                return ips
                
            except Exception as e:
                provider.mark_unhealthy()
                last_error = e
                self._failover_count += 1
                continue
        
        # All providers failed
        raise DNSResolutionError(
            f"All DoH providers failed for {hostname}: {last_error}"
        )
    
    async def _doh_query(self, hostname: str, provider_url: str) -> List[str]:
        """
        Perform DNS query via DoH.
        
        Uses DNS wire format over HTTPS POST.
        """
        # Build DNS query packet
        query_packet = self._build_dns_query(hostname)
        
        headers = {
            'Content-Type': 'application/dns-message',
            'Accept': 'application/dns-message',
        }
        
        async with self._session.post(
            provider_url,
            data=query_packet,
            headers=headers,
        ) as response:
            if response.status != 200:
                raise DNSResolutionError(
                    f"DoH provider returned status {response.status}"
                )
            
            response_data = await response.read()
            return self._parse_dns_response(response_data)
    
    def _build_dns_query(self, hostname: str) -> bytes:
        """Build DNS query packet in wire format."""
        # Transaction ID (2 bytes)
        transaction_id = b'\x00\x01'
        
        # Flags (2 bytes) - standard query
        flags = b'\x01\x00'
        
        # Questions count (2 bytes)
        qdcount = b'\x00\x01'
        
        # Answer count (2 bytes)
        ancount = b'\x00\x00'
        
        # Authority count (2 bytes)
        nscount = b'\x00\x00'
        
        # Additional count (2 bytes)
        arcount = b'\x00\x00'
        
        # Header
        header = transaction_id + flags + qdcount + ancount + nscount + arcount
        
        # Question section
        question = b''
        for label in hostname.split('.'):
            question += bytes([len(label)]) + label.encode('ascii')
        question += b'\x00'  # Null terminator
        
        # Query type (A record = 1)
        qtype = b'\x00\x01'
        
        # Query class (IN = 1)
        qclass = b'\x00\x01'
        
        return header + question + qtype + qclass
    
    def _parse_dns_response(self, data: bytes) -> List[str]:
        """Parse DNS response packet and extract IP addresses."""
        if len(data) < 12:
            raise DNSResolutionError("DNS response too short")
        
        # Parse header
        qdcount = struct.unpack('!H', data[4:6])[0]
        ancount = struct.unpack('!H', data[6:8])[0]
        
        # Skip to answer section
        offset = 12
        
        # Skip questions
        for _ in range(qdcount):
            while offset < len(data) and data[offset] != 0:
                offset += data[offset] + 1
            offset += 5  # null byte + qtype + qclass
        
        # Parse answers
        ips = []
        for _ in range(ancount):
            if offset >= len(data):
                break
            
            # Skip name (may be compressed)
            while offset < len(data) and data[offset] != 0:
                if (data[offset] & 0xC0) == 0xC0:
                    offset += 2
                    break
                offset += data[offset] + 1
            else:
                offset += 1
            
            # Read type, class, TTL, rdlength
            if offset + 10 > len(data):
                break
            
            rtype = struct.unpack('!H', data[offset:offset+2])[0]
            offset += 4  # skip type and class
            offset += 4  # skip TTL
            rdlength = struct.unpack('!H', data[offset:offset+2])[0]
            offset += 2
            
            # Extract A record (IPv4)
            if rtype == self.DNS_TYPE_A and rdlength == 4:
                ip = '.'.join(str(b) for b in data[offset:offset+4])
                ips.append(ip)
            
            offset += rdlength
        
        if not ips:
            raise DNSResolutionError("No A records in DNS response")
        
        return ips
    
    async def _cache_result(self, hostname: str, ips: List[str], provider: str):
        """Cache DNS resolution result."""
        async with self._cache_lock:
            now = time.time()
            entry = DNSCacheEntry(
                ip_addresses=ips,
                expires_at=now + self.cache_ttl_seconds,
                created_at=now,
                provider=provider,
            )
            
            self._cache[hostname] = entry
            
            # Enforce cache size limit
            while len(self._cache) > self.cache_max_size:
                self._cache.popitem(last=False)
    
    async def _refresh_entry(self, hostname: str, entry: DNSCacheEntry):
        """Refresh a cache entry in the background."""
        try:
            ips = await self._lookup_with_failover(hostname)
            
            async with self._cache_lock:
                if hostname in self._cache:
                    # Update existing entry
                    old_entry = self._cache[hostname]
                    old_entry.ip_addresses = ips
                    old_entry.expires_at = time.time() + self.cache_ttl_seconds
                    old_entry.provider = entry.provider
        except Exception as e:
            logger.debug("Cache refresh failed for %s, keeping stale entry: %s", hostname, e)
    
    async def _background_refresh_loop(self):
        """Periodically refresh cache entries approaching expiry."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                async with self._cache_lock:
                    to_refresh = [
                        hostname for hostname, entry in self._cache.items()
                        if entry.time_to_expiry() < self.refresh_threshold_seconds
                    ]
                
                for hostname in to_refresh:
                    asyncio.create_task(self._refresh_entry(hostname, self._cache[hostname]))
                    
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("Background DNS refresh loop error: %s", exc)
    
    async def _health_check_loop(self):
        """Periodically check health of failed providers."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                for provider in self.providers:
                    if not provider.healthy and provider.last_failure:
                        # Try to recover after 5 minutes
                        if time.time() - provider.last_failure > 300:
                            provider.healthy = True
                            
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("DNS health check loop error: %s", exc)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Return resolver metrics."""
        return {
            'query_count': self._query_count,
            'cache_hit_count': self._cache_hit_count,
            'cache_hit_rate': self._cache_hit_count / max(1, self._query_count),
            'failover_count': self._failover_count,
            'cache_size': len(self._cache),
            'providers': [
                {
                    'name': p.name,
                    'healthy': p.healthy,
                    'failure_count': p.failure_count,
                    'avg_latency_ms': p.avg_latency_ms,
                }
                for p in self.providers
            ],
        }


class DNSResolutionError(Exception):
    """Raised when DNS resolution fails."""
    pass
