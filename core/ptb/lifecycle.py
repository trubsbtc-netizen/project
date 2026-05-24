from __future__ import annotations

import asyncio
import hashlib
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from core.config import PolymarketConfig
from core.runtime.clock import bucket_5m, slug_for_bucket
from core.transport.json import loads
from core.types import Outcome, OutcomeToken, PTBMarket, PriceToBeat, TruthSource


class PTBMetadataUnavailable(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class _GammaMetadata:
    slug: str
    condition_id: str
    market_id: str
    token_ids: tuple[str, str]
    price_to_beat: float | None
    event_start_time: str
    raw_payload: Any


def deterministic_bucket(server_time: float | None = None) -> int:
    return bucket_5m(server_time)


def deterministic_slug(bucket: int) -> str:
    return slug_for_bucket(bucket)


class ImmutablePTBCache:
    def __init__(self) -> None:
        self._by_bucket: dict[int, PTBMarket] = {}
        self._lock = asyncio.Lock()

    async def get(self, bucket: int) -> PTBMarket | None:
        async with self._lock:
            return self._by_bucket.get(bucket)

    async def install(self, market: PTBMarket) -> PTBMarket:
        async with self._lock:
            existing = self._by_bucket.get(market.bucket)
            if existing is not None:
                return existing
            self._by_bucket[market.bucket] = market
            return market


class PTBProvider:
    def __init__(self, config: PolymarketConfig, cache: ImmutablePTBCache | None = None) -> None:
        self._config = config
        self._cache = cache or ImmutablePTBCache()
        self._inflight: dict[int, asyncio.Task[PTBMarket]] = {}
        self._bootstrap_lock = asyncio.Lock()
        self._startup_bootstrap_bucket: int | None = None
        self._startup_bootstrap_task: asyncio.Task[PTBMarket] | None = None

    async def install_round(self, market: PTBMarket) -> PTBMarket:
        return await self._cache.install(market)

    async def get_cached_round(self, server_time: float | None = None) -> PTBMarket | None:
        return await self._cache.get(deterministic_bucket(server_time))

    async def get_round(self, server_time: float | None = None) -> PTBMarket:
        bucket = deterministic_bucket(server_time)
        cached = await self._cache.get(bucket)
        if cached is not None:
            return cached
        if self._config.ptb_mode == "static":
            return await self._fetch_and_install(
                bucket,
                allow_http=False,
                timeout_s=self._config.startup_bootstrap_timeout_s,
                task_name=f"ptb.static.{bucket}",
            )
        raise PTBMetadataUnavailable("PTB metadata unavailable and runtime HTTP metadata disabled")

    async def bootstrap_startup_round(self, server_time: float | None = None) -> PTBMarket:
        bucket = deterministic_bucket(server_time)
        cached = await self._cache.get(bucket)
        if cached is not None:
            return cached
        async with self._bootstrap_lock:
            cached = await self._cache.get(bucket)
            if cached is not None:
                return cached
            if self._startup_bootstrap_task is None:
                self._startup_bootstrap_bucket = bucket
                task = asyncio.create_task(
                    self._fetch_once(
                        bucket,
                        allow_http=True,
                        timeout_s=self._config.startup_bootstrap_timeout_s,
                    ),
                    name=f"ptb.bootstrap.{bucket}",
                )
                self._startup_bootstrap_task = task
                self._inflight[bucket] = task
            elif self._startup_bootstrap_bucket != bucket:
                raise PTBMetadataUnavailable("startup PTB bootstrap already consumed")
            task = self._startup_bootstrap_task
        try:
            market = await task
            return await self._cache.install(market)
        finally:
            if task.done():
                self._inflight.pop(bucket, None)

    async def _fetch_and_install(self, bucket: int, *, allow_http: bool, timeout_s: float, task_name: str) -> PTBMarket:
        task = self._inflight.get(bucket)
        if task is None:
            task = asyncio.create_task(
                self._fetch_once(bucket, allow_http=allow_http, timeout_s=timeout_s),
                name=task_name,
            )
            self._inflight[bucket] = task
        try:
            market = await task
            return await self._cache.install(market)
        finally:
            if task.done():
                self._inflight.pop(bucket, None)

    async def _fetch_once(self, bucket: int, *, allow_http: bool, timeout_s: float) -> PTBMarket:
        if self._config.ptb_mode == "static":
            return self._static_market(bucket)
        if not allow_http:
            raise PTBMetadataUnavailable("PTB metadata unavailable and cold HTTP metadata disabled")
        return await asyncio.to_thread(self._gamma_fetch, bucket, timeout_s)

    def _static_market(self, bucket: int) -> PTBMarket:
        slug = deterministic_slug(bucket)
        if not (self._config.static_condition_id and self._config.static_up_token_id and self._config.static_down_token_id):
            raise RuntimeError("static PTB mode requires PTB_CONDITION_ID, PTB_UP_TOKEN_ID, PTB_DOWN_TOKEN_ID")
        ptb = PriceToBeat(
            source=TruthSource.PTB,
            value=self._config.static_price_to_beat,
            timestamp_ms=bucket * 1000,
        )
        raw = f"{slug}:{self._config.static_condition_id}:{self._config.static_up_token_id}:{self._config.static_down_token_id}"
        return PTBMarket(
            bucket=bucket,
            slug=slug,
            condition_id=self._config.static_condition_id,
            market_id=self._config.static_market_id or slug,
            up=OutcomeToken(Outcome.UP, self._config.static_up_token_id),
            down=OutcomeToken(Outcome.DOWN, self._config.static_down_token_id),
            price_to_beat=ptb,
            open_ts=bucket,
            close_ts=bucket + 300,
            fetched_mono_ns=time.monotonic_ns(),
            raw_hash=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        )

    def _gamma_fetch(self, bucket: int, timeout_s: float) -> PTBMarket:
        metadata = self._fetch_gamma_metadata(bucket, timeout_s)
        price_to_beat = metadata.price_to_beat
        price_payload: Any = None
        if price_to_beat is None:
            price_to_beat, price_payload = self._fetch_crypto_open_price(metadata, timeout_s)
        raw_bytes = repr({"gamma": metadata.raw_payload, "crypto_price": price_payload}).encode("utf-8")
        return PTBMarket(
            bucket=bucket,
            slug=metadata.slug,
            condition_id=metadata.condition_id,
            market_id=metadata.market_id,
            up=OutcomeToken(Outcome.UP, metadata.token_ids[0]),
            down=OutcomeToken(Outcome.DOWN, metadata.token_ids[1]),
            price_to_beat=PriceToBeat(TruthSource.PTB, price_to_beat, bucket * 1000),
            open_ts=bucket,
            close_ts=bucket + 300,
            fetched_mono_ns=time.monotonic_ns(),
            raw_hash=hashlib.sha256(raw_bytes).hexdigest(),
        )

    def _fetch_gamma_metadata(self, bucket: int, timeout_s: float) -> _GammaMetadata:
        slug = deterministic_slug(bucket)
        encoded = urllib.parse.quote(slug, safe="")
        urls = [
            f"{self._config.gamma_base_url.rstrip('/')}/events/slug/{encoded}",
            f"{self._config.gamma_base_url.rstrip('/')}/events?slug={encoded}",
        ]
        last_error: Exception | None = None
        for url in urls:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "btc-poly-institutional/1.0"})
                with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                    payload = loads(resp.read())
                return self._parse_gamma_metadata(bucket, slug, payload)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"failed to fetch immutable PTB metadata for {slug}: {last_error}")

    def _parse_gamma_metadata(self, bucket: int, slug: str, payload: Any) -> _GammaMetadata:
        event = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(event, dict):
            raise RuntimeError("invalid gamma PTB payload")
        markets = event.get("markets") or []
        market = self._select_btc_updown_market(markets)
        if market is None:
            market = event
        condition_id = str(market.get("conditionId") or market.get("condition_id") or event.get("conditionId") or "")
        market_id = str(market.get("id") or market.get("marketId") or event.get("id") or slug)
        token_ids = self._extract_token_ids(market)
        if len(token_ids) < 2:
            raise RuntimeError(f"could not extract UP/DOWN token ids for {slug}")
        return _GammaMetadata(
            slug=slug,
            condition_id=condition_id,
            market_id=market_id,
            token_ids=(token_ids[0], token_ids[1]),
            price_to_beat=self._extract_price_to_beat(event, market, bucket),
            event_start_time=str(market.get("eventStartTime") or event.get("startTime") or ""),
            raw_payload=payload,
        )

    def _fetch_crypto_open_price(self, metadata: _GammaMetadata, timeout_s: float) -> tuple[float, Any]:
        if not metadata.event_start_time:
            raise RuntimeError(f"price to beat missing for {metadata.slug}")
        query = urllib.parse.urlencode(
            {
                "symbol": "BTC",
                "eventStartTime": metadata.event_start_time,
                "variant": "fiveminute",
            }
        )
        url = f"{self._config.web_base_url.rstrip('/')}/api/crypto/crypto-price?{query}"
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "btc-poly-institutional/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            payload = loads(resp.read())
        if not isinstance(payload, dict):
            raise RuntimeError(f"invalid crypto open price payload for {metadata.slug}")
        raw = payload.get("openPrice") or payload.get("priceToBeat")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise RuntimeError(f"price to beat missing for {metadata.slug}") from None
        if value <= 0.0:
            raise RuntimeError(f"invalid price to beat for {metadata.slug}: {value}")
        return value, payload

    def _select_btc_updown_market(self, markets: list[Any]) -> dict[str, Any] | None:
        for item in markets:
            if not isinstance(item, dict):
                continue
            outcomes = str(item.get("outcomes") or "").lower()
            question = str(item.get("question") or item.get("title") or "").lower()
            if ("up" in outcomes and "down" in outcomes) or ("bitcoin" in question and ("up" in question or "down" in question)):
                return item
        for item in markets:
            if isinstance(item, dict):
                return item
        return None

    def _extract_token_ids(self, market: dict[str, Any]) -> list[str]:
        candidates = (
            market.get("clobTokenIds"),
            market.get("clob_token_ids"),
            market.get("tokenIds"),
            market.get("tokens"),
        )
        for candidate in candidates:
            values: list[Any] = []
            if isinstance(candidate, str) and candidate:
                try:
                    parsed = loads(candidate)
                except Exception:
                    parsed = None
                values = parsed if isinstance(parsed, list) else []
            elif isinstance(candidate, list):
                values = candidate
            if not values:
                continue
            token_ids: list[str] = []
            for item in values:
                if isinstance(item, str):
                    token_ids.append(item)
                elif isinstance(item, dict):
                    token_id = item.get("token_id") or item.get("tokenId") or item.get("id")
                    if token_id:
                        outcome = str(item.get("outcome") or item.get("name") or "").upper()
                        if outcome == "UP":
                            token_ids.insert(0, str(token_id))
                        elif outcome == "DOWN":
                            token_ids.append(str(token_id))
                        else:
                            token_ids.append(str(token_id))
            if len(token_ids) >= 2:
                return token_ids
        return []

    def _extract_price_to_beat(self, event: dict[str, Any], market: dict[str, Any], bucket: int) -> float | None:
        keys = (
            "priceToBeat",
            "price_to_beat",
            "ptb",
            "initialPrice",
            "initial_price",
            "startPrice",
            "start_price",
            "strike",
        )
        for obj in (market, event):
            for key in keys:
                raw = obj.get(key)
                if raw is None:
                    continue
                try:
                    value = float(raw)
                    if value > 0.0:
                        return value
                except (TypeError, ValueError):
                    continue
        return None
