from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from polymarket_bot.infrastructure.config import InfrastructureConfig
from polymarket_bot.infrastructure.network import ConnectionPool
from polymarket_bot.infrastructure.types import MarketInfo, _safe_float, _utc_timestamp

logger = logging.getLogger(__name__)


class MarketDiscovery:
    def __init__(self, config: InfrastructureConfig, pool: ConnectionPool):
        self.config = config
        self.pool = pool

    def deterministic_slug(self, start_ts: float) -> str:
        bucket = int(start_ts) - (int(start_ts) % 300)
        return f"{self.config.market_slug_prefix}{bucket}"

    def candidate_slugs(self, now_ts: Optional[float] = None) -> Tuple[str, ...]:
        now = int(now_ts or time.time())
        bucket = now - (now % 300)
        starts = (bucket - 300, bucket, bucket + 300, bucket + 600)
        return tuple(dict.fromkeys(self.deterministic_slug(start) for start in starts))

    async def find_active_market(self) -> Optional[MarketInfo]:
        candidates = await self._find_candidate_markets(max_time_to_end_s=600.0)
        return candidates[0] if candidates else None

    async def get_active_round(self) -> Optional[MarketInfo]:
        return await self.find_active_market()

    async def find_warm_markets(self, count: Optional[int] = None) -> Tuple[MarketInfo, ...]:
        candidates = await self._find_candidate_markets(max_time_to_end_s=self.config.market_prewarm_horizon_s)
        limit = max(1, count or self.config.market_prewarm_count)
        return tuple(candidates[:limit])

    async def refresh_market(self, market: MarketInfo) -> MarketInfo:
        if not market.condition_id:
            return market
        try:
            payload = await self._query_clob_market_by_condition(market.condition_id)
            if not payload:
                return market
            tick_size = str(payload.get("mts") or payload.get("minimum_tick_size") or market.tick_size)
            min_order_size = _safe_float(payload.get("mos") or payload.get("minimum_order_size"), market.min_order_size)
            fee_rate = market.fee_rate
            fee_exponent = market.fee_exponent
            fd = payload.get("fd") if isinstance(payload.get("fd"), dict) else {}
            if fd:
                fee_rate = _safe_float(fd.get("r"), fee_rate)
                fee_exponent = _safe_float(fd.get("e"), fee_exponent)
            from dataclasses import replace
            return replace(
                market,
                tick_size=tick_size,
                min_order_size=min_order_size,
                fee_rate=fee_rate,
                fee_exponent=fee_exponent,
                fees_enabled=fee_rate > 0,
                raw={**market.raw, "clob_refresh": payload},
            )
        except Exception as exc:
            logger.debug("market refresh failed for %s: %s", market.slug, exc)
            return market

    async def refresh_price_to_beat(self, slug: str) -> Optional[float]:
        if not slug:
            return None
        value = await self._query_official_price_to_beat(slug)
        if value is not None:
            return value
        market = await self._query_gamma_market_by_slug(slug, include_closed=True)
        value = self._extract_price_to_beat(market) if market else None
        if value is not None:
            return value
        event = await self._query_gamma_event_by_slug(slug, include_closed=True)
        value = self._extract_price_to_beat(event) if event else None
        return value

    async def _find_candidate_markets(self, max_time_to_end_s: float) -> List[MarketInfo]:
        raw: List[Dict[str, Any]] = []
        for slug in self.candidate_slugs():
            market = await self._query_gamma_market_by_slug(slug)
            if market and self._is_btc_5m(market):
                raw.append(market)
        if not raw:
            raw = await self._query_gamma_markets()
        if not raw:
            raw = await self._query_clob_markets()

        now = time.time()
        parsed: List[MarketInfo] = []
        seen: set[str] = set()
        for item in raw:
            info = self._parse_market_info(item, now=now, max_time_to_end_s=max_time_to_end_s)
            if not info or info.slug in seen:
                continue
            seen.add(info.slug)
            parsed.append(info)
        parsed.sort(key=lambda market: market.end_time)
        return parsed

    async def _query_official_price_to_beat(self, slug: str) -> Optional[float]:
        try:
            session = await self.pool.session()
            url = f"https://polymarket.com/api/equity/price-to-beat/{slug}"
            async with session.get(url, headers={"Accept": "application/json"}) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json(content_type=None)
            if isinstance(payload, (int, float)):
                return float(payload) if payload > 0 else None
            if isinstance(payload, dict):
                for key in ("price", "priceToBeat", "price_to_beat", "value", "strikePrice"):
                    value = _safe_float(payload.get(key), 0.0)
                    if value > 0:
                        return value
        except Exception:
            return None
        return None

    async def _query_gamma_market_by_slug(self, slug: str, include_closed: bool = False) -> Optional[Dict[str, Any]]:
        try:
            session = await self.pool.session()
            params = {"slug": slug}
            if include_closed:
                params["closed"] = "true"
            async with session.get(f"{self.config.gamma_api}/markets", params=params) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json(content_type=None)
            if isinstance(payload, list) and payload:
                return payload[0]
            if isinstance(payload, dict):
                items = payload.get("data") or payload.get("markets") or []
                if isinstance(items, list) and items:
                    return items[0]
        except Exception:
            return None
        return None

    async def _query_gamma_event_by_slug(self, slug: str, include_closed: bool = False) -> Optional[Dict[str, Any]]:
        try:
            session = await self.pool.session()
            params = {"slug": slug}
            if include_closed:
                params["closed"] = "true"
            async with session.get(f"{self.config.gamma_api}/events", params=params) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json(content_type=None)
            if isinstance(payload, list) and payload:
                return payload[0]
            if isinstance(payload, dict):
                items = payload.get("data") or payload.get("events") or []
                if isinstance(items, list) and items:
                    return items[0]
                return payload
        except Exception:
            return None
        return None

    async def _query_gamma_markets(self) -> List[Dict[str, Any]]:
        try:
            session = await self.pool.session()
            async with session.get(
                f"{self.config.gamma_api}/markets",
                params={"active": "true", "closed": "false", "limit": "200"},
            ) as resp:
                if resp.status != 200:
                    return []
                payload = await resp.json(content_type=None)
            items = payload if isinstance(payload, list) else payload.get("data") or payload.get("markets") or []
            return [item for item in items if isinstance(item, dict) and self._is_btc_5m(item)]
        except Exception:
            return []

    async def _query_clob_markets(self) -> List[Dict[str, Any]]:
        try:
            session = await self.pool.session()
            async with session.get(f"{self.config.polymarket_host}/simplified-markets", params={"next_cursor": ""}) as resp:
                if resp.status != 200:
                    return []
                payload = await resp.json(content_type=None)
            items = payload if isinstance(payload, list) else payload.get("data") or []
            return [item for item in items if isinstance(item, dict) and self._is_btc_5m(item)]
        except Exception:
            return []

    async def _query_clob_market_by_condition(self, condition_id: str) -> Optional[Dict[str, Any]]:
        try:
            session = await self.pool.session()
            async with session.get(f"{self.config.polymarket_host}/clob-markets/{condition_id}") as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json(content_type=None)
            return payload if isinstance(payload, dict) else None
        except Exception:
            return None

    def _parse_market_info(
        self,
        item: Dict[str, Any],
        *,
        now: float,
        max_time_to_end_s: float,
    ) -> Optional[MarketInfo]:
        end_time = (
            _utc_timestamp(item.get("endDate"))
            or _utc_timestamp(item.get("endDateIso"))
            or _utc_timestamp(item.get("end_time"))
        )
        if end_time is None or end_time <= now or (end_time - now) > max_time_to_end_s:
            return None
        slug = str(item.get("slug") or item.get("market_slug") or item.get("ticker") or "").strip()
        if not slug:
            return None
        tokens = self._parse_token_ids(item)
        if len(tokens) < 2:
            return None
        outcomes = self._parse_outcomes(item)
        up_idx, down_idx = self._outcome_indices(outcomes)
        start_from_slug = self._start_time_from_slug(slug)
        event_start = (
            start_from_slug
            or _utc_timestamp(item.get("eventStartTime"))
            or _utc_timestamp(item.get("event_start_time"))
            or _utc_timestamp(item.get("startDate"))
            or (end_time - 300.0)
        )
        fee_data = item.get("fd") if isinstance(item.get("fd"), dict) else {}
        return MarketInfo(
            slug=slug,
            condition_id=str(item.get("condition_id") or item.get("conditionId") or ""),
            question=str(item.get("question") or item.get("title") or "").strip(),
            up_token_id=tokens[min(up_idx, len(tokens) - 1)],
            down_token_id=tokens[min(down_idx, len(tokens) - 1)],
            price_to_beat=self._extract_price_to_beat(item) or 0.0,
            end_time=float(end_time),
            event_start_time=float(event_start),
            tick_size=str(item.get("orderPriceMinTickSize") or item.get("tick_size") or "0.01"),
            neg_risk=bool(item.get("negRisk", item.get("neg_risk", False))),
            min_order_size=_safe_float(item.get("orderMinSize") or item.get("min_order_size"), 5.0),
            fees_enabled=bool(item.get("feesEnabled", item.get("fees_enabled", True))),
            fee_rate=_safe_float(fee_data.get("r") if fee_data else item.get("feeRate"), 0.0),
            fee_exponent=_safe_float(fee_data.get("e") if fee_data else item.get("feeExponent"), 1.0),
            raw=dict(item),
        )

    def _is_btc_5m(self, item: Dict[str, Any]) -> bool:
        slug = str(item.get("slug") or item.get("market_slug") or item.get("ticker") or "").lower()
        if slug.startswith(self.config.market_slug_prefix):
            return True
        question = str(item.get("question") or item.get("title") or "").lower()
        tags = [str(tag).lower() for tag in (item.get("tags") or [])]
        if self.config.market_search_tag.lower() in tags:
            return True
        return ("btc" in question or "bitcoin" in question) and ("5" in question or "up or down" in question)

    def _start_time_from_slug(self, slug: str) -> Optional[float]:
        if not slug.startswith(self.config.market_slug_prefix):
            return None
        suffix = slug[len(self.config.market_slug_prefix):].split("-")[0]
        try:
            value = float(suffix)
            return value if value > 1_600_000_000 else None
        except ValueError:
            return None

    def _parse_jsonish_list(self, value: Any) -> List[str]:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
            return [part.strip() for part in value.split(",") if part.strip()]
        return []

    def _parse_token_ids(self, item: Dict[str, Any]) -> List[str]:
        for key in ("clobTokenIds", "clob_token_ids", "asset_ids", "assetIds", "assets_ids"):
            tokens = self._parse_jsonish_list(item.get(key))
            if tokens:
                return tokens
        raw_tokens = item.get("tokens") or []
        if isinstance(raw_tokens, list):
            parsed: List[str] = []
            for token in raw_tokens:
                if not isinstance(token, dict):
                    continue
                token_id = (
                    token.get("token_id")
                    or token.get("tokenId")
                    or token.get("asset_id")
                    or token.get("assetId")
                    or token.get("id")
                    or token.get("t")
                )
                if token_id:
                    parsed.append(str(token_id))
            if parsed:
                return parsed
        compact = item.get("t") or []
        if isinstance(compact, list):
            return [str(token.get("t")) for token in compact if isinstance(token, dict) and token.get("t")]
        return []

    def _parse_outcomes(self, item: Dict[str, Any]) -> List[str]:
        for key in ("outcomes", "outcome_names", "outcomeNames"):
            values = self._parse_jsonish_list(item.get(key))
            if values:
                return values
        raw_tokens = item.get("tokens") or []
        if isinstance(raw_tokens, list):
            values = [str(token.get("outcome", "")) for token in raw_tokens if isinstance(token, dict)]
            if any(values):
                return values
        compact = item.get("t") or []
        if isinstance(compact, list):
            return [str(token.get("o", "")) for token in compact if isinstance(token, dict)]
        return []

    def _outcome_indices(self, outcomes: Sequence[str]) -> Tuple[int, int]:
        up_idx, down_idx = 0, 1
        for i, outcome in enumerate(outcomes):
            text = str(outcome).lower()
            if "up" in text or "yes" in text or "above" in text:
                up_idx = i
            elif "down" in text or "no" in text or "below" in text:
                down_idx = i
        return up_idx, down_idx

    def _extract_price_to_beat(self, item: Any) -> Optional[float]:
        if not isinstance(item, dict):
            return None

        def scan(value: Any) -> Optional[float]:
            if isinstance(value, str) and value.strip().startswith(("{", "[")):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    return None
            if isinstance(value, dict):
                for key in ("priceToBeat", "price_to_beat", "price_to_beat_value", "strikePrice", "strike_price"):
                    parsed = _safe_float(value.get(key), 0.0)
                    if parsed > 0:
                        return parsed
                for nested in value.values():
                    parsed = scan(nested)
                    if parsed is not None:
                        return parsed
            elif isinstance(value, list):
                for nested in value:
                    parsed = scan(nested)
                    if parsed is not None:
                        return parsed
            return None

        parsed = scan(item)
        if parsed is not None:
            return parsed
        question = str(item.get("question") or item.get("title") or "")
        for m in re.finditer(r"\$\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.\d+)?)", question):
            value = _safe_float(m.group(1), 0.0)
            if value > 0:
                return value
        return None
        return None
