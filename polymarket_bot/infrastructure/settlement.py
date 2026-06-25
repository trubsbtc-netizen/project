from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from polymarket_bot.infrastructure.config import InfrastructureConfig
from polymarket_bot.infrastructure.network import ConnectionPool
from polymarket_bot.infrastructure.types import (
    MarketInfo,
    MarketResolution,
    Outcome,
    Outcome as ResolutionOutcome,
    _safe_float,
)

logger = logging.getLogger(__name__)


class SettlementEngine:
    def __init__(self, config: InfrastructureConfig, pool: ConnectionPool):
        self.config = config
        self.pool = pool
        self._ws_cache: Dict[str, Dict[str, Any]] = {}
        self._resolution_cache: Dict[str, MarketResolution] = {}

    def is_settled(self, slug: str) -> bool:
        resolution = self._resolution_cache.get(slug)
        return resolution is not None and resolution.is_known

    def get_resolution(self, slug: str) -> Optional[MarketResolution]:
        return self._resolution_cache.get(slug)

    def record_ws_event(self, payload: Dict[str, Any]) -> None:
        for key in self._resolution_keys(payload):
            self._ws_cache[key] = dict(payload)

    def cached_resolution(self, market: MarketInfo) -> Optional[MarketResolution]:
        return self._resolution_cache.get(market.slug)

    async def fetch_resolution(self, market: MarketInfo) -> Optional[MarketResolution]:
        candidates: List[Tuple[str, Optional[Dict[str, Any]]]] = []
        ws_payload = self._cached_ws_payload(market)
        if ws_payload:
            candidates.append(("clob_ws_market_resolved", ws_payload))
        if market.condition_id:
            candidates.append(("clob_condition", await self._query_clob_market_by_condition(market.condition_id)))
        if market.up_token_id:
            candidates.append(("clob_token_up", await self._query_clob_market_by_token(market.up_token_id)))
        if market.down_token_id:
            candidates.append(("clob_token_down", await self._query_clob_market_by_token(market.down_token_id)))
        if market.slug:
            candidates.append(("gamma_market", await self._query_gamma_market_by_slug(market.slug)))
            candidates.append(("gamma_event", await self._query_gamma_event_by_slug(market.slug)))

        for source, payload in candidates:
            resolution = self._parse_resolution_payload(payload, market, source)
            if resolution and resolution.is_known:
                onchain = await self.read_onchain_payouts(market.condition_id)
                if onchain:
                    resolution = self._merge_onchain(resolution, onchain)
                self._resolution_cache[market.slug] = resolution
                return resolution
        return None

    async def wait_for_resolution(self, market: MarketInfo) -> Optional[MarketResolution]:
        deadline = time.monotonic() + max(1.0, self.config.settlement_poll_timeout_s)
        interval = max(1.0, self.config.settlement_poll_interval_s)
        required = max(1, self.config.settlement_confirmations_required)
        last_key = ""
        confirmations = 0
        while time.monotonic() <= deadline:
            resolution = await self.fetch_resolution(market)
            if resolution and resolution.is_known:
                key = f"{resolution.outcome.value}:{resolution.winner_token_id}"
                confirmations = confirmations + 1 if key == last_key else 1
                last_key = key
                if confirmations >= required:
                    final = replace(resolution, confirmations=confirmations)
                    self._resolution_cache[market.slug] = final
                    return final
            await asyncio.sleep(interval)
        return None

    async def read_onchain_payouts(self, condition_id: str) -> Optional[MarketResolution]:
        if not condition_id:
            return None
        try:
            from eth_utils import keccak
        except ImportError:
            return None

        condition_hex = condition_id.replace("0x", "").replace("0X", "").rjust(64, "0")[-64:]
        numerator_selector = keccak(text="payoutNumerators(bytes32,uint256)")[:4].hex()
        denominator_selector = keccak(text="payoutDenominator(bytes32)")[:4].hex()

        async def eth_call(data: str) -> Optional[int]:
            for rpc_url in self.config.polygon_rpc_urls:
                try:
                    session = await self.pool.session()
                    payload = {
                        "jsonrpc": "2.0",
                        "id": int(time.time() * 1000) % 1_000_000,
                        "method": "eth_call",
                        "params": [{"to": self.config.ctf_address, "data": data}, "latest"],
                    }
                    async with session.post(rpc_url, json=payload) as resp:
                        if resp.status != 200:
                            continue
                        response = await resp.json(content_type=None)
                    result = response.get("result")
                    if isinstance(result, str) and result.startswith("0x"):
                        return int(result, 16)
                except Exception as exc:
                    logger.debug("On-chain eth_call failed for %s: %s", rpc_url, exc)
                    continue
            return None

        nums: List[int] = []
        for index in (0, 1):
            encoded_index = hex(index)[2:].rjust(64, "0")
            value = await eth_call(f"0x{numerator_selector}{condition_hex}{encoded_index}")
            if value is None:
                return None
            nums.append(value)
        denominator = await eth_call(f"0x{denominator_selector}{condition_hex}")
        if denominator is None:
            return None

        outcome = Outcome.UNKNOWN
        if denominator > 0 and len(nums) >= 2:
            if nums[0] > nums[1]:
                outcome = Outcome.UP
            elif nums[1] > nums[0]:
                outcome = Outcome.DOWN
        return MarketResolution(
            resolved=denominator > 0,
            outcome=outcome,
            payout_numerators=tuple(nums),
            payout_denominator=denominator,
            source="conditional_tokens",
            onchain_verified=denominator > 0,
        )

    def _merge_onchain(self, resolution: MarketResolution, onchain: MarketResolution) -> MarketResolution:
        verified = False
        if onchain.payout_denominator and onchain.payout_numerators:
            if resolution.outcome == Outcome.UP:
                verified = onchain.payout_numerators[0] > onchain.payout_numerators[1]
            elif resolution.outcome == Outcome.DOWN:
                verified = onchain.payout_numerators[1] > onchain.payout_numerators[0]
        return replace(
            resolution,
            payout_numerators=onchain.payout_numerators,
            payout_denominator=onchain.payout_denominator,
            onchain_verified=verified,
        )

    async def _query_clob_market_by_condition(self, condition_id: str) -> Optional[Dict[str, Any]]:
        try:
            session = await self.pool.session()
            async with session.get(f"{self.config.polymarket_host}/clob-markets/{condition_id}") as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json(content_type=None)
            return payload if isinstance(payload, dict) else None
        except Exception as exc:
            logger.debug("CLOB market query by condition failed for %s: %s", condition_id, exc)
            return None

    async def _query_clob_market_by_token(self, token_id: str) -> Optional[Dict[str, Any]]:
        try:
            session = await self.pool.session()
            async with session.get(f"{self.config.polymarket_host}/markets-by-token/{token_id}") as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json(content_type=None)
            return payload if isinstance(payload, dict) else None
        except Exception as exc:
            logger.debug("CLOB market query by token failed for %s: %s", token_id, exc)
            return None

    async def _query_gamma_market_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        try:
            session = await self.pool.session()
            async with session.get(f"{self.config.gamma_api}/markets", params={"slug": slug, "closed": "true"}) as resp:
                if resp.status != 200:
                    return None
                payload = await resp.json(content_type=None)
            if isinstance(payload, list) and payload:
                return payload[0]
            if isinstance(payload, dict):
                items = payload.get("data") or payload.get("markets") or []
                if isinstance(items, list) and items:
                    return items[0]
                return payload
        except Exception as exc:
            logger.debug("Gamma market query by slug failed for %s: %s", slug, exc)
            return None
        return None

    async def _query_gamma_event_by_slug(self, slug: str) -> Optional[Dict[str, Any]]:
        try:
            session = await self.pool.session()
            async with session.get(f"{self.config.gamma_api}/events", params={"slug": slug, "closed": "true"}) as resp:
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
        except Exception as exc:
            logger.debug("Gamma event query by slug failed for %s: %s", slug, exc)
            return None
        return None

    def _cached_ws_payload(self, market: MarketInfo) -> Optional[Dict[str, Any]]:
        for key in (market.condition_id.lower(), market.slug.lower(), market.up_token_id, market.down_token_id):
            if key and key in self._ws_cache:
                return dict(self._ws_cache[key])
        return None

    def _parse_resolution_payload(
        self,
        payload: Optional[Dict[str, Any]],
        market: MarketInfo,
        source: str,
    ) -> Optional[MarketResolution]:
        if not isinstance(payload, dict):
            return None
        for item in self._flatten(payload):
            resolution = self._resolution_from_object(item, market, source)
            if resolution:
                return resolution
        return None

    def _flatten(self, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], ...]:
        items: List[Dict[str, Any]] = [payload]
        for key in ("markets", "data", "children"):
            value = payload.get(key)
            if isinstance(value, list):
                items.extend(item for item in value if isinstance(item, dict))
            elif isinstance(value, dict):
                items.append(value)
        return tuple(items)

    def _resolution_from_object(
        self,
        obj: Dict[str, Any],
        market: MarketInfo,
        source: str,
    ) -> Optional[MarketResolution]:
        obj_slug = str(obj.get("slug") or obj.get("market_slug") or obj.get("ticker") or "").strip()
        if obj_slug and market.slug and obj_slug != market.slug:
            return None
        obj_condition = str(obj.get("condition_id") or obj.get("conditionId") or obj.get("market") or "").strip()
        if obj_condition and market.condition_id and obj_condition.lower() != market.condition_id.lower():
            return None

        tokens = self._parse_token_ids(obj)
        outcomes = self._parse_outcomes(obj)
        up_idx, down_idx = self._outcome_indices(outcomes)
        payouts = self._parse_payouts(obj)
        winner_token = self._winner_token(obj)
        winner_text = self._winner_text(obj)
        final_flag = self._final_flag(obj)

        up_payout = payouts[up_idx] if len(payouts) > up_idx else None
        down_payout = payouts[down_idx] if len(payouts) > down_idx else None

        outcome = Outcome.UNKNOWN
        if winner_token:
            if winner_token == market.up_token_id:
                outcome = Outcome.UP
            elif winner_token == market.down_token_id:
                outcome = Outcome.DOWN
        if outcome == Outcome.UNKNOWN and winner_text:
            text = winner_text.lower()
            if "up" in text or "yes" in text or "above" in text:
                outcome = Outcome.UP
            elif "down" in text or "no" in text or "below" in text:
                outcome = Outcome.DOWN
        if outcome == Outcome.UNKNOWN and up_payout is not None and down_payout is not None:
            if up_payout >= 0.999 and down_payout <= 0.001:
                outcome = Outcome.UP
            elif down_payout >= 0.999 and up_payout <= 0.001:
                outcome = Outcome.DOWN

        if outcome == Outcome.UNKNOWN or not (winner_token or winner_text or final_flag or payouts):
            return None
        if not winner_token:
            winner_token = market.up_token_id if outcome == Outcome.UP else market.down_token_id
        return MarketResolution(
            resolved=True,
            outcome=outcome,
            winner_token_id=winner_token,
            up_payout=up_payout,
            down_payout=down_payout,
            source=source,
            raw=obj,
        )

    def _parse_token_ids(self, obj: Dict[str, Any]) -> List[str]:
        raw = obj.get("clobTokenIds") or obj.get("clob_token_ids") or obj.get("asset_ids") or obj.get("assetIds")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                return [part.strip() for part in raw.split(",") if part.strip()]
        if isinstance(raw, list):
            return [str(item) for item in raw]
        tokens = obj.get("tokens") or []
        if isinstance(tokens, list):
            ids: List[str] = []
            for token in tokens:
                if isinstance(token, dict):
                    value = token.get("token_id") or token.get("tokenId") or token.get("asset_id") or token.get("assetId")
                    if value:
                        ids.append(str(value))
            return ids
        return []

    def _parse_outcomes(self, obj: Dict[str, Any]) -> List[str]:
        raw = obj.get("outcomes") or obj.get("outcome_names") or obj.get("outcomeNames")
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                return [part.strip() for part in raw.split(",") if part.strip()]
        if isinstance(raw, list):
            return [str(item) for item in raw]
        tokens = obj.get("tokens") or []
        if isinstance(tokens, list):
            return [str(token.get("outcome", "")) for token in tokens if isinstance(token, dict)]
        return []

    def _outcome_indices(self, outcomes: Sequence[str]) -> Tuple[int, int]:
        up_idx, down_idx = 0, 1
        for idx, name in enumerate(outcomes):
            text = str(name).lower()
            if "up" in text or "yes" in text or "above" in text:
                up_idx = idx
            elif "down" in text or "no" in text or "below" in text:
                down_idx = idx
        return up_idx, down_idx

    def _parse_payouts(self, obj: Dict[str, Any]) -> List[float]:
        raw = obj.get("outcomePrices") or obj.get("outcome_prices") or obj.get("prices") or obj.get("payouts")
        values: List[Any]
        if isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
                values = parsed if isinstance(parsed, list) else []
            except json.JSONDecodeError:
                values = [part.strip() for part in raw.split(",") if part.strip()]
        elif isinstance(raw, list):
            values = raw
        else:
            values = []
        return [_safe_float(value, 0.0) for value in values]

    def _winner_token(self, obj: Dict[str, Any]) -> str:
        for key in (
            "winningTokenId",
            "winning_token_id",
            "winningOutcomeTokenId",
            "winning_outcome_token_id",
            "winnerTokenId",
            "winning_asset_id",
            "winningAssetId",
        ):
            value = obj.get(key)
            if value not in (None, ""):
                return str(value)
        tokens = obj.get("tokens") or []
        if isinstance(tokens, list):
            for token in tokens:
                if not isinstance(token, dict):
                    continue
                if any(bool(token.get(flag)) for flag in ("winner", "winning", "isWinner", "is_winner")):
                    value = token.get("token_id") or token.get("tokenId") or token.get("asset_id") or token.get("assetId")
                    if value:
                        return str(value)
        return ""

    def _winner_text(self, obj: Dict[str, Any]) -> str:
        for key in ("winningOutcome", "winning_outcome", "winner", "resolvedOutcome", "resolution", "result"):
            value = obj.get(key)
            if value not in (None, ""):
                return str(value)
        return ""

    def _final_flag(self, obj: Dict[str, Any]) -> bool:
        for key in (
            "resolved",
            "isResolved",
            "settled",
            "readyForSettlement",
            "isFinalized",
            "finalized",
            "questionResolved",
        ):
            value = obj.get(key)
            if isinstance(value, bool) and value:
                return True
            if isinstance(value, str) and value.strip().lower() in {"true", "resolved", "settled", "finalized"}:
                return True
        return str(obj.get("status") or "").strip().lower() in {"resolved", "settled", "finalized", "final"}

    def _resolution_keys(self, payload: Dict[str, Any]) -> Tuple[str, ...]:
        keys: List[str] = []
        for key in (
            "condition_id",
            "conditionId",
            "market",
            "slug",
            "market_slug",
            "ticker",
            "asset_id",
            "assetId",
            "token_id",
            "tokenId",
            "winning_token_id",
            "winningTokenId",
            "winningOutcomeTokenId",
        ):
            value = payload.get(key)
            if value not in (None, ""):
                text = str(value).strip()
                keys.append(text.lower() if "token" not in key.lower() and "asset" not in key.lower() else text)
        return tuple(dict.fromkeys(item for item in keys if item))
