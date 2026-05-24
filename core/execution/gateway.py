from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any

from core.config import ExecutionConfig, PolymarketConfig
from core.runtime.clock import mono_ns
from core.types import OrderAck, OrderIntent


logger = logging.getLogger(__name__)


class ExecutionGateway(ABC):
    @abstractmethod
    async def submit(self, intent: OrderIntent) -> OrderAck:
        raise NotImplementedError


class DryRunGateway(ExecutionGateway):
    async def submit(self, intent: OrderIntent) -> OrderAck:
        logger.info(
            "dry-run order: %s %s size=%.4f limit=%.4f market=%s",
            intent.side.value,
            intent.outcome.value,
            intent.size,
            intent.limit_price,
            intent.market,
        )
        return OrderAck(
            accepted=True,
            order_id=f"dry-{intent.bucket}-{intent.outcome.value}-{mono_ns()}",
            status="dry_run",
            message="order simulated",
            recv_mono_ns=mono_ns(),
        )


class PyClobExecutionGateway(ExecutionGateway):
    def __init__(self, poly: PolymarketConfig, execution: ExecutionConfig) -> None:
        self._poly = poly
        self._execution = execution
        self._client: Any | None = None
        self._lock = asyncio.Lock()

    async def submit(self, intent: OrderIntent) -> OrderAck:
        if self._execution.dry_run or not self._execution.enable_live_trading:
            logger.info("live execution disabled; simulating order")
            return await DryRunGateway().submit(intent)
        return await asyncio.to_thread(self._submit_sync, intent)

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from py_clob_client.client import ClobClient
        except ImportError as exc:
            raise RuntimeError("py-clob-client is required for live execution") from exc
        client = ClobClient(
            host=self._poly.clob_host,
            key=self._poly.private_key,
            chain_id=self._poly.chain_id,
            signature_type=self._poly.signature_type,
            funder=self._poly.funder or None,
        )
        creds = None
        if hasattr(client, "create_or_derive_api_creds"):
            creds = client.create_or_derive_api_creds()
        elif hasattr(client, "derive_api_key"):
            creds = client.derive_api_key()
        if creds is not None and hasattr(client, "set_api_creds"):
            client.set_api_creds(creds)
        self._client = client
        return client

    def _submit_sync(self, intent: OrderIntent) -> OrderAck:
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
        except ImportError as exc:
            raise RuntimeError("py-clob-client clob_types unavailable") from exc
        client = self._ensure_client()
        args = OrderArgs(
            token_id=intent.token_id,
            price=round(intent.limit_price, 4),
            size=round(intent.size, 4),
            side=intent.side.value,
        )
        signed_order = client.create_order(args)
        order_type = getattr(OrderType, "FOK", getattr(OrderType, "GTC", None))
        response = client.post_order(signed_order, order_type) if order_type is not None else client.post_order(signed_order)
        order_id = ""
        status = "submitted"
        message = "accepted"
        if isinstance(response, dict):
            order_id = str(response.get("orderID") or response.get("order_id") or response.get("id") or "")
            status = str(response.get("status") or response.get("success") or status)
            message = str(response.get("message") or response.get("errorMsg") or message)
            accepted = not bool(response.get("error"))
        else:
            order_id = str(getattr(response, "order_id", "") or getattr(response, "id", ""))
            status = str(getattr(response, "status", status))
            accepted = True
        return OrderAck(accepted=accepted, order_id=order_id, status=status, message=message, recv_mono_ns=mono_ns())


class ExecutionEngine:
    def __init__(self, gateway: ExecutionGateway) -> None:
        self._gateway = gateway
        self._submitted: dict[int, OrderAck] = {}

    async def submit_once(self, intent: OrderIntent) -> OrderAck:
        existing = self._submitted.get(intent.bucket)
        if existing is not None:
            return existing
        ack = await self._gateway.submit(intent)
        if ack.accepted:
            self._submitted[intent.bucket] = ack
        return ack
