from __future__ import annotations

from typing import Any

from core.config import ContractConfig

try:
    from web3 import Web3
except ImportError:  # pragma: no cover
    Web3 = None


def build_web3(contracts: ContractConfig) -> Any | None:
    if Web3 is None:
        return None
    if contracts.polygon_ws_rpc:
        return Web3(Web3.WebsocketProvider(contracts.polygon_ws_rpc, websocket_timeout=8))
    if contracts.allow_http_rpc and contracts.polygon_http_rpc:
        return Web3(Web3.HTTPProvider(contracts.polygon_http_rpc, request_kwargs={"timeout": 8}))
    return None
