from __future__ import annotations

import asyncio
from dataclasses import dataclass

from core.config import ContractConfig, PolymarketConfig
from core.settlement.contracts import ERC20_ABI, ERC1155_ABI

try:
    from web3 import Web3
except ImportError:  # pragma: no cover
    Web3 = None


@dataclass(slots=True, frozen=True)
class WalletSnapshot:
    configured: bool
    address: str
    pusd_balance: int
    exchange_allowance: int
    neg_risk_exchange_allowance: int
    erc1155_exchange_approved: bool
    erc1155_neg_risk_approved: bool
    ready: bool


class WalletService:
    def __init__(self, poly: PolymarketConfig, contracts: ContractConfig) -> None:
        self._poly = poly
        self._contracts = contracts
        self._web3 = self._build_web3()
        self._erc20 = None
        self._erc1155 = None
        if self._web3 is not None:
            self._erc20 = self._web3.eth.contract(
                address=self._web3.to_checksum_address(contracts.pusd),
                abi=ERC20_ABI,
            )
            self._erc1155 = self._web3.eth.contract(
                address=self._web3.to_checksum_address(contracts.conditional_tokens),
                abi=ERC1155_ABI,
            )

    def _build_web3(self):
        if Web3 is None:
            return None
        if self._contracts.polygon_ws_rpc:
            return Web3(Web3.WebsocketProvider(self._contracts.polygon_ws_rpc, websocket_timeout=8))
        if self._contracts.allow_http_rpc and self._contracts.polygon_http_rpc:
            return Web3(Web3.HTTPProvider(self._contracts.polygon_http_rpc, request_kwargs={"timeout": 8}))
        return None

    @property
    def owner(self) -> str:
        if self._poly.funder:
            return self._poly.funder
        if self._poly.private_key and self._web3 is not None:
            return self._web3.eth.account.from_key(self._poly.private_key).address
        return ""

    async def snapshot(self) -> WalletSnapshot:
        if self._web3 is None or self._erc20 is None or self._erc1155 is None or not self.owner:
            return WalletSnapshot(False, self.owner, 0, 0, 0, False, False, False)
        return await asyncio.to_thread(self._snapshot_sync)

    def _snapshot_sync(self) -> WalletSnapshot:
        owner = self._web3.to_checksum_address(self.owner)
        exchange = self._web3.to_checksum_address(self._contracts.ctf_exchange)
        neg_exchange = self._web3.to_checksum_address(self._contracts.neg_risk_ctf_exchange)
        balance = int(self._erc20.functions.balanceOf(owner).call())
        allowance = int(self._erc20.functions.allowance(owner, exchange).call())
        neg_allowance = int(self._erc20.functions.allowance(owner, neg_exchange).call())
        approved = bool(self._erc1155.functions.isApprovedForAll(owner, exchange).call())
        neg_approved = bool(self._erc1155.functions.isApprovedForAll(owner, neg_exchange).call())
        ready = balance > 0 and allowance > 0 and neg_allowance > 0 and approved and neg_approved
        return WalletSnapshot(True, owner, balance, allowance, neg_allowance, approved, neg_approved, ready)
