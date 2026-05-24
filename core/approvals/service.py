from __future__ import annotations

import asyncio
from dataclasses import dataclass

from core.config import ContractConfig, PolymarketConfig
from core.settlement.contracts import ERC20_ABI, ERC1155_ABI

try:
    from web3 import Web3
except ImportError:  # pragma: no cover
    Web3 = None


MAX_UINT256 = (1 << 256) - 1


@dataclass(slots=True, frozen=True)
class ApprovalResult:
    submitted: bool
    tx_hashes: tuple[str, ...]
    message: str


class ApprovalService:
    def __init__(self, poly: PolymarketConfig, contracts: ContractConfig) -> None:
        self._poly = poly
        self._contracts = contracts
        self._web3 = self._build_web3()
        self._erc20 = None
        self._erc1155 = None
        if self._web3 is not None:
            self._erc20 = self._web3.eth.contract(self._web3.to_checksum_address(contracts.pusd), abi=ERC20_ABI)
            self._erc1155 = self._web3.eth.contract(
                self._web3.to_checksum_address(contracts.conditional_tokens),
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

    async def ensure_approvals(self) -> ApprovalResult:
        if self._web3 is None or self._erc20 is None or self._erc1155 is None:
            return ApprovalResult(False, (), "polygon RPC or web3 unavailable")
        if not self._poly.private_key:
            return ApprovalResult(False, (), "private key unavailable")
        return await asyncio.to_thread(self._ensure_approvals_sync)

    def _ensure_approvals_sync(self) -> ApprovalResult:
        account = self._web3.eth.account.from_key(self._poly.private_key)
        owner = self._web3.to_checksum_address(account.address)
        spenders = [
            self._web3.to_checksum_address(self._contracts.ctf_exchange),
            self._web3.to_checksum_address(self._contracts.neg_risk_ctf_exchange),
        ]
        tx_hashes: list[str] = []
        nonce = self._web3.eth.get_transaction_count(owner)
        for spender in spenders:
            allowance = int(self._erc20.functions.allowance(owner, spender).call())
            if allowance == 0:
                tx_hashes.append(self._send_tx(self._erc20.functions.approve(spender, MAX_UINT256), owner, account, nonce))
                nonce += 1
            approved = bool(self._erc1155.functions.isApprovedForAll(owner, spender).call())
            if not approved:
                tx_hashes.append(self._send_tx(self._erc1155.functions.setApprovalForAll(spender, True), owner, account, nonce))
                nonce += 1
        if not tx_hashes:
            return ApprovalResult(False, (), "approvals already ready")
        return ApprovalResult(True, tuple(tx_hashes), "approval maintenance submitted")

    def _send_tx(self, fn, owner: str, account, nonce: int) -> str:
        tx = fn.build_transaction(
            {
                "from": owner,
                "nonce": nonce,
                "chainId": int(self._web3.eth.chain_id),
                "gas": 180_000,
                "maxFeePerGas": self._web3.eth.gas_price * 2,
                "maxPriorityFeePerGas": self._web3.to_wei(35, "gwei"),
            }
        )
        signed = account.sign_transaction(tx)
        tx_hash = self._web3.eth.send_raw_transaction(signed.rawTransaction)
        return self._web3.to_hex(tx_hash)
