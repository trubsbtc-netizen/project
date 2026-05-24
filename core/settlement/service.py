from __future__ import annotations

import asyncio
import contextlib
import time
from dataclasses import dataclass

from core.config import ContractConfig, SettlementConfig
from core.settlement.contracts import CONDITIONAL_TOKENS_ABI
from core.transport.web3_provider import build_web3
from core.types import SettlementTruth


@dataclass(slots=True, frozen=True)
class SettlementReadiness:
    rpc_configured: bool
    connected: bool
    chain_id: int


class SettlementService:
    def __init__(self, contracts: ContractConfig, config: SettlementConfig) -> None:
        self._contracts = contracts
        self._config = config
        self._web3 = build_web3(contracts)
        self._ctf = None
        if self._web3 is not None:
            self._ctf = self._web3.eth.contract(
                address=self._web3.to_checksum_address(contracts.conditional_tokens),
                abi=CONDITIONAL_TOKENS_ABI,
            )

    async def readiness(self) -> SettlementReadiness:
        if self._web3 is None:
            return SettlementReadiness(False, False, 0)
        return await asyncio.to_thread(self._readiness_sync)

    def _readiness_sync(self) -> SettlementReadiness:
        try:
            connected = bool(self._web3.is_connected())
            chain_id = int(self._web3.eth.chain_id) if connected else 0
            return SettlementReadiness(True, connected, chain_id)
        except Exception:
            return SettlementReadiness(True, False, 0)

    async def verify_condition(self, condition_id: str) -> SettlementTruth | None:
        if self._ctf is None:
            return None
        return await asyncio.to_thread(self._verify_condition_sync, condition_id)

    def _verify_condition_sync(self, condition_id: str) -> SettlementTruth | None:
        cid = self._bytes32(condition_id)
        denominator = int(self._ctf.functions.payoutDenominator(cid).call())
        numerators = (
            int(self._ctf.functions.payoutNumerators(cid, 0).call()),
            int(self._ctf.functions.payoutNumerators(cid, 1).call()),
        )
        return SettlementTruth(
            condition_id=condition_id,
            payout_numerators=numerators,
            payout_denominator=denominator,
            verified_block=int(self._web3.eth.block_number),
        )

    async def wait_for_resolution(self, condition_id: str) -> SettlementTruth | None:
        deadline = time.monotonic() + self._config.poll_timeout_s
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                truth = await self.verify_condition(condition_id)
                if truth is not None and truth.resolved:
                    return truth
            await asyncio.sleep(self._config.poll_interval_s)
        return None

    async def redeem_positions(self, condition_id: str, private_key: str, collateral: str) -> str | None:
        if self._web3 is None or self._ctf is None or not private_key:
            return None
        return await asyncio.to_thread(self._redeem_positions_sync, condition_id, private_key, collateral)

    def _redeem_positions_sync(self, condition_id: str, private_key: str, collateral: str) -> str:
        account = self._web3.eth.account.from_key(private_key)
        nonce = self._web3.eth.get_transaction_count(account.address)
        tx = self._ctf.functions.redeemPositions(
            self._web3.to_checksum_address(collateral),
            b"\x00" * 32,
            self._bytes32(condition_id),
            [1, 2],
        ).build_transaction(
            {
                "from": account.address,
                "nonce": nonce,
                "chainId": int(self._web3.eth.chain_id),
                "gas": 350_000,
                "maxFeePerGas": self._web3.eth.gas_price * 2,
                "maxPriorityFeePerGas": self._web3.to_wei(35, "gwei"),
            }
        )
        signed = account.sign_transaction(tx)
        tx_hash = self._web3.eth.send_raw_transaction(signed.rawTransaction)
        return self._web3.to_hex(tx_hash)

    def _bytes32(self, value: str) -> bytes:
        clean = value[2:] if value.startswith("0x") else value
        raw = bytes.fromhex(clean)
        if len(raw) != 32:
            raise ValueError("condition id must be bytes32")
        return raw
