from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional

from polymarket_bot.infrastructure.config import InfrastructureConfig
from polymarket_bot.infrastructure.types import (
    ExecutionReadiness,
    WalletSnapshot,
    _safe_float,
)

logger = logging.getLogger(__name__)

try:
    from py_clob_client_v2.client import ClobClient
    from py_clob_client_v2.clob_types import BalanceAllowanceParams, AssetType, OpenOrderParams
    from py_clob_client_v2 import SignatureTypeV2
except ImportError:
    ClobClient = None
    BalanceAllowanceParams = None
    AssetType = None
    OpenOrderParams = None
    SignatureTypeV2 = None

try:
    from py_builder_relayer_client.client import RelayClient
    from py_builder_relayer_client.models import DepositWalletCall, RelayerTxType, Transaction, TransactionType
    from py_builder_signing_sdk.config import BuilderApiKeyCreds, BuilderConfig
except ImportError:
    RelayClient = None
    DepositWalletCall = None
    RelayerTxType = None
    Transaction = None
    TransactionType = None
    BuilderApiKeyCreds = None
    BuilderConfig = None


class WalletMaintenance:
    def __init__(self, config: InfrastructureConfig):
        self.config = config
        self._client: Optional[Any] = None
        self._relayer: Optional[Any] = None
        self._initialized = False
        self.collateral_snapshot: Optional[WalletSnapshot] = None
        self.conditional_snapshots: Dict[str, WalletSnapshot] = {}
        self.last_error = ""

    @property
    def is_live(self) -> bool:
        return self._initialized and self._client is not None

    async def initialize(self) -> None:
        if self.config.dry_run or not self.config.private_key or ClobClient is None:
            self._initialized = False
            return
        try:
            funder = self.config.funder_address or None
            signature_type = self.config.signature_type
            if self._has_builder_creds() and signature_type == 3 and not funder:
                deposit_wallet = await self._setup_deposit_wallet()
                if deposit_wallet:
                    funder = deposit_wallet
                    self.config = replace(self.config, deposit_wallet_address=deposit_wallet)
            if self._has_builder_creds() and self._relayer is None:
                self._relayer = self._new_relayer(self._relay_tx_type_for_signature(signature_type))
            client = ClobClient(
                self.config.polymarket_host,
                chain_id=self.config.chain_id,
                key=self.config.private_key,
                signature_type=signature_type,
                funder=funder,
                use_server_time=True,
                retry_on_error=True,
            )
            creds = client.create_or_derive_api_key()
            client.set_api_creds(creds)
            self._client = client
            self._initialized = True
        except Exception as exc:
            self.last_error = repr(exc)
            self._initialized = False

    async def close(self) -> None:
        self._client = None
        self._relayer = None
        self._initialized = False

    async def refresh_collateral_balance_allowance(self, *, update_cache: bool = False) -> Optional[WalletSnapshot]:
        if not self.is_live or BalanceAllowanceParams is None or AssetType is None:
            return None

        def call() -> WalletSnapshot:
            kwargs: Dict[str, Any] = {"asset_type": AssetType.COLLATERAL}
            if self.config.signature_type == 3 and SignatureTypeV2 is not None:
                kwargs["signature_type"] = SignatureTypeV2.POLY_1271
            params = BalanceAllowanceParams(**kwargs)
            if update_cache and hasattr(self._client, "update_balance_allowance"):
                self._client.update_balance_allowance(params=params)
            raw = self._client.get_balance_allowance(params=params)
            raw_dict = raw if isinstance(raw, dict) else {"raw": raw}
            return WalletSnapshot(
                asset_type="COLLATERAL",
                balance=self._micro_usdc_to_float(raw_dict.get("balance")),
                allowance=self._micro_usdc_to_float(raw_dict.get("allowance")),
                synced_at=time.time(),
                raw=raw_dict,
            )

        try:
            snapshot = await asyncio.to_thread(call)
            self.collateral_snapshot = snapshot
            return snapshot
        except Exception as exc:
            self.last_error = repr(exc)
            return None

    async def refresh_conditional_balance_allowance(
        self,
        token_id: str,
        *,
        update_cache: bool = False,
    ) -> Optional[WalletSnapshot]:
        if not self.is_live or not token_id or BalanceAllowanceParams is None or AssetType is None:
            return None

        def call() -> WalletSnapshot:
            kwargs: Dict[str, Any] = {"asset_type": AssetType.CONDITIONAL, "token_id": str(token_id)}
            if self.config.signature_type == 3 and SignatureTypeV2 is not None:
                kwargs["signature_type"] = SignatureTypeV2.POLY_1271
            params = BalanceAllowanceParams(**kwargs)
            if update_cache and hasattr(self._client, "update_balance_allowance"):
                self._client.update_balance_allowance(params=params)
            raw = self._client.get_balance_allowance(params=params)
            raw_dict = raw if isinstance(raw, dict) else {"raw": raw}
            return WalletSnapshot(
                asset_type="CONDITIONAL",
                balance=self._micro_usdc_to_float(raw_dict.get("balance")),
                allowance=self._micro_usdc_to_float(raw_dict.get("allowance")),
                token_id=str(token_id),
                synced_at=time.time(),
                raw=raw_dict,
            )

        try:
            snapshot = await asyncio.to_thread(call)
            self.conditional_snapshots[str(token_id)] = snapshot
            return snapshot
        except Exception as exc:
            self.last_error = repr(exc)
            return None

    async def execution_readiness(self, token_ids: Iterable[str] = ()) -> ExecutionReadiness:
        blockers: List[str] = []
        if not self.is_live:
            blockers.append("wallet-client-unavailable")
        collateral = await self.refresh_collateral_balance_allowance(update_cache=False)
        conditionals: List[WalletSnapshot] = []
        for token_id in token_ids:
            snapshot = await self.refresh_conditional_balance_allowance(str(token_id), update_cache=False)
            if snapshot:
                conditionals.append(snapshot)
        approvals_ready = bool(collateral and collateral.balance > 0 and collateral.allowance > 0)
        if self.is_live and not approvals_ready:
            blockers.append("collateral-not-ready")
        return ExecutionReadiness(
            live=self.is_live,
            collateral=collateral,
            conditionals=tuple(conditionals),
            approvals_ready=approvals_ready,
            blockers=tuple(blockers),
        )

    async def ensure_deposit_wallet_approvals(self) -> bool:
        if not self._relayer or not self.config.deposit_wallet_address:
            return False
        if DepositWalletCall is None or TransactionType is None:
            return False

        def submit() -> Any:
            nonce_payload = self._relayer.get_nonce(self._relayer.signer.address(), TransactionType.WALLET.value)
            nonce = str(nonce_payload["nonce"])
            spenders = [
                "0xE111180000d2663C0091e4f400237545B87B996B",
                self.config.ctf_collateral_adapter_address,
                self.config.neg_risk_ctf_collateral_adapter_address,
            ]
            spenders = [spender for spender in dict.fromkeys(spenders) if spender]
            approve_all = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
            approved_true = "0000000000000000000000000000000000000000000000000000000000000001"
            calls: List[Any] = []
            for spender in spenders:
                spender_hex = spender.lower().replace("0x", "").rjust(64, "0")
                calls.append(DepositWalletCall(target=self.config.pusd_address, value="0", data=f"0x095ea7b3{spender_hex}{approve_all}"))
                calls.append(DepositWalletCall(target=self.config.ctf_address, value="0", data=f"0xa22cb465{spender_hex}{approved_true}"))
            response = self._relayer.execute_deposit_wallet_batch(
                calls=calls,
                wallet_address=self.config.deposit_wallet_address,
                nonce=nonce,
                deadline=str(int(time.time()) + 240),
            )
            return response.wait()

        try:
            await asyncio.to_thread(submit)
            return True
        except Exception as exc:
            self.last_error = repr(exc)
            return False

    async def cancel_all_orders(self) -> bool:
        if not self.is_live:
            return True
        try:
            await asyncio.to_thread(self._client.cancel_all)
            return True
        except Exception as exc:
            self.last_error = repr(exc)
            return False

    async def get_open_orders(self) -> List[Dict[str, Any]]:
        if not self.is_live or OpenOrderParams is None:
            return []
        try:
            result = await asyncio.to_thread(lambda: self._client.get_orders(OpenOrderParams()) or [])
            return result if isinstance(result, list) else []
        except Exception as exc:
            self.last_error = repr(exc)
            return []

    async def redeem_standard_market(self, condition_id: str) -> Dict[str, Any]:
        if not condition_id:
            return {"success": False, "status": "SKIPPED", "reason": "missing-condition-id"}
        data = self._encode_standard_redeem_calldata(condition_id)
        return await self._submit_wallet_transaction(
            target=self.config.ctf_collateral_adapter_address,
            data=data,
            metadata=f"redeem:{condition_id}",
        )

    def _has_builder_creds(self) -> bool:
        return bool(
            self.config.builder_api_key
            and self.config.builder_secret
            and self.config.builder_passphrase
            and RelayClient is not None
            and BuilderConfig is not None
        )

    def _new_relayer(self, relay_tx_type: Any = None) -> Optional[Any]:
        if not self._has_builder_creds():
            return None
        creds = BuilderApiKeyCreds(
            key=self.config.builder_api_key,
            secret=self.config.builder_secret,
            passphrase=self.config.builder_passphrase,
        )
        builder_config = BuilderConfig(local_builder_creds=creds)
        kwargs = {"relay_tx_type": relay_tx_type} if relay_tx_type is not None else {}
        return RelayClient(self.config.relayer_url, self.config.chain_id, self.config.private_key, builder_config, **kwargs)

    def _relay_tx_type_for_signature(self, signature_type: int) -> Any:
        if RelayerTxType is None:
            return None
        if signature_type == 2:
            return RelayerTxType.SAFE
        return RelayerTxType.PROXY

    async def _setup_deposit_wallet(self) -> Optional[str]:
        relayer = self._new_relayer(self._relay_tx_type_for_signature(3))
        if relayer is None:
            return None
        self._relayer = relayer

        def setup() -> Optional[str]:
            wallet = str(relayer.get_expected_deposit_wallet())
            try:
                response = relayer.deploy_deposit_wallet()
                response.wait()
            except Exception as exc:
                logger.debug("Deposit wallet deployment failed (may already exist): %s", exc)
            return wallet

        try:
            return await asyncio.to_thread(setup)
        except Exception as exc:
            self.last_error = repr(exc)
            return None

    async def _submit_wallet_transaction(self, *, target: str, data: str, metadata: str) -> Dict[str, Any]:
        if not self.is_live or self.config.dry_run:
            return {"success": True, "status": "DRY_RUN", "target": target, "metadata": metadata}
        if not target or not data:
            return {"success": False, "status": "SKIPPED", "reason": "missing-transaction-data"}
        if self._relayer is None:
            return {"success": False, "status": "SKIPPED", "reason": "relayer-unavailable"}

        def submit() -> Any:
            if self.config.signature_type == 3 and self.config.deposit_wallet_address and DepositWalletCall is not None:
                nonce_payload = self._relayer.get_nonce(self._relayer.signer.address(), TransactionType.WALLET.value)
                response = self._relayer.execute_deposit_wallet_batch(
                    calls=[DepositWalletCall(target=target, value="0", data=data)],
                    wallet_address=self.config.deposit_wallet_address,
                    nonce=str(nonce_payload["nonce"]),
                    deadline=str(int(time.time()) + 300),
                )
                return response.wait()
            if Transaction is not None:
                response = self._relayer.execute([Transaction(to=target, data=data, value="0")], metadata=metadata)
                return response.wait()
            raise RuntimeError("relayer transaction model unavailable")

        try:
            response = await asyncio.to_thread(submit)
            return {"success": True, "status": "SUBMITTED", "target": target, "metadata": metadata, "response": response}
        except Exception as exc:
            self.last_error = repr(exc)
            return {"success": False, "status": "ERROR", "reason": str(exc)}

    def _encode_standard_redeem_calldata(self, condition_id: str) -> str:
        try:
            from eth_abi import encode
            from eth_utils import keccak, to_checksum_address
        except ImportError as exc:
            raise RuntimeError("eth_abi and eth_utils are required for redemption encoding") from exc
        selector = keccak(text="redeemPositions(address,bytes32,bytes32,uint256[])")[:4]
        parent_collection_id = b"\x00" * 32
        condition_bytes = bytes.fromhex(condition_id.replace("0x", "").replace("0X", "").rjust(64, "0")[-64:])
        body = encode(
            ["address", "bytes32", "bytes32", "uint256[]"],
            [to_checksum_address(self.config.pusd_address), parent_collection_id, condition_bytes, [1, 2]],
        )
        return "0x" + (selector + body).hex()

    @staticmethod
    def _micro_usdc_to_float(value: Any) -> float:
        try:
            if value is None or value == "":
                return 0.0
            return int(str(value)) / 1_000_000.0
        except (TypeError, ValueError):
            return _safe_float(value, 0.0)
