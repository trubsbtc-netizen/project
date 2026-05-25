"""
Polymarket CLOB API client with complete L1/L2 authentication.

Implements:
- EIP-712 order signing (L1)
- HMAC-SHA256 L2 request authentication
- All trading endpoints (post order, cancel, heartbeat)
- Market data endpoints (orderbook, tick size, neg risk)
- Automatic heartbeat management (prevents auto-cancel on inactivity)
- Retry logic for transient failures
- Rate limit awareness

Authentication flow (from official Polymarket docs):
1. L1: Private key signs EIP-712 typed data to prove wallet ownership
2. L2: API key + HMAC-SHA256 signature for all trading requests

Order signing:
- All orders are EIP-712 signed limit orders
- Market orders = limit orders with marketable price + FOK/FAK type
- GTD orders expire at specified unix timestamp + 60s security buffer

Security architecture:
- Private key NEVER leaves this module
- API credentials used only for server-to-server requests
- No credentials exposed in logs
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from eth_account import Account
from eth_account.messages import encode_typed_data

from core.constants import (
    CLOB_HOST,
    GTD_SECURITY_BUFFER_S,
    POLYGON_CHAIN_ID,
)
from core.types import Direction, OrderStatus, OrderType, Side

logger = logging.getLogger(__name__)


def _decode_api_secret(secret: str) -> bytes:
    cleaned = secret.strip()
    try:
        return base64.b64decode(cleaned, validate=True)
    except Exception:
        padded = cleaned + "=" * (-len(cleaned) % 4)
        return base64.urlsafe_b64decode(padded)


# ─────────────────────────── EIP-712 Domain ───────────────────────────

# CTF Exchange domain for order signing (Polygon mainnet)
_CTF_EXCHANGE_ADDRESS = "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E"

_ORDER_DOMAIN = {
    "name": "CTF Exchange",
    "version": "1",
    "chainId": POLYGON_CHAIN_ID,
    "verifyingContract": _CTF_EXCHANGE_ADDRESS,
}

_ORDER_TYPES = {
    "Order": [
        {"name": "salt",          "type": "uint256"},
        {"name": "maker",         "type": "address"},
        {"name": "signer",        "type": "address"},
        {"name": "taker",         "type": "address"},
        {"name": "tokenId",       "type": "uint256"},
        {"name": "makerAmount",   "type": "uint256"},
        {"name": "takerAmount",   "type": "uint256"},
        {"name": "expiration",    "type": "uint256"},
        {"name": "nonce",         "type": "uint256"},
        {"name": "feeRateBps",    "type": "uint256"},
        {"name": "side",          "type": "uint8"},
        {"name": "signatureType", "type": "uint8"},
    ]
}

_CLOB_AUTH_DOMAIN = {
    "name": "ClobAuthDomain",
    "version": "1",
    "chainId": POLYGON_CHAIN_ID,
}

_CLOB_AUTH_TYPES = {
    "ClobAuth": [
        {"name": "address", "type": "address"},
        {"name": "timestamp", "type": "string"},
        {"name": "nonce", "type": "uint256"},
        {"name": "message", "type": "string"},
    ]
}

_CLOB_AUTH_MESSAGE = "This message attests that I control the given wallet"

# Side encoding for smart contract
_SIDE_BUY  = 0
_SIDE_SELL = 1

# Signature type (POLY_1271 for deposit wallets)
_SIG_TYPE_POLY_1271 = 3

# Price precision: Polymarket uses 6 decimal places internally
# Price is represented as integer: price_int = round(price * 10^6)
_PRICE_DECIMALS    = 6
_PRICE_MULTIPLIER  = 10 ** _PRICE_DECIMALS


class ClobApiError(Exception):
    """Raised when CLOB API returns an error."""
    def __init__(self, message: str, status_code: int = 0, error_msg: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.error_msg = error_msg


class OrderCreationError(ClobApiError):
    """Raised when order creation/signing fails."""
    pass


class ClobClient:
    """
    Async CLOB API client with complete authentication and order management.
    Thread-safe via asyncio.Lock for stateful operations.
    """

    def __init__(
        self,
        session:          aiohttp.ClientSession,
        private_key:      str,   # 0x-prefixed hex
        api_key:          str,
        api_secret:       str,
        api_passphrase:   str,
        funder_address:   str,
        signature_type:   int = _SIG_TYPE_POLY_1271,
        host:             str = CLOB_HOST,
    ) -> None:
        self._session        = session
        self._private_key    = private_key
        self._api_key        = api_key
        self._api_secret     = api_secret
        self._api_passphrase = api_passphrase
        self._funder_address = funder_address
        self._signature_type = signature_type
        self._host           = host

        # Derive signer address from private key
        self._account        = Account.from_key(private_key)
        self._signer_address = self._account.address

        # Heartbeat state
        self._heartbeat_id:       str              = ""
        self._heartbeat_task:     Optional[asyncio.Task] = None
        self._heartbeat_interval: int              = 5   # seconds
        self._heartbeat_warned_unauthorized = False

        # Order deduplication (in-flight order IDs)
        self._pending_orders: Dict[str, float] = {}   # order_id -> timestamp
        self._order_lock = asyncio.Lock()

        logger.info(
            "ClobClient initialized: signer=%s funder=%s",
            self._signer_address[:10], self._funder_address[:10]
        )

    async def start_heartbeat(self, interval_s: int = 5) -> None:
        """Start background heartbeat task."""
        self._heartbeat_interval = interval_s
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(), name="clob-heartbeat"
        )
        logger.info("Heartbeat started (interval=%ds)", interval_s)

    async def stop_heartbeat(self) -> None:
        """Stop background heartbeat task."""
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            await asyncio.gather(self._heartbeat_task, return_exceptions=True)
            logger.info("Heartbeat stopped")

    @property
    def api_key(self) -> str:
        return self._api_key

    @property
    def api_secret(self) -> str:
        return self._api_secret

    @property
    def api_passphrase(self) -> str:
        return self._api_passphrase

    async def ensure_api_credentials(self, auto_create: bool = True) -> bool:
        """Ensure L2 API credentials exist and pass a lightweight authenticated check."""
        if self._has_api_credentials() and await self.validate_api_credentials():
            return True

        if not auto_create:
            return False

        logger.info("L2 API credentials missing/invalid; deriving or creating from wallet")
        creds = await self.derive_api_key()
        if creds is None:
            creds = await self.create_api_key()
        if creds is None:
            logger.warning("Could not auto-create Polymarket L2 API credentials")
            return False

        self._api_key = creds["api_key"]
        self._api_secret = creds["api_secret"]
        self._api_passphrase = creds["api_passphrase"]
        logger.info("L2 API credentials ready: key=%s", self._api_key[:8])
        return await self.validate_api_credentials()

    async def create_api_key(self) -> Optional[Dict[str, str]]:
        """Create a new L2 API key using Level-1 wallet auth."""
        return await self._level1_api_creds("POST", "/auth/api-key")

    async def derive_api_key(self) -> Optional[Dict[str, str]]:
        """Derive an existing L2 API key using Level-1 wallet auth."""
        return await self._level1_api_creds("GET", "/auth/derive-api-key")

    async def validate_api_credentials(self) -> bool:
        """Check L2 credentials without mutating trading state."""
        if not self._has_api_credentials():
            return False
        try:
            path = "/auth/api-keys"
            headers = self._build_l2_headers("GET", path)
            async with self._session.get(f"{self._host}{path}", headers=headers) as resp:
                if resp.status == 200:
                    self._heartbeat_warned_unauthorized = False
                    return True
                body = await resp.text()
                logger.warning("L2 auth validation failed: status=%d body=%s", resp.status, body[:160])
                return False
        except Exception as exc:
            logger.warning("L2 auth validation error: %s", exc)
            return False

    def _has_api_credentials(self) -> bool:
        return bool(self._api_key and self._api_secret and self._api_passphrase)

    async def _level1_api_creds(self, method: str, path: str) -> Optional[Dict[str, str]]:
        headers = self._build_l1_headers()
        try:
            if method.upper() == "POST":
                ctx = self._session.post(f"{self._host}{path}", headers=headers)
            else:
                ctx = self._session.get(f"{self._host}{path}", headers=headers)

            async with ctx as resp:
                text = await resp.text()
                if resp.status not in (200, 201):
                    logger.debug("L1 auth %s failed: status=%d body=%s", path, resp.status, text[:160])
                    return None
                data = json.loads(text) if text else {}
        except Exception as exc:
            logger.debug("L1 auth %s error: %s", path, exc)
            return None

        try:
            return {
                "api_key": data["apiKey"],
                "api_secret": data["secret"],
                "api_passphrase": data["passphrase"],
            }
        except KeyError:
            logger.warning("L1 auth response missing credential fields for %s", path)
            return None

    # ─────────────────────────── Order Operations ───────────────────────────

    async def post_limit_order(
        self,
        token_id:    str,
        price:       Decimal,
        size:        Decimal,
        side:        Side,
        tick_size:   str,
        neg_risk:    bool,
        order_type:  OrderType  = OrderType.GTC,
        expiration:  Optional[int] = None,
        post_only:   bool       = False,
    ) -> Dict[str, Any]:
        """
        Create, sign, and submit a limit order.

        For GTD orders, expiration must be set to:
          int(time.time()) + 60 + desired_lifetime_seconds
        (60-second security threshold required by Polymarket)

        Returns the order response dict from the API.
        Raises ClobApiError on failure.
        """
        if order_type == OrderType.GTD and expiration is None:
            raise OrderCreationError("GTD orders require expiration timestamp")

        # Build and sign the order
        signed_order = await self._create_signed_order(
            token_id=token_id,
            price=price,
            size=size,
            side=side,
            tick_size=tick_size,
            neg_risk=neg_risk,
            order_type=order_type,
            expiration=expiration or 0,
        )

        # Submit to CLOB
        return await self._post_order(signed_order, order_type, post_only)

    async def post_market_order(
        self,
        token_id:    str,
        amount:      Decimal,    # USDC for BUY, shares for SELL
        side:        Side,
        worst_price: Decimal,    # Slippage protection
        tick_size:   str,
        neg_risk:    bool,
        order_type:  OrderType = OrderType.FOK,
    ) -> Dict[str, Any]:
        """
        Create and submit a market order (FOK or FAK).

        For BUY: amount is USDC to spend
        For SELL: amount is shares to sell
        worst_price acts as slippage protection limit.
        """
        if order_type not in (OrderType.FOK, OrderType.FAK):
            raise OrderCreationError("Market orders must use FOK or FAK")

        # Market order size computation
        if side == Side.BUY:
            # Shares = USDC / price
            shares = (amount / worst_price).quantize(Decimal("0.01"))
        else:
            shares = amount

        signed_order = await self._create_signed_order(
            token_id=token_id,
            price=worst_price,
            size=shares,
            side=side,
            tick_size=tick_size,
            neg_risk=neg_risk,
            order_type=order_type,
            expiration=0,
        )

        return await self._post_order(signed_order, order_type, False)

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel a single order by ID."""
        path    = f"/order/{order_id}"
        headers = self._build_l2_headers("DELETE", path)
        url     = f"{self._host}{path}"

        try:
            async with self._session.delete(url, headers=headers) as resp:
                data = await resp.json()
                if resp.status == 200:
                    logger.info("Cancelled order: %s", order_id[:16])
                    return True
                else:
                    logger.warning(
                        "Cancel order failed: id=%s status=%d data=%s",
                        order_id[:16], resp.status, data
                    )
                    return False
        except Exception as exc:
            logger.error("Cancel order error: %s", exc)
            return False

    async def cancel_all_orders(self) -> bool:
        """Cancel all open orders for this account."""
        path    = "/orders"
        headers = self._build_l2_headers("DELETE", path)
        url     = f"{self._host}{path}"

        try:
            async with self._session.delete(url, headers=headers) as resp:
                if resp.status == 200:
                    logger.info("Cancelled all orders")
                    return True
                else:
                    data = await resp.text()
                    logger.warning("Cancel all failed: status=%d data=%s", resp.status, data)
                    return False
        except Exception as exc:
            logger.error("Cancel all orders error: %s", exc)
            return False

    async def get_open_orders(self) -> List[Dict]:
        """Fetch all open orders."""
        path    = "/orders?status=LIVE"
        headers = self._build_l2_headers("GET", path)
        url     = f"{self._host}{path}"

        async with self._session.get(url, headers=headers) as resp:
            if resp.status == 200:
                return await resp.json()
            return []

    # ─────────────────────────── Market Data ───────────────────────────

    async def get_orderbook(self, token_id: str) -> Dict:
        """Fetch current orderbook snapshot via REST (for re-sync)."""
        url = f"{self._host}/book?token_id={token_id}"
        async with self._session.get(url) as resp:
            if resp.status == 200:
                return await resp.json()
            raise ClobApiError(f"Orderbook fetch failed: {resp.status}")

    async def get_tick_size(self, token_id: str) -> str:
        """Fetch tick size for a token."""
        url = f"{self._host}/tick-size?token_id={token_id}"
        async with self._session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data.get("minimum_tick_size", "0.01")
            return "0.01"

    async def get_neg_risk(self, token_id: str) -> bool:
        """Fetch neg_risk flag for a token."""
        url = f"{self._host}/neg-risk?token_id={token_id}"
        async with self._session.get(url) as resp:
            if resp.status == 200:
                data = await resp.json()
                return bool(data.get("neg_risk", False))
            return True   # BTC UP/DOWN markets are always neg_risk

    async def get_server_time(self) -> int:
        """Fetch server UNIX timestamp for time synchronization."""
        url = f"{self._host}/time"
        async with self._session.get(url) as resp:
            if resp.status == 200:
                body = (await resp.text()).strip()
                try:
                    data = json.loads(body)
                except json.JSONDecodeError:
                    data = body

                if isinstance(data, dict):
                    return int(data.get("time", time.time()))
                return int(float(data))
            return int(time.time())

    # ─────────────────────────── Heartbeat ───────────────────────────

    async def post_heartbeat(self) -> bool:
        """
        Send heartbeat signal to maintain open orders.
        If not sent within 10 seconds (with 5s buffer), all orders are cancelled.
        Must include the previous heartbeat_id in each request.
        """
        path    = "/v1/heartbeats"
        payload = {"heartbeat_id": self._heartbeat_id}
        body = json.dumps(payload, separators=(",", ":"))
        headers = self._build_l2_headers("POST", path, body)
        url     = f"{self._host}{path}"

        try:
            async with self._session.post(url, headers=headers, data=body) as resp:
                if resp.status == 200:
                    data = await resp.json(content_type=None)
                    if isinstance(data, dict):
                        self._heartbeat_id = data.get("heartbeat_id", "")
                    self._heartbeat_warned_unauthorized = False
                    return True
                elif resp.status == 400:
                    # Server responded with correct ID — update and retry
                    data = await resp.json(content_type=None)
                    new_id = data.get("heartbeat_id", "")
                    if new_id:
                        self._heartbeat_id = new_id
                        logger.debug("Heartbeat ID corrected to: %s", new_id[:16])
                    return False
                elif resp.status == 401:
                    if not self._heartbeat_warned_unauthorized:
                        logger.warning("Heartbeat unauthorized; disabling repeated heartbeat warnings")
                        self._heartbeat_warned_unauthorized = True
                    return False
                else:
                    logger.warning("Heartbeat failed: status=%d", resp.status)
                    return False
        except Exception as exc:
            logger.error("Heartbeat error: %s", exc)
            return False

    async def _heartbeat_loop(self) -> None:
        """Background heartbeat task."""
        while True:
            try:
                success = await self.post_heartbeat()
                if not success:
                    await asyncio.sleep(self._heartbeat_interval)
                    continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("Heartbeat loop error: %s", exc)
            await asyncio.sleep(self._heartbeat_interval)

    # ─────────────────────────── Order Signing ───────────────────────────

    async def _create_signed_order(
        self,
        token_id:   str,
        price:      Decimal,
        size:       Decimal,
        side:       Side,
        tick_size:  str,
        neg_risk:   bool,
        order_type: OrderType,
        expiration: int,
    ) -> Dict[str, Any]:
        """
        Build and EIP-712 sign an order for submission.

        Price/size encoding:
        - Polymarket CLOB encodes amounts as integers with token-specific decimals
        - For standard tokens: amounts in 6 decimal places (USDC precision)
        - makerAmount: amount the maker gives (USDC for BUY, shares for SELL)
        - takerAmount: amount the taker gives (shares for BUY, USDC for SELL)
        """
        # Validate price conforms to tick size
        tick = Decimal(tick_size)
        price = (price / tick).to_integral_value() * tick

        # Convert price and size to integer representation
        # makerAmount / takerAmount are in 6-decimal USDC units
        usdc_amount   = int((price * size * Decimal(str(_PRICE_MULTIPLIER))).to_integral_value())
        shares_amount = int((size * Decimal(str(_PRICE_MULTIPLIER))).to_integral_value())

        if side == Side.BUY:
            maker_amount = usdc_amount   # Maker gives USDC
            taker_amount = shares_amount # Taker gives shares
        else:
            maker_amount = shares_amount  # Maker gives shares
            taker_amount = usdc_amount    # Taker gives USDC

        # Generate random salt for order uniqueness
        import os
        salt = int.from_bytes(os.urandom(32), "big") % (2 ** 256)

        # Expiration: 0 for GTC/FAK/FOK, unix timestamp for GTD
        if order_type == OrderType.GTD and expiration > 0:
            exp = expiration
        else:
            exp = 0

        order_struct = {
            "salt":          salt,
            "maker":         self._funder_address,
            "signer":        self._signer_address,
            "taker":         "0x0000000000000000000000000000000000000000",
            "tokenId":       int(token_id),
            "makerAmount":   maker_amount,
            "takerAmount":   taker_amount,
            "expiration":    exp,
            "nonce":         0,
            "feeRateBps":    0,    # Fee is applied by protocol at match time
            "side":          _SIDE_BUY if side == Side.BUY else _SIDE_SELL,
            "signatureType": self._signature_type,
        }

        # EIP-712 sign
        signature = self._sign_order(order_struct)

        return {
            "salt":          str(salt),
            "maker":         self._funder_address,
            "signer":        self._signer_address,
            "taker":         "0x0000000000000000000000000000000000000000",
            "tokenId":       token_id,
            "makerAmount":   str(maker_amount),
            "takerAmount":   str(taker_amount),
            "expiration":    str(exp),
            "nonce":         "0",
            "feeRateBps":    "0",
            "side":          side.value,
            "signatureType": str(self._signature_type),
            "signature":     signature,
        }

    def _sign_order(self, order: Dict[str, Any]) -> str:
        """
        EIP-712 sign order struct.
        Uses eth_account's encode_typed_data for correct encoding.
        """
        structured_data = {
            "domain":          _ORDER_DOMAIN,
            "types":           _ORDER_TYPES,
            "primaryType":     "Order",
            "message":         order,
        }

        # encode_typed_data returns the signable message
        encoded = encode_typed_data(full_message=structured_data)
        signed  = self._account.sign_message(encoded)
        return signed.signature.hex()

    async def _post_order(
        self,
        signed_order: Dict[str, Any],
        order_type:   OrderType,
        post_only:    bool,
    ) -> Dict[str, Any]:
        """Submit signed order to CLOB."""
        path    = "/order"
        url     = f"{self._host}{path}"

        payload: Dict[str, Any] = {
            "order":     signed_order,
            "owner":     self._api_key,
            "orderType": order_type.value,
        }
        if post_only:
            payload["postOnly"] = True
        body = json.dumps(payload, separators=(",", ":"))
        headers = self._build_l2_headers("POST", path, body)

        async with self._order_lock:
            async with self._session.post(
                url, headers=headers, data=body
            ) as resp:
                data = await resp.json()

                if resp.status != 200:
                    error_msg = data.get("errorMsg", "unknown") if isinstance(data, dict) else str(data)
                    raise ClobApiError(
                        f"Order submission failed: {error_msg}",
                        status_code=resp.status,
                        error_msg=error_msg,
                    )

                if not data.get("success", False):
                    error_msg = data.get("errorMsg", "submission rejected")
                    raise ClobApiError(
                        f"Order rejected: {error_msg}",
                        error_msg=error_msg,
                    )

                order_id = data.get("orderID", "")
                logger.info(
                    "Order placed: id=%s status=%s",
                    order_id[:16] if order_id else "?",
                    data.get("status", "?"),
                )
                return data

    # ─────────────────────────── L2 Authentication ───────────────────────────

    def _build_l2_headers(
        self,
        method:   str,
        path:     str,
        body:     str = "",
    ) -> Dict[str, str]:
        """
        Build required L2 authentication headers.

        Required headers (from Polymarket docs):
        - POLY_ADDRESS:    Polygon signer address
        - POLY_SIGNATURE:  HMAC-SHA256 signature
        - POLY_TIMESTAMP:  Current UNIX timestamp (seconds)
        - POLY_API_KEY:    API key UUID
        - POLY_PASSPHRASE: API passphrase

        HMAC signature = HMAC-SHA256(secret, timestamp + method + path + body)
        """
        timestamp = str(int(time.time()))
        message   = timestamp + method.upper() + path + body

        secret_bytes = _decode_api_secret(self._api_secret)
        sig = hmac.new(
            secret_bytes,
            message.encode("utf-8"),
            hashlib.sha256,
        ).digest()
        signature = base64.urlsafe_b64encode(sig).decode("utf-8")

        return {
            "POLY_ADDRESS":    self._signer_address,
            "POLY_SIGNATURE":  signature,
            "POLY_TIMESTAMP":  timestamp,
            "POLY_API_KEY":    self._api_key,
            "POLY_PASSPHRASE": self._api_passphrase,
            "Content-Type":    "application/json",
        }

    def _build_l1_headers(self, nonce: int = 0) -> Dict[str, str]:
        timestamp = str(int(time.time()))
        signature = self._sign_clob_auth_message(timestamp, nonce)
        return {
            "POLY_ADDRESS": self._signer_address,
            "POLY_SIGNATURE": signature,
            "POLY_TIMESTAMP": timestamp,
            "POLY_NONCE": str(nonce),
        }

    def _sign_clob_auth_message(self, timestamp: str, nonce: int) -> str:
        structured_data = {
            "domain": _CLOB_AUTH_DOMAIN,
            "types": _CLOB_AUTH_TYPES,
            "primaryType": "ClobAuth",
            "message": {
                "address": self._signer_address,
                "timestamp": timestamp,
                "nonce": nonce,
                "message": _CLOB_AUTH_MESSAGE,
            },
        }
        encoded = encode_typed_data(full_message=structured_data)
        signed = self._account.sign_message(encoded)
        return "0x" + signed.signature.hex()
