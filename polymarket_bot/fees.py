"""Polymarket fee helpers.

Docs: fee = shares * feeRate * price * (1 - price), rounded to 5 decimals.
"""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Optional

from core.constants import CRYPTO_TAKER_FEE_RATE
from shared.env import safe_float as finite_float

FEE_QUANT = Decimal("0.00001")


def default_taker_fee_rate_for_market(*, question: str = "", slug: str = "") -> Optional[float]:
    text = f"{question} {slug}".lower()
    crypto_markers = (
        "bitcoin",
        "btc",
        "ethereum",
        "eth",
        "solana",
        "sol",
        "xrp",
        "doge",
        "crypto",
    )
    if any(marker in text for marker in crypto_markers):
        return CRYPTO_TAKER_FEE_RATE
    return None


def normalize_taker_fee_rate(
    raw_rate: Any,
    *,
    default_rate: Optional[float] = None,
) -> float:
    """Return a decimal fee rate suitable for Polymarket's fee formula.

    Current CLOB market info exposes decimal ``fd.r`` values. Older/base-fee
    fields can be basis points; for known categories prefer the documented
    category rate over those base-fee fields.
    """
    if raw_rate is None or raw_rate == "":
        return max(0.0, float(default_rate or 0.0))

    rate = finite_float(raw_rate, 0.0)
    if rate == 1000.0 or rate == 1000:
        return max(0.0, float(default_rate or 0.0))

    if rate <= 0.0:
        return 0.0
    if 0.0 < rate <= 1.0:
        return rate
    if default_rate is not None and default_rate >= 0.0:
        return float(default_rate)

    bps_rate = rate / 10_000.0
    if 0.0 < bps_rate <= 1.0:
        return bps_rate
    return 0.0


def round_fee_usdc(value: float) -> float:
    fee = Decimal(str(max(0.0, finite_float(value, 0.0))))
    return float(fee.quantize(FEE_QUANT, rounding=ROUND_HALF_UP))


def estimate_taker_fee_usdc(*, shares: Any, price: Any, fee_rate: Any, fee_exponent: Any = 1.0) -> float:
    share_count = Decimal(str(max(0.0, finite_float(shares, 0.0))))
    p = Decimal(str(min(1.0, max(0.0, finite_float(price, 0.0)))))
    rate = Decimal(str(max(0.0, finite_float(fee_rate, 0.0))))
    exponent = float(finite_float(fee_exponent, 1.0))
    if exponent == 1.0:
        scaling = p * (Decimal("1") - p)
    else:
        p_factor = float(p * (Decimal("1") - p))
        scaling = Decimal(str(p_factor ** exponent))
    fee = share_count * rate * scaling
    return float(fee.quantize(FEE_QUANT, rounding=ROUND_HALF_UP))


def fee_rate_for_market_info(market_info: Any) -> float:
    question = str(getattr(market_info, "question", "") or "") if market_info else ""
    slug = str(getattr(market_info, "slug", "") or "") if market_info else ""
    default_rate = default_taker_fee_rate_for_market(question=question, slug=slug)
    raw_rate = getattr(market_info, "fee_rate", None) if market_info else None
    return normalize_taker_fee_rate(
        raw_rate,
        default_rate=default_rate,
    )


def sanitize_explicit_fee_usdc(
    explicit_fee: Any,
    *,
    expected_fee: float,
    notional: Any,
) -> Optional[float]:
    raw_fee = finite_float(explicit_fee, -1.0)
    if raw_fee < 0.0:
        return None
    if raw_fee == 0.0:
        return 0.0

    expected = max(0.0, finite_float(expected_fee, 0.0))
    trade_notional = max(0.0, finite_float(notional, 0.0))
    candidates = [raw_fee]
    if raw_fee > 1.0:
        candidates.append(raw_fee / 1_000_000.0)

    if expected > 0.0:
        lower = expected * 0.2
        upper = max(expected * 5.0, expected + 0.01)
        plausible = [
            candidate
            for candidate in candidates
            if lower <= candidate <= upper
        ]
        if plausible:
            return round_fee_usdc(min(plausible, key=lambda candidate: abs(candidate - expected)))
        return None

    if trade_notional > 0.0:
        plausible = [
            candidate
            for candidate in candidates
            if candidate <= max(trade_notional * 2.0, 0.01)
        ]
        if plausible:
            return round_fee_usdc(min(plausible))
    return None


def estimate_entry_fee_usdc(
    *,
    market_info: Any,
    entry_price: Any,
    shares: Any,
    explicit_fee: Any = None,
    notional: Any = None,
) -> float:
    fee_rate = fee_rate_for_market_info(market_info)
    fee_exponent = 1.0
    if market_info:
        fee_exp_val = getattr(market_info, "fee_exponent", 1.0)
        if fee_exp_val is not None:
            fee_exponent = fee_exp_val
    price = finite_float(entry_price, 0.0)
    size = finite_float(shares, 0.0)
    expected_fee = 0.0
    if fee_rate > 0.0 and 0.0 < price < 1.0 and size > 0.0:
        expected_fee = estimate_taker_fee_usdc(
            shares=size,
            price=price,
            fee_rate=fee_rate,
            fee_exponent=fee_exponent,
        )

    explicit = sanitize_explicit_fee_usdc(
        explicit_fee,
        expected_fee=expected_fee,
        notional=notional,
    )
    if explicit is not None:
        return explicit
    return expected_fee
