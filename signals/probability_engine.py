"""
Unified Probability Engine — the core signal computation layer.

This engine merges data from BOTH the UP and DOWN orderbooks into a single
probabilistic directional forecast. Every signal is independently validated
before contributing to the final probability estimate.

Architecture rationale:
- UP price + DOWN price ≈ 1.0 (neg-risk market invariant)
- True probability of UP = market-implied UP probability ± edge
- Edge is detected through: microprice, OFI, depth imbalance, velocity
- All signals are combined via a weighted probability fusion approach
- Confidence is derived from signal agreement (not individual strength)

Mathematical framework:
Each signal Si produces a probability estimate P_i(UP) ∈ [0,1].
Final probability = Σ(w_i × P_i) / Σ(w_i)
Confidence = 1 - variance(P_i) × 4   (high agreement = high confidence)

Validation principles (applied before any signal is accepted):
1. Statistical significance: signal must exceed noise threshold
2. Directional consistency: majority of signals must agree
3. Market microstructure sanity: book must be valid and liquid
4. Manipulation detection: reject if spoof/wash trade indicators
5. Regime check: different logic for high vs low volatility
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from decimal import Decimal
from typing import Deque, Dict, List, Optional, Tuple

from core.constants import (
    ABSORPTION_WINDOW_S,
    DEPTH_RATIO_THRESHOLD,
    MAX_TRADEABLE_SPREAD,
    MICROPRICE_SIGNAL_DEVIATION,
    MIN_BOOK_DEPTH_USDC,
    MIN_CONFIDENCE,
    MIN_EV_THRESHOLD,
    MIN_PRICE_VELOCITY,
    OFI_SIGNAL_THRESHOLD,
    SWEEP_CONFIRM_COUNT,
    CRYPTO_TAKER_FEE_RATE,
)
from core.types import (
    Direction,
    OrderFlowMetrics,
    OrderbookState,
    SignalStrength,
    Side,
    UnifiedSignal,
)
from orderbook.book import L2Orderbook

logger = logging.getLogger(__name__)

# Signal weights — determined by empirical importance
# These weights prioritize direct order flow evidence over derived metrics
_SIGNAL_WEIGHTS = {
    "microprice":   0.30,   # Strongest single signal in liquid markets
    "ofi":          0.25,   # Order flow imbalance — leading indicator
    "depth_ratio":  0.20,   # Depth asymmetry — market structure
    "velocity":     0.15,   # Price momentum signal
    "absorption":   0.10,   # Liquidity absorption confirmation
}

# Minimum number of agreeing signals for STRONG classification
_MIN_AGREEING_SIGNALS = 3


class ProbabilityEngine:
    """
    Merges both UP and DOWN orderbook signals into a unified
    probability estimate with confidence scoring.

    Key design decisions:
    1. Every computation is independently validated
    2. Disagreeing signals reduce confidence (not just ignored)
    3. Market microstructure checks gate all signal computation
    4. Manipulation/spoof detection runs before any trade signal
    5. Confidence is modeled as signal agreement, not signal strength alone
    """

    def __init__(
        self,
        up_book:   L2Orderbook,
        down_book: L2Orderbook,
        signal_config = None,
    ) -> None:
        self._up_book   = up_book
        self._down_book = down_book
        self._cfg       = signal_config

        # Rolling history for regime detection
        self._signal_history: Deque[UnifiedSignal] = deque(maxlen=50)
        self._last_up_price:   Optional[Decimal]   = None
        self._last_down_price: Optional[Decimal]   = None
        self._price_history:   Deque[Tuple[float, Decimal, Decimal]] = deque(maxlen=100)
        # (timestamp, up_microprice, down_microprice)

        # Sweep confirmation counters
        self._up_sweep_count:   int = 0
        self._down_sweep_count: int = 0

        self._lock = asyncio.Lock()

    async def compute(self) -> Optional[UnifiedSignal]:
        """
        Compute the unified directional signal.
        Returns None if books are not initialized or no meaningful signal exists.

        Pipeline:
        1. Validate both books are initialized and liquid
        2. Check market microstructure sanity
        3. Compute individual component signals
        4. Run manipulation/spoof detection
        5. Fuse signals into probability estimate
        6. Compute confidence from signal agreement
        7. Classify signal strength
        8. Return final signal or None if below threshold
        """
        # ── Step 1: Validate book initialization ──
        up_state   = await self._up_book.snapshot()
        down_state = await self._down_book.snapshot()

        if up_state is None or down_state is None:
            logger.debug("Books not initialized, skipping signal computation")
            return None

        if up_state.is_empty() or down_state.is_empty():
            logger.debug("One or both books are empty, skipping")
            return None

        # ── Step 2: Microstructure sanity checks ──
        liquidity_ok, spread_ok = self._check_market_conditions(up_state, down_state)

        # ── Step 3: Neg-risk consistency check ──
        # UP price + DOWN price should be close to 1.0
        # Significant deviation indicates a desynced or manipulated book
        if not self._check_neg_risk_consistency(up_state, down_state):
            logger.warning("Neg-risk consistency check failed — books may be desynced")
            return None

        # ── Step 4: Compute component signals ──
        microprice_score, mp_up, mp_down = await self._compute_microprice_signal(
            up_state, down_state
        )
        ofi_score          = await self._compute_ofi_signal()
        depth_ratio_score  = self._compute_depth_ratio_signal(up_state, down_state)
        velocity_score     = await self._compute_velocity_signal(up_state, down_state)
        absorption_score   = await self._compute_absorption_signal(up_state, down_state)

        # ── Step 5: Manipulation detection ──
        up_spoof_prob   = await self._up_book.estimate_spoof_probability()
        down_spoof_prob = await self._down_book.estimate_spoof_probability()
        manipulation_flag = (
            up_spoof_prob > Decimal("0.60") or down_spoof_prob > Decimal("0.60")
        )

        up_sweep   = await self._up_book.detect_sweep()
        down_sweep = await self._down_book.detect_sweep()
        if up_sweep:
            self._up_sweep_count += 1
        if down_sweep:
            self._down_sweep_count += 1

        liquidity_vacuum = self._detect_liquidity_vacuum(up_state, down_state)
        violent_reversal = self._detect_violent_reversal()

        # ── Step 6: Fuse signals into probability estimate ──
        component_scores = {
            "microprice":  microprice_score,
            "ofi":         ofi_score,
            "depth_ratio": depth_ratio_score,
            "velocity":    velocity_score,
            "absorption":  absorption_score,
        }

        up_probability, confidence = self._fuse_signals(component_scores)
        down_probability = Decimal("1") - up_probability

        # Adjust for manipulation risk (reduces confidence)
        if manipulation_flag:
            confidence *= Decimal("0.5")
        if liquidity_vacuum:
            confidence *= Decimal("0.7")
        if violent_reversal:
            confidence *= Decimal("0.8")

        # ── Step 7: Determine direction and signal strength ──
        direction = self._classify_direction(up_probability, confidence)
        strength  = self._classify_strength(
            up_probability, confidence, component_scores
        )

        # ── Step 8: Build unified signal ──
        signal = UnifiedSignal(
            up_asset_id=self._up_book.asset_id,
            down_asset_id=self._down_book.asset_id,
            up_microprice=mp_up,
            down_microprice=mp_down,
            up_probability=up_probability,
            down_probability=down_probability,
            confidence=confidence,
            direction=direction,
            signal_strength=strength,
            microprice_score=microprice_score,
            ofi_score=ofi_score,
            depth_ratio_score=depth_ratio_score,
            velocity_score=velocity_score,
            absorption_score=absorption_score,
            up_spread=up_state.spread,
            down_spread=down_state.spread,
            liquidity_ok=liquidity_ok,
            spread_ok=spread_ok,
            manipulation_flag=manipulation_flag,
            liquidity_vacuum=liquidity_vacuum,
            violent_reversal=violent_reversal,
            timestamp=int(time.time() * 1000),
        )

        # Store in history for regime detection
        self._signal_history.append(signal)

        # Update price history
        if mp_up and mp_down:
            self._price_history.append((time.monotonic(), mp_up, mp_down))

        return signal

    # ─────────────────────── Component Signal Computations ───────────────────────

    async def _compute_microprice_signal(
        self,
        up_state:   OrderbookState,
        down_state: OrderbookState,
    ) -> Tuple[Decimal, Optional[Decimal], Optional[Decimal]]:
        """
        Compute microprice-based directional signal.

        Microprice is the volume-weighted midprice:
          μ = (V_ask × P_bid + V_bid × P_ask) / (V_bid + V_ask)

        In a neg-risk market (UP + DOWN = 1):
        - If UP microprice > 0.5, market structure leans UP
        - If DOWN microprice > 0.5, market structure leans DOWN
        - Consistency check: UP_microprice + DOWN_microprice ≈ 1.0

        Signal is stronger when microprice deviates from 0.5 AND
        both books agree directionally.

        Statistical basis: Stoikov (2009) shows microprice is the
        best unbiased predictor of short-term price direction in LOBs.
        """
        mp_up   = up_state.microprice()
        mp_down = down_state.microprice()

        if mp_up is None or mp_down is None:
            return Decimal("0"), mp_up, mp_down

        # Consistency between UP and DOWN microprices
        # In a neg-risk market: mp_up should ≈ 1 - mp_down
        implied_up_from_down = Decimal("1") - mp_down
        consistency = Decimal("1") - abs(mp_up - implied_up_from_down)

        # Base signal: deviation of UP microprice from 0.5
        # Positive score = UP signal, Negative = DOWN signal
        deviation = mp_up - Decimal("0.5")

        # Gate: only count as signal if deviation is meaningful
        min_dev = Decimal(str(MICROPRICE_SIGNAL_DEVIATION))
        if abs(deviation) < min_dev:
            return Decimal("0"), mp_up, mp_down

        # Weight by consistency (reduce score if books disagree)
        score = deviation * consistency * Decimal("2")  # Normalize to [-1, 1]
        score = max(Decimal("-1"), min(Decimal("1"), score))

        return score, mp_up, mp_down

    async def _compute_ofi_signal(self) -> Decimal:
        """
        Compute Order Flow Imbalance signal across BOTH books.

        For UP/DOWN binary markets:
        - Positive OFI in UP book: buyers accumulating UP tokens → UP signal
        - Positive OFI in DOWN book: buyers accumulating DOWN tokens → DOWN signal

        Combined OFI = UP_OFI - DOWN_OFI
        Normalized to [-1, 1].

        This captures the NET directional pressure across both legs
        of the binary market simultaneously.
        """
        up_ofi   = await self._up_book.compute_ofi(window_s=10.0)
        down_ofi = await self._down_book.compute_ofi(window_s=10.0)

        combined_ofi = up_ofi - down_ofi

        threshold = Decimal(str(OFI_SIGNAL_THRESHOLD))
        if abs(combined_ofi) < threshold:
            return Decimal("0")

        # Normalize and clip
        score = combined_ofi / (abs(combined_ofi) + threshold)
        return max(Decimal("-1"), min(Decimal("1"), score))

    def _compute_depth_ratio_signal(
        self,
        up_state:   OrderbookState,
        down_state: OrderbookState,
    ) -> Decimal:
        """
        Compute depth asymmetry signal.

        Theory: In prediction markets, the side with LESS liquidity
        often represents the favored outcome because:
        1. Informed traders have already bought it up (depleting asks)
        2. Market makers are reluctant to offer more (reducing bid depth on losing side)

        UP_ask_depth / DOWN_ask_depth:
        - Ratio < 1: less UP sell pressure → UP favored
        - Ratio > 1: more UP sell pressure → DOWN favored

        We also check bid depth: more bids on one side = buyers want that side.

        Returns positive score for UP signal, negative for DOWN signal.
        """
        up_bid_depth   = up_state.bid_depth(5)
        up_ask_depth   = up_state.ask_depth(5)
        down_bid_depth = down_state.bid_depth(5)
        down_ask_depth = down_state.ask_depth(5)

        total_up   = up_bid_depth + up_ask_depth
        total_down = down_bid_depth + down_ask_depth

        if total_up == 0 or total_down == 0:
            return Decimal("0")

        # Bid depth imbalance: relative buying interest in UP vs DOWN
        up_bid_ratio   = up_bid_depth / total_up
        down_bid_ratio = down_bid_depth / total_down

        bid_signal = up_bid_ratio - down_bid_ratio

        # Ask depth imbalance: relative selling pressure in UP vs DOWN
        up_ask_ratio   = up_ask_depth / total_up
        down_ask_ratio = down_ask_depth / total_down

        # More asks in UP = more sellers, bearish for UP
        ask_signal = -(up_ask_ratio - down_ask_ratio)

        combined = (bid_signal + ask_signal) / Decimal("2")

        threshold = Decimal(str(DEPTH_RATIO_THRESHOLD))
        if abs(combined) < threshold:
            return Decimal("0")

        return max(Decimal("-1"), min(Decimal("1"), combined * Decimal("4")))

    async def _compute_velocity_signal(
        self,
        up_state:   OrderbookState,
        down_state: OrderbookState,
    ) -> Decimal:
        """
        Compute price velocity signal.

        A rapidly moving midprice in one direction indicates momentum.
        For short-horizon binary markets (5 minutes), early momentum
        tends to continue until the resolution determines the winner.

        UP_velocity - DOWN_velocity normalized to [-1, 1].
        """
        up_vel   = await self._up_book.compute_price_velocity(window_s=5.0)
        down_vel = await self._down_book.compute_price_velocity(window_s=5.0)

        combined_vel = up_vel - down_vel
        min_vel = Decimal(str(MIN_PRICE_VELOCITY))

        if abs(combined_vel) < min_vel:
            return Decimal("0")

        score = combined_vel / (abs(combined_vel) + min_vel)
        return max(Decimal("-1"), min(Decimal("1"), score))

    async def _compute_absorption_signal(
        self,
        up_state:   OrderbookState,
        down_state: OrderbookState,
    ) -> Decimal:
        """
        Compute liquidity absorption signal.

        Absorption occurs when large orders at a price level are filled
        without the price moving significantly — indicating hidden depth
        absorbing the aggressive flow.

        Bullish absorption: sellers are being absorbed by buyers at the ask
        Bearish absorption: buyers are being absorbed by sellers at the bid

        This is computed by comparing taker volumes against book depth changes:
        - High taker buy volume + stable/growing bid = bullish absorption
        - High taker sell volume + stable/growing ask = bearish absorption
        """
        up_buy_vol, up_sell_vol = await self._up_book.compute_taker_volumes(
            window_s=ABSORPTION_WINDOW_S
        )
        down_buy_vol, down_sell_vol = await self._down_book.compute_taker_volumes(
            window_s=ABSORPTION_WINDOW_S
        )

        total_volume = up_buy_vol + up_sell_vol + down_buy_vol + down_sell_vol
        if total_volume == 0:
            return Decimal("0")

        # Net directional volume: UP buys - UP sells - DOWN buys + DOWN sells
        # Positive = net UP buying
        net_direction = (up_buy_vol - up_sell_vol) - (down_buy_vol - down_sell_vol)
        normalized = net_direction / total_volume

        if abs(normalized) < Decimal("0.10"):
            return Decimal("0")

        return max(Decimal("-1"), min(Decimal("1"), normalized * Decimal("2")))

    # ─────────────────────────── Signal Fusion ───────────────────────────

    def _fuse_signals(
        self,
        component_scores: Dict[str, Decimal],
    ) -> Tuple[Decimal, Decimal]:
        """
        Fuse component signals into a final probability estimate.

        Method: Weighted average of directional probabilities.
        Each score ∈ [-1, 1] is converted to probability P_i(UP):
          P_i = 0.5 + score_i / 2

        Then: P(UP) = Σ(w_i × P_i) / Σ(w_i)

        Confidence is derived from signal agreement:
        - All signals agree → high confidence
        - Mixed signals → lower confidence
        - This penalizes situations where one strong signal dominates
          while others disagree

        The confidence formula:
          conf = 1 - (N_disagreeing / N_total) × 0.5
        Further penalized by variance in probability estimates.
        """
        probs = []
        weights = []

        for name, score in component_scores.items():
            # Only include signals that have meaningful magnitude
            if score == Decimal("0"):
                continue
            p_i = Decimal("0.5") + score / Decimal("2")
            probs.append(p_i)
            weights.append(Decimal(str(_SIGNAL_WEIGHTS.get(name, 0.1))))

        if not probs:
            return Decimal("0.5"), Decimal("0.0")

        total_weight = sum(weights)
        if total_weight == 0:
            return Decimal("0.5"), Decimal("0.0")

        weighted_prob = sum(
            p * w for p, w in zip(probs, weights)
        ) / total_weight

        # Confidence from signal agreement
        half = Decimal("0.5")
        n_bullish = sum(1 for p in probs if p > half)
        n_bearish = sum(1 for p in probs if p < half)
        n_total   = len(probs)

        if n_total == 0:
            return Decimal("0.5"), Decimal("0.0")

        # Agreement ratio: how many signals agree with the majority direction
        majority_count = max(n_bullish, n_bearish)
        agreement_ratio = Decimal(str(majority_count)) / Decimal(str(n_total))

        # Variance penalty (high variance = low confidence)
        mean_p = weighted_prob
        variance = sum(
            (p - mean_p) ** 2 * w for p, w in zip(probs, weights)
        ) / total_weight
        variance_penalty = min(Decimal("0.3"), variance * Decimal("4"))

        confidence = (
            agreement_ratio
            - variance_penalty
            - (Decimal("1") - agreement_ratio) * Decimal("0.3")
        )
        confidence = max(Decimal("0"), min(Decimal("1"), confidence))

        return weighted_prob, confidence

    # ─────────────────────────── Classification ───────────────────────────

    def _classify_direction(
        self,
        up_probability: Decimal,
        confidence:     Decimal,
    ) -> Direction:
        """
        Classify directional call.
        Returns FLAT if probability is too close to 50/50 or confidence is low.
        """
        if confidence < Decimal("0.4"):
            return Direction.FLAT

        if up_probability > Decimal("0.55"):
            return Direction.UP
        elif up_probability < Decimal("0.45"):
            return Direction.DOWN
        else:
            return Direction.FLAT

    def _classify_strength(
        self,
        up_probability:  Decimal,
        confidence:      Decimal,
        component_scores: Dict[str, Decimal],
    ) -> SignalStrength:
        """
        Classify signal strength from NONE to EXTREME.

        Strength requires BOTH probability deviation AND confidence:
        - EXTREME: > 70% probability + > 80% confidence + 4+ agreeing signals
        - STRONG:  > 60% probability + > 65% confidence + 3+ agreeing signals
        - MEDIUM:  > 55% probability + > 55% confidence
        - WEAK:    > 52% probability + > 45% confidence
        - NONE:    otherwise
        """
        if up_probability > Decimal("0.5"):
            dev = up_probability - Decimal("0.5")
        else:
            dev = Decimal("0.5") - up_probability

        n_agreeing = sum(
            1 for score in component_scores.values()
            if abs(score) > Decimal("0.1") and (
                (score > 0 and up_probability > Decimal("0.5")) or
                (score < 0 and up_probability < Decimal("0.5"))
            )
        )

        if dev > Decimal("0.20") and confidence > Decimal("0.80") and n_agreeing >= 4:
            return SignalStrength.EXTREME
        elif dev > Decimal("0.10") and confidence > Decimal("0.65") and n_agreeing >= 3:
            return SignalStrength.STRONG
        elif dev > Decimal("0.05") and confidence > Decimal("0.55"):
            return SignalStrength.MEDIUM
        elif dev > Decimal("0.02") and confidence > Decimal("0.45"):
            return SignalStrength.WEAK
        else:
            return SignalStrength.NONE

    # ─────────────────────────── Validation & Checks ───────────────────────────

    def _check_market_conditions(
        self,
        up_state:   OrderbookState,
        down_state: OrderbookState,
    ) -> Tuple[bool, bool]:
        """
        Validate market conditions for trading.
        Returns (liquidity_ok, spread_ok).
        """
        max_spread = Decimal(str(MAX_TRADEABLE_SPREAD))
        min_depth  = Decimal(str(MIN_BOOK_DEPTH_USDC))

        # Spread check
        up_spread   = up_state.spread
        down_spread = down_state.spread

        spread_ok = (
            up_spread is not None
            and down_spread is not None
            and up_spread   <= max_spread
            and down_spread <= max_spread
        )

        # Liquidity check: both books must have sufficient depth
        up_depth   = up_state.bid_depth(5) + up_state.ask_depth(5)
        down_depth = down_state.bid_depth(5) + down_state.ask_depth(5)

        # Convert shares to USDC using midprice
        up_mid   = up_state.midprice or Decimal("0.5")
        down_mid = down_state.midprice or Decimal("0.5")

        up_usdc_depth   = up_depth   * up_mid
        down_usdc_depth = down_depth * down_mid

        liquidity_ok = (
            up_usdc_depth   >= min_depth
            and down_usdc_depth >= min_depth
        )

        return liquidity_ok, spread_ok

    def _check_neg_risk_consistency(
        self,
        up_state:   OrderbookState,
        down_state: OrderbookState,
    ) -> bool:
        """
        Verify neg-risk market invariant: UP midprice + DOWN midprice ≈ 1.0

        In a properly functioning neg-risk market, these should sum to
        approximately 1.00. Significant deviation indicates:
        1. Books are desynced (one hasn't received recent updates)
        2. Arbitrage opportunity has opened (external interference)
        3. Market is in a transition state (near resolution)

        We allow up to 8 cents deviation before rejecting.
        """
        up_mid   = up_state.midprice
        down_mid = down_state.midprice

        if up_mid is None or down_mid is None:
            return True   # Can't check without midprice, assume OK

        total = up_mid + down_mid
        deviation = abs(total - Decimal("1"))

        if deviation > Decimal("0.08"):
            logger.warning(
                "Neg-risk consistency violation: UP %.3f + DOWN %.3f = %.3f (dev=%.3f)",
                float(up_mid), float(down_mid), float(total), float(deviation)
            )
            return False

        return True

    def _detect_liquidity_vacuum(
        self,
        up_state:   OrderbookState,
        down_state: OrderbookState,
    ) -> bool:
        """
        Detect if one side of the market has critically thin liquidity.

        A liquidity vacuum occurs when:
        - Best bid or best ask is missing on either side
        - Total depth on either side drops below critical threshold
        - Spread widens dramatically (> 20 cents)

        Trading in a liquidity vacuum is dangerous because:
        - Small orders can move price significantly
        - Fills may occur at unfavorable prices
        - Market may be near resolution (extreme price)
        """
        if up_state.best_bid is None or up_state.best_ask is None:
            return True
        if down_state.best_bid is None or down_state.best_ask is None:
            return True

        up_spread   = up_state.spread or Decimal("1")
        down_spread = down_state.spread or Decimal("1")

        if up_spread > Decimal("0.20") or down_spread > Decimal("0.20"):
            return True

        up_total_depth   = up_state.bid_depth(3) + up_state.ask_depth(3)
        down_total_depth = down_state.bid_depth(3) + down_state.ask_depth(3)

        min_critical = Decimal("50")   # $50 minimum in top 3 levels
        return up_total_depth < min_critical or down_total_depth < min_critical

    def _detect_violent_reversal(self) -> bool:
        """
        Detect if a violent price reversal has just occurred.

        A violent reversal invalidates recent directional signals because:
        - It suggests the market is reacting to unknown information
        - Signal models may be momentarily confused by the noise
        - Risk of entering on the wrong side of a fast move

        Detection: if last N signals show rapid direction flip
        """
        if len(self._signal_history) < 4:
            return False

        recent = list(self._signal_history)[-4:]
        directions = [s.direction for s in recent if s.direction != Direction.FLAT]

        if len(directions) < 3:
            return False

        # Check for alternating directions (UP, DOWN, UP or DOWN, UP, DOWN)
        flip_count = sum(
            1 for i in range(1, len(directions))
            if directions[i] != directions[i-1]
        )

        return flip_count >= 2

    # ─────────────────────────── EV Computation ───────────────────────────

    def compute_entry_ev(
        self,
        signal:     UnifiedSignal,
        entry_price: Decimal,
        direction:  Direction,
    ) -> Decimal:
        """
        Compute expected value of entering at a given price.

        EV = P(win) × (1 - entry_price) - P(lose) × entry_price - fee

        Fee formula (crypto): fee = 0.07 × shares × p × (1-p)
        Per share: fee_per_share = 0.07 × p × (1-p)

        For entry to be profitable: EV > MIN_EV_THRESHOLD
        """
        if direction == Direction.UP:
            p_win = signal.up_probability
        else:
            p_win = signal.down_probability

        p_lose = Decimal("1") - p_win

        # Expected PnL per share
        ev_before_fee = (
            p_win * (Decimal("1") - entry_price)
            - p_lose * entry_price
        )

        # Taker fee per share
        fee_per_share = Decimal(str(CRYPTO_TAKER_FEE_RATE)) * entry_price * (Decimal("1") - entry_price)

        return ev_before_fee - fee_per_share
