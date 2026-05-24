"""
Decision engine: the final decision gate that produces TradeDecision.

.

This is the central orchestrator of the entire bot. It takes all inference
 signal, execution, and risk, and and decorrelation outputs and produces
 the final TradeDecision that which either:
1. Executes a trade on Polymarket
2. Suppresses the trade (no edge, bad execution, risk limits)
3. Delays a trade (waiting for better conditions)

4. Modifies position size (scaling confidence)

The decision engine is deliberately conservative. It applies multiple gates
 before allowing execution, and each gate can independently veto a trade.
 This ensures that only high-quality, well-decorrelated, properly-sized
 trades with sufficient edge are executed.

Gate hierarchy (in order):
1. Settlement forecast gate: does the forecast recommend execution?
2. Execution quality gate: is execution quality acceptable?
3. Risk management gate: does risk state allow trading?
4. Decorrelation gate: is this signal decorrelated from recent trades?
5. Confidence gate: is scaled confidence above minimum threshold?
6. Final size gate: is position size within all constraints?

All gates must to pass for a trade to be executed.
"""

import time
from decimal import Decimal, ROUND_HALF_UP
import numpy as np
from typing import Optional

from polymarket_bot.bot_types import (
    TradeDecision,
    SettlementForecast,
    ExecutionEstimate,
    RiskState,
    RegimeStateEstimate,
    VolatilityEstimate,
    MultiTimescaleEstimate,
    BayesianPosterior,
    KalmanState,
    PolymarketOrderbook,
    OrderbookPressure,
    FlowMetrics,
    ContinuationSignal,
    ReversalSignal,
    ExhaustionSignal,
    BurstFailureSignal,
    EntropyFilterResult,
    Direction,
    RegimeState,
    SignalType,
    ExecutionQuality,
)
from polymarket_bot.config import BotConfig
from polymarket_bot.execution.execution_engine import ExecutionEngine
from polymarket_bot.decision.confidence_scaler import ConfidenceScaler
from polymarket_bot.decision.decorrelation import DecorrelationFilter


class DecisionEngine:
    """
    Final decision gate producing TradeDecision.
    
    Gate hierarchy:
    1. Settlement forecast gate
    2. Execution quality gate
    3. Risk management gate
    4. Decorrelation gate
    5. Confidence gate
    6. Final size gate
    
    Each gate can independently veto a trade. Only when ALL gates
    pass does a trade get executed.
    
    The engine also determines the dominant signal type and
    produces a human-readable reason for the decision.
    """

    # Signal type classification thresholds
    REVERSAL_DOMINANCE_THRESHOLD = 0.6  # If reversal_prob > this, continuation is secondary
    CONTINUATION_DOMINANCE_THRESHOLD = 0.6
    EXHAUSTION_THRESHOLD = 0.5
    BURST_FAILURE_THRESHOLD = 0.5

    # Minimum confidence for directional execution. The config can raise/lower it.
    MIN_EXECUTION_CONFIDENCE = 0.40

    def __init__(self, config: BotConfig):
        self.config = config
        self.execution_engine = ExecutionEngine(config)
        self.confidence_scaler = ConfidenceScaler(config)
        self.decorrelation_filter = DecorrelationFilter(config)
        self._decision_count = 0
        self._suppressed_count = 0
        self._entered_market_directions: dict[str, Direction] = {}
        self._active_market_slug = ""
        self._direction_confirmation_state: dict[str, dict[str, float | Direction | str]] = {}
        self._entry_price_wait_state: dict[str, dict[str, float | Direction | str]] = {}

    def record_filled_entry(self, market_slug: str, direction: Direction) -> None:
        """Record a round entry only after an order is actually filled/opened."""
        if not market_slug or direction not in {Direction.UP, Direction.DOWN}:
            return
        self._entered_market_directions[market_slug] = direction
        if len(self._entered_market_directions) > 256:
            recent_items = list(self._entered_market_directions.items())[-128:]
            self._entered_market_directions = dict(recent_items)
        self._direction_confirmation_state.pop(market_slug, None)
        self._entry_price_wait_state.pop(market_slug, None)

    @staticmethod
    def _round_to_tick(price: float, tick_size: float) -> float:
        tick = Decimal(str(tick_size or 0.01))
        rounded = (Decimal(str(price)) / tick).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
        return float(rounded * tick)

    def _classify_signal_type(
        self,
        settlement_forecast: Optional[SettlementForecast] = None,
        continuation_signal: Optional[ContinuationSignal] = None,
        reversal_signal: Optional[ReversalSignal] = None,
        exhaustion_signal: Optional[ExhaustionSignal] = None,
        burst_failure_signal: Optional[BurstFailureSignal] = None,
        entropy_filter: Optional[EntropyFilterResult] = None,
    ) -> SignalType:
        """
        Classify the dominant signal type driving this decision.
        
        Priority order (most important first):
        1. SUPPRESSED: entropy filter suppresses all signals
        2. BURST_FAILURE: failed breakout detected
        3. EXHAUSTION: momentum exhaustion detected
        4. REVERSAL: reversal precursors detected
        5. CONTINUATION: momentum continuation detected
        6. ABSORPTION: default when no strong signal
        """
        # Check entropy suppression first
        if entropy_filter is not None:
            if entropy_filter.suppression_factor < 0.3:
                return SignalType.SUPPRESSED

        # Check burst failure
        if burst_failure_signal is not None:
            if (
                burst_failure_signal.is_valid
                and burst_failure_signal.burst_failure_probability > self.BURST_FAILURE_THRESHOLD
            ):
                return SignalType.BURST_FAILURE

        # Check exhaustion
        if exhaustion_signal is not None:
            if (
                exhaustion_signal.is_valid
                and exhaustion_signal.exhaustion_probability > self.EXHAUSTION_THRESHOLD
            ):
                return SignalType.EXHAUSTION

        # Check reversal vs continuation
        reversal_prob = 0.0
        if reversal_signal is not None and reversal_signal.is_valid:
            reversal_prob = reversal_signal.reversal_probability

        continuation_prob = 0.0
        if continuation_signal is not None and continuation_signal.is_valid:
            continuation_prob = continuation_signal.continuation_probability

        if reversal_prob > self.REVERSAL_DOMINANCE_THRESHOLD:
            return SignalType.REVERSAL
        elif continuation_prob > self.CONTINUATION_DOMINANCE_THRESHOLD:
            return SignalType.CONTINUATION

        if settlement_forecast is not None and settlement_forecast.observation_valid:
            return SignalType.OBSERVATION

        # Default: no dominant signal
        return SignalType.ABSORPTION

    def _build_reason(
        self,
        settlement_forecast: SettlementForecast,
        execution_estimate: ExecutionEstimate,
        risk_state: Optional[RiskState],
        signal_type: SignalType,
        confidence: float,
        should_trade: bool,
        decorrelation_passed: bool,
        decorrelation_reason: str,
    ) -> str:
        """
        Build a human-readable reason string for the decision.
        
        Format: "GATE_RESULT| REASON_DETAIL"
        Example: "EXECUTE reversal_detected p=0.72 conf=0.85 edge=250bps"
        """
        parts = []

        # Directional forecast summary
        parts.append("DIRECTIONAL_EXEC" if settlement_forecast.execution_recommended else "DIRECTIONAL_REJECT")
        parts.append(f"p_up={settlement_forecast.p_up_settlement:.3f}")
        parts.append(f"p_dn={settlement_forecast.p_down_settlement:.3f}")
        parts.append(f"dir={settlement_forecast.expected_settlement_direction.value}")
        parts.append(f"dir_edge={settlement_forecast.edge_estimate*10000.0:.0f}bps")
        selected_probability = settlement_forecast.selected_probability
        if selected_probability is not None:
            parts.append(f"sel_p={selected_probability:.3f}")
        if settlement_forecast.reject_reason:
            parts.append(settlement_forecast.reject_reason)
        parts.append(f"tau={settlement_forecast.time_to_settlement_seconds:.0f}s")
        parts.append(f"z={settlement_forecast.terminal_z_score:.2f}")

        # Signal type
        parts.append(f"{signal_type.value}")

        # Confidence
        parts.append(f"conf={confidence:.2f}")

        # Execution quality
        quality = execution_estimate.execution_quality
        parts.append(f"exec={quality.value}")

        # Risk state
        if risk_state is not None:
            if risk_state.is_in_cooldown:
                parts.append("RISK_COOLDOWN")
            elif risk_state.should_stop_trading:
                parts.append("RISK_STOP")
            else:
                parts.append("RISK_OK")

        observation_wait = decorrelation_reason.startswith("observation_wait")
        observation_invalid = decorrelation_reason.startswith("observation_invalid")
        confirmation_wait = decorrelation_reason.startswith("post_observation_hold")
        # Decorrelation
        lock_wait = decorrelation_reason.startswith("round_direction_lock_pending")
        entry_pullback_wait = decorrelation_reason.startswith("entry_price_wait_for_pullback")
        decorrelation_was_skipped = decorrelation_reason in {
            "no_settlement_recommendation",
            "execution_rejected",
            "risk_stop_trading",
            "risk_cooldown",
        } or decorrelation_reason.startswith("round_entry_already_taken")
        if observation_wait:
            parts.append(f"OBS_WAIT({decorrelation_reason})")
        elif observation_invalid:
            parts.append(f"OBS_INVALID({decorrelation_reason})")
        elif confirmation_wait:
            parts.append(f"CONFIRM_WAIT({decorrelation_reason})")
        elif lock_wait:
            parts.append(f"LOCK_WAIT({decorrelation_reason})")
        elif entry_pullback_wait:
            parts.append(f"ENTRY_WAIT({decorrelation_reason})")
        elif decorrelation_reason.startswith("round_entry_already_taken"):
            parts.append(f"ROUND_SKIP({decorrelation_reason})")
        elif decorrelation_was_skipped:
            parts.append(f"DECOR_SKIP({decorrelation_reason})")
        elif decorrelation_passed:
            parts.append(f"DECOR_OK({decorrelation_reason})")
        else:
            parts.append(f"DECOR_REJECT({decorrelation_reason})")

        # Final decision
        if should_trade:
            parts.append("TRADE")
        else:
            parts.append("NO_TRADE")

        return " | ".join(parts)

    def _fallback_regime(self, timestamp: float) -> RegimeStateEstimate:
        return RegimeStateEstimate(
            timestamp=timestamp,
            current_regime=RegimeState.CALM_TRENDING,
            regime_probabilities={RegimeState.CALM_TRENDING: 1.0},
            regime_persistence=1.0,
            regime_transition_probability={RegimeState.CALM_TRENDING: 1.0},
            regime_duration_estimate=0.0,
            regime_volatility_estimate=0.0003,
            regime_flow_characteristic=0.0,
            regime_confidence=0.5,
            regime_entropy=0.0,
        )

    def _fallback_volatility(self, timestamp: float) -> VolatilityEstimate:
        default_vol = 0.00006
        return VolatilityEstimate(
            timestamp=timestamp,
            realized_volatility=default_vol,
            garch_volatility=default_vol,
            stochastic_volatility=default_vol,
            implied_volatility=default_vol,
            kalman_volatility=default_vol,
            regime_adjusted_volatility=default_vol,
            volatility_of_volatility=0.0,
            volatility_skew=0.0,
            volatility_forecast_5min=default_vol,
            volatility_confidence_interval=(default_vol / 3.0, default_vol * 3.0),
            volatility_regime_alignment=0.5,
        )

    def _decision_values(
        self,
        settlement_forecast: SettlementForecast,
        market_price_up: Optional[float],
        market_price_down: Optional[float] = None,
    ) -> tuple:
        p_up = float(np.clip(settlement_forecast.p_up_settlement, 0.001, 0.999))
        p_down = float(np.clip(settlement_forecast.p_down_settlement, 0.001, 0.999))
        if settlement_forecast.round_direction_locked:
            locked_direction = settlement_forecast.round_direction
            if locked_direction == Direction.UP:
                return Direction.UP, p_up, market_price_up or 0.5, p_up - 0.5, (p_up - 0.5) * 10000.0
            if locked_direction == Direction.DOWN:
                return Direction.DOWN, p_down, market_price_down or 0.5, p_down - 0.5, (p_down - 0.5) * 10000.0
        direction = settlement_forecast.expected_settlement_direction
        if direction == Direction.UP:
            return Direction.UP, p_up, market_price_up or 0.5, p_up - 0.5, (p_up - 0.5) * 10000.0
        if direction == Direction.DOWN:
            return Direction.DOWN, p_down, market_price_down or 0.5, p_down - 0.5, (p_down - 0.5) * 10000.0
        direction = Direction.UP if p_up >= p_down else Direction.DOWN
        probability = p_up if direction == Direction.UP else p_down
        market_price = market_price_up if direction == Direction.UP else market_price_down
        return direction, probability, market_price or 0.5, probability - 0.5, (probability - 0.5) * 10000.0

    def _forecast_confidence_for_reject(
        self,
        settlement_forecast: SettlementForecast,
    ) -> float:
        return float(np.clip(
            max(
                settlement_forecast.execution_confidence,
                settlement_forecast.settlement_confidence,
            ),
            0.0,
            1.0,
        ))

    @staticmethod
    def _direction_confirmation_key(market_key: str) -> str:
        return market_key or ""

    def _entry_price_wait_state_for(
        self,
        market_key: str,
        direction: Direction,
        live_market_price: Optional[float],
        raw_order_price: float,
        max_entry_price: float,
    ) -> tuple[bool, float, str]:
        key = self._direction_confirmation_key(market_key)
        if direction == Direction.NEUTRAL:
            self._entry_price_wait_state.pop(key, None)
            return False, raw_order_price, ""

        reference_price = None
        if live_market_price is not None and np.isfinite(live_market_price) and live_market_price > 0.0:
            reference_price = float(live_market_price)
        elif raw_order_price > 0.0 and np.isfinite(raw_order_price):
            reference_price = float(raw_order_price)

        if reference_price is None or reference_price <= 0.0:
            self._entry_price_wait_state.pop(key, None)
            return False, raw_order_price, "invalid_order_price_for_min_share_check"

        if reference_price > max_entry_price:
            now = time.time()
            state = self._entry_price_wait_state.get(key)
            if state is None or state.get("direction") != direction:
                state = {
                    "direction": direction,
                    "started_at": now,
                }
            state["last_price"] = reference_price
            state["updated_at"] = now
            state["max_entry_price"] = max_entry_price
            self._entry_price_wait_state[key] = state
            elapsed = now - float(state["started_at"])
            return (
                False,
                raw_order_price,
                (
                    "entry_price_wait_for_pullback("
                    f"direction={direction.value},"
                    f"price={reference_price:.4f}>max={max_entry_price:.4f},"
                    f"wait={elapsed:.1f}s)"
                ),
            )

        self._entry_price_wait_state.pop(key, None)
        adjusted_entry_price = min(max_entry_price, max(raw_order_price, reference_price))
        return True, adjusted_entry_price, ""

    def _direction_confirmation_state_for(
        self,
        market_key: str,
        settlement_forecast: SettlementForecast,
        direction: Direction,
        should_execute: bool,
    ) -> tuple[bool, float, str]:
        """
        Enforce the observation window, with an optional post-observation hold.

        The default is no extra hold: once observation is ready and valid, the
        signal can execute immediately through the remaining gates.
        """
        hold_seconds = float(getattr(self.config, "post_observation_confirmation_seconds", 3.0))
        hold_seconds = float(np.clip(hold_seconds, 0.0, 30.0))

        key = self._direction_confirmation_key(market_key)
        if direction == Direction.NEUTRAL:
            self._direction_confirmation_state.pop(key, None)
            return should_execute, 0.0, ""

        if should_execute and bool(getattr(self.config, "require_observation_window", True)):
            if not settlement_forecast.observation_ready:
                self._direction_confirmation_state.pop(key, None)
                return (
                    False,
                    hold_seconds,
                    f"observation_wait(obs={settlement_forecast.observation_seconds:.1f}s)",
                )
            if not settlement_forecast.observation_valid:
                self._direction_confirmation_state.pop(key, None)
                obs_reason = (settlement_forecast.observation_reason or "observation_invalid")
                obs_reason = obs_reason.replace("|", "/").replace(" ", "_")[:80]
                return (
                    False,
                    hold_seconds,
                    f"observation_invalid({obs_reason})",
                )

        if hold_seconds <= 0.0 or not market_key:
            self._direction_confirmation_state.pop(key, None)
            return should_execute, 0.0, ""

        now = time.time()
        expected_direction = settlement_forecast.expected_settlement_direction
        state = self._direction_confirmation_state.get(key)

        if state is None or state.get("direction") != direction:
            state = {
                "direction": direction,
                "started_at": now,
                "p_up": float(settlement_forecast.p_up_settlement),
            }
            self._direction_confirmation_state[key] = state
        elapsed = now - float(state["started_at"])

        if should_execute and settlement_forecast.observation_valid and elapsed < hold_seconds:
            remaining = max(0.0, hold_seconds - elapsed)
            return (
                False,
                remaining,
                f"post_observation_hold({direction.value},{elapsed:.1f}<{hold_seconds:.1f}s,expected={expected_direction.value})",
            )

        if should_execute and settlement_forecast.observation_valid:
            state["confirmed_at"] = now
            return should_execute, 0.0, ""

        return should_execute, 0.0, ""

    def _rejected_decision(
        self,
        timestamp: float,
        settlement_forecast: SettlementForecast,
        execution_estimate: ExecutionEstimate,
        regime_estimate: Optional[RegimeStateEstimate],
        market_price_up: Optional[float],
        market_price_down: Optional[float],
        confidence: float,
        reason: str,
        is_dry_run: bool,
        signal_type: SignalType = SignalType.SUPPRESSED,
    ) -> TradeDecision:
        direction, probability, market_price, edge, edge_bps = self._decision_values(
            settlement_forecast,
            market_price_up,
            market_price_down,
        )
        self._suppressed_count += 1
        return TradeDecision(
            timestamp=timestamp,
            direction=direction,
            token_id="",
            probability=probability,
            market_price=market_price,
            edge=edge,
            edge_bps=edge_bps,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            position_size=0.0,
            fill_probability=execution_estimate.expected_fill_probability,
            expected_slippage_bps=execution_estimate.expected_slippage_bps,
            settlement_forecast=settlement_forecast,
            execution_estimate=execution_estimate,
            regime_state=(
                regime_estimate.current_regime
                if regime_estimate
                else RegimeState.CALM_TRENDING
            ),
            signal_type=signal_type,
            is_dry_run=is_dry_run,
            reason=reason,
            should_trade=False,
        )

    def make_decision(
        self,
        settlement_forecast: SettlementForecast,
        execution_estimate: ExecutionEstimate,
        risk_state: Optional[RiskState] = None,
        regime_estimate: Optional[RegimeStateEstimate] = None,
        volatility_estimate: Optional[VolatilityEstimate] = None,
        multi_timescale: Optional[MultiTimescaleEstimate] = None,
        kalman_state: Optional[KalmanState] = None,
        orderbook_pressure: Optional[OrderbookPressure] = None,
        flow_metrics: Optional[FlowMetrics] = None,
        continuation_signal: Optional[ContinuationSignal] = None,
        reversal_signal: Optional[ReversalSignal] = None,
        exhaustion_signal: Optional[ExhaustionSignal] = None,
        burst_failure_signal: Optional[BurstFailureSignal] = None,
        entropy_filter: Optional[EntropyFilterResult] = None,
        polymarket_orderbook_up: Optional[PolymarketOrderbook] = None,
        polymarket_orderbook_down: Optional[PolymarketOrderbook] = None,
        market_price_up: Optional[float] = None,
        market_price_down: Optional[float] = None,
        token_id_up: Optional[str] = None,
        token_id_down: Optional[str] = None,
        market_slug: Optional[str] = None,
        minimum_order_size: Optional[float] = None,
        minimum_tick_size: Optional[float] = None,
        is_dry_run: bool = True,
    ) -> TradeDecision:
        """
        Make the final trade decision through all gates.
        
        Process:
        1. Check settlement forecast gate
        2. Check execution quality gate
        3. Check risk management gate
        4. Check decorrelation gate
        5. Scale confidence
        6. Compute final position size
        7. Build reason string
        8. Produce TradeDecision
        """
        timestamp = time.time()
        round_key = market_slug or ""
        if round_key and round_key != self._active_market_slug:
            self._active_market_slug = round_key
            self.decorrelation_filter.reset_for_new_round()
            self._direction_confirmation_state.clear()
            self._entry_price_wait_state.clear()

        # ---- Gate 1: Settlement forecast ----
        if not settlement_forecast.execution_recommended:
            # Settlement forecast does not recommend execution
            confidence = settlement_forecast.settlement_confidence
            should_trade = False
            reason = self._build_reason(
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                risk_state=risk_state,
                signal_type=SignalType.SUPPRESSED,
                confidence=confidence,
                should_trade=False,
                decorrelation_passed=True,
                decorrelation_reason="no_settlement_recommendation",
            )
            return self._rejected_decision(
                timestamp=timestamp,
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                regime_estimate=regime_estimate,
                market_price_up=market_price_up,
                market_price_down=market_price_down,
                confidence=confidence,
                reason=reason,
                is_dry_run=is_dry_run,
            )

        settlement_buffer_seconds = float(
            getattr(self.config.risk, "settlement_time_buffer_seconds", 30.0)
        )
        if settlement_forecast.time_to_settlement_seconds <= settlement_buffer_seconds:
            confidence = self._forecast_confidence_for_reject(settlement_forecast)
            buffer_reason = (
                "settlement_time_buffer("
                f"tau={settlement_forecast.time_to_settlement_seconds:.2f}s<="
                f"{settlement_buffer_seconds:.2f}s)"
            )
            reason = self._build_reason(
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                risk_state=risk_state,
                signal_type=SignalType.SUPPRESSED,
                confidence=confidence,
                should_trade=False,
                decorrelation_passed=True,
                decorrelation_reason=buffer_reason,
            )
            return self._rejected_decision(
                timestamp=timestamp,
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                regime_estimate=regime_estimate,
                market_price_up=market_price_up,
                market_price_down=market_price_down,
                confidence=confidence,
                reason=reason,
                is_dry_run=is_dry_run,
            )

        # ---- Gate 2: Execution quality ----
        if not self.execution_engine.should_execute(
            execution_estimate,
            settlement_forecast,
        ):
            confidence = self._forecast_confidence_for_reject(settlement_forecast)
            if execution_estimate.execution_quality == ExecutionQuality.REJECT:
                execution_reason = "execution_quality_reject"
            elif (
                execution_estimate.expected_fill_probability
                < self.config.execution.min_fill_probability
            ):
                execution_reason = (
                    "execution_fill_probability_low("
                    f"{execution_estimate.expected_fill_probability:.3f}<"
                    f"{self.config.execution.min_fill_probability:.3f})"
                )
            elif (
                execution_estimate.expected_slippage_bps
                > self.config.risk.min_edge_bps * 2.0
            ):
                execution_reason = (
                    "execution_slippage_too_high("
                    f"{execution_estimate.expected_slippage_bps:.1f}bps)"
                )
            else:
                execution_reason = "execution_engine_reject"
            reason = self._build_reason(
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                risk_state=risk_state,
                signal_type=SignalType.SUPPRESSED,
                confidence=confidence,
                should_trade=False,
                decorrelation_passed=True,
                decorrelation_reason=execution_reason,
            )
            return self._rejected_decision(
                timestamp=timestamp,
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                regime_estimate=regime_estimate,
                market_price_up=market_price_up,
                market_price_down=market_price_down,
                confidence=confidence,
                reason=reason,
                is_dry_run=is_dry_run,
            )

        # ---- Gate 3: Risk management ----
        if risk_state is not None:
            if risk_state.should_stop_trading:
                should_trade = False
                confidence = self._forecast_confidence_for_reject(settlement_forecast)
                reason = self._build_reason(
                    settlement_forecast=settlement_forecast,
                    execution_estimate=execution_estimate,
                    risk_state=risk_state,
                    signal_type=SignalType.SUPPRESSED,
                    confidence=confidence,
                    should_trade=False,
                    decorrelation_passed=True,
                    decorrelation_reason="risk_stop_trading",
                )
                return self._rejected_decision(
                    timestamp=timestamp,
                    settlement_forecast=settlement_forecast,
                    execution_estimate=execution_estimate,
                    regime_estimate=regime_estimate,
                    market_price_up=market_price_up,
                    market_price_down=market_price_down,
                    confidence=confidence,
                    reason=reason,
                    is_dry_run=is_dry_run,
                )

            if risk_state.is_in_cooldown:
                should_trade = False
                confidence = self._forecast_confidence_for_reject(settlement_forecast)
                reason = self._build_reason(
                    settlement_forecast=settlement_forecast,
                    execution_estimate=execution_estimate,
                    risk_state=risk_state,
                    signal_type=SignalType.SUPPRESSED,
                    confidence=confidence,
                    should_trade=False,
                    decorrelation_passed=True,
                    decorrelation_reason="risk_cooldown",
                )
                return self._rejected_decision(
                    timestamp=timestamp,
                    settlement_forecast=settlement_forecast,
                    execution_estimate=execution_estimate,
                    regime_estimate=regime_estimate,
                    market_price_up=market_price_up,
                    market_price_down=market_price_down,
                    confidence=confidence,
                    reason=reason,
                    is_dry_run=is_dry_run,
                )

        # ---- Gate 4: Decorrelation ----
        decorrelation_passed, decorrelation_reason, decorrelation_score = (
            self.decorrelation_filter.check_decorrelation(
                settlement_forecast=settlement_forecast,
                kalman_state=kalman_state,
                orderbook_pressure=orderbook_pressure,
                flow_metrics=flow_metrics,
                continuation_signal=continuation_signal,
                reversal_signal=reversal_signal,
                entropy_filter=entropy_filter,
                regime_state=regime_estimate.current_regime if regime_estimate else 0,
            )
        )

        if not decorrelation_passed:
            decorrelation_passed = True
            decorrelation_reason = f"soft_pass({decorrelation_reason})"

        # ---- Gate 5: Confidence scaling ----
        confidence = self.confidence_scaler.scale_confidence(
            settlement_forecast=settlement_forecast,
            execution_estimate=execution_estimate,
            regime_estimate=regime_estimate or self._fallback_regime(timestamp),
            volatility_estimate=volatility_estimate or self._fallback_volatility(timestamp),
            multi_timescale=multi_timescale,
            entropy_filter=entropy_filter,
            risk_state=risk_state,
        )

        confidence = float(np.clip(confidence, 0.0, 0.99))

        direction, probability, market_price, edge, edge_bps = self._decision_values(
            settlement_forecast,
            market_price_up,
            market_price_down,
        )
        if direction == Direction.NEUTRAL:
            reason = self._build_reason(
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                risk_state=risk_state,
                signal_type=SignalType.SUPPRESSED,
                confidence=confidence,
                should_trade=False,
                decorrelation_passed=decorrelation_passed,
                decorrelation_reason="directional_probability_below_threshold",
            )
            return self._rejected_decision(
                timestamp=timestamp,
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                regime_estimate=regime_estimate,
                market_price_up=market_price_up,
                market_price_down=market_price_down,
                confidence=confidence,
                reason=reason,
                is_dry_run=is_dry_run,
            )

        if (
            bool(getattr(self.config, "require_round_direction_lock_for_entry", True))
            and not settlement_forecast.round_direction_locked
        ):
            confidence = self._forecast_confidence_for_reject(settlement_forecast)
            reason = self._build_reason(
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                risk_state=risk_state,
                signal_type=SignalType.SUPPRESSED,
                confidence=confidence,
                should_trade=False,
                decorrelation_passed=True,
                decorrelation_reason="round_direction_lock_pending",
            )
            return self._rejected_decision(
                timestamp=timestamp,
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                regime_estimate=regime_estimate,
                market_price_up=market_price_up,
                market_price_down=market_price_down,
                confidence=confidence,
                reason=reason,
                is_dry_run=is_dry_run,
            )

        if bool(getattr(self.config, "one_entry_per_round", True)) and round_key:
            previous_direction = self._entered_market_directions.get(round_key)
            if previous_direction is not None:
                confidence = self._forecast_confidence_for_reject(settlement_forecast)
                reason = self._build_reason(
                    settlement_forecast=settlement_forecast,
                    execution_estimate=execution_estimate,
                    risk_state=risk_state,
                    signal_type=SignalType.SUPPRESSED,
                    confidence=confidence,
                    should_trade=False,
                    decorrelation_passed=True,
                    decorrelation_reason=(
                        f"round_entry_already_taken({round_key},{previous_direction.value})"
                    ),
                )
                return self._rejected_decision(
                    timestamp=timestamp,
                    settlement_forecast=settlement_forecast,
                    execution_estimate=execution_estimate,
                    regime_estimate=regime_estimate,
                    market_price_up=market_price_up,
                    market_price_down=market_price_down,
                    confidence=confidence,
                    reason=reason,
                    is_dry_run=is_dry_run,
                )

        should_trade, _confirm_delay_seconds, confirm_reason = self._direction_confirmation_state_for(
            market_key=round_key,
            settlement_forecast=settlement_forecast,
            direction=direction,
            should_execute=True,
        )
        if not should_trade:
            reason = self._build_reason(
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                risk_state=risk_state,
                signal_type=SignalType.SUPPRESSED,
                confidence=confidence,
                should_trade=False,
                decorrelation_passed=decorrelation_passed,
                decorrelation_reason=confirm_reason,
            )
            return self._rejected_decision(
                timestamp=timestamp,
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                regime_estimate=regime_estimate,
                market_price_up=market_price_up,
                market_price_down=market_price_down,
                confidence=confidence,
                reason=reason,
                is_dry_run=is_dry_run,
            )

        should_trade = True

        # ---- Gate 6: Final size and execution constraints ----
        # Compute final position size
        available_capital = 1000.0  # Default
        if risk_state is not None:
            available_capital = risk_state.available_capital

        position_size = self.execution_engine.compute_final_position_size(
            settlement_forecast=settlement_forecast,
            execution_estimate=execution_estimate,
            available_capital=available_capital,
        )

        min_order_shares = float(
            minimum_order_size
            if minimum_order_size is not None and minimum_order_size > 0.0
            else getattr(self.config.market, "min_order_size", 5.0)
        )
        directional_live_market_price = (
            market_price_up
            if direction == Direction.UP
            else market_price_down
            if direction == Direction.DOWN
            else None
        )
        raw_order_price = float(
            execution_estimate.limit_price
            or execution_estimate.effective_price
            or execution_estimate.best_ask_price
            or directional_live_market_price
            or 0.0
        )
        tick_size = float(
            minimum_tick_size
            if minimum_tick_size is not None and minimum_tick_size > 0.0
            else getattr(self.config.market, "price_tick", 0.01)
        )
        estimated_order_price = self._round_to_tick(raw_order_price, tick_size)
        max_entry_price = float(getattr(self.config.market, "max_entry_price", 0.65))
        enforce_max_entry_price = bool(
            getattr(self.config.market, "enforce_max_entry_price", False)
        )
        if estimated_order_price <= 0.0:
            should_trade = False
            reason = self._build_reason(
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                risk_state=risk_state,
                signal_type=SignalType.SUPPRESSED,
                confidence=confidence,
                should_trade=False,
                decorrelation_passed=decorrelation_passed,
                decorrelation_reason="invalid_order_price_for_min_share_check",
            )
            return self._rejected_decision(
                timestamp=timestamp,
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                regime_estimate=regime_estimate,
                market_price_up=market_price_up,
                market_price_down=market_price_down,
                confidence=confidence,
                reason=reason,
                is_dry_run=is_dry_run,
            )
        if enforce_max_entry_price:
            should_trade, adjusted_order_price, wait_reason = self._entry_price_wait_state_for(
                market_key=round_key,
                direction=direction,
                live_market_price=directional_live_market_price,
                raw_order_price=estimated_order_price,
                max_entry_price=max_entry_price,
            )
            if not should_trade:
                reason = self._build_reason(
                    settlement_forecast=settlement_forecast,
                    execution_estimate=execution_estimate,
                    risk_state=risk_state,
                    signal_type=SignalType.SUPPRESSED,
                    confidence=confidence,
                    should_trade=False,
                    decorrelation_passed=decorrelation_passed,
                    decorrelation_reason=wait_reason,
                )
                return self._rejected_decision(
                    timestamp=timestamp,
                    settlement_forecast=settlement_forecast,
                    execution_estimate=execution_estimate,
                    regime_estimate=regime_estimate,
                    market_price_up=market_price_up,
                    market_price_down=market_price_down,
                    confidence=confidence,
                    reason=reason,
                    is_dry_run=is_dry_run,
                )
            estimated_order_price = min(max_entry_price, self._round_to_tick(adjusted_order_price, tick_size))
            execution_estimate.limit_price = estimated_order_price
        max_viable_size = min(
            max(0.0, available_capital),
            max(0.0, self.config.risk.max_position_notional),
        )
        min_order_notional = min_order_shares * estimated_order_price
        if max_viable_size + 1.0e-9 < min_order_notional:
            should_trade = False
            reason = self._build_reason(
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                risk_state=risk_state,
                signal_type=SignalType.SUPPRESSED,
                confidence=confidence,
                should_trade=False,
                decorrelation_passed=decorrelation_passed,
                decorrelation_reason=(
                    "capital_below_min_order("
                    f"cap={max_viable_size:.2f}<"
                    f"min_notional={min_order_notional:.2f},"
                    f"min_shares={min_order_shares:.4f},"
                    f"price={estimated_order_price:.4f})"
                ),
            )
            return self._rejected_decision(
                timestamp=timestamp,
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                regime_estimate=regime_estimate,
                market_price_up=market_price_up,
                market_price_down=market_price_down,
                confidence=confidence,
                reason=reason,
                is_dry_run=is_dry_run,
            )
        position_size = min(max(position_size, min_order_notional), max_viable_size)
        estimated_shares = (
            position_size / estimated_order_price
            if estimated_order_price > 0.0
            else 0.0
        )
        # Polymarket FOK/FAK BUY market orders submit dollar amount, but the
        # CLOB minimum order size is still measured in conditional-token shares.
        if estimated_shares + 1.0e-9 < min_order_shares:
            should_trade = False
            reason = self._build_reason(
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                risk_state=risk_state,
                signal_type=SignalType.SUPPRESSED,
                confidence=confidence,
                should_trade=False,
                decorrelation_passed=decorrelation_passed,
                decorrelation_reason=(
                    f"below_clob_min_order_size("
                    f"shares={estimated_shares:.4f}<min_shares={min_order_shares:.4f},"
                    f"amount={position_size:.2f},price={estimated_order_price:.4f})"
                ),
            )
            return self._rejected_decision(
                timestamp=timestamp,
                settlement_forecast=settlement_forecast,
                execution_estimate=execution_estimate,
                regime_estimate=regime_estimate,
                market_price_up=market_price_up,
                market_price_down=market_price_down,
                confidence=confidence,
                reason=reason,
                is_dry_run=is_dry_run,
            )

        # ---- All gates passed: determine direction and execute ----

        # Token ID
        token_id = token_id_up if direction == Direction.UP and token_id_up else (
            token_id_down if direction == Direction.DOWN and token_id_down else ""
        )

        # Classify signal type
        signal_type = self._classify_signal_type(
            settlement_forecast=settlement_forecast,
            continuation_signal=continuation_signal,
            reversal_signal=reversal_signal,
            exhaustion_signal=exhaustion_signal,
            burst_failure_signal=burst_failure_signal,
            entropy_filter=entropy_filter,
        )

        # Build reason
        reason = self._build_reason(
            settlement_forecast=settlement_forecast,
            execution_estimate=execution_estimate,
            risk_state=risk_state,
            signal_type=signal_type,
            confidence=confidence,
            should_trade=True,
            decorrelation_passed=decorrelation_passed,
            decorrelation_reason=decorrelation_reason,
        )

        # Record trade in decorrelation filter
        self.decorrelation_filter.record_trade(
            direction=direction,
            settlement_forecast=settlement_forecast,
            kalman_state=kalman_state,
            orderbook_pressure=orderbook_pressure,
            flow_metrics=flow_metrics,
            continuation_signal=continuation_signal,
            reversal_signal=reversal_signal,
            entropy_filter=entropy_filter,
            regime_state=regime_estimate.current_regime if regime_estimate else 0,
        )
        self._decision_count += 1

        return TradeDecision(
            timestamp=timestamp,
            direction=direction,
            token_id=token_id,
            probability=probability,
            market_price=market_price,
            edge=edge,
            edge_bps=edge_bps,
            confidence=confidence,
            position_size=position_size,
            fill_probability=execution_estimate.expected_fill_probability,
            expected_slippage_bps=execution_estimate.expected_slippage_bps,
            settlement_forecast=settlement_forecast,
            execution_estimate=execution_estimate,
            regime_state=regime_estimate.current_regime if regime_estimate else RegimeState.CALM_TRENDING,
            signal_type=signal_type,
            is_dry_run=is_dry_run,
            reason=reason,
            should_trade=True,
        )

    def get_decision_stats(self) -> dict:
        """Return decision engine statistics."""
        return {
            "total_decisions": self._decision_count,
            "suppressed_decisions": self._suppressed_count,
            "confidence_stats": self.confidence_scaler.get_confidence_stats(),
            "decorrelation_stats": self.decorrelation_filter.get_decorrelation_stats(),
        }
