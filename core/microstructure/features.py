from __future__ import annotations

import math
from dataclasses import dataclass

from core.buffering.ring import EWMoment, RingBuffer
from core.config import StrategyConfig
from core.orderbook.top import CompositeTopOfBook
from core.probability.math import binary_entropy, clamp, sigmoid
from core.runtime.health import HealthMonitor
from core.types import MicrostructureSnapshot, PriceTick, TradeTick


@dataclass(slots=True)
class FlowState:
    halflife_s: float
    signed_flow: float = 0.0
    abs_flow: float = 0.0
    acceleration: float = 0.0
    last_signed_flow: float = 0.0
    last_ts_ns: int = 0

    def update(self, trade: TradeTick) -> None:
        if self.last_ts_ns == 0:
            dt = 0.05
        else:
            dt = clamp((trade.recv_mono_ns - self.last_ts_ns) * 1e-9, 1e-4, 5.0)
        alpha = 1.0 - 2.0 ** (-dt / max(1e-9, self.halflife_s))
        signed = trade.signed_size
        self.last_signed_flow = self.signed_flow
        self.signed_flow = (1.0 - alpha) * self.signed_flow + alpha * (signed / dt)
        self.abs_flow = (1.0 - alpha) * self.abs_flow + alpha * (abs(signed) / dt)
        flow_delta = self.signed_flow - self.last_signed_flow
        self.acceleration = (1.0 - alpha) * self.acceleration + alpha * (flow_delta / dt)
        self.last_ts_ns = trade.recv_mono_ns

    @property
    def aggression(self) -> float:
        return clamp(self.signed_flow / max(1e-9, self.abs_flow), -1.0, 1.0)

    @property
    def normalized_acceleration(self) -> float:
        return clamp(self.acceleration / max(1e-9, self.abs_flow), -4.0, 4.0)


class MicrostructureEngine:
    def __init__(self, config: StrategyConfig, health: HealthMonitor) -> None:
        self._config = config
        self._health = health
        self.books = CompositeTopOfBook()
        self._prices = RingBuffer[PriceTick](4096)
        self._returns = EWMoment(config.vol_halflife_s)
        self._return_abs = EWMoment(config.vol_halflife_s * 0.7)
        self._jumps = EWMoment(8.0)
        self._flow = FlowState(config.flow_halflife_s)
        self._last_truth_price = 0.0
        self._last_truth_ts_ns = 0
        self._last_signal_mid = 0.0
        self._breakout_peak = 0.0
        self._breakout_trough = 0.0
        self._last_direction = 0.0

    def update_truth_price(self, tick: PriceTick) -> None:
        if tick.price <= 0.0:
            return
        if self._last_truth_price > 0.0:
            dt = max(1e-6, (tick.recv_mono_ns - self._last_truth_ts_ns) * 1e-9)
            ret = math.log(tick.price / self._last_truth_price)
            ret_per_s = ret / dt
            self._returns.update(ret_per_s, tick.recv_mono_ns)
            self._return_abs.update(abs(ret_per_s), tick.recv_mono_ns)
            z = abs(ret_per_s - self._returns.mean) / max(1e-9, self._returns.std)
            self._jumps.update(z, tick.recv_mono_ns)
        self._prices.append(tick)
        self._last_truth_price = tick.price
        self._last_truth_ts_ns = tick.recv_mono_ns

    def update_trade(self, trade: TradeTick) -> None:
        self._flow.update(trade)

    def snapshot(
        self,
        ts_ns: int,
        seconds_to_expiry: float,
        price_to_beat: float,
        regime_trend: float = 0.0,
        regime_volatility: float = 0.0,
    ) -> MicrostructureSnapshot | None:
        truth = self._last_truth_price
        if truth <= 0.0 or price_to_beat <= 0.0:
            return None
        signal_mid = self.books.signal_mid or truth
        signed_distance = truth - price_to_beat
        realized_vol = max(1e-8, self._realized_vol_per_s())
        drift = self._returns.mean if self._returns.initialized else 0.0
        drift_uncertainty = self._returns.std if self._returns.initialized else realized_vol
        taker_aggression = self._flow.aggression
        flow_acceleration = self._flow.normalized_acceleration
        book_pressure = self._book_pressure(signal_mid)
        spoof_imb = self.books.spoof_resistant_imbalance
        queue_survival = self.books.queue_survival
        liquidity_stability = self.books.liquidity_stability
        spread_stability = self.books.spread_stability
        liquidity_vacuum = self._liquidity_vacuum()
        absorption_up, absorption_down = self._absorption(taker_aggression, book_pressure, signed_distance, realized_vol)
        exhaustion = self._exhaustion(taker_aggression, flow_acceleration, queue_survival)
        burst_failure = self._burst_failure(truth, signed_distance, taker_aggression, realized_vol)
        distance_horizon_vol = max(1e-8, truth * realized_vol * (60.0 ** 0.5))
        distance_z = clamp(signed_distance / distance_horizon_vol, -3.0, 3.0)
        entropy = binary_entropy(sigmoid(1.7 * taker_aggression + 1.3 * book_pressure + distance_z))
        return MicrostructureSnapshot(
            ts_mono_ns=ts_ns,
            seconds_to_expiry=seconds_to_expiry,
            truth_price=truth,
            price_to_beat=price_to_beat,
            signed_distance=signed_distance,
            signal_mid=signal_mid,
            exchange_divergence=self.books.exchange_divergence,
            realized_vol=realized_vol,
            jump_intensity=clamp(self._jumps.mean / 8.0, 0.0, 2.0) if self._jumps.initialized else 0.0,
            drift=drift,
            drift_uncertainty=drift_uncertainty,
            taker_aggression=taker_aggression,
            flow_acceleration=flow_acceleration,
            book_pressure=book_pressure,
            spoof_resistant_imbalance=spoof_imb,
            queue_survival=queue_survival,
            liquidity_stability=liquidity_stability,
            spread_stability=spread_stability,
            absorption_up=absorption_up,
            absorption_down=absorption_down,
            exhaustion=exhaustion,
            burst_failure=burst_failure,
            liquidity_vacuum=liquidity_vacuum,
            entropy=entropy,
            regime_trend=regime_trend,
            regime_volatility=regime_volatility,
            stale_penalty=self._health.stale_penalty(),
        )

    def _realized_vol_per_s(self) -> float:
        if not self._returns.initialized:
            return 1e-5
        garch = 0.78 * self._returns.std + 0.22 * self._return_abs.mean
        return max(1e-8, min(0.02, garch))

    def _book_pressure(self, signal_mid: float) -> float:
        b = self.books.binance.last
        c = self.books.coinbase.last
        numerator = 0.0
        denom = 0.0
        for tob, state in ((b, self.books.binance), (c, self.books.coinbase)):
            if tob is None:
                continue
            microprice = (tob.ask * tob.bid_size + tob.bid * tob.ask_size) / max(1e-9, tob.bid_size + tob.ask_size)
            pressure = (microprice - tob.mid) / max(1e-9, tob.spread + tob.mid * 1e-7)
            w = 0.5 + state.liquidity_stability + state.queue_survival
            numerator += pressure * w
            denom += w
        return clamp(numerator / max(1e-9, denom), -1.0, 1.0)

    def _liquidity_vacuum(self) -> float:
        vacuum = self.books.liquidity_vacuum
        spread_instability = 1.0 - self.books.spread_stability
        divergence = clamp(self.books.exchange_divergence * 5000.0, 0.0, 1.0)
        return clamp(0.52 * vacuum + 0.30 * spread_instability + 0.18 * divergence, 0.0, 1.0)

    def _absorption(
        self,
        taker_aggression: float,
        book_pressure: float,
        signed_distance: float,
        realized_vol: float,
    ) -> tuple[float, float]:
        price_response = clamp(signed_distance / max(1e-8, realized_vol * 3.0), -2.0, 2.0)
        up_abs = sigmoid(1.8 * taker_aggression - 1.3 * price_response - 0.9 * book_pressure) - 0.5
        down_abs = sigmoid(-1.8 * taker_aggression + 1.3 * price_response + 0.9 * book_pressure) - 0.5
        return clamp(2.0 * up_abs, 0.0, 1.0), clamp(2.0 * down_abs, 0.0, 1.0)

    def _exhaustion(self, taker_aggression: float, flow_acceleration: float, queue_survival: float) -> float:
        inv = abs(taker_aggression) * max(0.0, -math.copysign(flow_acceleration, taker_aggression or 1.0))
        queue_decay = 1.0 - queue_survival
        return clamp(0.58 * inv + 0.30 * queue_decay + 0.12 * self.books.liquidity_vacuum, 0.0, 1.0)

    def _burst_failure(
        self,
        price: float,
        signed_distance: float,
        taker_aggression: float,
        realized_vol: float,
    ) -> float:
        direction = 1.0 if taker_aggression > 0.08 else -1.0 if taker_aggression < -0.08 else self._last_direction
        if direction > 0:
            self._breakout_peak = max(self._breakout_peak or price, price)
            failure = max(0.0, self._breakout_peak - price) / max(1e-8, realized_vol * 2.0)
        elif direction < 0:
            self._breakout_trough = min(self._breakout_trough or price, price)
            failure = max(0.0, price - self._breakout_trough) / max(1e-8, realized_vol * 2.0)
        else:
            failure = 0.0
        if direction != self._last_direction:
            self._breakout_peak = price
            self._breakout_trough = price
        self._last_direction = direction
        boundary_fail = 1.0 if signed_distance * direction < 0 and abs(taker_aggression) > 0.35 else 0.0
        return clamp(0.72 * failure + 0.28 * boundary_fail, 0.0, 1.0)

