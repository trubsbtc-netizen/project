from __future__ import annotations

from dataclasses import dataclass

from core.buffering.ring import EWMoment
from core.probability.math import clamp
from core.types import SignalSource, TopOfBook


@dataclass(slots=True)
class QueueSideState:
    price: float = 0.0
    size: float = 0.0
    last_ts_ns: int = 0
    cumulative_added: float = 0.0
    cumulative_removed: float = 0.0
    cancellation_velocity: float = 0.0
    survival_ewm: float = 1.0

    def update(self, price: float, size: float, ts_ns: int) -> None:
        if self.last_ts_ns == 0:
            self.price = price
            self.size = size
            self.last_ts_ns = ts_ns
            return
        dt = max(1e-6, (ts_ns - self.last_ts_ns) * 1e-9)
        same_price = abs(price - self.price) <= max(1e-12, self.price * 1e-8)
        if not same_price:
            removed = self.size
            added = size
            survival = 0.0
        else:
            delta = size - self.size
            added = max(0.0, delta)
            removed = max(0.0, -delta)
            survival = min(1.0, size / max(1e-9, self.size)) if self.size > 0.0 else 1.0
        alpha = 1.0 - 2.0 ** (-dt / 3.0)
        self.cumulative_added += added
        self.cumulative_removed += removed
        self.cancellation_velocity = (1.0 - alpha) * self.cancellation_velocity + alpha * (removed / dt)
        self.survival_ewm = (1.0 - alpha) * self.survival_ewm + alpha * survival
        self.price = price
        self.size = size
        self.last_ts_ns = ts_ns


class TopOfBookState:
    def __init__(self, source: SignalSource, symbol: str, halflife_s: float = 8.0) -> None:
        self.source = source
        self.symbol = symbol
        self.bid = QueueSideState()
        self.ask = QueueSideState()
        self._spread = EWMoment(halflife_s)
        self._mid = EWMoment(halflife_s)
        self._imbalance = EWMoment(halflife_s)
        self.last: TopOfBook | None = None

    def update(self, tob: TopOfBook) -> None:
        self.last = tob
        self.bid.update(tob.bid, tob.bid_size, tob.recv_mono_ns)
        self.ask.update(tob.ask, tob.ask_size, tob.recv_mono_ns)
        spread = tob.spread / max(1e-9, tob.mid)
        imb = (tob.bid_size - tob.ask_size) / max(1e-9, tob.bid_size + tob.ask_size)
        self._spread.update(spread, tob.recv_mono_ns)
        self._mid.update(tob.mid, tob.recv_mono_ns)
        self._imbalance.update(imb, tob.recv_mono_ns)

    @property
    def mid(self) -> float:
        return self.last.mid if self.last is not None else 0.0

    @property
    def spread_stability(self) -> float:
        if not self._spread.initialized:
            return 0.0
        cv = self._spread.std / max(1e-9, self._spread.mean + 1e-9)
        return clamp(1.0 / (1.0 + 8.0 * cv), 0.0, 1.0)

    @property
    def queue_survival(self) -> float:
        return clamp(0.5 * (self.bid.survival_ewm + self.ask.survival_ewm), 0.0, 1.0)

    @property
    def cancellation_pressure(self) -> float:
        total = self.bid.cancellation_velocity + self.ask.cancellation_velocity
        if total <= 0.0:
            return 0.0
        return clamp((self.ask.cancellation_velocity - self.bid.cancellation_velocity) / total, -1.0, 1.0)

    @property
    def raw_imbalance(self) -> float:
        if self.last is None:
            return 0.0
        total = self.last.bid_size + self.last.ask_size
        return 0.0 if total <= 0.0 else clamp((self.last.bid_size - self.last.ask_size) / total, -1.0, 1.0)

    @property
    def spoof_resistant_imbalance(self) -> float:
        persistence = self.queue_survival
        cancellation_skew = self.cancellation_pressure
        raw = self.raw_imbalance
        spoof_penalty = clamp(1.0 - abs(cancellation_skew) * (1.0 - persistence), 0.05, 1.0)
        return clamp(raw * persistence * spoof_penalty - 0.25 * cancellation_skew * (1.0 - persistence), -1.0, 1.0)

    @property
    def liquidity_stability(self) -> float:
        if self.last is None:
            return 0.0
        depth = self.last.bid_size + self.last.ask_size
        cancel = self.bid.cancellation_velocity + self.ask.cancellation_velocity
        survival = self.queue_survival
        return clamp(survival * depth / max(depth + 2.0 * cancel, 1e-9), 0.0, 1.0)

    @property
    def liquidity_vacuum(self) -> float:
        if self.last is None:
            return 1.0
        depth = self.last.bid_size + self.last.ask_size
        spread = self.last.spread / max(1e-9, self.last.mid)
        return clamp((1.0 - self.liquidity_stability) * (1.0 + 12.0 * spread) / (1.0 + depth), 0.0, 1.0)


class CompositeTopOfBook:
    def __init__(self) -> None:
        self.binance = TopOfBookState(SignalSource.BINANCE, "BTCUSDT")
        self.coinbase = TopOfBookState(SignalSource.COINBASE, "BTC-USD")
        self.polymarket_up = TopOfBookState(SignalSource.POLYMARKET_BOOK, "UP")
        self.polymarket_down = TopOfBookState(SignalSource.POLYMARKET_BOOK, "DOWN")

    def update_exchange(self, tob: TopOfBook) -> None:
        if tob.source is SignalSource.BINANCE:
            self.binance.update(tob)
        elif tob.source is SignalSource.COINBASE:
            self.coinbase.update(tob)
        elif tob.symbol.upper() == "UP":
            self.polymarket_up.update(tob)
        elif tob.symbol.upper() == "DOWN":
            self.polymarket_down.update(tob)

    @property
    def signal_mid(self) -> float:
        b = self.binance.mid
        c = self.coinbase.mid
        if b > 0.0 and c > 0.0:
            return 0.5 * (b + c)
        return b or c

    @property
    def exchange_divergence(self) -> float:
        b = self.binance.mid
        c = self.coinbase.mid
        if b <= 0.0 or c <= 0.0:
            return 0.0
        return abs(b - c) / max(1e-9, 0.5 * (b + c))

    @property
    def spoof_resistant_imbalance(self) -> float:
        values = []
        weights = []
        for state in (self.binance, self.coinbase):
            if state.last is None:
                continue
            values.append(state.spoof_resistant_imbalance)
            weights.append(0.5 + state.liquidity_stability)
        if not values:
            return 0.0
        return clamp(sum(v * w for v, w in zip(values, weights)) / sum(weights), -1.0, 1.0)

    @property
    def queue_survival(self) -> float:
        states = [s for s in (self.binance, self.coinbase) if s.last is not None]
        return sum(s.queue_survival for s in states) / len(states) if states else 0.0

    @property
    def spread_stability(self) -> float:
        states = [s for s in (self.binance, self.coinbase) if s.last is not None]
        return sum(s.spread_stability for s in states) / len(states) if states else 0.0

    @property
    def liquidity_stability(self) -> float:
        states = [s for s in (self.binance, self.coinbase) if s.last is not None]
        return sum(s.liquidity_stability for s in states) / len(states) if states else 0.0

    @property
    def liquidity_vacuum(self) -> float:
        states = [s for s in (self.binance, self.coinbase) if s.last is not None]
        return sum(s.liquidity_vacuum for s in states) / len(states) if states else 1.0

