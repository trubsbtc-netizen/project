"""
Validated multi-timeframe RSI/MACD momentum evidence for BTC 5-minute markets.

The model resamples settlement-basis BTC ticks onto a 1-second grid, then
computes scale-free RSI/MACD evidence on horizons from 5 seconds to 5 minutes.
Each timeframe contributes a bounded log-likelihood ratio with robust
volatility normalization and Newey-West trend validation.  The output is used
as stabilizing evidence for the settlement forecast, not as a standalone
trade trigger.
"""

import time
from typing import Dict, Optional, Tuple
from threading import Lock

import numpy as np

from polymarket_bot.config import BotConfig, Direction
from polymarket_bot.bot_types import TechnicalMomentumSignal, TickBuffer


class TechnicalMomentumStrategy:
    """Validated RSI/MACD multi-timeframe directional signal."""

    TIMEFRAMES = (5, 15, 30, 60, 120, 300)

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.grid_seconds = 1.0
        self.min_samples = int(getattr(config, "technical_momentum_min_samples", 32))
        self.min_timeframes = int(getattr(config, "technical_momentum_min_timeframes", 2))
        self.min_consensus = float(getattr(config, "technical_momentum_min_consensus", 0.56))
        self.max_conflict = float(getattr(config, "technical_momentum_max_conflict", 0.52))
        self.max_log_odds = float(getattr(config, "technical_momentum_max_log_odds", 2.6))
        self.smoothing_alpha = float(getattr(config, "technical_momentum_smoothing_alpha", 0.62))
        self.hysteresis_log_odds = float(
            getattr(config, "technical_momentum_hysteresis_log_odds", 0.22)
        )
        self.rsi50_macd50_required = bool(
            getattr(config, "technical_momentum_rsi50_macd50_required", True)
        )
        self.rsi50_min_distance = float(
            getattr(config, "technical_momentum_rsi50_min_distance", 1.0)
        )
        self.macd50_min_z = float(
            getattr(config, "technical_momentum_macd50_min_z", 0.15)
        )
        self.tick_buffer_slack_seconds = float(
            getattr(config, "technical_momentum_tick_buffer_slack_seconds", 8.0)
        )
        self._last_market_slug = ""
        self._last_log_odds = 0.0
        self._last_direction = Direction.NEUTRAL
        self._state_by_slug: Dict[str, Dict[str, object]] = {}
        self._state_lock = Lock()

    @staticmethod
    def _sigmoid(log_odds: float) -> float:
        if log_odds >= 0.0:
            z = np.exp(-log_odds)
            return float(1.0 / (1.0 + z))
        z = np.exp(log_odds)
        return float(z / (1.0 + z))

    @staticmethod
    def _logit(probability: float) -> float:
        p = float(np.clip(probability, 1e-8, 1.0 - 1e-8))
        return float(np.log(p / (1.0 - p)))

    @staticmethod
    def _bounded(value: float, scale: float) -> float:
        if scale <= 0.0 or not np.isfinite(scale):
            return 0.0
        return float(np.tanh(float(value) / float(scale)))

    @staticmethod
    def _ewm(values: np.ndarray, span: int) -> np.ndarray:
        if len(values) == 0:
            return values
        alpha = 2.0 / (float(max(span, 1)) + 1.0)
        decay = 1.0 - alpha
        out = np.empty_like(values, dtype=np.float64)
        numerator = 0.0
        denominator = 0.0
        for idx, value in enumerate(values):
            numerator = float(value) + decay * numerator
            denominator = 1.0 + decay * denominator
            out[idx] = numerator / max(denominator, 1.0e-12)
        return out

    @staticmethod
    def _mad_scale(values: np.ndarray, floor: float = 1.0e-9) -> float:
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return float(floor)
        center = float(np.median(values))
        mad = float(np.median(np.abs(values - center)))
        return float(max(1.4826 * mad, floor))

    @staticmethod
    def _wilder_rsi(log_prices: np.ndarray, period: int) -> Optional[float]:
        if len(log_prices) < period + 2:
            return None
        deltas = np.diff(log_prices)
        gains = np.maximum(deltas, 0.0)
        losses = np.maximum(-deltas, 0.0)
        avg_gain = float(np.mean(gains[:period]))
        avg_loss = float(np.mean(losses[:period]))
        for idx in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + float(gains[idx])) / period
            avg_loss = (avg_loss * (period - 1) + float(losses[idx])) / period
        if avg_gain <= 1.0e-14 and avg_loss <= 1.0e-14:
            return 50.0
        if avg_loss <= 1.0e-14:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    @staticmethod
    def _newey_west_t_stat(returns: np.ndarray) -> float:
        returns = returns[np.isfinite(returns)]
        n = len(returns)
        if n < 4:
            return 0.0
        mean = float(np.mean(returns))
        centered = returns - mean
        max_lag = int(np.clip(round(np.sqrt(n)), 1, min(8, n - 1)))
        gamma0 = float(np.dot(centered, centered) / n)
        long_run_var = gamma0
        for lag in range(1, max_lag + 1):
            cov = float(np.dot(centered[lag:], centered[:-lag]) / n)
            weight = 1.0 - lag / float(max_lag + 1)
            long_run_var += 2.0 * weight * cov
        long_run_var = max(long_run_var, 1.0e-16)
        standard_error = np.sqrt(long_run_var / n)
        return float(np.clip(mean / standard_error, -8.0, 8.0))

    @staticmethod
    def _resample_grid(
        tick_buffer: TickBuffer,
        lookback_seconds: float,
        grid_seconds: float,
        now: float,
        slack_seconds: float = 5.0,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        if tick_buffer.len() < 3:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                float("inf"),
            )

        timestamps = np.array(tick_buffer.timestamps, dtype=np.float64)
        prices = np.array(tick_buffer.prices, dtype=np.float64)
        mask = (
            np.isfinite(timestamps)
            & np.isfinite(prices)
            & (timestamps > 0.0)
            & (prices > 0.0)
            & (timestamps >= now - lookback_seconds - max(float(slack_seconds), 0.0))
        )
        timestamps = timestamps[mask]
        prices = prices[mask]
        if len(timestamps) < 3:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                float("inf"),
            )

        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        prices = prices[order]

        unique_ts = []
        unique_prices = []
        for ts, price in zip(timestamps, prices):
            if unique_ts and abs(float(ts) - unique_ts[-1]) < 1.0e-6:
                unique_prices[-1] = float(price)
            else:
                unique_ts.append(float(ts))
                unique_prices.append(float(price))

        timestamps = np.array(unique_ts, dtype=np.float64)
        prices = np.array(unique_prices, dtype=np.float64)
        gaps = np.diff(timestamps)
        max_gap = float(np.max(gaps)) if len(gaps) else float("inf")

        end = float(np.floor(min(now, timestamps[-1]) / grid_seconds) * grid_seconds)
        start = float(
            max(
                np.ceil((end - lookback_seconds) / grid_seconds) * grid_seconds,
                np.ceil(timestamps[0] / grid_seconds) * grid_seconds,
            )
        )
        if end - start < grid_seconds * 8.0:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                max_gap,
            )

        grid_count = int(round((end - start) / grid_seconds)) + 1
        if grid_count < 2:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                max_gap,
            )
        grid = start + np.arange(grid_count, dtype=np.float64) * float(grid_seconds)
        sampled = np.interp(grid, timestamps, prices)
        return grid, sampled, max_gap

    def _timeframe_signal(
        self,
        prices: np.ndarray,
        timeframe_seconds: int,
    ) -> Optional[Tuple[float, float, float, float, float, float, float, float]]:
        bars = int(round(float(timeframe_seconds) / self.grid_seconds))
        if bars < 4 or len(prices) < bars + 1:
            return None

        fast = int(np.clip(round(max(2.0, bars * 0.22)), 2, 34))
        slow = int(np.clip(round(max(float(fast + 1), bars * 0.55)), fast + 1, 89))
        signal_span = int(np.clip(round(max(2.0, bars * 0.18)), 2, 34))
        burn_in = min(max(3 * slow, 3 * signal_span), max(0, len(prices) - (bars + 1)))
        segment = prices[-(bars + 1 + burn_in):]
        if np.any(segment <= 0.0) or not np.all(np.isfinite(segment)):
            return None
        log_prices = np.log(segment)
        analysis_log_prices = log_prices[-(bars + 1):]
        returns = np.diff(analysis_log_prices)
        if len(returns) < 4:
            return None

        realized_scale = self._mad_scale(returns, floor=2.0e-7)
        trend_t = self._newey_west_t_stat(returns)
        trend_score = self._bounded(trend_t, 2.75)

        rsi_cap = max(21, bars // 4)
        rsi_period = int(np.clip(round(max(3.0, bars * 0.34)), 3, rsi_cap))
        rsi = self._wilder_rsi(analysis_log_prices, rsi_period)
        if rsi is None:
            return None
        rsi_score = self._bounded(rsi - 50.0, 17.5)
        rsi_log_odds = 1.20 * rsi_score

        if len(log_prices) < slow + signal_span + 1:
            return None

        ema_fast = self._ewm(log_prices, fast)
        ema_slow = self._ewm(log_prices, slow)
        macd = ema_fast - ema_slow
        signal = self._ewm(macd, signal_span)
        hist = macd - signal
        hist_scale = self._mad_scale(hist[-max(6, min(len(hist), slow)):], floor=realized_scale * 0.35)
        macd_score = self._bounded(hist[-1], 2.35 * hist_scale)

        slope_window = min(len(hist), max(3, signal_span + 1))
        hist_slope = float((hist[-1] - hist[-slope_window]) / max(slope_window - 1, 1))
        slope_scale = max(hist_scale / max(signal_span, 1), realized_scale * 0.05)
        slope_score = self._bounded(hist_slope, 2.50 * slope_scale)

        impulse = float(np.sum(returns) / max(realized_scale * np.sqrt(len(returns)), 1.0e-9))
        impulse_score = self._bounded(impulse, 2.50)

        macd_log_odds = 1.45 * macd_score + 0.45 * slope_score
        trend_log_odds = 0.82 * trend_score + 0.38 * impulse_score
        log_odds = rsi_log_odds + macd_log_odds + trend_log_odds
        log_odds = float(np.clip(log_odds, -3.0, 3.0))
        score = float(np.clip(np.tanh(log_odds / 2.4), -0.985, 0.985))

        components = np.array(
            [rsi_score, macd_score, slope_score, trend_score, impulse_score],
            dtype=np.float64,
        )
        component_dispersion = float(np.std(components))
        max_component_dispersion = np.sqrt(0.80)
        agreement = float(np.clip(1.0 - component_dispersion / max_component_dispersion, 0.0, 1.0))
        p_up = float(np.clip(self._sigmoid(log_odds), 0.001, 0.999))
        return (
            p_up,
            score,
            float(rsi),
            float(hist[-1]),
            agreement,
            abs(score),
            float(rsi_log_odds),
            float(macd_log_odds),
        )

    def _rsi50_macd50_confirmation(
        self,
        prices: np.ndarray,
    ) -> Tuple[Direction, float, float, float, bool, str]:
        if len(prices) < 101:
            return Direction.NEUTRAL, 50.0, 0.0, 0.0, False, "rsi50_macd50_samples<101"
        if np.any(prices <= 0.0) or not np.all(np.isfinite(prices)):
            return Direction.NEUTRAL, 50.0, 0.0, 0.0, False, "rsi50_macd50_bad_prices"

        log_prices = np.log(prices)
        rsi50 = self._wilder_rsi(log_prices, 50)
        if rsi50 is None or not np.isfinite(rsi50):
            return Direction.NEUTRAL, 50.0, 0.0, 0.0, False, "rsi50_unavailable"

        ema_fast = self._ewm(log_prices, 50)
        ema_slow = self._ewm(log_prices, 100)
        macd50 = ema_fast - ema_slow
        macd50_signal = self._ewm(macd50, 50)
        macd50_hist = macd50 - macd50_signal
        hist_window = macd50_hist[-50:]
        macd50_scale = self._mad_scale(hist_window, floor=2.0e-7)
        macd50_z = float(np.clip(macd50_hist[-1] / max(macd50_scale, 1.0e-12), -8.0, 8.0))

        rsi_delta = float(rsi50 - 50.0)
        rsi_up = rsi_delta >= self.rsi50_min_distance
        rsi_down = rsi_delta <= -self.rsi50_min_distance
        macd_up = macd50_z >= self.macd50_min_z
        macd_down = macd50_z <= -self.macd50_min_z

        if rsi_up and macd_up:
            return Direction.UP, float(rsi50), float(macd50_hist[-1]), macd50_z, True, "rsi50_macd50_up"
        if rsi_down and macd_down:
            return Direction.DOWN, float(rsi50), float(macd50_hist[-1]), macd50_z, True, "rsi50_macd50_down"
        return (
            Direction.NEUTRAL,
            float(rsi50),
            float(macd50_hist[-1]),
            macd50_z,
            False,
            "rsi50_macd50_conflict",
        )

    def _empty_signal(
        self,
        *,
        now: float,
        market_slug: str,
        reason: str,
        tf_probabilities: Optional[Dict[str, float]] = None,
        tf_weights: Optional[Dict[str, float]] = None,
        rsi_values: Optional[Dict[str, float]] = None,
        macd_histograms: Optional[Dict[str, float]] = None,
    ) -> TechnicalMomentumSignal:
        return TechnicalMomentumSignal(
            timestamp=now,
            market_slug=market_slug,
            is_valid=False,
            direction=Direction.NEUTRAL,
            p_up=0.5,
            p_down=0.5,
            confidence=0.0,
            validation_score=0.0,
            consensus_score=0.0,
            conflict_score=1.0,
            log_odds=0.0,
            rsi_log_odds=0.0,
            macd_log_odds=0.0,
            timeframe_probabilities=tf_probabilities or {},
            timeframe_weights=tf_weights or {},
            rsi_values=rsi_values or {},
            macd_histograms=macd_histograms or {},
            dominant_timeframe="",
            reason=reason,
        )

    def evaluate(
        self,
        *,
        market_slug: str,
        tick_buffer: TickBuffer,
        time_to_settlement: float,
        now: Optional[float] = None,
    ) -> TechnicalMomentumSignal:
        now = time.time() if now is None else float(now)
        with self._state_lock:
            state = self._state_by_slug.get(market_slug)
            if state is None:
                state = {
                    "last_log_odds": 0.0,
                    "last_direction": Direction.NEUTRAL,
                }
                self._state_by_slug[market_slug] = state
            if len(self._state_by_slug) > 128:
                stale_keys = list(self._state_by_slug.keys())[:-64]
                for stale_key in stale_keys:
                    self._state_by_slug.pop(stale_key, None)
            last_log_odds = float(state.get("last_log_odds", 0.0))
            last_direction = state.get("last_direction", Direction.NEUTRAL)

        if market_slug != self._last_market_slug:
            self._last_market_slug = market_slug
            self._last_log_odds = last_log_odds
            self._last_direction = last_direction if isinstance(last_direction, Direction) else Direction.NEUTRAL

        lookback = float(max(90.0, min(360.0, max(self.TIMEFRAMES) + 30.0)))
        _, prices, max_gap = self._resample_grid(
            tick_buffer,
            lookback,
            self.grid_seconds,
            now,
            self.tick_buffer_slack_seconds,
        )
        if len(prices) < self.min_samples:
            return self._empty_signal(
                now=now,
                market_slug=market_slug,
                reason=f"samples<{self.min_samples}",
            )

        tf_probabilities: Dict[str, float] = {}
        tf_weights: Dict[str, float] = {}
        rsi_values: Dict[str, float] = {}
        macd_histograms: Dict[str, float] = {}
        scores = []
        log_odds_values = []
        weights = []
        rsi_llrs = []
        macd_llrs = []

        tau = max(float(time_to_settlement), 1.0)
        for timeframe in self.TIMEFRAMES:
            signal = self._timeframe_signal(prices, timeframe)
            if signal is None:
                continue
            (
                p_up,
                score,
                rsi,
                macd_hist,
                agreement,
                magnitude,
                rsi_llr,
                macd_llr,
            ) = signal
            tf_key = f"{timeframe}s"
            horizon_match = np.exp(-abs(np.log(max(timeframe, 5.0) / max(tau, 5.0))) / 1.10)
            recency_weight = 1.0 / np.sqrt(max(timeframe / 5.0, 1.0))
            evidence_weight = (0.45 + 0.55 * agreement) * (0.35 + 0.65 * magnitude)
            weight = float((0.30 + 0.70 * horizon_match) * evidence_weight * recency_weight)
            tf_probabilities[tf_key] = float(p_up)
            tf_weights[tf_key] = weight
            rsi_values[tf_key] = float(rsi)
            macd_histograms[tf_key] = float(macd_hist)
            scores.append(float(score))
            log_odds_values.append(self._logit(p_up))
            weights.append(weight)
            rsi_llrs.append(rsi_llr)
            macd_llrs.append(macd_llr)

        if len(scores) < self.min_timeframes or sum(weights) <= 1.0e-12:
            return self._empty_signal(
                now=now,
                market_slug=market_slug,
                reason=f"timeframes<{self.min_timeframes}",
                tf_probabilities=tf_probabilities,
                tf_weights=tf_weights,
                rsi_values=rsi_values,
                macd_histograms=macd_histograms,
            )

        (
            rsi50_macd50_direction,
            rsi50_value,
            macd50_hist,
            macd50_z,
            rsi50_macd50_valid,
            rsi50_macd50_reason,
        ) = self._rsi50_macd50_confirmation(prices)
        rsi_values["RSI50"] = float(rsi50_value)
        macd_histograms["MACD50"] = float(macd50_hist)

        weight_arr = np.array(weights, dtype=np.float64)
        weight_arr = weight_arr / max(float(np.sum(weight_arr)), 1.0e-12)
        score_arr = np.array(scores, dtype=np.float64)
        llr_arr = np.array(log_odds_values, dtype=np.float64)

        weighted_score = float(np.sum(weight_arr * score_arr))
        raw_log_odds = float(np.sum(weight_arr * llr_arr))
        signs = np.sign(score_arr)
        dominant_sign = 1.0 if weighted_score >= 0.0 else -1.0
        sign_strength = np.maximum(np.abs(score_arr), 1.0e-6)
        consensus_score = float(
            np.sum(weight_arr * sign_strength * (signs == dominant_sign))
            / max(np.sum(weight_arr * sign_strength), 1.0e-12)
        )
        dispersion = float(np.sqrt(np.sum(weight_arr * (score_arr - weighted_score) ** 2)))
        llr_dispersion = float(np.sqrt(np.sum(weight_arr * (llr_arr - raw_log_odds) ** 2)))
        conflict_score = float(
            np.clip(0.55 * dispersion / 0.70 + 0.35 * llr_dispersion / 2.0 + 0.10 * (1.0 - consensus_score), 0.0, 1.0)
        )

        sample_score = float(1.0 - np.exp(-len(prices) / max(float(self.min_samples), 1.0)))
        timeframe_score = float(np.clip(len(scores) / max(float(len(self.TIMEFRAMES)), 1.0), 0.0, 1.0))
        gap_score = float(np.exp(-max(0.0, max_gap - 3.0) / 3.0)) if np.isfinite(max_gap) else 0.0
        magnitude_score = float(np.clip(abs(weighted_score) / 0.52, 0.0, 1.0))
        validation_score = float(
            np.clip(
                (
                    max(sample_score, 1.0e-3)
                    * max(consensus_score, 1.0e-3)
                    * max(1.0 - conflict_score, 1.0e-3)
                    * max(0.45 + 0.55 * timeframe_score, 1.0e-3)
                    * max(gap_score, 1.0e-3)
                )
                ** 0.20,
                0.0,
                1.0,
            )
        )
        confidence = float(
            np.clip(
                0.24 * sample_score
                + 0.28 * consensus_score
                + 0.22 * magnitude_score
                + 0.16 * (1.0 - conflict_score)
                + 0.10 * gap_score,
                0.0,
                0.99,
            )
        )

        bounded_raw = float(np.clip(raw_log_odds * validation_score, -self.max_log_odds, self.max_log_odds))
        previous_direction = last_direction if isinstance(last_direction, Direction) else Direction.NEUTRAL
        if previous_direction != Direction.NEUTRAL:
            previous_sign = 1.0 if previous_direction == Direction.UP else -1.0
            current_sign = 1.0 if bounded_raw >= 0.0 else -1.0
            switch_barrier = self.hysteresis_log_odds + 0.14 * (1.0 - confidence) + 0.10 * conflict_score
            if current_sign != previous_sign and abs(bounded_raw) < switch_barrier:
                bounded_raw = 0.0

        alpha = float(np.clip(self.smoothing_alpha, 0.05, 1.0))
        smoothed_log_odds = alpha * bounded_raw + (1.0 - alpha) * last_log_odds
        smoothed_log_odds = float(np.clip(smoothed_log_odds, -self.max_log_odds, self.max_log_odds))

        p_up = float(np.clip(self._sigmoid(smoothed_log_odds), 0.001, 0.999))
        direction = Direction.UP if p_up >= 0.5 else Direction.DOWN
        rsi50_macd50_aligned = (
            not self.rsi50_macd50_required
            or (
                rsi50_macd50_valid
                and rsi50_macd50_direction == direction
            )
        )
        is_valid = (
            consensus_score >= self.min_consensus
            and conflict_score <= self.max_conflict
            and validation_score >= 0.48
            and confidence >= 0.34
            and abs(smoothed_log_odds) >= 0.08
            and rsi50_macd50_aligned
        )

        if is_valid:
            next_direction = direction if abs(smoothed_log_odds) >= 0.05 else previous_direction
            with self._state_lock:
                self._state_by_slug[market_slug] = {
                    "last_log_odds": smoothed_log_odds,
                    "last_direction": next_direction,
                }
            self._last_log_odds = smoothed_log_odds
            self._last_direction = next_direction

        total_tf_weight = max(sum(tf_weights.values()), 1.0e-12)
        tf_weights = {
            key: float(value / total_tf_weight)
            for key, value in tf_weights.items()
        }
        dominant_timeframe = max(tf_weights.items(), key=lambda item: item[1])[0] if tf_weights else ""
        reason = (
            (
                f"valid(n_tf={len(scores)},gap={max_gap:.1f}s,"
                f"rsi50={rsi50_value:.1f},macd50z={macd50_z:+.2f})"
            )
            if is_valid
            else (
                f"weak(n_tf={len(scores)},cons={consensus_score:.2f},"
                f"conflict={conflict_score:.2f},val={validation_score:.2f},"
                f"{rsi50_macd50_reason},rsi50={rsi50_value:.1f},macd50z={macd50_z:+.2f})"
            )
        )
        return TechnicalMomentumSignal(
            timestamp=now,
            market_slug=market_slug,
            is_valid=is_valid,
            direction=direction,
            p_up=p_up,
            p_down=1.0 - p_up,
            confidence=confidence if is_valid else min(confidence, 0.33),
            validation_score=validation_score,
            consensus_score=consensus_score,
            conflict_score=conflict_score,
            log_odds=smoothed_log_odds,
            rsi_log_odds=float(np.clip(np.dot(weight_arr, np.array(rsi_llrs, dtype=np.float64)) if rsi_llrs else 0.0, -self.max_log_odds, self.max_log_odds)),
            macd_log_odds=float(np.clip(np.dot(weight_arr, np.array(macd_llrs, dtype=np.float64)) if macd_llrs else 0.0, -self.max_log_odds, self.max_log_odds)),
            timeframe_probabilities=tf_probabilities,
            timeframe_weights=tf_weights,
            rsi_values=rsi_values,
            macd_histograms=macd_histograms,
            dominant_timeframe=dominant_timeframe,
            reason=reason,
        )
