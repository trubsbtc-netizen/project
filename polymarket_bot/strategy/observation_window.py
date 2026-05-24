"""
Opening observation strategy for BTC 5-minute Polymarket markets.

The opening seconds of each market are treated as a calibration window, not
as an entry window.  After that window closes, this engine estimates the
terminal UP probability with a heavy-tailed Bayesian predictive distribution:

    r_i = log(S_i / S_{i-1})
    r_i | mu, sigma^2 ~ Normal(mu, sigma^2)
    (mu, sigma^2) ~ Normal-Inverse-Gamma(mu_0, kappa_0, alpha_0, beta_0)

The future cumulative log-return is Student-t after marginalizing parameter
uncertainty. Microstructure and EMA-slope evidence are added only as bounded
log-likelihood ratios, so short-horizon noise cannot overwhelm the PTB
probability model.
"""

import time
from dataclasses import replace
from typing import Optional, Tuple

import numpy as np
from scipy.special import ndtr, stdtr

from polymarket_bot.config import BotConfig, Direction
from polymarket_bot.bot_types import (
    FlowMetrics,
    LiquidityMetrics,
    ObservationSignal,
    OrderbookPressure,
    SpreadMetrics,
    TickBuffer,
    VolatilityEstimate,
)


class ObservationWindowStrategy:
    """Compute a validated post-observation terminal direction signal."""

    SLUG_PREFIX = "btc-updown-5m-"

    def __init__(self, config: BotConfig) -> None:
        self.config = config
        self.window_seconds = float(getattr(config, "observation_window_seconds", 40.0))
        self.min_samples = int(getattr(config, "observation_min_samples", 32))
        self.min_coverage = float(getattr(config, "observation_min_coverage", 0.86))
        self.max_gap_seconds = float(getattr(config, "observation_max_gap_seconds", 6.0))
        self.min_validation_score = float(
            getattr(config, "observation_min_validation_score", 0.60)
        )
        self.min_abs_log_odds = float(
            getattr(config, "observation_min_abs_log_odds", 0.45)
        )
        self.min_direction_confidence = float(
            getattr(config, "observation_min_direction_confidence", 0.58)
        )
        self.ema_slope_enabled = bool(
            getattr(config, "observation_ema_slope_enabled", True)
        )
        self.ema_slope_fusion_weight = float(np.clip(
            getattr(config, "observation_ema_slope_fusion_weight", 0.42),
            0.0,
            0.70,
        ))
        self.ema_slope_max_log_odds = float(np.clip(
            getattr(config, "observation_ema_slope_max_log_odds", 2.4),
            0.4,
            6.0,
        ))
        self.ema_slope_min_quality = float(np.clip(
            getattr(config, "observation_ema_slope_min_quality", 0.38),
            0.0,
            0.95,
        ))
        self._final_signals: dict[str, ObservationSignal] = {}
        self._default_sigma = 6.0e-5

    @classmethod
    def market_start_from_slug(cls, slug: str) -> Optional[float]:
        if not slug or not slug.startswith(cls.SLUG_PREFIX):
            return None
        try:
            return float(int(slug[len(cls.SLUG_PREFIX):]))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _sigmoid(log_odds: float) -> float:
        if log_odds >= 0:
            z = np.exp(-log_odds)
            return float(1.0 / (1.0 + z))
        z = np.exp(log_odds)
        return float(z / (1.0 + z))

    @staticmethod
    def _logit(probability: float) -> float:
        p = float(np.clip(probability, 1e-8, 1.0 - 1e-8))
        return float(np.log(p / (1.0 - p)))

    @staticmethod
    def _normal_cdf(value: float) -> float:
        return float(ndtr(value))

    @staticmethod
    def _weighted_quantile(
        values: np.ndarray,
        weights: np.ndarray,
        quantile: float,
    ) -> float:
        if len(values) == 0:
            return 0.0
        order = np.argsort(values)
        sorted_values = values[order]
        sorted_weights = weights[order]
        total_weight = float(np.sum(sorted_weights))
        if total_weight <= 1.0e-12:
            return float(np.quantile(values, quantile))
        cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
        cumulative = np.clip(cumulative / total_weight, 0.0, 1.0)
        return float(np.interp(float(quantile), cumulative, sorted_values))

    @classmethod
    def _weighted_median(cls, values: np.ndarray, weights: np.ndarray) -> float:
        return cls._weighted_quantile(values, weights, 0.5)

    @staticmethod
    def _weighted_autocorr_lag1(values: np.ndarray, weights: np.ndarray) -> float:
        if len(values) < 4:
            return 0.0
        x0 = values[:-1]
        x1 = values[1:]
        pair_weights = np.sqrt(np.maximum(weights[:-1], 0.0) * np.maximum(weights[1:], 0.0))
        weight_sum = float(np.sum(pair_weights))
        if weight_sum <= 1.0e-12:
            return 0.0
        mean0 = float(np.sum(pair_weights * x0) / weight_sum)
        mean1 = float(np.sum(pair_weights * x1) / weight_sum)
        cov = float(np.sum(pair_weights * (x0 - mean0) * (x1 - mean1)) / weight_sum)
        var0 = float(np.sum(pair_weights * (x0 - mean0) ** 2) / weight_sum)
        var1 = float(np.sum(pair_weights * (x1 - mean1) ** 2) / weight_sum)
        denom = np.sqrt(max(var0 * var1, 1.0e-24))
        return float(np.clip(cov / denom, -0.95, 0.95))

    @staticmethod
    def _atanh_clip(value: float, limit: float = 0.985) -> float:
        # Fisher z-transform: maps bounded imbalance/correlation evidence into log-odds space.
        x = float(np.clip(value, -limit, limit))
        return 0.5 * float(np.log((1.0 + x) / (1.0 - x)))

    @staticmethod
    def _ema(values: np.ndarray, span: int) -> np.ndarray:
        if len(values) == 0:
            return values
        alpha = 2.0 / (float(max(span, 1)) + 1.0)
        out = np.empty_like(values, dtype=np.float64)
        out[0] = float(values[0])
        for idx in range(1, len(values)):
            out[idx] = alpha * float(values[idx]) + (1.0 - alpha) * out[idx - 1]
        return out

    @staticmethod
    def _linear_slope(values: np.ndarray) -> float:
        if len(values) < 2:
            return 0.0
        y = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(y)):
            y = y[np.isfinite(y)]
        if len(y) < 2:
            return 0.0
        x = np.arange(len(y), dtype=np.float64)
        x -= float(np.mean(x))
        y = y - float(np.mean(y))
        denom = float(np.dot(x, x))
        if denom <= 1.0e-12:
            return 0.0
        return float(np.dot(x, y) / denom)

    @staticmethod
    def _bounded(value: float, scale: float) -> float:
        if scale <= 0.0 or not np.isfinite(scale):
            return 0.0
        return float(np.tanh(float(value) / float(scale)))

    def _multi_speed_ema_slope_signal(
        self,
        prices: np.ndarray,
        *,
        prior_sigma: float,
        time_to_settlement: float,
    ) -> dict:
        neutral = {
            "is_valid": False,
            "log_odds": 0.0,
            "confidence": 0.0,
            "quality": 0.0,
            "velocity": 0.0,
            "acceleration": 0.0,
            "curvature": 0.0,
            "spread_acceleration": 0.0,
            "turning_pressure": 1.0,
            "coherence": 0.0,
            "span_count": 0,
        }
        if not self.ema_slope_enabled or len(prices) < 16:
            return neutral
        valid = np.isfinite(prices) & (prices > 0.0)
        prices = prices[valid]
        if len(prices) < 16:
            return neutral

        log_prices = np.log(prices.astype(np.float64))
        returns = np.diff(log_prices)
        if len(returns) < 8:
            return neutral
        realized_scale = float(np.median(np.abs(returns - np.median(returns))) * 1.4826)
        scale = max(float(prior_sigma), realized_scale, self._default_sigma * 0.35, 1.0e-8)
        spans = (4, 6, 9, 13, 21, 34)
        records = []

        for span in spans:
            if len(log_prices) < max(16, span + 6):
                continue
            ema = self._ema(log_prices, span)
            segment = int(np.clip(round(max(4.0, span * 0.35)), 4, max(4, len(ema) // 3)))
            if len(ema) < segment * 3:
                continue

            early = ema[-3 * segment : -2 * segment]
            middle = ema[-2 * segment : -segment]
            late = ema[-segment:]
            v0 = self._linear_slope(early)
            v1 = self._linear_slope(middle)
            v2 = self._linear_slope(late)
            velocity = float(v2)
            acceleration = float(v2 - v1)
            curvature = float(v2 - 2.0 * v1 + v0)
            turning_raw = max(0.0, -velocity * acceleration)
            turning_pressure = float(np.clip(
                turning_raw / max(abs(velocity) + abs(acceleration) + scale, scale),
                0.0,
                1.0,
            ))

            velocity_score = self._bounded(velocity, 2.05 * scale)
            acceleration_score = self._bounded(acceleration, 2.45 * scale)
            curvature_score = self._bounded(curvature, 2.95 * scale)
            base_span_score = (
                0.54 * velocity_score
                + 0.28 * acceleration_score
                + 0.18 * curvature_score
            )
            span_score = base_span_score * (1.0 - 0.45 * turning_pressure)
            horizon_match = np.exp(
                -abs(np.log(max(float(span), 2.0) / max(float(time_to_settlement), 5.0))) / 1.45
            )
            speed_weight = 1.0 / np.sqrt(float(span))
            records.append({
                "span": float(span),
                "weight": float((0.55 + 0.45 * horizon_match) * speed_weight),
                "score": float(span_score),
                "velocity": velocity,
                "acceleration": acceleration,
                "curvature": curvature,
                "turning_pressure": turning_pressure,
                "velocity_score": velocity_score,
                "acceleration_score": acceleration_score,
                "curvature_score": curvature_score,
            })

        if len(records) < 3:
            return neutral

        weights = np.array([row["weight"] for row in records], dtype=np.float64)
        weights /= max(float(np.sum(weights)), 1.0e-12)
        scores = np.array([row["score"] for row in records], dtype=np.float64)
        velocities = np.array([row["velocity"] for row in records], dtype=np.float64)
        accelerations = np.array([row["acceleration"] for row in records], dtype=np.float64)
        curvatures = np.array([row["curvature"] for row in records], dtype=np.float64)
        turning = np.array([row["turning_pressure"] for row in records], dtype=np.float64)

        velocity_score = float(np.sum(weights * np.array([row["velocity_score"] for row in records])))
        acceleration_score = float(np.sum(weights * np.array([row["acceleration_score"] for row in records])))
        curvature_score = float(np.sum(weights * np.array([row["curvature_score"] for row in records])))
        turning_pressure = float(np.clip(np.sum(weights * turning), 0.0, 1.0))

        order = np.argsort([row["span"] for row in records])
        fast = order[0]
        slow = order[-1]
        spread_velocity = float(velocities[fast] - velocities[slow])
        spread_acceleration = float(accelerations[fast] - accelerations[slow])
        spread_curvature = float(curvatures[fast] - curvatures[slow])
        spread_velocity_score = self._bounded(spread_velocity, 2.10 * scale)
        spread_acceleration_score = self._bounded(spread_acceleration, 2.45 * scale)
        spread_curvature_score = self._bounded(spread_curvature, 2.95 * scale)

        raw_log_odds = (
            1.24 * velocity_score
            + 0.92 * acceleration_score
            + 0.58 * curvature_score
            + 0.70 * spread_velocity_score
            + 0.72 * spread_acceleration_score
            + 0.38 * spread_curvature_score
        )
        raw_log_odds *= 1.0 - 0.55 * turning_pressure
        raw_log_odds = float(np.clip(
            raw_log_odds,
            -self.ema_slope_max_log_odds,
            self.ema_slope_max_log_odds,
        ))

        dominant_sign = 1.0 if raw_log_odds >= 0.0 else -1.0
        sign_strength = np.maximum(np.abs(scores), 1.0e-6)
        score_signs = np.sign(scores)
        coherence = float(
            np.sum(weights * sign_strength * (score_signs == dominant_sign))
            / max(float(np.sum(weights * sign_strength)), 1.0e-12)
        )
        magnitude = float(np.clip(abs(raw_log_odds) / max(self.ema_slope_max_log_odds, 1.0e-6), 0.0, 1.0))
        spread_support = float(
            np.clip(
                0.50 * abs(spread_velocity_score)
                + 0.35 * abs(spread_acceleration_score)
                + 0.15 * abs(spread_curvature_score),
                0.0,
                1.0,
            )
        )
        quality = float(np.clip(
            0.34 * coherence
            + 0.24 * magnitude
            + 0.20 * (1.0 - turning_pressure)
            + 0.14 * spread_support
            + 0.08 * min(len(records) / len(spans), 1.0),
            0.0,
            1.0,
        ))
        is_valid = bool(
            quality >= self.ema_slope_min_quality
            and coherence >= 0.58
            and abs(raw_log_odds) >= 0.10
            and turning_pressure <= 0.74
        )
        return {
            "is_valid": is_valid,
            "log_odds": raw_log_odds,
            "confidence": quality,
            "quality": quality,
            "velocity": velocity_score,
            "acceleration": acceleration_score,
            "curvature": curvature_score,
            "spread_acceleration": spread_acceleration_score,
            "turning_pressure": turning_pressure,
            "coherence": coherence,
            "span_count": len(records),
        }

    def _prior_sigma(self, volatility_estimate: Optional[VolatilityEstimate]) -> float:
        if volatility_estimate is None:
            return self._default_sigma
        weighted_candidates = [
            (volatility_estimate.realized_volatility, 0.42),
            (volatility_estimate.garch_volatility, 0.22),
            (volatility_estimate.kalman_volatility, 0.18),
            (volatility_estimate.regime_adjusted_volatility, 0.18),
        ]
        valid_values = []
        valid_weights = []
        for value, weight in weighted_candidates:
            if value is None:
                continue
            parsed = float(value)
            if np.isfinite(parsed) and parsed > 0.0:
                valid_values.append(parsed)
                valid_weights.append(float(weight))
        if not valid_values:
            return self._default_sigma
        valid = np.array(valid_values, dtype=np.float64)
        weights = np.array(valid_weights, dtype=np.float64)
        center = self._weighted_median(valid, weights)
        floor = max(self._default_sigma * 0.35, center * 0.35, 1.0e-6)
        capped = np.clip(valid, floor, max(center * 4.0, floor * 2.0))
        log_sigma = float(np.sum(weights * np.log(capped)) / max(float(np.sum(weights)), 1.0e-12))
        blended = 0.70 * float(np.exp(log_sigma)) + 0.30 * center
        return float(np.clip(blended, 1.0e-6, 0.005))

    def _empty_signal(
        self,
        *,
        now: float,
        slug: str,
        market_start: float,
        observed_seconds: float,
        ready: bool,
        reason: str,
    ) -> ObservationSignal:
        return ObservationSignal(
            timestamp=now,
            market_slug=slug,
            market_start_timestamp=market_start,
            observation_start_timestamp=market_start,
            observation_end_timestamp=market_start + max(self.window_seconds, observed_seconds),
            observed_seconds=float(max(0.0, observed_seconds)),
            required_seconds=self.window_seconds,
            sample_count=0,
            effective_sample_size=0.0,
            coverage_ratio=0.0,
            max_gap_seconds=0.0,
            outlier_fraction=1.0,
            is_ready=ready,
            is_valid=False,
            direction=Direction.NEUTRAL,
            p_up=0.5,
            p_down=0.5,
            confidence=0.0,
            validation_score=0.0,
            uncertainty=0.5,
            terminal_z_score=0.0,
            drift_per_second=0.0,
            sigma_per_sqrt_second=0.0,
            student_t_df=0.0,
            terminal_log_odds=0.0,
            microstructure_log_odds=0.0,
            combined_log_odds=0.0,
            reason=reason,
        )

    def _extract_ticks(
        self,
        tick_buffer: TickBuffer,
        start_time: float,
        end_time: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if tick_buffer.len() < 2:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        timestamps = np.array(tick_buffer.timestamps, dtype=np.float64)
        prices = np.array(tick_buffer.prices, dtype=np.float64)
        boundary_slack = max(2.0, float(self.max_gap_seconds))
        mask = (
            np.isfinite(timestamps)
            & np.isfinite(prices)
            & (timestamps >= start_time - boundary_slack)
            & (timestamps <= end_time + boundary_slack)
            & (prices > 0.0)
        )
        timestamps = timestamps[mask]
        prices = prices[mask]
        if len(timestamps) < 2:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        order = np.argsort(timestamps)
        timestamps = timestamps[order]
        prices = prices[order]

        unique_timestamps = []
        unique_prices = []
        for timestamp, price in zip(timestamps, prices):
            if unique_timestamps and abs(timestamp - unique_timestamps[-1]) < 1.0e-6:
                unique_prices[-1] = float(price)
            else:
                unique_timestamps.append(float(timestamp))
                unique_prices.append(float(price))

        timestamps = np.array(unique_timestamps, dtype=np.float64)
        prices = np.array(unique_prices, dtype=np.float64)

        def price_at(target: float) -> Optional[float]:
            idx = int(np.searchsorted(timestamps, target))
            max_nearest_skew = min(max(float(self.max_gap_seconds), 0.25), 2.0)
            if idx < len(timestamps) and abs(float(timestamps[idx]) - target) <= 1.0e-6:
                return float(prices[idx])
            before_idx = idx - 1 if idx > 0 else None
            after_idx = idx if idx < len(timestamps) else None
            if before_idx is not None and after_idx is not None:
                t0 = float(timestamps[before_idx])
                t1 = float(timestamps[after_idx])
                if t0 <= target <= t1 and t1 - t0 <= self.max_gap_seconds:
                    weight = (target - t0) / max(t1 - t0, 1.0e-9)
                    return float(prices[before_idx] + weight * (prices[after_idx] - prices[before_idx]))
            candidates = []
            if before_idx is not None:
                candidates.append((abs(target - float(timestamps[before_idx])), float(prices[before_idx])))
            if after_idx is not None:
                candidates.append((abs(float(timestamps[after_idx]) - target), float(prices[after_idx])))
            if candidates:
                skew, price = min(candidates, key=lambda item: item[0])
                if skew <= max_nearest_skew:
                    return price
            return None

        window_timestamps = []
        window_prices = []
        start_price = price_at(float(start_time))
        if start_price is not None:
            window_timestamps.append(float(start_time))
            window_prices.append(start_price)

        interior = (
            (timestamps > start_time + 1.0e-6)
            & (timestamps < end_time - 1.0e-6)
        )
        for timestamp, price in zip(timestamps[interior], prices[interior]):
            window_timestamps.append(float(timestamp))
            window_prices.append(float(price))

        end_price = price_at(float(end_time))
        if end_price is not None:
            window_timestamps.append(float(end_time))
            window_prices.append(end_price)

        if len(window_timestamps) < 2:
            interior_or_boundary = (
                (timestamps >= start_time)
                & (timestamps <= end_time)
            )
            window_timestamps = [float(value) for value in timestamps[interior_or_boundary]]
            window_prices = [float(value) for value in prices[interior_or_boundary]]

        return (
            np.array(window_timestamps, dtype=np.float64),
            np.array(window_prices, dtype=np.float64),
        )

    def _cutoff_time_to_settlement(
        self,
        market_start: float,
        observation_end: float,
        live_time_to_settlement: float,
        now: float,
    ) -> float:
        interval = float(
            getattr(getattr(self.config, "market", None), "settlement_interval_seconds", 300.0)
            or 300.0
        )
        if np.isfinite(interval) and interval > self.window_seconds:
            market_end = market_start + interval
            return float(max(market_end - observation_end, 0.25))
        elapsed_since_cutoff = max(0.0, float(now) - float(observation_end))
        return float(max(float(live_time_to_settlement) + elapsed_since_cutoff, 0.25))

    def _metric_within_observation(
        self,
        metric: Optional[object],
        market_start: float,
        observation_end: float,
    ) -> Optional[object]:
        if metric is None:
            return None
        timestamp = getattr(metric, "timestamp", None)
        if timestamp is None:
            return None
        try:
            ts = float(timestamp)
        except (TypeError, ValueError):
            return None
        slack = min(max(float(self.max_gap_seconds), 0.25), 2.0)
        if not np.isfinite(ts):
            return None
        if market_start - slack <= ts <= observation_end + slack:
            return metric
        return None

    def _cache_final_signal(self, signal: ObservationSignal) -> ObservationSignal:
        data_quality_complete = (
            signal.is_ready
            and signal.sample_count > 0
            and signal.validation_score >= self.min_validation_score
            and signal.coverage_ratio >= self.min_coverage
            and signal.max_gap_seconds <= self.max_gap_seconds
            and "samples<" not in signal.reason
            and "coverage<" not in signal.reason
            and "gap>" not in signal.reason
            and "validation<" not in signal.reason
        )
        if data_quality_complete:
            self._final_signals[signal.market_slug] = signal
            if len(self._final_signals) > 64:
                keys = list(self._final_signals.keys())
                for stale_key in keys[:-32]:
                    self._final_signals.pop(stale_key, None)
        return signal

    def _build_second_returns(
        self,
        timestamps: np.ndarray,
        prices: np.ndarray,
        start_time: float,
        end_time: float,
    ) -> Tuple[np.ndarray, np.ndarray, float, float, float, float]:
        if len(timestamps) < 2:
            return (
                np.array([], dtype=np.float64),
                np.array([], dtype=np.float64),
                0.0,
                0.0,
                1.0,
                0.0,
            )

        observed_start = max(float(timestamps[0]), start_time)
        observed_end = min(float(timestamps[-1]), end_time)
        observed_seconds = max(0.0, observed_end - observed_start)
        required_seconds = max(float(self.window_seconds), 1.0)
        coverage_ratio = float(np.clip(observed_seconds / required_seconds, 0.0, 1.0))
        gaps = np.diff(timestamps)
        max_gap = float(np.max(gaps)) if len(gaps) else 0.0

        grid_start = float(np.ceil(observed_start))
        grid_end = float(np.floor(observed_end))
        if grid_end - grid_start >= 4.0:
            grid = np.arange(grid_start, grid_end + 1.0, 1.0, dtype=np.float64)
            sampled_prices = np.interp(grid, timestamps, prices)
            log_prices = np.log(sampled_prices)
            raw_returns = np.diff(log_prices)
            interval_end = grid[1:]
        else:
            dt = np.diff(timestamps)
            valid_dt = (dt >= 0.05) & (dt <= self.max_gap_seconds)
            raw_returns = np.diff(np.log(prices))[valid_dt] / np.maximum(dt[valid_dt], 1.0e-6)
            interval_end = timestamps[1:][valid_dt]

        if len(raw_returns) == 0:
            return raw_returns, interval_end, observed_seconds, coverage_ratio, max_gap, 0.0

        center = float(np.median(raw_returns))
        mad = float(np.median(np.abs(raw_returns - center)))
        robust_scale = max(1.4826 * mad, self._default_sigma, 1.0e-8)
        standardized = np.abs(raw_returns - center) / robust_scale
        adaptive_abs_cap = max(0.02, 6.0 * robust_scale)
        valid = (standardized <= 8.0) & (np.abs(raw_returns) <= adaptive_abs_cap)
        outlier_count = int(len(raw_returns) - np.count_nonzero(valid))
        returns = raw_returns[valid]
        interval_end = interval_end[valid]
        outlier_fraction = outlier_count / max(len(raw_returns), 1)

        return (
            returns,
            interval_end,
            observed_seconds,
            coverage_ratio,
            max_gap,
            float(outlier_fraction),
        )

    def _fit_terminal_student_t(
        self,
        *,
        returns: np.ndarray,
        interval_end: np.ndarray,
        btc_price: float,
        price_to_beat: float,
        time_to_settlement: float,
        prior_sigma: float,
    ) -> Tuple[float, float, float, float, float, float, float, float]:
        if len(returns) < 3:
            return 0.5, 0.0, 0.0, prior_sigma, 0.0, 0.0, 0.0, 0.0

        center = float(np.median(returns))
        mad = float(np.median(np.abs(returns - center)))
        robust_scale = max(1.4826 * mad, prior_sigma * 0.65, 1.0e-8)
        u = (returns - center) / (4.685 * robust_scale)
        tukey = np.where(np.abs(u) < 1.0, (1.0 - u * u) ** 2, 0.0)

        half_life = max(8.0, min(24.0, self.window_seconds * 0.42))
        recency = np.exp(-(float(interval_end[-1]) - interval_end) / half_life)
        weights = tukey * recency
        if float(np.sum(weights)) <= 1.0e-9:
            weights = recency

        weight_sum = float(np.sum(weights))
        raw_n_eff = float((weight_sum ** 2) / max(float(np.sum(weights * weights)), 1.0e-12))
        autocorr = max(0.0, self._weighted_autocorr_lag1(returns, weights))
        n_eff = float(raw_n_eff * (1.0 - autocorr) / max(1.0 + autocorr, 1.0e-9))
        n_eff = float(np.clip(n_eff, 1.0, raw_n_eff))

        q01 = self._weighted_quantile(returns, weights, 0.01)
        q99 = self._weighted_quantile(returns, weights, 0.99)
        clipped_returns = np.clip(returns, q01, q99)
        median_return = self._weighted_median(clipped_returns, weights)
        mean_return_raw = float(np.sum(weights * clipped_returns) / max(weight_sum, 1.0e-12))
        mean_return = float(0.72 * mean_return_raw + 0.28 * median_return)
        variance_w = float(np.sum(weights * (clipped_returns - mean_return) ** 2) / max(weight_sum, 1.0e-12))
        variance_floor = (prior_sigma * (1.0 + 0.35 * autocorr)) ** 2
        variance_w = max(variance_w, variance_floor, 1.0e-16)
        sse_eff = max(variance_w * max(n_eff - 1.0, 1.0), 1.0e-14)

        mu0 = 0.0
        kappa0 = 6.0
        alpha0 = 4.0
        beta0 = (alpha0 - 1.0) * (prior_sigma ** 2)

        kappa_n = kappa0 + n_eff
        alpha_n = alpha0 + 0.5 * n_eff
        beta_n = (
            beta0
            + 0.5 * sse_eff
            + (kappa0 * n_eff * (mean_return - mu0) ** 2) / (2.0 * kappa_n)
        )
        mu_n = (kappa0 * mu0 + n_eff * mean_return) / kappa_n

        drift_se = np.sqrt(max(beta_n / (alpha_n * kappa_n), 1.0e-16))
        drift_shrink = n_eff / (n_eff + 14.0)
        drift_shrink *= abs(mu_n) / (abs(mu_n) + 3.5 * drift_se + 1.0e-12)
        mu_n = float(mu_n * np.clip(drift_shrink, 0.0, 1.0))

        tau = max(float(time_to_settlement), 0.25)
        df = max(2.1, 2.0 * alpha_n)
        posterior_variance = max(beta_n / alpha_n, 1.0e-16)
        autocorr_inflation = 1.0 + 0.65 * autocorr
        scale = float(np.sqrt(posterior_variance * (tau + (tau * tau) / kappa_n) * autocorr_inflation))
        loc = float(mu_n * tau)

        threshold = float(np.log1p((price_to_beat - btc_price) / btc_price))
        if scale <= 1.0e-12 or not np.isfinite(scale):
            z = (loc - threshold) / max(prior_sigma * np.sqrt(tau), 1.0e-12)
            p_up = self._normal_cdf(float(z))
        else:
            z_t = (threshold - loc) / scale
            p_up = 1.0 - float(stdtr(df, z_t))

        sigma = float(np.sqrt(posterior_variance))
        terminal_z = float((loc - threshold) / max(scale, 1.0e-12))
        return (
            float(np.clip(p_up, 0.001, 0.999)),
            terminal_z,
            float(mu_n),
            sigma,
            df,
            n_eff,
            loc,
            scale,
        )

    def _microstructure_log_odds(
        self,
        *,
        btc_price: float,
        prior_sigma: float,
        validation_score: float,
        flow_metrics: Optional[FlowMetrics],
        orderbook_pressure: Optional[OrderbookPressure],
        spread_metrics: Optional[SpreadMetrics],
        liquidity_metrics: Optional[LiquidityMetrics],
    ) -> float:
        flow_llr = 0.0
        if flow_metrics is not None and flow_metrics.total_volume > 0.0:
            persistence = float(np.clip(flow_metrics.flow_persistence, 0.0, 1.0))
            volume_info = np.sqrt(float(np.clip(flow_metrics.trade_count / 120.0, 0.0, 1.0)))
            aggression = float(np.clip(flow_metrics.taker_aggression_score, 0.0, 1.0))
            flow_llr = (
                self._atanh_clip(flow_metrics.flow_imbalance)
                * (0.25 + 0.75 * persistence)
                * (0.50 + 0.50 * aggression)
                * volume_info
            )

        pressure_llr = 0.0
        microprice_llr = 0.0
        anti_spoof = 1.0
        if orderbook_pressure is not None:
            anti_spoof = float(np.clip(
                (1.0 - orderbook_pressure.spoof_score)
                * (1.0 - orderbook_pressure.fake_liquidity_ratio),
                0.0,
                1.0,
            ))
            pressure_llr = self._atanh_clip(orderbook_pressure.real_imbalance) * anti_spoof
            if (
                orderbook_pressure.microprice is not None
                and orderbook_pressure.microprice > 0.0
                and btc_price > 0.0
            ):
                micro_edge = np.log(orderbook_pressure.microprice / btc_price)
                micro_scale = max(prior_sigma * np.sqrt(5.0), 1.0e-7)
                microprice_llr = float(np.clip(micro_edge / micro_scale, -2.0, 2.0))
                microprice_llr *= 0.35 * anti_spoof

        liquidity_llr = 0.0
        if liquidity_metrics is not None:
            depth_ratio = float(np.clip(liquidity_metrics.depth_ratio, 1.0e-4, 1.0e4))
            stability = float(np.clip(liquidity_metrics.liquidity_stability_score, 0.0, 1.0))
            real_fraction = float(np.clip(liquidity_metrics.real_liquidity_fraction, 0.0, 1.0))
            liquidity_llr = 0.25 * np.clip(np.log(depth_ratio), -2.0, 2.0)
            liquidity_llr *= 0.5 + 0.5 * min(stability, real_fraction)

        spread_quality = 1.0
        if spread_metrics is not None:
            spread_quality = float(np.exp(-max(spread_metrics.spread_bps, 0.0) / 12.0))
            spread_quality = float(np.clip(spread_quality, 0.15, 1.0))

        raw_llr = (
            0.38 * flow_llr
            + 0.32 * pressure_llr
            + 0.18 * microprice_llr
            + 0.12 * liquidity_llr
        )
        quality = validation_score * spread_quality * (0.50 + 0.50 * anti_spoof)
        return float(np.clip(raw_llr * quality, -2.5, 2.5))

    def evaluate(
        self,
        *,
        market_slug: str,
        tick_buffer: TickBuffer,
        btc_price: float,
        price_to_beat: float,
        time_to_settlement: float,
        flow_metrics: Optional[FlowMetrics] = None,
        orderbook_pressure: Optional[OrderbookPressure] = None,
        spread_metrics: Optional[SpreadMetrics] = None,
        liquidity_metrics: Optional[LiquidityMetrics] = None,
        volatility_estimate: Optional[VolatilityEstimate] = None,
        now: Optional[float] = None,
    ) -> ObservationSignal:
        now = time.time() if now is None else float(now)
        market_start = self.market_start_from_slug(market_slug)
        if market_start is None:
            return self._empty_signal(
                now=now,
                slug=market_slug,
                market_start=0.0,
                observed_seconds=0.0,
                ready=False,
                reason="invalid_market_slug",
            )

        minimum_observation_end = market_start + self.window_seconds
        cached_signal = self._final_signals.get(market_slug)
        if cached_signal is not None and now >= minimum_observation_end:
            return replace(cached_signal, timestamp=now)

        observation_end = minimum_observation_end
        observed_cutoff = min(now, observation_end)
        observed_seconds_clock = max(0.0, observed_cutoff - market_start)
        ready = now >= minimum_observation_end
        if not ready:
            return self._empty_signal(
                now=now,
                slug=market_slug,
                market_start=market_start,
                observed_seconds=observed_seconds_clock,
                ready=False,
                reason=f"observation_pending({observed_seconds_clock:.1f}/{self.window_seconds:.0f}s)",
            )

        if btc_price <= 0.0 or price_to_beat <= 0.0:
            return self._empty_signal(
                now=now,
                slug=market_slug,
                market_start=market_start,
                observed_seconds=observed_seconds_clock,
                ready=True,
                reason="invalid_price_or_ptb",
            )

        timestamps, prices = self._extract_ticks(tick_buffer, market_start, observation_end)
        (
            returns,
            interval_end,
            observed_seconds,
            coverage_ratio,
            max_gap,
            outlier_fraction,
        ) = self._build_second_returns(timestamps, prices, market_start, observation_end)
        sample_count = int(len(returns))

        if sample_count == 0:
            reason = (
                "observation_window_not_recorded"
                if now > minimum_observation_end + self.max_gap_seconds
                else "no_observation_ticks"
            )
            return self._empty_signal(
                now=now,
                slug=market_slug,
                market_start=market_start,
                observed_seconds=observed_seconds_clock,
                ready=True,
                reason=reason,
            )

        cutoff_price = float(prices[-1]) if len(prices) else float(btc_price)
        if not np.isfinite(cutoff_price) or cutoff_price <= 0.0:
            return self._empty_signal(
                now=now,
                slug=market_slug,
                market_start=market_start,
                observed_seconds=observed_seconds_clock,
                ready=True,
                reason="invalid_cutoff_price",
            )
        cutoff_tau = self._cutoff_time_to_settlement(
            market_start=market_start,
            observation_end=observation_end,
            live_time_to_settlement=time_to_settlement,
            now=now,
        )
        cutoff_flow_metrics = self._metric_within_observation(
            flow_metrics,
            market_start,
            observation_end,
        )
        cutoff_orderbook_pressure = self._metric_within_observation(
            orderbook_pressure,
            market_start,
            observation_end,
        )
        cutoff_spread_metrics = self._metric_within_observation(
            spread_metrics,
            market_start,
            observation_end,
        )
        cutoff_liquidity_metrics = self._metric_within_observation(
            liquidity_metrics,
            market_start,
            observation_end,
        )

        possible_samples = max(1, int(np.floor(min(observed_seconds, self.window_seconds))) - 1)
        adaptive_min_samples = int(np.clip(
            min(self.min_samples, possible_samples),
            8,
            max(self.min_samples, 8),
        ))
        sample_score = 1.0 - np.exp(-sample_count / max(float(adaptive_min_samples), 1.0))
        gap_score = float(np.exp(-max(0.0, max_gap - 2.0) / max(self.max_gap_seconds, 1.0)))
        coverage_score = float(np.clip(coverage_ratio / max(self.min_coverage, 1.0e-6), 0.0, 1.0))
        outlier_score = float(np.clip(1.0 - outlier_fraction, 0.0, 1.0))
        validation_components = [
            max(1.0e-3, sample_score),
            max(1.0e-3, gap_score),
            max(1.0e-3, coverage_score),
            max(1.0e-3, outlier_score),
        ]
        validation_score = float(np.exp(np.mean(np.log(validation_components))))

        data_invalid_reasons = []
        if sample_count < adaptive_min_samples:
            data_invalid_reasons.append(f"samples<{adaptive_min_samples}")
        if coverage_ratio < self.min_coverage:
            data_invalid_reasons.append(f"coverage<{self.min_coverage:.2f}")
        if max_gap > self.max_gap_seconds:
            data_invalid_reasons.append(f"gap>{self.max_gap_seconds:.1f}s")
        if validation_score < self.min_validation_score:
            data_invalid_reasons.append(f"validation<{self.min_validation_score:.2f}")

        prior_sigma = self._prior_sigma(volatility_estimate)
        ema_slope = self._multi_speed_ema_slope_signal(
            prices,
            prior_sigma=prior_sigma,
            time_to_settlement=cutoff_tau,
        )
        ema_log_odds = float(ema_slope["log_odds"])
        ema_weight = 0.0
        ema_agreement = 0.5
        ema_component = 0.5
        (
            p_terminal,
            terminal_z,
            drift,
            sigma,
            df,
            n_eff,
            _predictive_loc,
            predictive_scale,
        ) = self._fit_terminal_student_t(
            returns=returns,
            interval_end=interval_end,
            btc_price=cutoff_price,
            price_to_beat=price_to_beat,
            time_to_settlement=cutoff_tau,
            prior_sigma=prior_sigma,
        )

        student_log_odds = self._logit(p_terminal)
        terminal_log_odds = student_log_odds
        terminal_z_out = float(terminal_z)
        drift_out = float(drift)
        sigma_out = float(sigma)
        pre_ema_terminal_log_odds = float(terminal_log_odds)
        if bool(ema_slope["is_valid"]) and self.ema_slope_fusion_weight > 0.0:
            ema_quality = float(ema_slope["quality"])
            ema_weight = float(np.clip(
                self.ema_slope_fusion_weight
                * (0.45 + 0.55 * validation_score * ema_quality),
                0.0,
                self.ema_slope_fusion_weight,
            ))
            base_weight = 1.0 - ema_weight
            terminal_log_odds = (
                base_weight * float(terminal_log_odds)
                + ema_weight * ema_log_odds
            )
            ema_agreement = float(np.exp(-abs(ema_log_odds - pre_ema_terminal_log_odds) / 3.5))
            ema_component = float(np.clip(0.35 + 0.65 * ema_quality, 0.01, 1.0))

        micro_llr = self._microstructure_log_odds(
            btc_price=cutoff_price,
            prior_sigma=prior_sigma,
            validation_score=validation_score,
            flow_metrics=cutoff_flow_metrics,
            orderbook_pressure=cutoff_orderbook_pressure,
            spread_metrics=cutoff_spread_metrics,
            liquidity_metrics=cutoff_liquidity_metrics,
        )
        n_eff_weight = float(np.clip(n_eff / (n_eff + 12.0), 0.0, 1.0))
        terminal_weight = float(np.clip(0.55 + 0.45 * validation_score * n_eff_weight, 0.50, 1.0))
        micro_weight = float(np.clip(0.10 + 0.30 * validation_score * n_eff_weight, 0.0, 0.40))

        raw_combined_log_odds = terminal_weight * terminal_log_odds + micro_weight * micro_llr
        combined_log_odds = raw_combined_log_odds
        p_up = float(np.clip(self._sigmoid(combined_log_odds), 0.001, 0.999))
        p_down = 1.0 - p_up

        probability_entropy = 0.0
        if 0.0 < p_up < 1.0:
            probability_entropy = -(
                p_up * np.log(p_up) + p_down * np.log(p_down)
            ) / np.log(2.0)
        direction_certainty = abs(p_up - 0.5) * 2.0
        log_odds_evidence = 1.0 - np.exp(-abs(combined_log_odds))
        terminal_z_evidence = 1.0 - np.exp(-abs(terminal_z_out))
        sample_confidence = n_eff / (n_eff + 10.0)
        model_confidence = 1.0 / (
            1.0 + predictive_scale / max(prior_sigma * np.sqrt(max(cutoff_tau, 1.0)), 1.0e-8)
        )
        evidence_component = max(
            direction_certainty,
            log_odds_evidence,
            terminal_z_evidence,
        )
        entropy_component = float(np.clip(1.0 - probability_entropy * 0.35, 0.05, 1.0))
        model_component = float(np.clip(0.55 + 0.45 * model_confidence, 0.05, 1.0))
        confidence_components = [
            (max(1.0e-3, validation_score), 0.22),
            (max(1.0e-3, sample_confidence), 0.17),
            (max(1.0e-3, evidence_component), 0.31),
            (max(1.0e-3, entropy_component), 0.12),
            (max(1.0e-3, model_component), 0.10),
        ]
        if bool(ema_slope["is_valid"]):
            confidence_components.extend([
                (max(1.0e-3, ema_component), 0.04),
                (max(1.0e-3, ema_agreement), 0.04),
            ])
        weight_total = max(sum(weight for _, weight in confidence_components), 1.0e-12)
        confidence = float(np.exp(
            sum(weight * np.log(value) for value, weight in confidence_components)
            / weight_total
        ))
        confidence = float(np.clip(confidence, 0.01, 0.99))
        uncertainty = float(np.clip((1.0 - confidence) * 0.50 + np.sqrt(p_up * p_down / (n_eff + 3.0)), 0.01, 0.50))

        direction_log_odds = abs(combined_log_odds)
        selected_probability = max(p_up, p_down)
        required_log_odds = self.min_abs_log_odds
        min_probability = self._sigmoid(required_log_odds)
        directional_reasons = []
        if direction_log_odds < required_log_odds:
            directional_reasons.append(f"edge<{min_probability:.3f}")
        if (
            int(ema_slope["span_count"]) >= 3
            and float(ema_slope["turning_pressure"]) > 0.82
            and selected_probability < 0.72
        ):
            directional_reasons.append("ema_turning>0.82")
        if (
            bool(ema_slope["is_valid"])
            and np.sign(ema_log_odds) != 0.0
            and np.sign(pre_ema_terminal_log_odds) != 0.0
            and np.sign(ema_log_odds) != np.sign(pre_ema_terminal_log_odds)
            and abs(ema_log_odds) >= 0.35
            and abs(pre_ema_terminal_log_odds) >= 0.35
            and ema_agreement < 0.74
        ):
            directional_reasons.append("ema_terminal_conflict")
        if confidence < self.min_direction_confidence:
            directional_reasons.append(f"conf<{self.min_direction_confidence:.2f}")

        is_valid = len(data_invalid_reasons) == 0
        candidate_direction = (
            Direction.UP
            if p_up >= p_down and p_up > 0.5
            else Direction.DOWN
            if p_down > p_up and p_down > 0.5
            else Direction.NEUTRAL
        )
        direction = candidate_direction if is_valid else Direction.NEUTRAL
        direction_state = "trade_ready" if not directional_reasons else "watch"
        reason = (
            f"valid({direction_state},n_eff={n_eff:.1f},z={terminal_z_out:+.2f},"
            f"cutoff={cutoff_price:.2f},tau0={cutoff_tau:.0f},"
            f"edge_min={min_probability:.3f},"
            f"ema_w={ema_weight:.2f},ev={float(ema_slope['velocity']):+.2f},"
            f"ea={float(ema_slope['acceleration']):+.2f},"
            f"esa={float(ema_slope['spread_acceleration']):+.2f},"
            f"ec={float(ema_slope['curvature']):+.2f},"
            f"etp={float(ema_slope['turning_pressure']):.2f},"
            f"p={selected_probability:.3f}"
            f"{',gate=' + '/'.join(directional_reasons) if directional_reasons else ''})"
            if is_valid
            else ",".join(data_invalid_reasons)
        )
        signal = ObservationSignal(
            timestamp=now,
            market_slug=market_slug,
            market_start_timestamp=market_start,
            observation_start_timestamp=market_start,
            observation_end_timestamp=observation_end,
            observed_seconds=float(observed_seconds),
            required_seconds=self.window_seconds,
            sample_count=sample_count,
            effective_sample_size=float(n_eff),
            coverage_ratio=float(coverage_ratio),
            max_gap_seconds=float(max_gap),
            outlier_fraction=float(outlier_fraction),
            is_ready=True,
            is_valid=is_valid,
            direction=direction,
            p_up=float(p_up),
            p_down=float(p_down),
            confidence=float(confidence),
            validation_score=float(validation_score),
            uncertainty=uncertainty,
            terminal_z_score=float(terminal_z_out),
            drift_per_second=float(drift_out),
            sigma_per_sqrt_second=float(sigma_out),
            student_t_df=float(df),
            terminal_log_odds=float(terminal_log_odds),
            microstructure_log_odds=float(micro_llr),
            combined_log_odds=float(combined_log_odds),
            ema_slope_log_odds=float(ema_log_odds),
            ema_slope_velocity=float(ema_slope["velocity"]),
            ema_spread_acceleration=float(ema_slope["spread_acceleration"]),
            ema_curvature=float(ema_slope["curvature"]),
            ema_turning_pressure=float(ema_slope["turning_pressure"]),
            reason=reason,
        )
        return self._cache_final_signal(signal)
