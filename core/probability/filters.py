from __future__ import annotations

import math
from dataclasses import dataclass

from core.probability.math import clamp, logsumexp, normal_cdf, normal_pdf, softmax


@dataclass(slots=True)
class KalmanDriftFilter:
    process_price_var: float = 2.5e-4
    process_drift_var: float = 2.5e-7
    observation_var_floor: float = 2.5e-6
    price: float = 0.0
    drift: float = 0.0
    p00: float = 1.0
    p01: float = 0.0
    p10: float = 0.0
    p11: float = 1.0
    last_ts_ns: int = 0
    initialized: bool = False

    def update(self, price: float, obs_var: float, ts_ns: int, vol_per_s: float) -> tuple[float, float, float]:
        if not self.initialized:
            self.price = price
            self.drift = 0.0
            self.p00 = max(obs_var, self.observation_var_floor)
            self.p11 = max(vol_per_s * vol_per_s, 1e-8)
            self.p01 = self.p10 = 0.0
            self.last_ts_ns = ts_ns
            self.initialized = True
            return self.price, self.drift, self.p00

        dt = clamp((ts_ns - self.last_ts_ns) * 1e-9, 1e-4, 10.0)
        q_price = self.process_price_var * max(vol_per_s * vol_per_s, 1e-10) * dt
        q_drift = self.process_drift_var * (1.0 + 100.0 * vol_per_s) * dt

        pred_price = self.price + self.drift * dt
        pred_drift = self.drift

        a00 = self.p00 + dt * (self.p10 + self.p01) + dt * dt * self.p11 + q_price
        a01 = self.p01 + dt * self.p11
        a10 = self.p10 + dt * self.p11
        a11 = self.p11 + q_drift

        r = max(self.observation_var_floor, obs_var)
        innovation = price - pred_price
        s = max(1e-12, a00 + r)
        k0 = a00 / s
        k1 = a10 / s

        self.price = pred_price + k0 * innovation
        self.drift = pred_drift + k1 * innovation
        self.p00 = max(1e-12, (1.0 - k0) * a00)
        self.p01 = (1.0 - k0) * a01
        self.p10 = a10 - k1 * a00
        self.p11 = max(1e-12, a11 - k1 * a01)
        self.last_ts_ns = ts_ns
        return self.price, self.drift, self.p00

    def terminal_up_probability(self, barrier: float, seconds: float, vol_per_s: float) -> tuple[float, float]:
        t = max(1e-3, seconds)
        mean = self.price + self.drift * t
        var = (
            self.p00
            + t * t * self.p11
            + 2.0 * t * self.p01
            + max(vol_per_s * vol_per_s * t, 1e-12)
        )
        std = max(1e-6, var**0.5)
        z = (mean - barrier) / std
        p = normal_cdf(z)
        density = normal_pdf(z) / std
        return clamp(p, 1e-5, 1.0 - 1e-5), density


class HMMRegimeFilter:
    DOWN_TREND = 0
    MEAN_REVERT = 1
    NEUTRAL = 2
    UP_TREND = 3
    HIGH_VOL = 4

    def __init__(self) -> None:
        self._logp = [math.log(0.2)] * 5
        self.last_ts_ns = 0

    @property
    def probabilities(self) -> list[float]:
        return softmax(self._logp)

    def update(
        self,
        ret_z: float,
        flow_z: float,
        imbalance_z: float,
        absorption: float,
        exhaustion: float,
        jump: float,
        ts_ns: int,
    ) -> list[float]:
        dt = 1.0 if self.last_ts_ns == 0 else clamp((ts_ns - self.last_ts_ns) * 1e-9, 0.02, 5.0)
        transition = self._transition_matrix(dt, jump, exhaustion)
        pred: list[float] = []
        for j in range(5):
            pred.append(logsumexp([self._logp[i] + math.log(max(1e-12, transition[i][j])) for i in range(5)]))

        emission = self._emission_loglik(ret_z, flow_z, imbalance_z, absorption, exhaustion, jump)
        joint = [pred[i] + emission[i] for i in range(5)]
        norm = logsumexp(joint)
        self._logp = [v - norm for v in joint]
        self.last_ts_ns = ts_ns
        return self.probabilities

    def _transition_matrix(self, dt: float, jump: float, exhaustion: float) -> list[list[float]]:
        persistence = clamp(math.exp(-dt / max(1.0, 25.0 / (1.0 + 2.0 * jump + exhaustion))), 0.40, 0.995)
        flip = clamp(0.012 * dt * (1.0 + 2.5 * exhaustion + jump), 0.001, 0.22)
        to_high = clamp(0.015 * dt * (1.0 + 5.0 * jump), 0.001, 0.35)
        to_neutral = clamp(1.0 - persistence - flip - to_high, 0.0, 0.35)
        m = [[0.0] * 5 for _ in range(5)]
        for i in range(5):
            m[i][i] = persistence
            m[i][self.NEUTRAL] += to_neutral
            m[i][self.HIGH_VOL] += to_high
        m[self.DOWN_TREND][self.UP_TREND] += flip
        m[self.UP_TREND][self.DOWN_TREND] += flip
        m[self.MEAN_REVERT][self.NEUTRAL] += flip
        m[self.NEUTRAL][self.MEAN_REVERT] += flip
        m[self.HIGH_VOL][self.MEAN_REVERT] += flip
        for i, row in enumerate(m):
            s = sum(row)
            if s <= 0.0:
                m[i] = [0.2] * 5
            else:
                m[i] = [v / s for v in row]
        return m

    def _emission_loglik(
        self,
        ret_z: float,
        flow_z: float,
        imbalance_z: float,
        absorption: float,
        exhaustion: float,
        jump: float,
    ) -> list[float]:
        def ll(x: float, mu: float, sigma: float) -> float:
            z = (x - mu) / max(1e-6, sigma)
            return -0.5 * z * z - math.log(max(1e-6, sigma))

        trend_signal = 0.55 * ret_z + 0.35 * flow_z + 0.25 * imbalance_z
        rev_signal = -0.35 * ret_z + 0.25 * absorption + 0.35 * exhaustion
        return [
            ll(trend_signal, -1.15, 1.10) + ll(jump, 0.2, 1.6),
            ll(rev_signal, 0.90, 1.05) + ll(absorption, 0.55, 1.2),
            ll(trend_signal, 0.0, 1.45) + ll(jump, 0.2, 1.4),
            ll(trend_signal, 1.15, 1.10) + ll(jump, 0.2, 1.6),
            ll(jump + exhaustion, 1.25, 1.10) + ll(abs(ret_z), 1.0, 1.5),
        ]

    def trend_score(self) -> float:
        p = self.probabilities
        return p[self.UP_TREND] - p[self.DOWN_TREND]

    def volatility_score(self) -> float:
        p = self.probabilities
        return p[self.HIGH_VOL]

    def mean_reversion_score(self) -> float:
        p = self.probabilities
        return p[self.MEAN_REVERT]


class OnlineBetaCalibrator:
    def __init__(self, bins: int = 25, prior: float = 3.0) -> None:
        self._bins = bins
        self._alpha = [prior] * bins
        self._beta = [prior] * bins

    def _idx(self, p: float) -> int:
        return min(self._bins - 1, max(0, int(clamp(p, 0.0, 0.999999) * self._bins)))

    def calibrate(self, p: float, uncertainty: float) -> float:
        idx = self._idx(p)
        empirical = self._alpha[idx] / (self._alpha[idx] + self._beta[idx])
        shrink = clamp(1.0 - uncertainty, 0.10, 0.85)
        return clamp(shrink * empirical + (1.0 - shrink) * p, 1e-5, 1.0 - 1e-5)

    def update(self, p: float, outcome_up: bool, weight: float = 1.0) -> None:
        idx = self._idx(p)
        w = clamp(weight, 0.0, 10.0)
        if outcome_up:
            self._alpha[idx] += w
        else:
            self._beta[idx] += w

