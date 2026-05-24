from __future__ import annotations

import math
from dataclasses import dataclass

from core.buffering.ring import EWCorrelation
from core.config import StrategyConfig
from core.probability.filters import HMMRegimeFilter, KalmanDriftFilter, OnlineBetaCalibrator
from core.probability.math import binary_entropy, clamp, dot, logit, sigmoid
from core.types import Direction, MicrostructureSnapshot, ProbabilityEstimate


@dataclass(slots=True)
class EvidenceVector:
    continuation: float
    reversal: float
    flow: float
    book: float
    distance: float
    absorption: float
    vacuum: float
    regime: float

    def as_list(self) -> list[float]:
        return [
            self.continuation,
            self.reversal,
            self.flow,
            self.book,
            self.distance,
            self.absorption,
            self.vacuum,
            self.regime,
        ]


class BayesianEvidenceFusion:
    def __init__(self, config: StrategyConfig) -> None:
        self._config = config
        self._kalman = KalmanDriftFilter()
        self._hmm = HMMRegimeFilter()
        self._calibrator = OnlineBetaCalibrator()
        self._corr = EWCorrelation(dim=8, halflife_s=config.covariance_halflife_s)
        self._last_raw_p = 0.5

    def update_settlement_label(self, predicted_p_up: float, outcome_up: bool) -> None:
        self._calibrator.update(predicted_p_up, outcome_up, weight=1.0)

    def estimate(self, snap: MicrostructureSnapshot) -> ProbabilityEstimate:
        truth_price = max(1e-8, snap.truth_price)
        realized_vol = max(1e-8, snap.realized_vol)
        price_vol_per_s = truth_price * realized_vol
        obs_var = max(
            1e-12,
            (truth_price * snap.exchange_divergence) ** 2 + price_vol_per_s * price_vol_per_s * 0.02,
        )
        filtered_price, filtered_drift, filtered_var = self._kalman.update(
            snap.truth_price,
            obs_var=obs_var,
            ts_ns=snap.ts_mono_ns,
            vol_per_s=price_vol_per_s,
        )
        base_p, boundary_density = self._kalman.terminal_up_probability(
            snap.price_to_beat,
            snap.seconds_to_expiry,
            price_vol_per_s,
        )
        ret_z = clamp(filtered_drift / price_vol_per_s, -8.0, 8.0)
        flow_z = clamp(snap.taker_aggression + 0.5 * snap.flow_acceleration, -8.0, 8.0)
        imbalance_z = clamp(0.55 * snap.spoof_resistant_imbalance + 0.45 * snap.book_pressure, -8.0, 8.0)
        absorption = max(snap.absorption_up, snap.absorption_down)
        regimes = self._hmm.update(
            ret_z=ret_z,
            flow_z=flow_z,
            imbalance_z=imbalance_z,
            absorption=absorption,
            exhaustion=snap.exhaustion,
            jump=snap.jump_intensity + snap.liquidity_vacuum,
            ts_ns=snap.ts_mono_ns,
        )
        evidence = self._evidence(snap, filtered_drift, boundary_density)
        decorrelated = self._decorrelate(evidence.as_list(), snap.ts_mono_ns)
        weights = self._regime_weights(regimes, snap)
        evidence_logodds = dot(weights, decorrelated)
        time_weight = clamp((snap.seconds_to_expiry / 300.0) ** 0.35, 0.15, 1.0)
        settlement_weight = 1.0 - 0.42 * time_weight
        raw_logodds = settlement_weight * logit(base_p) + time_weight * evidence_logodds

        hazard = self._directional_hazard(snap, filtered_drift, boundary_density)
        raw_logodds += hazard
        uncertainty = self._posterior_uncertainty(snap, filtered_var, evidence, boundary_density)
        raw_p = sigmoid(raw_logodds)
        calibrated = self._calibrator.calibrate(raw_p, uncertainty)

        entropy = binary_entropy(calibrated)
        confidence = clamp(abs(calibrated - 0.5) * 2.0, 0.0, 1.0)
        confidence *= clamp(1.0 - 0.72 * entropy, 0.05, 1.0)
        confidence *= clamp(1.0 - uncertainty, 0.05, 1.0)
        confidence *= clamp(1.0 - self._config.stale_suppression_weight * snap.stale_penalty, 0.0, 1.0)

        cont_p, rev_p = self._continuation_reversal_probabilities(snap, regimes, calibrated)
        direction = Direction.UP if calibrated > 0.5 else Direction.DOWN
        edge = abs(calibrated - 0.5)
        self._last_raw_p = raw_p
        return ProbabilityEstimate(
            ts_mono_ns=snap.ts_mono_ns,
            p_up=calibrated,
            p_down=1.0 - calibrated,
            confidence=confidence,
            posterior_uncertainty=uncertainty,
            continuation_probability=cont_p,
            reversal_probability=rev_p,
            settlement_probability=calibrated if direction is Direction.UP else 1.0 - calibrated,
            entropy=entropy,
            direction=direction,
            edge=edge,
            fair_price_up=calibrated,
            fair_price_down=1.0 - calibrated,
            diagnostics={
                "base_p_up": base_p,
                "raw_p_up": raw_p,
                "kalman_price": filtered_price,
                "kalman_drift": filtered_drift,
                "boundary_density": boundary_density,
                "hmm_down": regimes[0],
                "hmm_mean_revert": regimes[1],
                "hmm_neutral": regimes[2],
                "hmm_up": regimes[3],
                "hmm_high_vol": regimes[4],
                "evidence_logodds": evidence_logodds,
                "hazard_logodds": hazard,
            },
        )

    def _evidence(self, snap: MicrostructureSnapshot, drift: float, boundary_density: float) -> EvidenceVector:
        price_vol_per_s = max(1e-8, snap.truth_price) * max(1e-8, snap.realized_vol)
        distance_scale = max(1e-8, price_vol_per_s * max(1.0, snap.seconds_to_expiry) ** 0.5)
        signed_dist_z = clamp(snap.signed_distance / distance_scale, -8.0, 8.0)
        drift_z = clamp(drift / price_vol_per_s, -8.0, 8.0)
        continuation = (
            0.46 * drift_z
            + 0.24 * snap.taker_aggression
            + 0.18 * snap.flow_acceleration
            + 0.18 * snap.spoof_resistant_imbalance
            - 0.36 * snap.exhaustion
            - 0.24 * snap.burst_failure
        )
        reversal = (
            -0.48 * drift_z * clamp(boundary_density * distance_scale, 0.0, 2.0)
            - 0.36 * snap.taker_aggression * snap.exhaustion
            - 0.24 * snap.flow_acceleration * snap.burst_failure
            + 0.32 * (snap.absorption_up - snap.absorption_down)
        )
        flow = 0.62 * snap.taker_aggression + 0.38 * snap.flow_acceleration
        book = (
            0.52 * snap.book_pressure
            + 0.36 * snap.spoof_resistant_imbalance
            + 0.20 * snap.queue_survival * snap.book_pressure
            - 0.18 * (1.0 - snap.spread_stability) * snap.book_pressure
        )
        absorption = 0.58 * (snap.absorption_down - snap.absorption_up)
        vacuum = -0.32 * snap.liquidity_vacuum * math.copysign(1.0, drift_z if abs(drift_z) > 1e-9 else flow)
        regime = 0.78 * snap.regime_trend - 0.26 * snap.regime_volatility * math.copysign(1.0, drift_z)
        return EvidenceVector(
            continuation=clamp(continuation, -8.0, 8.0),
            reversal=clamp(reversal, -8.0, 8.0),
            flow=clamp(flow, -8.0, 8.0),
            book=clamp(book, -8.0, 8.0),
            distance=signed_dist_z,
            absorption=clamp(absorption, -8.0, 8.0),
            vacuum=clamp(vacuum, -8.0, 8.0),
            regime=clamp(regime, -8.0, 8.0),
        )

    def _decorrelate(self, x: list[float], ts_ns: int) -> list[float]:
        cov = self._corr.update(x, ts_ns)
        out: list[float] = []
        for i, xi in enumerate(x):
            redundancy = 0.0
            var_i = max(1e-6, cov[i][i])
            for j, xj in enumerate(x):
                if i == j:
                    continue
                corr = cov[i][j] / max(1e-6, (var_i * cov[j][j]) ** 0.5)
                redundancy += abs(corr) * 0.035 * xj
            out.append(clamp(xi - redundancy, -8.0, 8.0))
        return out

    def _regime_weights(self, regimes: list[float], snap: MicrostructureSnapshot) -> list[float]:
        trend = regimes[HMMRegimeFilter.UP_TREND] + regimes[HMMRegimeFilter.DOWN_TREND]
        mean_rev = regimes[HMMRegimeFilter.MEAN_REVERT]
        high_vol = regimes[HMMRegimeFilter.HIGH_VOL]
        t_decay = clamp((snap.seconds_to_expiry / 300.0) ** 0.25, 0.20, 1.0)
        return [
            0.36 + 0.32 * trend - 0.16 * high_vol,
            0.30 + 0.45 * mean_rev + 0.22 * high_vol,
            0.34 + 0.16 * trend,
            0.25 + 0.16 * snap.liquidity_stability,
            0.28 + 0.36 * (1.0 - t_decay),
            0.26 + 0.28 * mean_rev,
            0.18 + 0.35 * high_vol,
            0.32 + 0.30 * trend,
        ]

    def _directional_hazard(self, snap: MicrostructureSnapshot, drift: float, boundary_density: float) -> float:
        t = max(1e-3, snap.seconds_to_expiry)
        price_vol_per_s = max(1e-8, snap.truth_price) * max(1e-8, snap.realized_vol)
        drift_side = math.tanh(drift / price_vol_per_s)
        crossing_hazard = boundary_density * price_vol_per_s / max(1.0, t**0.5)
        exhaustion_hazard = snap.exhaustion * (snap.burst_failure + 0.35 * snap.liquidity_vacuum)
        absorption_hazard = snap.absorption_down - snap.absorption_up
        return clamp(
            0.42 * drift_side * math.exp(-t / 240.0)
            - 0.72 * drift_side * exhaustion_hazard
            + 0.34 * absorption_hazard
            + 0.18 * crossing_hazard * math.copysign(1.0, snap.signed_distance),
            -4.0,
            4.0,
        )

    def _posterior_uncertainty(
        self,
        snap: MicrostructureSnapshot,
        filtered_var: float,
        evidence: EvidenceVector,
        boundary_density: float,
    ) -> float:
        conflict = abs(evidence.continuation + evidence.reversal) / (1.0 + abs(evidence.continuation) + abs(evidence.reversal))
        conflict = 1.0 - conflict
        vol_unc = clamp(snap.regime_volatility + snap.jump_intensity + snap.liquidity_vacuum, 0.0, 3.0) / 3.0
        stale = snap.stale_penalty
        price_vol_per_s = max(1e-8, snap.truth_price) * max(1e-8, snap.realized_vol)
        boundary = clamp(boundary_density * price_vol_per_s * 30.0, 0.0, 1.0)
        kalman = clamp(filtered_var / (price_vol_per_s * price_vol_per_s), 0.0, 1.0)
        return clamp(0.23 * conflict + 0.22 * vol_unc + 0.22 * stale + 0.20 * boundary + 0.13 * kalman, 0.0, 0.98)

    def _continuation_reversal_probabilities(
        self,
        snap: MicrostructureSnapshot,
        regimes: list[float],
        p_up: float,
    ) -> tuple[float, float]:
        dir_sign = 1.0 if p_up >= 0.5 else -1.0
        trend_p = regimes[HMMRegimeFilter.UP_TREND] if dir_sign > 0 else regimes[HMMRegimeFilter.DOWN_TREND]
        flow_align = sigmoid(dir_sign * (snap.taker_aggression + 0.55 * snap.flow_acceleration))
        book_align = sigmoid(dir_sign * (snap.spoof_resistant_imbalance + 0.35 * snap.book_pressure))
        exhaustion = sigmoid(1.4 * snap.exhaustion + 1.1 * snap.burst_failure + 0.9 * snap.liquidity_vacuum)
        continuation = clamp(0.42 * trend_p + 0.26 * flow_align + 0.22 * book_align + 0.10 * snap.liquidity_stability, 0.0, 1.0)
        continuation *= clamp(1.0 - 0.62 * exhaustion, 0.02, 1.0)
        reversal = clamp(
            0.34 * regimes[HMMRegimeFilter.MEAN_REVERT]
            + 0.24 * regimes[HMMRegimeFilter.HIGH_VOL]
            + 0.27 * exhaustion
            + 0.15 * (1.0 - book_align),
            0.0,
            1.0,
        )
        return continuation, reversal

