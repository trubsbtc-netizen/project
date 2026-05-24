from __future__ import annotations

import math
from collections.abc import Sequence


SQRT_2 = 2.0**0.5
SQRT_2PI = (2.0 * math.pi) ** 0.5


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-min(60.0, x))
        return 1.0 / (1.0 + z)
    z = math.exp(max(-60.0, x))
    return z / (1.0 + z)


def logit(p: float) -> float:
    p = clamp(p, 1e-9, 1.0 - 1e-9)
    return math.log(p / (1.0 - p))


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def binary_entropy(p: float) -> float:
    p = clamp(p, 1e-9, 1.0 - 1e-9)
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)) / math.log(2.0)


def softmax(log_weights: Sequence[float]) -> list[float]:
    m = max(log_weights)
    exps = [math.exp(clamp(v - m, -60.0, 60.0)) for v in log_weights]
    s = sum(exps)
    if s <= 0.0:
        return [1.0 / len(log_weights)] * len(log_weights)
    return [v / s for v in exps]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def robust_z(x: float, mean: float, std: float, cap: float = 8.0) -> float:
    return clamp((x - mean) / max(1e-12, std), -cap, cap)


def ew_alpha(dt_s: float, halflife_s: float) -> float:
    return 1.0 - 2.0 ** (-max(0.0, dt_s) / max(1e-9, halflife_s))


def logsumexp(values: Sequence[float]) -> float:
    m = max(values)
    return m + math.log(sum(math.exp(clamp(v - m, -60.0, 60.0)) for v in values))

