"""Two-sample sample size and minimum detectable effect.

Equal allocation, two-sided test, normal approximation. No continuity
correction and no adjustment for multiple looks or multiple metrics.
"""

import math
from typing import Literal

from scipy.stats import norm

from experiment_mcp.models import MetricKind, PowerResult

_PROPORTION_FORMULA = (
    "equal-allocation two-sided normal approximation for two proportions; "
    "unpooled variance under the alternative and pooled variance under the null"
)
_MEAN_FORMULA = (
    "equal-allocation two-sided normal approximation for a difference of means with known common standard deviation"
)


def _z(probability: float) -> float:
    return float(norm.ppf(probability))


def _critical_pair(alpha: float, power: float) -> tuple[float, float]:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if not 0 < power < 1:
        raise ValueError("power must be between 0 and 1")
    return _z(1 - alpha / 2), _z(power)


def sample_size_means(delta: float, sigma: float, alpha: float, power: float) -> int:
    if delta <= 0:
        raise ValueError("mde must be a positive absolute difference")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z_alpha, z_beta = _critical_pair(alpha, power)
    n = 2 * sigma**2 * (z_alpha + z_beta) ** 2 / delta**2
    return max(2, math.ceil(n - 1e-12))


def mde_means(n_per_arm: int, sigma: float, alpha: float, power: float) -> float:
    if n_per_arm < 2:
        raise ValueError("n_per_arm must be at least 2")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    z_alpha, z_beta = _critical_pair(alpha, power)
    return (z_alpha + z_beta) * sigma * math.sqrt(2 / n_per_arm)


def sample_size_proportions(baseline: float, mde: float, alpha: float, power: float) -> int:
    if mde == 0:
        raise ValueError("mde must be a non-zero absolute difference")
    treated = baseline + mde
    if not 0 < baseline < 1 or not 0 < treated < 1:
        raise ValueError("baseline and baseline + mde must both lie strictly between 0 and 1")
    z_alpha, z_beta = _critical_pair(alpha, power)
    pooled = (baseline + treated) / 2
    se_null = math.sqrt(2 * pooled * (1 - pooled))
    se_alt = math.sqrt(baseline * (1 - baseline) + treated * (1 - treated))
    n = (z_alpha * se_null + z_beta * se_alt) ** 2 / (treated - baseline) ** 2
    return max(2, math.ceil(n - 1e-9))


def mde_proportions(n_per_arm: int, baseline: float, alpha: float, power: float) -> float:
    """Smallest absolute MDE whose required per-arm sample size is at most ``n_per_arm``."""
    if n_per_arm < 2:
        raise ValueError("n_per_arm must be at least 2")
    if not 0 < baseline < 1:
        raise ValueError("baseline must lie strictly between 0 and 1")
    # Search the positive direction. A negative MDE of the same magnitude is a
    # different design (rates cannot go below 0), so callers pass the sign they want
    # by inspecting the business question. This search returns a positive MDE.
    hi = 1 - baseline - 1e-6
    if hi <= 1e-8:
        raise ValueError("baseline is too close to 0 or 1 to search for an MDE")
    if sample_size_proportions(baseline, hi, alpha, power) > n_per_arm:
        raise ValueError(f"n_per_arm={n_per_arm} is too small to detect any MDE that keeps both rates in (0, 1)")
    lo = 1e-8
    for _ in range(60):
        mid = (lo + hi) / 2
        if sample_size_proportions(baseline, mid, alpha, power) <= n_per_arm:
            hi = mid
        else:
            lo = mid
    return hi


def power_calculation(
    *,
    metric: MetricKind,
    alpha: float,
    power: float,
    baseline: float | None,
    sigma: float | None,
    mde: float | None,
    n_per_arm: int | None,
) -> PowerResult:
    if (mde is None) == (n_per_arm is None):
        raise ValueError("provide exactly one of mde or n_per_arm")
    if metric == "proportion":
        if baseline is None:
            raise ValueError("proportion designs require baseline, the control rate")
        if sigma is not None:
            raise ValueError("sigma is only used for mean designs")
        formula = _PROPORTION_FORMULA
        if n_per_arm is None:
            assert mde is not None
            solved_n = sample_size_proportions(baseline, mde, alpha, power)
            solved_mde = mde
            solved_for: Literal["n_per_arm", "mde"] = "n_per_arm"
        else:
            solved_mde = mde_proportions(n_per_arm, baseline, alpha, power)
            solved_n = n_per_arm
            solved_for = "mde"
        solved_sigma = None
    else:
        if sigma is None:
            raise ValueError("mean designs require sigma, the outcome standard deviation")
        if baseline is not None:
            raise ValueError("baseline is only used for proportion designs")
        formula = _MEAN_FORMULA
        if n_per_arm is None:
            assert mde is not None
            solved_n = sample_size_means(mde, sigma, alpha, power)
            solved_mde = mde
            solved_for = "n_per_arm"
        else:
            solved_mde = mde_means(n_per_arm, sigma, alpha, power)
            solved_n = n_per_arm
            solved_for = "mde"
        solved_sigma = sigma
    return PowerResult(
        metric=metric,
        alpha=alpha,
        power=power,
        baseline=baseline,
        sigma=solved_sigma,
        mde=solved_mde,
        n_per_arm=solved_n,
        solved_for=solved_for,
        formula=formula,
    )
