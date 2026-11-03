"""Difference-in-means lift, with an optional CUPED adjustment.

The interval is a normal approximation using the sample variance of each arm
(``ddof=1``). For a 0/1 outcome that variance is the unbiased Bernoulli
variance, so the same routine covers redemption rates and continuous spend.

CUPED follows Deng, Xu, Kohavi, and Walker: one theta from the pooled
sample, ``theta = Cov(X, Y) / Var(X)``, and
``Y_adj = Y - theta * (X - mean(X))``. The point estimate moves only when the
covariate means differ across arms. Relative-lift intervals use the delta
method on the ratio of arm means.
"""

import math

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from experiment_mcp.models import CiMethod, GroupSummary, LiftResult

FloatArray = NDArray[np.float64]


def _as_float(values: list[float] | FloatArray, label: str) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional list of numbers")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains a non-finite value")
    return array


def apply_cuped(
    control: FloatArray,
    control_covariate: FloatArray,
    treatment: FloatArray,
    treatment_covariate: FloatArray,
) -> tuple[FloatArray, FloatArray, float]:
    if control.shape != control_covariate.shape or treatment.shape != treatment_covariate.shape:
        raise ValueError("each covariate list must match the length of its outcome list")
    covariate = np.concatenate([control_covariate, treatment_covariate])
    outcome = np.concatenate([control, treatment])
    if covariate.size < 2:
        raise ValueError("CUPED needs at least two observations in total")
    variance = float(np.var(covariate, ddof=1))
    if variance == 0:
        theta = 0.0
    else:
        covariance = float(np.cov(covariate, outcome, ddof=1)[0, 1])
        theta = covariance / variance
    center = float(covariate.mean())
    adjusted_control = control - theta * (control_covariate - center)
    adjusted_treatment = treatment - theta * (treatment_covariate - center)
    return adjusted_control, adjusted_treatment, theta


def _summarize(values: FloatArray) -> tuple[float, float]:
    if values.size < 2:
        raise ValueError("each arm needs at least 2 observations")
    return float(values.mean()), float(values.var(ddof=1))


def _relative_interval(
    mean_c: float,
    var_c: float,
    n_c: int,
    mean_t: float,
    var_t: float,
    n_t: int,
    z: float,
) -> tuple[float | None, float | None, float | None]:
    if mean_c == 0:
        return None, None, None
    relative = (mean_t - mean_c) / mean_c
    if mean_t == 0:
        var_ratio = (var_t / n_t) / mean_c**2
    else:
        ratio = mean_t / mean_c
        var_ratio = ratio**2 * ((var_t / n_t) / mean_t**2 + (var_c / n_c) / mean_c**2)
    se = math.sqrt(var_ratio)
    return relative, relative - z * se, relative + z * se


def analyze_lift(
    *,
    outcome: str,
    control: list[float] | FloatArray,
    treatment: list[float] | FloatArray,
    alpha: float = 0.05,
    control_covariate: list[float] | FloatArray | None = None,
    treatment_covariate: list[float] | FloatArray | None = None,
    ci_method: CiMethod = "normal",
    bootstrap_iterations: int | None = None,
    bootstrap_low: float | None = None,
    bootstrap_high: float | None = None,
    relative_bootstrap_low: float | None = None,
    relative_bootstrap_high: float | None = None,
    cancelled: bool = False,
) -> LiftResult:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    control_arr = _as_float(control, "control")
    treatment_arr = _as_float(treatment, "treatment")
    raw_c_mean, _ = _summarize(control_arr)
    raw_t_mean, _ = _summarize(treatment_arr)
    theta: float | None = None
    analyzed_c = control_arr
    analyzed_t = treatment_arr
    use_cuped = control_covariate is not None or treatment_covariate is not None
    if use_cuped:
        if control_covariate is None or treatment_covariate is None:
            raise ValueError("CUPED requires a covariate for both arms")
        analyzed_c, analyzed_t, theta = apply_cuped(
            control_arr,
            _as_float(control_covariate, "control_covariate"),
            treatment_arr,
            _as_float(treatment_covariate, "treatment_covariate"),
        )
    mean_c, var_c = _summarize(analyzed_c)
    mean_t, var_t = _summarize(analyzed_t)
    n_c = int(analyzed_c.size)
    n_t = int(analyzed_t.size)
    absolute = mean_t - mean_c
    z = float(norm.ppf(1 - alpha / 2))
    if ci_method == "bootstrap":
        if bootstrap_low is None or bootstrap_high is None or bootstrap_iterations is None:
            raise ValueError("bootstrap intervals require low, high, and iteration count")
        low, high = bootstrap_low, bootstrap_high
        rel = None if raw_c_mean == 0 else (raw_t_mean - raw_c_mean) / raw_c_mean
        # CUPED changes the absolute contrast. Relative lift stays on the original
        # outcome scale, using the bootstrap interval when one was supplied.
        if use_cuped:
            rel = None if mean_c == 0 else absolute / mean_c
        rel_low, rel_high = relative_bootstrap_low, relative_bootstrap_high
    else:
        se = math.sqrt(var_t / n_t + var_c / n_c)
        low, high = absolute - z * se, absolute + z * se
        rel, rel_low, rel_high = _relative_interval(mean_c, var_c, n_c, mean_t, var_t, n_t, z)
    return LiftResult(
        outcome=outcome,
        ci_method=ci_method,
        alpha=alpha,
        use_cuped=use_cuped,
        cuped_theta=theta,
        control=GroupSummary(n=n_c, mean=raw_c_mean, variance=float(control_arr.var(ddof=1))),
        treatment=GroupSummary(n=n_t, mean=raw_t_mean, variance=float(treatment_arr.var(ddof=1))),
        absolute_lift=absolute,
        absolute_ci_low=low,
        absolute_ci_high=high,
        relative_lift=rel,
        relative_ci_low=rel_low,
        relative_ci_high=rel_high,
        bootstrap_iterations=bootstrap_iterations,
        cancelled=cancelled,
    )
