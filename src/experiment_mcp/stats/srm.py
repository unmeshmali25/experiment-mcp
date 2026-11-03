"""Pearson chi-square goodness of fit for sample-ratio mismatch."""

from scipy.stats import chisquare

from experiment_mcp.models import SrmResult


def sample_ratio_mismatch(
    counts: dict[str, int],
    expected_weights: dict[str, float],
    *,
    alpha: float = 0.001,
) -> SrmResult:
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if set(counts) != set(expected_weights):
        raise ValueError("counts and expected_weights must name the same arms")
    if len(counts) < 2:
        raise ValueError("SRM needs at least two arms")
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts.values()):
        raise ValueError("counts must be non-negative integers")
    if any(weight <= 0 for weight in expected_weights.values()):
        raise ValueError("expected weights must be positive")
    weight_total = sum(expected_weights.values())
    if abs(weight_total - 1.0) > 1e-6:
        raise ValueError(f"expected weights must sum to 1 (got {weight_total})")
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("counts must include at least one observation")
    arms = sorted(counts)
    observed = [counts[arm] for arm in arms]
    expected = [expected_weights[arm] * total for arm in arms]
    statistic, p_value = chisquare(observed, expected)
    p = float(p_value)
    return SrmResult(
        counts={arm: counts[arm] for arm in arms},
        expected_counts={arm: expected_weights[arm] * total for arm in arms},
        expected_weights={arm: expected_weights[arm] for arm in arms},
        chi_square=float(statistic),
        degrees_of_freedom=len(arms) - 1,
        p_value=p,
        alpha=alpha,
        mismatch=p < alpha,
    )
