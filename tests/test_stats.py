import hashlib
import math

import numpy as np
import pytest
from scipy.stats import norm

from experiment_mcp.stats.assignment import arm_for_bucket, assign_units, unit_bucket
from experiment_mcp.stats.bootstrap import iter_bootstrap_diffs, percentile_interval
from experiment_mcp.stats.lift import analyze_lift, apply_cuped
from experiment_mcp.stats.power import (
    mde_means,
    power_calculation,
    sample_size_means,
    sample_size_proportions,
)
from experiment_mcp.stats.srm import sample_ratio_mismatch


def test_mean_sample_size_matches_textbook_case() -> None:
    # n = 2 * (z_{1-a/2} + z_power)^2 * sigma^2 / delta^2, sigma=1, delta=0.5.
    z = float(norm.ppf(0.975)) + float(norm.ppf(0.8))
    expected = math.ceil(2 * z**2 / 0.25 - 1e-12)
    assert expected == 63
    assert sample_size_means(0.5, 1.0, 0.05, 0.8) == 63
    result = power_calculation(metric="mean", alpha=0.05, power=0.8, baseline=None, sigma=1.0, mde=0.5, n_per_arm=None)
    assert result.n_per_arm == 63
    assert result.solved_for == "n_per_arm"


def test_mean_mde_is_the_closed_form_inverse() -> None:
    z = float(norm.ppf(0.975)) + float(norm.ppf(0.8))
    expected = z * math.sqrt(2 / 63)
    assert mde_means(63, 1.0, 0.05, 0.8) == pytest.approx(expected)
    result = power_calculation(metric="mean", alpha=0.05, power=0.8, baseline=None, sigma=1.0, mde=None, n_per_arm=63)
    assert result.mde == pytest.approx(expected)
    assert result.solved_for == "mde"


def test_proportion_sample_size_unpooled_formula() -> None:
    z_alpha = float(norm.ppf(0.975))
    z_beta = float(norm.ppf(0.8))
    baseline, treated = 0.1, 0.12
    pooled = (baseline + treated) / 2
    se_null = math.sqrt(2 * pooled * (1 - pooled))
    se_alt = math.sqrt(baseline * (1 - baseline) + treated * (1 - treated))
    expected = math.ceil((z_alpha * se_null + z_beta * se_alt) ** 2 / (treated - baseline) ** 2 - 1e-9)
    assert expected == 3841
    assert sample_size_proportions(0.1, 0.02, 0.05, 0.8) == 3841


def test_proportion_mde_search_round_trips() -> None:
    n = 3841
    mde = power_calculation(
        metric="proportion",
        alpha=0.05,
        power=0.8,
        baseline=0.1,
        sigma=None,
        mde=None,
        n_per_arm=n,
    ).mde
    assert mde is not None
    assert sample_size_proportions(0.1, mde, 0.05, 0.8) <= n
    assert sample_size_proportions(0.1, mde * 0.99, 0.05, 0.8) >= n


def test_power_rejects_ambiguous_target() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        power_calculation(metric="mean", alpha=0.05, power=0.8, baseline=None, sigma=1, mde=0.2, n_per_arm=100)


def test_assignment_is_deterministic_and_matches_the_hash() -> None:
    salt = "synthetic-v1"
    unit_id = "c00042"
    digest = hashlib.sha256(f"{salt}\x1f{unit_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") / 2**64
    assert unit_bucket(unit_id, salt) == bucket
    result = assign_units([unit_id, "c00007"], salt, {"treatment": 0.5, "control": 0.5})
    assert result.arm_order == ["control", "treatment"]
    again = assign_units([unit_id, "c00007"], salt, {"control": 0.5, "treatment": 0.5})
    assert [row.arm for row in result.assignments] == [row.arm for row in again.assignments]
    thresholds = [("control", 0.5), ("treatment", 1.0)]
    assert result.assignments[0].arm == arm_for_bucket(bucket, thresholds)


def test_srm_balanced_and_mismatch() -> None:
    balanced = sample_ratio_mismatch({"control": 500, "treatment": 500}, {"treatment": 0.5, "control": 0.5})
    assert balanced.chi_square == pytest.approx(0.0)
    assert balanced.p_value == pytest.approx(1.0)
    assert balanced.mismatch is False
    skewed = sample_ratio_mismatch({"control": 600, "treatment": 400}, {"control": 0.5, "treatment": 0.5})
    assert skewed.chi_square == pytest.approx(40.0)
    assert skewed.p_value == pytest.approx(2.5396285894708634e-10)
    assert skewed.mismatch is True
    assert skewed.degrees_of_freedom == 1


def test_cuped_removes_a_perfect_linear_covariate() -> None:
    control_y = np.array([1.0, 2.0, 3.0])
    treatment_y = np.array([2.0, 3.0, 4.0])
    covariate = np.array([1.0, 2.0, 3.0])
    adjusted_c, adjusted_t, theta = apply_cuped(control_y, covariate, treatment_y, covariate)
    assert theta == pytest.approx(1.0)
    assert adjusted_c == pytest.approx([2.0, 2.0, 2.0])
    assert adjusted_t == pytest.approx([3.0, 3.0, 3.0])
    result = analyze_lift(
        outcome="demo",
        control=control_y,
        treatment=treatment_y,
        control_covariate=covariate,
        treatment_covariate=covariate,
    )
    assert result.absolute_lift == pytest.approx(1.0)
    assert result.absolute_ci_low == pytest.approx(1.0)
    assert result.absolute_ci_high == pytest.approx(1.0)
    assert result.relative_lift == pytest.approx(0.5)
    assert result.control.mean == pytest.approx(2.0)
    assert result.treatment.mean == pytest.approx(3.0)
    assert result.cuped_theta == pytest.approx(1.0)


def test_unadjusted_lift_interval() -> None:
    result = analyze_lift(outcome="demo", control=[0.0, 0.0, 1.0, 1.0], treatment=[1.0, 1.0, 1.0, 1.0])
    assert result.absolute_lift == pytest.approx(0.5)
    assert result.control.mean == pytest.approx(0.5)
    assert result.treatment.mean == pytest.approx(1.0)
    assert result.relative_lift == pytest.approx(1.0)
    assert result.absolute_ci_low < result.absolute_lift < result.absolute_ci_high


def test_bootstrap_percentile_and_early_stop() -> None:
    control = np.array([1.0, 2.0, 3.0, 4.0])
    treatment = np.array([2.0, 3.0, 4.0, 5.0])
    chunks = []
    seen = 0
    for done, total, diffs, _relative in iter_bootstrap_diffs(control, treatment, iterations=50, seed=7, batch_size=20):
        chunks.append(diffs)
        seen = done
        assert total == 50
        if done >= 20:
            break
    assert seen == 20
    assert sum(len(chunk) for chunk in chunks) == 20
    full = np.concatenate(
        [
            diffs
            for _done, _total, diffs, _relative in iter_bootstrap_diffs(
                control, treatment, iterations=200, seed=7, batch_size=50
            )
        ]
    )
    low, high = percentile_interval(full, 0.05)
    assert low < high
    assert low < float(full.mean()) < high
