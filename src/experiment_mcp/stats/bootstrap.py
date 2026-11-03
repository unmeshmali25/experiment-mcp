"""Chunked percentile bootstrap for a difference of means.

The caller decides how often to report progress and whether to stop. Each
yielded chunk is independent given the seeded generator, and stopping early
simply returns fewer replicates.
"""

from collections.abc import Iterator

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def iter_bootstrap_diffs(
    control: FloatArray,
    treatment: FloatArray,
    *,
    iterations: int,
    seed: int,
    batch_size: int,
) -> Iterator[tuple[int, int, FloatArray, FloatArray]]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if control.size < 2 or treatment.size < 2:
        raise ValueError("each arm needs at least 2 observations")
    rng = np.random.default_rng(seed)
    done = 0
    while done < iterations:
        take = min(batch_size, iterations - done)
        control_index = rng.integers(0, control.size, size=(take, control.size))
        treatment_index = rng.integers(0, treatment.size, size=(take, treatment.size))
        control_means = control[control_index].mean(axis=1)
        treatment_means = treatment[treatment_index].mean(axis=1)
        diffs = treatment_means - control_means
        with np.errstate(divide="ignore", invalid="ignore"):
            relative = treatment_means / control_means - 1
        done += take
        yield done, iterations, diffs, relative


def percentile_interval(diffs: FloatArray, alpha: float) -> tuple[float, float]:
    if diffs.size == 0:
        raise ValueError("no bootstrap replicates")
    low, high = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)
