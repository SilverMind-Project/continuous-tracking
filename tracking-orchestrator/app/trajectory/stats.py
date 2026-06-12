"""Robust statistics for dementia signal baselines.

Pure functions, no I/O. Uses numpy/scipy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobustZ:
    """Result of a robust z-score computation using median + MAD."""

    median: float
    mad: float  # median absolute deviation (scaled)
    modified_z: float  # 0.6745 * (value - median) / MAD
    n: int  # number of samples in baseline


def robust_z(value: float, samples: list[float]) -> RobustZ:
    """Compute a modified z-score using median and MAD.

    The modified z-score uses median absolute deviation (MAD) scaled by
    the constant 0.6745 to make it consistent with the standard deviation
    for normally-distributed data.

    Returns ``RobustZ`` with ``n`` samples. The caller should decide the
    ``n`` floor — when ``n`` is too small the baseline is unreliable.
    """
    if not samples:
        return RobustZ(median=0.0, mad=0.0, modified_z=0.0, n=0)

    arr = np.array(samples, dtype=np.float64)
    median = float(np.median(arr))
    abs_dev = np.abs(arr - median)
    mad = float(np.median(abs_dev))

    if mad == 0.0:
        # When all samples are identical, MAD is zero.
        # If value == median: z = 0. If value != median: arbitrary large z.
        if abs(value - median) < 1e-9:
            return RobustZ(median=median, mad=mad, modified_z=0.0, n=len(samples))
        return RobustZ(median=median, mad=mad, modified_z=float("inf"), n=len(samples))

    modified_z_val = 0.6745 * (value - median) / mad
    return RobustZ(
        median=round(median, 6),
        mad=round(mad, 6),
        modified_z=round(modified_z_val, 4),
        n=len(samples),
    )


def weighted_median(values: list[float], weights: list[float]) -> float:
    """Duration-weighted median of a non-empty parallel list of values and weights.

    Sorts values by value, accumulates weights, and returns the value at the
    50th percentile of cumulative weight.  Equal to the ordinary median when all
    weights are identical.

    Raises ValueError on empty input or mismatched lengths.
    """
    if not values:
        raise ValueError("weighted_median requires at least one value")
    if len(values) != len(weights):
        raise ValueError("values and weights must have the same length")

    total = sum(weights)
    if total <= 0:
        # Fall back to unweighted median when all weights are zero/negative.
        return float(np.median(np.array(values, dtype=np.float64)))

    pairs = sorted(zip(values, weights, strict=True), key=lambda p: p[0])
    half = total / 2.0
    cumulative = 0.0
    for v, w in pairs:
        cumulative += w
        if cumulative >= half:
            return v
    # Should be unreachable; return last value as safety fallback.
    return pairs[-1][0]


def ewma(samples: list[float], halflife: float) -> float:
    """Exponentially-weighted moving average.

    Args:
        samples: time-ordered values (oldest to newest).
        halflife: number of periods for the weight to halve.
    """
    if not samples:
        return 0.0
    alpha = 1.0 - np.exp(np.log(0.5) / halflife)
    weights = np.array(
        [(1.0 - alpha) ** i for i in range(len(samples) - 1, -1, -1)],
        dtype=np.float64,
    )
    weights *= alpha
    weights[-1] = (1.0 - alpha) ** (len(samples) - 1)  # initial weight
    return float(np.average(samples, weights=weights))
