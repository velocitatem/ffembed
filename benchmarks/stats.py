"""Statistics helpers used by every benchmark suite.

Bootstrap confidence intervals, paired permutation tests, Wilson intervals.
Deliberately dependency-free so numbers mean the same thing everywhere.
"""

from __future__ import annotations

import math
import random


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def percentile(sorted_xs: list[float], p: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not sorted_xs:
        return 0.0
    idx = min(len(sorted_xs) - 1, max(0, int(round(p / 100 * (len(sorted_xs) - 1)))))
    return sorted_xs[idx]


def bootstrap_ci(xs: list[float], *, stat=mean, confidence: float = 0.95,
                 iters: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile-bootstrap CI for ``stat`` over resamples of ``xs``."""
    if not xs:
        return 0.0, 0.0
    rng = random.Random(seed)
    n = len(xs)
    stats: list[float] = []
    for _ in range(iters):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        stats.append(stat(sample))
    stats.sort()
    lo_q = (1 - confidence) / 2 * 100
    hi_q = 100 - lo_q
    return percentile(stats, lo_q), percentile(stats, hi_q)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def paired_permutation_test(a: list[float], b: list[float], *,
                            iters: int = 5000, seed: int = 0) -> float:
    """Two-sided paired permutation test on the mean of b - a.

    Returns a p-value. Assumes lists are paired index-wise (same task in both
    arms). Zero differences are included as-is; sign flips are applied to all
    non-zero differences.
    """
    assert len(a) == len(b) and a, "paired test needs equally long, non-empty lists"
    diffs = [y - x for x, y in zip(a, b)]
    observed = abs(mean(diffs))
    rng = random.Random(seed)
    count = 0
    for _ in range(iters):
        flipped = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(mean(flipped)) >= observed:
            count += 1
    return (count + 1) / (iters + 1)


def pct(values: list[float]) -> list[float]:
    return [v * 100 for v in values]
