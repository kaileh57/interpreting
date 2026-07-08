"""Bootstrap confidence intervals for means and paired differences.

`rcg.metrics.stats.bootstrap_ci` already covers single-sample means; this
module adds a plain-tuple-returning variant plus a *paired* bootstrap for
comparing two aligned measurements per example (e.g. hidden vs. distractor
causal effect in notebook 11).
"""

from __future__ import annotations

import random
from collections.abc import Sequence


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI for the mean. Returns `(mean, low, high)`."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return vals[0], vals[0], vals[0]

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [vals[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    mean = sum(vals) / n
    return mean, lo, hi


def paired_bootstrap_diff_ci(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """
    Percentile bootstrap CI for the paired mean difference ``mean(a) - mean(b)``.

    `a` and `b` must be the same length and index-aligned (paired
    observations, e.g. hidden vs. distractor causal effect for the same
    example). Returns `(mean_diff, low, high)`.
    """
    va = [float(x) for x in a]
    vb = [float(x) for x in b]
    if len(va) != len(vb):
        raise ValueError("paired_bootstrap_diff_ci requires equal-length paired sequences")
    n = len(va)
    if n == 0:
        return 0.0, 0.0, 0.0

    diffs = [x - y for x, y in zip(va, vb, strict=True)]
    if n == 1:
        return diffs[0], diffs[0], diffs[0]

    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [diffs[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    mean_diff = sum(diffs) / n
    return mean_diff, lo, hi
