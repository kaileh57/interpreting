"""Statistical helpers: bootstrap confidence intervals for defensible reporting."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class Estimate:
    """A point estimate with a bootstrap confidence interval."""

    mean: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.mean:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"

    def as_dict(self) -> dict[str, float]:
        return {"mean": self.mean, "lo": self.lo, "hi": self.hi, "n": self.n}


def bootstrap_ci(
    values: list[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Estimate:
    """Percentile bootstrap CI for the mean of `values`."""
    n = len(values)
    if n == 0:
        return Estimate(0.0, 0.0, 0.0, 0)
    if n == 1:
        v = float(values[0])
        return Estimate(v, v, v, 1)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    mean = sum(values) / n
    return Estimate(mean, lo, hi, n)


def chance_reportability(vocab_size: int, k: int) -> float:
    """Chance level for top-k membership readout over a vocab of size `vocab_size`."""
    if vocab_size <= 0:
        return 0.0
    return min(1.0, k / vocab_size)
