"""Reportability–causality gap metrics."""

from __future__ import annotations


def rcg_gap(reportability_score: float, normalized_causal_effect: float) -> float:
    """
    Core RCG metric: reportability minus normalized causal effect.

    Positive gap suggests readable-but-non-causal explanations.
    Negative gap suggests causal-but-unreadable variables.
    """
    return reportability_score - normalized_causal_effect


def failure_mode(
    reportability: float,
    causal_effect: float,
    threshold: float = 0.5,
) -> str:
    """Classify an example into readable+causal, readable-only, causal-only, or neither."""
    readable = reportability >= threshold
    causal = causal_effect >= threshold
    if readable and causal:
        return "readable_and_causal"
    if readable and not causal:
        return "readable_only"
    if not readable and causal:
        return "causal_only"
    return "neither"


def failure_mode_counts(modes: list[str]) -> dict[str, int]:
    """Aggregate a list of failure-mode labels into counts for all four cells."""
    keys = ["readable_and_causal", "readable_only", "causal_only", "neither"]
    counts = dict.fromkeys(keys, 0)
    for m in modes:
        counts[m] = counts.get(m, 0) + 1
    return counts


def mean(values: list[float]) -> float:
    """Mean of a list, 0.0 when empty."""
    return sum(values) / len(values) if values else 0.0
