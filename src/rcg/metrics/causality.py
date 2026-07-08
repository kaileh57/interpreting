"""Causality scoring metrics."""

from __future__ import annotations


def causal_effect_score(intervention_delta: float, threshold: float = 0.0) -> float:
    """
    Binary or continuous score for whether an intervention changed behavior.

    Stub: returns 1.0 if |delta| exceeds threshold, else 0.0.
    """
    return 1.0 if abs(intervention_delta) > threshold else 0.0


def readable_only_rate(
    reportability: float,
    causal_effect: float,
    reportability_threshold: float = 0.5,
    causal_threshold: float = 0.5,
) -> bool:
    """True when readout is correct but intervention has little/no effect."""
    return reportability >= reportability_threshold and causal_effect < causal_threshold


def causal_only_rate(
    reportability: float,
    causal_effect: float,
    reportability_threshold: float = 0.5,
    causal_threshold: float = 0.5,
) -> bool:
    """True when intervention changes behavior but readout misses the variable."""
    return reportability < reportability_threshold and causal_effect >= causal_threshold


def causal_calibration(
    confidences: list[float],
    causal_effects: list[float],
) -> float:
    """
    Pearson correlation between readout confidence and causal effect.

    High positive value means confident readouts really do have larger causal
    effect (well-calibrated causal confidence). Returns 0.0 if undefined.
    """
    n = len(confidences)
    if n < 2 or n != len(causal_effects):
        return 0.0
    mx = sum(confidences) / n
    my = sum(causal_effects) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(confidences, causal_effects, strict=True))
    vx = sum((x - mx) ** 2 for x in confidences)
    vy = sum((y - my) ** 2 for y in causal_effects)
    denom = (vx * vy) ** 0.5
    return cov / denom if denom > 1e-12 else 0.0


def self_report_disagreement(
    self_report_concept: str,
    readout_concept: str,
) -> bool:
    """True when behavioral self-report and a white-box readout disagree."""
    a = (self_report_concept or "").strip().lower()
    b = (readout_concept or "").strip().lower()
    if not a or not b:
        return False
    return a not in b and b not in a
