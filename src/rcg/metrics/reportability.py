"""Reportability scoring metrics."""

from __future__ import annotations

from rcg.readouts.base import ReadoutResult


def reportability_score(
    readout_results: list[ReadoutResult],
    ground_truth: str,
    k: int = 10,
) -> float:
    """
    Score how accurately a readout identifies the ground-truth latent variable.

    Returns 1.0 if ground_truth appears in top-k concepts, else 0.0 (stub).
    """
    top_concepts = [r.concept for r in readout_results[:k]]
    return 1.0 if ground_truth in top_concepts else 0.0


def causal_precision_at_k(
    readout_results: list[ReadoutResult],
    causal_concepts: set[str],
    k: int = 10,
) -> float:
    """Fraction of top-k readout concepts that are causal under intervention."""
    if k == 0:
        return 0.0
    top_concepts = [r.concept for r in readout_results[:k]]
    hits = sum(1 for c in top_concepts if c in causal_concepts)
    return hits / k


def readout_hallucination_rate(
    readout_results: list[ReadoutResult],
    ground_truth_concepts: set[str],
    causal_concepts: set[str],
    k: int = 5,
) -> float:
    """
    Fraction of top-k named concepts that are absent from ground truth AND
    non-causal under intervention (confident-but-wrong readouts).
    """
    top_concepts = [r.concept for r in readout_results[:k]]
    if not top_concepts:
        return 0.0
    halluc = sum(
        1
        for c in top_concepts
        if c not in ground_truth_concepts and c not in causal_concepts
    )
    return halluc / len(top_concepts)
