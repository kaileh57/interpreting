"""Plotting helpers for RCG-Bench analysis."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt


def rcg_scatter(
    reportabilities: list[float],
    causal_effects: list[float],
    labels: list[str] | None = None,
    title: str = "Reportability vs causal effect",
) -> Any:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(reportabilities, causal_effects, alpha=0.7)
    if labels:
        for x, y, label in zip(reportabilities, causal_effects, labels, strict=False):
            ax.annotate(label, (x, y), fontsize=7, alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Reportability score")
    ax.set_ylabel("Normalized causal effect")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    return fig


def failure_mode_matrix(counts: dict[str, int], title: str = "Failure-mode matrix") -> Any:
    modes = [
        "readable_and_causal",
        "readable_only",
        "causal_only",
        "neither",
    ]
    values = [counts.get(m, 0) for m in modes]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(modes, values, color=["#2ecc71", "#f39c12", "#3498db", "#95a5a6"])
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def causal_precision_bar(
    method_scores: dict[str, float],
    title: str = "Causal precision@k by method",
) -> Any:
    methods = list(method_scores.keys())
    values = [method_scores[m] for m in methods]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(methods, values, color="#3498db")
    ax.set_ylabel("Causal precision@k")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def layer_curve(
    layers: list[int],
    reportability: list[float],
    causal_effect: list[float],
    title: str = "Reportability vs causal effect by layer",
) -> Any:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(layers, reportability, marker="o", label="reportability")
    ax.plot(layers, causal_effect, marker="s", label="causal effect")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def calibration_scatter(
    confidences: list[float],
    causal_effects: list[float],
    title: str = "Causal calibration",
) -> Any:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(confidences, causal_effects, alpha=0.7)
    ax.set_xlabel("Readout confidence")
    ax.set_ylabel("Causal effect")
    ax.set_title(title)
    fig.tight_layout()
    return fig
