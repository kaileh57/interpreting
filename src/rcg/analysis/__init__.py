"""Analysis and plotting utilities for RCG-Bench."""

from rcg.analysis.plotting import (
    calibration_scatter,
    causal_precision_bar,
    failure_mode_matrix,
    layer_curve,
    rcg_scatter,
)

__all__ = [
    "rcg_scatter",
    "failure_mode_matrix",
    "causal_precision_bar",
    "layer_curve",
    "calibration_scatter",
]
