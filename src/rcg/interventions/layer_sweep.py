"""Residual patching across candidate layers and strengths.

The earlier tool-comparison notebooks patched only the middle layer. That
hides layer-dependent causal effects (e.g. late-layer patches being far more
decisive than mid-layer ones). This module sweeps a set of layer fractions and
optional interpolation strengths and returns metadata-rich rows suitable for
direct merging with readout rows on `(example_id, layer)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rcg.interventions.causal_effects import (
    intervention_delta,
    logit_diff,
    normalize_causal_effect,
)
from rcg.models.hooks import capture_last_activation, layer_module, num_hidden_layers


@dataclass
class LayerSweepConfig:
    """Which layer fractions and patch strengths to sweep, and the causal-effect threshold."""

    layer_fractions: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    strengths: tuple[float, ...] = (1.0,)
    causal_threshold: float = 0.5


def fraction_to_layer(model: Any, fraction: float) -> int:
    """
    Map a fraction in [0, 1] to a valid decoder-layer index for `model`.

    Clamped to [1, n_layers - 1] to avoid the embedding-adjacent layer 0
    (matching `rcg.models.hooks.middle_layer` convention), except for
    single-layer models where 0 is the only valid index.
    """
    n_layers = num_hidden_layers(model)
    if n_layers <= 1:
        return 0
    idx = round(fraction * (n_layers - 1))
    return max(1, min(n_layers - 1, idx))


def _task_field(task: Any, *names: str) -> Any:
    if isinstance(task, dict):
        for name in names:
            if name in task:
                return task[name]
        return None
    for name in names:
        value = getattr(task, name, None)
        if value is not None:
            return value
    return None


def _patch_and_measure_with_strength(
    model: Any,
    tokenizer: Any,
    clean_prompt: str,
    corrupt_prompt: str,
    layer: int,
    strength: float,
    positive_token: str,
    negative_token: str,
) -> dict[str, float]:
    """
    Residual patch with an interpolation strength in activation space:
    ``h' = (1 - strength) * h_clean + strength * h_corrupt``.

    ``strength=1.0`` reproduces a standard full patch (equivalent to
    `ResidualPatcher.patch_and_measure`); intermediate strengths give a genuine
    (not stubbed) dose-response measurement.
    """
    baseline = logit_diff(model, tokenizer, clean_prompt, positive_token, negative_token)
    clean_act = capture_last_activation(model, tokenizer, clean_prompt, layer)
    corrupt_act = capture_last_activation(model, tokenizer, corrupt_prompt, layer)
    mixed = (1.0 - strength) * clean_act + strength * corrupt_act

    def patch_hook(_module: torch.nn.Module, _inputs: tuple, output: Any) -> Any:
        tensor = output[0] if isinstance(output, tuple) else output
        tensor = tensor.clone()
        tensor[0, -1] = mixed
        return (tensor,) if isinstance(output, tuple) else tensor

    handle = layer_module(model, layer).register_forward_hook(patch_hook)
    try:
        patched = logit_diff(model, tokenizer, clean_prompt, positive_token, negative_token)
    finally:
        handle.remove()

    delta = intervention_delta(baseline, patched)
    return {"baseline": baseline, "patched": patched, "delta": delta}


def run_residual_patch_layer_sweep(
    model: Any,
    tokenizer: Any,
    tasks: list[Any],
    *,
    config: LayerSweepConfig,
) -> list[dict[str, Any]]:
    """
    Run residual patching for every task across every swept layer and strength.

    Returns one row per (task, layer, strength) with raw and normalized causal
    effect, plus the metadata needed to merge with readout rows. Errors are
    captured per-row (never silently dropped) via the `error` field.
    """
    n_layers = num_hidden_layers(model)
    layers = sorted({fraction_to_layer(model, f) for f in config.layer_fractions})
    rows: list[dict[str, Any]] = []

    for task in tasks:
        example_id = _task_field(task, "example_id")
        prompt = _task_field(task, "prompt")
        clean_prompt = _task_field(task, "clean_prompt") or prompt
        corrupt_prompt = _task_field(task, "corrupt_prompt") or prompt
        target_metric = _task_field(task, "target_metric")
        if isinstance(target_metric, dict):
            pos_tok = target_metric.get("positive_token")
            neg_tok = target_metric.get("negative_token")
        else:
            pos_tok = getattr(target_metric, "positive_token", None)
            neg_tok = getattr(target_metric, "negative_token", None)

        for layer in layers:
            layer_fraction = layer / max(1, n_layers - 1)
            for strength in config.strengths:
                try:
                    res = _patch_and_measure_with_strength(
                        model,
                        tokenizer,
                        clean_prompt,
                        corrupt_prompt,
                        layer,
                        strength,
                        pos_tok,
                        neg_tok,
                    )
                    raw_effect = normalize_causal_effect(res["delta"], res["baseline"])
                    clipped = min(1.0, abs(raw_effect))
                    rows.append(
                        {
                            "example_id": example_id,
                            "layer": layer,
                            "layer_fraction": layer_fraction,
                            "strength": strength,
                            "baseline_logit_diff": res["baseline"],
                            "patched_logit_diff": res["patched"],
                            "patch_delta": res["delta"],
                            "normalized_causal_effect_raw": raw_effect,
                            "normalized_causal_effect": clipped,
                            "is_causal": clipped >= config.causal_threshold,
                            "intervention_type": "residual_patch",
                            "error": None,
                        }
                    )
                except Exception as exc:
                    rows.append(
                        {
                            "example_id": example_id,
                            "layer": layer,
                            "layer_fraction": layer_fraction,
                            "strength": strength,
                            "baseline_logit_diff": None,
                            "patched_logit_diff": None,
                            "patch_delta": None,
                            "normalized_causal_effect_raw": None,
                            "normalized_causal_effect": None,
                            "is_causal": False,
                            "intervention_type": "residual_patch",
                            "error": str(exc),
                        }
                    )
    return rows
