"""Unified RCG-Bench evaluation pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from rcg.interventions.causal_effects import logit_diff, normalize_causal_effect
from rcg.interventions.residual_patch import PatchConfig, ResidualPatcher
from rcg.metrics.gap import failure_mode, rcg_gap
from rcg.metrics.reportability import reportability_score
from rcg.readouts.base import ReadoutResult
from rcg.tasks.latent_slot import LatentSlotTask

ReadoutFn = Callable[[str], list[ReadoutResult]]


@dataclass
class EvalResult:
    example_id: str
    method: str
    reportability: float
    causal_effect: float
    rcg: float
    failure_mode: str
    baseline_logit_diff: float
    patch_delta: float
    readout_top: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "method": self.method,
            "reportability": self.reportability,
            "causal_effect": self.causal_effect,
            "rcg": self.rcg,
            "failure_mode": self.failure_mode,
            "baseline_logit_diff": self.baseline_logit_diff,
            "patch_delta": self.patch_delta,
            "readout_top": ", ".join(self.readout_top),
        }


def _ground_truth(task: LatentSlotTask) -> str:
    lv = task.latent_variables
    return str(lv.get("hidden_city") or lv.get("target_item") or lv.get("target_format") or "")


def _causal_effect_for_task(
    model: Any,
    tokenizer: Any,
    task: LatentSlotTask,
    layer: int,
    baseline: float,
) -> float:
    """Residual-patch causal effect, normalized to [0, 1]."""
    tm = task.target_metric
    gt = _ground_truth(task)
    clean = task.clean_prompt or task.prompt
    if task.corrupt_prompt:
        corrupt = task.corrupt_prompt
    else:
        corrupt_city = task.latent_variables.get("corrupt_city", "Paris")
        corrupt = task.prompt.replace(gt, str(corrupt_city)) if gt else task.prompt
    patcher = ResidualPatcher(model, tokenizer)
    patched = patcher.patch_and_measure(
        clean, corrupt, PatchConfig(layer=layer), tm.positive_token, tm.negative_token
    )
    causal = min(1.0, abs(normalize_causal_effect(patched["delta"], baseline)))
    return causal, patched["delta"]


def evaluate_latent_slot(
    model: Any,
    tokenizer: Any,
    task: LatentSlotTask,
    readout_fn: ReadoutFn,
    layer: int,
    method: str = "readout",
) -> EvalResult:
    """Run one readout + residual patch on a latent-slot task."""
    gt = _ground_truth(task)
    tm = task.target_metric
    baseline = logit_diff(model, tokenizer, task.prompt, tm.positive_token, tm.negative_token)

    readout: list[ReadoutResult] = readout_fn(task.prompt)
    rep = reportability_score(readout, gt, k=5)
    causal, patch_delta = _causal_effect_for_task(model, tokenizer, task, layer, baseline)

    return EvalResult(
        example_id=task.example_id,
        method=method,
        reportability=rep,
        causal_effect=causal,
        rcg=rcg_gap(rep, causal),
        failure_mode=failure_mode(rep, causal),
        baseline_logit_diff=baseline,
        patch_delta=patch_delta,
        readout_top=[r.concept for r in readout[:5]],
    )


def evaluate_tasks(
    model: Any,
    tokenizer: Any,
    tasks: list[LatentSlotTask],
    readouts: dict[str, ReadoutFn],
    layer: int,
) -> list[EvalResult]:
    """
    Evaluate readout methods across tasks. The causal effect is computed once
    per task (it does not depend on the readout) and reused across methods.
    """
    results: list[EvalResult] = []
    for task in tasks:
        gt = _ground_truth(task)
        tm = task.target_metric
        try:
            baseline = logit_diff(
                model, tokenizer, task.prompt, tm.positive_token, tm.negative_token
            )
            causal, patch_delta = _causal_effect_for_task(
                model, tokenizer, task, layer, baseline
            )
        except Exception as exc:
            for method in readouts:
                results.append(_error_result(task.example_id, method, str(exc)))
            continue

        for method, fn in readouts.items():
            try:
                readout: list[ReadoutResult] = fn(task.prompt)
                rep = reportability_score(readout, gt, k=5)
                results.append(
                    EvalResult(
                        example_id=task.example_id,
                        method=method,
                        reportability=rep,
                        causal_effect=causal,
                        rcg=rcg_gap(rep, causal),
                        failure_mode=failure_mode(rep, causal),
                        baseline_logit_diff=baseline,
                        patch_delta=patch_delta,
                        readout_top=[r.concept for r in readout[:5]],
                    )
                )
            except Exception as exc:
                results.append(_error_result(task.example_id, method, str(exc)))
    return results


def _error_result(example_id: str, method: str, error: str) -> EvalResult:
    return EvalResult(
        example_id=example_id,
        method=method,
        reportability=0.0,
        causal_effect=0.0,
        rcg=0.0,
        failure_mode="error",
        baseline_logit_diff=0.0,
        patch_delta=0.0,
        metadata={"error": error},
    )
