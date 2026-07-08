"""Sanity checks so a run's numbers are trustworthy (or flagged as noise).

If the base task doesn't produce a real logit signal, or no readout beats
chance, or no intervention moves the logit, the reportability–causality gap is
measuring noise. These checks make that explicit rather than silently reporting
meaningless means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rcg.pipeline.evaluate import EvalResult


@dataclass
class SanityReport:
    passed: bool
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        head = "SANITY: PASS" if self.passed else "SANITY: WARN"
        lines = [head]
        for name, ok in self.checks.items():
            mark = "ok" if ok else "FAIL"
            extra = self.details.get(name)
            lines.append(f"  [{mark}] {name}" + (f" = {extra:.3f}" if extra is not None else ""))
        lines.extend(f"  ! {w}" for w in self.warnings)
        return "\n".join(lines)


def sanity_checks(
    results: list[EvalResult],
    chance_reportability: float = 0.5,
    min_baseline: float = 0.5,
    min_causal_rate: float = 0.1,
) -> SanityReport:
    """
    Validate a run:
      * mean |baseline logit diff| indicates the task actually discriminates,
      * the best readout beats chance reportability,
      * a non-trivial fraction of examples have a real causal effect.
    """
    real = [r for r in results if r.failure_mode != "error"]
    errors = [r for r in results if r.failure_mode == "error"]
    checks: dict[str, bool] = {}
    details: dict[str, float] = {}
    warnings: list[str] = []

    if errors:
        checks["no_eval_errors"] = False
        details["eval_error_count"] = float(len(errors))
        sample = errors[0].metadata.get("error", "unknown")
        warnings.append(
            f"{len(errors)} evaluation error(s) — excluded from means below. "
            f"First error ({errors[0].method}): {sample}"
        )
    else:
        checks["no_eval_errors"] = True
        details["eval_error_count"] = 0.0

    if not real:
        return SanityReport(False, warnings=["no successful results"])

    mean_base = sum(abs(r.baseline_logit_diff) for r in real) / len(real)
    details["mean_abs_baseline_logit_diff"] = mean_base
    checks["task_has_signal"] = mean_base >= min_baseline
    if not checks["task_has_signal"]:
        warnings.append(
            f"mean |baseline logit diff| = {mean_base:.3f} < {min_baseline}; the task may "
            "not discriminate the target — check prompts/tokens before trusting the gap."
        )

    by_method: dict[str, list[EvalResult]] = {}
    for r in real:
        by_method.setdefault(r.method, []).append(r)
    best_rep = max(sum(x.reportability for x in v) / len(v) for v in by_method.values())
    details["best_mean_reportability"] = best_rep
    details["chance_reportability"] = chance_reportability
    checks["readout_beats_chance"] = best_rep > chance_reportability
    if not checks["readout_beats_chance"]:
        warnings.append(
            f"best mean reportability {best_rep:.3f} <= chance {chance_reportability:.3f}; "
            "no readout recovers the latent above chance — reportability is uninformative."
        )

    causal_rate = sum(1 for r in real if r.causal_effect >= 0.5) / len(real)
    details["causal_effect_rate"] = causal_rate
    checks["intervention_moves_logit"] = causal_rate >= min_causal_rate
    if not checks["intervention_moves_logit"]:
        warnings.append(
            f"only {causal_rate:.1%} of examples show a real causal effect; interventions "
            "may be too weak or mis-located — check patch layer/strength."
        )

    return SanityReport(
        all(checks.values()) and not errors, checks=checks, details=details, warnings=warnings
    )


def _field(row: Any, *names: str, default: Any = None) -> Any:
    """Read the first present, non-None field from a dict row or an attribute-bearing object."""
    if isinstance(row, dict):
        for name in names:
            value = row.get(name)
            if value is not None:
                return value
        return default
    for name in names:
        value = getattr(row, name, None)
        if value is not None:
            return value
    return default


def _row_is_error(row: Any) -> bool:
    if _field(row, "failure_mode") == "error":
        return True
    error = _field(row, "error")
    return bool(error)


def structured_sanity(
    results_or_rows: list[Any],
    *,
    chance_reportability: float,
    causal_threshold: float = 0.5,
    min_baseline: float = 0.5,
    min_causal_rate: float = 0.1,
) -> dict[str, Any]:
    """
    Dict-returning sanity report for the decisive-rerun notebooks (10+).

    Unlike `sanity_checks` (which requires `EvalResult` objects), this accepts
    either `EvalResult`s or plain dict rows (e.g. a pandas `to_dict("records")`
    export), so it can validate the metadata-rich CSV rows those notebooks
    build directly. Always returns the same required keys so callers can save
    it straight into a manifest.
    """
    rows = list(results_or_rows)
    warnings: list[str] = []
    failures: list[str] = []

    errors = [r for r in rows if _row_is_error(r)]
    real = [r for r in rows if not _row_is_error(r)]
    no_eval_errors = len(errors) == 0
    if not no_eval_errors:
        warnings.append(
            f"{len(errors)} evaluation/intervention error(s) present — excluded from means below."
        )
        failures.append("no_eval_errors")

    if not real:
        return {
            "no_eval_errors": no_eval_errors,
            "task_has_signal": False,
            "readout_beats_chance": False,
            "intervention_moves_logit": False,
            "fraction_causal_examples": 0.0,
            "best_mean_reportability": 0.0,
            "chance_reportability": chance_reportability,
            "warnings": warnings + ["no successful rows"],
            "failures": failures
            + ["task_has_signal", "readout_beats_chance", "intervention_moves_logit"],
        }

    baselines = [abs(float(_field(r, "baseline_logit_diff", default=0.0) or 0.0)) for r in real]
    mean_base = sum(baselines) / len(baselines) if baselines else 0.0
    task_has_signal = mean_base >= min_baseline
    if not task_has_signal:
        warnings.append(f"mean |baseline logit diff| = {mean_base:.3f} < {min_baseline}")
        failures.append("task_has_signal")

    by_method: dict[str, list[float]] = {}
    for r in real:
        method = str(_field(r, "readout_method", "method", default="unknown"))
        rep = _field(r, "reportability_score", "reportability")
        if rep is None:
            continue
        by_method.setdefault(method, []).append(float(rep))
    best_rep = max((sum(v) / len(v) for v in by_method.values()), default=0.0)
    readout_beats_chance = best_rep > chance_reportability
    if not readout_beats_chance:
        warnings.append(
            f"best mean reportability {best_rep:.3f} <= chance {chance_reportability:.3f}"
        )
        failures.append("readout_beats_chance")

    causal_vals = []
    for r in real:
        ce = _field(r, "normalized_causal_effect", "causal_effect")
        if ce is not None:
            causal_vals.append(float(ce))
    causal_rate = (
        sum(1 for c in causal_vals if abs(c) >= causal_threshold) / len(causal_vals)
        if causal_vals
        else 0.0
    )
    intervention_moves_logit = causal_rate >= min_causal_rate
    if not intervention_moves_logit:
        warnings.append(
            f"only {causal_rate:.1%} of examples show causal effect >= {causal_threshold}"
        )
        failures.append("intervention_moves_logit")

    return {
        "no_eval_errors": no_eval_errors,
        "task_has_signal": task_has_signal,
        "readout_beats_chance": readout_beats_chance,
        "intervention_moves_logit": intervention_moves_logit,
        "fraction_causal_examples": causal_rate,
        "best_mean_reportability": best_rep,
        "chance_reportability": chance_reportability,
        "warnings": warnings,
        "failures": failures,
    }
