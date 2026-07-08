"""Sanity checks so a run's numbers are trustworthy (or flagged as noise).

If the base task doesn't produce a real logit signal, or no readout beats
chance, or no intervention moves the logit, the reportability–causality gap is
measuring noise. These checks make that explicit rather than silently reporting
meaningless means.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    checks: dict[str, bool] = {}
    details: dict[str, float] = {}
    warnings: list[str] = []

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

    return SanityReport(all(checks.values()), checks=checks, details=details, warnings=warnings)
