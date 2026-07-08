"""Persist RCG-Bench experiment outputs to results/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rcg.metrics.gap import failure_mode_counts
from rcg.metrics.stats import bootstrap_ci
from rcg.pipeline.evaluate import EvalResult


def results_dir(root: Path | None = None) -> Path:
    base = root or Path.cwd()
    for candidate in (base, base.parent):
        if (candidate / "src" / "rcg").exists():
            out = candidate / "results"
            out.mkdir(parents=True, exist_ok=True)
            return out
    out = base / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out


def summarize(results: list[EvalResult]) -> dict[str, Any]:
    by_method: dict[str, list[EvalResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)
    summary: dict[str, Any] = {"n_examples": len(results), "by_method": {}}
    for method, rows in by_method.items():
        rep = bootstrap_ci([r.reportability for r in rows])
        causal = bootstrap_ci([r.causal_effect for r in rows])
        gap = bootstrap_ci([r.rcg for r in rows])
        summary["by_method"][method] = {
            "mean_reportability": rep.mean,
            "reportability_ci": [rep.lo, rep.hi],
            "mean_causal_effect": causal.mean,
            "causal_effect_ci": [causal.lo, causal.hi],
            "mean_rcg": gap.mean,
            "rcg_ci": [gap.lo, gap.hi],
            "failure_modes": failure_mode_counts([r.failure_mode for r in rows]),
        }
    return summary


def summary_table(results: list[EvalResult]) -> list[dict[str, Any]]:
    """Flat per-method rows with CIs, ready for a DataFrame / paper table."""
    s = summarize(results)
    rows = []
    for method, m in s["by_method"].items():
        rows.append({
            "method": method,
            "reportability": round(m["mean_reportability"], 3),
            "reportability_ci": f"[{m['reportability_ci'][0]:.2f}, {m['reportability_ci'][1]:.2f}]",
            "causal_effect": round(m["mean_causal_effect"], 3),
            "causal_ci": f"[{m['causal_effect_ci'][0]:.2f}, {m['causal_effect_ci'][1]:.2f}]",
            "rcg": round(m["mean_rcg"], 3),
        })
    return rows


def save_run(
    name: str,
    results: list[EvalResult],
    extra: dict[str, Any] | None = None,
    root: Path | None = None,
) -> Path:
    """Write results/<name>.json (rows + summary) and results/<name>.csv."""
    out = results_dir(root)
    rows = [r.as_row() for r in results]
    summary = summarize(results)
    payload = {"experiment": name, "summary": summary, "rows": rows, "extra": extra or {}}
    json_path = out / f"{name}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if rows:
        csv_path = out / f"{name}.csv"
        headers = list(rows[0].keys())
        lines = [",".join(headers)]
        for row in rows:
            lines.append(",".join(str(row[h]).replace(",", ";") for h in headers))
        csv_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path
