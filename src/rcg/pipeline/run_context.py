"""Centralized run identity, output directories, and manifests.

Every decisive-rerun notebook (10+) creates one `RunContext` at startup so every
saved artifact — tasks, results, figures, manifest — lives under a single
`runs/<run_id>/` directory and carries the same run metadata.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _repo_commit(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _find_repo_root(root: Path | None = None) -> Path:
    base = root or Path.cwd()
    for candidate in (base, base.parent, *base.parents):
        if (candidate / "src" / "rcg").exists():
            return candidate
    return base


def _slug(notebook: str) -> str:
    return notebook.replace(".ipynb", "").strip()


@dataclass
class RunContext:
    """Identity and output paths for one notebook execution."""

    run_id: str
    notebook: str
    model_id: str | None
    created_at_utc: str
    repo_commit: str | None
    root: Path
    run_dir: Path
    results_dir: Path
    figures_dir: Path
    tasks_path: Path
    manifest_path: Path


def create_run_context(
    notebook: str,
    model_id: str | None = None,
    root: Path | None = None,
) -> RunContext:
    """Create a fresh run directory (`runs/<run_id>/{results,figures}/`) and identity."""
    repo_root = _find_repo_root(root)
    slug = _slug(notebook)
    created = datetime.now(UTC)
    run_id = f"{created.strftime('%Y%m%d_%H%M%S')}_{slug}"

    run_dir = repo_root / "runs" / run_id
    results_dir = run_dir / "results"
    figures_dir = run_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return RunContext(
        run_id=run_id,
        notebook=slug,
        model_id=model_id,
        created_at_utc=created.isoformat(),
        repo_commit=_repo_commit(repo_root),
        root=repo_root,
        run_dir=run_dir,
        results_dir=results_dir,
        figures_dir=figures_dir,
        tasks_path=run_dir / "tasks.jsonl",
        manifest_path=run_dir / "manifest.json",
    )


def write_manifest(ctx: RunContext, payload: dict[str, Any]) -> Path:
    """Write `runs/<run_id>/manifest.json`, always including base run identity fields."""
    base: dict[str, Any] = {
        "run_id": ctx.run_id,
        "notebook": ctx.notebook,
        "model_id": ctx.model_id,
        "created_at_utc": ctx.created_at_utc,
        "repo_commit": ctx.repo_commit,
    }
    base.update(payload)
    ctx.manifest_path.write_text(json.dumps(base, indent=2, default=str), encoding="utf-8")
    return ctx.manifest_path
