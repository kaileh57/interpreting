"""JSONL task/result IO and prompt hashing.

Used by the decisive-rerun notebooks (10+) to persist the exact task list a
run used (`runs/<run_id>/tasks.jsonl`) so every result row can be traced back
to the prompt that produced it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def stable_hash(text: str) -> str:
    """Deterministic short hash for prompts/ids (stable across processes and runs)."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def _as_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {}


def task_to_dict(
    task: Any,
    *,
    seed: int,
    index: int,
    task_family: str,
    model_id: str,
) -> dict[str, Any]:
    """
    Permissive task serializer for JSONL persistence.

    Works with pydantic `LatentSlotTask` instances, plain dicts (e.g. the
    prefill/hidden-objective example generators), or any object exposing the
    same attributes via `getattr`.
    """
    base = _as_dict(task)

    latent = _as_dict(base.get("latent_variables", {}))
    target_metric = _as_dict(base.get("target_metric"))

    prompt = base.get("prompt", "") or ""
    clean_prompt = base.get("clean_prompt") or prompt
    corrupt_prompt = base.get("corrupt_prompt") or ""

    row: dict[str, Any] = {
        "example_id": base.get("example_id") or f"{task_family}_seed{seed:03d}_i{index:04d}",
        "task_family": base.get("task_family") or task_family,
        "model_id": base.get("model_id") or model_id,
        "seed": seed,
        "index": index,
        "prompt": prompt,
        "clean_prompt": clean_prompt,
        "corrupt_prompt": corrupt_prompt,
        "prompt_hash": stable_hash(prompt),
        "clean_prompt_hash": stable_hash(clean_prompt),
        "corrupt_prompt_hash": stable_hash(corrupt_prompt),
        "positive_token": target_metric.get("positive_token"),
        "negative_token": target_metric.get("negative_token"),
        "target_metric_type": target_metric.get("type"),
    }
    for key, value in latent.items():
        row.setdefault(f"latent_{key}", value)
    return row


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    """Write one JSON object per line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=str) + "\n")
    return path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file back into a list of dicts."""
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
