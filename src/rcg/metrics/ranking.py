"""Rank/margin metrics for readout concepts vs. a target and optional distractor.

Reportability-only (in-top-k) scoring hides *how* readable a concept was. These
helpers let notebooks 10/11 report exact rank, score, and margin between the
ground-truth concept and a named distractor concept.
"""

from __future__ import annotations

import json
import re
from typing import Any

_STRIP_RE = re.compile(r"^[\s.,;:!?'\"()\[\]]+|[\s.,;:!?'\"()\[\]]+$")


def normalize_concept(s: Any) -> str:
    """Lowercase and strip surrounding whitespace/punctuation for robust matching."""
    if s is None:
        return ""
    text = str(s).strip().lower()
    return _STRIP_RE.sub("", text)


def _concept_of(item: Any) -> str:
    if isinstance(item, dict):
        return item.get("concept", "")
    return getattr(item, "concept", "")


def _score_of(item: Any) -> float:
    if isinstance(item, dict):
        return float(item.get("score", 0.0))
    return float(getattr(item, "score", 0.0))


def concept_rank(readout_items: list[Any], concept: str) -> int | None:
    """1-indexed rank of `concept` in `readout_items` (already assumed sorted best-first)."""
    target = normalize_concept(concept)
    if not target:
        return None
    for i, item in enumerate(readout_items):
        if normalize_concept(_concept_of(item)) == target:
            return i + 1
    return None


def concept_score(readout_items: list[Any], concept: str) -> float | None:
    """Score assigned to `concept`, or None if it does not appear in `readout_items`."""
    target = normalize_concept(concept)
    if not target:
        return None
    for item in readout_items:
        if normalize_concept(_concept_of(item)) == target:
            return _score_of(item)
    return None


def topk_as_json(readout_items: list[Any], k: int = 10) -> str:
    """Serialize the top-k `(concept, score)` pairs as a JSON string for CSV storage."""
    payload = [{"concept": _concept_of(x), "score": _score_of(x)} for x in readout_items[:k]]
    return json.dumps(payload)


def rank_metrics(
    readout_items: list[Any],
    *,
    target: str,
    distractor: str | None = None,
    k: int = 10,
) -> dict[str, Any]:
    """
    Rank/score/margin of `target` (and optionally `distractor`) within `readout_items`.

    Returns a dict with `target_rank`, `target_score`, `target_in_topk`,
    `distractor_rank`, `distractor_score`, `distractor_in_topk`,
    `target_minus_distractor_score`, and `topk_json`.
    """
    target_rank = concept_rank(readout_items, target)
    target_score = concept_score(readout_items, target)
    target_in_topk = target_rank is not None and target_rank <= k

    result: dict[str, Any] = {
        "target_rank": target_rank,
        "target_score": target_score,
        "target_in_topk": target_in_topk,
        "distractor_rank": None,
        "distractor_score": None,
        "distractor_in_topk": False,
        "target_minus_distractor_score": None,
        "topk_json": topk_as_json(readout_items, k),
    }

    if distractor:
        distractor_rank = concept_rank(readout_items, distractor)
        distractor_score = concept_score(readout_items, distractor)
        result["distractor_rank"] = distractor_rank
        result["distractor_score"] = distractor_score
        result["distractor_in_topk"] = distractor_rank is not None and distractor_rank <= k
        if target_score is not None and distractor_score is not None:
            result["target_minus_distractor_score"] = target_score - distractor_score

    return result
