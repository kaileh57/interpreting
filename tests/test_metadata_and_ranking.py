"""Tests for the decisive-rerun run metadata and ranking helpers.

See RCG_AGENT_IMPLEMENTATION_INSTRUCTIONS.md acceptance tests.
"""

from __future__ import annotations

import json

from rcg.metrics.ranking import concept_rank, rank_metrics, topk_as_json
from rcg.pipeline.sanity import structured_sanity
from rcg.readouts.base import ReadoutResult
from rcg.tasks.generators import unique_latent_slot_examples
from rcg.tasks.task_io import stable_hash


def test_stable_hash_deterministic() -> None:
    assert stable_hash("hello world") == stable_hash("hello world")
    assert stable_hash("hello world") != stable_hash("hello world!")
    assert len(stable_hash("x")) == 16


def test_concept_rank_case_and_punctuation_tolerant() -> None:
    items = [
        ReadoutResult(method="m", concept="Tokyo,", score=0.9),
        ReadoutResult(method="m", concept=" Paris ", score=0.5),
    ]
    assert concept_rank(items, "tokyo") == 1
    assert concept_rank(items, "PARIS.") == 2
    assert concept_rank(items, "Cairo") is None


def test_topk_as_json_is_valid_json() -> None:
    items = [ReadoutResult(method="m", concept="Tokyo", score=0.9)]
    parsed = json.loads(topk_as_json(items, k=5))
    assert isinstance(parsed, list)
    assert parsed[0]["concept"] == "Tokyo"


def test_rank_metrics_with_distractor() -> None:
    items = [
        ReadoutResult(method="m", concept="Paris", score=0.9),
        ReadoutResult(method="m", concept="Tokyo", score=0.4),
    ]
    result = rank_metrics(items, target="Tokyo", distractor="Paris", k=5)
    assert result["target_rank"] == 2
    assert result["distractor_rank"] == 1
    assert result["target_in_topk"] is True
    assert result["distractor_in_topk"] is True
    assert result["target_minus_distractor_score"] == 0.4 - 0.9
    assert json.loads(result["topk_json"])


def test_generated_example_ids_unique_across_seeds() -> None:
    tasks = unique_latent_slot_examples(n_per_seed=10, seeds=[1, 2, 3])
    ids = [t.example_id for t in tasks]
    assert len(ids) == 30
    assert len(ids) == len(set(ids)), "example_id collision across seeds"


def test_structured_sanity_returns_required_keys() -> None:
    rows = [
        {
            "reportability_score": 1.0, "normalized_causal_effect": 0.8,
            "baseline_logit_diff": 1.2, "readout_method": "jlens", "error": None,
        },
        {
            "reportability_score": 0.0, "normalized_causal_effect": 0.1,
            "baseline_logit_diff": 0.9, "readout_method": "jlens", "error": None,
        },
    ]
    result = structured_sanity(rows, chance_reportability=0.2, causal_threshold=0.5)
    required = {
        "no_eval_errors", "task_has_signal", "readout_beats_chance",
        "intervention_moves_logit", "fraction_causal_examples",
        "best_mean_reportability", "chance_reportability", "warnings", "failures",
    }
    assert required.issubset(result.keys())
    assert result["chance_reportability"] == 0.2


def test_structured_sanity_flags_eval_errors() -> None:
    rows = [
        {"reportability_score": 1.0, "normalized_causal_effect": 0.8,
         "baseline_logit_diff": 1.0, "readout_method": "jlens", "error": None},
        {"reportability_score": None, "normalized_causal_effect": None,
         "baseline_logit_diff": None, "readout_method": "jlens", "error": "boom"},
    ]
    result = structured_sanity(rows, chance_reportability=0.2)
    assert result["no_eval_errors"] is False
    assert "no_eval_errors" in result["failures"]
