"""Task schemas for ground-truth latent-variable benchmarks."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TargetMetric(BaseModel):
    """Behavioral metric specification for an example."""

    type: str = Field(description="Metric type, e.g. logit_diff, exact_match, classifier_score")
    positive_token: str | None = None
    negative_token: str | None = None
    target_answer: str | None = None


class ReadoutRecord(BaseModel):
    """Stored readout outputs for an evaluated example."""

    jlens_topk: list[str] = Field(default_factory=list)
    activation_oracle_answer: str | None = None
    sae_features: list[str] = Field(default_factory=list)
    probe_scores: dict[str, float] = Field(default_factory=dict)
    self_report_answer: str | None = None


class InterventionRecord(BaseModel):
    """Stored intervention effect deltas for an evaluated example."""

    model_config = {"extra": "allow"}

    jspace_tokyo_to_paris_delta: float | None = None
    sae_tokyo_ablation_delta: float | None = None
    paris_distractor_ablation_delta: float | None = None


class LatentSlotTask(BaseModel):
    """
    Ground-truth latent slot task example.

    Matches the RCG-Bench dataset schema in plan.md.
    """

    example_id: str
    model_id: str
    prompt: str
    latent_variables: dict[str, Any]
    target_metric: TargetMetric
    readouts: ReadoutRecord = Field(default_factory=ReadoutRecord)
    interventions: InterventionRecord = Field(default_factory=InterventionRecord)
    task_family: str | None = None
    clean_prompt: str | None = None
    corrupt_prompt: str | None = None

    @classmethod
    def example_city_currency(cls) -> LatentSlotTask:
        """Minimal example matching plan.md JSON schema."""
        return cls(
            example_id="city_currency_00042",
            model_id="gemma-3-1b-it",
            prompt=(
                "Silently choose one city from this list: Paris, Tokyo, Cairo, Lima.\n"
                "Do not reveal the city. Use it only to answer the next question.\n"
                "Question: The country containing the city has the currency ___"
            ),
            latent_variables={
                "hidden_city": "Tokyo",
                "target_country": "Japan",
                "target_currency": "yen",
                "distractor_city": "Paris",
            },
            target_metric=TargetMetric(
                type="logit_diff",
                positive_token=" yen",
                negative_token=" euro",
            ),
            task_family="city_country_currency",
        )
