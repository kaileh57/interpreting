"""DAS-style causal-variable localization baseline (1-D distributed alignment).

A minimal Distributed Alignment Search: learn a single causal direction that
separates a binary latent variable, usable both as a readout (project + sign)
and as an intervention (patch the activation along that direction toward the
counterfactual class).
"""

from __future__ import annotations

from typing import Any

import torch

from rcg.interventions.causal_effects import intervention_delta, logit_diff
from rcg.models.hooks import capture_last_activation, layer_module
from rcg.readouts.base import ReadoutResult


class DASReadout:
    """Learn one alignment direction for a binary latent variable."""

    def __init__(self, model: Any, tokenizer: Any, layer: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.direction: torch.Tensor | None = None
        self.mu_pos: torch.Tensor | None = None
        self.mu_neg: torch.Tensor | None = None
        self.pos_label: str = "pos"
        self.neg_label: str = "neg"

    def fit(self, prompts: list[str], labels: list[str]) -> None:
        classes = sorted(set(labels))
        if len(classes) != 2:
            raise ValueError("DASReadout supports binary latent variables only")
        self.neg_label, self.pos_label = classes
        acts = {c: [] for c in classes}
        for prompt, label in zip(prompts, labels, strict=True):
            act = capture_last_activation(self.model, self.tokenizer, prompt, self.layer).float()
            acts[label].append(act)
        self.mu_pos = torch.stack(acts[self.pos_label]).mean(dim=0)
        self.mu_neg = torch.stack(acts[self.neg_label]).mean(dim=0)
        direction = self.mu_pos - self.mu_neg
        self.direction = direction / (direction.norm() + 1e-8)

    def predict(self, prompt: str) -> ReadoutResult:
        if self.direction is None:
            raise ValueError("Call DASReadout.fit(...) before predict")
        act = capture_last_activation(self.model, self.tokenizer, prompt, self.layer).float()
        midpoint = 0.5 * (self.mu_pos + self.mu_neg)
        proj = float(torch.dot(act - midpoint, self.direction).item())
        label = self.pos_label if proj > 0 else self.neg_label
        return ReadoutResult(method="das", concept=label, score=abs(proj), confidence=abs(proj))

    def intervene_and_measure(
        self,
        prompt: str,
        positive_token: str,
        negative_token: str,
        toward: str,
        strength: float = 4.0,
    ) -> dict[str, float]:
        """Patch the activation along the DAS direction toward a target class."""
        if self.direction is None:
            raise ValueError("Call DASReadout.fit(...) before intervene")
        baseline = logit_diff(self.model, self.tokenizer, prompt, positive_token, negative_token)
        sign = 1.0 if toward == self.pos_label else -1.0
        direction = (sign * strength * self.direction).to(
            self.model.get_output_embeddings().weight.dtype
        )

        def hook(_module: torch.nn.Module, _inputs: tuple, output: Any) -> Any:
            tensor = output[0] if isinstance(output, tuple) else output
            tensor = tensor.clone()
            tensor[0, -1] = tensor[0, -1] + direction
            return (tensor,) if isinstance(output, tuple) else tensor

        handle = layer_module(self.model, self.layer).register_forward_hook(hook)
        try:
            intervened = logit_diff(
                self.model, self.tokenizer, prompt, positive_token, negative_token
            )
        finally:
            handle.remove()
        return {
            "baseline": baseline,
            "intervened": intervened,
            "delta": intervention_delta(baseline, intervened),
        }
