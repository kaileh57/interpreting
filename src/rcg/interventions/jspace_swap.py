"""J-space coordinate swap interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rcg.interventions.causal_effects import intervention_delta, logit_diff
from rcg.models.hooks import layer_module
from rcg.readouts.jlens import JLensReadout  # noqa: F401 — optional full J-lens path


@dataclass
class JSpaceSwapConfig:
    """Configuration for swapping one J-lens coordinate for another."""

    layer: int
    subtract_concept: str
    add_concept: str
    vocabulary: list[str]
    alpha: float = 1.0
    beta: float = 1.0


class JSpaceSwapper:
    """Swap J-space coordinates: h' = h - α proj_subtract(h) + β j_add."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.lm_head = model.get_output_embeddings().weight

    def _direction(self, concept: str) -> torch.Tensor:
        token_ids = self.tokenizer.encode(concept, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Unknown concept token: {concept}")
        direction = self.lm_head[token_ids[0]].detach()
        return direction / (direction.norm() + 1e-8)

    def swap_and_measure(
        self,
        prompt: str,
        config: JSpaceSwapConfig,
        positive_token: str,
        negative_token: str,
    ) -> dict[str, float]:
        baseline = logit_diff(
            self.model, self.tokenizer, prompt, positive_token, negative_token
        )
        sub_dir = self._direction(config.subtract_concept)
        add_dir = self._direction(config.add_concept)

        def hook(_module: torch.nn.Module, _inputs: tuple, output: Any) -> Any:
            tensor = output[0] if isinstance(output, tuple) else output
            tensor = tensor.clone()
            h = tensor[0, -1]
            proj = torch.dot(h, sub_dir) * sub_dir
            h_new = h - config.alpha * proj + config.beta * add_dir
            tensor[0, -1] = h_new
            return (tensor,) if isinstance(output, tuple) else tensor

        handle = layer_module(self.model, config.layer).register_forward_hook(hook)
        try:
            swapped = logit_diff(
                self.model, self.tokenizer, prompt, positive_token, negative_token
            )
        finally:
            handle.remove()
        delta = intervention_delta(baseline, swapped)
        return {"baseline": baseline, "swapped": swapped, "delta": delta}
