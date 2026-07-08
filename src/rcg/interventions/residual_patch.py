"""Residual stream patching interventions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from rcg.interventions.causal_effects import intervention_delta, logit_diff
from rcg.models.hooks import capture_last_activation, layer_module


@dataclass
class PatchConfig:
    """Configuration for a residual patching intervention."""

    layer: int
    position: int = -1
    component: str = "resid_post"


class ResidualPatcher:
    """Patch residual activations from a counterfactual (corrupt) prompt."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def patch_and_measure(
        self,
        clean_prompt: str,
        corrupt_prompt: str,
        config: PatchConfig,
        positive_token: str,
        negative_token: str,
    ) -> dict[str, float]:
        """Run clean/corrupt patch and return baseline, patched, and delta logit diffs."""
        baseline = logit_diff(
            self.model, self.tokenizer, clean_prompt, positive_token, negative_token
        )
        corrupt_act = capture_last_activation(
            self.model, self.tokenizer, corrupt_prompt, config.layer, config.position
        )

        def patch_hook(_module: torch.nn.Module, _inputs: tuple, output: Any) -> Any:
            tensor = output[0] if isinstance(output, tuple) else output
            tensor = tensor.clone()
            tensor[0, config.position] = corrupt_act
            return (tensor,) if isinstance(output, tuple) else tensor

        handle = layer_module(self.model, config.layer).register_forward_hook(patch_hook)
        try:
            patched = logit_diff(
                self.model, self.tokenizer, clean_prompt, positive_token, negative_token
            )
        finally:
            handle.remove()

        delta = intervention_delta(baseline, patched)
        return {"baseline": baseline, "patched": patched, "delta": delta}
