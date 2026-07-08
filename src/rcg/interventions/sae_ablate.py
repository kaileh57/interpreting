"""SAE feature ablation and steering interventions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from rcg.interventions.causal_effects import intervention_delta, logit_diff
from rcg.models.hooks import layer_module


@dataclass
class SAEAblationConfig:
    """Configuration for SAE feature ablation, clamping, or steering."""

    layer: int
    feature_ids: list[int]
    mode: str = "ablate"  # ablate | clamp | steer
    strength: float = 1.0
    position: int = -1
    clamp_values: dict[int, float] = field(default_factory=dict)


class SAEAblator:
    """
    Ablate, clamp, or steer SAE/transcoder features during the forward pass.

    `sae` must expose ``encode(acts) -> features`` and ``decode(features) -> acts``.
    The intervention edits features at `position` and splices the SAE
    reconstruction delta back into the residual stream, then measures the
    change in a logit-difference metric.
    """

    def __init__(self, model: Any, tokenizer: Any, sae: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sae = sae

    def _edit_features(self, features: torch.Tensor, config: SAEAblationConfig) -> torch.Tensor:
        edited = features.clone()
        for fid in config.feature_ids:
            if config.mode == "ablate":
                edited[..., fid] = 0.0
            elif config.mode == "clamp":
                edited[..., fid] = config.clamp_values.get(fid, config.strength)
            elif config.mode == "steer":
                edited[..., fid] = edited[..., fid] + config.strength
            else:
                raise ValueError(f"Unknown SAE intervention mode: {config.mode}")
        return edited

    def intervene_and_measure(
        self,
        prompt: str,
        config: SAEAblationConfig,
        positive_token: str,
        negative_token: str,
    ) -> dict[str, float]:
        """Apply the SAE intervention and return baseline, intervened, and delta."""
        baseline = logit_diff(self.model, self.tokenizer, prompt, positive_token, negative_token)

        def hook(_module: torch.nn.Module, _inputs: tuple, output: Any) -> Any:
            tensor = output[0] if isinstance(output, tuple) else output
            tensor = tensor.clone()
            act = tensor[0, config.position].float()
            features = self.sae.encode(act)
            if isinstance(features, tuple):
                features = features[0]
            recon = self.sae.decode(features)
            edited = self._edit_features(features, config)
            recon_edited = self.sae.decode(edited)
            delta_vec = (recon_edited - recon).to(tensor.dtype)
            tensor[0, config.position] = tensor[0, config.position] + delta_vec
            return (tensor,) if isinstance(output, tuple) else tensor

        handle = layer_module(self.model, config.layer).register_forward_hook(hook)
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


class SyntheticSAE:
    """
    A lightweight dictionary SAE fit from cached activations (PCA-style).

    Stand-in so the SAE experiment runs anywhere Gemma Scope 2 is unavailable.
    Not a trained SAE; clearly labeled and used only for pipeline demonstration.
    """

    def __init__(self, dictionary: torch.Tensor, mean: torch.Tensor) -> None:
        self.dictionary = dictionary  # [n_features, d_model], orthonormal rows
        self.mean = mean

    @classmethod
    def fit(cls, activations: torch.Tensor, n_features: int = 32) -> SyntheticSAE:
        acts = activations.float()
        mean = acts.mean(dim=0)
        centered = acts - mean
        # SVD gives orthonormal components as a proxy sparse dictionary.
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        n = min(n_features, vh.shape[0])
        return cls(dictionary=vh[:n], mean=mean)

    def encode(self, act: torch.Tensor) -> torch.Tensor:
        centered = act.float() - self.mean
        return self.dictionary @ centered

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.dictionary.t() @ features + self.mean
