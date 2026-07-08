"""Best-effort Gemma Scope 2 SAE / transcoder integration — no fake fallback.

Unlike `rcg.readouts.sae_features.SAEFeatureReadout` (which requires a
pre-loaded `sae` object, e.g. a `SyntheticSAE` for smoke testing), this
wrapper resolves a real Gemma Scope artifact from env vars via `sae_lens` and
never silently substitutes a synthetic one. If unavailable, `status()`
explains why and callers must record the result as `sae_not_run`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SAEStatus:
    available: bool
    reason: str | None
    artifact_id: str | None


def _sae_lens_importable() -> bool:
    try:
        import sae_lens  # noqa: F401

        return True
    except Exception:
        return False


def _env_int(name: str) -> int | None:
    val = os.environ.get(name)
    if not val:
        return None
    try:
        return int(val)
    except ValueError:
        return None


class GemmaScopeSAEReadout:
    """
    Best-effort Gemma Scope 2 SAE integration via `sae_lens`.

    Resolves `artifact_id` from the constructor argument or `RCG_SAE_MODEL_ID`,
    `layer` from the constructor argument or `RCG_SAE_LAYER`, and the hook site
    from `RCG_SAE_SITE` (default `resid_post`, currently informational only —
    `sae_lens` release/sae_id strings determine the actual hook point).
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        layer: int | None = None,
        artifact_id: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer if layer is not None else _env_int("RCG_SAE_LAYER")
        self.artifact_id = artifact_id or os.environ.get("RCG_SAE_MODEL_ID")
        self.site = os.environ.get("RCG_SAE_SITE", "resid_post")
        self._sae: Any = None
        self._status = self._probe_status()

    def _probe_status(self) -> SAEStatus:
        if not self.artifact_id:
            return SAEStatus(available=False, reason="No RCG_SAE_MODEL_ID set.", artifact_id=None)
        if not _sae_lens_importable():
            return SAEStatus(
                available=False,
                reason="sae_lens not installed. Install with `pip install .[sae]`.",
                artifact_id=self.artifact_id,
            )
        try:
            from sae_lens import SAE

            kwargs: dict[str, Any] = {"release": self.artifact_id, "device": "cpu"}
            if self.layer is not None:
                kwargs["sae_id"] = f"layer_{self.layer}/width_16k/canonical"
            loaded = SAE.from_pretrained(**kwargs)
            self._sae = loaded[0] if isinstance(loaded, tuple) else loaded
            return SAEStatus(available=True, reason=None, artifact_id=self.artifact_id)
        except Exception as exc:
            return SAEStatus(
                available=False,
                reason=f"Failed to load SAE artifact {self.artifact_id!r}: {exc}",
                artifact_id=self.artifact_id,
            )

    def status(self) -> SAEStatus:
        return self._status

    def _activation(self, prompt: str) -> torch.Tensor:
        from rcg.models.hooks import capture_last_activation

        if self.model is None or self.tokenizer is None or self.layer is None:
            raise RuntimeError(
                "GemmaScopeSAEReadout requires model, tokenizer, and layer to encode activations"
            )
        return capture_last_activation(self.model, self.tokenizer, prompt, self.layer)

    def top_features(self, prompt: str, k: int = 10) -> list[dict[str, Any]]:
        """
        Top-k SAE feature IDs and raw activations for `prompt`.

        Descriptions are left `None`: without a loaded Neuronpedia/feature
        description table we do not guess semantic labels from feature IDs.
        """
        if not self._status.available or self._sae is None:
            raise RuntimeError(f"GemmaScopeSAEReadout unavailable: {self._status.reason}")
        act = self._activation(prompt).float()
        features = self._sae.encode(act.unsqueeze(0))
        if isinstance(features, tuple):
            features = features[0]
        flat = features.flatten()
        top = torch.topk(flat, min(k, flat.numel()))
        return [
            {"feature_id": int(idx), "activation": float(score), "description": None}
            for score, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True)
        ]

    def ablate_and_measure(
        self,
        prompt: str,
        feature_ids: list[int],
        positive_token: str,
        negative_token: str,
    ) -> dict[str, Any]:
        """Zero-ablate `feature_ids`, reconstruct, and re-measure logit diff. Never guesses."""
        if not self._status.available or self._sae is None:
            return {"available": False, "reason": self._status.reason or "sae_not_available"}
        if not hasattr(self._sae, "decode"):
            return {"available": False, "reason": "ablation_not_implemented"}
        try:
            from rcg.interventions.causal_effects import intervention_delta, logit_diff
            from rcg.models.hooks import layer_module

            baseline = logit_diff(
                self.model, self.tokenizer, prompt, positive_token, negative_token
            )
            act = self._activation(prompt).float()
            features = self._sae.encode(act.unsqueeze(0))
            if isinstance(features, tuple):
                features = features[0]
            features = features.clone()
            for fid in feature_ids:
                features[..., fid] = 0.0
            recon = self._sae.decode(features).squeeze(0)

            def hook(_module: torch.nn.Module, _inputs: tuple, output: Any) -> Any:
                tensor = output[0] if isinstance(output, tuple) else output
                tensor = tensor.clone()
                tensor[0, -1] = recon.to(tensor.dtype)
                return (tensor,) if isinstance(output, tuple) else tensor

            handle = layer_module(self.model, self.layer).register_forward_hook(hook)
            try:
                patched = logit_diff(
                    self.model, self.tokenizer, prompt, positive_token, negative_token
                )
            finally:
                handle.remove()

            delta = intervention_delta(baseline, patched)
            return {
                "available": True,
                "method": "sae_feature_ablation",
                "baseline": baseline,
                "patched": patched,
                "delta": delta,
                "feature_ids": feature_ids,
            }
        except Exception as exc:
            return {"available": False, "reason": f"ablation_failed: {exc}"}
