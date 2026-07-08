"""SAE / transcoder feature readout baseline."""

from __future__ import annotations

from typing import Any

import torch

from rcg.readouts.base import ReadoutResult


class SAEFeatureReadout:
    """
    Read top activated SAE features and their natural-language descriptions.

    Pass an `sae` object exposing ``encode()`` (or precomputed feature
    activations as a dict). Full Gemma Scope 2 integration via `load_gemma_scope_sae`.
    """

    def __init__(self, sae: Any, feature_descriptions: dict[int, str] | None = None) -> None:
        self.sae = sae
        self.feature_descriptions = feature_descriptions or {}

    def top_k(self, activations: torch.Tensor | dict, k: int = 10) -> list[ReadoutResult]:
        if hasattr(self.sae, "encode"):
            features = self.sae.encode(activations)
            if isinstance(features, tuple):
                features = features[0]
            flat = features.flatten()
        elif isinstance(activations, dict):
            flat = torch.tensor(list(activations.values()))
        else:
            raise ValueError("Provide sae.encode() or a feature activation dict")

        top = torch.topk(flat, min(k, flat.numel()))
        results: list[ReadoutResult] = []
        for score, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True):
            fid = int(idx)
            label = self.feature_descriptions.get(fid, f"feature_{fid}")
            results.append(
                ReadoutResult(method="sae", concept=label, score=float(score), metadata={"id": fid})
            )
        return results


def gemma_scope_available() -> bool:
    try:
        import sae_lens  # noqa: F401

        return True
    except Exception:
        return False


def load_gemma_scope_sae(
    release: str = "gemma-scope-2b-pt-res-canonical",
    sae_id: str = "layer_12/width_16k/canonical",
    device: str = "cpu",
) -> Any:
    """
    Load a Gemma Scope SAE via sae_lens. Raises an informative error if the
    `sae` optional dependency is not installed (``pip install .[sae]``).
    """
    if not gemma_scope_available():
        raise RuntimeError(
            "sae_lens not installed. Install with `pip install .[sae]` and see "
            "sources/tooling/gemma_scope2.md."
        )
    from sae_lens import SAE

    sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    return sae[0] if isinstance(sae, tuple) else sae
