"""Logit-lens readout baseline."""

from __future__ import annotations

from typing import Any

import torch

from rcg.models.hooks import capture_last_activation
from rcg.readouts.base import ReadoutResult


class LogitLensReadout:
    """Decode residual stream through unembedding at a given layer."""

    def __init__(self, model: Any, tokenizer: Any, layer: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer

    def top_k(self, prompt: str, k: int = 10) -> list[ReadoutResult]:
        hidden = capture_last_activation(self.model, self.tokenizer, prompt, self.layer)
        lm_head = self.model.get_output_embeddings()
        with torch.no_grad():
            logits = lm_head(hidden)
        top = torch.topk(logits, k)
        results: list[ReadoutResult] = []
        for score, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True):
            token = self.tokenizer.decode([idx]).strip()
            results.append(
                ReadoutResult(method="logit_lens", concept=token, score=float(score))
            )
        return results
