"""J-lens / J-space readout for open models.

Two implementations:

* ``JLensReadout`` — fast proxy: cosine similarity between the residual
  activation and unembedding directions on a restricted vocabulary.
* ``GradientJLens`` — the plan's Week-3 approximation: build a vector
  ``j_{l,v}`` per vocabulary word as the corpus-averaged gradient of
  ``log p(v)`` w.r.t. the residual stream, then score by cosine similarity.
"""

from __future__ import annotations

from typing import Any

import torch

from rcg.models.hooks import capture_activation_grad, capture_last_activation
from rcg.readouts.base import ReadoutResult


def _first_token_id(tokenizer: Any, word: str) -> int | None:
    ids = tokenizer.encode(word, add_special_tokens=False)
    if not ids:
        ids = tokenizer.encode(" " + word.strip(), add_special_tokens=False)
    return ids[0] if ids else None


class JLensReadout:
    """Fast proxy J-lens via cosine to unembedding columns on a restricted vocab."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        layer: int,
        vocabulary: list[str] | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.vocabulary = vocabulary or []

    def top_k(self, prompt: str, k: int = 10) -> list[ReadoutResult]:
        if not self.vocabulary:
            raise ValueError("JLensReadout requires a restricted vocabulary list")
        hidden = capture_last_activation(self.model, self.tokenizer, prompt, self.layer)
        hidden = hidden.float()
        hidden = hidden / (hidden.norm() + 1e-8)
        lm_head = self.model.get_output_embeddings().weight
        scores: list[tuple[str, float]] = []
        for word in self.vocabulary:
            tid = _first_token_id(self.tokenizer, word)
            if tid is None:
                continue
            direction = lm_head[tid].float()
            direction = direction / (direction.norm() + 1e-8)
            scores.append((word, float(torch.dot(hidden, direction).item())))
        scores.sort(key=lambda x: x[1], reverse=True)
        return [ReadoutResult(method="jlens", concept=w, score=s) for w, s in scores[:k]]


class GradientJLens:
    """
    Gradient-based J-lens approximation (plan.md Week 3 / Detailed methodology).

    For each word v, average grad of log p(v) w.r.t. the residual at `layer`
    over a small corpus of contexts, producing a J-vector j_{l,v}. Score an
    activation by cosine(h, j_{l,v}).
    """

    def __init__(self, model: Any, tokenizer: Any, layer: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.jvectors: dict[str, torch.Tensor] = {}

    def build(self, vocabulary: list[str], corpus: list[str], normalize: bool = True) -> None:
        """Estimate j_{l,v} for each word by averaging gradients across the corpus."""
        for word in vocabulary:
            tid = _first_token_id(self.tokenizer, word)
            if tid is None:
                continue
            grads = []
            for context in corpus:
                grad = capture_activation_grad(
                    self.model, self.tokenizer, context, self.layer, tid
                )
                grads.append(grad.float())
            if not grads:
                continue
            vec = torch.stack(grads).mean(dim=0)
            if normalize:
                vec = vec / (vec.norm() + 1e-8)
            self.jvectors[word] = vec

    def top_k(self, prompt: str, k: int = 10) -> list[ReadoutResult]:
        if not self.jvectors:
            raise ValueError("Call GradientJLens.build(...) before top_k")
        hidden = capture_last_activation(self.model, self.tokenizer, prompt, self.layer).float()
        hidden = hidden / (hidden.norm() + 1e-8)
        scores = [
            (word, float(torch.dot(hidden, vec).item()))
            for word, vec in self.jvectors.items()
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [ReadoutResult(method="jlens_grad", concept=w, score=s) for w, s in scores[:k]]
