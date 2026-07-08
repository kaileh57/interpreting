"""Tuned-lens readout baseline (lightweight affine translator)."""

from __future__ import annotations

from typing import Any

import torch

from rcg.models.hooks import capture_last_activation, num_hidden_layers
from rcg.readouts.base import ReadoutResult


class TunedLensReadout:
    """
    Approximate tuned lens: learn a linear translator A mapping layer-`layer`
    activations to the final-layer residual, then decode with the unembedding.

    This is the standard tuned-lens idea (a learned per-layer translator that
    corrects the logit lens) in a minimal, calibratable form.
    """

    def __init__(self, model: Any, tokenizer: Any, layer: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self.final_layer = num_hidden_layers(model) - 1
        self.translator: torch.Tensor | None = None

    def calibrate(self, prompts: list[str], ridge: float = 1e-2) -> None:
        src, dst = [], []
        for prompt in prompts:
            h_l = capture_last_activation(self.model, self.tokenizer, prompt, self.layer)
            h_f = capture_last_activation(self.model, self.tokenizer, prompt, self.final_layer)
            src.append(h_l.float())
            dst.append(h_f.float())
        x = torch.stack(src)  # [n, d]
        y = torch.stack(dst)  # [n, d]
        d = x.shape[1]
        eye = torch.eye(d, device=x.device, dtype=x.dtype)
        gram = x.t() @ x + ridge * eye
        self.translator = torch.linalg.solve(gram, x.t() @ y)  # [d, d]

    def top_k(self, prompt: str, k: int = 10) -> list[ReadoutResult]:
        if self.translator is None:
            raise ValueError("Call TunedLensReadout.calibrate(...) before top_k")
        h = capture_last_activation(self.model, self.tokenizer, prompt, self.layer).float()
        translator = self.translator.to(device=h.device, dtype=h.dtype)
        translated = h @ translator
        lm_head = self.model.get_output_embeddings()
        with torch.no_grad():
            logits = lm_head(translated.to(lm_head.weight.dtype))
        top = torch.topk(logits, k)
        return [
            ReadoutResult(method="tuned_lens", concept=self.tokenizer.decode([idx]).strip(),
                          score=float(score))
            for score, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True)
        ]
