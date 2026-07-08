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
        self.bias: torch.Tensor | None = None
        self.mu: torch.Tensor | None = None
        self.sd: torch.Tensor | None = None

    def calibrate(self, prompts: list[str], ridge: float = 1e-1) -> None:
        """
        Fit an affine translator ``y ≈ (x - mu)/sd @ A + b`` from layer-`layer`
        to final-layer activations.

        Gemma-style residual streams have a few massive-magnitude dimensions that
        make raw activations nearly collinear, so the naive normal-equations Gram
        matrix is numerically singular. To stay well-conditioned we (1) standardize
        features per dimension, (2) scale the ridge to the data magnitude, and
        (3) solve in float64. With n < d we use the dual (kernel) form so the
        system is n×n instead of d×d.
        """
        src, dst = [], []
        for prompt in prompts:
            h_l = capture_last_activation(self.model, self.tokenizer, prompt, self.layer)
            h_f = capture_last_activation(self.model, self.tokenizer, prompt, self.final_layer)
            src.append(h_l.float())
            dst.append(h_f.float())
        x = torch.stack(src).double()  # [n, d]
        y = torch.stack(dst).double()  # [n, d]
        n, d = x.shape
        if n < 2:
            raise ValueError("TunedLensReadout.calibrate needs at least 2 prompts")

        # Standardize inputs so no single massive-activation dimension dominates.
        self.mu = x.mean(dim=0, keepdim=True)
        self.sd = x.std(dim=0, keepdim=True).clamp_min(1e-6)
        xs = (x - self.mu) / self.sd

        # Center targets; recover the offset as a bias term after solving.
        y_mean = y.mean(dim=0, keepdim=True)
        yc = y - y_mean

        if n <= d:
            gram = xs @ xs.t()  # [n, n]
            # Floor the ridge so gram + lam·I is positive-definite even when the
            # standardized features are exactly collinear (lam would else be 0).
            lam = ridge * gram.diagonal().mean().clamp_min(1.0)
            gram = gram + lam * torch.eye(n, dtype=torch.float64, device=x.device)
            dual = torch.linalg.solve(gram, yc)  # [n, d]
            translator = xs.t() @ dual  # [d, d]
        else:
            gram = xs.t() @ xs  # [d, d]
            lam = ridge * gram.diagonal().mean().clamp_min(1.0)
            gram = gram + lam * torch.eye(d, dtype=torch.float64, device=x.device)
            translator = torch.linalg.solve(gram, xs.t() @ yc)  # [d, d]

        self.translator = translator.to(torch.float32)
        self.bias = y_mean.squeeze(0).to(torch.float32)
        self.mu = self.mu.squeeze(0).to(torch.float32)
        self.sd = self.sd.squeeze(0).to(torch.float32)

    def top_k(self, prompt: str, k: int = 10) -> list[ReadoutResult]:
        if self.translator is None:
            raise ValueError("Call TunedLensReadout.calibrate(...) before top_k")
        h = capture_last_activation(self.model, self.tokenizer, prompt, self.layer).float()
        mu = self.mu.to(device=h.device, dtype=h.dtype)
        sd = self.sd.to(device=h.device, dtype=h.dtype)
        translator = self.translator.to(device=h.device, dtype=h.dtype)
        bias = self.bias.to(device=h.device, dtype=h.dtype)
        translated = ((h - mu) / sd) @ translator + bias
        lm_head = self.model.get_output_embeddings()
        with torch.no_grad():
            logits = lm_head(translated.to(lm_head.weight.dtype))
        top = torch.topk(logits, k)
        return [
            ReadoutResult(method="tuned_lens", concept=self.tokenizer.decode([idx]).strip(),
                          score=float(score))
            for score, idx in zip(top.values.tolist(), top.indices.tolist(), strict=True)
        ]
