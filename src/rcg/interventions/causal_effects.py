"""Causal effect measurement utilities."""

from __future__ import annotations

from typing import Any

import torch


def _answer_logits(model: Any, tokenizer: Any, prompt: str) -> torch.Tensor:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    return logits[0, -1]


def token_id(tokenizer: Any, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Could not encode token text: {text!r}")
    return ids[0]


def logit_diff(
    model: Any,
    tokenizer: Any,
    prompt: str,
    positive_token: str,
    negative_token: str,
) -> float:
    """Compute logit(positive_token) - logit(negative_token) at the last position."""
    logits = _answer_logits(model, tokenizer, prompt)
    pos_id = token_id(tokenizer, positive_token)
    neg_id = token_id(tokenizer, negative_token)
    return (logits[pos_id] - logits[neg_id]).item()


def intervention_delta(baseline: float, intervened: float) -> float:
    """Return intervened - baseline metric delta."""
    return intervened - baseline


def normalize_causal_effect(delta: float, baseline: float, eps: float = 1e-8) -> float:
    """Normalize causal effect relative to baseline magnitude."""
    return delta / (abs(baseline) + eps)
