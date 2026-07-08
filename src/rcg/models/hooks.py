"""Shared hook utilities for activation access."""

from __future__ import annotations

from typing import Any

import torch


def decoder_layers(model: Any) -> torch.nn.ModuleList:
    """
    Return the ModuleList of decoder layers across architectures:

      * text models (Gemma3ForCausalLM, Llama, Qwen): model.model.layers
      * multimodal wrappers (Gemma3ForConditionalGeneration 4B/12B/27B):
        model.model.language_model.layers or model.language_model.model.layers
      * GPT-2: model.transformer.h
    """
    candidates = [
        lambda m: m.model.layers,
        lambda m: m.model.language_model.layers,
        lambda m: m.language_model.model.layers,
        lambda m: m.language_model.layers,
        lambda m: m.transformer.h,
        lambda m: m.gpt_neox.layers,
    ]
    for get in candidates:
        try:
            layers = get(model)
        except AttributeError:
            continue
        if layers is not None and len(layers) > 0:
            return layers
    raise ValueError(
        f"Could not locate decoder layers for model type {type(model).__name__}"
    )


def num_hidden_layers(model: Any) -> int:
    """Resolve layer count across architectures (incl. multimodal text_config)."""
    cfg = model.config
    text_cfg = getattr(cfg, "text_config", None)
    for source in (text_cfg, cfg):
        if source is None:
            continue
        for attr in ("num_hidden_layers", "n_layer", "num_layers"):
            value = getattr(source, attr, None)
            if value is not None:
                return int(value)
    return len(decoder_layers(model))


def middle_layer(model: Any) -> int:
    """A safe default intervention/readout layer."""
    return max(1, num_hidden_layers(model) // 2)


def layer_module(model: Any, layer: int) -> torch.nn.Module:
    return decoder_layers(model)[layer]


def _hidden_from_output(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


def capture_last_activation(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer: int,
    position: int = -1,
) -> torch.Tensor:
    """Capture the residual activation at `position` after `layer` (no grad)."""
    captured: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: tuple, output: Any) -> None:
        tensor = _hidden_from_output(output)
        captured["act"] = tensor[0, position].detach()

    handle = layer_module(model, layer).register_forward_hook(hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
    finally:
        handle.remove()
    return captured["act"]


def capture_activation_grad(
    model: Any,
    tokenizer: Any,
    prompt: str,
    layer: int,
    target_token_id: int,
    position: int = -1,
) -> torch.Tensor:
    """
    Gradient of log p(target_token) at the final position w.r.t. the residual
    activation at `position` after `layer`. Used to build J-lens vectors.
    """
    stash: dict[str, torch.Tensor] = {}

    def hook(_module: torch.nn.Module, _inputs: tuple, output: Any) -> Any:
        tensor = _hidden_from_output(output)
        tensor.retain_grad()
        stash["act"] = tensor
        return output

    handle = layer_module(model, layer).register_forward_hook(hook)
    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        model.zero_grad(set_to_none=True)
        logits = model(**inputs).logits
        log_probs = torch.log_softmax(logits[0, -1], dim=-1)
        log_probs[target_token_id].backward()
        grad = stash["act"].grad[0, position].detach().clone()
    finally:
        handle.remove()
    return grad
