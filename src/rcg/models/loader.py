"""Model loading utilities for RCG-Bench experiments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ModelConfig:
    """Configuration for loading an open-weight language model."""

    model_id: str
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False


def default_model_id() -> str:
    """Pick model from env or fall back to the main science model."""
    return os.environ.get("RCG_MODEL_ID", "google/gemma-3-4b-it")


def resolve_dtype(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(dtype, "auto")


class ModelLoader:
    """Load and cache language models for activation extraction and interventions."""

    def __init__(self, config: ModelConfig | None = None) -> None:
        if config is None:
            config = ModelConfig(model_id=default_model_id())
        self.config = config
        self._model: Any | None = None
        self._tokenizer: Any | None = None

    def load(self) -> tuple[Any, Any]:
        """Load model and tokenizer. Returns (model, tokenizer)."""
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            trust_remote_code=self.config.trust_remote_code,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs: dict[str, Any] = {
            "dtype": resolve_dtype(self.config.dtype),
            "trust_remote_code": self.config.trust_remote_code,
        }
        if torch.cuda.is_available():
            load_kwargs["device_map"] = (
                self.config.device if self.config.device != "auto" else "auto"
            )
        else:
            load_kwargs["device_map"] = None

        model = AutoModelForCausalLM.from_pretrained(self.config.model_id, **load_kwargs)
        if not torch.cuda.is_available():
            model = model.to("cpu")
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device
