"""Behavioral self-report readout."""

from __future__ import annotations

from typing import Any

from rcg.readouts.base import ReadoutResult


class SelfReportReadout:
    """Query the model directly about its internal state or intentions."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def ask(self, context_prompt: str, question: str, max_new_tokens: int = 32) -> ReadoutResult:
        prompt = f"{context_prompt.strip()}\n\n{question.strip()}\nAnswer:"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        import torch

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        text = self.tokenizer.decode(
            output_ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()
        return ReadoutResult(
            method="self_report",
            concept=text,
            score=1.0,
            raw_output=text,
        )
