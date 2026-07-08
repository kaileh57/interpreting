"""Activation Oracle readout baseline."""

from __future__ import annotations

import os
from typing import Any

from rcg.readouts.base import ReadoutResult


class ActivationOracleReadout:
    """
    Query an Activation Oracle with forced-choice or open-ended prompts.

    Activation Oracles require a separate environment (transformers==4.55.2) and
    pretrained checkpoints, so they are not runnable inside the main Colab
    kernel. Provide a loaded `oracle` object (exposing ``answer(acts, question,
    choices=...)``) or set RCG_AO_CHECKPOINT. Use `is_available()` to skip AO
    gracefully. See sources/tooling/activation_oracles.md.
    """

    def __init__(self, oracle: Any | None = None, checkpoint: str | None = None) -> None:
        self.oracle = oracle
        self.checkpoint = checkpoint or os.environ.get("RCG_AO_CHECKPOINT")

    def is_available(self) -> bool:
        return self.oracle is not None

    def query(
        self,
        activations: Any,
        question: str,
        choices: list[str] | None = None,
    ) -> ReadoutResult:
        if self.oracle is None:
            raise RuntimeError(
                "Activation Oracle not loaded. Provide an `oracle` or skip AO. "
                "See sources/tooling/activation_oracles.md."
            )
        answer = self.oracle.answer(activations, question, choices=choices)
        concept = answer if isinstance(answer, str) else str(answer.get("answer", ""))
        return ReadoutResult(
            method="activation_oracle",
            concept=concept,
            score=1.0,
            raw_output=concept,
        )


def activation_oracles_available() -> bool:
    """True if the external adamkarvonen/activation_oracles package is importable."""
    try:
        import activation_oracles  # noqa: F401

        return True
    except Exception:
        return False


def load_activation_oracle(checkpoint: str, device: str = "cuda") -> Any:
    """
    Load a real Activation Oracle from the external package.

    Requires a separate env pinning transformers==4.55.2 and the
    adamkarvonen/activation_oracles package. Raises an informative error
    otherwise. See sources/tooling/activation_oracles.md.
    """
    if not activation_oracles_available():
        raise RuntimeError(
            "activation_oracles not installed. It pins transformers==4.55.2 and must "
            "run in a separate env; do not merge into the main rcg env. "
            "See sources/tooling/activation_oracles.md."
        )
    from activation_oracles import Oracle  # type: ignore

    return Oracle.from_pretrained(checkpoint, device=device)


class SelfProbeOracle:
    """
    Runnable forced-choice baseline that stands in for a real Activation Oracle.

    NOT a real AO: instead of reading another model's activations, it asks the
    subject model itself a forced-choice question and scores the option with the
    highest first-token logit. It exercises the exact forced-choice reportability
    pipeline so notebook 09 produces real numbers without external checkpoints.
    Clearly labeled `self_probe_oracle` so it is never confused with the real AO.
    """

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model = model
        self.tokenizer = tokenizer

    def answer(
        self,
        context_prompt: str,
        question: str,
        choices: list[str] | None = None,
    ) -> dict[str, Any]:
        import torch

        if not choices:
            raise ValueError("SelfProbeOracle requires forced-choice `choices`")
        letters = [chr(ord("A") + i) for i in range(len(choices))]
        options = "\n".join(f"{ltr}. {c}" for ltr, c in zip(letters, choices, strict=True))
        prompt = (
            f"{context_prompt.strip()}\n\n{question.strip()}\n{options}\n"
            "Answer with the single letter only. Answer:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, -1]
        best_letter, best_score = letters[0], float("-inf")
        for ltr in letters:
            ids = self.tokenizer.encode(ltr, add_special_tokens=False)
            if not ids:
                ids = self.tokenizer.encode(" " + ltr, add_special_tokens=False)
            if not ids:
                continue
            score = float(logits[ids[0]].item())
            if score > best_score:
                best_letter, best_score = ltr, score
        idx = letters.index(best_letter)
        return {"answer": choices[idx], "letter": best_letter, "score": best_score}


class SelfProbeOracleReadout:
    """Wrap SelfProbeOracle to return a ReadoutResult with method `self_probe_oracle`."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.oracle = SelfProbeOracle(model, tokenizer)

    def query(
        self,
        context_prompt: str,
        question: str,
        choices: list[str],
    ) -> ReadoutResult:
        out = self.oracle.answer(context_prompt, question, choices=choices)
        return ReadoutResult(
            method="self_probe_oracle",
            concept=out["answer"],
            score=out["score"],
            raw_output=out["letter"],
        )
