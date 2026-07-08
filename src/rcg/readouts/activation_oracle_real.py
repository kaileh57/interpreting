"""Explicit real Activation Oracle wrapper — no fake fallback.

Unlike `rcg.readouts.activation_oracle.SelfProbeOracle` (a runnable stand-in,
clearly labeled `self_probe_oracle`), this wrapper never silently substitutes
anything. If a real checkpoint isn't available, `status()` explains why and
`query_forced_choice` raises; callers must catch that and record the result as
`activation_oracle_not_run`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class AOStatus:
    available: bool
    reason: str | None
    checkpoint: str | None


def _activation_oracles_importable() -> bool:
    try:
        import activation_oracles  # noqa: F401

        return True
    except Exception:
        return False


class RealActivationOracleReadout:
    """
    Wraps a real Activation Oracle checkpoint (adamkarvonen/activation_oracles).

    Resolves the checkpoint from the constructor argument, then
    `RCG_AO_CHECKPOINT`, then `RCG_AO_MODEL_ID`. Requires a separate
    environment pinning `transformers==4.55.2`
    (see sources/tooling/activation_oracles.md).
    """

    def __init__(
        self,
        model: Any = None,
        tokenizer: Any = None,
        checkpoint: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint = (
            checkpoint
            or os.environ.get("RCG_AO_CHECKPOINT")
            or os.environ.get("RCG_AO_MODEL_ID")
        )
        self._oracle: Any = None
        self._status = self._probe_status()

    def _probe_status(self) -> AOStatus:
        if not self.checkpoint:
            return AOStatus(
                available=False,
                reason="No RCG_AO_CHECKPOINT or RCG_AO_MODEL_ID set.",
                checkpoint=None,
            )
        if not _activation_oracles_importable():
            return AOStatus(
                available=False,
                reason=(
                    "activation_oracles package not installed (separate env; pins "
                    "transformers==4.55.2). See sources/tooling/activation_oracles.md."
                ),
                checkpoint=self.checkpoint,
            )
        try:
            from activation_oracles import Oracle  # type: ignore[import-not-found]

            self._oracle = Oracle.from_pretrained(self.checkpoint)
            return AOStatus(available=True, reason=None, checkpoint=self.checkpoint)
        except Exception as exc:
            return AOStatus(
                available=False,
                reason=f"Failed to load Activation Oracle checkpoint {self.checkpoint!r}: {exc}",
                checkpoint=self.checkpoint,
            )

    def status(self) -> AOStatus:
        return self._status

    def query_forced_choice(
        self,
        activation_or_prompt: Any,
        question: str,
        choices: list[str],
    ) -> dict[str, Any]:
        """Ask the real oracle a forced-choice question. Raises if unavailable."""
        if not self._status.available or self._oracle is None:
            raise RuntimeError(f"RealActivationOracleReadout unavailable: {self._status.reason}")

        answer = self._oracle.answer(activation_or_prompt, question, choices=choices)
        if isinstance(answer, dict):
            concept = str(answer.get("answer", ""))
            score = float(answer.get("score", 1.0))
        else:
            concept = str(answer)
            score = 1.0
        return {
            "method": "activation_oracle_forced_choice",
            "concept": concept,
            "score": score,
            "choices": choices,
            "question": question,
        }
