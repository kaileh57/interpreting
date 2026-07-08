"""Natural Language Autoencoder (NLA) readout interface.

NLAs (activation verbalizer + reconstructor) are external artifacts that are
not openly runnable in a single Colab kernel. This module provides a clean
interface: pass a `verbalizer` callable ``(activation) -> str`` to use a real
NLA, otherwise `is_available()` is False and callers should skip NLA rather
than crash. See sources/papers/02_nla.md.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from rcg.readouts.base import ReadoutResult

Verbalizer = Callable[[Any], str]


class NLAReadout:
    """Wrap an external activation verbalizer to produce natural-language readouts."""

    def __init__(self, verbalizer: Verbalizer | None = None) -> None:
        self.verbalizer = verbalizer

    def is_available(self) -> bool:
        return self.verbalizer is not None

    def verbalize(self, activation: Any) -> ReadoutResult:
        if self.verbalizer is None:
            raise RuntimeError(
                "No NLA verbalizer wired. Provide a verbalizer callable or skip NLA. "
                "See sources/papers/02_nla.md."
            )
        text = self.verbalizer(activation)
        return ReadoutResult(method="nla", concept=text, score=1.0, raw_output=text)
