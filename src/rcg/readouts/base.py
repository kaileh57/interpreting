"""Base types for readout methods."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ReadoutResult:
    """Result from a single readout method on one activation snapshot."""

    method: str
    concept: str
    score: float
    confidence: float | None = None
    raw_output: str | None = None
    metadata: dict = field(default_factory=dict)
