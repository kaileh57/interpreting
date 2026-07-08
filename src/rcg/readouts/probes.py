"""Linear probe readout baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from rcg.models.hooks import capture_last_activation
from rcg.readouts.base import ReadoutResult


class ProbeReadout:
    """Supervised linear probes for ground-truth latent variables."""

    def __init__(self, model: Any, tokenizer: Any, layer: int) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.layer = layer
        self._probes: dict[str, LogisticRegression] = {}

    def fit(self, prompts: list[str], labels: list[str], variable: str) -> None:
        xs = []
        for prompt in prompts:
            act = capture_last_activation(self.model, self.tokenizer, prompt, self.layer)
            xs.append(act.cpu().float().numpy())
        x = np.stack(xs)
        y = np.array(labels)
        clf = LogisticRegression(max_iter=1000)
        clf.fit(x, y)
        self._probes[variable] = clf

    def predict(self, prompt: str, variable: str) -> ReadoutResult:
        if variable not in self._probes:
            raise KeyError(f"No probe trained for variable {variable!r}")
        act = capture_last_activation(self.model, self.tokenizer, prompt, self.layer)
        x = act.cpu().float().numpy().reshape(1, -1)
        clf = self._probes[variable]
        pred = clf.predict(x)[0]
        prob = float(clf.predict_proba(x).max())
        return ReadoutResult(method="probe", concept=str(pred), score=prob, confidence=prob)
