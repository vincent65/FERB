from __future__ import annotations

from typing import Protocol


class Scorer(Protocol):
    def score(self, reference_summary: dict, candidate_summary: dict) -> float:
        ...
