from __future__ import annotations

from typing import Protocol


class MemoryStrategy(Protocol):
    def summarize(self, history: list[dict]) -> str:
        ...
