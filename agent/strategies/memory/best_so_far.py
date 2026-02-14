from __future__ import annotations


class BestSoFarMemory:
    def summarize(self, history: list[dict]) -> str:
        if not history:
            return "No prior iterations."
        best = max(history, key=lambda x: x.get("score", float("-inf")))
        latest = history[-1]
        return (
            f"Best iteration: {best.get('iteration')} score={best.get('score'):.4f}. "
            f"Latest iteration: {latest.get('iteration')} score={latest.get('score'):.4f}. "
            "Avoid regressions while improving speedup."
        )
