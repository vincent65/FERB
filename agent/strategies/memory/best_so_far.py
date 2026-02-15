from __future__ import annotations

import json


class BestSoFarMemory:
    """Memory strategy that provides the LLM with a structured history of
    all prior iterations for the current problem, including what failed and
    why — not just the best score.
    """

    # Maximum number of recent entries to show in detail.
    _MAX_DETAIL_ENTRIES = 6

    def summarize(self, history: list[dict]) -> str:
        if not history:
            return "No prior iterations."

        best = max(history, key=lambda x: x.get("score", float("-inf")))
        latest = history[-1]

        parts: list[str] = []
        parts.append(
            f"Best iteration so far: iter {best.get('iteration')} "
            f"score={best.get('score', 0):.4f}."
        )
        parts.append(
            f"Latest iteration: iter {latest.get('iteration')} "
            f"score={latest.get('score', 0):.4f}."
        )

        # Build a concise per-iteration timeline.
        parts.append("")
        parts.append("Iteration history (most recent last):")
        recent = history[-self._MAX_DETAIL_ENTRIES:]
        for entry in recent:
            it = entry.get("iteration", "?")
            score = entry.get("score", 0)
            failed = entry.get("eval_failed", False)
            failure_kind = entry.get("eval_failure_kind", "")
            error_msg = entry.get("eval_error", "")

            if failed:
                # Truncate very long errors but keep enough for diagnosis.
                error_snippet = (error_msg[:300] + "...") if len(error_msg) > 300 else error_msg
                parts.append(
                    f"  iter {it}: FAILED ({failure_kind}) — {error_snippet}"
                )
            else:
                correctness = entry.get("candidate_feedback", {}).get("correctness", {})
                all_ok = correctness.get("all_ok", True)
                agg = entry.get("candidate_feedback", {}).get("aggregate", {})
                cand_ms = agg.get("candidate_mean_ms")
                ref_ms = agg.get("reference_mean_ms")
                timing_str = ""
                if cand_ms is not None and ref_ms is not None:
                    timing_str = f" (cand={cand_ms:.2f}ms, ref={ref_ms:.2f}ms)"
                correctness_str = "correct" if all_ok else "INCORRECT"
                parts.append(
                    f"  iter {it}: score={score:.4f}, {correctness_str}{timing_str}"
                )

        parts.append("")
        parts.append(
            "Avoid repeating approaches that previously failed. "
            "Build on what worked. Prioritize correctness, then improve speedup."
        )
        return "\n".join(parts)
