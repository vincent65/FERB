from __future__ import annotations


class SpeedupMeanScorer:
    def score(self, reference_summary: dict, candidate_summary: dict) -> float:
        ref_mean = reference_summary.get("aggregate", {}).get("reference_mean_ms")
        cand_mean = candidate_summary.get("aggregate", {}).get("candidate_mean_ms")

        # Worker summaries can also only contain candidate/reference sections, depending on backend.
        if ref_mean is None:
            ref_mean = reference_summary.get("aggregate", {}).get("candidate_mean_ms")
        if cand_mean is None:
            cand_mean = candidate_summary.get("aggregate", {}).get("candidate_mean_ms")

        if not ref_mean or not cand_mean:
            return 0.0
        return float(ref_mean) / float(cand_mean)
