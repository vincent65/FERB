from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agent.config import PromptConfig
from agent.llm_client import LLMPatchClient
from agent.strategies.proposers.base import IterationSnapshot, ProposalContext

if TYPE_CHECKING:
    from agent.strategies.retrieval.retriever import LLMRetriever

# Maximum number of prior kernels to include in full.
# Beyond this, only the most recent ones are shown in full; older ones are
# summarized with just their score/error status.
_MAX_FULL_KERNELS = 4


class SingleShotProposer:
    def __init__(
        self,
        client: LLMPatchClient,
        prompt_config: PromptConfig,
        retriever: LLMRetriever | None = None,
    ):
        self.client = client
        self.prompt_config = prompt_config
        self.retriever = retriever

    @staticmethod
    def _build_kernel_history_block(
        prior_kernels: list[IterationSnapshot] | None,
    ) -> str:
        """Build the Kernel Devin-style history block.

        Each prior kernel is presented with its code and evaluation result so
        the LLM can see the full optimization trajectory and learn from every
        attempt — successful or not.
        """
        if not prior_kernels:
            return ""

        parts: list[str] = []
        parts.append("=" * 60)
        parts.append("PRIOR KERNEL HISTORY (oldest first)")
        parts.append(
            "Study each attempt and its evaluation result. "
            "Build on what worked, avoid repeating what failed."
        )
        parts.append("=" * 60)

        n = len(prior_kernels)
        for i, snap in enumerate(prior_kernels):
            parts.append(f"\n--- KERNEL #{snap.iteration} ---")

            # For older kernels beyond the window, show summary only.
            if n > _MAX_FULL_KERNELS and i < (n - _MAX_FULL_KERNELS):
                if snap.eval_failed:
                    parts.append(
                        f"Score: 0.0 (FAILED: {snap.eval_error[:200]})"
                    )
                else:
                    parts.append(f"Score: {snap.score:.4f}")
                parts.append("[code omitted for brevity — see recent kernels below]")
                continue

            # Show full code + eval info for recent kernels.
            parts.append(f"Score: {snap.score:.4f}")
            if snap.eval_failed:
                parts.append(f"Status: FAILED ({snap.eval_error[:300]})")
            else:
                # Extract useful eval signals.
                fb = snap.eval_feedback
                status = fb.get("status", "unknown")
                correctness = fb.get("correctness", {})
                aggregate = fb.get("aggregate", {})
                parts.append(f"Status: {status}")
                if not correctness.get("all_ok", True):
                    parts.append(
                        f"Correctness: FAILED (ranks: {correctness.get('failed_ranks', [])})"
                    )
                if aggregate:
                    cand_ms = aggregate.get("candidate_mean_ms")
                    ref_ms = aggregate.get("reference_mean_ms")
                    if cand_ms is not None and ref_ms is not None:
                        parts.append(
                            f"Timing: candidate={cand_ms:.3f}ms, reference={ref_ms:.3f}ms"
                        )
                # Include rank-level errors if present (e.g. shape mismatch details).
                error_msg = fb.get("error", "")
                if error_msg:
                    parts.append(f"Error: {error_msg[:300]}")

            parts.append("```python")
            parts.append(snap.code.strip())
            parts.append("```")

        parts.append("\n" + "=" * 60)
        parts.append("END PRIOR KERNEL HISTORY")
        parts.append("=" * 60 + "\n")
        return "\n".join(parts)

    def propose(self, ctx: ProposalContext) -> dict:
        template = self.prompt_config.performance_patch_template
        if ctx.proposal_mode == "correctness_fix":
            template = self.prompt_config.correctness_patch_template

        prompt = template.format(
            problem_id=ctx.problem_id,
            candidate_file=str(ctx.candidate_file),
            memory_summary=ctx.memory_summary,
            retrieved_docs=ctx.retrieved_docs or "No documentation retrieved.",
            latest_metrics=json.dumps(ctx.latest_metrics, indent=2, default=str),
            eval_feedback=json.dumps(ctx.eval_feedback, indent=2, default=str),
            current_code=ctx.current_code,
        )

        # ── Kernel Devin pattern: append full history of prior kernels ──
        kernel_history = self._build_kernel_history_block(ctx.prior_kernels)
        if kernel_history:
            prompt += kernel_history

        if self.retriever is not None:
            proposal = self.client.propose_patch_with_tools(prompt, self.retriever)
        else:
            proposal = self.client.propose_patch(prompt)

        return {
            "diagnosis": proposal.diagnosis,
            "hypotheses": proposal.hypotheses,
            "candidate_code": proposal.candidate_code,
            "test_expectations": proposal.test_expectations,
        }
