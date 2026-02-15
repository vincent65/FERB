from __future__ import annotations

import json
from typing import TYPE_CHECKING

from agent.config import PromptConfig
from agent.openai_client import OpenAIPatchClient
from agent.strategies.proposers.base import ProposalContext

if TYPE_CHECKING:
    from agent.strategies.retrieval.retriever import LLMRetriever


class SingleShotProposer:
    def __init__(
        self,
        client: OpenAIPatchClient,
        prompt_config: PromptConfig,
        retriever: LLMRetriever | None = None,
    ):
        self.client = client
        self.prompt_config = prompt_config
        self.retriever = retriever

    def propose(self, ctx: ProposalContext) -> dict:
        template = self.prompt_config.performance_patch_template
        if ctx.proposal_mode == "correctness_fix":
            template = self.prompt_config.correctness_patch_template

        # Build the failed-code context block when available.
        failed_code_block = ""
        if ctx.failed_code:
            failed_code_block = (
                "\n--- FAILED CODE FROM PREVIOUS ATTEMPT ---\n"
                "The following code was attempted in the previous iteration but "
                "was rolled back due to the error shown in eval_feedback above. "
                "Analyze what went wrong and avoid repeating the same mistake.\n"
                "```python\n"
                f"{ctx.failed_code}\n"
                "```\n"
                "--- END FAILED CODE ---\n"
            )

        prompt = template.format(
            problem_id=ctx.problem_id,
            candidate_file=str(ctx.candidate_file),
            memory_summary=ctx.memory_summary,
            retrieved_docs=ctx.retrieved_docs or "No documentation retrieved.",
            latest_metrics=json.dumps(ctx.latest_metrics, indent=2, default=str),
            eval_feedback=json.dumps(ctx.eval_feedback, indent=2, default=str),
            current_code=ctx.current_code,
        )

        # Append the failed code block after the template so the LLM can
        # compare the working code (current_code) with the broken attempt.
        if failed_code_block:
            prompt += failed_code_block

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
