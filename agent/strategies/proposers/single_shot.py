from __future__ import annotations

import json

from agent.config import PromptConfig
from agent.openai_client import OpenAIPatchClient
from agent.strategies.proposers.base import ProposalContext


class SingleShotProposer:
    def __init__(self, client: OpenAIPatchClient, prompt_config: PromptConfig):
        self.client = client
        self.prompt_config = prompt_config

    def propose(self, ctx: ProposalContext) -> dict:
        template = self.prompt_config.performance_patch_template
        if ctx.proposal_mode == "correctness_fix":
            template = self.prompt_config.correctness_patch_template

        prompt = template.format(
            problem_id=ctx.problem_id,
            candidate_file=str(ctx.candidate_file),
            memory_summary=ctx.memory_summary,
            latest_metrics=json.dumps(ctx.latest_metrics, indent=2, default=str),
            eval_feedback=json.dumps(ctx.eval_feedback, indent=2, default=str),
            current_code=ctx.current_code,
        )
        proposal = self.client.propose_patch(prompt)
        return {
            "diagnosis": proposal.diagnosis,
            "hypotheses": proposal.hypotheses,
            "candidate_code": proposal.candidate_code,
            "test_expectations": proposal.test_expectations,
        }
