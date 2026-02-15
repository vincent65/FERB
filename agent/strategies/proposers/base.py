from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass
class ProposalContext:
    problem_id: int
    candidate_file: Path
    current_code: str
    latest_metrics: dict
    memory_summary: str
    proposal_mode: str
    eval_feedback: dict
    retrieved_docs: str = ""  # RLM-retrieved documentation context.
    failed_code: str = ""  # Code that was rolled back (if any), so the LLM knows what it tried.


class Proposer(Protocol):
    def propose(self, ctx: ProposalContext) -> dict:
        ...
