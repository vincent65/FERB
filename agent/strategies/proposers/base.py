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


class Proposer(Protocol):
    def propose(self, ctx: ProposalContext) -> dict:
        ...
