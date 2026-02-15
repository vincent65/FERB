from __future__ import annotations

from typing import TYPE_CHECKING

from agent.config import PromptConfig
from agent.openai_client import OpenAIPatchClient
from agent.strategies.memory.best_so_far import BestSoFarMemory
from agent.strategies.proposers.single_shot import SingleShotProposer
from agent.strategies.scorers.speedup_mean import SpeedupMeanScorer

if TYPE_CHECKING:
    from agent.strategies.retrieval.retriever import LLMRetriever


def make_proposer(
    name: str,
    openai_client: OpenAIPatchClient,
    prompt_config: PromptConfig,
    retriever: LLMRetriever | None = None,
):
    if name == "single_shot":
        return SingleShotProposer(openai_client, prompt_config, retriever=retriever)
    raise ValueError(f"Unknown proposer strategy: {name}")


def make_memory(name: str):
    if name == "best_so_far":
        return BestSoFarMemory()
    raise ValueError(f"Unknown memory strategy: {name}")


def make_scorer(name: str):
    if name == "speedup_mean":
        return SpeedupMeanScorer()
    raise ValueError(f"Unknown scorer strategy: {name}")
