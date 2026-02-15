from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from agent.strategies.retrieval.corpus import CorpusEntry, TritonCorpus


class LLMRetriever:
    """LLM-based retriever that ranks corpus entries by relevance to a query.

    Used as the backend for the ``search_examples`` tool-use call in the
    proposer.  A lightweight LLM call ranks the available corpus entries and
    returns the top-k most relevant ones, fully formatted with code.
    """

    def __init__(
        self,
        corpus: TritonCorpus,
        model: str | None = None,
        *,
        default_top_k: int = 3,
    ) -> None:
        self.corpus = corpus
        self.model = model or "gpt-4o-mini"
        self.default_top_k = default_top_k
        self._client = OpenAI()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> str:
        """Rank corpus entries against *query* and return formatted top-k.

        Returns a formatted string of (reference, triton_solution) pairs ready
        to be injected as tool-call output back into the proposer conversation.
        """
        top_k = top_k or self.default_top_k
        entries = self.corpus.get_all_entries()
        if not entries:
            return "No corpus entries available."

        # If the corpus is small enough, skip ranking and return all.
        if len(entries) <= top_k:
            return TritonCorpus.format_entries(entries)

        selected = self._rank_and_select(query, entries, top_k)
        return TritonCorpus.format_entries(selected)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rank_and_select(
        self,
        query: str,
        entries: list[CorpusEntry],
        top_k: int,
    ) -> list[CorpusEntry]:
        """Ask an LLM to pick the *top_k* most relevant entries."""
        summaries = "\n".join(
            f"- [{e.problem_id}] {e.summary}" for e in entries
        )

        ranking_prompt = (
            "You are a code-retrieval assistant. Given the user's query and a list of "
            "available Triton kernel examples (each identified by a problem ID), return "
            "the IDs of the most relevant examples.\n\n"
            f"Query:\n{query}\n\n"
            f"Available examples:\n{summaries}\n\n"
            f"Return ONLY a JSON array of the top {top_k} problem IDs, most relevant first. "
            "Example: [10, 2, 1]"
        )

        request: dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": "system", "content": "You are a helpful code-retrieval assistant."},
                {"role": "user", "content": ranking_prompt},
            ],
        }
        # gpt-5 models reject the temperature parameter.
        if not self.model.startswith("gpt-5"):
            request["temperature"] = 0.0

        response = self._client.responses.create(**request)
        text = response.output_text.strip()

        selected_ids = self._parse_ids(text, top_k)
        id_to_entry = {e.problem_id: e for e in entries}
        selected = [id_to_entry[pid] for pid in selected_ids if pid in id_to_entry]

        # Fallback: if parsing failed or returned nothing, return the first top_k.
        if not selected:
            selected = entries[:top_k]

        return selected

    @staticmethod
    def _parse_ids(text: str, top_k: int) -> list[int]:
        """Parse a JSON array of ints from the LLM response."""
        # Try direct JSON parse first.
        try:
            ids = json.loads(text)
            if isinstance(ids, list):
                return [int(x) for x in ids[:top_k]]
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback: extract all integers from the text.
        found = re.findall(r"\d+", text)
        return [int(x) for x in found[:top_k]]
