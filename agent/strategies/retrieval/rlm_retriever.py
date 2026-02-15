from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from rlm import RLM

from agent.strategies.retrieval.docs_loader import DocsCorpus

if TYPE_CHECKING:
    from agent.config import RetrievalConfig


# Default system prompt for the RLM documentation retriever.
_DEFAULT_RLM_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a documentation retrieval assistant for Triton/NVSHMEM GPU kernel
    optimization.  You have access to the full NVSHMEM documentation corpus via
    the ``context`` variable.

    Given a retrieval query that describes a specific kernel optimization
    problem, the current proposal mode (performance optimization or correctness
    fix), and any error or eval feedback, your job is to:

    1. Search through the NVSHMEM documentation in ``context``.
    2. Identify the most relevant API references, code examples, usage patterns,
       and caveats that would help an engineer fix or optimize the kernel.
    3. Return a concise but complete excerpt of the relevant documentation
       sections.  Include function signatures, parameter descriptions, and code
       snippets where applicable.

    Do NOT summarize vaguely — quote the actual documentation content so the
    engineer can use it directly.  Keep your final answer under 4000 characters.
""")


class RLMDocRetriever:
    """Uses the RLM library to recursively search NVSHMEM docs for relevant content.

    The retriever loads the docs corpus once and reuses it across calls.  Each
    ``retrieve()`` invocation makes a single ``RLM.completion()`` call, which
    internally may spawn sub-LLM queries to process large documentation.
    """

    def __init__(self, docs_corpus: DocsCorpus, cfg: RetrievalConfig) -> None:
        self.corpus = docs_corpus
        self.cfg = cfg

        system_prompt = cfg.rlm_custom_system_prompt or _DEFAULT_RLM_SYSTEM_PROMPT

        self.rlm = RLM(
            backend="openai",
            backend_kwargs={"model_name": cfg.retrieval_model or "gpt-4o-mini"},
            max_iterations=cfg.rlm_max_iterations,
            max_depth=cfg.rlm_max_depth,
            custom_system_prompt=system_prompt,
            verbose=False,
        )

        # Pre-build the context payload once (docs don't change between iterations).
        self._context_payload = self.corpus.as_context_payload()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> str:
        """Use RLM to find documentation relevant to *query*.

        Returns a string of relevant documentation excerpts ready to be
        injected into the proposer prompt.  On failure returns a short
        fallback message so the proposer is never blocked.
        """
        try:
            payload = {**self._context_payload, "query": query}
            result = self.rlm.completion(prompt=payload, root_prompt=query)
            text = result.response.strip() if result.response else ""
            if text:
                return text
            return "RLM retrieval returned no results."
        except Exception as exc:  # noqa: BLE001
            return f"RLM retrieval failed: {exc}"
