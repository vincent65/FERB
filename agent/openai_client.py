"""Legacy compatibility shim — everything now lives in ``agent.llm_client``.

Import ``LLMPatchClient`` (aliased as ``OpenAIPatchClient``) and
``PatchProposal`` from the new unified module so that existing code and tests
that import from ``agent.openai_client`` continue to work.
"""

from agent.llm_client import LLMPatchClient as OpenAIPatchClient  # noqa: F401
from agent.llm_client import PatchProposal  # noqa: F401

__all__ = ["OpenAIPatchClient", "PatchProposal"]
