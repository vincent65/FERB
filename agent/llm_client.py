"""Unified LLM client that supports both OpenAI and Anthropic (Claude) backends.

The public interface mirrors the original ``OpenAIPatchClient`` so that all
callers (optimizer, proposer, bootstrap) work unchanged.  The ``provider``
parameter selects which SDK to use under the hood.

Config example (YAML)::

    llm:
      provider: anthropic          # or "openai" (default)
      model: claude-sonnet-4-5-20250929
      temperature: 0.2

Environment variables:
    - ``OPENAI_API_KEY``    – required when provider is "openai"
    - ``ANTHROPIC_API_KEY`` – required when provider is "anthropic"
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.strategies.retrieval.retriever import LLMRetriever


# ── Data model (unchanged from openai_client.py) ──────────────────────


@dataclass
class PatchProposal:
    diagnosis: list[str]
    hypotheses: list[str]
    candidate_code: str  # Complete rewritten file content
    test_expectations: list[str]

    @property
    def proposed_patch(self) -> str:
        return self.candidate_code


# ── Helpers ────────────────────────────────────────────────────────────


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _parse_patch_json(text: str) -> PatchProposal:
    text = _strip_code_fences(text.strip())
    if not text:
        raise ValueError(
            "LLM returned an empty response (no text content). "
            "The model may not have produced a final JSON answer."
        )
    # Some models wrap JSON in prose — try to extract the JSON object.
    if not text.startswith("{"):
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start : end + 1]
    data: dict[str, Any] = json.loads(text)
    candidate_code = str(data.get("candidate_code", data.get("proposed_patch", "")))
    return PatchProposal(
        diagnosis=list(data.get("diagnosis", [])),
        hypotheses=list(data.get("hypotheses", [])),
        candidate_code=candidate_code,
        test_expectations=list(data.get("test_expectations", [])),
    )


# ── OpenAI backend ────────────────────────────────────────────────────


class _OpenAIBackend:
    """Thin wrapper around the OpenAI Responses API (the original path)."""

    def __init__(self, model: str, temperature: float) -> None:
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.temperature = temperature

    # Temperature is omitted for some reasoning models that reject it.
    def _base_request(self, input_payload: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
        request: dict[str, Any] = {"model": self.model, "input": input_payload, **extra}
        if not self.model.startswith("gpt-5"):
            request["temperature"] = self.temperature
        return request

    def create(self, messages: list[dict[str, Any]], **extra: Any) -> str:
        request = self._base_request(messages, **extra)
        response = self.client.responses.create(**request)
        return response.output_text.strip()

    def create_raw(self, messages: list[dict[str, Any]], **extra: Any) -> Any:
        """Return the raw OpenAI response object (needed for tool-use loop)."""
        request = self._base_request(messages, **extra)
        return self.client.responses.create(**request)


# ── Anthropic backend ─────────────────────────────────────────────────


class _AnthropicBackend:
    """Wrapper around the Anthropic Messages API."""

    def __init__(self, model: str, temperature: float) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.temperature = temperature

    def _split_system(
        self, messages: list[dict[str, Any]]
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """Anthropic requires the system prompt as a separate arg."""
        system: str | None = None
        user_messages: list[dict[str, Any]] = []
        for m in messages:
            if m.get("role") == "system":
                system = m.get("content", "")
            else:
                user_messages.append({"role": m["role"], "content": m["content"]})
        return system, user_messages

    def create(self, messages: list[dict[str, Any]], **extra: Any) -> str:
        system, user_messages = self._split_system(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": extra.pop("max_tokens", 16384),
            "temperature": self.temperature,
            "messages": user_messages,
        }
        if system:
            kwargs["system"] = system

        # Map any extra tool definitions from OpenAI format to Anthropic format.
        tools = extra.pop("tools", None)
        if tools:
            kwargs["tools"] = [self._convert_tool(t) for t in tools]

        response = self.client.messages.create(**kwargs)
        # Extract text blocks from the response.
        return self._extract_text(response)

    def create_with_tools(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **extra: Any
    ) -> Any:
        """Return the raw Anthropic response for tool-use conversations."""
        system, user_messages = self._split_system(messages)
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": extra.pop("max_tokens", 16384),
            "temperature": self.temperature,
            "messages": user_messages,
            "tools": [self._convert_tool(t) for t in tools],
        }
        if system:
            kwargs["system"] = system
        return self.client.messages.create(**kwargs)

    @staticmethod
    def _convert_tool(openai_tool: dict[str, Any]) -> dict[str, Any]:
        """Convert an OpenAI function-calling tool spec to Anthropic format."""
        return {
            "name": openai_tool["name"],
            "description": openai_tool.get("description", ""),
            "input_schema": openai_tool.get("parameters", {}),
        }

    @staticmethod
    def _extract_text(response: Any) -> str:
        parts: list[str] = []
        for block in response.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "\n".join(parts).strip()


# ── Unified client (public API) ──────────────────────────────────────


class LLMPatchClient:
    """Drop-in replacement for ``OpenAIPatchClient`` that supports multiple providers."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        provider: str = "openai",
        patch_system_prompt: str | None = None,
        patch_schema_hint: str | None = None,
        bootstrap_system_prompt: str | None = None,
        # Legacy aliases
        system_prompt: str | None = None,
        schema_hint: str | None = None,
    ) -> None:
        self.provider = provider.lower()
        self.model = model
        self.temperature = temperature

        if self.provider == "openai":
            self._backend = _OpenAIBackend(model, temperature)
        elif self.provider in ("anthropic", "claude"):
            self._backend = _AnthropicBackend(model, temperature)
        else:
            raise ValueError(
                f"Unknown LLM provider {provider!r}. Supported: 'openai', 'anthropic'."
            )

        if patch_system_prompt is None:
            patch_system_prompt = system_prompt
        if patch_schema_hint is None:
            patch_schema_hint = schema_hint

        self.patch_system_prompt = patch_system_prompt or (
            "You are an expert Triton/NVSHMEM performance engineer. "
            "Suggest concrete code improvements grounded in profiling and timing data."
        )
        self.patch_schema_hint = patch_schema_hint or (
            "Return ONLY valid JSON with keys: diagnosis (string[]), "
            "hypotheses (string[]), candidate_code (string), test_expectations (string[]).\n"
            "candidate_code must be the COMPLETE rewritten Python source file (not a diff/patch). "
            "It must define a top-level `solution` function."
        )
        self.bootstrap_system_prompt = bootstrap_system_prompt or (
            "You are an expert Triton/NVSHMEM kernel engineer. "
            "Given a reference implementation and solved Triton examples, "
            "generate a correct and optimization-ready Triton candidate."
        )

    # ------------------------------------------------------------------
    # Simple proposal (no tools)
    # ------------------------------------------------------------------

    def propose_patch(self, prompt: str) -> PatchProposal:
        messages = [
            {"role": "system", "content": self.patch_system_prompt},
            {"role": "user", "content": f"{self.patch_schema_hint}\n\n{prompt}"},
        ]
        text = self._backend.create(messages)
        return _parse_patch_json(text)

    # ------------------------------------------------------------------
    # Tool-use proposal (RAG retrieval)
    # ------------------------------------------------------------------

    _SEARCH_EXAMPLES_TOOL: dict[str, Any] = {
        "type": "function",
        "name": "search_examples",
        "description": (
            "Search the corpus of solved Triton kernel examples for ones relevant "
            "to your current task.  Returns paired (reference, triton_solution) code "
            "that you can study for patterns, optimizations, or correctness fixes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "A natural-language description of what you are looking for."
                    ),
                },
                "top_k": {
                    "type": "integer",
                    "description": "Number of examples to retrieve (default 3).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    }

    def propose_patch_with_tools(
        self,
        prompt: str,
        retriever: LLMRetriever,
        *,
        max_tool_rounds: int = 3,
    ) -> PatchProposal:
        if self.provider in ("anthropic", "claude"):
            return self._propose_patch_with_tools_anthropic(prompt, retriever, max_tool_rounds)
        return self._propose_patch_with_tools_openai(prompt, retriever, max_tool_rounds)

    # ── OpenAI tool-use loop ──

    def _propose_patch_with_tools_openai(
        self,
        prompt: str,
        retriever: LLMRetriever,
        max_tool_rounds: int,
    ) -> PatchProposal:
        tools = [self._SEARCH_EXAMPLES_TOOL]

        input_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.patch_system_prompt},
            {"role": "user", "content": f"{self.patch_schema_hint}\n\n{prompt}"},
        ]

        backend: _OpenAIBackend = self._backend  # type: ignore[assignment]

        for _round in range(max_tool_rounds + 1):
            response = backend.create_raw(input_messages, tools=tools)

            tool_calls = [
                item
                for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]

            if not tool_calls:
                break

            def _item_to_dict(obj: Any) -> dict:
                if hasattr(obj, "model_dump"):
                    return obj.model_dump()
                if isinstance(obj, dict):
                    return obj
                return dict(obj)

            for item in response.output:
                input_messages.append(_item_to_dict(item))
            for tc in tool_calls:
                args = json.loads(tc.arguments)
                query = args.get("query", "")
                top_k = args.get("top_k")
                result_text = retriever.retrieve(query, top_k=top_k)
                input_messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": tc.call_id,
                        "output": result_text,
                    }
                )
        else:
            response = backend.create_raw(input_messages)

        text = response.output_text.strip()
        return _parse_patch_json(text)

    # ── Anthropic tool-use loop ──

    def _propose_patch_with_tools_anthropic(
        self,
        prompt: str,
        retriever: LLMRetriever,
        max_tool_rounds: int,
    ) -> PatchProposal:
        tools = [self._SEARCH_EXAMPLES_TOOL]

        backend: _AnthropicBackend = self._backend  # type: ignore[assignment]

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": f"{self.patch_schema_hint}\n\n{prompt}"},
        ]

        system_prompt = self.patch_system_prompt
        response = None

        for _round in range(max_tool_rounds + 1):
            kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 16384,
                "temperature": self.temperature,
                "messages": messages,
                "tools": [backend._convert_tool(t) for t in tools],
            }
            if system_prompt:
                kwargs["system"] = system_prompt

            response = backend.client.messages.create(**kwargs)

            # Anthropic signals tool use via stop_reason == "tool_use".
            # If stop_reason is "end_turn" the model is done, even if it
            # included tool_use blocks (shouldn't happen, but be safe).
            if getattr(response, "stop_reason", None) != "tool_use":
                break

            # Check for tool use blocks.
            tool_use_blocks = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

            if not tool_use_blocks:
                break

            # Append assistant response as-is.
            assistant_content = []
            for block in response.content:
                if getattr(block, "type", None) == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif getattr(block, "type", None) == "tool_use":
                    assistant_content.append(
                        {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                    )
            messages.append({"role": "assistant", "content": assistant_content})

            # Build tool results.
            tool_results: list[dict[str, Any]] = []
            for tb in tool_use_blocks:
                args = tb.input if isinstance(tb.input, dict) else json.loads(tb.input)
                query = args.get("query", "")
                top_k = args.get("top_k")
                result_text = retriever.retrieve(query, top_k=top_k)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": tb.id, "content": result_text}
                )
            messages.append({"role": "user", "content": tool_results})
        else:
            # Exhausted tool rounds — make one final call without tools so the
            # model is forced to produce a text response.
            kwargs = {
                "model": self.model,
                "max_tokens": 16384,
                "temperature": self.temperature,
                "messages": messages,
            }
            if system_prompt:
                kwargs["system"] = system_prompt
            response = backend.client.messages.create(**kwargs)

        assert response is not None
        text = _AnthropicBackend._extract_text(response)

        # Safety net: if the model produced no text (e.g. it only emitted
        # tool_use blocks in its final turn), make one more call without tools
        # to force a text response.
        if not text.strip():
            # Append whatever the model said as the assistant turn.
            assistant_content = []
            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "text":
                    assistant_content.append({"type": "text", "text": block.text})
                elif btype == "tool_use":
                    assistant_content.append(
                        {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
                    )
            if assistant_content:
                messages.append({"role": "assistant", "content": assistant_content})
            # Ask explicitly for the JSON result.
            messages.append(
                {"role": "user", "content": "Please provide your final answer as the JSON object now."}
            )
            fallback_kwargs: dict[str, Any] = {
                "model": self.model,
                "max_tokens": 16384,
                "temperature": self.temperature,
                "messages": messages,
            }
            if system_prompt:
                fallback_kwargs["system"] = system_prompt
            response = backend.client.messages.create(**fallback_kwargs)
            text = _AnthropicBackend._extract_text(response)

        return _parse_patch_json(text)

    # ------------------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------------------

    def generate_initial_candidate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": self.bootstrap_system_prompt},
            {"role": "user", "content": prompt},
        ]
        text = self._backend.create(messages)
        try:
            data: dict[str, Any] = json.loads(text)
            candidate_code = str(data.get("candidate_code", "")).strip()
            if candidate_code:
                return candidate_code
        except json.JSONDecodeError:
            pass
        return _strip_code_fences(text)


# ── Factory ──────────────────────────────────────────────────────────


def make_llm_client(
    provider: str = "openai",
    model: str = "gpt-5",
    temperature: float = 0.2,
) -> _OpenAIBackend | _AnthropicBackend:
    """Create a raw backend client (used by the retriever)."""
    provider = provider.lower()
    if provider == "openai":
        return _OpenAIBackend(model, temperature)
    if provider in ("anthropic", "claude"):
        return _AnthropicBackend(model, temperature)
    raise ValueError(f"Unknown LLM provider {provider!r}. Supported: 'openai', 'anthropic'.")
