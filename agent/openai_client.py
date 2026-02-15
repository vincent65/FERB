from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from openai import OpenAI

if TYPE_CHECKING:
    from agent.strategies.retrieval.retriever import LLMRetriever


@dataclass
class PatchProposal:
    diagnosis: list[str]
    hypotheses: list[str]
    candidate_code: str          # Complete rewritten file content
    test_expectations: list[str]

    # Backward-compat alias so callers using .proposed_patch still work
    # during transition.
    @property
    def proposed_patch(self) -> str:
        return self.candidate_code


class OpenAIPatchClient:
    def __init__(
        self,
        model: str,
        temperature: float = 0.2,
        patch_system_prompt: str | None = None,
        patch_schema_hint: str | None = None,
        bootstrap_system_prompt: str | None = None,
        system_prompt: str | None = None,
        schema_hint: str | None = None,
    ):
        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
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

    def _create_response(self, input_payload: list[dict[str, str]]):
        request: dict[str, Any] = {
            "model": self.model,
            "input": input_payload,
        }
        # Some OpenAI responses models (for example gpt-5) reject temperature.
        if not self.model.startswith("gpt-5"):
            request["temperature"] = self.temperature
        return self.client.responses.create(**request)

    def propose_patch(self, prompt: str) -> PatchProposal:
        response = self._create_response(
            [
                {
                    "role": "system",
                    "content": self.patch_system_prompt,
                },
                {"role": "user", "content": f"{self.patch_schema_hint}\n\n{prompt}"},
            ]
        )

        text = response.output_text.strip()
        # Strip markdown fences if the model wrapped JSON in ```json ... ```
        text = self._strip_code_fences(text)
        data: dict[str, Any] = json.loads(text)

        # Accept either "candidate_code" (new) or "proposed_patch" (legacy).
        candidate_code = str(
            data.get("candidate_code", data.get("proposed_patch", ""))
        )

        return PatchProposal(
            diagnosis=list(data.get("diagnosis", [])),
            hypotheses=list(data.get("hypotheses", [])),
            candidate_code=candidate_code,
            test_expectations=list(data.get("test_expectations", [])),
        )

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
                        "A natural-language description of what you are looking for. "
                        "For example: 'Triton kernel that does tiled matrix multiply with "
                        "shared memory' or 'NVSHMEM all-to-all exchange pattern'."
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
        """Like :meth:`propose_patch` but gives the model a ``search_examples`` tool.

        The model can call the tool to retrieve relevant solved Triton examples
        from the corpus.  A lightweight LLM ranker inside *retriever* selects
        the top-k entries and returns formatted code.

        The conversation loops until the model produces a final text response
        (the JSON patch proposal) or *max_tool_rounds* tool calls have been
        handled.
        """
        tools = [self._SEARCH_EXAMPLES_TOOL]

        input_messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.patch_system_prompt},
            {"role": "user", "content": f"{self.patch_schema_hint}\n\n{prompt}"},
        ]

        for _round in range(max_tool_rounds + 1):
            request: dict[str, Any] = {
                "model": self.model,
                "input": input_messages,
                "tools": tools,
            }
            if not self.model.startswith("gpt-5"):
                request["temperature"] = self.temperature

            response = self.client.responses.create(**request)

            # Check whether the model produced tool calls.
            tool_calls = [
                item for item in response.output
                if getattr(item, "type", None) == "function_call"
            ]

            if not tool_calls:
                # Model produced a final text answer — parse it.
                break

            # Process each tool call and append results.
            # For reasoning models (gpt-5, o3, o4-mini, etc.), response.output may
            # contain reasoning items that must precede function_call items when
            # passed back. Include full output to preserve required ordering.
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
            # Exhausted tool rounds — make one final call without tools so the
            # model is forced to produce a text response.
            request = {
                "model": self.model,
                "input": input_messages,
            }
            if not self.model.startswith("gpt-5"):
                request["temperature"] = self.temperature
            response = self.client.responses.create(**request)

        # Parse the final text response the same way as propose_patch.
        text = response.output_text.strip()
        text = self._strip_code_fences(text)
        data: dict[str, Any] = json.loads(text)

        candidate_code = str(
            data.get("candidate_code", data.get("proposed_patch", ""))
        )

        return PatchProposal(
            diagnosis=list(data.get("diagnosis", [])),
            hypotheses=list(data.get("hypotheses", [])),
            candidate_code=candidate_code,
            test_expectations=list(data.get("test_expectations", [])),
        )

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                return "\n".join(lines[1:-1]).strip()
        return stripped

    def generate_initial_candidate(self, prompt: str) -> str:
        response = self._create_response(
            [
                {
                    "role": "system",
                    "content": self.bootstrap_system_prompt,
                },
                {"role": "user", "content": prompt},
            ]
        )
        text = response.output_text.strip()
        try:
            data: dict[str, Any] = json.loads(text)
            candidate_code = str(data.get("candidate_code", "")).strip()
            if candidate_code:
                return candidate_code
        except json.JSONDecodeError:
            pass

        # Fallback for non-JSON model outputs.
        return self._strip_code_fences(text)
