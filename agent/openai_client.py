from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class PatchProposal:
    diagnosis: list[str]
    hypotheses: list[str]
    proposed_patch: str
    test_expectations: list[str]


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
            "Suggest concrete code patches grounded in profiling and timing data."
        )
        self.patch_schema_hint = patch_schema_hint or (
            "Return ONLY valid JSON with keys: diagnosis (string[]), "
            "hypotheses (string[]), proposed_patch (string), test_expectations (string[])."
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
        data: dict[str, Any] = json.loads(text)
        return PatchProposal(
            diagnosis=list(data.get("diagnosis", [])),
            hypotheses=list(data.get("hypotheses", [])),
            proposed_patch=str(data.get("proposed_patch", "")),
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
