from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ProblemConfig:
    problem_id: int
    rows: int = 1024
    cols: int = 1024
    dtype: str = "float32"


@dataclass
class EvalConfig:
    reference_backend: str = "reference"
    candidate_backend: str = "agent"
    warmup_iters: int = 3
    measure_iters: int = 10
    profile: bool = True
    download: bool = True
    # Timeout (seconds) for the local `modal run` subprocess.
    # If the subprocess doesn't finish in this time it is killed and the
    # candidate is treated as a failed evaluation (timeout).
    eval_timeout_s: int = 10 * 60
    # Timeout forwarded to the Modal worker (--worker-timeout-s flag).
    # This is the *remote* watchdog inside run_modal.py that kills torchrun.
    worker_timeout_s: int = 8 * 60


@dataclass
class OpenAIConfig:
    model: str = "gpt-5"
    temperature: float = 0.2


@dataclass
class PromptConfig:
    patch_system_prompt: str = (
        "You are an expert Triton/NVSHMEM performance engineer. "
        "Suggest concrete code patches grounded in profiling and timing data."
    )
    patch_schema_hint: str = (
        "Return ONLY valid JSON with keys: diagnosis (string[]), "
        "hypotheses (string[]), candidate_code (string), test_expectations (string[]).\n"
        "candidate_code must be the COMPLETE rewritten Python source file (not a diff/patch). "
        "It must define a top-level `solution` function."
    )
    performance_patch_template: str = (
        "Problem: {problem_id}\n"
        "Candidate file: {candidate_file}\n\n"
        "Mode: performance_optimization\n"
        "Focus on improving runtime while preserving correctness.\n\n"
        "Recent memory summary:\n"
        "{memory_summary}\n\n"
        "Latest evaluation feedback:\n"
        "{eval_feedback}\n\n"
        "Current candidate code:\n"
        "```python\n"
        "{current_code}\n"
        "```\n\n"
        "Return a JSON object with keys: diagnosis (string[]), hypotheses (string[]), "
        "candidate_code (string), test_expectations (string[]).\n"
        "candidate_code must be the COMPLETE rewritten Python file (not a diff/patch).\n"
        "It must define a top-level `solution` function.\n"
    )
    correctness_patch_template: str = (
        "Problem: {problem_id}\n"
        "Candidate file: {candidate_file}\n\n"
        "Mode: correctness_fix\n"
        "The most recent evaluation indicates correctness issues. "
        "Prioritize making candidate outputs match reference outputs. "
        "Only apply safe performance tweaks after correctness is restored.\n\n"
        "Recent memory summary:\n"
        "{memory_summary}\n\n"
        "Latest evaluation feedback:\n"
        "{eval_feedback}\n\n"
        "Current candidate code:\n"
        "```python\n"
        "{current_code}\n"
        "```\n\n"
        "Return a JSON object with keys: diagnosis (string[]), hypotheses (string[]), "
        "candidate_code (string), test_expectations (string[]).\n"
        "candidate_code must be the COMPLETE rewritten Python file (not a diff/patch).\n"
        "It must define a top-level `solution` function.\n"
    )
    bootstrap_system_prompt: str = (
        "You are an expert Triton/NVSHMEM kernel engineer. "
        "Given a reference implementation and solved Triton examples, "
        "generate a correct and optimization-ready Triton candidate."
    )
    bootstrap_template: str = (
        "You are bootstrapping a brand-new problem.\n"
        "Problem: {problem_id}\n"
        "Output file: {candidate_file}\n\n"
        "Reference implementation:\n"
        "```python\n"
        "{reference_code}\n"
        "```\n\n"
        "Solved Triton examples to use as style/context:\n"
        "{context_examples}\n\n"
        "Return ONLY valid JSON with key: candidate_code (string).\n"
        "candidate_code must be a complete Python file that defines `solution`.\n"
    )
    # Backward-compatible aliases for older configs.
    system_prompt: str = ""
    schema_hint: str = ""
    single_shot_template: str = ""


@dataclass
class RetrievalConfig:
    enabled: bool = False
    top_k: int = 3
    retrieval_model: str | None = None  # None = use same model as main proposer.


@dataclass
class StrategyConfig:
    proposer: str = "single_shot"
    memory: str = "best_so_far"
    scorer: str = "speedup_mean"


@dataclass
class ExperimentConfig:
    name: str
    max_iterations: int = 3
    early_stop_speedup: float | None = None
    seed_from_backend: str = "triton"
    candidate_dir: str = "solutions_agent"
    run_root: str = "runs"
    problems: list[ProblemConfig] = field(default_factory=list)
    eval: EvalConfig = field(default_factory=EvalConfig)
    openai: OpenAIConfig = field(default_factory=OpenAIConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    strategies: StrategyConfig = field(default_factory=StrategyConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    @staticmethod
    def _get(raw: dict[str, Any], key: str, default: Any) -> Any:
        return raw.get(key, default) if isinstance(raw, dict) else default

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        problems = [ProblemConfig(**p) for p in raw.get("problems", [])]
        eval_cfg = EvalConfig(**cls._get(raw, "eval", {}))
        openai_cfg = OpenAIConfig(**cls._get(raw, "openai", {}))
        prompt_raw = cls._get(raw, "prompts", {})
        if isinstance(prompt_raw, dict):
            # Accept legacy field names and map to new contracts.
            if "system_prompt" in prompt_raw and "patch_system_prompt" not in prompt_raw:
                prompt_raw["patch_system_prompt"] = prompt_raw["system_prompt"]
            if "schema_hint" in prompt_raw and "patch_schema_hint" not in prompt_raw:
                prompt_raw["patch_schema_hint"] = prompt_raw["schema_hint"]
            if "single_shot_template" in prompt_raw and "performance_patch_template" not in prompt_raw:
                prompt_raw["performance_patch_template"] = prompt_raw["single_shot_template"]
        prompt_cfg = PromptConfig(**prompt_raw)
        strategy_cfg = StrategyConfig(**cls._get(raw, "strategies", {}))
        retrieval_cfg = RetrievalConfig(**cls._get(raw, "retrieval", {}))

        return cls(
            name=raw["name"],
            max_iterations=raw.get("max_iterations", 3),
            early_stop_speedup=raw.get("early_stop_speedup"),
            seed_from_backend=raw.get("seed_from_backend", "triton"),
            candidate_dir=raw.get("candidate_dir", "solutions_agent"),
            run_root=raw.get("run_root", "runs"),
            problems=problems,
            eval=eval_cfg,
            openai=openai_cfg,
            prompts=prompt_cfg,
            strategies=strategy_cfg,
            retrieval=retrieval_cfg,
        )
