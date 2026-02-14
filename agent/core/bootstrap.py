from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from agent.config import PromptConfig
from agent.openai_client import OpenAIPatchClient


@dataclass
class BootstrapResult:
    problem_id: int
    candidate_file: Path
    event: str
    source: str


class BootstrapStage:
    def __init__(
        self,
        repo_root: Path,
        candidate_dir: str,
        seed_from_backend: str,
        prompt_config: PromptConfig,
        openai_client: OpenAIPatchClient,
    ):
        self.repo_root = repo_root
        self.candidate_dir = candidate_dir
        self.seed_from_backend = seed_from_backend
        self.prompt_config = prompt_config
        self.openai_client = openai_client

    def candidate_file(self, problem_id: int) -> Path:
        return self.repo_root / self.candidate_dir / f"{problem_id}_agent.py"

    def ensure_candidate(self, problem_id: int) -> BootstrapResult:
        dst = self.candidate_file(problem_id)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            return BootstrapResult(
                problem_id=problem_id,
                candidate_file=dst,
                event="bootstrap_skipped_existing",
                source=str(dst),
            )

        src = self.repo_root / f"solutions_{self.seed_from_backend}" / f"{problem_id}_{self.seed_from_backend}.py"
        if src.exists():
            shutil.copy2(src, dst)
            return BootstrapResult(
                problem_id=problem_id,
                candidate_file=dst,
                event="bootstrap_seed_copy",
                source=str(src),
            )

        self._bootstrap_candidate(problem_id, dst)
        return BootstrapResult(
            problem_id=problem_id,
            candidate_file=dst,
            event="bootstrap_generated",
            source=f"reference/{problem_id}.py + solutions_{self.seed_from_backend}",
        )

    def _collect_context_examples(self, problem_id: int) -> str:
        context_dir = self.repo_root / f"solutions_{self.seed_from_backend}"
        if not context_dir.exists():
            return "No solved context examples available."

        suffix = f"_{self.seed_from_backend}.py"
        chunks: list[str] = []
        for path in sorted(context_dir.glob(f"*{suffix}")):
            match = re.match(r"^(\d+)_" + re.escape(self.seed_from_backend) + r"\.py$", path.name)
            if not match:
                continue
            other_id = int(match.group(1))
            if other_id == problem_id:
                continue
            code = path.read_text(encoding="utf-8")
            ref_path = self.repo_root / "reference" / f"{other_id}.py"
            ref_code = ref_path.read_text(encoding="utf-8") if ref_path.exists() else "Reference file not found."
            chunks.append(
                f"### Example problem {other_id} ({path.name})\n"
                f"Reference:\n```python\n{ref_code}\n```\n"
                f"Triton solution:\n```python\n{code}\n```"
            )
        return "\n\n".join(chunks) if chunks else "No solved context examples available."

    def _bootstrap_candidate(self, problem_id: int, dst: Path) -> None:
        reference_file = self.repo_root / "reference" / f"{problem_id}.py"
        if not reference_file.exists():
            raise FileNotFoundError(
                f"Cannot bootstrap problem {problem_id}: missing seed and missing reference file {reference_file}"
            )

        reference_code = reference_file.read_text(encoding="utf-8")
        context_examples = self._collect_context_examples(problem_id)
        prompt = self.prompt_config.bootstrap_template.format(
            problem_id=problem_id,
            candidate_file=str(dst),
            reference_code=reference_code,
            context_examples=context_examples,
        )
        candidate_code = self.openai_client.generate_initial_candidate(prompt).strip()
        if not candidate_code:
            raise RuntimeError(f"Bootstrap generation returned empty candidate for problem {problem_id}")

        if "def solution" not in candidate_code:
            raise RuntimeError(
                f"Bootstrap candidate for problem {problem_id} does not define `solution`."
            )

        dst.write_text(candidate_code + "\n", encoding="utf-8")
