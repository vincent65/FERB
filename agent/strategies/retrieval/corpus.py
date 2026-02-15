from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CorpusEntry:
    """A paired (reference, triton_solution) example from the knowledge base."""

    problem_id: int
    reference_code: str
    triton_code: str
    summary: str  # Short description for the LLM ranker.


class TritonCorpus:
    """Scans solutions_triton/ + reference/ and builds paired corpus entries."""

    def __init__(self, repo_root: Path, seed_from_backend: str = "triton") -> None:
        self.repo_root = repo_root
        self.seed_from_backend = seed_from_backend
        self._entries: list[CorpusEntry] = []
        self._build()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def _build(self) -> None:
        solutions_dir = self.repo_root / f"solutions_{self.seed_from_backend}"
        if not solutions_dir.exists():
            return

        suffix = f"_{self.seed_from_backend}.py"
        for path in sorted(solutions_dir.glob(f"*{suffix}")):
            match = re.match(
                r"^(\d+)_" + re.escape(self.seed_from_backend) + r"\.py$",
                path.name,
            )
            if not match:
                continue

            problem_id = int(match.group(1))
            triton_code = path.read_text(encoding="utf-8")

            ref_path = self.repo_root / "reference" / f"{problem_id}.py"
            reference_code = (
                ref_path.read_text(encoding="utf-8")
                if ref_path.exists()
                else "Reference file not found."
            )

            # Build a short summary from the first docstring / comment lines.
            summary = self._extract_summary(problem_id, triton_code)
            self._entries.append(
                CorpusEntry(
                    problem_id=problem_id,
                    reference_code=reference_code,
                    triton_code=triton_code,
                    summary=summary,
                )
            )

    @staticmethod
    def _extract_summary(problem_id: int, code: str) -> str:
        """Return a one-line summary for the ranker prompt."""
        lines = code.splitlines()
        # Try to grab the first docstring or comment.
        for i, line in enumerate(lines[:30]):
            stripped = line.strip()
            if stripped.startswith('"""') or stripped.startswith("'''"):
                # Inline docstring: """Some text"""
                clean = stripped.strip('"').strip("'").strip()
                if clean:
                    return f"Problem {problem_id}: {clean}"
                # Multi-line docstring: look at the next non-empty line.
                for following in lines[i + 1 : i + 5]:
                    following = following.strip()
                    if following and not following.startswith('"""') and not following.startswith("'''"):
                        return f"Problem {problem_id}: {following}"
                return f"Problem {problem_id}: Triton kernel solution"
            if stripped.startswith("#") and len(stripped) > 2:
                return f"Problem {problem_id}: {stripped.lstrip('#').strip()}"
        return f"Problem {problem_id}: Triton kernel solution"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_all_entries(self) -> list[CorpusEntry]:
        return list(self._entries)

    def get_entry(self, problem_id: int) -> CorpusEntry | None:
        for entry in self._entries:
            if entry.problem_id == problem_id:
                return entry
        return None

    def get_summaries_text(self) -> str:
        """Return a numbered list of summaries for the ranker prompt."""
        if not self._entries:
            return "No corpus entries available."
        lines = []
        for entry in self._entries:
            lines.append(f"- [{entry.problem_id}] {entry.summary}")
        return "\n".join(lines)

    @staticmethod
    def format_entries(entries: list[CorpusEntry]) -> str:
        """Format selected entries as context text for the proposer."""
        if not entries:
            return "No relevant examples found."
        chunks: list[str] = []
        for entry in entries:
            chunks.append(
                f"### Example problem {entry.problem_id}\n"
                f"Reference:\n```python\n{entry.reference_code}\n```\n"
                f"Triton solution:\n```python\n{entry.triton_code}\n```"
            )
        return "\n\n".join(chunks)

    def __len__(self) -> int:
        return len(self._entries)
