from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunLogger:
    run_dir: Path

    def __post_init__(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "iterations.jsonl"
        self.llm_path = self.run_dir / "llm_outputs.jsonl"
        self.llm_dir = self.run_dir / "llm_outputs"
        self.llm_dir.mkdir(parents=True, exist_ok=True)

    def append(self, payload: dict[str, Any]) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

    def append_llm(self, payload: dict[str, Any]) -> Path:
        with open(self.llm_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        iteration = payload.get("iteration", "na")
        problem_id = payload.get("problem_id", "na")
        output_path = self.llm_dir / f"iter_{iteration}_problem_{problem_id}.json"
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return output_path
