from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agent.config import EvalConfig, ProblemConfig


@dataclass
class BackendEvalResult:
    backend: str
    logs_dir: Path
    summary_rank0: dict
    eval_feedback: dict


class ModalEvaluator:
    def __init__(self, repo_root: Path, eval_cfg: EvalConfig):
        self.repo_root = repo_root
        self.eval_cfg = eval_cfg
        self._reference_results: dict[int, BackendEvalResult] = {}

    @staticmethod
    def _to_eval_feedback(summary: dict) -> dict:
        ranks = summary.get("ranks", []) if isinstance(summary, dict) else []
        failed_ranks = [r.get("rank") for r in ranks if isinstance(r, dict) and not r.get("correctness_ok", True)]
        trace_candidate = [
            r.get("trace_candidate")
            for r in ranks
            if isinstance(r, dict) and r.get("trace_candidate")
        ]
        trace_reference = [
            r.get("trace_reference")
            for r in ranks
            if isinstance(r, dict) and r.get("trace_reference")
        ]
        used_cached_reference_outputs = any(
            bool(r.get("used_cached_reference_outputs", False))
            for r in ranks
            if isinstance(r, dict)
        )
        reference_timing_skipped = any(
            bool(r.get("reference_timing_skipped", False))
            for r in ranks
            if isinstance(r, dict)
        )
        return {
            "status": summary.get("status"),
            "aggregate": summary.get("aggregate", {}),
            "correctness": {
                "all_ok": len(failed_ranks) == 0,
                "failed_ranks": failed_ranks,
            },
            "traces": {
                "candidate": trace_candidate,
                "reference": trace_reference,
            },
            "cache": {
                "used_cached_reference_outputs": used_cached_reference_outputs,
                "reference_timing_skipped": reference_timing_skipped,
            },
        }

    def _run_backend(
        self,
        problem: ProblemConfig,
        backend: str,
        *,
        use_cached_reference: bool = False,
    ) -> BackendEvalResult:
        download_flag = "--download" if self.eval_cfg.download else "--no-download"
        cmd = [
            "modal",
            "run",
            "run_modal.py",
            "--problem",
            str(problem.problem_id),
            "--solution",
            backend,
            "--m",
            str(problem.rows),
            "--n",
            str(problem.cols),
            "--dtype",
            problem.dtype,
            download_flag,
            "--save-outputs",
        ]
        if use_cached_reference:
            cmd.append("--use-cached-reference")
        subprocess.run(cmd, cwd=self.repo_root, check=True)

        logs_dir = self.repo_root / "logs" / f"problem_{problem.problem_id}" / backend
        summary_path = logs_dir / "summary_rank0.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing summary file: {summary_path}")
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        return BackendEvalResult(
            backend=backend,
            logs_dir=logs_dir,
            summary_rank0=summary,
            eval_feedback=self._to_eval_feedback(summary),
        )

    def precompute_references(self, problems: list[ProblemConfig]) -> None:
        for problem in problems:
            if problem.problem_id in self._reference_results:
                continue
            self._reference_results[problem.problem_id] = self._run_backend(
                problem,
                self.eval_cfg.reference_backend,
            )

    def evaluate_pair(self, problem: ProblemConfig) -> tuple[BackendEvalResult, BackendEvalResult]:
        reference = self._reference_results.get(problem.problem_id)
        if reference is None:
            reference = self._run_backend(problem, self.eval_cfg.reference_backend)
            self._reference_results[problem.problem_id] = reference
        candidate = self._run_backend(
            problem,
            self.eval_cfg.candidate_backend,
            use_cached_reference=True,
        )
        return reference, candidate
