from __future__ import annotations

import json
import subprocess
import traceback
from dataclasses import dataclass
from pathlib import Path

from agent.config import EvalConfig, ProblemConfig


@dataclass
class BackendEvalResult:
    backend: str
    logs_dir: Path
    summary_rank0: dict
    eval_feedback: dict


def _make_error_result(
    backend: str,
    logs_dir: Path,
    error_message: str,
    *,
    is_timeout: bool = False,
) -> BackendEvalResult:
    """Return a synthetic BackendEvalResult that signals a failed evaluation.

    The optimizer can inspect ``eval_feedback["status"]`` to decide whether to
    rollback and re-propose.
    """
    summary: dict = {
        "problem_id": None,
        "backend": backend,
        "status": "timeout" if is_timeout else "error",
        "n_ranks": 0,
        "ranks": [],
        "aggregate": {},
        "error": error_message,
    }
    feedback: dict = {
        "status": summary["status"],
        "aggregate": {},
        "correctness": {
            "all_ok": False,
            "failed_ranks": [],
        },
        "traces": {"candidate": [], "reference": []},
        "cache": {
            "used_cached_reference_outputs": False,
            "reference_timing_skipped": False,
        },
        "error": error_message,
        "is_timeout": is_timeout,
    }
    return BackendEvalResult(
        backend=backend,
        logs_dir=logs_dir,
        summary_rank0=summary,
        eval_feedback=feedback,
    )


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
        logs_dir = self.repo_root / "logs" / f"problem_{problem.problem_id}" / backend

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
            "--warmup-iters",
            str(self.eval_cfg.warmup_iters),
            "--measure-iters",
            str(self.eval_cfg.measure_iters),
            download_flag,
            "--save-outputs",
            "--worker-timeout-s",
            str(self.eval_cfg.worker_timeout_s),
        ]
        cmd.append("--profile" if self.eval_cfg.profile else "--no-profile")
        if use_cached_reference:
            cmd.append("--use-cached-reference")
        # Skip downloading .pt files locally — the evaluator only needs
        # summary_rank0.json which is now inlined in the Modal return value.
        # The .pt files stay on the Modal volume for cached-reference usage.
        cmd.append("--skip-pt-download")

        # ── Run with a local timeout so a hung worker can never block the
        #    optimizer loop forever. ──
        timeout_s = self.eval_cfg.eval_timeout_s
        try:
            print(
                f"[evaluator] Running {backend} for problem {problem.problem_id} "
                f"(timeout={timeout_s}s) ...",
                flush=True,
            )
            subprocess.run(
                cmd,
                cwd=self.repo_root,
                check=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            msg = (
                f"modal run timed out after {timeout_s}s for problem "
                f"{problem.problem_id}, backend {backend}. "
                "The candidate likely caused a hang (deadlock / infinite loop)."
            )
            print(f"[evaluator] TIMEOUT: {msg}", flush=True)
            return _make_error_result(backend, logs_dir, msg, is_timeout=True)
        except subprocess.CalledProcessError as exc:
            msg = (
                f"modal run failed (exit code {exc.returncode}) for problem "
                f"{problem.problem_id}, backend {backend}: {exc}"
            )
            print(f"[evaluator] ERROR: {msg}", flush=True)
            return _make_error_result(backend, logs_dir, msg, is_timeout=False)
        except Exception as exc:  # pylint: disable=broad-except
            msg = (
                f"Unexpected error running modal for problem "
                f"{problem.problem_id}, backend {backend}: "
                f"{exc}\n{traceback.format_exc()}"
            )
            print(f"[evaluator] ERROR: {msg}", flush=True)
            return _make_error_result(backend, logs_dir, msg, is_timeout=False)

        # ── Parse results ──
        summary_path = logs_dir / "summary_rank0.json"
        if not summary_path.exists():
            msg = (
                f"Missing summary file after successful modal run: {summary_path}. "
                "The worker may have crashed before writing results."
            )
            print(f"[evaluator] ERROR: {msg}", flush=True)
            return _make_error_result(backend, logs_dir, msg, is_timeout=False)

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
            result = self._run_backend(
                problem,
                self.eval_cfg.reference_backend,
            )
            if result.eval_feedback.get("status") in ("error", "timeout"):
                raise RuntimeError(
                    f"Reference evaluation failed for problem {problem.problem_id}. "
                    f"Cannot continue without a working reference. "
                    f"Error: {result.eval_feedback.get('error', 'unknown')}"
                )
            self._reference_results[problem.problem_id] = result

    def evaluate_pair(self, problem: ProblemConfig) -> tuple[BackendEvalResult, BackendEvalResult]:
        reference = self._reference_results.get(problem.problem_id)
        if reference is None:
            reference = self._run_backend(problem, self.eval_cfg.reference_backend)
            if reference.eval_feedback.get("status") in ("error", "timeout"):
                raise RuntimeError(
                    f"Reference evaluation failed for problem {problem.problem_id}: "
                    f"{reference.eval_feedback.get('error', 'unknown')}"
                )
            self._reference_results[problem.problem_id] = reference

        candidate = self._run_backend(
            problem,
            self.eval_cfg.candidate_backend,
            use_cached_reference=True,
        )
        # NOTE: candidate errors are NOT raised here — they are returned as
        # error results so the optimizer can log, rollback, and re-propose.
        return reference, candidate
