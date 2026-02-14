from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from agent.config import ExperimentConfig, ProblemConfig
from agent.core.bootstrap import BootstrapStage
from agent.core.rollback import RollbackManager
from agent.core.run_log import RunLogger
from agent.eval.modal_evaluator import ModalEvaluator
from agent.openai_client import OpenAIPatchClient
from agent.strategies.proposers.base import ProposalContext
from agent.strategies.registry import make_memory, make_proposer, make_scorer


@dataclass
class Optimizer:
    repo_root: Path
    cfg: ExperimentConfig

    def __post_init__(self) -> None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.repo_root / self.cfg.run_root / f"{now}_{self.cfg.name}"
        self.logger = RunLogger(self.run_dir)
        self.evaluator = ModalEvaluator(self.repo_root, self.cfg.eval)
        self.openai_client = OpenAIPatchClient(
            model=self.cfg.openai.model,
            temperature=self.cfg.openai.temperature,
            patch_system_prompt=self.cfg.prompts.patch_system_prompt,
            patch_schema_hint=self.cfg.prompts.patch_schema_hint,
            bootstrap_system_prompt=self.cfg.prompts.bootstrap_system_prompt,
        )
        self.bootstrap = BootstrapStage(
            repo_root=self.repo_root,
            candidate_dir=self.cfg.candidate_dir,
            seed_from_backend=self.cfg.seed_from_backend,
            prompt_config=self.cfg.prompts,
            openai_client=self.openai_client,
        )
        self.proposer = make_proposer(
            self.cfg.strategies.proposer,
            self.openai_client,
            self.cfg.prompts,
        )
        self.memory = make_memory(self.cfg.strategies.memory)
        self.scorer = make_scorer(self.cfg.strategies.scorer)
        self._progress_total_steps = 1
        self._progress_completed_steps = 0
        self._progress_start_time = 0.0

    def _init_progress(self) -> None:
        problems_count = len(self.cfg.problems)
        # bootstrap(problem) + precompute(all refs) + per-iteration(eval + propose-per-problem)
        self._progress_total_steps = max(
            1,
            problems_count + 1 + (self.cfg.max_iterations * (1 + problems_count)),
        )
        self._progress_completed_steps = 0
        self._progress_start_time = time.perf_counter()
        print(
            f"Starting experiment with {self._progress_total_steps} steps "
            f"({problems_count} problem(s), {self.cfg.max_iterations} iteration(s))",
            flush=True,
        )

    def _advance_progress(self, stage: str) -> None:
        self._progress_completed_steps = min(
            self._progress_total_steps,
            self._progress_completed_steps + 1,
        )
        elapsed = time.perf_counter() - self._progress_start_time
        ratio = self._progress_completed_steps / self._progress_total_steps
        remaining_steps = self._progress_total_steps - self._progress_completed_steps
        eta_seconds = (
            (elapsed / self._progress_completed_steps) * remaining_steps
            if self._progress_completed_steps > 0
            else 0.0
        )
        bar_width = 28
        filled = int(bar_width * ratio)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(
            f"[{bar}] {self._progress_completed_steps}/{self._progress_total_steps} "
            f"{stage} | elapsed {elapsed:.1f}s | eta {eta_seconds:.1f}s",
            flush=True,
        )

    def _candidate_file(self, problem_id: int) -> Path:
        return self.repo_root / self.cfg.candidate_dir / f"{problem_id}_agent.py"

    def _apply_patch(self, patch_text: str) -> None:
        patch_file = self.run_dir / "candidate.patch"
        patch_file.write_text(patch_text, encoding="utf-8")
        subprocess.run(
            ["git", "apply", "--recount", "--reject", "--whitespace=nowarn", str(patch_file)],
            cwd=self.repo_root,
            check=True,
        )

    def _evaluate_all(self) -> tuple[float, list[dict]]:
        entries: list[dict] = []
        scores: list[float] = []
        for problem in self.cfg.problems:
            reference, candidate = self.evaluator.evaluate_pair(problem)
            score = self.scorer.score(reference.summary_rank0, candidate.summary_rank0)
            scores.append(score)
            entries.append(
                {
                    "problem_id": problem.problem_id,
                    "reference_summary": reference.summary_rank0,
                    "candidate_summary": candidate.summary_rank0,
                    "reference_feedback": reference.eval_feedback,
                    "candidate_feedback": candidate.eval_feedback,
                    "score": score,
                }
            )
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return mean_score, entries

    def _proposal_for_problem(self, problem: ProblemConfig, history: list[dict]) -> dict:
        candidate_file = self._candidate_file(problem.problem_id)
        current_code = candidate_file.read_text(encoding="utf-8")
        latest = next((h for h in reversed(history) if h["problem_id"] == problem.problem_id), {})
        latest_feedback = latest.get("candidate_feedback", {})
        correctness = latest_feedback.get("correctness", {})
        proposal_mode = "perf_opt" if correctness.get("all_ok", True) else "correctness_fix"
        ctx = ProposalContext(
            problem_id=problem.problem_id,
            candidate_file=candidate_file,
            current_code=current_code,
            latest_metrics=latest,
            memory_summary=self.memory.summarize(history),
            proposal_mode=proposal_mode,
            eval_feedback=latest_feedback,
        )
        proposal = self.proposer.propose(ctx)
        proposal["mode"] = proposal_mode
        return proposal

    def run(self) -> Path:
        history: list[dict] = []
        self._init_progress()
        for p in self.cfg.problems:
            bootstrap_result = self.bootstrap.ensure_candidate(p.problem_id)
            self.logger.append(
                {
                    "event": bootstrap_result.event,
                    "problem_id": p.problem_id,
                    "candidate_file": str(bootstrap_result.candidate_file),
                    "source": bootstrap_result.source,
                }
            )
            self._advance_progress(f"bootstrapped problem {p.problem_id}")
        self.evaluator.precompute_references(self.cfg.problems)
        self._advance_progress("precomputed references")

        for iteration in range(1, self.cfg.max_iterations + 1):
            score, eval_entries = self._evaluate_all()
            for entry in eval_entries:
                row = {"iteration": iteration, **entry}
                history.append(row)
                self.logger.append(row)
            self._advance_progress(f"evaluated iteration {iteration}")

            if self.cfg.early_stop_speedup is not None and score >= self.cfg.early_stop_speedup:
                self.logger.append(
                    {
                        "iteration": iteration,
                        "event": "early_stop",
                        "score": score,
                        "threshold": self.cfg.early_stop_speedup,
                    }
                )
                break

            for p in self.cfg.problems:
                candidate_file = self._candidate_file(p.problem_id)
                rollback = RollbackManager(candidate_file)
                rollback.snapshot()
                proposal = self._proposal_for_problem(p, history)
                cycle_event = (
                    "perf_opt_cycle"
                    if proposal.get("mode", "perf_opt") == "perf_opt"
                    else "correctness_fix_cycle"
                )
                self.logger.append(
                    {
                        "iteration": iteration,
                        "event": cycle_event,
                        "problem_id": p.problem_id,
                    }
                )
                try:
                    self._apply_patch(proposal.get("proposed_patch", ""))
                    self.logger.append(
                        {
                            "iteration": iteration,
                            "event": "proposal_applied",
                            "problem_id": p.problem_id,
                            "proposal": proposal,
                        }
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    rollback.rollback()
                    self.logger.append(
                        {
                            "iteration": iteration,
                            "event": "proposal_failed",
                            "problem_id": p.problem_id,
                            "error": str(exc),
                            "proposal": proposal,
                        }
                    )
                self._advance_progress(
                    f"completed iteration {iteration} patch cycle for problem {p.problem_id}"
                )

        summary = {"name": self.cfg.name, "run_dir": str(self.run_dir), "iterations": len(history)}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        total_elapsed = time.perf_counter() - self._progress_start_time
        print(f"Experiment finished in {total_elapsed:.1f}s", flush=True)
        return self.run_dir
