from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.config import ExperimentConfig, ProblemConfig
from agent.core.bootstrap import BootstrapStage
from agent.core.rollback import RollbackManager
from agent.core.run_log import RunLogger
from agent.eval.modal_evaluator import ModalEvaluator
from agent.llm_client import LLMPatchClient
from agent.strategies.proposers.base import IterationSnapshot, ProposalContext
from agent.strategies.registry import make_memory, make_proposer, make_scorer
from agent.strategies.retrieval.corpus import TritonCorpus
from agent.strategies.retrieval.retriever import LLMRetriever

# Lazy-imported when RLM retrieval is enabled:
#   from agent.strategies.retrieval.docs_loader import DocsCorpus
#   from agent.strategies.retrieval.rlm_retriever import RLMDocRetriever


@dataclass
class Optimizer:
    repo_root: Path
    cfg: ExperimentConfig

    _RAG_SYSTEM_PROMPT_HINT: str = (
        "\nYou have access to a `search_examples` tool that retrieves relevant "
        "solved Triton kernel examples (paired reference + optimized solution). "
        "Call it when you want to see how similar kernels were written or optimized."
    )

    def __post_init__(self) -> None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = self.repo_root / self.cfg.run_root / f"{now}_{self.cfg.name}"
        self.logger = RunLogger(self.run_dir)
        self.evaluator = ModalEvaluator(self.repo_root, self.cfg.eval)

        # -- Optional RAG retrieval --
        retriever = None
        patch_system_prompt = self.cfg.prompts.patch_system_prompt
        if self.cfg.retrieval.enabled:
            corpus = TritonCorpus(
                self.repo_root,
                seed_from_backend=self.cfg.seed_from_backend,
            )
            retrieval_model = (
                self.cfg.retrieval.retrieval_model or self.cfg.openai.model
            )
            retriever = LLMRetriever(
                corpus,
                model=retrieval_model,
                default_top_k=self.cfg.retrieval.top_k,
                provider=self.cfg.openai.provider,
            )
            # Augment the system prompt so the model knows the tool exists.
            patch_system_prompt += self._RAG_SYSTEM_PROMPT_HINT
            print(
                f"[optimizer] RAG retrieval enabled: {len(corpus)} corpus entries, "
                f"model={retrieval_model}, top_k={self.cfg.retrieval.top_k}",
                flush=True,
            )

        self.openai_client = LLMPatchClient(
            model=self.cfg.openai.model,
            temperature=self.cfg.openai.temperature,
            provider=self.cfg.openai.provider,
            patch_system_prompt=patch_system_prompt,
            patch_schema_hint=self.cfg.prompts.patch_schema_hint,
            bootstrap_system_prompt=self.cfg.prompts.bootstrap_system_prompt,
        )
        # -- Optional RLM-based documentation retrieval (needed for bootstrap) --
        rlm_retriever = None
        if self.cfg.retrieval.enabled and self.cfg.retrieval.docs_dir:
            docs_path = self.repo_root / self.cfg.retrieval.docs_dir
            if docs_path.exists():
                from agent.strategies.retrieval.docs_loader import DocsCorpus
                from agent.strategies.retrieval.rlm_retriever import RLMDocRetriever

                docs_corpus = DocsCorpus(docs_path)
                if len(docs_corpus) > 0:
                    rlm_retriever = RLMDocRetriever(docs_corpus, self.cfg.retrieval)
                    print(
                        f"[optimizer] RLM doc retrieval enabled: {len(docs_corpus)} pages "
                        f"({docs_corpus.total_chars} chars), "
                        f"model={self.cfg.retrieval.retrieval_model or 'gpt-4o-mini'}, "
                        f"max_iterations={self.cfg.retrieval.rlm_max_iterations}",
                        flush=True,
                    )
                else:
                    print(
                        f"[optimizer] RLM doc retrieval: no pages found in {docs_path}",
                        flush=True,
                    )
        self.rlm_retriever = rlm_retriever
        self.bootstrap = BootstrapStage(
            repo_root=self.repo_root,
            candidate_dir=self.cfg.candidate_dir,
            seed_from_backend=self.cfg.seed_from_backend,
            prompt_config=self.cfg.prompts,
            openai_client=self.openai_client,
            rlm_retriever=rlm_retriever,
        )
        self.proposer = make_proposer(
            self.cfg.strategies.proposer,
            self.openai_client,
            self.cfg.prompts,
            retriever=retriever,
        )
        self.memory = make_memory(self.cfg.strategies.memory)
        self.scorer = make_scorer(self.cfg.strategies.scorer)
        self._progress_total_steps = 1
        self._progress_completed_steps = 0
        self._progress_start_time = 0.0

        # Directory for per-iteration code snapshots.
        self.snapshots_dir = self.run_dir / "snapshots"
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_llm_output(
        self,
        *,
        iteration: int,
        problem: ProblemConfig,
        proposal: dict[str, Any],
        proposal_mode: str,
    ) -> None:
        llm_payload = {
            "event": "llm_proposal",
            "iteration": iteration,
            "problem_id": problem.problem_id,
            "candidate_file": str(self._candidate_file(problem.problem_id)),
            "proposal_mode": proposal_mode,
            "model": self.cfg.openai.model,
            "diagnosis": proposal.get("diagnosis", []),
            "hypotheses": proposal.get("hypotheses", []),
            "test_expectations": proposal.get("test_expectations", []),
            "candidate_code": proposal.get("candidate_code", ""),
        }
        out_path = self.logger.append_llm(llm_payload)
        print(f"Saved LLM output: {out_path}", flush=True)

    # ------------------------------------------------------------------
    # Code snapshots — lets you see how the solution evolves over time
    # ------------------------------------------------------------------

    def _snapshot_code(self, problem_id: int, iteration: int, tag: str = "") -> Path:
        """Save a timestamped copy of the current candidate code.

        Files are saved to ``<run_dir>/snapshots/problem_<id>/iter_<n>[_<tag>].py``.
        """
        candidate_file = self._candidate_file(problem_id)
        problem_snap_dir = self.snapshots_dir / f"problem_{problem_id}"
        problem_snap_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        snap_name = f"iter_{iteration}{suffix}.py"
        snap_path = problem_snap_dir / snap_name
        shutil.copy2(candidate_file, snap_path)
        return snap_path

    # ------------------------------------------------------------------
    # Progress bar
    # ------------------------------------------------------------------

    def _init_progress(self) -> None:
        problems_count = len(self.cfg.problems)
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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _candidate_file(self, problem_id: int) -> Path:
        return self.repo_root / self.cfg.candidate_dir / f"{problem_id}_agent.py"

    def _apply_code(self, problem_id: int, candidate_code: str) -> None:
        """Overwrite the candidate file with the full code from the LLM.

        Raises ``ValueError`` if the code is empty or does not look like a
        Python file that defines ``solution``.
        """
        code = candidate_code.strip()
        if not code:
            raise ValueError("LLM returned empty candidate_code")
        if "def solution" not in code and "solution" not in code:
            raise ValueError(
                "LLM candidate_code does not appear to define `solution`"
            )
        candidate_file = self._candidate_file(problem_id)
        candidate_file.write_text(code + "\n", encoding="utf-8")

    @staticmethod
    def _is_eval_failure(result) -> bool:
        """Check if a BackendEvalResult represents a hard infrastructure failure.

        Correctness errors (shape mismatches, value diffs) are NOT treated as
        eval failures — they produced useful feedback the LLM can act on.
        Only treat as failure if the modal run itself crashed (no rank data)
        or timed out.
        """
        status = result.eval_feedback.get("status")
        if status == "timeout":
            return True
        if status == "error":
            # If we got rank-level data back, this is a correctness error, not
            # an infrastructure failure.  The LLM can learn from the per-rank
            # error messages.
            ranks = result.summary_rank0.get("ranks", [])
            if len(ranks) > 0:
                return False
            return True
        return False

    def _save_traces_for_iteration(
        self,
        iteration: int,
        problem_id: int,
        reference: Any,
        candidate: Any,
    ) -> None:
        """Copy profiler traces to run_dir so they persist per iteration."""
        traces_dir = self.run_dir / "traces" / f"problem_{problem_id}" / f"iter_{iteration}"
        traces_dir.mkdir(parents=True, exist_ok=True)
        for backend, result in [("reference", reference), ("agent", candidate)]:
            if result is None or not hasattr(result, "logs_dir"):
                continue
            try:
                src = Path(result.logs_dir)
            except (TypeError, ValueError):
                continue
            if not src.exists():
                continue
            dst = traces_dir / backend
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.glob("trace_*.json"):
                try:
                    shutil.copy2(f, dst / f.name)
                except OSError:
                    pass

    @staticmethod
    def _extract_error_message(result) -> str:
        """Extract the most informative error message from a BackendEvalResult.

        Prefers rank-level error messages (which contain stack traces and
        correctness details) over the top-level error string.
        """
        # Try rank-level errors first — they have the real details.
        ranks = result.summary_rank0.get("ranks", [])
        rank_errors = []
        for r in ranks:
            if isinstance(r, dict) and r.get("error"):
                rank_errors.append(f"rank {r.get('rank', '?')}: {r['error']}")
        if rank_errors:
            # Show unique errors only to avoid repetition.
            seen = set()
            unique = []
            for e in rank_errors:
                # Normalize to just the error text (strip rank prefix for dedup).
                key = e.split(": ", 1)[-1] if ": " in e else e
                if key not in seen:
                    seen.add(key)
                    unique.append(e)
            return "; ".join(unique[:3])  # Cap at 3 unique errors.

        # Fall back to top-level error.
        return (
            result.eval_feedback.get("error")
            or result.summary_rank0.get("error")
            or "unknown error"
        )

    def _collect_prior_kernels(
        self,
        problem_id: int,
        history: list[dict],
    ) -> list[IterationSnapshot]:
        """Build the list of prior iteration kernel snapshots for this problem.

        Each snapshot includes the actual code and its evaluation result,
        following the Kernel Devin pattern where the LLM sees every prior
        attempt so it can learn from the full trajectory.
        """
        snapshots: list[IterationSnapshot] = []
        problem_snap_dir = self.snapshots_dir / f"problem_{problem_id}"

        # Build a map of iteration -> history entry for this problem.
        iter_entries: dict[int, dict] = {}
        for entry in history:
            if entry.get("problem_id") == problem_id and "iteration" in entry:
                iter_entries[entry["iteration"]] = entry

        # Walk snapshots in order.
        if problem_snap_dir.exists():
            snap_files = sorted(problem_snap_dir.glob("iter_*.py"))
            for snap_file in snap_files:
                # Parse iteration number from filename (iter_0_bootstrap.py, iter_1.py, etc.)
                name = snap_file.stem  # e.g. "iter_0_bootstrap" or "iter_1"
                parts = name.split("_")
                try:
                    iteration = int(parts[1])
                except (IndexError, ValueError):
                    continue

                try:
                    code = snap_file.read_text(encoding="utf-8")
                except Exception:
                    continue

                entry = iter_entries.get(iteration, {})
                snapshots.append(
                    IterationSnapshot(
                        iteration=iteration,
                        code=code,
                        eval_feedback=entry.get("candidate_feedback", {}),
                        score=entry.get("score", 0.0),
                        eval_failed=entry.get("eval_failed", False),
                        eval_error=entry.get("eval_error", ""),
                    )
                )

        return snapshots

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _evaluate_all(self, iteration: int) -> tuple[float, list[dict]]:
        entries: list[dict] = []
        scores: list[float] = []
        for problem in self.cfg.problems:
            reference, candidate = self.evaluator.evaluate_pair(problem)

            self._save_traces_for_iteration(iteration, problem.problem_id, reference, candidate)

            if self._is_eval_failure(candidate):
                error_msg = self._extract_error_message(candidate)
                is_timeout = candidate.eval_feedback.get("is_timeout", False)
                failure_kind = "timeout" if is_timeout else "error"
                print(
                    f"[optimizer] Candidate eval FAILED ({failure_kind}) for "
                    f"problem {problem.problem_id}: {error_msg}",
                    flush=True,
                )
                scores.append(0.0)
                entries.append(
                    {
                        "problem_id": problem.problem_id,
                        "reference_summary": reference.summary_rank0,
                        "candidate_summary": candidate.summary_rank0,
                        "reference_feedback": reference.eval_feedback,
                        "candidate_feedback": candidate.eval_feedback,
                        "score": 0.0,
                        "eval_failed": True,
                        "eval_failure_kind": failure_kind,
                        "eval_error": error_msg,
                    }
                )
                continue

            # Enrich candidate feedback with rank-level error details when
            # the status is "error" (correctness failures like shape mismatches)
            # so the LLM gets actionable error information.
            candidate_feedback = dict(candidate.eval_feedback)
            if candidate.eval_feedback.get("status") == "error":
                error_detail = self._extract_error_message(candidate)
                candidate_feedback["error"] = error_detail
                # Mark correctness as failed since rank-level errors exist.
                ranks = candidate.summary_rank0.get("ranks", [])
                failed_ranks = [
                    r.get("rank") for r in ranks
                    if isinstance(r, dict) and r.get("status") == "error"
                ]
                candidate_feedback["correctness"] = {
                    "all_ok": False,
                    "failed_ranks": failed_ranks,
                }

            try:
                score = self.scorer.score(reference.summary_rank0, candidate.summary_rank0)
            except Exception:
                # Scorer may fail if no aggregate timing data (all ranks errored).
                score = 0.0
            scores.append(score)
            entries.append(
                {
                    "problem_id": problem.problem_id,
                    "reference_summary": reference.summary_rank0,
                    "candidate_summary": candidate.summary_rank0,
                    "reference_feedback": reference.eval_feedback,
                    "candidate_feedback": candidate_feedback,
                    "score": score,
                }
            )
        mean_score = sum(scores) / len(scores) if scores else 0.0
        return mean_score, entries

    # ------------------------------------------------------------------
    # RLM retrieval
    # ------------------------------------------------------------------

    @staticmethod
    def _build_retrieval_query(
        problem: ProblemConfig,
        eval_feedback: dict,
        proposal_mode: str,
        current_code: str,
    ) -> str:
        """Build a targeted retrieval query from eval feedback signals.

        The resulting query guides the RLM to search for the most relevant
        NVSHMEM documentation given what went wrong (or right) in the latest
        evaluation.
        """
        parts = [f"Problem {problem.problem_id}: distributed Triton/NVSHMEM kernel."]
        parts.append(f"Mode: {proposal_mode}")

        # Extract specific signals from eval feedback.
        correctness = eval_feedback.get("correctness", {})
        if not correctness.get("all_ok", True):
            parts.append(f"Correctness issue: {json.dumps(correctness, default=str)}")

        error = eval_feedback.get("error", "")
        if error:
            parts.append(f"Error: {error}")

        if eval_feedback.get("is_timeout"):
            parts.append(
                "Code timed out -- likely a synchronization/barrier hang. "
                "Look for NVSHMEM barrier, sync, and team-based synchronization APIs."
            )

        status = eval_feedback.get("status", "")
        if status == "error":
            parts.append(
                "The evaluation crashed. Look for common NVSHMEM pitfalls: "
                "symmetric memory allocation, buffer alignment, correct PE indexing."
            )

        # Add a truncated code snippet for API-matching context.
        parts.append(f"Current code excerpt:\n{current_code[:800]}")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Proposal
    # ------------------------------------------------------------------

    def _proposal_for_problem(
        self,
        problem: ProblemConfig,
        history: list[dict],
        *,
        prior_kernels: list[IterationSnapshot] | None = None,
    ) -> dict:
        candidate_file = self._candidate_file(problem.problem_id)
        current_code = candidate_file.read_text(encoding="utf-8")
        latest = next((h for h in reversed(history) if h["problem_id"] == problem.problem_id), {})
        latest_feedback = latest.get("candidate_feedback", {})
        correctness = latest_feedback.get("correctness", {})

        # Determine proposal mode: also check rank-level errors for correctness
        # issues even when status is "error" (e.g. shape mismatches).
        if latest.get("eval_failed"):
            proposal_mode = "correctness_fix"
        elif not correctness.get("all_ok", True):
            proposal_mode = "correctness_fix"
        elif latest_feedback.get("status") == "error":
            # Rank-level errors (shape mismatch, assertion errors) are
            # correctness issues even though the status is "error".
            proposal_mode = "correctness_fix"
        else:
            proposal_mode = "perf_opt"

        # -- RLM documentation retrieval (feedback-driven) --
        retrieved_docs = ""
        if self.rlm_retriever is not None:
            query = self._build_retrieval_query(
                problem, latest_feedback, proposal_mode, current_code
            )
            print(
                f"[optimizer] Running RLM doc retrieval for problem {problem.problem_id} "
                f"(mode={proposal_mode})...",
                flush=True,
            )
            retrieved_docs = self.rlm_retriever.retrieve(query)
            print(
                f"[optimizer] RLM retrieval returned {len(retrieved_docs)} chars",
                flush=True,
            )

        # Filter history to this problem only for the memory summary.
        problem_history = [h for h in history if h.get("problem_id") == problem.problem_id]

        ctx = ProposalContext(
            problem_id=problem.problem_id,
            candidate_file=candidate_file,
            current_code=current_code,
            latest_metrics=latest,
            memory_summary=self.memory.summarize(problem_history),
            proposal_mode=proposal_mode,
            eval_feedback=latest_feedback,
            retrieved_docs=retrieved_docs,
            prior_kernels=prior_kernels,
        )
        proposal = self.proposer.propose(ctx)
        proposal["mode"] = proposal_mode
        return proposal

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> Path:
        history: list[dict] = []
        self._init_progress()

        # ── Bootstrap ──
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
            # Snapshot the initial bootstrapped code as iter_0.
            self._snapshot_code(p.problem_id, 0, tag="bootstrap")
            self._advance_progress(f"bootstrapped problem {p.problem_id}")

        self.evaluator.precompute_references(self.cfg.problems)
        self._advance_progress("precomputed references")

        # ── Iterative loop ──
        for iteration in range(1, self.cfg.max_iterations + 1):
            score, eval_entries = self._evaluate_all(iteration)
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

                # ── NO ROLLBACK on eval failure ──
                # Instead of rolling back, we keep the current (possibly broken)
                # code and pass the LLM the full history of prior kernels + eval
                # results so it can learn from its mistakes and iterate forward
                # (Kernel Devin pattern).
                latest_entry = next(
                    (e for e in reversed(eval_entries) if e["problem_id"] == p.problem_id),
                    None,
                )
                if latest_entry and latest_entry.get("eval_failed"):
                    failure_kind = latest_entry.get("eval_failure_kind", "error")
                    print(
                        f"[optimizer] Eval {failure_kind} for problem {p.problem_id} — "
                        f"keeping current code and passing full history to LLM.",
                        flush=True,
                    )
                    self.logger.append(
                        {
                            "iteration": iteration,
                            "event": "eval_failure_no_rollback",
                            "problem_id": p.problem_id,
                            "eval_failure_kind": failure_kind,
                            "eval_error": latest_entry.get("eval_error", ""),
                        }
                    )

                # Collect all prior kernel snapshots for this problem.
                prior_kernels = self._collect_prior_kernels(p.problem_id, history)

                proposal = self._proposal_for_problem(
                    p, history, prior_kernels=prior_kernels,
                )
                cycle_event = (
                    "perf_opt_cycle"
                    if proposal.get("mode", "perf_opt") == "perf_opt"
                    else "correctness_fix_cycle"
                )
                self._log_llm_output(
                    iteration=iteration,
                    problem=p,
                    proposal=proposal,
                    proposal_mode=str(proposal.get("mode", "perf_opt")),
                )
                self.logger.append(
                    {
                        "iteration": iteration,
                        "event": cycle_event,
                        "problem_id": p.problem_id,
                    }
                )
                try:
                    self._apply_code(
                        p.problem_id,
                        proposal.get("candidate_code", ""),
                    )
                    # Snapshot the newly written code for history tracking.
                    snap = self._snapshot_code(p.problem_id, iteration)
                    self.logger.append(
                        {
                            "iteration": iteration,
                            "event": "proposal_applied",
                            "problem_id": p.problem_id,
                            "snapshot": str(snap),
                        }
                    )
                    print(
                        f"[optimizer] Applied new code for problem {p.problem_id} "
                        f"(snapshot: {snap.name})",
                        flush=True,
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    rollback.rollback()
                    self.logger.append(
                        {
                            "iteration": iteration,
                            "event": "proposal_failed",
                            "problem_id": p.problem_id,
                            "error": str(exc),
                        }
                    )
                    print(
                        f"[optimizer] Proposal failed for problem {p.problem_id}: {exc}",
                        flush=True,
                    )
                self._advance_progress(
                    f"completed iteration {iteration} patch cycle for problem {p.problem_id}"
                )

        summary = {"name": self.cfg.name, "run_dir": str(self.run_dir), "iterations": len(history)}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"LLM outputs index: {self.run_dir / 'llm_outputs.jsonl'}", flush=True)
        print(f"Code snapshots:    {self.snapshots_dir}", flush=True)
        total_elapsed = time.perf_counter() - self._progress_start_time
        print(f"Experiment finished in {total_elapsed:.1f}s", flush=True)
        return self.run_dir
