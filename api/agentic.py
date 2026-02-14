"""
Agentic optimization loop for FERB kernel solutions.

This module provides:
- GPT chat replies with FERB context
- Iterative candidate generation + optional evaluation
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DIR = ROOT_DIR / "reference"
RUNS_DIR = ROOT_DIR / ".agent_runs"


SYSTEM_CHAT_PROMPT = """You are an expert multi-GPU systems and CUDA engineer working on FERB.
Focus on practical kernel optimization advice for distributed training speedups.
Keep answers concise and action-oriented.
"""


SYSTEM_OPTIMIZER_PROMPT = """You are an autonomous kernel optimization agent for FERB.
You are iteratively improving solution quality.
Always return valid Python code for a FERB solution module with a `solution(...)` function.
Do not include markdown fences.
"""


@dataclass
class IterationResult:
    iteration: int
    candidate_path: str
    score: float | None
    evaluator_stdout: str
    evaluator_stderr: str
    model_feedback: str


def _openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return OpenAI(api_key=api_key)


def _extract_python_code(text: str) -> str:
    """
    Accept raw code or fenced markdown and return Python source.
    """
    src = (text or "").strip()
    fence = re.search(r"```(?:python)?\s*(.*?)```", src, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    return src


def _read_problem_reference(problem_id: int) -> str:
    path = REFERENCE_DIR / f"{problem_id}.py"
    if not path.exists():
        return f"# Missing reference file for problem {problem_id}: {path}"
    return path.read_text(encoding="utf-8")


def _read_problem_descriptions() -> str:
    path = REFERENCE_DIR / "problems.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def gpt_chat_reply(message: str, model: str = "gpt-4o-mini") -> str:
    """
    GPT-backed chat with FERB context.
    """
    client = _openai_client()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_CHAT_PROMPT},
            {"role": "user", "content": message},
        ],
    )
    return (response.output_text or "").strip()


def _generate_candidate(
    *,
    model: str,
    objective: str,
    problem_id: int,
    iteration_idx: int,
    previous_best_code: str | None,
    previous_feedback: str | None,
    topology_json: str | None,
) -> tuple[str, str]:
    """
    Returns (candidate_code, model_feedback).
    """
    client = _openai_client()
    reference_code = _read_problem_reference(problem_id)
    problem_docs = _read_problem_descriptions()
    best_code_section = previous_best_code if previous_best_code else "# none yet"
    feedback_section = previous_feedback if previous_feedback else "# first iteration"
    topology_section = topology_json if topology_json else "{}"

    user_prompt = f"""
Objective:
{objective}

Problem ID:
{problem_id}

Iteration:
{iteration_idx}

Problem notes:
{problem_docs}

Reference implementation:
{reference_code}

Current best candidate:
{best_code_section}

Feedback from previous iteration:
{feedback_section}

Topology JSON:
{topology_section}

Task:
1) Write an improved solution module.
2) Keep function signature compatible with reference.
3) Prioritize correctness first, then performance.
4) Return ONLY code.
"""

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_OPTIMIZER_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    code = _extract_python_code(response.output_text)

    feedback_prompt = f"""
You proposed this candidate. Give a short self-critique with:
- likely strengths
- likely risks
- what to change next iteration

Candidate code:
{code}
"""
    feedback_resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": "You are a strict code reviewer for FERB kernels."},
            {"role": "user", "content": feedback_prompt},
        ],
    )
    feedback = (feedback_resp.output_text or "").strip()
    return code, feedback


def _parse_score(stdout: str) -> float | None:
    """
    Accept score in either:
    - JSON object: {"score": 123.4}
    - plain text line: score=123.4
    """
    txt = (stdout or "").strip()
    if not txt:
        return None

    lines = [line.strip() for line in txt.splitlines() if line.strip()]
    if not lines:
        return None

    # Try JSON from last line then full text
    for candidate in (lines[-1], txt):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and "score" in obj:
                return float(obj["score"])
        except Exception:
            pass

    # Try score=...
    for line in reversed(lines):
        m = re.search(r"score\s*=\s*([0-9]+(?:\.[0-9]+)?)", line, flags=re.IGNORECASE)
        if m:
            return float(m.group(1))

    return None


def _run_evaluator(command_template: str, candidate_path: Path, timeout_s: int) -> tuple[float | None, str, str]:
    """
    Run evaluator command and parse score from stdout.
    """
    command = command_template.format(candidate_path=str(candidate_path))
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    score = _parse_score(stdout)
    return score, stdout, stderr


def run_agentic_optimization(
    *,
    objective: str,
    problem_id: int,
    iterations: int = 3,
    model: str = "gpt-4o-mini",
    topology_json_path: str | None = None,
    evaluator_command: str | None = None,
    evaluator_timeout_s: int = 240,
) -> dict[str, Any]:
    """
    Iterative generate/evaluate/select loop.
    """
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    topology_json = None
    if topology_json_path:
        topo_path = Path(topology_json_path)
        if topo_path.exists():
            topology_json = topo_path.read_text(encoding="utf-8")

    run_id = uuid.uuid4().hex[:12]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    best_code: str | None = None
    best_score: float | None = None
    best_candidate_path: str | None = None
    previous_feedback: str | None = None
    trace: list[IterationResult] = []

    for idx in range(1, iterations + 1):
        code, model_feedback = _generate_candidate(
            model=model,
            objective=objective,
            problem_id=problem_id,
            iteration_idx=idx,
            previous_best_code=best_code,
            previous_feedback=previous_feedback,
            topology_json=topology_json,
        )
        candidate_path = run_dir / f"candidate_iter_{idx}.py"
        candidate_path.write_text(code, encoding="utf-8")

        score = None
        eval_stdout = ""
        eval_stderr = ""
        if evaluator_command:
            score, eval_stdout, eval_stderr = _run_evaluator(
                evaluator_command,
                candidate_path,
                timeout_s=evaluator_timeout_s,
            )

        choose_new_best = False
        if best_code is None:
            choose_new_best = True
        elif score is not None and (best_score is None or score > best_score):
            choose_new_best = True

        if choose_new_best:
            best_code = code
            best_score = score
            best_candidate_path = str(candidate_path)

        previous_feedback = model_feedback
        trace.append(
            IterationResult(
                iteration=idx,
                candidate_path=str(candidate_path),
                score=score,
                evaluator_stdout=eval_stdout,
                evaluator_stderr=eval_stderr,
                model_feedback=model_feedback,
            )
        )

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "problem_id": problem_id,
        "objective": objective,
        "iterations": iterations,
        "model": model,
        "best_score": best_score,
        "best_candidate_path": best_candidate_path,
        "best_code": best_code,
        "trace": [
            {
                "iteration": t.iteration,
                "candidate_path": t.candidate_path,
                "score": t.score,
                "model_feedback": t.model_feedback,
                "evaluator_stdout": t.evaluator_stdout,
                "evaluator_stderr": t.evaluator_stderr,
            }
            for t in trace
        ],
    }
