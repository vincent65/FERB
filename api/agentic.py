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
import shlex
import shutil
import subprocess
import selectors
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from typing import Any

from openai import OpenAI


ROOT_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DIR = ROOT_DIR / "reference"
TRITON_DIR = ROOT_DIR / "solutions_triton"
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


def _clip(text: str, limit: int) -> str:
    if limit <= 0:
        return text or ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... [truncated]"


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


def _extract_plan_and_code(text: str) -> tuple[str, str]:
    src = (text or "").strip()
    marker = "###CODE_START"
    if marker in src:
        head, tail = src.split(marker, 1)
        plan = head.strip()
        code = _extract_python_code(tail.strip())
        return plan, code
    return "No explicit plan returned.", _extract_python_code(src)


def _read_problem_reference(problem_id: int) -> str:
    path = REFERENCE_DIR / f"{problem_id}.py"
    if not path.exists():
        return f"# Missing reference file for problem {problem_id}: {path}"
    return path.read_text(encoding="utf-8")


def _read_triton_seed(problem_id: int) -> str:
    path = TRITON_DIR / f"{problem_id}_triton.py"
    if not path.exists():
        return ""
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
    target_backend: str,
    previous_best_code: str | None,
    previous_feedback: str | None,
    previous_eval_feedback: str | None,
    quality_feedback: str | None,
    topology_json: str | None,
) -> tuple[str, str]:
    """
    Returns (candidate_code, model_feedback).
    """
    client = _openai_client()
    reference_code = _read_problem_reference(problem_id)
    triton_seed = _read_triton_seed(problem_id)
    problem_docs = _read_problem_descriptions()
    best_code_section = previous_best_code if previous_best_code else "# none yet"
    feedback_section = previous_feedback if previous_feedback else "# first iteration"
    eval_feedback_section = previous_eval_feedback if previous_eval_feedback else "# no evaluator feedback yet"
    quality_feedback_section = quality_feedback if quality_feedback else "# no quality issues from last attempt"
    topology_section = topology_json if topology_json else "{}"

    backend_requirements = (
        "Target backend is triton+nvshmem. Use Triton kernels and NVSHMEM APIs; avoid plain NCCL-only all_reduce wrappers."
        if target_backend == "triton"
        else "Target backend is reference/pytorch distributed baseline."
    )

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
4) Return with this exact format:
PLAN: <1-3 short bullets of what you'll try>
###CODE_START
<python code only>

Backend constraint:
{backend_requirements}

Triton seed (if available):
{triton_seed if triton_seed else "# none"}

Quality feedback from previous attempt:
{quality_feedback_section}

Evaluator feedback from previous attempt (errors, logs, metrics):
{eval_feedback_section}
"""

    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_OPTIMIZER_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    plan_text, code = _extract_plan_and_code(response.output_text)
    return code, plan_text


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


def _normalize_evaluator_command(command: str) -> str:
    """
    If torchrun is unavailable, replace it with:
      python -m torch.distributed.run ...
    """
    if shutil.which("torchrun"):
        return command
    try:
        tokens = shlex.split(command)
    except Exception:
        return command
    if not tokens:
        return command
    if tokens[0] != "torchrun":
        return command
    py = _find_python_with_torch() or (sys.executable or "python3")
    rewritten = [py, "-m", "torch.distributed.run"] + tokens[1:]
    return " ".join(shlex.quote(t) for t in rewritten)


def _python_has_torch(python_exe: str) -> bool:
    try:
        proc = subprocess.run(
            [python_exe, "-c", "import torch; print(torch.__version__)"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _find_python_with_torch() -> str | None:
    """
    Find a Python interpreter with torch installed.
    Priority:
    1) FERB_EVAL_PYTHON env var
    2) python3 in PATH
    3) python in PATH
    4) current interpreter
    """
    candidates: list[str] = []
    env_py = os.environ.get("FERB_EVAL_PYTHON", "").strip()
    if env_py:
        candidates.append(env_py)
    for name in ("python3", "python"):
        p = shutil.which(name)
        if p:
            candidates.append(p)
    if sys.executable:
        candidates.append(sys.executable)

    seen: set[str] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if _python_has_torch(c):
            return c
    return None


def _candidate_quality_issues(code: str, target_backend: str) -> list[str]:
    src = (code or "").lower()
    issues: list[str] = []
    if not src.strip():
        issues.append("empty candidate code")
        return issues
    if "def solution" not in src:
        issues.append("missing solution(...) function")
    if target_backend == "triton":
        if "import triton" not in src:
            issues.append("missing Triton import")
        if "nvshmem" not in src:
            issues.append("missing NVSHMEM usage")
        if "dist.all_reduce" in src and "triton" not in src:
            issues.append("NCCL-only fallback detected; expected triton/nvshmem solution")
    return issues


def _is_infra_blocker(stderr: str, target_backend: str) -> str | None:
    s = (stderr or "").lower()
    if "token missing" in s and "modal" in s:
        return "Modal authentication token missing (modal CLI not logged in)."
    if "could not authenticate client" in s and "modal" in s:
        return "Modal authentication failed (check your modal token credentials)."
    if target_backend == "triton":
        if "no module named 'triton'" in s or "no module named triton" in s:
            return "Evaluator environment is missing Triton."
        if "no module named 'nvshmem'" in s or "no module named nvshmem" in s:
            return "Evaluator environment is missing NVSHMEM bindings."
        if "torch.cuda" in s and ("not available" in s or "not compiled" in s):
            return "Evaluator environment has no CUDA."
        if "torch.cuda.set_device" in s:
            return "Evaluator environment cannot bind CUDA devices (likely no GPU)."
    return None


def _rewrite_python_launcher(command: str, python_exe: str | None) -> str:
    """
    If command begins with python/python3, replace with explicit interpreter.
    """
    if not python_exe:
        return command
    try:
        tokens = shlex.split(command)
    except Exception:
        return command
    if not tokens:
        return command
    head = tokens[0]
    if head in {"python", "python3"}:
        tokens[0] = python_exe
        return " ".join(shlex.quote(t) for t in tokens)
    return command


def _run_evaluator(
    command_template: str,
    candidate_path: Path,
    timeout_s: int,
    evaluator_python: str | None = None,
) -> tuple[float, str, str]:
    """
    Run evaluator command and parse score from stdout.
    """
    command = command_template.format(candidate_path=str(candidate_path))
    py_with_torch = None
    if evaluator_python:
        # Only honor explicit evaluator_python if it exists and imports torch.
        if os.path.exists(evaluator_python) and _python_has_torch(evaluator_python):
            py_with_torch = evaluator_python
    if py_with_torch is None:
        py_with_torch = _find_python_with_torch()
    command = _rewrite_python_launcher(command, py_with_torch)
    command = _normalize_evaluator_command(command)
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
    if score is None:
        score = 0.0
    if proc.returncode != 0:
        if stderr:
            stderr = stderr + f"\n[evaluator_exit_code={proc.returncode}]"
        else:
            stderr = f"[evaluator_exit_code={proc.returncode}]"
    if py_with_torch is None:
        extra = (
            "\n[no_python_with_torch_found] Set FERB_EVAL_PYTHON to a Python interpreter "
            "that has torch installed."
        )
        stderr = (stderr + extra) if stderr else extra.lstrip()
    return score, stdout, stderr


def _run_evaluator_live(
    command_template: str,
    candidate_path: Path,
    timeout_s: int,
    evaluator_python: str | None = None,
    *,
    heartbeat_every_s: float = 2.0,
    tail_chars: int = 1200,
) -> Iterator[dict[str, Any]]:
    """
    Run evaluator command while yielding periodic heartbeats.

    This is mainly to keep the SSE UI "alive" while long-running evaluators
    (e.g. `modal run ...`) build images / wait for GPUs / execute benchmarks.
    """
    command = command_template.format(candidate_path=str(candidate_path))
    py_with_torch = None
    if evaluator_python:
        if os.path.exists(evaluator_python) and _python_has_torch(evaluator_python):
            py_with_torch = evaluator_python
    if py_with_torch is None:
        py_with_torch = _find_python_with_torch()
    command = _rewrite_python_launcher(command, py_with_torch)
    command = _normalize_evaluator_command(command)

    start = time.time()
    last_hb = 0.0
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []

    proc = subprocess.Popen(
        command,
        shell=True,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    assert proc.stdout is not None
    assert proc.stderr is not None

    sel = selectors.DefaultSelector()
    sel.register(proc.stdout, selectors.EVENT_READ, data="stdout")
    sel.register(proc.stderr, selectors.EVENT_READ, data="stderr")

    def _tail(txt: str) -> str:
        if tail_chars <= 0:
            return ""
        if len(txt) <= tail_chars:
            return txt
        return txt[-tail_chars:]

    while True:
        now = time.time()
        elapsed = now - start
        if elapsed > timeout_s:
            try:
                proc.kill()
            except Exception:
                pass
            stderr_chunks.append(f"\n[evaluator_timeout_s={timeout_s}]")
            break

        # Drain any available output.
        events = sel.select(timeout=0.5)
        for key, _mask in events:
            stream_name = key.data
            try:
                line = key.fileobj.readline()
            except Exception:
                line = ""
            if not line:
                continue
            if stream_name == "stdout":
                stdout_chunks.append(line)
            else:
                stderr_chunks.append(line)

        if (now - last_hb) >= heartbeat_every_s:
            last_hb = now
            out = "".join(stdout_chunks)
            err = "".join(stderr_chunks)
            yield {
                "elapsed_s": int(elapsed),
                "stdout_tail": _tail(out),
                "stderr_tail": _tail(err),
            }

        rc = proc.poll()
        if rc is not None:
            # Drain remaining buffered output
            for stream, name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
                try:
                    rest = stream.read()
                except Exception:
                    rest = ""
                if not rest:
                    continue
                if name == "stdout":
                    stdout_chunks.append(rest)
                else:
                    stderr_chunks.append(rest)
            break

    stdout = "".join(stdout_chunks)
    stderr = "".join(stderr_chunks)
    score = _parse_score(stdout)
    if score is None:
        score = 0.0
    if proc.returncode not in (0, None):
        if stderr:
            stderr = stderr + f"\n[evaluator_exit_code={proc.returncode}]"
        else:
            stderr = f"[evaluator_exit_code={proc.returncode}]"
    if py_with_torch is None:
        extra = (
            "\n[no_python_with_torch_found] Set FERB_EVAL_PYTHON to a Python interpreter "
            "that has torch installed."
        )
        stderr = (stderr + extra) if stderr else extra.lstrip()
    return (score, stdout, stderr)


def run_agentic_optimization(
    *,
    objective: str,
    problem_id: int,
    iterations: int = 3,
    model: str = "gpt-4o-mini",
    target_backend: str = "triton",
    topology_json_path: str | None = None,
    evaluator_command: str | None = None,
    evaluator_timeout_s: int = 240,
    evaluator_python: str | None = None,
    include_full_code: bool = False,
    include_trace_output: bool = True,
    trace_text_limit: int = 0,
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
    previous_eval_feedback: str | None = None
    trace: list[IterationResult] = []

    for idx in range(1, iterations + 1):
        quality_feedback = None
        model_feedback = ""
        code = ""
        for _attempt in range(1, 4):
            code, model_feedback = _generate_candidate(
                model=model,
                objective=objective,
                problem_id=problem_id,
                iteration_idx=idx,
                target_backend=target_backend,
                previous_best_code=best_code,
                previous_feedback=previous_feedback,
                previous_eval_feedback=previous_eval_feedback,
                quality_feedback=quality_feedback,
                topology_json=topology_json,
            )
            issues = _candidate_quality_issues(code, target_backend)
            if not issues:
                break
            quality_feedback = "Quality issues: " + "; ".join(issues)

        candidate_path = run_dir / f"candidate_iter_{idx}.py"
        candidate_path.write_text(code, encoding="utf-8")

        score = 0.0
        eval_stdout = ""
        eval_stderr = ""
        if evaluator_command:
            score, eval_stdout, eval_stderr = _run_evaluator(
                evaluator_command,
                candidate_path,
                timeout_s=evaluator_timeout_s,
                evaluator_python=evaluator_python,
            )
            previous_eval_feedback = (
                "Evaluator stdout:\n"
                + (eval_stdout or "")
                + "\n\nEvaluator stderr:\n"
                + (eval_stderr or "")
                + f"\n\nScore: {score}"
            )
            blocker = _is_infra_blocker(eval_stderr, target_backend)
            if blocker:
                # Don't keep iterating when the evaluator environment can't run the target backend.
                if best_code is None:
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
                        evaluator_stderr=(eval_stderr or "") + f"\n[blocked] {blocker}",
                        model_feedback=model_feedback,
                    )
                )
                break
        else:
            previous_eval_feedback = None

        choose_new_best = False
        if best_code is None:
            choose_new_best = True
        elif best_score is None or score > best_score:
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

    result = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "problem_id": problem_id,
        "objective": objective,
        "iterations": iterations,
        "model": model,
        "best_score": best_score,
        "best_candidate_path": best_candidate_path,
        "trace": [
            {
                "iteration": t.iteration,
                "candidate_path": t.candidate_path,
                "score": t.score,
                "model_feedback": _clip(t.model_feedback, trace_text_limit),
                "evaluator_stdout": (
                    _clip(t.evaluator_stdout, trace_text_limit) if include_trace_output else ""
                ),
                "evaluator_stderr": (
                    _clip(t.evaluator_stderr, trace_text_limit) if include_trace_output else ""
                ),
            }
            for t in trace
        ],
    }
    if include_full_code:
        result["best_code"] = best_code
    return result


def stream_agentic_optimization_events(
    *,
    objective: str,
    problem_id: int,
    iterations: int = 3,
    model: str = "gpt-4o-mini",
    target_backend: str = "triton",
    topology_json_path: str | None = None,
    evaluator_command: str | None = None,
    evaluator_timeout_s: int = 240,
    evaluator_python: str | None = None,
    feedback_preview_chars: int = 1200,
) -> Iterator[dict[str, Any]]:
    """
    Stream per-iteration events for real-time visibility.
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
    previous_eval_feedback: str | None = None
    trace: list[IterationResult] = []

    yield {
        "type": "run_started",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "objective": objective,
        "problem_id": problem_id,
        "iterations": iterations,
        "model": model,
        "target_backend": target_backend,
    }

    for idx in range(1, iterations + 1):
        yield {"type": "iteration_started", "iteration": idx}

        quality_feedback = None
        code = ""
        model_feedback = ""
        for attempt in range(1, 4):
            code, model_feedback = _generate_candidate(
                model=model,
                objective=objective,
                problem_id=problem_id,
                iteration_idx=idx,
                target_backend=target_backend,
                previous_best_code=best_code,
                previous_feedback=previous_feedback,
                previous_eval_feedback=previous_eval_feedback,
                quality_feedback=quality_feedback,
                topology_json=topology_json,
            )
            yield {
                "type": "agent_thought",
                "iteration": idx,
                "attempt": attempt,
                "text": model_feedback,
            }
            issues = _candidate_quality_issues(code, target_backend)
            if not issues:
                break
            quality_feedback = "Quality issues: " + "; ".join(issues)
            yield {
                "type": "quality_reject",
                "iteration": idx,
                "attempt": attempt,
                "issues": issues,
            }

        candidate_path = run_dir / f"candidate_iter_{idx}.py"
        candidate_path.write_text(code, encoding="utf-8")
        yield {
            "type": "candidate_generated",
            "iteration": idx,
            "candidate_path": str(candidate_path),
            "feedback_preview": _clip(model_feedback, feedback_preview_chars),
            "candidate_code_preview": _clip(code, 20000),
        }

        score = 0.0
        eval_stdout = ""
        eval_stderr = ""
        if evaluator_command:
            yield {
                "type": "evaluation_started",
                "iteration": idx,
                "command": evaluator_command,
            }
            timeout_s = evaluator_timeout_s
            if "modal run" in (evaluator_command or "") and timeout_s < 1800:
                # Modal runs often include image build/pull and GPU scheduling latency.
                timeout_s = 1800

            live = _run_evaluator_live(
                evaluator_command,
                candidate_path,
                timeout_s=timeout_s,
                evaluator_python=evaluator_python,
                heartbeat_every_s=2.0,
                tail_chars=1500,
            )
            while True:
                try:
                    hb = next(live)
                    yield {
                        "type": "evaluation_heartbeat",
                        "iteration": idx,
                        **hb,
                    }
                except StopIteration as stop:
                    score, eval_stdout, eval_stderr = stop.value
                    break
            previous_eval_feedback = (
                "Evaluator stdout:\n"
                + (eval_stdout or "")
                + "\n\nEvaluator stderr:\n"
                + (eval_stderr or "")
                + f"\n\nScore: {score}"
            )
            yield {
                "type": "evaluation_completed",
                "iteration": idx,
                "score": score,
                "stdout_preview": _clip(eval_stdout, 0),
                "stderr_preview": _clip(eval_stderr, 0),
            }

            blocker = _is_infra_blocker(eval_stderr, target_backend)
            if blocker:
                if best_code is None:
                    best_code = code
                    best_score = score
                    best_candidate_path = str(candidate_path)
                yield {
                    "type": "blocked",
                    "iteration": idx,
                    "reason": blocker,
                    "hint": (
                        "If this is a Modal auth issue, run `modal token new` locally (same environment running the API), "
                        "then rerun. For Triton/NVSHMEM acceleration you must evaluate on a CUDA multi-GPU environment "
                        "(e.g. Modal H100x8) with triton + nvshmem installed."
                    ),
                }
                break
        else:
            previous_eval_feedback = None

        choose_new_best = False
        if best_code is None:
            choose_new_best = True
        elif best_score is None or score > best_score:
            choose_new_best = True

        if choose_new_best:
            best_code = code
            best_score = score
            best_candidate_path = str(candidate_path)
            yield {
                "type": "best_updated",
                "iteration": idx,
                "best_score": best_score,
                "best_candidate_path": best_candidate_path,
            }
        else:
            yield {
                "type": "best_unchanged",
                "iteration": idx,
                "best_score": best_score,
                "best_candidate_path": best_candidate_path,
            }

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

    result = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "problem_id": problem_id,
        "objective": objective,
        "iterations": iterations,
        "model": model,
        "best_score": best_score,
        "best_candidate_path": best_candidate_path,
        "trace": [
            {
                "iteration": t.iteration,
                "candidate_path": t.candidate_path,
                "score": t.score,
                "model_feedback": _clip(t.model_feedback, 0),
            }
            for t in trace
        ],
    }
    yield {"type": "run_completed", "result": result}
