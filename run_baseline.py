#!/usr/bin/env python3
"""Naive baseline: iteratively ask an LLM to improve kernel code.

No evaluation, no Modal, no feedback, no RAG, no memory. Just:

    1. Bootstrap from seed/reference
    2. Show the LLM the current code
    3. Ask it to make it faster
    4. Write the new code
    5. Repeat

At the end, we run ONE evaluation pass to score the final result.

This exists to demonstrate that the full FERB agent architecture (eval-driven
feedback, memory, retrieval, kernel history) meaningfully outperforms the
"obvious" approach of just asking an LLM to keep improving code.

Usage:
    python run_baseline.py --config experiments/problem6.yaml
    python run_baseline.py --config experiments/problem3.yaml --max-iters 6
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.config import ExperimentConfig


# ── Simple LLM wrapper ───────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a GPU kernel engineer. You write Triton/NVSHMEM kernels that "
    "implement distributed collective operations. Your goal is to produce "
    "correct, fast code."
)

IMPROVE_TEMPLATE = """\
Here is a Triton/NVSHMEM kernel for problem {problem_id} (a distributed collective op running on 8× H100 GPUs).

```python
{current_code}
```

Make this code faster. Focus on memory access patterns, parallelism, reducing \
synchronization overhead, and leveraging NVSHMEM features effectively. \
Make sure the code stays correct — the output must match the reference NCCL implementation exactly.

Return ONLY the complete Python source file. No explanations, no markdown fences.
The file must define a top-level `solution` function.
"""

BOOTSTRAP_TEMPLATE = """\
Write a Triton/NVSHMEM implementation for the following distributed collective operation.

## Reference implementation (PyTorch/NCCL)
```python
{reference_code}
```

The Triton version should use NVSHMEM for GPU-side communication instead of NCCL.
Return ONLY the complete Python source file. No explanations, no markdown fences.
The file must define a top-level `solution` function.
"""


def _call_llm(
    system: str,
    user: str,
    *,
    model: str = "gpt-5",
    temperature: float = 0.2,
    provider: str = "openai",
) -> str:
    """Single LLM call — no tools, no retries, no fancy parsing."""
    if provider in ("anthropic", "claude"):
        import anthropic
        client = anthropic.Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=16384,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        parts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
        return "\n".join(parts).strip()
    else:
        from openai import OpenAI
        client = OpenAI()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        request: dict[str, Any] = {"model": model, "input": messages}
        if not model.startswith("gpt-5"):
            request["temperature"] = temperature
        resp = client.responses.create(**request)
        return resp.output_text.strip()


def _strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    t = text.strip()
    if t.startswith("```") and t.endswith("```"):
        lines = t.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return t


# ── Main loop ─────────────────────────────────────────────────────────

def run_baseline(config_path: str, max_iters_override: int | None = None) -> None:
    repo_root = Path(__file__).resolve().parent
    cfg = ExperimentConfig.from_yaml(repo_root / config_path)

    if max_iters_override is not None:
        cfg.max_iterations = max_iters_override

    model = cfg.openai.model
    temperature = cfg.openai.temperature
    provider = cfg.openai.provider

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / cfg.run_root / f"{now}_baseline_{cfg.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = run_dir / "snapshots"
    snapshots_dir.mkdir(exist_ok=True)
    log_path = run_dir / "log.jsonl"

    # Write to solutions_baseline/ so we don't clobber agent solutions
    baseline_dir = repo_root / "solutions_baseline"
    baseline_dir.mkdir(exist_ok=True)

    def candidate_file(pid: int) -> Path:
        return baseline_dir / f"{pid}_baseline.py"

    def log(entry: dict) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

    def snapshot(pid: int, iteration: int, tag: str = "") -> Path:
        d = snapshots_dir / f"problem_{pid}"
        d.mkdir(parents=True, exist_ok=True)
        suffix = f"_{tag}" if tag else ""
        p = d / f"iter_{iteration}{suffix}.py"
        shutil.copy2(candidate_file(pid), p)
        return p

    print("=" * 60)
    print(f"BASELINE RUN (no eval): {cfg.name}")
    print(f"  Model: {provider}/{model}")
    print(f"  Problems: {[p.problem_id for p in cfg.problems]}")
    print(f"  Improvement iterations: {cfg.max_iterations}")
    print(f"  Run dir: {run_dir}")
    print("=" * 60)

    start_time = time.perf_counter()

    # ── Phase 0: Bootstrap ──
    for p in cfg.problems:
        cf = candidate_file(p.problem_id)
        if cf.exists():
            print(f"[baseline] Using existing candidate for problem {p.problem_id}")
        else:
            # Try seed from triton solutions
            seed = repo_root / f"solutions_triton/{p.problem_id}_triton.py"
            if seed.exists():
                shutil.copy2(seed, cf)
                print(f"[baseline] Seeded problem {p.problem_id} from triton solution")
            else:
                # Bootstrap via LLM from reference
                ref_path = repo_root / f"reference/{p.problem_id}.py"
                ref_code = ref_path.read_text(encoding="utf-8")
                print(f"[baseline] Bootstrapping problem {p.problem_id} from reference...")
                prompt = BOOTSTRAP_TEMPLATE.format(reference_code=ref_code)
                code = _call_llm(SYSTEM_PROMPT, prompt, model=model,
                                 temperature=temperature, provider=provider)
                code = _strip_fences(code)
                cf.write_text(code + "\n", encoding="utf-8")
                print(f"[baseline] Bootstrapped problem {p.problem_id} ({len(code)} chars)")

        snapshot(p.problem_id, 0, "bootstrap")
        log({"event": "bootstrap", "problem_id": p.problem_id,
             "file": str(candidate_file(p.problem_id))})

    # ── Phase 1: Blind iterative improvement (no eval feedback) ──
    for iteration in range(1, cfg.max_iterations + 1):
        print(f"\n{'─' * 50}")
        print(f"ITERATION {iteration}/{cfg.max_iterations}")
        print(f"{'─' * 50}")

        for p in cfg.problems:
            cf = candidate_file(p.problem_id)
            current_code = cf.read_text(encoding="utf-8")

            prompt = IMPROVE_TEMPLATE.format(
                problem_id=p.problem_id,
                current_code=current_code,
            )

            print(f"[baseline] Asking LLM to improve problem {p.problem_id}...")
            new_code = _call_llm(SYSTEM_PROMPT, prompt, model=model,
                                 temperature=temperature, provider=provider)
            new_code = _strip_fences(new_code)

            # Basic sanity check
            if not new_code or "def solution" not in new_code:
                print(f"[baseline] WARNING: LLM returned invalid code, keeping previous version")
                log({"iteration": iteration, "problem_id": p.problem_id,
                     "event": "invalid_response"})
                continue

            cf.write_text(new_code + "\n", encoding="utf-8")
            snap = snapshot(p.problem_id, iteration)
            print(f"[baseline] Applied new code ({len(new_code)} chars, snapshot: {snap.name})")
            log({"iteration": iteration, "problem_id": p.problem_id,
                 "event": "improved", "chars": len(new_code)})

    # ── Phase 2: One final evaluation to score the result ──
    print(f"\n{'=' * 60}")
    print("FINAL EVALUATION")
    print("=" * 60)

    from agent.eval.modal_evaluator import ModalEvaluator
    cfg.eval.candidate_backend = "baseline"
    evaluator = ModalEvaluator(repo_root, cfg.eval)

    print("[baseline] Precomputing references...")
    evaluator.precompute_references(cfg.problems)

    for p in cfg.problems:
        print(f"[baseline] Evaluating final result for problem {p.problem_id}...")
        ref_result, cand_result = evaluator.evaluate_pair(p)

        status = cand_result.eval_feedback.get("status", "unknown")
        correctness = cand_result.eval_feedback.get("correctness", {})
        is_correct = correctness.get("all_ok", False)

        ref_agg = ref_result.summary_rank0.get("aggregate", {})
        cand_agg = cand_result.summary_rank0.get("aggregate", {})
        ref_ms = ref_agg.get("reference_mean_ms") or ref_agg.get("wall_time_ms", 0)
        cand_ms = cand_agg.get("candidate_mean_ms") or cand_agg.get("wall_time_ms", 0)
        speedup = (ref_ms / cand_ms) if (cand_ms > 0 and ref_ms > 0) else 0.0

        print(f"  Problem {p.problem_id}: status={status}, correct={is_correct}, "
              f"speedup={speedup:.2f}x (ref={ref_ms:.3f}ms, cand={cand_ms:.3f}ms)")

        if not is_correct:
            for r in cand_result.summary_rank0.get("ranks", []):
                if isinstance(r, dict) and r.get("error"):
                    print(f"    rank {r.get('rank', '?')}: {r['error']}")

        log({"event": "final_eval", "problem_id": p.problem_id,
             "status": status, "correct": is_correct, "speedup": speedup,
             "ref_ms": ref_ms, "cand_ms": cand_ms})

    elapsed = time.perf_counter() - start_time

    summary = {
        "name": f"baseline_{cfg.name}",
        "run_dir": str(run_dir),
        "elapsed_s": elapsed,
        "model": model,
        "provider": provider,
        "max_iterations": cfg.max_iterations,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"BASELINE COMPLETE in {elapsed:.1f}s")
    print(f"  Run dir: {run_dir}")
    print(f"  Log: {log_path}")
    print(f"  Snapshots: {snapshots_dir}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run naive baseline kernel optimization")
    parser.add_argument("--config", type=str, required=True, help="Experiment YAML config")
    parser.add_argument("--max-iters", type=int, default=None,
                        help="Override max iterations from config")
    args = parser.parse_args()
    run_baseline(args.config, args.max_iters)


if __name__ == "__main__":
    main()
