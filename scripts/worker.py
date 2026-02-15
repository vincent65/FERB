#!/usr/bin/env python3
"""
Distributed evaluation worker launched by torchrun on Modal.

This worker:
- initializes backend runtimes (`reference`, `triton`, `agent`, `numba_cuda`)
- loads the problem module dynamically
- creates standardized inputs via utils/input_output_tensors.py
- runs correctness + timing + optional profiler trace
- writes per-rank metrics and a rank0 summary JSON
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

# Ensure project root is importable when launched as /workspace/scripts/worker.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.init_and_finalize_backends import (
    finalize_numba_cuda,
    finalize_reference,
    finalize_triton,
    init_numba_cuda,
    init_reference,
    init_triton,
)
from utils.input_output_tensors import (
    create_input_tensor,
    save_performance_metrics,
    save_tensor,
)


def _load_module_from_path(module_path: str, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import module from {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _clone_arg(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, tuple):
        return tuple(_clone_arg(v) for v in x)
    if isinstance(x, list):
        return [_clone_arg(v) for v in x]
    if isinstance(x, dict):
        return {k: _clone_arg(v) for k, v in x.items()}
    return x


def _resolve_backend_mode(backend: str) -> str:
    # `agent` solutions are expected to be Triton/NVSHMEM-based by default.
    if backend == "reference":
        return "reference"
    if backend in {"triton", "agent"}:
        return "triton"
    if backend == "numba_cuda":
        return "numba_cuda"
    raise ValueError(f"Unsupported backend: {backend}")


def _init_backend(backend_mode: str, rank: int, world_size: int) -> None:
    if backend_mode == "reference":
        init_reference(rank, world_size)
    elif backend_mode == "triton":
        init_reference(rank, world_size)
        init_triton(rank, world_size)
    elif backend_mode == "numba_cuda":
        init_reference(rank, world_size)
        init_numba_cuda(rank, world_size)
    else:
        raise ValueError(f"Unsupported backend mode: {backend_mode}")


def _finalize_backend(backend_mode: str) -> None:
    if backend_mode == "reference":
        finalize_reference()
    elif backend_mode == "triton":
        finalize_triton()
    elif backend_mode == "numba_cuda":
        finalize_numba_cuda()


def _compare_outputs(
    out_ref: Any,
    out_candidate: Any,
    rtol: float,
    atol: float,
) -> None:
    if isinstance(out_ref, torch.Tensor):
        if not isinstance(out_candidate, torch.Tensor):
            raise AssertionError("Reference output is tensor but candidate is not")
        torch.testing.assert_close(out_candidate, out_ref, rtol=rtol, atol=atol)
        return

    if isinstance(out_ref, (list, tuple)):
        if type(out_ref) is not type(out_candidate):
            raise AssertionError("Output container type mismatch")
        if len(out_ref) != len(out_candidate):
            raise AssertionError("Output container length mismatch")
        for a, b in zip(out_ref, out_candidate):
            _compare_outputs(a, b, rtol=rtol, atol=atol)
        return

    if isinstance(out_ref, dict):
        if not isinstance(out_candidate, dict):
            raise AssertionError("Reference output is dict but candidate is not")
        if set(out_ref.keys()) != set(out_candidate.keys()):
            raise AssertionError("Output dict keys mismatch")
        for key in out_ref:
            _compare_outputs(out_ref[key], out_candidate[key], rtol=rtol, atol=atol)
        return

    if out_ref != out_candidate:
        raise AssertionError(f"Non-tensor output mismatch: {out_candidate} != {out_ref}")


def _to_cpu_recursive(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, tuple):
        return tuple(_to_cpu_recursive(v) for v in x)
    if isinstance(x, list):
        return [_to_cpu_recursive(v) for v in x]
    if isinstance(x, dict):
        return {k: _to_cpu_recursive(v) for k, v in x.items()}
    return x


def _load_saved_output(outputs_dir: str, rank: int) -> Any:
    path = os.path.join(outputs_dir, f"rank_{rank}.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing cached reference output for rank {rank}: {path}")
    return torch.load(path, map_location="cpu")


def _measure_latency_ms(
    fn: Callable[..., Any],
    inputs: tuple[Any, ...],
    warmup_iters: int,
    measure_iters: int,
) -> dict[str, Any]:
    for _ in range(warmup_iters):
        _ = fn(*inputs)
        torch.cuda.synchronize()
        dist.barrier()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(measure_iters)]
    times_ms: list[float] = []

    for i in range(measure_iters):
        dist.barrier()
        start_events[i].record()
        _ = fn(*inputs)
        end_events[i].record()
        torch.cuda.synchronize()
        dist.barrier()

    for i in range(measure_iters):
        times_ms.append(float(start_events[i].elapsed_time(end_events[i])))

    times = torch.tensor(times_ms, dtype=torch.float32)
    return {
        "wall_time_ms": float(times.mean().item()),
        "wall_time_std_ms": float(times.std(unbiased=False).item()),
        "min_time_ms": float(times.min().item()),
        "max_time_ms": float(times.max().item()),
        "iterations": measure_iters,
    }


def _profile_once(
    fn: Callable[..., Any],
    inputs: tuple[Any, ...],
    trace_path: str,
    active_iters: int = 3,
) -> None:
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    schedule = torch.profiler.schedule(wait=0, warmup=1, active=max(1, active_iters - 1), repeat=1)
    with torch.profiler.profile(activities=activities, schedule=schedule, record_shapes=True) as prof:
        for _ in range(active_iters):
            _ = fn(*inputs)
            torch.cuda.synchronize()
            prof.step()
    prof.export_chrome_trace(trace_path)


def _get_rank_world() -> tuple[int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, world_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Distributed worker for kernel evaluation")
    parser.add_argument("--backend", type=str, required=True)
    parser.add_argument("--problem_py", type=str, required=True)
    parser.add_argument("--logs_dir", type=str, required=True)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--problem_id", type=int, required=True)
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--measure_iters", type=int, default=10)
    parser.add_argument("--profile", action="store_true", default=True)
    parser.add_argument("--no-profile", action="store_false", dest="profile")
    parser.add_argument("--save_outputs", action="store_true", default=True)
    parser.add_argument("--no-save_outputs", action="store_false", dest="save_outputs")
    parser.add_argument("--reference_outputs_dir", type=str, default="")
    parser.add_argument("--skip_reference_timing", action="store_true", default=False)
    args = parser.parse_args()

    os.makedirs(args.logs_dir, exist_ok=True)
    rank, world_size = _get_rank_world()
    backend_mode = _resolve_backend_mode(args.backend)

    result_payload: dict[str, Any] = {
        "rank": rank,
        "world_size": world_size,
        "problem_id": args.problem_id,
        "backend": args.backend,
        "backend_mode": backend_mode,
        "status": "ok",
    }

    try:
        _init_backend(backend_mode, rank, world_size)

        ref_module = _load_module_from_path(
            str(Path("/workspace/reference") / f"{args.problem_id}.py"),
            f"reference_problem_{args.problem_id}",
        )
        cand_module = _load_module_from_path(args.problem_py, f"candidate_problem_{args.problem_id}_{args.backend}")

        ref_solution = getattr(ref_module, "solution", None)
        cand_solution = getattr(cand_module, "solution", None)
        if ref_solution is None or cand_solution is None:
            raise RuntimeError("Both reference and candidate modules must define `solution`")

        dtype = getattr(torch, args.dtype)
        base_inputs = create_input_tensor(
            rank=rank,
            world_size=world_size,
            problem_id=args.problem_id,
            base_shape=(args.rows, args.cols),
            dtype=dtype,
            device=torch.device("cuda", rank),
        )
        if not isinstance(base_inputs, tuple):
            raise RuntimeError("create_input_tensor must return a tuple")

        ref_inputs = _clone_arg(base_inputs)
        cand_inputs = _clone_arg(base_inputs)

        use_cached_reference_outputs = bool(args.reference_outputs_dir)
        if use_cached_reference_outputs:
            out_ref = _load_saved_output(args.reference_outputs_dir, rank)
            out_cand = cand_solution(*cand_inputs)
        else:
            out_ref = ref_solution(*ref_inputs)
            out_cand = cand_solution(*cand_inputs)

        should_check = True
        if hasattr(ref_module, "output_is_valid"):
            should_check = bool(ref_module.output_is_valid(rank))

        if hasattr(ref_module, "normalize_output"):
            if not use_cached_reference_outputs:
                out_ref = ref_module.normalize_output(out_ref, rank)
            out_cand = ref_module.normalize_output(out_cand, rank)
            should_check = should_check and (out_ref is not None and out_cand is not None)

        tolerances = {"rtol": 1e-3, "atol": 1e-3}
        if hasattr(ref_module, "get_tolerances"):
            custom_tol = ref_module.get_tolerances()
            if isinstance(custom_tol, dict):
                tolerances["rtol"] = float(custom_tol.get("rtol", tolerances["rtol"]))
                tolerances["atol"] = float(custom_tol.get("atol", tolerances["atol"]))

        correctness_ok = True
        correctness_note = "checked"
        if should_check:
            if use_cached_reference_outputs:
                out_cand = _to_cpu_recursive(out_cand)
            _compare_outputs(out_ref, out_cand, rtol=tolerances["rtol"], atol=tolerances["atol"])
        else:
            correctness_note = "skipped_by_problem_hook"

        if args.save_outputs:
            save_tensor(out_ref, os.path.join(args.logs_dir, "reference_outputs"), rank)
            save_tensor(out_cand, os.path.join(args.logs_dir, "candidate_outputs"), rank)

        ref_metrics = None
        if not args.skip_reference_timing:
            ref_metrics = _measure_latency_ms(
                ref_solution,
                _clone_arg(base_inputs),
                args.warmup_iters,
                args.measure_iters,
            )
        cand_metrics = _measure_latency_ms(cand_solution, _clone_arg(base_inputs), args.warmup_iters, args.measure_iters)

        speedup_vs_ref = None
        if ref_metrics is not None:
            speedup_vs_ref = (
                ref_metrics["wall_time_ms"] / cand_metrics["wall_time_ms"]
                if cand_metrics["wall_time_ms"] > 0
                else 0.0
            )

        result_payload.update(
            {
                "correctness_ok": correctness_ok,
                "correctness_note": correctness_note,
                "tolerances": tolerances,
                "reference": ref_metrics,
                "candidate": cand_metrics,
                "speedup_vs_ref": float(speedup_vs_ref) if speedup_vs_ref is not None else None,
                "reference_timing_skipped": bool(args.skip_reference_timing),
                "used_cached_reference_outputs": use_cached_reference_outputs,
            }
        )

        if args.profile:
            trace_cand = os.path.join(args.logs_dir, f"trace_candidate_rank{rank}.json")
            _profile_once(cand_solution, _clone_arg(base_inputs), trace_cand)
            result_payload["trace_candidate"] = trace_cand
            # Only profile the reference if we aren't skipping reference timing
            # (i.e., this is a reference run or a combined run, not a cached-reference candidate eval).
            if not args.skip_reference_timing:
                trace_ref = os.path.join(args.logs_dir, f"trace_reference_rank{rank}.json")
                _profile_once(ref_solution, _clone_arg(base_inputs), trace_ref)
                result_payload["trace_reference"] = trace_ref

    except Exception as exc:  # pylint: disable=broad-except
        result_payload["status"] = "error"
        result_payload["error"] = str(exc)
        result_payload["traceback"] = traceback.format_exc()
    finally:
        # Persist per-rank metrics regardless of success.
        save_performance_metrics(result_payload, args.logs_dir, rank)

        # Gather rank payloads and write rank0 summary.
        if dist.is_initialized():
            gathered: list[Any] = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered, result_payload)
            if rank == 0:
                candidate_times = [r["candidate"]["wall_time_ms"] for r in gathered if r and r.get("status") == "ok"]
                reference_times = [
                    r["reference"]["wall_time_ms"]
                    for r in gathered
                    if r
                    and r.get("status") == "ok"
                    and isinstance(r.get("reference"), dict)
                    and r["reference"].get("wall_time_ms") is not None
                ]
                summary = {
                    "problem_id": args.problem_id,
                    "backend": args.backend,
                    "status": "ok" if all((r or {}).get("status") == "ok" for r in gathered) else "error",
                    "n_ranks": len(gathered),
                    "ranks": gathered,
                    "aggregate": {},
                }
                if candidate_times and reference_times:
                    cand_t = torch.tensor(candidate_times, dtype=torch.float32)
                    ref_t = torch.tensor(reference_times, dtype=torch.float32)
                    cand_mean = float(cand_t.mean().item())
                    ref_mean = float(ref_t.mean().item())
                    summary["aggregate"] = {
                        "candidate_mean_ms": cand_mean,
                        "candidate_p95_ms": float(torch.quantile(cand_t, 0.95).item()),
                        "reference_mean_ms": ref_mean,
                        "reference_p95_ms": float(torch.quantile(ref_t, 0.95).item()),
                        "speedup_vs_ref_mean": (ref_mean / cand_mean) if cand_mean > 0 else 0.0,
                    }
                elif candidate_times:
                    cand_t = torch.tensor(candidate_times, dtype=torch.float32)
                    summary["aggregate"] = {
                        "candidate_mean_ms": float(cand_t.mean().item()),
                        "candidate_p95_ms": float(torch.quantile(cand_t, 0.95).item()),
                        "reference_mean_ms": None,
                        "reference_p95_ms": None,
                        "speedup_vs_ref_mean": None,
                    }
                with open(os.path.join(args.logs_dir, "summary_rank0.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2)

        try:
            _finalize_backend(backend_mode)
        except Exception:
            pass


if __name__ == "__main__":
    main()
