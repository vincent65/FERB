#!/usr/bin/env python3
"""
Benchmark reference vs candidate solution under torchrun.

Outputs:
- JSON summary (default)
- `score=<speedup>` for evaluator mode (--score-only)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Ensure repo root is importable when launched by torch.distributed.run.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist

from utils.init_and_finalize_backends import finalize_reference
from utils.init_and_finalize_backends import init_reference
from utils.input_output_tensors import create_input_tensor


def _dtype_from_string(name: str) -> torch.dtype:
    lookup = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int32": torch.int32,
        "int64": torch.int64,
    }
    key = (name or "").strip().lower()
    if key not in lookup:
        raise ValueError(f"Unsupported dtype: {name}")
    return lookup[key]


def _load_solution(path_str: str):
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Solution file not found: {path_str}")
    mod_name = f"ferb_bench_{path.stem}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import: {path_str}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "solution"):
        raise AttributeError(f"{path_str} has no solution(...)")
    return getattr(module, "solution")


def _clone_obj(obj: Any) -> Any:
    if isinstance(obj, torch.Tensor):
        return obj.clone()
    if isinstance(obj, tuple):
        return tuple(_clone_obj(x) for x in obj)
    if isinstance(obj, list):
        return [_clone_obj(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _clone_obj(v) for k, v in obj.items()}
    return obj


def _compare_outputs(a: Any, b: Any, atol: float, rtol: float) -> tuple[bool, float]:
    # Returns (allclose, max_abs_diff)
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        ac = torch.allclose(a, b, atol=atol, rtol=rtol)
        max_diff = float((a - b).abs().max().item()) if a.numel() > 0 else 0.0
        return ac, max_diff
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        oks = []
        diffs = []
        for x, y in zip(a, b):
            ok, d = _compare_outputs(x, y, atol, rtol)
            oks.append(ok)
            diffs.append(d)
        return all(oks), (max(diffs) if diffs else 0.0)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        oks = []
        diffs = []
        for x, y in zip(a, b):
            ok, d = _compare_outputs(x, y, atol, rtol)
            oks.append(ok)
            diffs.append(d)
        return all(oks), (max(diffs) if diffs else 0.0)
    if isinstance(a, dict) and isinstance(b, dict) and set(a.keys()) == set(b.keys()):
        oks = []
        diffs = []
        for k in sorted(a.keys()):
            ok, d = _compare_outputs(a[k], b[k], atol, rtol)
            oks.append(ok)
            diffs.append(d)
        return all(oks), (max(diffs) if diffs else 0.0)
    return False, float("inf")


def _tensor_sample(obj: Any, n: int = 8):
    if isinstance(obj, torch.Tensor):
        flat = obj.detach().flatten()
        k = min(n, flat.numel())
        return flat[:k].tolist()
    if isinstance(obj, tuple):
        return [_tensor_sample(x, n) for x in obj]
    if isinstance(obj, list):
        return [_tensor_sample(x, n) for x in obj]
    if isinstance(obj, dict):
        return {k: _tensor_sample(v, n) for k, v in obj.items()}
    return str(type(obj))


def _run_timed(fn, inputs: tuple[Any, ...], warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        _ = fn(*_clone_obj(inputs))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

    times_ms: list[float] = []
    for _ in range(iters):
        if dist.is_initialized():
            dist.barrier()
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _ = fn(*_clone_obj(inputs))
            end.record()
            torch.cuda.synchronize()
            times_ms.append(float(start.elapsed_time(end)))
        else:
            t0 = time.perf_counter()
            _ = fn(*_clone_obj(inputs))
            t1 = time.perf_counter()
            times_ms.append((t1 - t0) * 1000.0)
        if dist.is_initialized():
            dist.barrier()
    return times_ms


def _global_max_time(local_ms: float, device: torch.device) -> float:
    t = torch.tensor([local_ms], device=device, dtype=torch.float32)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark FERB candidate speedup")
    parser.add_argument("--problem", type=int, required=True)
    parser.add_argument("--candidate", type=str, required=True)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    init_reference(rank, world_size)
    try:
        device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
        dtype = _dtype_from_string(args.dtype)
        base_shape = (args.rows, args.cols)
        inputs = create_input_tensor(
            rank=rank,
            world_size=world_size,
            problem_id=args.problem,
            base_shape=base_shape,
            dtype=dtype,
            device=device,
        )

        ref_path = Path(__file__).resolve().parent.parent / "reference" / f"{args.problem}.py"
        ref_solution = _load_solution(str(ref_path))
        cand_solution = _load_solution(args.candidate)

        # Correctness pass
        with torch.no_grad():
            ref_out = ref_solution(*_clone_obj(inputs))
            cand_out = cand_solution(*_clone_obj(inputs))
        ok_local, max_diff_local = _compare_outputs(ref_out, cand_out, atol=args.atol, rtol=args.rtol)
        ok_tensor = torch.tensor([1 if ok_local else 0], device=device, dtype=torch.int32)
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        all_ok = bool(ok_tensor.item() == 1)

        diff_tensor = torch.tensor([max_diff_local], device=device, dtype=torch.float32)
        dist.all_reduce(diff_tensor, op=dist.ReduceOp.MAX)
        max_abs_diff = float(diff_tensor.item())

        # Timing
        ref_times = _run_timed(ref_solution, inputs, warmup=args.warmup, iters=args.iters)
        cand_times = _run_timed(cand_solution, inputs, warmup=args.warmup, iters=args.iters)
        ref_local_mean = statistics.mean(ref_times)
        cand_local_mean = statistics.mean(cand_times)
        ref_global_ms = _global_max_time(ref_local_mean, device)
        cand_global_ms = _global_max_time(cand_local_mean, device)

        speedup = (ref_global_ms / cand_global_ms) if cand_global_ms > 0 else 0.0
        score = speedup if all_ok else 0.0

        if rank == 0:
            payload = {
                "problem": args.problem,
                "candidate": args.candidate,
                "shape": [args.rows, args.cols],
                "dtype": args.dtype,
                "world_size": world_size,
                "allclose": all_ok,
                "max_abs_diff": max_abs_diff,
                "reference_ms": ref_global_ms,
                "candidate_ms": cand_global_ms,
                "speedup": speedup,
                "score": score,
                "reference_output_sample": _tensor_sample(ref_out, 8),
                "candidate_output_sample": _tensor_sample(cand_out, 8),
            }
            if args.score_only:
                print(f"score={score:.6f}", flush=True)
            else:
                print(json.dumps(payload), flush=True)
        return 0
    except Exception as exc:
        if rank == 0:
            msg = {"score": 0.0, "error": str(exc)}
            if args.score_only:
                print("score=0.0", flush=True)
            else:
                print(json.dumps(msg), flush=True)
        return 1
    finally:
        finalize_reference()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ModuleNotFoundError as exc:
        # Provide a clean machine-parseable error for agent loops.
        print(json.dumps({"score": 0.0, "error": str(exc)}), flush=True)
        raise SystemExit(1)
    except Exception as exc:
        print(json.dumps({"score": 0.0, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        raise SystemExit(1)
