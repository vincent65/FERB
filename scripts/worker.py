#!/usr/bin/env python3
"""
Distributed worker for FERB runs (used by run_modal.py).

Per rank:
1) Initialize distributed backend
2) Load problem solution module
3) Create input tensor(s)
4) Run solution
5) Save outputs
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

# Ensure repo root is importable under torchrun.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist

from utils.init_and_finalize_backends import finalize_numba_cuda
from utils.init_and_finalize_backends import finalize_reference
from utils.init_and_finalize_backends import finalize_triton
from utils.init_and_finalize_backends import init_numba_cuda
from utils.init_and_finalize_backends import init_reference
from utils.init_and_finalize_backends import init_triton
from utils.input_output_tensors import create_input_tensor
from utils.input_output_tensors import save_tensor


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


def _load_solution_module(problem_py: str):
    path = Path(problem_py)
    if not path.exists():
        raise FileNotFoundError(f"Problem file not found: {problem_py}")

    mod_name = f"ferb_solution_{path.stem}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {problem_py}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "solution"):
        raise AttributeError(f"{problem_py} does not define solution(...)")
    return module


def _init_backend(backend: str, rank: int, world_size: int) -> None:
    backend = backend.strip().lower()
    # Always initialize torch.distributed first.
    init_reference(rank, world_size)
    if backend == "reference":
        return
    if backend == "triton":
        init_triton(rank, world_size)
        return
    if backend == "numba_cuda":
        init_numba_cuda(rank, world_size)
        return
    # Unknown backend: keep NCCL-only path.


def _finalize_backend(backend: str) -> None:
    backend = backend.strip().lower()
    if backend == "triton":
        finalize_triton()
        return
    if backend == "numba_cuda":
        finalize_numba_cuda()
        return
    finalize_reference()


def _save_rank_metadata(logs_dir: str, rank: int, payload: dict[str, Any]) -> None:
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, f"rank_{rank}_meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(description="FERB distributed worker")
    parser.add_argument("--backend", required=True, help="reference|triton|numba_cuda")
    parser.add_argument("--problem_py", required=True, help="Absolute path to problem file")
    parser.add_argument("--logs_dir", required=True, help="Output logs directory")
    parser.add_argument("--rows", required=True, type=int)
    parser.add_argument("--cols", required=True, type=int)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--problem_id", required=True, type=int)
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    status = "ok"
    error_text = ""
    output_path = ""
    try:
        _init_backend(args.backend, rank, world_size)

        module = _load_solution_module(args.problem_py)
        solution_fn = getattr(module, "solution")

        dtype = _dtype_from_string(args.dtype)
        base_shape = (args.rows, args.cols)
        inputs = create_input_tensor(
            rank=rank,
            world_size=world_size,
            problem_id=args.problem_id,
            base_shape=base_shape,
            dtype=dtype,
        )

        with torch.no_grad():
            output = solution_fn(*inputs)
        output_path = save_tensor(output, args.logs_dir, rank)

        if dist.is_initialized():
            dist.barrier()
    except Exception:
        status = "error"
        error_text = traceback.format_exc()
        print(error_text, file=sys.stderr, flush=True)
    finally:
        _save_rank_metadata(
            args.logs_dir,
            rank,
            {
                "status": status,
                "rank": rank,
                "world_size": world_size,
                "backend": args.backend,
                "problem_id": args.problem_id,
                "problem_py": args.problem_py,
                "dtype": args.dtype,
                "shape": [args.rows, args.cols],
                "output_path": output_path,
                "error": error_text,
            },
        )
        try:
            _finalize_backend(args.backend)
        except Exception:
            traceback.print_exc()

    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
