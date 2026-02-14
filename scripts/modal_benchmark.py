#!/usr/bin/env python3
"""
Run the candidate benchmark on Modal (8x H100) and print JSON metrics.

This exists so the agent loop can evaluate Triton/NVSHMEM candidates in a real GPU
environment (instead of a local CPU/Mac environment that lacks Triton/CUDA).

Typical usage (from repo root):
  modal run scripts/modal_benchmark.py \
    --problem 1 \
    --candidate /Users/rohk/FERB/.agent_runs/<run_id>/candidate_iter_1.py \
    --rows 1024 --cols 1024 --dtype float32 --warmup 3 --iters 10
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import modal


APP_NAME = "ferb-agent-eval"
REMOTE_ROOT = "/workspace"

app = modal.App(APP_NAME)

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("wget", "xz-utils", "gnupg", "software-properties-common", "git")
    # Install NVSHMEM from tarball (avoids version mismatch issues with apt package)
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.2.5_cuda12-archive.tar.xz -O /tmp/nvshmem.tar.xz",
        "mkdir -p /opt/nvshmem",
        "tar -xf /tmp/nvshmem.tar.xz -C /opt/nvshmem --strip-components=1",
        "rm /tmp/nvshmem.tar.xz",
    )
    .env(
        {
            "NVSHMEM_HOME": "/opt/nvshmem",
            "LD_LIBRARY_PATH": "/opt/nvshmem/lib:/usr/local/cuda/lib64",
            "CUDA_HOME": "/usr/local/cuda",
            "PATH": "/usr/local/cuda/bin:${PATH}",
        }
    )
    .pip_install(
        "torch",
        "triton",
        "numpy",
        "mpi4py",
        "nvshmem4py-cu12",
        "cuda-python>=12.0",
        "numba",
        "numba-cuda",
        "cffi",
    )
)

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
image_with_code = image.add_local_dir(project_dir, remote_path=REMOTE_ROOT, copy=True)


@app.function(image=image_with_code, gpu="H100:8", timeout=60 * 30)
def run_modal_benchmark(
    *,
    problem: int,
    candidate_relpath: str,
    rows: int,
    cols: int,
    dtype: str,
    warmup: int,
    iters: int,
) -> dict:
    import subprocess

    candidate_path = f"{REMOTE_ROOT}/{candidate_relpath.lstrip('/')}"
    cmd = [
        "torchrun",
        "--nproc-per-node",
        "8",
        f"{REMOTE_ROOT}/scripts/benchmark_candidate.py",
        "--problem",
        str(problem),
        "--candidate",
        candidate_path,
        "--rows",
        str(rows),
        "--cols",
        str(cols),
        "--dtype",
        dtype,
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()

    # Try to parse JSON metrics from stdout (benchmark prints JSON).
    metrics: dict = {"score": 0.0}
    try:
        metrics = json.loads(stdout.splitlines()[-1]) if stdout else {"score": 0.0}
        if not isinstance(metrics, dict):
            metrics = {"score": 0.0}
    except Exception:
        metrics = {"score": 0.0}

    metrics["modal_returncode"] = proc.returncode
    if stderr:
        metrics["modal_stderr"] = stderr
    return metrics


@app.local_entrypoint()
def main(
    problem: int = 1,
    candidate: str = "",
    rows: int = 1024,
    cols: int = 1024,
    dtype: str = "float32",
    warmup: int = 3,
    iters: int = 10,
) -> None:
    if not candidate:
        raise SystemExit("--candidate is required")

    repo_root = Path(project_dir).resolve()
    cand_path = Path(candidate).expanduser().resolve()
    try:
        rel = cand_path.relative_to(repo_root)
    except Exception:
        raise SystemExit(f"candidate must be within repo: {repo_root} (got {cand_path})")

    metrics = run_modal_benchmark.remote(
        problem=problem,
        candidate_relpath=str(rel),
        rows=rows,
        cols=cols,
        dtype=dtype,
        warmup=warmup,
        iters=iters,
    )
    print(json.dumps(metrics))

