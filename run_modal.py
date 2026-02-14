#!/usr/bin/env python3
"""
Modal launcher for distributed kernel evaluation.

Usage:
    modal run run_modal.py                     # Run reference (default)
    modal run run_modal.py --solution triton   # Run triton solution
    modal run run_modal.py --problem 2         # Run problem 2

The script will:
1. Spin up 8 H100 GPUs on Modal
2. Run the distributed evaluation (8 processes, one per GPU)
3. Download the .pt/.json output files to your local logs/ directory
"""

import modal
import os
import shutil

# ---------------------------------------------------------------------------
# Modal App & Image Setup
# ---------------------------------------------------------------------------

app = modal.App("parallel-kernel-bench")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install(
        "wget", "xz-utils", "gnupg", "software-properties-common", "git",
    )
    # Install NVSHMEM from tarball (avoids version mismatch issues with apt package)
    .run_commands(
        "wget -q https://developer.download.nvidia.com/compute/nvshmem/redist/libnvshmem/linux-x86_64/libnvshmem-linux-x86_64-3.2.5_cuda12-archive.tar.xz -O /tmp/nvshmem.tar.xz",
        "mkdir -p /opt/nvshmem",
        "tar -xf /tmp/nvshmem.tar.xz -C /opt/nvshmem --strip-components=1",
        "rm /tmp/nvshmem.tar.xz",
    )
    # environment variables needed for using NVSHMEM
    .env({
        "NVSHMEM_HOME": "/opt/nvshmem",
        "LD_LIBRARY_PATH": "/opt/nvshmem/lib:/usr/local/cuda/lib64",
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:${PATH}",
    })
    .pip_install(
        "torch",
        "triton",
        "numpy",
        "mpi4py",
        "nvshmem4py-cu12",
        "cuda-python>=12.0",
        "numba",
        "numba-cuda",
        "cffi",     # needed for nvshmem.core.device.numba RMA bindings
    )
)

# Volume to persist logs across runs (optional, for debugging)
volume = modal.Volume.from_name("pkb-logs", create_if_missing=True)

# ---------------------------------------------------------------------------
# Modal Function
# ---------------------------------------------------------------------------

# Add local project files to the image
project_dir = os.path.dirname(os.path.abspath(__file__))
image_with_code = image.add_local_dir(project_dir, remote_path="/workspace", copy=True)

@app.function(
    image=image_with_code,
    gpu="H100:8",
    timeout=60 * 30,  # 30 minutes
    volumes={"/logs": volume},
)
def run_distributed_eval(
    problem_id: str = "1",
    solution_type: str = "reference",  # this is your backend. "reference" or "triton" etc.
    m: int = 1024,
    n: int = 1024,
    dtype: str = "float32",
    save_outputs: bool = True,
    use_cached_reference: bool = False,
) -> dict:
    """
    Run distributed evaluation on 8 H100 GPUs.

    Args:
        problem_id: Problem number (e.g., "1", "2")
        solution_type: "reference" uses reference/<id>.py, "triton" uses solutions_triton/<id>_triton.py
        m, n: Tensor dimensions
        dtype: Tensor dtype

    Returns:
        dict with paths to output files
    """
    import subprocess
    
    # Determine which Python file to run
    if solution_type == "reference":
        problem_py = f"/workspace/reference/{problem_id}.py"
    else:
        problem_py = f"/workspace/solutions_{solution_type}/{problem_id}_{solution_type}.py"
    
    # Output directory (on the Modal volume)
    logs_dir = f"/logs/problem_{problem_id}/{solution_type}"
    
    print(f"Running: {problem_py}")
    print(f"Output:  {logs_dir}")
    print(f"Shape:   ({m}, {n}), dtype={dtype}")
    print(f"GPUs:    {8}")
    print(f"Backend: {solution_type}")
    print("-" * 60)

    # Ensure logs directory exists
    os.makedirs(logs_dir, exist_ok=True)
    
    # Single worker script for all backends; backend selects init/solution path
    worker_script_path = "/workspace/scripts/worker.py"
    cmd = [
        "torchrun",
        "--nproc-per-node", "8",
        "--master-addr", "127.0.0.1",
        "--master-port", "29500",
        worker_script_path,
        "--backend", solution_type,
        "--problem_py", problem_py,
        "--logs_dir", logs_dir,
        "--rows", str(m),
        "--cols", str(n),
        "--dtype", dtype,
        "--problem_id", str(problem_id),
        "--save_outputs" if save_outputs else "--no-save_outputs",
    ]
    if use_cached_reference and solution_type != "reference":
        cached_ref_dir = f"/logs/problem_{problem_id}/reference/reference_outputs"
        if not os.path.isdir(cached_ref_dir):
            raise FileNotFoundError(
                f"Cached reference outputs not found at {cached_ref_dir}. "
                "Run the reference backend first."
            )
        cmd.extend(
            [
                "--reference_outputs_dir",
                cached_ref_dir,
                "--skip_reference_timing",
            ]
        )

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"Worker exited with code {result.returncode}")
    elif result.stderr:
        print("STDERR:", result.stderr)
    
    # Commit volume changes
    volume.commit()
    
    # List output files
    output_files = []
    if os.path.isdir(logs_dir):
        for root, _, files in os.walk(logs_dir):
            for fname in sorted(files):
                if fname.endswith(".pt") or fname.endswith(".json"):
                    output_files.append(os.path.join(root, fname))

    return {
        "problem_id": problem_id,
        "solution_type": solution_type,
        "logs_dir": logs_dir,
        "output_files": output_files,
    }


@app.function(
    image=image,
    volumes={"/logs": volume},
)
def download_logs(problem_id: str, solution_type: str) -> list[bytes]:
    """Download the .pt and .json files for a given problem/solution."""
    logs_dir = f"/logs/problem_{problem_id}/{solution_type}"
    results = []
    
    if not os.path.isdir(logs_dir):
        print(f"No logs found at {logs_dir}")
        return results
    
    for root, _, files in os.walk(logs_dir):
        for fname in sorted(files):
            if fname.endswith(".pt") or fname.endswith(".json"):
                path = os.path.join(root, fname)
                rel = os.path.relpath(path, logs_dir)
                with open(path, "rb") as f:
                    results.append((rel, f.read()))
    
    return results


# ---------------------------------------------------------------------------
# Local Entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    problem: str = "1",
    solution: str = "reference",
    m: int = 1024,                  # TODO: m and n are how you currently sweep dimensions, want to refactor later
    n: int = 1024,
    dtype: str = "float32",
    download: bool = True,
    save_outputs: bool = True,
    use_cached_reference: bool = False,
):
    """
    Run distributed kernel evaluation on Modal.

    Examples:
        modal run run_modal.py                          # Run reference for problem 1
        modal run run_modal.py --problem 2              # Run reference for problem 2
        modal run run_modal.py --solution triton       # Run triton solution
    """
    print(f"Launching distributed eval on Modal...")
    print(f"  Problem:  {problem}")
    print(f"  Solution: {solution}")
    print(f"  Shape:    ({m}, {n})")
    print(f"  Dtype:    {dtype}")
    print()

    # Run the evaluation
    result = run_distributed_eval.remote(
        problem_id=problem,
        solution_type=solution,
        m=m,
        n=n,
        dtype=dtype,
        save_outputs=save_outputs,
        use_cached_reference=use_cached_reference,
    )
    
    print()
    print("=" * 60)
    print("RESULT:")
    print(f"  Problem ID:    {result['problem_id']}")
    print(f"  Solution Type: {result['solution_type']}")
    print(f"  Logs Dir:      {result['logs_dir']}")
    print(f"  Output Files:  {len(result['output_files'])} files")
    for f in result['output_files']:
        print(f"    - {f}")
    print("=" * 60)
    
    # Download files to local logs/ directory
    if download:
        print()
        print("Downloading .pt and .json files to local logs/ directory...")
        
        local_logs_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "logs",
            f"problem_{problem}",
            solution,
        )
        os.makedirs(local_logs_dir, exist_ok=True)
        
        files = download_logs.remote(problem, solution)
        for fname, data in files:
            local_path = os.path.join(local_logs_dir, fname)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            with open(local_path, "wb") as f:
                f.write(data)
            print(f"  Downloaded: {local_path}")
        
        print()
        print(f"All files downloaded to: {local_logs_dir}")

