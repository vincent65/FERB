#!/usr/bin/env python3
import argparse
import json
import os
import sys
from typing import Optional


"""
The prompt will look something like:

    1) {BASE PROMPT}
    2) Description of the problem/what you want it to do (task and problem statement)
    3) Function Signature to fill in
    4) Reference implemnentation (torch distributed)
    5) SYSTEM TOPOLOGY: extracted from NCCL automatically
"""


base_prompt = """
You are an expert multi-GPU systems and CUDA engineer. Given a problem, a reference implementation, and a GPU topology, write the FASTEST correct multi-GPU solution.

You are tasked with developing CUDA kernel(s) that use NVIDIA's NVSHMEM library.

NVSHMEM quick context (assumptions and APIs you can use):
- Rank setup is already done for you (torch.distributed initialized, PE mapping established, and NVSHMEM host init handled externally). Do NOT call nvshmem_init/nvshmem_finalize.
- You can include and use NVSHMEM from both host and device code:
  - Host includes:  #include <nvshmem.h>, #include <nvshmemx.h>
  - Device-side queries:  nvshmem_my_pe(), nvshmem_n_pes() return the calling PE id and world size inside kernels
  - Device-side one-sided ops (examples):  nvshmem_float_p, nvshmem_float_g, nvshmem_float_atomic_add
  - Device-side collectives (when appropriate):  nvshmemx_* device variants on supported types (e.g., reduce, broadcast)
- Symmetric memory pointers you are given are valid for NVSHMEM one-sided/device operations.
- Prefer NVLink/NVSwitch paths implicitly (NVSHMEM/transport chooses routes); still minimize communication volume and overlap comm/compute.

Minimal example that you should seek to emulate (headers are already included; you may directly use NVSHMEM device APIs):
```cuda
// NOTE: Do NOT include headers here; they are already included by the build wrapper.
// You may define helper __device__ functions if needed, but define exactly ONE __global__ kernel
// and launch ONLY that single kernel from run_solution.

__global__ void demo_nvshmem_kernel(float* buf, size_t n) {
  int me = nvshmem_my_pe();            // my PE (rank)
  int np = nvshmem_n_pes();            // world size
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    int pe_next = (me + 1) % np;       // example target PE
    // Write my value to neighbor's symmetric buffer
    nvshmem_float_p(buf + idx, buf[idx], pe_next);
  }
}

extern "C" void run_solution(float* buf, size_t numel, cudaStream_t stream) {
  dim3 block(256);
  dim3 grid((numel + block.x - 1) / block.x);
  // Launch exactly ONE kernel here
  demo_nvshmem_kernel<<<grid, block, 0, stream>>>(buf, numel);
}
```

Output format requirements (CRITICAL):
- Return ONLY the contents of a single compilable .cu source file (no markdown prose outside the code).
- Do NOT include any headers; they are already included by the build wrapper. You may directly call NVSHMEM device APIs.
- Define EXACTLY ONE __global__ kernel implementing the requested functionality using NVSHMEM device APIs.
- Expose a single host wrapper with the exact signature:
  extern "C" void run_solution(float* data, size_t numel, cudaStream_t stream);
  (This wrapper will be called from PyTorch in place of the reference code and must launch exactly one kernel on the provided CUDA stream.)
- Do NOT perform NVSHMEM init/finalize or torch.distributed setup inside the file.
- Avoid host-device synchronizations beyond what is strictly necessary; prefer stream-ordered semantics.
- Code must compile with: nvcc -std=c++17 -Xcompiler -fPIC -O3 -arch=sm_90

Ensure to respect the provided topology. Prefer high-bandwidth links (NVLink/NVSwitch) and minimize PCIe hops. Overlap communication with computation using CUDA streams and events where beneficial. Ensure numerical correctness and deterministic synchronization across ranks. Favor memory-efficient strategies when choices are equivalent performance-wise.
"""


problem_statement = """
Replace the following NCCL-based all-reduce (sum) reference with an NVSHMEM kernel implementation that performs the same collective entirely from GPU kernels (kernel-initiated communication), minimizing host synchronization and maximizing overlap. Your solution must:
- Produce identical numerical results to the reference (elementwise sum across all ranks)
- Work for arbitrary tensor shapes and dtypes provided at runtime (assume dense, contiguous Tensors)
- Respect the provided system topology (prefer NVLink/NVSwitch paths when relevant)
- Run one rank per GPU with torch.distributed already initialized
"""






def construct_prompt(problem_id: str = None, topology: str = None):
    """Construct the prompt. Can be called with parameters or via argparse."""
    if problem_id is None or topology is None:
        # Called as script - use argparse
        parser = argparse.ArgumentParser(description="Construct a master PKB prompt from components.")
        parser.add_argument("--problem_id", required=True, help="Numerical ID of the problem")
        parser.add_argument("--topology", required=True, help="Path to topology JSON (LLM-friendly JSON)")
        parser.add_argument("--out", help="Optional path to save the constructed prompt")
        args = parser.parse_args()
        problem_id = args.problem_id
        topology = args.topology
    else:
        # Called as function - use provided parameters
        args = None

    reference_impl_path = os.path.join(os.path.dirname(__file__), "../reference", f"{problem_id}.py")
    try:
        with open(reference_impl_path, "r") as _f:
            _ref_impl_text = _f.read()
    except Exception as _e:
        _ref_impl_text = f"# ERROR reading {reference_impl_path}: {_e}"

    system_topo_path = topology
    try:
        with open(system_topo_path, "r") as _f:
            _system_topo_obj = json.load(_f)
            _system_topo_text = json.dumps(_system_topo_obj, indent=2)
    except Exception as _e:
        _system_topo_text = f"# ERROR reading {system_topo_path}: {_e}"

    # Construct reference implementation string
    reference_implementation = f"""```python
{_ref_impl_text}
```"""

    # Construct system topology string
    system_topology = f"""The following SYSTEM TOPOLOGY JSON is provided (parsed from NCCL topology dump):
- Use `nvlink_peers`, `bidirectional_bandwidth_gb_s`, and `numa_node` to route and schedule communication.
- Prefer NVLink/NVSwitch paths over PCIe when orchestrating collectives.
- Overlap collectives with compute using separate CUDA streams.

```json
{_system_topo_text}
```"""

    # Assemble the final prompt
    prompt = f"""{base_prompt}

# PROBLEM STATEMENT
{problem_statement}

# REFERENCE IMPLEMENTATION
{reference_implementation}

# SYSTEM TOPOLOGY
{system_topology}
"""
    return prompt
    


if __name__ == "__main__":
    prompt = construct_prompt()
    # Print to stdout
    print(prompt)

    # # Optionally save
    # if args.out:
    #     with open(args.out, "w") as f:
    #         f.write(prompt)
    #     print(f"\n# Saved prompt to {os.path.abspath(args.out)}", file=sys.stderr)










# # Single process on SLURM with all 8 GPUs
# #   srun -N1 -n1 --gpus=8 python /home/willychan/vllm_test/delme.py
# # Or locally force visibility
# #   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /home/willychan/vllm_test/delme.py



# def dump_nccl_topology_xml(output_path: str = "nccl_topology.xml") -> str:
# 	"""Dump NCCL hardware topology XML to the given path and return the absolute path.
# 	Requires a working NCCL installation and at least one CUDA-visible GPU.
# 	"""
# 	import os
# 	os.environ["NCCL_TOPO_DUMP_FILE"] = os.path.abspath(output_path)
# 	# Try PyTorch (single-process NCCL init)
# 	try:
# 		import torch
# 		import torch.distributed as dist
# 		if not torch.cuda.is_available():
# 			raise RuntimeError("CUDA not available")
# 		os.environ.setdefault("RANK", "0")
# 		os.environ.setdefault("WORLD_SIZE", "1")
# 		os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
# 		os.environ.setdefault("MASTER_PORT", "29500")
# 		os.environ.setdefault("LOCAL_RANK", "0")
# 		# Support torchrun-style env and SLURM env
# 		local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_LOCALID", "0")))
# 		rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
# 		world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
# 		torch.cuda.set_device(local_rank)
# 		# Torch 2.4+ supports device binding via device_id. Fall back if not available.
# 		try:
# 			dist.init_process_group(
# 				backend="nccl",
# 				init_method="env://",
# 				rank=rank,
# 				world_size=world_size,
# 				device_id=local_rank,
# 			)
# 		except TypeError:
# 			# Older torch versions without device_id
# 			dist.init_process_group(
# 				backend="nccl",
# 				init_method="env://",
# 				rank=rank,
# 				world_size=world_size,
# 			)
# 		# Barrier with explicit device_ids when supported
# 		try:
# 			dist.barrier(device_ids=[local_rank])
# 		except TypeError:
# 			dist.barrier()
# 		dist.destroy_process_group()
# 	except Exception:
# 		# Fallback: use CuPy's NCCL bindings if available
# 		try:
# 			import cupy as cp  # noqa: F401
# 			import cupy.cuda.nccl as nccl
# 			uid = nccl.get_unique_id()
# 			comm = nccl.NcclCommunicator(1, uid, 0)
# 			del comm
# 		except Exception as e:
# 			raise RuntimeError(
# 				f"Failed to trigger NCCL topology dump; ensure NCCL/PyTorch or CuPy is installed. Error: {e}"
# 			)
# 	return os.environ["NCCL_TOPO_DUMP_FILE"]



# dump_nccl_topology_xml()

# reference_ring_allreduce = """
# ```python

# ```
# """