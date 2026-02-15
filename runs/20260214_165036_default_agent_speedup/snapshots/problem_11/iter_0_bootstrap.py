import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem


@triton.jit
def _copy_to_peer_kernel(
    remote_ptr,       # Pointer to the remote symmetric buffer (full buffer view)
    input_ptr,        # Pointer to local input flattened buffer
    dest_offset,      # Offset (in elements) inside the remote buffer where we write
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load from local input
    vals = tl.load(input_ptr + offsets, mask=mask)

    # Store into remote buffer at remote_ptr + dest_offset + offsets
    tl.store(remote_ptr + dest_offset + offsets, vals, mask=mask)


class PyTorchStreamWrapper:
    """Wrap a torch.cuda.Stream for nvshmem barrier calls."""
    def __init__(self, pt_stream: torch.cuda.Stream):
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        return (0, self.pt_stream.cuda_stream)


@torch.no_grad()
def solution(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Distributed GEMM with All-Scatter using NVSHMEM + Triton.

    Each PE (rank) holds a column-shard of B with shape [K, N_local].
    Algorithm:
      1. Compute local GEMM: C_local = A @ B_local  -> shape [M, N_local]
      2. Allocate symmetric buffer `output_sym` sized world_size * M * N_local
      3. Write local shard into own segment of `output_sym`
      4. For each remote PE, copy local shard into the remote PE's segment
         at offset = rank * (M*N_local) using a Triton kernel writing to the
         peer view returned by nvshmem.get_peer_tensor
      5. Barrier and return the gathered full matrix reshaped to [M, world_size*N_local]

    Preconditions:
      - nvshmem must be initialized by the environment
      - Inputs must be CUDA tensors on the current device
      - A: [M, K], B: [K, N_local]
    """
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    assert A.ndim == 2 and B.ndim == 2, "A and B must be 2D matrices"

    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    # Shapes
    M, K = A.shape
    K_B, N_local = B.shape
    assert K == K_B, f"A and B must have matching K dimension: {K} != {K_B}"

    # Ensure contiguous
    A = A.contiguous()
    B = B.contiguous()

    # Step 1: local GEMM
    # Use PyTorch's matmul for best perf on single GPU
    C_local = torch.matmul(A, B)  # [M, N_local]
    C_local = C_local.contiguous()

    # Flatten local shard
    n_elems_per_rank = C_local.numel()  # M * N_local
    total_elems = n_pes * n_elems_per_rank

    # NVSHMEM symmetric allocation for gathered output
    # dtype must match C_local.dtype
    output_sym = nvshmem.tensor((total_elems,), dtype=C_local.dtype)

    # Flattened view of local shard
    local_flat = C_local.view(-1)

    # Copy local shard into our own segment of the symmetric buffer
    my_offset = my_pe * n_elems_per_rank
    output_sym[my_offset: my_offset + n_elems_per_rank] = local_flat

    # Prepare stream wrapper and synchronize
    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)

    # Ensure local writes are visible
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Prepare Triton kernel launch parameters
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elems_per_rank, BLOCK_SIZE),)

    # For each peer, get a peer view of the symmetric buffer and copy our local shard
    for dest_pe in range(n_pes):
        if dest_pe == my_pe:
            continue

        # Get peer view of the full symmetric buffer on dest_pe
        dest_view = nvshmem.get_peer_tensor(output_sym, dest_pe)

        # Destination offset inside remote buffer is our rank's segment
        dest_offset = my_offset  # in elements

        # Launch Triton kernel to copy local_flat -> dest_view[dest_offset:dest_offset+n_elems_per_rank]
        _copy_to_peer_kernel[grid](
            dest_view,
            local_flat,
            dest_offset,
            n_elements=n_elems_per_rank,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Wait for device activity and synchronize all PEs to ensure copies are complete
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Retrieve the full gathered result and reshape to [M, n_pes * N_local]
    gathered = output_sym.clone().detach()
    nvshmem.free_tensor(output_sym)

    C = gathered.view(M, n_pes * N_local)
    return C
