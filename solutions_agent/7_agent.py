import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem


@triton.jit
def reduce_scatter_atomic_add_kernel(
    remote_ptr,      # Pointer to remote PE's output buffer (symmetric memory)
    local_ptr,       # Pointer to local chunk data
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    vals = tl.load(local_ptr + offsets, mask=mask)
    tl.atomic_add(remote_ptr + offsets, vals, mask=mask)


class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream
    def __cuda_stream__(self):
        return (0, self.pt_stream.cuda_stream)


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    """
    NVSHMEM/Triton Reduce-Scatter (SUM):
    - Input tensor shape: [world_size * chunk_size, ...]
    - Output tensor shape: [chunk_size, ...] (this rank's chunk of the global sum)

    Implementation: Each PE allocates a symmetric output buffer for its chunk and
    all PEs atomically add their corresponding local chunk into the destination PE's
    buffer. After accumulation, each PE returns its local symmetric buffer reshaped
    to [chunk_size, ...].
    """
    assert tensor.is_cuda, "Input must be a CUDA tensor"
    assert tensor.is_contiguous(), "Input must be contiguous"
    # Triton atomic_add support is most reliable for float32
    assert tensor.dtype == torch.float32, "This implementation currently supports only torch.float32"

    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    # Validate shape divisibility
    assert tensor.shape[0] % n_pes == 0, (
        f"First dimension ({tensor.shape[0]}) must be divisible by world_size ({n_pes})"
    )

    chunk_size = tensor.shape[0] // n_pes
    inner_elems = tensor.numel() // tensor.shape[0]  # product of remaining dims
    chunk_elems = chunk_size * inner_elems

    # Prepare flattened views
    input_flat = tensor.flatten().contiguous()

    # Symmetric output buffer for this PE's reduced chunk
    out_sym = nvshmem.tensor((chunk_elems,), dtype=tensor.dtype)
    out_sym.zero_()

    # Stream wrapper for NVSHMEM barriers
    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)

    # Ensure all PEs have zeroed buffers before accumulation
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Kernel launch config
    BLOCK_SIZE = 512
    grid = (triton.cdiv(chunk_elems, BLOCK_SIZE),)

    # For each destination PE, atomically add our local chunk destined for that PE
    for dest_pe in range(n_pes):
        # Local slice corresponding to chunk 'dest_pe'
        start = dest_pe * chunk_elems
        local_chunk = input_flat[start:start + chunk_elems]

        # Remote view of destination PE's symmetric output buffer
        remote_view = nvshmem.get_peer_tensor(out_sym, dest_pe)

        # Launch kernel to atomically add into the remote buffer
        reduce_scatter_atomic_add_kernel[grid](
            remote_view,
            local_chunk,
            n_elements=chunk_elems,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Synchronize to ensure all remote atomics completed
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Gather local result and reshape
    out = out_sym.clone().detach().reshape((chunk_size,) + tuple(tensor.shape[1:]))

    # Free symmetric memory
    nvshmem.free_tensor(out_sym)

    return out
