import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem


@triton.jit
def reduce_sum_kernel(
    output_ptr,
    input_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Vectorized atomic add with coalesced memory access pattern."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input with vectorized reads for better memory throughput
    local_val = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Atomic add to destination buffer
    tl.atomic_add(output_ptr + offsets, local_val, mask=mask)


class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        return (0, self.pt_stream.cuda_stream)


@torch.no_grad()
def solution(
    tensor: torch.Tensor,
    dst: int = 0,
) -> torch.Tensor:
    """
    NCCL Reduce (sum): Sum tensors from all ranks, result only on destination rank.
    Uses NVSHMEM for the reduction.
    
    Strategy v2: Push-based atomic add with optimized memory access.
    - Increased BLOCK_SIZE to 2048 for better memory coalescing
    - Removed redundant torch.cuda.synchronize() before barrier (barrier implies fence)
    - Reduced overhead by reusing stream wrapper
    """
    assert tensor.is_cuda, "Input must be a CUDA tensor"

    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    original_shape = tensor.shape
    original_dtype = tensor.dtype
    n_elements = tensor.numel()

    # Convert to float32 for atomic_add support
    input_flat = tensor.flatten().contiguous().float()

    # Allocate symmetric memory on all PEs
    output_sym = nvshmem.tensor(n_elements, dtype=torch.float32)
    output_sym.zero_()

    # Wrap stream for NVSHMEM barriers
    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)

    # Single barrier to ensure all PEs have zeroed their output_sym
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Every PE atomically adds its local data to the dst PE's buffer
    BLOCK_SIZE = 2048
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Get a view of the destination PE's symmetric buffer
    dst_view = nvshmem.get_peer_tensor(output_sym, dst)

    # Launch Triton kernel to perform atomic addition on the destination PE's buffer
    reduce_sum_kernel[grid](
        dst_view,
        input_flat,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    # Synchronize to ensure all atomic additions have completed
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # On destination rank, return the reduced result
    if my_pe == dst:
        result = output_sym.clone().reshape(original_shape)
        if original_dtype != torch.float32:
            result = result.to(original_dtype)
    else:
        result = tensor.clone()

    nvshmem.free_tensor(output_sym)
    return result
