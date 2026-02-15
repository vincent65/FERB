import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem


@triton.jit
def copy_kernel(
    output_ptr,
    input_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    val = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, val, mask=mask)


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
    
    Strategy: Each PE writes its data into its own symmetric buffer.
    After barrier, the destination PE reads from all peers and sums locally.
    This avoids remote atomic writes which can cause hangs.
    """
    assert tensor.is_cuda, "Input must be a CUDA tensor"

    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    original_shape = tensor.shape
    original_dtype = tensor.dtype
    n_elements = tensor.numel()

    # Convert to float32 for consistent handling
    input_flat = tensor.flatten().contiguous().float()

    # Allocate symmetric memory - each PE stores its own data here
    data_sym = nvshmem.tensor(n_elements, dtype=torch.float32)

    # Copy local data into symmetric buffer
    data_sym.copy_(input_flat)

    # Wrap stream for NVSHMEM barriers
    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)

    # Synchronize to ensure all PEs have written their data
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # On destination rank, read from all peers and sum
    if my_pe == dst:
        # Start with our own data
        result_flat = data_sym.clone()

        # Read from each other PE and accumulate
        for src_pe in range(n_pes):
            if src_pe == my_pe:
                continue
            # Get a view of the source PE's symmetric buffer
            peer_view = nvshmem.get_peer_tensor(data_sym, src_pe)
            # Read peer data using a Triton kernel for efficiency
            peer_data = torch.empty(n_elements, dtype=torch.float32, device=tensor.device)
            BLOCK_SIZE = 512
            grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
            copy_kernel[grid](
                peer_data,
                peer_view,
                n_elements=n_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            # Accumulate
            result_flat.add_(peer_data)

        result = result_flat.reshape(original_shape)
        if original_dtype != torch.float32:
            result = result.to(original_dtype)
    else:
        # Non-destination ranks return original tensor unchanged
        result = tensor.clone()

    # Synchronize before cleanup
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Clean up symmetric memory
    nvshmem.free_tensor(data_sym)

    return result
