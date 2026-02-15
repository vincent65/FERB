import torch
import triton
import triton.language as tl
import torch.distributed as dist
import torch.cuda as cuda


@triton.jit
def reduce_accumulate_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for in-place reduction accumulation.
    Uses coalesced memory access and minimizes synchronization.
    Processes multiple elements per thread for better throughput.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Load with coalesced access pattern
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    out_data = tl.load(output_ptr + offsets, mask=mask, other=0.0)
    
    # Accumulate
    result = out_data + data
    
    # Store back with coalesced access
    tl.store(output_ptr + offsets, result, mask=mask)


def solution(
    tensor: torch.Tensor,
    dst: int = 0,
) -> torch.Tensor:
    """
    NCCL-based distributed reduce operation optimized for performance.
    Sums tensors from all ranks, with result only on destination rank.
    Uses NCCL reduce for minimal communication overhead and maximum throughput.
    """
    # Get distributed context
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if not (0 <= dst < world_size):
        raise ValueError(f"dst={dst} is not a valid rank (0 <= dst < {world_size})")
    
    # Ensure tensor is contiguous and on CUDA
    if not tensor.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    
    tensor_contiguous = tensor if tensor.is_contiguous() else tensor.contiguous()
    
    # Initialize output tensor as clone for non-dst ranks, or in-place for dst
    if rank == dst:
        out = tensor_contiguous.clone()
    else:
        out = tensor_contiguous.clone()
    
    # Use NCCL reduce operation which is highly optimized for multi-GPU
    if world_size > 1:
        # NCCL reduce is the most efficient collective operation
        # It uses optimized algorithms (tree-based, ring, etc.) depending on topology
        dist.reduce(out, dst=dst, op=dist.ReduceOp.SUM)
    
    return out
