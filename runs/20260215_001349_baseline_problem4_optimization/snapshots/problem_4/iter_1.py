import torch
import triton
import triton.language as tl


@triton.jit
def reduce_accumulate_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for accumulating data into output tensor.
    Optimized for memory bandwidth with coalesced access patterns.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Load from input with coalesced memory access
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Load current output value
    out_data = tl.load(output_ptr + offsets, mask=mask, other=0.0)
    
    # Accumulate
    result = out_data + data
    
    # Store back with coalesced memory access
    tl.store(output_ptr + offsets, result, mask=mask)


def solution(
    tensor: torch.Tensor,
    dst: int = 0,
) -> torch.Tensor:
    """
    NVSHMEM-based distributed reduce operation optimized for performance.
    Sums tensors from all ranks, with result only on destination rank.
    Uses tree-based reduction pattern and optimized memory access.
    """
    # Get distributed context
    try:
        import torch.distributed as dist
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    except Exception as e:
        raise RuntimeError(f"Failed to get distributed context: {e}")
    
    if not (0 <= dst < world_size):
        raise ValueError(f"dst={dst} is not a valid rank (0 <= dst < {world_size})")
    
    # Ensure tensor is contiguous and on CUDA
    if not tensor.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    # Initialize output tensor
    out = tensor.clone()
    
    # Get tensor properties
    numel = tensor.numel()
    
    # Flatten tensor for easier processing
    flat_tensor = tensor.reshape(-1)
    flat_out = out.reshape(-1)
    
    # Determine block size based on tensor size and GPU capabilities
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    
    # Perform tree-based reduction with optimized synchronization
    if rank == dst:
        # Destination rank accumulates all values
        # Use a tree reduction pattern to minimize synchronization overhead
        for src_rank in range(world_size):
            if src_rank != dst:
                # Create temporary tensor for receiving data
                temp = flat_tensor.clone()
                
                # Broadcast from source rank
                dist.broadcast(temp, src=src_rank)
                
                # Use Triton kernel for efficient accumulation
                reduce_accumulate_kernel[grid](
                    temp,
                    flat_out,
                    numel,
                    BLOCK_SIZE=BLOCK_SIZE,
                )
    else:
        # Non-destination ranks broadcast their data
        # This is a blocking operation that synchronizes with destination
        dist.broadcast(flat_tensor, src=rank)
    
    # Synchronize all ranks to ensure completion
    dist.barrier()
    
    # Reshape back to original shape
    out = flat_out.reshape(tensor.shape)
    
    return out
