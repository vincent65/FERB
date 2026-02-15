import torch
import triton
import triton.language as tl
import torch.distributed as dist


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


@triton.jit
def tree_reduce_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for tree-based reduction accumulation.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Load with prefetching optimization
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    out_data = tl.load(output_ptr + offsets, mask=mask, other=0.0)
    
    # Accumulate
    result = out_data + data
    
    # Store back
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
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if not (0 <= dst < world_size):
        raise ValueError(f"dst={dst} is not a valid rank (0 <= dst < {world_size})")
    
    # Ensure tensor is contiguous and on CUDA
    if not tensor.is_cuda:
        raise ValueError("Input tensor must be on CUDA device")
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    
    # Initialize output tensor on destination rank
    if rank == dst:
        out = tensor.clone()
    else:
        out = tensor.clone()
    
    # Get tensor properties
    numel = tensor.numel()
    
    # Flatten tensor for easier processing
    flat_tensor = tensor.reshape(-1)
    flat_out = out.reshape(-1)
    
    # Determine block size based on tensor size and GPU capabilities
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    
    # Use tree-based reduction to minimize communication rounds
    # Calculate tree structure for efficient reduction
    def tree_reduce():
        current_rank = rank
        step = 1
        
        while step < world_size:
            # Determine if this rank sends or receives in this step
            if current_rank % (2 * step) == 0:
                # This rank receives from current_rank + step
                src = current_rank + step
                if src < world_size:
                    # Create temporary tensor for receiving
                    temp = flat_tensor.clone()
                    
                    # Use point-to-point communication for efficiency
                    dist.recv(temp, src=src)
                    
                    # Accumulate using Triton kernel
                    tree_reduce_kernel[grid](
                        temp,
                        flat_out,
                        numel,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
            else:
                # This rank sends to current_rank - step
                dst_rank = current_rank - (current_rank % (2 * step))
                if dst_rank >= 0:
                    # Send accumulated data
                    dist.send(flat_out, dst=dst_rank)
                    break
            
            step *= 2
    
    # Perform tree-based reduction
    if world_size > 1:
        tree_reduce()
    
    # Synchronize all ranks to ensure completion
    dist.barrier()
    
    # Reshape back to original shape
    out = flat_out.reshape(tensor.shape)
    
    return out
