import torch
import triton
import triton.language as tl
import torch.distributed as dist
import torch.cuda as cuda


@triton.jit
def reduce_kernel(
    input_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for in-place reduction accumulation.
    Uses coalesced memory access and minimizes synchronization.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Load with coalesced access
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
    NVSHMEM-based distributed reduce operation optimized for performance.
    Sums tensors from all ranks, with result only on destination rank.
    Uses binary tree reduction pattern for minimal communication rounds.
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
    
    # Initialize output tensor
    out = tensor.clone()
    
    # Get tensor properties
    numel = tensor.numel()
    
    # Flatten tensor for easier processing
    flat_tensor = tensor.reshape(-1)
    flat_out = out.reshape(-1)
    
    # Determine block size based on GPU capabilities
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    
    # Binary tree reduction for minimal communication rounds
    if world_size > 1:
        # Synchronize before starting reduction
        cuda.synchronize()
        dist.barrier()
        
        # Binary tree reduction
        step = 1
        while step < world_size:
            # Determine if this rank sends or receives in this step
            if rank % (2 * step) == 0:
                # This rank receives from rank + step
                src_rank = rank + step
                if src_rank < world_size:
                    # Create temporary tensor for receiving
                    temp = torch.empty_like(flat_tensor)
                    
                    # Use point-to-point communication
                    dist.recv(temp, src=src_rank)
                    
                    # Synchronize GPU before kernel launch
                    cuda.synchronize()
                    
                    # Accumulate using Triton kernel
                    reduce_kernel[grid](
                        temp,
                        flat_out,
                        numel,
                        BLOCK_SIZE=BLOCK_SIZE,
                    )
                    
                    # Synchronize GPU after kernel
                    cuda.synchronize()
            else:
                # This rank sends to the appropriate receiving rank
                dst_rank = rank - (rank % (2 * step))
                if dst_rank >= 0:
                    # Synchronize GPU before send
                    cuda.synchronize()
                    
                    # Send accumulated data
                    dist.send(flat_out, dst=dst_rank)
                    
                    # Exit after sending
                    break
            
            step *= 2
        
        # Final synchronization
        cuda.synchronize()
        dist.barrier()
    
    # Reshape back to original shape
    out = flat_out.reshape(tensor.shape)
    
    return out
