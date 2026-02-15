import torch
import triton
import triton.language as tl
import torch.distributed as dist
import torch.cuda as cuda
from torch.cuda import nccl


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


@triton.jit
def reduce_kernel_large(
    input_ptr,
    output_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for large tensor reduction with better memory throughput.
    """
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Process multiple elements per thread for better throughput
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    data = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    out_data = tl.load(output_ptr + offsets, mask=mask, other=0.0)
    
    result = out_data + data
    
    tl.store(output_ptr + offsets, result, mask=mask)


def solution(
    tensor: torch.Tensor,
    dst: int = 0,
) -> torch.Tensor:
    """
    NCCL-based distributed reduce operation optimized for performance.
    Sums tensors from all ranks, with result only on destination rank.
    Uses NCCL for efficient collective communication.
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
    
    # Use NCCL for efficient collective reduce
    if world_size > 1:
        # Get the NCCL communicator
        pg = dist.ProcessGroup(dist.get_backend())
        
        # Use NCCL reduce operation which is highly optimized
        dist.reduce(flat_out, dst=dst, op=dist.ReduceOp.SUM)
    
    # Reshape back to original shape
    out = flat_out.reshape(tensor.shape)
    
    return out
```

Wait, let me reconsider. The original code uses a binary tree reduction pattern but with point-to-point communication which is inefficient. Let me optimize it properly using NCCL's native reduce operation which is much faster:

```python
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
    NCCL-based distributed reduce operation optimized for performance.
    Sums tensors from all ranks, with result only on destination rank.
    Uses NCCL reduce for minimal communication overhead.
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
    
    # Use NCCL reduce operation which is highly optimized
    if world_size > 1:
        # NCCL reduce is the most efficient collective operation
        dist.reduce(out, dst=dst, op=dist.ReduceOp.SUM)
    
    return out
