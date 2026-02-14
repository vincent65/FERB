import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    tensor: torch.Tensor,  # Input tensor: Must be a contiguous CUDA tensor on the current rank's device.
                           # Shape: [world_size * chunk_size, ...] - must be divisible by world_size
                           # Dtype: Any numeric dtype (float32, float16, int32, etc.)
                           # Device: Must be on the CUDA device corresponding to this rank
                           # All ranks must provide tensors of identical shape and dtype.
) -> torch.Tensor:  # Returns: A tensor of shape [chunk_size, ...] containing this rank's portion
                    #          of the reduced result. The full reduced result is distributed across
                    #          all ranks (rank i gets chunk i of the elementwise sum).
    """
    NCCL Reduce-Scatter: Reduce tensors from all ranks, then scatter result chunks.
    
    This is a fused reduce + scatter operation. First, all tensors are summed
    elementwise. Then, the result is divided into world_size chunks, and each
    rank receives one chunk. This is useful for distributed optimizers.
    
    Complexity: O(N) with ring algorithm (same as ring all-reduce's reduce-scatter phase)
    
    Example with 4 ranks, input shape [4, 3]:
        Before: Rank 0: [[1,1,1], [1,1,1], [1,1,1], [1,1,1]]
                Rank 1: [[2,2,2], [2,2,2], [2,2,2], [2,2,2]]
                Rank 2: [[3,3,3], [3,3,3], [3,3,3], [3,3,3]]
                Rank 3: [[4,4,4], [4,4,4], [4,4,4], [4,4,4]]
        
        Sum:    [[10,10,10], [10,10,10], [10,10,10], [10,10,10]]
        
        After:  Rank 0: [10,10,10]  (chunk 0 of sum)
                Rank 1: [10,10,10]  (chunk 1 of sum)
                Rank 2: [10,10,10]  (chunk 2 of sum)
                Rank 3: [10,10,10]  (chunk 3 of sum)
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - All ranks must call this function with tensors of the same shape and dtype
        - First dimension must be divisible by world_size
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    
    world_size = dist.get_world_size()
    
    # Verify tensor is divisible by world_size
    assert tensor.shape[0] % world_size == 0, \
        f"First dimension ({tensor.shape[0]}) must be divisible by world_size ({world_size})"
    
    chunk_size = tensor.shape[0] // world_size
    out_shape = (chunk_size,) + tensor.shape[1:]
    
    out = torch.empty(out_shape, dtype=tensor.dtype, device=tensor.device)
    dist.reduce_scatter_tensor(out, tensor, op=dist.ReduceOp.SUM)
    
    return out

