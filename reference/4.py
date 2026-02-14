import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    tensor: torch.Tensor,  # Input tensor: Must be a contiguous CUDA tensor on the current rank's device.
                           # Shape: Arbitrary (any number of dimensions, e.g., [M, N], [B, C, H, W], etc.)
                           # Dtype: Any numeric dtype (float32, float16, int32, etc.)
                           # Device: Must be on the CUDA device corresponding to this rank
                           # All ranks must provide tensors of identical shape and dtype.
    dst: int = 0,          # Destination rank that receives the reduced result
) -> torch.Tensor:  # Returns: On dst rank: tensor with elementwise sum of all ranks' inputs
                    #          On other ranks: the input tensor (unchanged or undefined)
    """
    NCCL Reduce: Sum tensors from all ranks, result only on destination rank.
    
    Similar to all-reduce, but the result is only stored on the destination rank.
    More efficient than all-reduce when only one rank needs the result.
    
    Complexity: O(log N) with tree-based reduction
    
    Example with 4 ranks, dst=0:
        Before: Rank 0: [1,1,1], Rank 1: [2,2,2], Rank 2: [3,3,3], Rank 3: [4,4,4]
        After:  Rank 0: [10,10,10], Rank 1: [2,2,2], Rank 2: [3,3,3], Rank 3: [4,4,4]
                        ^ sum of all          ^ unchanged (or undefined)
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - All ranks must call this function with tensors of the same shape and dtype
        - dst must be a valid rank (0 <= dst < world_size)
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    out = tensor.clone()
    dist.reduce(out, dst=dst, op=dist.ReduceOp.SUM)
    return out

