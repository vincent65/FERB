import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    tensor: torch.Tensor,  # Input tensor: Must be a contiguous CUDA tensor on the current rank's device.
                           # Shape: Arbitrary (any number of dimensions, e.g., [M, N], [B, C, H, W], etc.)
                           # Dtype: Any numeric dtype (float32, float16, int32, etc.)
                           # Device: Must be on the CUDA device corresponding to this rank
                           # The tensor on rank 0 contains the data to broadcast.
                           # Tensors on other ranks will be overwritten.
    src: int = 0,          # Source rank that broadcasts its tensor to all others
) -> torch.Tensor:  # Returns: A tensor of the same shape and dtype as input, containing the data
                    #          from the source rank. All ranks will have identical output.
    """
    NCCL Broadcast: One rank sends its tensor to all other ranks.
    
    This is the simplest collective operation - one sender, multiple receivers.
    The source rank's tensor is copied to all other ranks.
    
    Complexity: O(log N) with tree-based implementation
    
    Example with 4 ranks, src=0:
        Before: Rank 0: [1,2,3], Rank 1: [4,5,6], Rank 2: [7,8,9], Rank 3: [10,11,12]
        After:  Rank 0: [1,2,3], Rank 1: [1,2,3], Rank 2: [1,2,3], Rank 3: [1,2,3]
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - All ranks must call this function with tensors of the same shape and dtype
        - src must be a valid rank (0 <= src < world_size)
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    out = tensor.clone()
    dist.broadcast(out, src=src)
    return out

