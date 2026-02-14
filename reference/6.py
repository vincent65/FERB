import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    tensor: torch.Tensor,  # Input tensor: Must be a contiguous CUDA tensor on the current rank's device.
                           # Shape: [*chunk_shape] - each rank contributes one chunk
                           # Dtype: Any numeric dtype (float32, float16, int32, etc.)
                           # Device: Must be on the CUDA device corresponding to this rank
                           # All ranks must provide tensors of identical shape and dtype.
    dst: int = 0,          # Destination rank that gathers all tensors
) -> torch.Tensor:  # Returns: On dst rank: tensor of shape [world_size, *chunk_shape] containing
                    #                        all ranks' tensors concatenated
                    #          On other ranks: the input tensor (unchanged)
    """
    NCCL Gather: All ranks send their tensors to one destination rank.
    
    The destination rank collects tensors from all ranks into a single tensor
    with an extra dimension. This is the inverse of scatter.
    
    Complexity: O(N) naive, O(log N) with tree-based implementation
    
    Example with 4 ranks, dst=0:
        Before: Rank 0: [1,1,1], Rank 1: [2,2,2], Rank 2: [3,3,3], Rank 3: [4,4,4]
        After:  Rank 0: [[1,1,1], [2,2,2], [3,3,3], [4,4,4]]  (shape [4, 3])
                Other ranks: [unchanged]
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - All ranks must call this function with tensors of the same shape and dtype
        - dst must be a valid rank (0 <= dst < world_size)
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == dst:
        # Destination rank: prepare list to receive all tensors
        gather_list = [torch.empty_like(tensor) for _ in range(world_size)]
    else:
        gather_list = None
    
    dist.gather(tensor, gather_list=gather_list, dst=dst)
    
    if rank == dst:
        # Stack gathered tensors into single tensor
        return torch.stack(gather_list, dim=0)
    else:
        return tensor

