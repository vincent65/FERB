import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    tensor: torch.Tensor,  # Input tensor: Must be a contiguous CUDA tensor on the current rank's device.
                           # Shape: [world_size, *chunk_shape] on src rank (contains data for all ranks)
                           #        [*chunk_shape] on other ranks (will receive one chunk)
                           # Dtype: Any numeric dtype (float32, float16, int32, etc.)
                           # Device: Must be on the CUDA device corresponding to this rank
                           # On src rank: tensor[i] will be sent to rank i
                           # On other ranks: tensor shape should match the chunk shape
    src: int = 0,          # Source rank that scatters its tensor chunks to all others
) -> torch.Tensor:  # Returns: A tensor of shape [*chunk_shape] containing this rank's chunk
                    #          from the source rank's tensor.
    """
    NCCL Scatter: One rank distributes different chunks to each rank.
    
    The source rank has a tensor with world_size chunks, and each rank receives
    one chunk. This is the inverse of gather.
    
    Complexity: O(N) naive, O(log N) with tree-based implementation
    
    Example with 4 ranks, src=0, tensor shape [4, 3] on rank 0:
        Before: Rank 0: [[1,1,1], [2,2,2], [3,3,3], [4,4,4]]
                Other ranks: don't care (will be overwritten)
        After:  Rank 0: [1,1,1], Rank 1: [2,2,2], Rank 2: [3,3,3], Rank 3: [4,4,4]
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - src rank must have tensor of shape [world_size, *chunk_shape]
        - All other ranks must have tensor of shape [*chunk_shape]
        - src must be a valid rank (0 <= src < world_size)
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == src:
        # Source rank: split tensor into chunks for each rank
        assert tensor.shape[0] == world_size, f"Source tensor must have {world_size} chunks"
        scatter_list = list(tensor.chunk(world_size, dim=0))
        scatter_list = [chunk.squeeze(0).contiguous() for chunk in scatter_list]
        out = torch.empty_like(scatter_list[0])
    else:
        # Other ranks: prepare output buffer
        scatter_list = None
        out = tensor.clone()
    
    dist.scatter(out, scatter_list=scatter_list, src=src)
    return out

