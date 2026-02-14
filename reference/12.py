# Distributed GEMM with all-gather on A using NCCL
import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    A_local: torch.Tensor,  # Input: Column shard of A, shape [M, K_local] where K_local = K // world_size
                            # Each rank holds columns [rank * K_local : (rank + 1) * K_local] of A_global
                            # Dtype: Any numeric dtype (float32, float16, etc.)
                            # Device: Must be on CUDA device corresponding to this rank
    B: torch.Tensor,  # Input: Full B matrix, shape [K, N] - fully replicated on all ranks
                     # Dtype: Same as A_local
                     # Device: Must be on CUDA device corresponding to this rank
) -> torch.Tensor:  # Returns: Complete C matrix, shape [M, N] - fully replicated on all ranks
                    #          Contains the result of A_global @ B
    """
    Distributed GEMM with All-Gather on A using NCCL.
    
    Computes C = A_global @ B where A is column-sharded across ranks. Each rank:
    1. Gathers all A shards from all ranks using all-gather along K dimension
    2. Computes local GEMM: C = A_global @ B (shape [M, N])
    3. Returns the complete C matrix (fully replicated on all ranks)
        
    Algorithm:
    1. Each rank starts with A_local [M, K_local] - its column shard of A_global
    2. Use all-gather to collect all A_local shards into A_global [M, K] on every rank
    3. Compute C = A_global @ B locally (B is already replicated)
    4. All ranks have the same complete C matrix
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - Input tensors must be on the current CUDA device (torch.cuda.current_device())
        - All ranks must have A_local of shape [M, K_local] (same M, K_local across ranks)
        - All ranks must have B of shape [K, N] where K = world_size * K_local
        - A_local and B must be contiguous CUDA tensors
    
    Returns:
        Complete C matrix of shape [M, N], fully replicated on all ranks.
        C = A_global @ B where A_global is the concatenation of all A_local shards along dim=1.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    M, K_local = A_local.shape
    K_B, N = B.shape
    
    # Verify K dimension matches: B should have K = world_size * K_local
    K_global = world_size * K_local
    assert K_B == K_global, f"B must have K dimension = world_size * K_local: {K_B} != {world_size} * {K_local}"
    
    # Step 1: All-gather A_local shards along K dimension
    # Each rank sends its [M, K_local] shard, all ranks receive all shards
    # This reconstructs A_global [M, K] on every rank
    A_gathered = [torch.zeros_like(A_local) for _ in range(world_size)]
    dist.all_gather(A_gathered, A_local)
    
    # Concatenate all A shards along K dimension (dim=1) to form A_global
    # A_gathered is a list of [M, K_local] tensors, one from each rank
    # Concatenate to get [M, K_global] where K_global = world_size * K_local
    A_global = torch.cat(A_gathered, dim=1)
    
    # Step 2: Compute local GEMM: C = A_global @ B
    # Ensure contiguous for better performance
    A_global = A_global.contiguous()
    B = B.contiguous()
    C = torch.matmul(A_global, B)
    
    dist.barrier()
    
    return C
