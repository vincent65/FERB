# Distributed GEMM with all-reduce using NCCL
import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    A_local: torch.Tensor,  # Input: Local A matrix, shape [M, K]
                           # Each rank has its own A (not a shard - full local matrix)
                           # Dtype: Any numeric dtype (float32, float16, etc.)
                           # Device: Must be on CUDA device corresponding to this rank
    B_local: torch.Tensor,  # Input: Local B matrix, shape [K, N]
                           # Each rank has its own B (not a shard - full local matrix)
                           # Dtype: Same as A_local
                           # Device: Must be on CUDA device corresponding to this rank
) -> torch.Tensor:  # Returns: Reduced C matrix, shape [M, N] - fully replicated on all ranks
                    #          Contains sum of all ranks' partial products: sum_r(A_r @ B_r)
    """
    Distributed GEMM with All-Reduce using NCCL.
    
    Computes C = sum_r(A_r @ B_r) where each rank has its own local A and B matrices.
    Each rank:
    1. Computes local GEMM: C_local = A_local @ B_local (shape [M, N])
    2. All-reduces C_local across all ranks with sum operation
    3. Returns the complete reduced C matrix (fully replicated on all ranks)
        
    Algorithm:
    1. Each rank computes its local partial product: C_local = A_local @ B_local
    2. Use all-reduce with sum to combine all partial products across ranks
    3. All ranks have the same complete reduced C matrix
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - Input tensors must be on the current CUDA device (torch.cuda.current_device())
        - All ranks must have A_local of shape [M, K] (same M, K across ranks)
        - All ranks must have B_local of shape [K, N] (same K, N across ranks)
        - A_local and B_local must be contiguous CUDA tensors
    
    Returns:
        Reduced C matrix of shape [M, N], fully replicated on all ranks.
        C = sum_r(A_r @ B_r) where the sum is over all ranks.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B_local.is_cuda, "Inputs must be CUDA tensors"
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    M, K = A_local.shape
    K_B, N = B_local.shape
    assert K == K_B, f"A_local and B_local must have matching K dimension: {K} != {K_B}"
    
    # Step 1: Compute local GEMM partial product
    # C_local = A_local @ B_local, shape [M, N]
    # Ensure contiguous for better performance
    A_local = A_local.contiguous()
    B_local = B_local.contiguous()
    C_local = torch.matmul(A_local, B_local)
    
    # Step 2: All-reduce with sum to combine all partial products
    # Each rank contributes its C_local, all ranks receive the sum
    # This implements: C = sum_r(C_r) where C_r = A_r @ B_r
    C = C_local.clone()
    dist.all_reduce(C, op=dist.ReduceOp.SUM)
    
    dist.barrier()
    
    return C
