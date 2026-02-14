# Distributed GEMM with all-scatter using NCCL
import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    A: torch.Tensor,  # Input: Matrix A, shape [M, K]
                     # Dtype: Any numeric dtype (float32, float16, etc.)
                     # Device: Must be on CUDA device corresponding to this rank
    B: torch.Tensor,  # Input: Matrix B, shape [K, N_local] where N_local = N // world_size
                      # Each rank holds a column shard of B
                      # Dtype: Same as A
                      # Device: Must be on CUDA device corresponding to this rank
) -> torch.Tensor:  # Returns: Complete C matrix, shape [M, N] where N = world_size * N_local
                    #          Contains the result of A @ B_global after all-scatter
    """
    Distributed GEMM with All-Scatter using NCCL.
    
    Computes C = A @ B where B is column-sharded across ranks. Each rank:
    1. Computes local GEMM: C_local = A @ B_local (shape [M, N_local])
    2. Scatters C_local to all other ranks using all-gather
    3. Returns the complete C matrix (shape [M, N] where N = world_size * N_local)
        
    Algorithm:
    1. Each rank computes its local GEMM shard: C_local = A @ B
    2. Use all-gather: each rank sends its C_local, all ranks receive all C_local chunks
    3. Concatenate received chunks along N dimension to form complete C matrix
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - Input tensors must be on the current CUDA device (torch.cuda.current_device())
        - All ranks must have A of shape [M, K] (same M, K across ranks)
        - All ranks must have B of shape [K, N_local] (same N_local across ranks)
        - A and B must be contiguous CUDA tensors
    
    Returns:
        Complete C matrix of shape [M, N] where N = world_size * N_local.
        C[:, rank * N_local : (rank + 1) * N_local] contains the data computed by that rank.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A.is_cuda and B.is_cuda, "Inputs must be CUDA tensors"
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    M, K = A.shape
    K_B, N_local = B.shape
    assert K == K_B, f"A and B must have matching K dimension: {K} != {K_B}"
    
    # Step 1: Compute local GEMM shard
    # C_local = A @ B, shape [M, N_local]
    # Ensure contiguous for better performance
    A = A.contiguous()
    B = B.contiguous()
    C_local = torch.matmul(A, B)
    
    # Step 2: All-scatter: each rank scatters its C_local to all other ranks
    # Use all-gather: each rank sends its C_local, all ranks receive all C_local chunks
    # This implements the "scatter" half of all-gather: every rank distributes its shard to all others
    C_gathered = [torch.zeros_like(C_local) for _ in range(world_size)]
    dist.all_gather(C_gathered, C_local)
    
    # Step 3: Concatenate all chunks along N dimension to form complete C
    # C_gathered is a list of [M, N_local] tensors, one from each rank
    # Concatenate to get [M, world_size * N_local]
    # C[:, rank * N_local : (rank + 1) * N_local] contains data computed by rank
    C = torch.cat(C_gathered, dim=1)
    
    dist.barrier()
    
    return C
