# Distributed GEMM with reduce-scatter using NCCL
import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    A_local: torch.Tensor,  # Input: Column shard of A, shape [M, K_local] where K_local = K // world_size
                           # Each rank holds columns [rank * K_local : (rank + 1) * K_local] of A_global
                           # Dtype: Any numeric dtype (float32, float16, etc.)
                           # Device: Must be on CUDA device corresponding to this rank
    B_local: torch.Tensor,  # Input: Row shard of B, shape [K_local, N]
                           # Each rank holds rows [rank * K_local : (rank + 1) * K_local] of B_global
                           # Dtype: Same as A_local
                           # Device: Must be on CUDA device corresponding to this rank
) -> torch.Tensor:  # Returns: Row shard of C, shape [M_local, N] where M_local = M // world_size
                    #          Contains rank's portion of sum_r(A_r @ B_r) after reduce-scatter
    """
    Distributed GEMM with Reduce-Scatter using NCCL.
    
    Computes C = sum_r(A_r @ B_r) where A and B are sharded along K dimension, then reduce-scatters
    the result along M dimension. Each rank:
    1. Computes local partial product: C_partial = A_local @ B_local (shape [M, N])
    2. Reduce-scatters C_partial along M dimension with sum operation
    3. Returns its row shard of the reduced C matrix (shape [M_local, N])
    
    Algorithm:
    1. Each rank computes C_partial = A_local @ B_local (full [M, N] partial result)
    2. Use reduce-scatter to split C_partial along dim=0 into world_size chunks
    3. Sum chunk i across all ranks and place result on rank i
    4. Each rank receives its [M_local, N] shard where M_local = M // world_size
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - Input tensors must be on the current CUDA device (torch.cuda.current_device())
        - All ranks must have A_local of shape [M, K_local] (same M, K_local across ranks)
        - All ranks must have B_local of shape [K_local, N] (same K_local, N across ranks)
        - A_local and B_local must be contiguous CUDA tensors
        - M must be divisible by world_size
    
    Returns:
        Row shard of C matrix of shape [M_local, N] where M_local = M // world_size.
        C_local = [sum_r(A_r @ B_r)][rank * M_local : (rank + 1) * M_local, :]
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert A_local.is_cuda and B_local.is_cuda, "Inputs must be CUDA tensors"
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    M, K_local = A_local.shape
    K_B, N = B_local.shape
    assert K_local == K_B, f"A_local and B_local must have matching K_local dimension: {K_local} != {K_B}"
    assert M % world_size == 0, f"M ({M}) must be divisible by world_size ({world_size})"
    
    M_local = M // world_size
    
    # Step 1: Compute local partial product
    # C_partial = A_local @ B_local, shape [M, N]
    # This is a partial contribution to the global sum
    A_local = A_local.contiguous()
    B_local = B_local.contiguous()
    C_partial = torch.matmul(A_local, B_local)
    
    # Step 2: Reduce-scatter along M dimension (dim=0)
    # Split C_partial into world_size chunks along dim=0
    # Sum chunk i across all ranks and place result on rank i
    # Output: each rank gets [M_local, N] shard
    C_local = torch.empty((M_local, N), dtype=C_partial.dtype, device=C_partial.device)
    dist.reduce_scatter_tensor(C_local, C_partial, op=dist.ReduceOp.SUM)
    
    dist.barrier()
    
    return C_local
