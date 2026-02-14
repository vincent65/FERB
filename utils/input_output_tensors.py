"""
Utility functions for creating input tensors and saving output tensors.

These functions are shared across different worker scripts to ensure
consistent tensor creation and saving behavior.
"""

import os
import torch
import json

def save_tensor(output, logs_dir: str, rank: int) -> str:
    """
    Save output tensor(s) to file.
    
    Handles:
    - Single tensor: saves as rank_X.pt
    - Tuple/list of tensors: saves as dict with keys 'output_0', 'output_1', etc.
    - Dict: saves as-is
    """
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, f"rank_{rank}.pt")
    
    # Handle different output types
    if isinstance(output, torch.Tensor):
        # Single tensor
        torch.save(output.detach().cpu(), path)
    elif isinstance(output, (tuple, list)):
        # Multiple tensors - save as dict
        output_dict = {f'output_{i}': t.detach().cpu() if isinstance(t, torch.Tensor) else t 
                      for i, t in enumerate(output)}
        torch.save(output_dict, path)
    elif isinstance(output, dict):
        # Dict - convert tensors to CPU
        output_dict = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v 
                      for k, v in output.items()}
        torch.save(output_dict, path)
    else:
        # Fallback: try to save as-is
        torch.save(output, path)
    
    return path


# ---------------------------------------------------------------------------
# INPUT TENSOR STANDARD (tuple-only)
# ---------------------------------------------------------------------------
# create_input_tensor() ALWAYS returns a tuple of tensors. The worker calls
# solution_fn(*x), so solution signatures are:
#   - solution(tensor) for single-tensor problems: x is (tensor,), so solution_fn(*x) → solution(tensor)
#   - solution(t1, t2) for multi-arg problems: x is (t1, t2), so solution_fn(*x) → solution(t1, t2)
# Single-tensor problems return (tensor,); multi-arg problems return (t1, t2, ...).
# Output from solution_fn may still be a single tensor or a tuple; save_tensor() handles both.
# ---------------------------------------------------------------------------


def create_input_tensor(rank: int, world_size: int, problem_id: int, base_shape: tuple, dtype: torch.dtype, device=None):
    """
    Create appropriate input tensors for this problem. Always returns a tuple of tensors.

    Different collective operations need different input shapes:
    - Problems 1-4 (allreduce, allgather, broadcast, reduce): [M, N] on all ranks → (tensor,)
    - Problem 5 (scatter): [world_size, M, N] on src, [M, N] on others → (tensor,)
    - Problem 6 (gather): [M, N] on all ranks → (tensor,)
    - Problem 7 (reduce-scatter): [world_size * M, N] on all ranks → (tensor,)
    - Problem 8 (all-to-all): [world_size, M, N] on all ranks → (tensor,)
    - Problem 9: (X_hat, dY) for layernorm backward
    - Problem 10: (indices, local_shard) for embedding lookup

    Args:
        rank: Process rank (0..world_size-1)
        world_size: Total number of processes
        problem_id: Problem ID determining input shape
        base_shape: Base tensor shape tuple (e.g., (M, N))
        dtype: Tensor data type
        device: PyTorch device or device string. If None, uses torch.device("cuda", rank)
    """
    if device is None:
        dev = torch.device("cuda", rank)
    elif isinstance(device, str):
        dev = device
    else:
        dev = device
    
    val = float(rank + 1)
    
    if problem_id in [1, 2, 3, 4, 6]:
        # Standard shape: [M, N] with rank-specific values
        return (torch.full(base_shape, val, dtype=dtype, device=dev),)

    elif problem_id == 5:
        # Scatter: src rank has [world_size, M, N], others have [M, N]
        src = 0
        if rank == src:
            # Source rank: create [world_size, M, N] with different values per chunk
            chunks = []
            for i in range(world_size):
                chunk = torch.full(base_shape, float(i + 1), dtype=dtype, device=dev)
                chunks.append(chunk)
            return (torch.stack(chunks, dim=0),)
        else:
            # Other ranks: just need buffer of right shape
            return (torch.zeros(base_shape, dtype=dtype, device=dev),)

    elif problem_id == 7:
        # Reduce-scatter: [world_size * M, N] on all ranks
        expanded_shape = (world_size * base_shape[0],) + base_shape[1:]
        return (torch.full(expanded_shape, val, dtype=dtype, device=dev),)

    elif problem_id == 8:
        # All-to-all: [world_size, M, N] with different values per destination
        chunks = []
        for dest in range(world_size):
            # Value encodes: sender * 10 + destination
            chunk_val = float(rank * 10 + dest)
            chunk = torch.full(base_shape, chunk_val, dtype=dtype, device=dev)
            chunks.append(chunk)
        return (torch.stack(chunks, dim=0),)
    
    elif problem_id == 9:
        # Level 1: simple data parallel example with layernorm backward: each GPU computes d_beta, d_gamma, and we allreducesum across all GPUs
        # Solution signature: solution(X_hat, dY) -> (d_gamma, d_beta)
        
        B, H = base_shape    # base shape = (M, N) = (num_tokens_in_each_rank_also_known_as_batch_size, hidden_dim_of_each_token)

        # Use fixed seed for reproducibility - same inputs for reference and solution
        # Seed based on problem_id and rank to ensure deterministic but different per rank
        torch.manual_seed(42 + problem_id * 1000 + rank)
        
        # Per-rank local inputs (data-parallel)
        # X_hat: normalized activations on this rank (for testing, we use random normalized values)
        X_hat = torch.randn((B, H), dtype=dtype, device=dev)
        X_hat = X_hat / (X_hat.norm(dim=-1, keepdim=True) + 1e-5)  # Normalize for realism
        
        # dY: upstream gradient w.r.t. LayerNorm output on this rank
        dY = torch.randn((B, H), dtype=dtype, device=dev)
        
        # Return tuple matching solution function signature: solution(X_hat, dY)
        return (X_hat, dY)
    
    elif problem_id == 10:
        # Embedding table lookup: distributed embedding lookup with block partitioning
        # Solution signature: solution(indices, local_shard) -> output_vectors
        # Each rank holds a shard of the global embedding table
        # Rank r owns indices [r * shard_size, (r+1) * shard_size)
        
        # base_shape = (M, N) where:
        #   M = shard_size (number of embeddings per rank)
        #   N = embed_dim (embedding dimension)
        shard_size, embed_dim = base_shape
        
        # Use fixed seed for reproducibility
        torch.manual_seed(42 + problem_id * 1000 + rank)
        
        # Create local embedding shard for this rank
        # Shape: [shard_size, embed_dim]
        local_shard = torch.randn((shard_size, embed_dim), dtype=dtype, device=dev)
        
        # Create query indices: each rank queries some indices from the global table
        # We'll query a subset of indices that may come from any rank
        num_queries = shard_size  # Query as many indices as we have in our shard
        global_vocab_size = world_size * shard_size
        
        # Generate random global indices in range [0, global_vocab_size)
        # This ensures we test lookups across different ranks
        indices = torch.randint(0, global_vocab_size, (num_queries,), dtype=torch.long, device=dev)
        
        # Return tuple matching solution function signature: solution(indices, local_shard)
        return (indices, local_shard)
    
    elif problem_id == 11:
        # GEMM all-scatter: A @ B where B is column-sharded
        # A: [M, K] replicated, B: [K, N_local] sharded, C: [M, N] where N = world_size * N_local
        M, N_local = base_shape
        K = 512
        torch.manual_seed(42 + problem_id * 1000 + rank)
        A = torch.randn((M, K), dtype=dtype, device=dev)
        B = torch.randn((K, N_local), dtype=dtype, device=dev)
        return (A, B)
    
    elif problem_id == 12:
        # GEMM all-gather on A: A_global @ B where A is column-sharded
        # A_local: [M, K_local] sharded, B: [K_global, N] replicated, C: [M, N] replicated
        M, K_local = base_shape
        K_global = world_size * K_local
        N = 512
        torch.manual_seed(42 + problem_id * 1000 + rank)
        A_local = torch.randn((M, K_local), dtype=dtype, device=dev)
        B = torch.randn((K_global, N), dtype=dtype, device=dev)
        return (A_local, B)
    
    elif problem_id == 13:
        # GEMM all-reduce: sum_r(A_r @ B_r) where each rank has full local A and B
        # A_local: [M, K] local, B_local: [K, N] local, C: [M, N] = sum_r(A_r @ B_r) replicated
        M, N = base_shape
        K = 512
        torch.manual_seed(42 + problem_id * 1000 + rank)
        A_local = torch.randn((M, K), dtype=dtype, device=dev)
        B_local = torch.randn((K, N), dtype=dtype, device=dev)
        return (A_local, B_local)
    
    elif problem_id == 14:
        # GEMM reduce-scatter: sum_r(A_r @ B_r) then scatter along M dimension
        # A_local: [M, K_local] sharded along K, B_local: [K_local, N] sharded along K
        # C_partial: [M, N] partial, C_local: [M_local, N] = reduce_scatter(C_partial) sharded along M
        M, N = base_shape
        assert M % world_size == 0, f"M ({M}) must be divisible by world_size ({world_size})"
        K_global = 512
        K_local = K_global // world_size
        torch.manual_seed(42 + problem_id * 1000 + rank)
        A_local = torch.randn((M, K_local), dtype=dtype, device=dev)
        B_local = torch.randn((K_local, N), dtype=dtype, device=dev)
        return (A_local, B_local)
    
    else:
        # Default: standard shape
        return (torch.full(base_shape, val, dtype=dtype, device=dev),)


def save_performance_metrics(metrics: dict, logs_dir: str, rank: int) -> str:
    """Save performance metrics to a JSON file."""
    os.makedirs(logs_dir, exist_ok=True)
    path = os.path.join(logs_dir, f"rank_{rank}_perf.json")
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    return path