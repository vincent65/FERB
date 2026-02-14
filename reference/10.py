# embedding table lookup
import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    indices: torch.Tensor,  # Input: Global indices this GPU wants to look up.
                           # Shape: [N] where N is the number of queries
                           # Dtype: torch.long (int64)
                           # Device: Must be on CUDA device corresponding to this rank
    local_shard: torch.Tensor,  # Input: The slice of the global embedding table held by this GPU.
                               # Shape: [ShardSize, D] where ShardSize is the number of embeddings
                               #        in this rank's shard, D is the embedding dimension
                               # Dtype: Any numeric dtype (float32, float16, etc.)
                               # Device: Must be on CUDA device corresponding to this rank
) -> torch.Tensor:  # Returns: [N, D] tensor containing the embedding vectors for the requested indices
    """
    Distributed Embedding Lookup using NCCL All-to-All.
    
    Performs a distributed embedding lookup where each rank holds a shard of the global
    embedding table. Indices are distributed across ranks using block partitioning:
    rank r owns indices [r * shard_size, (r+1) * shard_size).
    
    The algorithm:
    1. Each rank determines which other ranks own the indices it needs
    2. Indices are exchanged via All-to-All
    3. Each rank performs local lookups for indices it received
    4. Embedding vectors are exchanged back via All-to-All
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - Input tensors must be on the current CUDA device (torch.cuda.current_device())
        - All ranks must have the same shard_size (local_shard.shape[0])
        - All ranks must have the same embedding dimension (local_shard.shape[1])
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert indices.is_cuda and local_shard.is_cuda, "Inputs must be CUDA tensors"
    assert indices.dtype == torch.long, "indices must be torch.long"
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    shard_size = local_shard.shape[0]
    embed_dim = local_shard.shape[1]
    
    # Ensure indices are contiguous and on the right device
    indices = indices.contiguous().to(torch.cuda.current_device())
    
    # 1. Determine which rank owns each index (block partitioning)
    # Rank r owns indices [r * shard_size, (r+1) * shard_size)
    target_ranks = indices // shard_size
    
    # 2. Sort/Group indices by target rank
    # In NCCL, we must prepare 'send_counts' for All-to-All
    send_indices_list = [indices[target_ranks == r] for r in range(world_size)]
    send_counts = torch.tensor([len(idx) for idx in send_indices_list], dtype=torch.long, device='cuda')
    
    # 3. Exchange counts so we know how many indices we are receiving from others
    recv_counts = torch.zeros(world_size, dtype=torch.long, device='cuda')
    dist.all_to_all_single(recv_counts, send_counts)
    
    # 4. Exchange the actual indices (All-to-All)
    # This tells Rank B: "Rank A needs you to look up these indices"
    # Concatenate all indices to send (filter out empty tensors)
    non_empty_lists = [idx_list for idx_list in send_indices_list if len(idx_list) > 0]
    if non_empty_lists:
        flat_send_indices = torch.cat(non_empty_lists)
    else:
        flat_send_indices = torch.empty(0, dtype=torch.long, device='cuda')
    
    total_recv = recv_counts.sum().item()
    total_send = send_counts.sum().item()
    received_indices = torch.empty(total_recv, dtype=torch.long, device='cuda')
    
    # Only perform all_to_all if there's data to exchange
    if total_recv > 0 or total_send > 0:
        dist.all_to_all_single(
            received_indices, 
            flat_send_indices,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist()
        )
    
    # 5. Local Lookup: Every rank looks up the requests it received
    # We subtract the offset to get the index relative to our local shard
    if total_recv > 0:
        local_lookup_indices = received_indices - (rank * shard_size)
        # Clamp to valid range for safety
        local_lookup_indices = torch.clamp(local_lookup_indices, 0, shard_size - 1)
        retrieved_vectors = local_shard[local_lookup_indices]  # Shape: [num_received, D]
    else:
        retrieved_vectors = torch.empty((0, embed_dim), dtype=local_shard.dtype, device='cuda')
    
    # 6. Exchange the vectors back (All-to-All)
    # We must send vectors back to the ranks that requested them
    output_vectors = torch.empty((len(indices), embed_dim), dtype=local_shard.dtype, device='cuda')
    
    # Split sizes for vectors: all_to_all_single with split sizes works on the first dimension (rows)
    # input_split_sizes: how many rows to send to each rank from retrieved_vectors
    # output_split_sizes: how many rows to receive from each rank into output_vectors
    # Note: We use row counts, NOT multiplied by embed_dim, because all_to_all_single
    # automatically handles the 2D nature by splitting along dim 0
    input_split_sizes = recv_counts.cpu().tolist()  # Rows in retrieved_vectors to send to each rank
    output_split_sizes = send_counts.cpu().tolist()  # Rows in output_vectors to receive from each rank
    
    # Only perform all_to_all if there's data to exchange
    if len(indices) > 0 or retrieved_vectors.numel() > 0:
        dist.all_to_all_single(
            output_vectors,
            retrieved_vectors,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes
        )
    
    return output_vectors