import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem

@triton.jit
def embedding_lookup_kernel_unified(
    indices_ptr,         # Pointer to SORTED indices
    output_ptr,          # Pointer to output buffer
    shard_ptrs_ptr,      # Pointer to the ARRAY of peer pointers (uint64)
    shard_size: tl.constexpr,
    embed_dim: tl.constexpr,
    n_queries: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_queries

    # 1. Load the Global Index
    global_idx = tl.load(indices_ptr + idx, mask=mask)      # Foreach of this rank's tokens to lookup, their index into the global embedding table

    # 2. Calculate Routing
    target_pe = global_idx // shard_size
    local_offset = (global_idx % shard_size) * embed_dim    # Start_idx within target_pe's shard

    # 3. Load the Base Pointer for that PE
    # We load the raw 64-bit address of the remote shard
    raw_base_addr = tl.load(shard_ptrs_ptr + target_pe, mask=mask)
    
    # 4. FIX: Cast int64 address to a float32 pointer
    # This ensures adding '1' moves 4 bytes, not 1 byte
    remote_base_ptr = raw_base_addr.to(tl.pointer_type(tl.float32))

    # 5. Vectorized Load
    d_range = tl.arange(0, embed_dim)
    
    # Broadcast pointers
    # [BLOCK, DIM]
    ptr_vec = remote_base_ptr[:, None] + (local_offset[:, None] + d_range[None, :])     # matrix of offsets, add to remote base pointer
    mask_vec = mask[:, None]
    
    val_vec = tl.load(ptr_vec, mask=mask_vec)
    
    # 6. Store LINEARLY to match the Reference's "Stacked" format
    # Because we passed sorted indices, storing linearly here preserves the 
    # "sorted by rank" order of the reference.

    # OUTPUT_OFFSET JUST CONVERTS THE 2D TO 1D: TELLS TRITON WHERE EACH ELEMENT GOES IN THE FLAT MEMORY BUFFER
    # Conceptually equivalent to:
    # for i in range(BLOCK_SIZE):
    #     for j in range(embed_dim):
    #         output[idx[i], j] = val_vec[i, j]
    output_offset = idx[:, None] * embed_dim + d_range[None, :]
    tl.store(output_ptr + output_offset, val_vec, mask=mask_vec)

# TODO: PUT THIS IN COMMON FILE
class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream
    def __cuda_stream__(self):
        return (0, self.pt_stream.cuda_stream)


@torch.no_grad()
def solution(indices: torch.Tensor, local_shard: torch.Tensor) -> torch.Tensor:
    assert indices.is_cuda and local_shard.is_cuda
    local_shard = local_shard.contiguous()
    indices = indices.contiguous()
    
    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()
    shard_size, embed_dim = local_shard.shape
    n_queries = indices.numel()
    
    # --- STEP 1: Sort Indices to Match Reference Behavior ---
    # The reference sorts queries by target rank. We must do the same to pass correctness.
    target_ranks = indices // shard_size
    
    # Stable sort by target rank ensures we match the reference's grouping
    sort_order = torch.argsort(target_ranks, stable=True)
    sorted_indices = indices[sort_order].contiguous()

    # --- STEP 2: NVSHMEM Setup ---
    shard_sym = nvshmem.tensor(local_shard.shape, dtype=local_shard.dtype)
    shard_sym[:] = local_shard
    
    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)
    
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Build Phonebook (Peer Pointers)
    peer_ptrs = []
    for pe in range(n_pes):
        ptr = nvshmem.get_peer_tensor(shard_sym, pe).data_ptr()
        peer_ptrs.append(ptr)
    
    peer_ptrs_gpu = torch.tensor(peer_ptrs, dtype=torch.int64, device='cuda')

    # --- STEP 3: Launch Kernel ---
    # We write directly into 'output' in the sorted order
    output = torch.empty((n_queries, embed_dim), device='cuda', dtype=local_shard.dtype)
    
    # Assuming embed_dim fits in block (e.g. <= 256). If larger, use tiling.
    BLOCK_SIZE = 128 
    grid = (triton.cdiv(n_queries, BLOCK_SIZE),)
    
    embedding_lookup_kernel_unified[grid](
        sorted_indices,      # Use the SORTED indices
        output,
        peer_ptrs_gpu,
        shard_size=shard_size,
        embed_dim=embed_dim,
        n_queries=n_queries,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)
    nvshmem.free_tensor(shard_sym)
    
    # # --- STEP 4: Unsort Output to Match Reference Order ---
    # # The reference preserves the original input order (not sorted by target rank).
    # # We sorted the indices, so we need to unsort the output to match.
    # unsort_order = torch.argsort(sort_order)  # Inverse permutation
    # output = output[unsort_order]
    
    return output