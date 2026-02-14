# Multi-Head Attention Combine: AllToAll Head Gathering + Out Projection
# Ulysses-style combine phase: compute attention on local heads, gather results, apply out projection.
# Optionally supports sequence parallelism (split along sequence length after attention).

import math
import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    q: torch.Tensor,                  # Query for local heads. Shape: [B, T, num_heads_local, head_dim].
                                      # Contiguous CUDA tensor. Only the heads assigned to this GPU.
    k: torch.Tensor,                  # Key for local heads. Same shape as q.
    v: torch.Tensor,                  # Value for local heads. Same shape as q.
    head_assignments: torch.Tensor,   # Per-head GPU assignment. Shape: [num_heads], dtype long.
                                      # Values in [0, world_size). Same as used in dispatch (53.py).
    w_out: torch.Tensor,              # Output projection weight. Shape: [D_out, num_heads * head_dim] or [D_out, D].
                                      # Full weight matrix (not sharded).
    scale: float | None = None,       # Attention scale factor. If None, uses 1/sqrt(head_dim).
    sequence_parallel: bool = False,  # If True, split output along sequence dimension after attention.
                                      # If False, standard Ulysses (gather all heads, then out projection).
) -> torch.Tensor:                    # Returns: [B, T, D_out] or [B, T_local, D_out] if sequence_parallel.
    """
    MHA combine phase: compute attention on local heads, gather results, apply out projection.

    Flow:
        1. COMPUTE ATTENTION — Q @ K^T / scale, softmax, @ V (on local heads only).
        2. GATHER HEADS      — AllToAll collects attention outputs from all GPUs.
        3. OUT PROJECTION    — Apply output projection weight.
        4. SEQUENCE SPLIT    — (if sequence_parallel) AllToAll splits along sequence dimension.

    Example (num_heads=8, world_size=4, heads_per_gpu=2):
        Input (per GPU):
            GPU 0: QKV for heads [0, 1] → [B, T, 2, head_dim]
            GPU 1: QKV for heads [2, 3] → [B, T, 2, head_dim]
            GPU 2: QKV for heads [4, 5] → [B, T, 2, head_dim]
            GPU 3: QKV for heads [6, 7] → [B, T, 2, head_dim]
        
        After attention + gather:
            All GPUs: [B, T, 8, head_dim] (all heads combined)
        
        After out projection:
            All GPUs: [B, T, D_out]

    Preconditions:
        - torch.distributed initialized with NCCL backend.
        - q, k, v must have the same shape (local heads only).
        - head_assignments values in [0, world_size).
        - All ranks must have the same B, T initially.
        - w_out shape must match: [D_out, num_heads * head_dim] where num_heads = len(head_assignments).
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"
    assert q.is_cuda and q.is_contiguous(), "q must be contiguous CUDA tensor"
    assert k.is_cuda and k.is_contiguous(), "k must be contiguous CUDA tensor"
    assert v.is_cuda and v.is_contiguous(), "v must be contiguous CUDA tensor"
    assert w_out.is_cuda and w_out.is_contiguous(), "w_out must be contiguous CUDA tensor"
    assert head_assignments.dtype == torch.long, "head_assignments must be torch.long"

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    B, T, num_heads_local, head_dim = q.shape
    num_heads = len(head_assignments)

    # ──────────────────────────── COMPUTE ATTENTION ──────────────────────────────
    # Attention on local heads: Q @ K^T / scale, softmax, @ V
    # q, k: [B, T, num_heads_local, head_dim]
    # attn_scores: [B, num_heads_local, T, T]
    
    # Reshape for batched matmul: [B, num_heads_local, T, head_dim]
    q_reshaped = q.transpose(1, 2).contiguous()  # [B, num_heads_local, T, head_dim]
    k_reshaped = k.transpose(1, 2).contiguous()  # [B, num_heads_local, T, head_dim]
    v_reshaped = v.transpose(1, 2).contiguous()  # [B, num_heads_local, T, head_dim]

    # Compute attention scores: Q @ K^T
    attn_scores = torch.matmul(q_reshaped, k_reshaped.transpose(-2, -1))  # [B, num_heads_local, T, T]

    # Scale
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    attn_scores = attn_scores * scale

    # Softmax
    attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)  # [B, num_heads_local, T, T]

    # Apply to values
    attn_out = torch.matmul(attn_probs, v_reshaped)  # [B, num_heads_local, T, head_dim]

    # Reshape back: [B, T, num_heads_local, head_dim]
    attn_out = attn_out.transpose(1, 2).contiguous()  # [B, T, num_heads_local, head_dim]

    # ──────────────────────────── GATHER HEADS (AllToAll) ──────────────────────────────
    # Each GPU has attention outputs for its local heads. We need to gather all heads.
    # Count how many heads each GPU has
    heads_per_gpu = torch.bincount(head_assignments, minlength=world_size).to(
        device=q.device, dtype=torch.long
    )
    
    # Flatten head dimension: [B, T, num_heads_local, head_dim] -> [B*T, num_heads_local, head_dim]
    BT = B * T
    attn_out_flat = attn_out.view(BT, num_heads_local, head_dim)  # [BT, num_heads_local, head_dim]
    
    # Flatten further: [BT, num_heads_local, head_dim] -> [BT * num_heads_local, head_dim]
    attn_out_packed = attn_out_flat.view(BT * num_heads_local, head_dim)  # [BT * num_heads_local, head_dim]

    # AllToAll to gather all heads
    # Each GPU sends its local heads, receives heads from all GPUs
    send_counts = (heads_per_gpu * BT).tolist()  # How many head-tokens each GPU sends
    recv_counts = torch.empty(world_size, dtype=torch.long, device=q.device)
    dist.all_to_all_single(recv_counts, torch.tensor(send_counts, dtype=torch.long, device=q.device))

    total_recv = int(recv_counts.sum().item())
    attn_gathered_flat = torch.empty((total_recv, head_dim), dtype=q.dtype, device=q.device)
    
    dist.all_to_all_single(
        attn_gathered_flat, attn_out_packed,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts,
    )

    # Reshape: [BT * num_heads, head_dim] -> [B, T, num_heads, head_dim]
    attn_gathered = attn_gathered_flat.view(B, T, num_heads, head_dim)

    # ──────────────────────────── OUT PROJECTION ──────────────────────────────
    # Reshape for matmul: [B, T, num_heads, head_dim] -> [B*T, num_heads * head_dim]
    attn_gathered_flat = attn_gathered.view(B * T, num_heads * head_dim)
    
    # Out projection: [B*T, num_heads * head_dim] @ [num_heads * head_dim, D_out]^T
    out = torch.matmul(attn_gathered_flat, w_out.t())  # [B*T, D_out]
    
    # Reshape: [B*T, D_out] -> [B, T, D_out]
    out = out.view(B, T, -1)

    # ──────────────────────────── SEQUENCE PARALLELISM (optional) ──────────────────────────────
    if sequence_parallel:
        # Split output along sequence dimension (T) across GPUs
        # Each GPU gets a chunk of the sequence
        T_per_gpu = T // world_size
        assert T % world_size == 0, f"T ({T}) must be divisible by world_size ({world_size}) for sequence parallelism"
        
        # Reshape: [B, T, D_out] -> [B*T, D_out]
        out_flat = out.view(B * T, -1)
        
        # Split into chunks: [B*T_per_gpu, D_out] per GPU
        chunks = torch.chunk(out_flat, world_size, dim=0)
        out_chunk = chunks[rank]  # This GPU's chunk: [B*T_per_gpu, D_out]
        
        # AllToAll to distribute chunks
        D_out = out.shape[-1]
        send_chunk = out_chunk.contiguous()
        recv_chunks = [torch.empty((B * T_per_gpu, D_out), dtype=out.dtype, device=out.device) 
                      for _ in range(world_size)]
        
        # Stack for AllToAll: [world_size, B*T_per_gpu, D_out]
        send_buffer = torch.stack([chunks[i] for i in range(world_size)], dim=0)
        recv_buffer = torch.empty((world_size, B * T_per_gpu, D_out), dtype=out.dtype, device=out.device)
        
        dist.all_to_all_single(recv_buffer, send_buffer)
        
        # Each GPU now has one chunk from each GPU (including its own)
        # For sequence parallel, we typically want the chunk corresponding to this GPU's sequence range
        out = recv_buffer[rank].view(B, T_per_gpu, D_out)  # [B, T_per_gpu, D_out]
    else:
        # Standard: all GPUs have full output
        pass

    return out
