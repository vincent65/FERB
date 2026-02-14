# End-to-End Ulysses Attention: Dispatch + Compute + Combine
# Full multi-head attention with head parallelism via AllToAll.
# QKV is replicated initially, heads are distributed, attention computed, then gathered back.

import math
import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    q: torch.Tensor,                  # Query tensor. Shape: [B, T, num_heads, head_dim] or [B*T, num_heads, head_dim].
                                      # REPLICATED across all GPUs (all heads).
    k: torch.Tensor,                  # Key tensor. Same shape as q. REPLICATED across all GPUs.
    v: torch.Tensor,                  # Value tensor. Same shape as q. REPLICATED across all GPUs.
    head_assignments: torch.Tensor,   # Per-head GPU assignment. Shape: [num_heads], dtype long.
                                      # Values in [0, world_size). head_assignments[h] = GPU rank that handles head h.
    w_out: torch.Tensor,              # Output projection weight. Shape: [D_out, num_heads * head_dim] or [D_out, D].
                                      # Full weight matrix (not sharded). REPLICATED across all GPUs.
    scale: float | None = None,       # Attention scale factor. If None, uses 1/sqrt(head_dim).
    sequence_parallel: bool = False,  # If True, split output along sequence dimension after attention.
                                      # If False, standard Ulysses (all GPUs have full output).
) -> torch.Tensor:                    # Returns: [B, T, D_out] or [B, T_local, D_out] if sequence_parallel.
    """
    End-to-end Ulysses attention: dispatch heads → compute attention → gather heads → out projection.

    Full flow:
        1. DISPATCH HEADS    — AllToAll routes QKV head slices to GPUs (each GPU gets its assigned heads).
        2. COMPUTE ATTENTION — Q @ K^T / scale, softmax, @ V (on local heads only).
        3. GATHER HEADS      — AllToAll collects attention outputs from all GPUs.
        4. OUT PROJECTION    — Apply output projection weight.
        5. SEQUENCE SPLIT    — (if sequence_parallel) AllToAll splits along sequence dimension.

    Example (num_heads=8, world_size=4, heads_per_gpu=2):
        Initial state (all GPUs identical):
            GPU 0-3: QKV with all 8 heads [B, T, 8, head_dim]
        
        After dispatch:
            GPU 0: QKV with heads [0, 1] → [B, T, 2, head_dim]
            GPU 1: QKV with heads [2, 3] → [B, T, 2, head_dim]
            GPU 2: QKV with heads [4, 5] → [B, T, 2, head_dim]
            GPU 3: QKV with heads [6, 7] → [B, T, 2, head_dim]
        
        After attention + gather + out projection:
            All GPUs: [B, T, D_out] (full output)

    Preconditions:
        - torch.distributed initialized with NCCL backend.
        - q, k, v must have the same shape and be REPLICATED across all GPUs.
        - head_assignments values in [0, world_size).
        - All ranks must have the same input shapes.
        - w_out must be REPLICATED across all GPUs.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"
    assert q.is_cuda and q.is_contiguous(), "q must be contiguous CUDA tensor"
    assert k.is_cuda and k.is_contiguous(), "k must be contiguous CUDA tensor"
    assert v.is_cuda and v.is_contiguous(), "v must be contiguous CUDA tensor"
    assert w_out.is_cuda and w_out.is_contiguous(), "w_out must be contiguous CUDA tensor"
    assert head_assignments.dtype == torch.long, "head_assignments must be torch.long"
    assert head_assignments.is_cuda, "head_assignments must be CUDA tensor"

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    num_heads = head_assignments.shape[0]

    # Handle different input shapes: [B, T, num_heads, head_dim] or [B*T, num_heads, head_dim]
    if q.dim() == 4:
        B, T, num_heads_in, head_dim = q.shape
        assert num_heads_in == num_heads, f"q shape mismatch: num_heads={num_heads_in}, head_assignments={num_heads}"
        # Reshape to [B*T, num_heads, head_dim] for easier head-wise operations
        q_flat = q.view(-1, num_heads, head_dim)
        k_flat = k.view(-1, num_heads, head_dim)
        v_flat = v.view(-1, num_heads, head_dim)
        original_shape = (B, T)
        BT = B * T
    elif q.dim() == 3:
        BT, num_heads_in, head_dim = q.shape
        assert num_heads_in == num_heads, f"q shape mismatch: num_heads={num_heads_in}, head_assignments={num_heads}"
        q_flat = q
        k_flat = k
        v_flat = v
        original_shape = (BT,)
        B = 1  # Will infer from output shape
        T = BT
    else:
        raise ValueError(f"q must be 3D or 4D, got shape {q.shape}")

    # ──────────────────────────── DISPATCH HEADS ──────────────────────────────
    # Extract head slices and group by destination GPU
    head_slices_q = [q_flat[:, h, :] for h in range(num_heads)]  # Each: [BT, head_dim]
    head_slices_k = [k_flat[:, h, :] for h in range(num_heads)]
    head_slices_v = [v_flat[:, h, :] for h in range(num_heads)]

    # Count how many heads each GPU receives
    heads_per_gpu = torch.bincount(head_assignments, minlength=world_size).to(
        device=q.device, dtype=torch.long
    )
    num_heads_local = int(heads_per_gpu[rank].item())

    # Pack head slices grouped by destination GPU
    q_packed_list = []
    k_packed_list = []
    v_packed_list = []
    send_counts = []

    for gpu_rank in range(world_size):
        head_mask = head_assignments == gpu_rank
        head_indices = torch.where(head_mask)[0]

        if len(head_indices) == 0:
            send_counts.append(0)
            continue

        # Stack head slices for this GPU: [BT, num_heads_for_gpu, head_dim]
        q_gpu = torch.stack([head_slices_q[h] for h in head_indices], dim=1)
        k_gpu = torch.stack([head_slices_k[h] for h in head_indices], dim=1)
        v_gpu = torch.stack([head_slices_v[h] for h in head_indices], dim=1)

        # Flatten to [BT * num_heads_for_gpu, head_dim] for AllToAll
        num_heads_gpu = len(head_indices)
        q_gpu_flat = q_gpu.view(BT * num_heads_gpu, head_dim)
        k_gpu_flat = k_gpu.view(BT * num_heads_gpu, head_dim)
        v_gpu_flat = v_gpu.view(BT * num_heads_gpu, head_dim)

        q_packed_list.append(q_gpu_flat)
        k_packed_list.append(k_gpu_flat)
        v_packed_list.append(v_gpu_flat)
        send_counts.append(BT * num_heads_gpu)

    # Concatenate all packed tensors
    q_packed = torch.cat(q_packed_list, dim=0) if q_packed_list else torch.empty((0, head_dim), dtype=q.dtype, device=q.device)
    k_packed = torch.cat(k_packed_list, dim=0) if k_packed_list else torch.empty((0, head_dim), dtype=k.dtype, device=k.device)
    v_packed = torch.cat(v_packed_list, dim=0) if v_packed_list else torch.empty((0, head_dim), dtype=v.dtype, device=v.device)

    # AllToAll to dispatch heads
    recv_counts = torch.empty(world_size, dtype=torch.long, device=q.device)
    dist.all_to_all_single(recv_counts, torch.tensor(send_counts, dtype=torch.long, device=q.device))

    total_recv = int(recv_counts.sum().item())
    q_local_flat = torch.empty((total_recv, head_dim), dtype=q.dtype, device=q.device)
    k_local_flat = torch.empty((total_recv, head_dim), dtype=k.dtype, device=k.device)
    v_local_flat = torch.empty((total_recv, head_dim), dtype=v.dtype, device=v.device)

    dist.all_to_all_single(q_local_flat, q_packed, output_split_sizes=recv_counts.tolist(), input_split_sizes=send_counts)
    dist.all_to_all_single(k_local_flat, k_packed, output_split_sizes=recv_counts.tolist(), input_split_sizes=send_counts)
    dist.all_to_all_single(v_local_flat, v_packed, output_split_sizes=recv_counts.tolist(), input_split_sizes=send_counts)

    # Reshape to [B, T, num_heads_local, head_dim]
    if len(original_shape) == 2:
        q_local = q_local_flat.view(B, T, num_heads_local, head_dim)
        k_local = k_local_flat.view(B, T, num_heads_local, head_dim)
        v_local = v_local_flat.view(B, T, num_heads_local, head_dim)
    else:
        q_local = q_local_flat.view(BT, num_heads_local, head_dim)
        k_local = k_local_flat.view(BT, num_heads_local, head_dim)
        v_local = v_local_flat.view(BT, num_heads_local, head_dim)
        B = 1
        T = BT

    # ──────────────────────────── COMPUTE ATTENTION ──────────────────────────────
    # Reshape for batched matmul: [B, num_heads_local, T, head_dim]
    q_local = q_local.transpose(1, 2).contiguous() if q_local.dim() == 4 else q_local.unsqueeze(0).transpose(1, 2).contiguous()
    k_local = k_local.transpose(1, 2).contiguous() if k_local.dim() == 4 else k_local.unsqueeze(0).transpose(1, 2).contiguous()
    v_local = v_local.transpose(1, 2).contiguous() if v_local.dim() == 4 else v_local.unsqueeze(0).transpose(1, 2).contiguous()
    
    if q_local.dim() == 3:
        # Handle [BT, num_heads_local, head_dim] case
        q_local = q_local.unsqueeze(0)  # [1, num_heads_local, BT, head_dim]
        k_local = k_local.unsqueeze(0)
        v_local = v_local.unsqueeze(0)
        B = 1
        T = BT

    # Compute attention scores: Q @ K^T
    attn_scores = torch.matmul(q_local, k_local.transpose(-2, -1))  # [B, num_heads_local, T, T]

    # Scale
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    attn_scores = attn_scores * scale

    # Softmax
    attn_probs = torch.nn.functional.softmax(attn_scores, dim=-1)  # [B, num_heads_local, T, T]

    # Apply to values
    attn_out = torch.matmul(attn_probs, v_local)  # [B, num_heads_local, T, head_dim]

    # Reshape back: [B, T, num_heads_local, head_dim]
    attn_out = attn_out.transpose(1, 2).contiguous()  # [B, T, num_heads_local, head_dim]

    # ──────────────────────────── GATHER HEADS (AllToAll) ──────────────────────────────
    # Flatten: [B, T, num_heads_local, head_dim] -> [B*T * num_heads_local, head_dim]
    attn_out_flat = attn_out.view(B * T * num_heads_local, head_dim)

    # AllToAll to gather all heads
    send_counts_gather = (heads_per_gpu * B * T).tolist()
    recv_counts_gather = torch.empty(world_size, dtype=torch.long, device=q.device)
    dist.all_to_all_single(recv_counts_gather, torch.tensor(send_counts_gather, dtype=torch.long, device=q.device))

    total_recv_gather = int(recv_counts_gather.sum().item())
    attn_gathered_flat = torch.empty((total_recv_gather, head_dim), dtype=q.dtype, device=q.device)
    
    dist.all_to_all_single(
        attn_gathered_flat, attn_out_flat,
        output_split_sizes=recv_counts_gather.tolist(),
        input_split_sizes=send_counts_gather,
    )

    # Reshape: [B, T, num_heads, head_dim]
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
        T_per_gpu = T // world_size
        assert T % world_size == 0, f"T ({T}) must be divisible by world_size ({world_size}) for sequence parallelism"
        
        # Reshape: [B, T, D_out] -> [B*T, D_out]
        out_flat = out.view(B * T, -1)
        
        # Split into chunks: [B*T_per_gpu, D_out] per GPU
        chunks = torch.chunk(out_flat, world_size, dim=0)
        
        # AllToAll to distribute chunks
        D_out = out.shape[-1]
        send_buffer = torch.stack([chunks[i] for i in range(world_size)], dim=0)  # [world_size, B*T_per_gpu, D_out]
        recv_buffer = torch.empty((world_size, B * T_per_gpu, D_out), dtype=out.dtype, device=out.device)
        
        dist.all_to_all_single(recv_buffer, send_buffer)
        
        # Each GPU gets its sequence chunk
        out = recv_buffer[rank].view(B, T_per_gpu, D_out)  # [B, T_per_gpu, D_out]

    return out
