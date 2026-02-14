# Multi-Head Attention Dispatch: AllToAll Head Routing
# Routes QKV tensors to GPUs based on which heads each GPU handles.
# This is the "dispatch" phase only — no attention computation.

import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    q: torch.Tensor,                  # Query tensor. Shape: [B, T, num_heads, head_dim] or [B*T, num_heads, head_dim].
                                      # Contiguous CUDA tensor.
    k: torch.Tensor,                  # Key tensor. Same shape as q.
    v: torch.Tensor,                  # Value tensor. Same shape as q.
    head_assignments: torch.Tensor,   # Per-head GPU assignment. Shape: [num_heads], dtype long.
                                      # Values in [0, world_size). head_assignments[h] = GPU rank that handles head h.
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:  # Returns: (q_dispatched, k_dispatched, v_dispatched)
    """
    MHA dispatch phase: route QKV head slices to GPUs via AllToAll (Ulysses-style).

    ASSUMPTION: QKV is REPLICATED across all GPUs initially (standard in Ulysses).
    Each GPU starts with the full QKV for ALL heads, then heads are distributed.

    This performs ONLY the dispatch step:
        1. SPLIT BY HEADS — Extract head slices from QKV tensors (each GPU has all heads initially).
        2. GROUP BY GPU   — Group head slices by destination GPU rank.
        3. DISPATCH       — AllToAll sends head slices to destination GPUs.

    After dispatch, each GPU has QKV for only the heads it's responsible for.

    Example (num_heads=8, world_size=4, heads_per_gpu=2):
        Initial state (all GPUs identical):
            GPU 0-3: QKV with all 8 heads [B, T, 8, head_dim]
        
        After dispatch:
            GPU 0: QKV with heads [0, 1] → [B, T, 2, head_dim]
            GPU 1: QKV with heads [2, 3] → [B, T, 2, head_dim]
            GPU 2: QKV with heads [4, 5] → [B, T, 2, head_dim]
            GPU 3: QKV with heads [6, 7] → [B, T, 2, head_dim]

    Returns:
        - q_dispatched: [B, T, num_heads_local, head_dim] — Q heads received on this GPU.
        - k_dispatched: [B, T, num_heads_local, head_dim] — K heads received on this GPU.
        - v_dispatched: [B, T, num_heads_local, head_dim] — V heads received on this GPU.

    Preconditions:
        - torch.distributed initialized with NCCL backend.
        - q, k, v must have the same shape.
        - head_assignments values in [0, world_size).
        - All ranks must have the SAME input shapes (QKV replicated).
        - QKV contains ALL heads initially (will be split during dispatch).
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"
    assert q.is_cuda and q.is_contiguous(), "q must be contiguous CUDA tensor"
    assert k.is_cuda and k.is_contiguous(), "k must be contiguous CUDA tensor"
    assert v.is_cuda and v.is_contiguous(), "v must be contiguous CUDA tensor"
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
        q_flat = q.view(B * T, num_heads, head_dim)
        k_flat = k.view(B * T, num_heads, head_dim)
        v_flat = v.view(B * T, num_heads, head_dim)
        original_shape = (B, T)
    elif q.dim() == 3:
        BT, num_heads_in, head_dim = q.shape
        assert num_heads_in == num_heads, f"q shape mismatch: num_heads={num_heads_in}, head_assignments={num_heads}"
        q_flat = q
        k_flat = k
        v_flat = v
        original_shape = (BT,)
    else:
        raise ValueError(f"q must be 3D or 4D, got shape {q.shape}")

    BT = q_flat.shape[0]

    # ──────────────────────────── SPLIT BY HEADS ──────────────────────────────
    # Extract head slices: [BT, num_heads, head_dim] -> list of [BT, head_dim] per head
    # Then group by destination GPU
    head_slices_q = [q_flat[:, h, :] for h in range(num_heads)]  # Each: [BT, head_dim]
    head_slices_k = [k_flat[:, h, :] for h in range(num_heads)]
    head_slices_v = [v_flat[:, h, :] for h in range(num_heads)]

    # ──────────────────────────── GROUP BY GPU ──────────────────────────────
    # Count how many heads each GPU receives
    heads_per_gpu = torch.bincount(head_assignments, minlength=world_size).to(
        device=q.device, dtype=torch.long
    )
    num_heads_local = int(heads_per_gpu[rank].item())

    # Pack head slices grouped by destination GPU
    # We'll create one tensor per GPU containing all its heads: [BT, num_heads_for_gpu, head_dim]
    q_packed_list = []
    k_packed_list = []
    v_packed_list = []
    send_counts = []

    for gpu_rank in range(world_size):
        # Find heads assigned to this GPU
        head_mask = head_assignments == gpu_rank
        head_indices = torch.where(head_mask)[0]

        if len(head_indices) == 0:
            send_counts.append(0)
            continue

        # Stack head slices for this GPU: [BT, num_heads_for_gpu, head_dim]
        q_gpu = torch.stack([head_slices_q[h] for h in head_indices], dim=1)  # [BT, num_heads_for_gpu, head_dim]
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

    # Concatenate all packed tensors: [total_tokens, head_dim]
    q_packed = torch.cat(q_packed_list, dim=0) if q_packed_list else torch.empty((0, head_dim), dtype=q.dtype, device=q.device)
    k_packed = torch.cat(k_packed_list, dim=0) if k_packed_list else torch.empty((0, head_dim), dtype=k.dtype, device=k.device)
    v_packed = torch.cat(v_packed_list, dim=0) if v_packed_list else torch.empty((0, head_dim), dtype=v.dtype, device=v.device)

    # ──────────────────────────── DISPATCH ────────────────────────────
    # Exchange counts
    recv_counts = torch.empty(world_size, dtype=torch.long, device=q.device)
    dist.all_to_all_single(recv_counts, torch.tensor(send_counts, dtype=torch.long, device=q.device))

    # AllToAll QKV
    total_recv = int(recv_counts.sum().item())
    q_dispatched_flat = torch.empty((total_recv, head_dim), dtype=q.dtype, device=q.device)
    k_dispatched_flat = torch.empty((total_recv, head_dim), dtype=k.dtype, device=k.device)
    v_dispatched_flat = torch.empty((total_recv, head_dim), dtype=v.dtype, device=v.device)

    dist.all_to_all_single(
        q_dispatched_flat, q_packed,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts,
    )
    dist.all_to_all_single(
        k_dispatched_flat, k_packed,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts,
    )
    dist.all_to_all_single(
        v_dispatched_flat, v_packed,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts,
    )

    # ──────────────────────────── RESHAPE BACK ──────────────────────────────
    # Reshape from [BT * num_heads_local, head_dim] to [B, T, num_heads_local, head_dim] or [BT, num_heads_local, head_dim]
    if len(original_shape) == 2:
        B, T = original_shape
        q_dispatched = q_dispatched_flat.view(B, T, num_heads_local, head_dim)
        k_dispatched = k_dispatched_flat.view(B, T, num_heads_local, head_dim)
        v_dispatched = v_dispatched_flat.view(B, T, num_heads_local, head_dim)
    else:
        BT = original_shape[0]
        q_dispatched = q_dispatched_flat.view(BT, num_heads_local, head_dim)
        k_dispatched = k_dispatched_flat.view(BT, num_heads_local, head_dim)
        v_dispatched = v_dispatched_flat.view(BT, num_heads_local, head_dim)

    return q_dispatched, k_dispatched, v_dispatched
