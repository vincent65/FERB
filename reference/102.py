# MoE with Multiple Experts per GPU (Expert Sharding)
# Each GPU hosts multiple experts. Tokens route to expert IDs, which map to GPUs.
# Common case: more experts than GPUs (e.g., 8 experts on 4 GPUs = 2 experts per GPU).

import math
import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    x: torch.Tensor,                  # Original tokens on this rank. Shape: [T, D]. Contiguous CUDA.
                                      # All ranks must have the SAME T (pad if needed).
    expert_assignments: torch.Tensor,  # Per-token expert ID assignment. Shape: [T] or [T, top_k], dtype long.
                                      # Values are expert IDs in [0, num_experts).
                                      # These map to GPU ranks via: gpu_rank = expert_id // experts_per_gpu.
    w1_list: list[torch.Tensor],      # List of first weights, one per local expert on this GPU.
                                      # len(w1_list) == experts_per_gpu.
                                      # Each w1: Shape [D_hidden, D].
    w2_list: list[torch.Tensor],      # List of second weights, one per local expert on this GPU.
                                      # len(w2_list) == experts_per_gpu.
                                      # Each w2: Shape [D, D_hidden].
    num_experts: int,                 # Total number of experts across all GPUs.
    experts_per_gpu: int,             # Number of experts hosted on each GPU.
                                      # num_experts == world_size * experts_per_gpu.
    top_k: int = 1,                    # Number of experts each token routes to.
    gate_weights: torch.Tensor | None = None,  # Per-token gating weights (required if top_k > 1).
                                      # Shape: [T, top_k], dtype float.
    capacity_factor: float | None = None,  # Per-expert capacity multiplier. None = no capacity limit.
) -> torch.Tensor:                    # Returns: [T, D].
    """
    MoE forward with multiple experts per GPU (expert sharding).

    Layout (example: num_experts=8, world_size=4, experts_per_gpu=2):
        GPU 0: experts [0, 1]
        GPU 1: experts [2, 3]
        GPU 2: experts [4, 5]
        GPU 3: experts [6, 7]

    Flow:
        1. MAP         — Convert expert IDs to GPU ranks: gpu_rank = expert_id // experts_per_gpu.
        2. EXPAND      — (if top_k > 1) Duplicate tokens for all K expert assignments.
        3. DISPATCH    — AllToAll sends tokens to GPU ranks (grouped by GPU, not by expert).
        4. SEPARATE    — On receiving GPU, separate tokens by local expert ID.
        5. CAPACITY    — (if enabled) Truncate per local expert to capacity.
        6. COMPUTE     — Run each local expert's MLP on its tokens.
        7. COMBINE     — AllToAll sends results back.
        8. WEIGHTED SUM — (if top_k > 1) Combine K expert outputs per token.

    Preconditions:
        - torch.distributed initialized with NCCL backend.
        - num_experts == world_size * experts_per_gpu.
        - expert_assignments values in [0, num_experts).
        - All ranks have the same T.
        - If top_k > 1: gate_weights must be provided.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert top_k >= 1, "top_k must be >= 1"
    assert num_experts == dist.get_world_size() * experts_per_gpu, \
        f"num_experts ({num_experts}) must equal world_size * experts_per_gpu ({dist.get_world_size()} * {experts_per_gpu})"
    assert len(w1_list) == experts_per_gpu, f"w1_list length ({len(w1_list)}) must equal experts_per_gpu ({experts_per_gpu})"
    assert len(w2_list) == experts_per_gpu, f"w2_list length ({len(w2_list)}) must equal experts_per_gpu ({experts_per_gpu})"

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    T, D = x.shape

    # ──────────────────────────── MAP & EXPAND ──────────────────────────────
    if top_k == 1:
        # Map expert IDs to GPU ranks
        expert_ids = expert_assignments  # [T]
        gpu_assignments = expert_ids // experts_per_gpu  # [T], values in [0, world_size)
        local_expert_ids = expert_ids % experts_per_gpu  # [T], values in [0, experts_per_gpu)
        x_dispatch = x.contiguous()
        T_dispatch = T
    else:
        # Multi-expert routing
        assert expert_assignments.dim() == 2, f"expert_assignments must be [T, top_k] for top_k={top_k}"
        assert expert_assignments.shape[1] == top_k, f"expert_assignments shape mismatch"
        assert gate_weights is not None, f"gate_weights required for top_k={top_k}"
        assert gate_weights.shape == (T, top_k), f"gate_weights shape mismatch"

        # Expand: duplicate each token top_k times
        x_dispatch = x.repeat_interleave(top_k, dim=0)  # [T*top_k, D]
        expert_ids_flat = expert_assignments.flatten()  # [T*top_k]
        gpu_assignments = expert_ids_flat // experts_per_gpu  # [T*top_k]
        local_expert_ids = expert_ids_flat % experts_per_gpu  # [T*top_k]
        T_dispatch = T * top_k

    # ──────────────────────────── DISPATCH ────────────────────────────
    # Dispatch tokens to GPU ranks (not expert IDs)
    send_counts = torch.bincount(gpu_assignments, minlength=world_size).to(
        device=x.device, dtype=torch.long
    )
    recv_counts = torch.empty(world_size, dtype=torch.long, device=x.device)
    dist.all_to_all_single(recv_counts, send_counts)

    perm = torch.argsort(gpu_assignments)
    packed_x = x_dispatch.index_select(0, perm).contiguous()
    packed_local_expert_ids = local_expert_ids.index_select(0, perm).contiguous()

    total_recv = int(recv_counts.sum().item())
    recv_x = torch.empty((total_recv, D), dtype=x.dtype, device=x.device)
    recv_local_expert_ids = torch.empty((total_recv,), dtype=torch.long, device=x.device)

    # AllToAll tokens and their local expert IDs
    dist.all_to_all_single(
        recv_x, packed_x,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
    )
    dist.all_to_all_single(
        recv_local_expert_ids, packed_local_expert_ids,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
    )

    # ──────────────────────────── SEPARATE BY LOCAL EXPERT ──────────────────────────────
    # Group received tokens by which local expert they belong to
    expert_capacity = None
    if capacity_factor is not None:
        expert_capacity = int(math.ceil(T * capacity_factor))

    # Initialize output tensor (will be filled per-expert)
    computed = torch.zeros((total_recv, D), dtype=x.dtype, device=x.device)

    for local_expert_idx in range(experts_per_gpu):
        # Find tokens assigned to this local expert
        mask = recv_local_expert_ids == local_expert_idx
        num_tokens_for_expert = mask.sum().item()

        if num_tokens_for_expert == 0:
            # No tokens for this expert, skip
            continue

        expert_tokens = recv_x[mask]  # [num_tokens_for_expert, D]

        # ────────────────── CAPACITY ENFORCEMENT ────────────────────────────
        if capacity_factor is not None:
            num_to_compute = min(num_tokens_for_expert, expert_capacity)
            if num_to_compute < num_tokens_for_expert:
                # Only compute first num_to_compute tokens
                active_tokens = expert_tokens[:num_to_compute]
            else:
                active_tokens = expert_tokens
        else:
            active_tokens = expert_tokens
            num_to_compute = num_tokens_for_expert

        # ──────────────────────────── COMPUTE ─────────────────────────────
        w1 = w1_list[local_expert_idx]
        w2 = w2_list[local_expert_idx]
        h = torch.matmul(active_tokens, w1.t())  # [num_to_compute, D_hidden]
        h = torch.nn.functional.gelu(h)
        expert_output = torch.matmul(h, w2.t())  # [num_to_compute, D]

        # Place results back in the same positions (preserving order)
        # If capacity was enforced, excess positions remain zero (already initialized)
        # Use advanced indexing to place results correctly
        mask_indices = torch.where(mask)[0]  # Get indices where mask is True
        computed[mask_indices[:num_to_compute]] = expert_output

    # ──────────────────────────── COMBINE ─────────────────────────────
    # Reverse AllToAll: send results back to originating ranks
    result_packed = torch.empty((T_dispatch, D), dtype=x.dtype, device=x.device)
    dist.all_to_all_single(
        result_packed, computed,
        output_split_sizes=send_counts.tolist(),
        input_split_sizes=recv_counts.tolist(),
    )

    # Restore original dispatch order
    inv_perm = torch.argsort(perm)
    result_ordered = result_packed.index_select(0, inv_perm)

    # ──────────────────────── WEIGHTED SUM (if top_k > 1) ────────────────────────────
    if top_k == 1:
        out = result_ordered  # [T, D]
    else:
        # Reshape: [T*top_k, D] -> [T, top_k, D]
        result_shaped = result_ordered.view(T, top_k, D)  # [T, top_k, D]
        # Weighted sum: [T, top_k, D] * [T, top_k, 1] -> [T, D]
        out = (result_shaped * gate_weights.unsqueeze(-1)).sum(dim=1)  # [T, D]

    return out
