# MoE Dispatch: AllToAll Token Routing
# Unified dispatch implementation supporting top-K routing and expert ID mapping.
# This is the "dispatch" phase only — no compute, no combine.

import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    x: torch.Tensor,                  # Original tokens on this rank. Shape: [T, D]. Contiguous CUDA.
                                      # All ranks must have the SAME T (pad if needed).
    expert_assignments: torch.Tensor,  # Per-token expert assignments.
                                      #   If top_k=1: Shape [T], dtype long.
                                      #   If top_k>1: Shape [T, top_k], dtype long.
                                      #   Values interpretation depends on experts_per_gpu:
                                      #     - If experts_per_gpu=None: direct GPU ranks [0, world_size).
                                      #     - If experts_per_gpu=int: expert IDs [0, num_experts).
    top_k: int = 1,                    # Number of experts each token routes to. 1 = top-1, 2 = top-2, etc.
    experts_per_gpu: int | None = None,  # Number of experts per GPU. None = direct GPU mapping.
                                      #   If None: expert_assignments are GPU ranks [0, world_size).
                                      #   If int: expert_assignments are expert IDs [0, world_size * experts_per_gpu).
) -> tuple[torch.Tensor, torch.Tensor]:  # Returns: (dispatched_tokens, local_expert_ids)
    """
    MoE dispatch phase: route tokens to expert GPUs via AllToAll.

    This performs ONLY the dispatch step:
        1. EXPAND (if top_k > 1) — Duplicate each token for all K expert assignments.
        2. MAP (if experts_per_gpu) — Convert expert IDs to GPU ranks and local expert IDs.
        3. DISPATCH — AllToAll sends tokens to destination GPU ranks.

    Returns:
        - dispatched_tokens: [total_recv, D] — tokens received on this rank.
        - local_expert_ids: [total_recv] — which local expert each token belongs to.
                                      Values in [0, experts_per_gpu) or [0, 1) if experts_per_gpu=None.

    Preconditions:
        - torch.distributed initialized with NCCL backend.
        - All ranks have the same T (token count).
        - If experts_per_gpu is not None: num_experts = world_size * experts_per_gpu.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert top_k >= 1, "top_k must be >= 1"

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    T, D = x.shape

    # ──────────────────────────── EXPAND (if top_k > 1) ──────────────────────────────
    if top_k == 1:
        # Simple case: no expansion needed
        x_dispatch = x.contiguous()
        assignments_flat = expert_assignments.contiguous()  # [T]
        T_dispatch = T
    else:
        # Multi-expert routing: duplicate each token top_k times
        assert expert_assignments.dim() == 2, f"expert_assignments must be [T, top_k] for top_k={top_k}"
        assert expert_assignments.shape[1] == top_k, \
            f"expert_assignments shape mismatch: got {expert_assignments.shape}, expected [T, {top_k}]"

        # Expand: [T, D] -> [T*top_k, D] by repeating each token top_k times
        x_dispatch = x.repeat_interleave(top_k, dim=0)  # [T*top_k, D]
        assignments_flat = expert_assignments.flatten()  # [T*top_k]
        T_dispatch = T * top_k

    # ──────────────────────────── MAP (if experts_per_gpu) ──────────────────────────────
    if experts_per_gpu is None:
        # Direct GPU mapping: expert_assignments are already GPU ranks
        gpu_assignments = assignments_flat  # [T_dispatch], values in [0, world_size)
        local_expert_ids_dispatch = torch.zeros(T_dispatch, dtype=torch.long, device=x.device)  # All 0 (single expert per GPU)
    else:
        # Expert ID mapping: convert expert IDs to GPU ranks and local expert IDs
        num_experts = world_size * experts_per_gpu
        assert assignments_flat.max() < num_experts, \
            f"expert_assignments values must be < {num_experts} (world_size * experts_per_gpu)"
        assert assignments_flat.min() >= 0, "expert_assignments values must be >= 0"

        gpu_assignments = assignments_flat // experts_per_gpu  # [T_dispatch], values in [0, world_size)
        local_expert_ids_dispatch = assignments_flat % experts_per_gpu  # [T_dispatch], values in [0, experts_per_gpu)

    # ──────────────────────────── DISPATCH ────────────────────────────
    # 1) Count how many tokens this rank sends to each GPU
    send_counts = torch.bincount(gpu_assignments, minlength=world_size).to(
        device=x.device, dtype=torch.long
    )

    # 2) Exchange counts so every rank knows how many tokens it will receive
    recv_counts = torch.empty(world_size, dtype=torch.long, device=x.device)
    dist.all_to_all_single(recv_counts, send_counts)

    # 3) Sort tokens by destination GPU so each GPU's payload is contiguous
    perm = torch.argsort(gpu_assignments)
    packed_x = x_dispatch.index_select(0, perm).contiguous()

    # 4) AllToAll the packed token payloads
    total_recv = int(recv_counts.sum().item())
    dispatched_tokens = torch.empty((total_recv, D), dtype=x.dtype, device=x.device)
    dist.all_to_all_single(
        dispatched_tokens, packed_x,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
    )

    # 5) AllToAll the local expert IDs (so receiving ranks know which local expert each token belongs to)
    if experts_per_gpu is None:
        # Single expert per GPU: all local_expert_ids are 0, no need to send
        local_expert_ids = torch.zeros(total_recv, dtype=torch.long, device=x.device)
    else:
        packed_local_expert_ids = local_expert_ids_dispatch.index_select(0, perm).contiguous()
        local_expert_ids = torch.empty((total_recv,), dtype=torch.long, device=x.device)
        dist.all_to_all_single(
            local_expert_ids, packed_local_expert_ids,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
        )

    return dispatched_tokens, local_expert_ids
