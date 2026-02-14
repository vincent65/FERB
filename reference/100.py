# Full MoE: AllToAll Dispatch + Expert Compute + AllToAll Combine
# Unified implementation supporting top-K routing and capacity factor.
# Each rank hosts one expert. Routing skew is parameterized via expert_assignments.

import math
import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    x: torch.Tensor,                  # Original tokens on this rank. Shape: [T, D]. Contiguous CUDA.
                                      # All ranks must have the SAME T (pad if needed).
    expert_assignments: torch.Tensor,  # Per-token expert assignments.
                                      #   If top_k=1: Shape [T], dtype long, values in [0, world_size).
                                      #   If top_k>1: Shape [T, top_k], dtype long, values in [0, world_size).
                                      #   Caller controls routing skew.
    w1: torch.Tensor,                  # This rank's expert first weight. Shape: [D_hidden, D].
    w2: torch.Tensor,                  # This rank's expert second weight. Shape: [D, D_hidden].
    top_k: int = 1,                    # Number of experts each token routes to. 1 = top-1, 2 = top-2, etc.
    gate_weights: torch.Tensor | None = None,  # Per-token gating weights for combining (required if top_k > 1).
                                      #   Shape: [T, top_k], dtype float. Typically sums to 1 per token.
    capacity_factor: float | None = None,  # Per-expert capacity multiplier. None = no capacity limit.
                                      #   expert_capacity = ceil(T * capacity_factor).
                                      #   1.0 = perfectly-balanced budget, >1.0 = slack for imbalance.
) -> torch.Tensor:                    # Returns: [T, D] — each token processed by its assigned expert(s).
    """
    Full MoE forward with NCCL AllToAll, supporting top-K routing and capacity limits.

    Flow:
        1. EXPAND (if top_k > 1) — Duplicate each token for all K expert assignments.
        2. DISPATCH              — AllToAll sends tokens to expert ranks.
        3. CAPACITY (if enabled) — Truncate received tokens to expert_capacity.
        4. COMPUTE               — Each rank runs: GELU(x @ W1^T) @ W2^T
        5. COMBINE               — AllToAll sends results back.
        6. WEIGHTED SUM (if top_k > 1) — Combine K expert outputs per token.

    Preconditions:
        - torch.distributed initialized with NCCL backend.
        - *** One expert per rank. expert_assignments values index ranks [0, world_size).
        - All ranks have the same T (token count). Pad shorter sequences if needed.
        - If top_k > 1: gate_weights must be provided and match expert_assignments shape.
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
        assert expert_assignments.shape[1] == top_k, f"expert_assignments shape mismatch: got {expert_assignments.shape}, expected [T, {top_k}]"
        assert gate_weights is not None, f"gate_weights required for top_k={top_k}"
        assert gate_weights.shape == (T, top_k), f"gate_weights shape mismatch: got {gate_weights.shape}, expected [T, {top_k}]"

        # Expand: [T, D] -> [T*top_k, D] by repeating each token top_k times
        x_dispatch = x.repeat_interleave(top_k, dim=0)  # [T*top_k, D]
        assignments_flat = expert_assignments.flatten()  # [T*top_k]
        T_dispatch = T * top_k

    # ──────────────────────────── DISPATCH ────────────────────────────
    # 1) Count how many tokens this rank sends to each expert/rank
    send_counts = torch.bincount(assignments_flat, minlength=world_size).to(
        device=x.device, dtype=torch.long
    )

    # 2) Exchange counts so every rank knows how many tokens it will receive
    recv_counts = torch.empty(world_size, dtype=torch.long, device=x.device)
    dist.all_to_all_single(recv_counts, send_counts)

    # 3) Sort tokens by destination rank so each rank's payload is contiguous
    perm = torch.argsort(assignments_flat)
    packed_x = x_dispatch.index_select(0, perm).contiguous()

    # 4) AllToAll the packed token payloads
    total_recv = int(recv_counts.sum().item())
    recv_x = torch.empty((total_recv, D), dtype=x.dtype, device=x.device)
    dist.all_to_all_single(
        recv_x, packed_x,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
    )

    # ────────────────── CAPACITY ENFORCEMENT ────────────────────────────
    if capacity_factor is not None:
        expert_capacity = int(math.ceil(T * capacity_factor))
        num_to_compute = min(total_recv, expert_capacity)
        # We'll zero out the rest after compute
    else:
        num_to_compute = total_recv
        expert_capacity = None

    # ──────────────────────────── COMPUTE ─────────────────────────────
    # Fixed compute: matmul(w1) -> gelu -> matmul(w2)
    if num_to_compute > 0:
        active = recv_x[:num_to_compute]
        h = torch.matmul(active, w1.t())          # [num_to_compute, D_hidden]
        h = torch.nn.functional.gelu(h)            # [num_to_compute, D_hidden]
        computed_active = torch.matmul(h, w2.t())  # [num_to_compute, D]
    else:
        computed_active = torch.empty((0, D), dtype=x.dtype, device=x.device)

    # If capacity was enforced, pad with zeros for dropped tokens
    if capacity_factor is not None and total_recv > expert_capacity:
        computed = torch.zeros((total_recv, D), dtype=x.dtype, device=x.device)
        computed[:num_to_compute] = computed_active
    else:
        computed = computed_active

    # ──────────────────────────── COMBINE ─────────────────────────────
    # Reverse AllToAll: send results back to originating ranks.
    # send/recv counts are swapped relative to dispatch.
    result_packed = torch.empty((T_dispatch, D), dtype=x.dtype, device=x.device)
    dist.all_to_all_single(
        result_packed, computed,
        output_split_sizes=send_counts.tolist(),
        input_split_sizes=recv_counts.tolist(),
    )

    # Restore original dispatch order (inverse of the dispatch permutation)
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
