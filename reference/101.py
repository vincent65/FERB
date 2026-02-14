# EP + TP: Expert-Parallel Dispatch + Tensor-Parallel Expert Compute
# Expert weights are sharded across a TP group (column-parallel W_in, row-parallel W_out).
# Dispatch/combine use AllToAll within EP groups; expert compute uses AllReduce within TP groups.
# Parameterized: tp_size controls the tensor-parallel degree.

import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    x: torch.Tensor,                  # Original tokens on this rank. Shape: [T, D]. Contiguous CUDA.
                                      # Must be IDENTICAL across all TP peers (replicated input).
                                      # All EP peers must have the same T.
    expert_assignments: torch.Tensor,  # Per-token expert assignment. Shape: [T], dtype long.
                                      # Values in [0, ep_size) where ep_size = world_size // tp_size.
                                      # Must be IDENTICAL across all TP peers (deterministic routing).
    w1_shard: torch.Tensor,           # Column shard of this expert's first weight.
                                      # Shape: [D_hidden // tp_size, D].
    w2_shard: torch.Tensor,           # Row shard of this expert's second weight.
                                      # Shape: [D, D_hidden // tp_size].
    tp_size: int = 2,                 # Tensor-parallel degree within each expert.
                                      # world_size must be divisible by tp_size.
                                      # ep_size (number of experts) = world_size // tp_size.
) -> torch.Tensor:                    # Returns: [T, D].
    """
    MoE forward with Expert Parallelism + Tensor Parallelism.

    Layout (example: world_size=8, tp_size=2, ep_size=4):
        Expert 0: ranks [0, 1]  (TP group)
        Expert 1: ranks [2, 3]
        Expert 2: ranks [4, 5]
        Expert 3: ranks [6, 7]

        EP group A: ranks [0, 2, 4, 6]  (TP position 0)
        EP group B: ranks [1, 3, 5, 7]  (TP position 1)

    Flow:
        1. EP DISPATCH  — AllToAll within EP group sends tokens to expert ranks.
                          Since all TP peers have the same input + routing, each TP peer
                          independently dispatches and receives the same tokens for its expert.
        2. TP COMPUTE   — Each TP peer computes with its weight shard:
                            h = x @ W1_shard^T   →  GELU  →  partial = h @ W2_shard^T
                          AllReduce within TP group sums the partial outputs.
        3. EP COMBINE   — AllToAll within EP group sends results back.

    Preconditions:
        - torch.distributed initialized with NCCL backend.
        - world_size divisible by tp_size.
        - x and expert_assignments must be identical across TP peers.
        - expert_assignments values in [0, ep_size).

    NOTE: Process groups are created inside this function for clarity.
          In production, cache them outside the hot path.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size % tp_size == 0, f"world_size ({world_size}) must be divisible by tp_size ({tp_size})"

    ep_size = world_size // tp_size
    tp_rank = rank % tp_size       # position within TP group
    ep_rank = rank // tp_size      # which expert this rank belongs to

    T, D = x.shape

    # ──────────────── CREATE PROCESS GROUPS ───────────────────────────
    # All ranks must participate in every new_group() call (it's collective).

    # EP groups: ranks with the same tp_rank across all experts
    my_ep_group = None
    for t in range(tp_size):
        ranks = list(range(t, world_size, tp_size))  # e.g. [0, 2, 4, 6] for t=0
        g = dist.new_group(ranks)
        if t == tp_rank:
            my_ep_group = g

    # TP groups: ranks within the same expert
    my_tp_group = None
    for e in range(ep_size):
        ranks = list(range(e * tp_size, (e + 1) * tp_size))  # e.g. [0, 1] for e=0
        g = dist.new_group(ranks)
        if e == ep_rank:
            my_tp_group = g

    # ──────────────────────── EP DISPATCH ─────────────────────────────
    # AllToAll within this rank's EP group (ep_size peers)
    send_counts = torch.bincount(expert_assignments, minlength=ep_size).to(
        device=x.device, dtype=torch.long
    )
    recv_counts = torch.empty(ep_size, dtype=torch.long, device=x.device)
    dist.all_to_all_single(recv_counts, send_counts, group=my_ep_group)

    perm = torch.argsort(expert_assignments)
    packed_x = x.index_select(0, perm).contiguous()

    total_recv = int(recv_counts.sum().item())
    recv_x = torch.empty((total_recv, D), dtype=x.dtype, device=x.device)
    dist.all_to_all_single(
        recv_x, packed_x,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
        group=my_ep_group,
    )

    # ──────────────────────── TP COMPUTE ──────────────────────────────
    # Column-parallel first layer: each TP peer computes a slice of the hidden dim
    h = torch.matmul(recv_x, w1_shard.t())    # [total_recv, D_hidden // tp_size]
    h = torch.nn.functional.gelu(h)

    # Row-parallel second layer: each TP peer produces a partial output in full D space
    partial = torch.matmul(h, w2_shard.t())    # [total_recv, D]

    # AllReduce within TP group to combine partial outputs
    dist.all_reduce(partial, op=dist.ReduceOp.SUM, group=my_tp_group)

    # ──────────────────────── EP COMBINE ──────────────────────────────
    result_packed = torch.empty((T, D), dtype=x.dtype, device=x.device)
    dist.all_to_all_single(
        result_packed, partial,
        output_split_sizes=send_counts.tolist(),
        input_split_sizes=recv_counts.tolist(),
        group=my_ep_group,
    )

    # Restore original token order
    inv_perm = torch.argsort(perm)
    out = result_packed.index_select(0, inv_perm)
    return out
