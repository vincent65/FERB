"""
Optional per-problem input override registry.

Use this for irregular signatures that need richer synthetic data than the
default `create_input_tensor` logic.
"""

from __future__ import annotations

from typing import Callable


InputOverride = Callable[..., tuple]
_INPUT_OVERRIDES: dict[int, InputOverride] = {}


def register_input_override(problem_id: int):
    def decorator(fn: InputOverride) -> InputOverride:
        _INPUT_OVERRIDES[problem_id] = fn
        return fn

    return decorator


def get_problem_input_override(problem_id: int) -> InputOverride | None:
    return _INPUT_OVERRIDES.get(problem_id)


# Add overrides below as needed:

import torch


@register_input_override(50)
def make_inputs_problem_50(
    *,
    rank: int,
    world_size: int,
    problem_id: int,
    base_shape: tuple,
    dtype: torch.dtype,
    device,
) -> tuple:
    """
    MoE Combine: AllToAll Token Output Routing Back to Original Ranks.

    solution(x, dst_ranks, dst_indices, out_len, top_k=1, gate_weights=None, capacity_factor=None)

    We generate synthetic routing data that simulates the dispatch phase output:
      - x:           [T_local, D_out]  expert-computed outputs on this rank
      - dst_ranks:   [T_local]         which rank each output should be sent to
      - dst_indices:  [T_local]         row index on the destination rank
      - out_len:     int               number of output tokens this rank expects
      - top_k:       int               experts per token (we use 1 for the reference)
    """
    T_local, D_out = base_shape          # base_shape = (rows, cols) from the YAML
    top_k = 1                            # simple top-1 routing

    # Use a *global* seed so every rank generates the same routing table,
    # then each rank picks only its own slice.  This ensures dst_indices are
    # globally unique per destination rank (no collisions after all-to-all).
    torch.manual_seed(42 + problem_id * 1000)

    # Total tokens across all ranks
    T_total = T_local * world_size

    # Generate a global routing: for every token on every rank, pick a
    # destination rank uniformly at random.
    all_dst_ranks = torch.randint(0, world_size, (T_total,), dtype=torch.long)

    # Assign globally-unique dst_indices per destination rank.
    all_dst_indices = torch.zeros(T_total, dtype=torch.long)
    for r in range(world_size):
        mask = all_dst_ranks == r
        count = int(mask.sum().item())
        all_dst_indices[mask] = torch.arange(count, dtype=torch.long)

    # Each rank's out_len = number of tokens routed TO this rank from all ranks
    out_len = int((all_dst_ranks == rank).sum().item())

    # Slice out this rank's portion
    start = rank * T_local
    end = start + T_local
    dst_ranks_local = all_dst_ranks[start:end].to(device=device)
    dst_indices_local = all_dst_indices[start:end].to(device=device)

    # Now generate per-rank random data (rank-specific seed for x)
    torch.manual_seed(42 + problem_id * 1000 + rank)
    x = torch.randn((T_local, D_out), dtype=dtype, device=device)

    return (x, dst_ranks_local, dst_indices_local, out_len)
