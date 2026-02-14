# MoE Combine: AllToAll Token Output Routing Back to Original Ranks
# Unified combine implementation supporting top-K weighted combination and capacity handling.
# This is the "combine" phase only — assumes tokens have already been computed by experts.

import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    x: torch.Tensor,  # Expert-computed token outputs on this rank. Shape: [T_local, D_out].
                      # These are outputs from expert computation (already processed).
                      # NOTE: For all-to-all, per-rank token counts may differ.
    dst_ranks: torch.Tensor,    # For each local output row i, which rank should receive it.
                               # Shape: [T_local], dtype torch.long, values in [0, world_size).
    dst_indices: torch.Tensor,  # For each local output row i, which row index to place it at on dst_ranks[i].
                               # Shape: [T_local], dtype torch.long.
                               # For top_k > 1: indices may repeat (same token, different expert outputs).
    out_len: int,               # Number of tokens this rank expects to receive (size of output sequence).
    top_k: int = 1,             # Number of experts each original token routed to. 1 = top-1, 2 = top-2, etc.
    gate_weights: torch.Tensor | None = None,  # Per-token gating weights for combining (required if top_k > 1).
                               # Shape: [out_len, top_k], dtype float. Typically sums to 1 per token.
    capacity_factor: float | None = None,  # Per-expert capacity multiplier used during dispatch.
                               # None = no capacity limit was enforced.
                               # If set, some tokens may have been dropped (will have zero output).
) -> torch.Tensor:
    """
    MoE combine phase: route expert outputs back to original ranks via AllToAll.

    This performs ONLY the combine step (assumes tokens have already been computed):
        1. DISPATCH — AllToAll sends computed outputs to destination ranks.
        2. RESTORE ORDER — Use dst_indices to place outputs in correct positions.
        3. WEIGHTED SUM — (if top_k > 1) Combine K expert outputs per token.

    Preconditions:
        - torch.distributed initialized with NCCL backend.
        - If top_k > 1: gate_weights must be provided and match out_len.
        - dst_ranks and dst_indices must correspond to the routing used in dispatch.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert top_k >= 1, "top_k must be >= 1"
    assert x.is_cuda and x.is_contiguous(), "x must be contiguous CUDA tensor"
    assert dst_ranks.is_cuda and dst_ranks.is_contiguous(), "dst_ranks must be contiguous CUDA tensor"
    assert dst_indices.is_cuda and dst_indices.is_contiguous(), "dst_indices must be contiguous CUDA tensor"
    assert dst_ranks.dtype == torch.long, "dst_ranks must be torch.long"
    assert dst_indices.dtype == torch.long, "dst_indices must be torch.long"

    if top_k > 1:
        assert gate_weights is not None, f"gate_weights required for top_k={top_k}"
        assert gate_weights.shape == (out_len, top_k), \
            f"gate_weights shape mismatch: got {gate_weights.shape}, expected [out_len, {top_k}]"
        assert gate_weights.is_cuda, "gate_weights must be CUDA tensor"

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    T_local, D_out = x.shape

    # x is already the computed expert outputs (no computation needed here)
    local_out = x.contiguous()



    # ---- AllToAll setup (variable number of tokens per destination rank) ----
    # 1) Count how many token outputs we are sending to each rank, EXCHANGE "recv_counts" BETWEEN  RANKS
    send_counts = torch.bincount(dst_ranks, minlength=world_size).to(device=x.device, dtype=torch.long)
    recv_counts = torch.empty((world_size,), device=x.device, dtype=torch.long)
    dist.all_to_all_single(recv_counts, send_counts)  # exchange counts


    # 2) Pack outputs (and indices) grouped by destination rank
    # Sort by dst_ranks so tokens for the same destination become contiguous
    perm = torch.argsort(dst_ranks)
    packed_out = local_out.index_select(0, perm).contiguous()
    packed_idx = dst_indices.index_select(0, perm).contiguous()

    # TWO alltoalls needed: one to exchange the outputs and another to exchange the actual indices.
    # 3) AllToAll the packed outputs
    total_recv = int(recv_counts.sum().item())
    D_out = packed_out.shape[1]
    recv_out = torch.empty((total_recv, D_out), device=x.device, dtype=packed_out.dtype)
    dist.all_to_all_single(
        recv_out,
        packed_out,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
    )
    # 4) AllToAll the destination indices (same split sizes as outputs)
    recv_idx = torch.empty((total_recv,), device=x.device, dtype=torch.long)
    dist.all_to_all_single(
        recv_idx,
        packed_idx,
        output_split_sizes=recv_counts.tolist(),
        input_split_sizes=send_counts.tolist(),
    )



    # 5) Restore per-rank output order and handle top-K weighted combination
    out = torch.zeros((out_len, D_out), device=x.device, dtype=recv_out.dtype)
    
    if top_k == 1:
        # Simple case: one output per token, direct placement
        # Use index_copy_ to handle potential duplicates (shouldn't happen in top-1, but be safe)
        # If capacity dropped tokens, missing positions remain zero (already initialized)
        out.index_copy_(0, recv_idx, recv_out)
    else:
        # Top-K case: multiple outputs per token, need to group by token then weight
        # For top_k > 1, we expect each token to have top_k outputs (one per expert)
        # dst_indices tells us which token each output belongs to
        
        # Group outputs by token index: for each token i, collect its top_k expert outputs
        # Then apply weighted sum: sum(weight[k] * output[k] for k in range(top_k))
        
        # Sort by dst_index to group outputs for the same token together
        sort_idx = torch.argsort(recv_idx)
        sorted_out = recv_out[sort_idx]
        sorted_dst_idx = recv_idx[sort_idx]
        
        # Count outputs per token
        counts_per_token = torch.bincount(recv_idx, minlength=out_len).to(device=x.device, dtype=torch.long)
        
        # For each token, collect its outputs and apply weighted sum
        for token_idx in range(out_len):
            count = counts_per_token[token_idx].item()
            if count == 0:
                continue  # Token was dropped, already zero
            elif count == top_k:
                # Perfect case: all k experts sent outputs
                token_mask = sorted_dst_idx == token_idx
                token_outputs = sorted_out[token_mask]  # [top_k, D_out]
                token_weights = gate_weights[token_idx]  # [top_k]
                out[token_idx] = (token_outputs * token_weights.unsqueeze(-1)).sum(dim=0)
            else:
                # Partial: some experts dropped (capacity)
                # Use available outputs, renormalize weights for available experts only
                token_mask = sorted_dst_idx == token_idx
                token_outputs = sorted_out[token_mask]  # [count, D_out]
                available_weights = gate_weights[token_idx][:count]  # [count]
                # Renormalize available weights to sum to 1
                available_weights = available_weights / available_weights.sum()
                out[token_idx] = (token_outputs * available_weights.unsqueeze(-1)).sum(dim=0)
    
    return out

