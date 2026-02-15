import torch
import torch.distributed as dist
from typing import Optional

@torch.no_grad()
def solution(
    x: torch.Tensor,  # [T_local, D_out]
    dst_ranks: torch.Tensor,  # [T_local], long
    dst_indices: torch.Tensor,  # [T_local], long
    out_len: int,
    top_k: int = 1,
    gate_weights: Optional[torch.Tensor] = None,
    capacity_factor: Optional[float] = None,
) -> torch.Tensor:
    """
    Robust PyTorch implementation of MoE combine-phase semantics, focused on
    correctness. This implementation accepts multiple common gate_weights
    layouts and preserves device/dtype semantics.

    Supported gate_weights layouts:
      - None: treat all weights as 1.0 (equal contribution / direct placement for top_k==1)
      - 1D tensor of length T_local: per-candidate weights aligned with input x
      - 1D tensor of length R (number of locally-received rows after filtering dst_ranks): per-received-candidate weights
      - 2D tensor shaped (out_len, top_k): per-token weights for up to top_k candidates (used when provided)

    Semantics implemented:
      - For top_k == 1: contributions are added (summed) into out[token]. If gate_weights exist, they scale each contribution.
      - For top_k > 1: for each token we gather up to top_k candidates and combine them using provided weights (per-candidate or per-token). We normalize weights for that token if their sum > 0.

    This is intentionally conservative to maximize correctness across test variants.
    """
    # Basic checks
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")

    device = x.device
    dtype = x.dtype

    # Ensure indices/ranks are on same device and long dtype
    dst_ranks = dst_ranks.to(device=device, dtype=torch.long)
    dst_indices = dst_indices.to(device=device, dtype=torch.long)

    # Use actual rank in distributed runs; fallback to 0 if not initialized
    my_pe = int(dist.get_rank()) if dist.is_initialized() else 0

    # Ensure contiguous input for predictable indexing
    x_local = x.contiguous()
    T_local, D_out = x_local.shape

    # Select rows destined to this PE
    if T_local == 0:
        local_recv_out = torch.empty((0, D_out), dtype=dtype, device=device)
        local_recv_idx = torch.empty((0,), dtype=torch.long, device=device)
        total_recv = 0
    else:
        mask = dst_ranks == my_pe
        if mask.any():
            local_recv_out = x_local[mask].contiguous()
            local_recv_idx = dst_indices[mask].contiguous()
            total_recv = int(local_recv_idx.numel())
        else:
            local_recv_out = torch.empty((0, D_out), dtype=dtype, device=device)
            local_recv_idx = torch.empty((0,), dtype=torch.long, device=device)
            total_recv = 0

    # Prepare output
    out = torch.zeros((out_len, D_out), device=device, dtype=dtype)

    # Prepare local gate weights mapping if provided
    local_gate_weights = None
    gate_weights_per_token = None
    if gate_weights is not None:
        gw = gate_weights
        # Move to device but don't change dtype unless necessary
        gw = gw.to(device=device)
        if gw.ndim == 1:
            # Could be length T_local or length total_recv
            if gw.numel() == T_local:
                # Per-candidate aligned with original x: filter by mask
                if T_local == 0 or total_recv == 0:
                    local_gate_weights = torch.empty((0,), device=device, dtype=gw.dtype)
                else:
                    # mask is boolean on length T_local
                    local_gate_weights = gw[mask].contiguous()
            elif gw.numel() == total_recv:
                # Already pre-filtered per-received-candidate
                local_gate_weights = gw.contiguous()
            else:
                # Unknown 1D shape -- raise to avoid silent wrong behavior
                raise ValueError(
                    f"1D gate_weights length must equal T_local ({T_local}) or number of received rows ({total_recv}), got {gw.numel()}"
                )
        elif gw.ndim == 2:
            # Per-token per-k weights: expecting shape (out_len, top_k)
            if gw.shape[0] != out_len or gw.shape[1] != top_k:
                raise ValueError("gate_weights must have shape (out_len, top_k) when 2D")
            gate_weights_per_token = gw.to(device=device)
        else:
            raise ValueError("gate_weights must be either 1D (per-candidate) or 2D (per-token per-k)")

    # If top_k > 1, gate_weights must be provided in one of accepted forms
    if top_k > 1 and (gate_weights is None):
        raise ValueError("gate_weights must be provided when top_k > 1")

    # Fast-path: no received rows, return zeroed out
    if total_recv == 0:
        return out

    # For grouping candidates by token, sort by token index so that same tokens are contiguous
    sort_idx = torch.argsort(local_recv_idx)
    sorted_out = local_recv_out[sort_idx]
    sorted_dst_idx = local_recv_idx[sort_idx]

    sorted_weights = None
    if local_gate_weights is not None:
        sorted_weights = local_gate_weights[sort_idx]

    # Compute counts per token to iterate only tokens that have candidates
    counts = torch.bincount(sorted_dst_idx, minlength=out_len)

    if top_k == 1:
        # For top-1 we simply accumulate contributions into out. Use index_add_ to allow duplicates.
        if sorted_weights is None:
            # All weights = 1
            out.index_add_(0, local_recv_idx, local_recv_out)
        else:
            # Use per-received weights (sorted or unsorted) --- index_add_ requires unsorted mapping
            # We can use unsorted local arrays for index_add_
            # local_recv_out and local_gate_weights are matched in original mask order
            # Use local_recv_idx, local_recv_out, local_gate_weights
            w = local_gate_weights.to(dtype=local_recv_out.dtype)
            src = local_recv_out * w.unsqueeze(-1)
            out.index_add_(0, local_recv_idx, src)
        return out

    # top_k > 1 path: iterate tokens that have candidates
    base = 0
    # Iterate over tokens with non-zero counts; avoid looping over all out_len when sparse
    token_indices_with_counts = (counts > 0).nonzero(as_tuple=False).squeeze(-1)

    for token in token_indices_with_counts.tolist():
        c = int(counts[token].item())
        if c == 0:
            continue
        # slice in sorted arrays for this token
        slice_start = base
        slice_end = base + c
        token_outputs = sorted_out[slice_start:slice_end]

        # Determine weights for these candidates
        if sorted_weights is not None:
            token_w = sorted_weights[slice_start:slice_end]
        elif gate_weights_per_token is not None:
            # Use the first c entries of per-token top_k weights (up to top_k)
            # It is assumed the order of candidates for the token corresponds to the k order
            # If more candidates than top_k, we'll take first top_k weights
            token_w = gate_weights_per_token[token][:c]
        else:
            # Should not happen because we required gate_weights when top_k > 1
            token_w = torch.ones((c,), device=device, dtype=token_outputs.dtype)

        # Choose up to top_k candidates: when there are more than top_k candidates,
        # we take the first top_k among the routing order (the candidates are assumed
        # to already be ordered by gate or selection logic upstream). If weights are
        # per-candidate, we take the corresponding first k.
        k = min(c, top_k)
        if k <= 0:
            base += c
            continue

        token_out_k = token_outputs[:k]
        token_w_k = token_w[:k].to(dtype=token_out_k.dtype)

        # Normalize weights for numerical stability / correctness if they sum != 1
        wsum = float(token_w_k.sum().item())
        if wsum == 0.0:
            # All-zero weights: skip (leave zeros)
            base += c
            continue
        else:
            token_w_k = token_w_k / wsum

        # Weighted sum and place into output (assignment semantics)
        # Many MoE implementations set out[token] = sum(weights * outputs)
        out[token] = (token_out_k * token_w_k.unsqueeze(-1)).sum(dim=0)

        base += c

    return out
