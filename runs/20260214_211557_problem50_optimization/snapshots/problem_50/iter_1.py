import torch
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
    Correct and robust PyTorch implementation of the MoE combine semantics.

    Emphasis on correctness and defensive programming to avoid crashes in
    environments where torch.distributed may be absent/uninitialized and to
    handle edge cases: empty inputs, different gate_weights shapes, and
    device/dtype mismatches.

    Behavior summary:
      - top_k == 1: accumulate contributions into out (summing). If gate_weights
        are provided (per-candidate), contributions are scaled accordingly.
      - top_k > 1: for each token that has received candidates, choose up to
        top_k candidates (preserving input order after filtering to this PE),
        normalize weights (if provided) and set out[token] = weighted sum.

    This function is intentionally conservative: it validates shapes and
    raises errors on ambiguous gate_weights shapes to avoid silent mistakes.
    """
    # Basic validation
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")

    device = x.device
    dtype = x.dtype

    # Determine current PE robustly without assuming torch.distributed exists
    my_pe = 0
    dist_mod = getattr(torch, "distributed", None)
    try:
        if dist_mod is not None and hasattr(dist_mod, "is_available") and dist_mod.is_available():
            if dist_mod.is_initialized():
                # safe call now
                my_pe = int(dist_mod.get_rank())
    except Exception:
        # Fallback to 0 if any issues querying the distributed runtime
        my_pe = 0

    # Ensure indices/ranks are on the same device and long dtype
    # Copying to device early so subsequent ops are safe
    dst_ranks = dst_ranks.to(device=device, dtype=torch.long)
    dst_indices = dst_indices.to(device=device, dtype=torch.long)

    # Ensure contiguous input for predictable indexing
    x_local = x.contiguous()
    # T_local might be 0
    if x_local.ndim != 2:
        raise ValueError("x must be 2D tensor of shape [T_local, D_out]")
    T_local, D_out = x_local.shape

    # Build a mask of rows destined to this PE. This mask is always defined
    if T_local == 0:
        mask = torch.zeros((0,), dtype=torch.bool, device=device)
    else:
        # dst_ranks is already on correct device and long dtype
        mask = (dst_ranks == my_pe)

    # Select rows destined to this PE
    if T_local == 0 or mask.numel() == 0 or not mask.any():
        local_recv_out = torch.empty((0, D_out), dtype=dtype, device=device)
        local_recv_idx = torch.empty((0,), dtype=torch.long, device=device)
        total_recv = 0
    else:
        local_recv_out = x_local[mask].contiguous()
        local_recv_idx = dst_indices[mask].contiguous()
        total_recv = int(local_recv_idx.numel())

    # Prepare output
    out = torch.zeros((out_len, D_out), device=device, dtype=dtype)

    # Prepare local gate weights mapping if provided
    local_gate_weights = None
    gate_weights_per_token = None
    if gate_weights is not None:
        gw = gate_weights
        # Move to device (preserve dtype unless explicitly cast later)
        gw = gw.to(device=device)
        if gw.ndim == 1:
            # Could be per-input (length T_local) or per-received (length total_recv)
            if gw.numel() == T_local:
                # Per-candidate aligned with original x: filter by mask
                # If there are no selected rows, create empty tensor
                if T_local == 0 or total_recv == 0:
                    local_gate_weights = torch.empty((0,), device=device, dtype=gw.dtype)
                else:
                    local_gate_weights = gw[mask].contiguous()
            elif gw.numel() == total_recv:
                # Already pre-filtered per-received-candidate
                local_gate_weights = gw.contiguous()
            else:
                raise ValueError(
                    f"1D gate_weights length must equal T_local ({T_local}) or number of received rows ({total_recv}), got {int(gw.numel())}"
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
    # Ensure sorted_dst_idx is 1D long
    counts = torch.bincount(sorted_dst_idx, minlength=out_len)

    if top_k == 1:
        # For top-1 we simply accumulate contributions into out. Use index_add_ to allow duplicates.
        if local_gate_weights is None:
            # All weights = 1
            # index_add_ expects index on CPU or same device; local_recv_idx is on device
            out.index_add_(0, local_recv_idx, local_recv_out)
        else:
            # Use per-received weights (local_gate_weights is aligned with local_recv_out)
            w = local_gate_weights.to(dtype=local_recv_out.dtype)
            src = local_recv_out * w.unsqueeze(-1)
            out.index_add_(0, local_recv_idx, src)
        return out

    # top_k > 1 path: iterate tokens that have candidates. We iterate only tokens with counts > 0
    base = 0
    token_indices_with_counts = (counts > 0).nonzero(as_tuple=False).squeeze(-1)

    # If token_indices_with_counts is zero-dim (single index), ensure it's 1D
    if token_indices_with_counts.dim() == 0:
        token_indices = [int(token_indices_with_counts.item())]
    else:
        token_indices = [int(t.item()) for t in token_indices_with_counts]

    for token in token_indices:
        c = int(counts[token].item())
        if c == 0:
            continue
        slice_start = base
        slice_end = base + c
        token_outputs = sorted_out[slice_start:slice_end]

        # Determine weights for these candidates
        if sorted_weights is not None:
            token_w = sorted_weights[slice_start:slice_end]
        elif gate_weights_per_token is not None:
            # Use per-token weights; it's shaped (out_len, top_k). We may have c <= top_k
            token_w = gate_weights_per_token[token][:c]
        else:
            # Should not happen due to earlier guard, but safe fallback
            token_w = torch.ones((c,), device=device, dtype=token_outputs.dtype)

        k = min(c, top_k)
        if k <= 0:
            base += c
            continue

        token_out_k = token_outputs[:k]
        token_w_k = token_w[:k].to(dtype=token_out_k.dtype)

        # Normalize weights for numerical stability / correctness if they sum != 1
        wsum = float(token_w_k.sum().item())
        if wsum == 0.0:
            # All-zero weights: leave out[token] as zeros
            base += c
            continue
        token_w_k = token_w_k / wsum

        # Weighted sum and place into output (assignment semantics)
        out[token] = (token_out_k * token_w_k.unsqueeze(-1)).sum(dim=0)

        base += c

    return out
