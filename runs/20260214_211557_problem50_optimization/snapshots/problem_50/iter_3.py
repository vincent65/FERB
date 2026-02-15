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
    Robust, correctness-first PyTorch implementation of MoE combine semantics.

    Semantics implemented:
    - Only rows whose dst_ranks == current PE are considered local-received.
    - If top_k == 1: accumulate (sum) contributions into out[token]. If per-candidate
      gate_weights (1D) are provided, scale each candidate by its weight before summing.
    - If top_k > 1: select up to top_k candidates per token by their gate_weights
      when per-candidate weights are provided (1D). When gate_weights is a
      per-token-per-k tensor (2D of shape [out_len, top_k]) we assume candidate
      ordering is already the chosen ordering and use the first k candidates.
      Selected candidates are normalized (weights sum to 1) and combined.

    The implementation focuses on safety across devices and empty inputs.
    """
    # Basic validation and standardization
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    device = x.device
    dtype = x.dtype

    # Determine current PE safely (fallback to 0)
    my_pe = 0
    try:
        if hasattr(torch, "distributed"):
            dist = torch.distributed
            if callable(getattr(dist, "is_available", None)) and dist.is_available():
                if callable(getattr(dist, "is_initialized", None)) and dist.is_initialized():
                    if callable(getattr(dist, "get_rank", None)):
                        my_pe = int(dist.get_rank())
    except Exception:
        my_pe = 0

    # Ensure indices/ranks are on the same device and long dtype
    dst_ranks = dst_ranks.to(device=device, dtype=torch.long)
    dst_indices = dst_indices.to(device=device, dtype=torch.long)

    # Ensure x is 2D
    x_local = x.contiguous()
    if x_local.ndim != 2:
        raise ValueError("x must be 2D tensor of shape [T_local, D_out]")
    T_local, D_out = x_local.shape

    # Build mask for rows destined to this PE
    if T_local == 0:
        mask = torch.zeros((0,), dtype=torch.bool, device=device)
    else:
        mask = (dst_ranks == my_pe)

    # Select local-received rows
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

    # Prepare gate weight representations
    local_gate_weights = None  # 1D per-candidate, aligned with local_recv_*
    gate_weights_per_token = None  # 2D [out_len, top_k]

    if gate_weights is not None:
        gw = gate_weights
        # move to same device if needed but preserve dtype
        gw = gw.to(device=device)
        if gw.ndim == 1:
            # Could be per-input (length T_local) or per-received (length total_recv)
            if gw.numel() == T_local:
                # per-input: pick those destined to this PE
                if T_local == 0 or total_recv == 0:
                    local_gate_weights = torch.empty((0,), device=device, dtype=gw.dtype)
                else:
                    local_gate_weights = gw[mask].contiguous()
            elif gw.numel() == total_recv:
                # already per-received
                local_gate_weights = gw.contiguous()
            else:
                raise ValueError(
                    f"1D gate_weights length must equal T_local ({T_local}) or number of received rows ({total_recv}), got {int(gw.numel())}"
                )
        elif gw.ndim == 2:
            # per-token per-k
            if gw.shape[0] != out_len or gw.shape[1] != top_k:
                raise ValueError("gate_weights must have shape (out_len, top_k) when 2D")
            gate_weights_per_token = gw.to(device=device)
        else:
            raise ValueError("gate_weights must be either 1D (per-candidate) or 2D (per-token per-k)")

    # top_k > 1 requires some gate_weights information to choose/weight candidates
    if top_k > 1 and gate_weights is None:
        raise ValueError("gate_weights must be provided when top_k > 1")

    # Fast path: no local received rows
    if total_recv == 0:
        return out

    # Group candidates by token using a stable sort so candidates for each token are contiguous
    sort_idx = torch.argsort(local_recv_idx)
    sorted_out = local_recv_out[sort_idx]
    sorted_dst_idx = local_recv_idx[sort_idx]
    sorted_weights = None
    if local_gate_weights is not None:
        sorted_weights = local_gate_weights[sort_idx]

    # Compute counts per token
    counts = torch.zeros((out_len,), dtype=torch.long, device=device)
    ones = torch.ones_like(sorted_dst_idx, dtype=torch.long, device=device)
    if sorted_dst_idx.numel() > 0:
        counts.index_add_(0, sorted_dst_idx, ones)

    # Handle top_k == 1 by summing (possibly weighted) contributions
    if top_k == 1:
        if local_gate_weights is None:
            # unsorted local_recv_idx and local_recv_out are aligned
            # use index_add_ to sum contributions for duplicates
            out.index_add_(0, local_recv_idx, local_recv_out)
        else:
            w = local_gate_weights.to(dtype=local_recv_out.dtype)
            src = local_recv_out * w.unsqueeze(-1)
            out.index_add_(0, local_recv_idx, src)
        return out

    # top_k > 1: iterate over tokens that have candidates
    token_positions = (counts > 0).nonzero(as_tuple=False).view(-1)

    base = 0
    for token in token_positions.tolist():
        c = int(counts[token].item())
        if c == 0:
            continue
        slice_start = base
        slice_end = base + c
        token_outputs = sorted_out[slice_start:slice_end]  # [c, D_out]

        # Determine candidate weights for this token
        if sorted_weights is not None:
            token_w = sorted_weights[slice_start:slice_end].to(dtype=token_outputs.dtype)
            # select up to top_k candidates by highest weight
            k = min(c, top_k)
            if k <= 0:
                base += c
                continue
            if c == 1 or k == c:
                # Use all
                chosen_out = token_outputs[:k]
                chosen_w = token_w[:k]
            else:
                # pick topk indices by weight
                topk_vals, topk_idxs = torch.topk(token_w, k=k, largest=True, sorted=False)
                chosen_out = token_outputs[topk_idxs]
                chosen_w = topk_vals
        elif gate_weights_per_token is not None:
            # We assume candidates are ordered in preference order for this token;
            # choose the first k candidates (as in many MoE pipelines where top-k routing
            # ensured ordering). If there are fewer than top_k candidates, use what exists.
            k = min(c, top_k)
            if k <= 0:
                base += c
                continue
            chosen_out = token_outputs[:k]
            # get the per-token per-k weights (length top_k) and take first k entries
            chosen_w = gate_weights_per_token[token][:k].to(dtype=token_outputs.dtype)
        else:
            # Should not reach here since top_k>1 requires gate_weights, but handle robustly
            k = min(c, top_k)
            chosen_out = token_outputs[:k]
            chosen_w = torch.ones((k,), device=device, dtype=token_outputs.dtype)

        # Normalize chosen_w
        wsum = float(chosen_w.sum().item())
        if wsum == 0.0:
            # If weights sum to zero, treat as no contribution
            base += c
            continue
        chosen_w = chosen_w / wsum

        # Weighted sum
        out[token] = (chosen_out * chosen_w.unsqueeze(-1)).sum(dim=0)

        base += c

    return out
