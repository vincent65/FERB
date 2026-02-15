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

    This implementation focuses on correctness and robustness across devices
    and PyTorch versions. It avoids device- or version-sensitive ops (e.g.,
    torch.bincount on some runtimes) and uses safe shape handling.
    """
    # Basic validation
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")

    device = x.device
    dtype = x.dtype

    # Determine current PE robustly: default to 0 if distributed isn't available
    my_pe = 0
    try:
        # Use top-level functions when available; guard with try/except
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

    # Ensure contiguous input for predictable indexing
    x_local = x.contiguous()
    if x_local.ndim != 2:
        raise ValueError("x must be 2D tensor of shape [T_local, D_out]")
    T_local, D_out = x_local.shape

    # Build mask of rows destined to this PE
    if T_local == 0:
        mask = torch.zeros((0,), dtype=torch.bool, device=device)
    else:
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
        gw = gw.to(device=device)
        if gw.ndim == 1:
            # Could be per-input (length T_local) or per-received (length total_recv)
            if gw.numel() == T_local:
                if T_local == 0 or total_recv == 0:
                    local_gate_weights = torch.empty((0,), device=device, dtype=gw.dtype)
                else:
                    local_gate_weights = gw[mask].contiguous()
            elif gw.numel() == total_recv:
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

    # If top_k > 1, gate_weights must be provided
    if top_k > 1 and (gate_weights is None):
        raise ValueError("gate_weights must be provided when top_k > 1")

    # Fast-path: no received rows
    if total_recv == 0:
        return out

    # Group candidates by token. Sort by token index so same tokens are contiguous
    sort_idx = torch.argsort(local_recv_idx)
    sorted_out = local_recv_out[sort_idx]
    sorted_dst_idx = local_recv_idx[sort_idx]

    sorted_weights = None
    if local_gate_weights is not None:
        sorted_weights = local_gate_weights[sort_idx]

    # Compute counts per token without relying on torch.bincount (more portable)
    counts = torch.zeros((out_len,), dtype=torch.long, device=device)
    ones = torch.ones_like(sorted_dst_idx, dtype=torch.long, device=device)
    # index_add_ accumulates ones into counts at positions sorted_dst_idx
    counts.index_add_(0, sorted_dst_idx, ones)

    if top_k == 1:
        # For top-1 we simply accumulate contributions into out. Use index_add_ to allow duplicates.
        if local_gate_weights is None:
            out.index_add_(0, local_recv_idx, local_recv_out)
        else:
            w = local_gate_weights.to(dtype=local_recv_out.dtype)
            src = local_recv_out * w.unsqueeze(-1)
            out.index_add_(0, local_recv_idx, src)
        return out

    # top_k > 1 path: iterate tokens that have candidates
    token_indices_tensor = (counts > 0).nonzero(as_tuple=False).view(-1)
    token_indices = token_indices_tensor.tolist()

    base = 0
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
            token_w = gate_weights_per_token[token][:c]
        else:
            token_w = torch.ones((c,), device=device, dtype=token_outputs.dtype)

        k = min(c, top_k)
        if k <= 0:
            base += c
            continue

        token_out_k = token_outputs[:k]
        token_w_k = token_w[:k].to(dtype=token_out_k.dtype)

        wsum = float(token_w_k.sum().item())
        if wsum == 0.0:
            base += c
            continue
        token_w_k = token_w_k / wsum

        out[token] = (token_out_k * token_w_k.unsqueeze(-1)).sum(dim=0)

        base += c

    return out
