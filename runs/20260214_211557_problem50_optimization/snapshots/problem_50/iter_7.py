import torch
from typing import Optional


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
    Robust correctness-first MoE combine for a single-PE (assumes this PE is 0).

    Defensive changes vs. previous attempt:
    - Ensure all tensors are moved to the same device and appropriate dtypes.
    - Filter out-of-range indices early to avoid index_add_ and indexing errors.
    - Handle empty inputs and no-local-candidate paths uniformly.
    - Avoid ambiguous tensor truthiness; use .numel()/.item() explicitly.
    - Deterministic iteration and guarded indexing for top_k > 1.

    The function returns a tensor of shape [out_len, D_out].
    """

    # Basic validation
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not isinstance(dst_ranks, torch.Tensor) or not isinstance(dst_indices, torch.Tensor):
        raise TypeError("dst_ranks and dst_indices must be torch.Tensors")
    if not isinstance(out_len, int):
        raise TypeError("out_len must be an int")
    if top_k < 1:
        raise ValueError("top_k must be >= 1")

    device = x.device
    dtype = x.dtype

    # Single-PE assumption for correctness-mode
    my_pe = 0

    # Move index/rank tensors to same device/dtype for indexing
    dst_ranks = dst_ranks.to(device=device, dtype=torch.long)
    dst_indices = dst_indices.to(device=device, dtype=torch.long)

    # Ensure x is 2D and contiguous
    x_local = x.contiguous()
    if x_local.ndim != 2:
        raise ValueError("x must be 2D tensor of shape [T_local, D_out]")
    T_local, D_out = x_local.shape

    # Build mask of entries destined to this PE
    if T_local == 0:
        # No local inputs at all
        return torch.zeros((out_len, D_out), device=device, dtype=dtype)

    mask = dst_ranks == my_pe  # boolean tensor on device

    # If nothing for this PE, return zeros
    if mask.sum().item() == 0:
        return torch.zeros((out_len, D_out), device=device, dtype=dtype)

    # Local received rows & indices
    local_recv_out = x_local[mask]  # [R, D_out]
    local_recv_idx = dst_indices[mask].to(device=device, dtype=torch.long)  # [R]
    R = local_recv_out.shape[0]

    # Filter out invalid indices (safety guard). Keep only 0 <= idx < out_len
    if R == 0:
        return torch.zeros((out_len, D_out), device=device, dtype=dtype)

    valid_mask = (local_recv_idx >= 0) & (local_recv_idx < out_len)
    if valid_mask.sum().item() == 0:
        return torch.zeros((out_len, D_out), device=device, dtype=dtype)

    if valid_mask.all().item() is False:
        local_recv_out = local_recv_out[valid_mask]
        local_recv_idx = local_recv_idx[valid_mask]
        R = local_recv_out.shape[0]
        if R == 0:
            return torch.zeros((out_len, D_out), device=device, dtype=dtype)

    # Prepare output
    out = torch.zeros((out_len, D_out), device=device, dtype=dtype)

    # Parse gate_weights into two canonical forms:
    # - gate_per_candidate: 1D tensor length R (aligned with local_recv_out rows)
    # - gate_per_token: 2D tensor [out_len, K]
    gate_per_candidate = None
    gate_per_token = None

    if gate_weights is not None:
        if not isinstance(gate_weights, torch.Tensor):
            raise TypeError("gate_weights must be a torch.Tensor if provided")
        gw = gate_weights
        # Move to device but do not force dtype for 2D per-token case beyond float conversion below
        gw = gw.to(device=device)
        if gw.ndim == 1:
            # Could be length T_local (per-input) or length R (per-received)
            if gw.numel() == T_local:
                gw_on_dev = gw.contiguous()
                # select those destined here
                gw_masked = gw_on_dev[mask]
                # After previous valid_mask filter, need to keep only valid rows
                gw_masked = gw_masked[valid_mask]
                gate_per_candidate = gw_masked.to(dtype=dtype)
            elif gw.numel() == R:
                gate_per_candidate = gw.contiguous().to(dtype=dtype)
            else:
                raise ValueError(
                    f"1D gate_weights length must equal T_local ({T_local}) or number of received rows ({R}), got {int(gw.numel())}"
                )
        elif gw.ndim == 2:
            # per-token-per-k: shape [out_len, K]
            if gw.shape[0] != out_len or gw.shape[1] < 1:
                raise ValueError("gate_weights 2D must have shape (out_len, K) with K >= 1")
            gate_per_token = gw.to(dtype=dtype).contiguous()
        else:
            raise ValueError("gate_weights must be either 1D (per-candidate) or 2D (per-token per-k)")

    if top_k > 1 and gate_weights is None:
        raise ValueError("gate_weights must be provided when top_k > 1")

    # Fast path: top_k == 1
    if top_k == 1:
        # If gate_per_token provided, use first column as per-token scalar
        if gate_per_token is not None:
            # Get per-token first weight (shape [out_len]) and gather for local indices
            per_token_first = gate_per_token[:, 0].to(device=device, dtype=dtype)
            gate_per_candidate = per_token_first[local_recv_idx]

        if gate_per_candidate is None:
            # Sum all candidate vectors into their destination rows
            # Ensure local_recv_idx is long and in-range by previous filter
            out.index_add_(0, local_recv_idx, local_recv_out)
        else:
            # Multiply each candidate vector by its scalar weight then add
            if gate_per_candidate.numel() != local_recv_out.shape[0]:
                raise RuntimeError("Internal: gate_per_candidate length mismatch")
            src = local_recv_out * gate_per_candidate.unsqueeze(-1)
            out.index_add_(0, local_recv_idx, src)
        return out

    # top_k > 1: correctness-first path
    # We'll process each token deterministically in ascending order of tokens present
    unique_tokens = torch.unique(local_recv_idx)
    if unique_tokens.numel() == 0:
        return out

    # Convert to CPU python ints for iteration to avoid device-side scalar pitfalls
    # But keep a filtered list only for tokens that actually are within valid range (they are by earlier filter)
    unique_tokens_cpu = unique_tokens.to('cpu')

    for t_idx in unique_tokens_cpu.tolist():
        token = int(t_idx)
        # Select rows for this token
        sel_mask = (local_recv_idx == token)
        c = int(sel_mask.sum().item())
        if c == 0:
            continue
        vals = local_recv_out[sel_mask]  # [c, D_out]
        k = min(c, top_k)

        # Determine weights and chosen candidate rows
        if gate_per_candidate is not None:
            w = gate_per_candidate[sel_mask]  # [c]
            if k == c:
                chosen_vals = vals
                chosen_w = w.to(dtype=dtype)
            else:
                # pick top-k by weight
                # torch.topk returns (values, indices) into w
                topk_vals, topk_idx = torch.topk(w, k=k, largest=True, sorted=False)
                chosen_vals = vals[topk_idx.to(dtype=torch.long)]
                chosen_w = topk_vals.to(dtype=dtype)
        elif gate_per_token is not None:
            provided_k = gate_per_token.shape[1]
            # Take up to provided_k weights for this token
            take_k = min(k, provided_k)
            token_ws = gate_per_token[token][:take_k].to(device=device, dtype=dtype)
            chosen_vals = vals[:take_k]
            chosen_w = token_ws
            if take_k < k:
                pad_count = k - take_k
                pad_w = torch.ones((pad_count,), device=device, dtype=dtype)
                chosen_w = torch.cat([chosen_w, pad_w], dim=0)
                # Append the next candidate vectors if available (values beyond take_k)
                extra_vals = vals[take_k: take_k + pad_count]
                if extra_vals.shape[0] < pad_count:
                    # If not enough extra values, pad with zeros to keep sizes consistent
                    pad_vals = torch.zeros((pad_count - extra_vals.shape[0], D_out), device=device, dtype=dtype)
                    extra_vals = torch.cat([extra_vals, pad_vals], dim=0)
                chosen_vals = torch.cat([chosen_vals, extra_vals], dim=0)
        else:
            # Uniform weights across the first k candidates
            chosen_vals = vals[:k]
            chosen_w = torch.ones((k,), device=device, dtype=dtype)

        # Normalize weights; skip if sum is zero
        wsum = float(chosen_w.sum().item())
        if wsum == 0.0:
            continue
        chosen_w = chosen_w / wsum
        contrib = (chosen_vals * chosen_w.unsqueeze(-1)).sum(dim=0)

        # Write contribution into output at this token index
        # If multiple contributions to same token processed serially, we overwrite (per spec we combine per-token once)
        out[token] = contrib

    return out
