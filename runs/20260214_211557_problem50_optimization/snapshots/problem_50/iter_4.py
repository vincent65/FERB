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
    Correctness-first implementation of MoE combine semantics.

    Behavior summary:
    - Only entries where dst_ranks == current PE are considered local receipts.
    - If top_k == 1: contributions for the same output index are summed. If per-candidate
      gate_weights (1D) are provided they scale each candidate before summing.
    - If top_k > 1: up to top_k candidates are selected per token and combined using
      normalized weights. If per-candidate 1D weights are provided we select the top-k
      by those weights; if a 2D gate_weights (out_len, top_k) is provided we use the
      first k weights for that token (assuming ordering was pre-determined).

    This implementation prefers simplicity and robustness over micro-optimizations to
    avoid environment-dependent failures.
    """
    # Basic validation
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not isinstance(dst_ranks, torch.Tensor) or not isinstance(dst_indices, torch.Tensor):
        raise TypeError("dst_ranks and dst_indices must be torch.Tensors")

    device = x.device
    dtype = x.dtype

    # Avoid any torch.distributed calls here to prevent environment-dependent failures.
    my_pe = 0

    # Move / cast index tensors to correct device and dtype
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
        # dst_ranks already moved to device
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
    local_gate_weights = None  # 1D per-candidate aligned with local_recv_*
    gate_weights_per_token = None  # 2D [out_len, top_k]

    if gate_weights is not None:
        gw = gate_weights
        if not isinstance(gw, torch.Tensor):
            raise TypeError("gate_weights must be a torch.Tensor if provided")
        gw = gw.to(device=device)
        if gw.ndim == 1:
            if gw.numel() == T_local:
                # per-input: select those destined here
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
            # per-token per-k
            if gw.shape[0] != out_len or gw.shape[1] < 1:
                raise ValueError("gate_weights 2D must have shape (out_len, K) with K >= 1")
            # We don't require gw.shape[1] == top_k exactly because top_k may be <= provided K;
            # we'll take first k entries where appropriate.
            gate_weights_per_token = gw.to(device=device)
        else:
            raise ValueError("gate_weights must be either 1D (per-candidate) or 2D (per-token per-k)")

    if top_k > 1 and gate_weights is None:
        raise ValueError("gate_weights must be provided when top_k > 1")

    # Fast path: no local received rows
    if total_recv == 0:
        return out

    # Handle top_k == 1 with efficient index_add_ when possible
    if top_k == 1:
        if local_gate_weights is None:
            # simple sum per destination index
            # local_recv_idx aligned with local_recv_out
            out.index_add_(0, local_recv_idx, local_recv_out)
        else:
            w = local_gate_weights.to(dtype=local_recv_out.dtype)
            src = local_recv_out * w.unsqueeze(-1)
            out.index_add_(0, local_recv_idx, src)
        return out

    # top_k > 1: build per-token candidate lists (explicitly) for safety
    per_token_vals = {}     # token -> list of tensors (vectors)
    per_token_w = {}        # token -> list of weight scalars (only for per-candidate weights)

    # Populate lists
    for i in range(total_recv):
        token = int(local_recv_idx[i].item())
        vec = local_recv_out[i]
        if token < 0 or token >= out_len:
            # skip invalid token indices defensively
            continue
        if token not in per_token_vals:
            per_token_vals[token] = []
            per_token_w[token] = []
        per_token_vals[token].append(vec)
        if local_gate_weights is not None:
            per_token_w[token].append(float(local_gate_weights[i].item()))

    # Combine per token
    for token, vec_list in per_token_vals.items():
        c = len(vec_list)
        if c == 0:
            continue
        k = min(c, top_k)

        values = torch.stack(vec_list, dim=0)  # [c, D_out]

        if len(per_token_w[token]) > 0:
            # per-candidate weights available: select top-k by weight
            weights_tensor = torch.tensor(per_token_w[token], dtype=values.dtype, device=device)
            if k == c:
                chosen_vals = values
                chosen_w = weights_tensor
            else:
                # topk returns (vals, indices)
                topk_vals, topk_idx = torch.topk(weights_tensor, k=k, largest=True, sorted=False)
                chosen_vals = values[topk_idx]
                chosen_w = topk_vals
        elif gate_weights_per_token is not None:
            # use provided per-token per-k weights (take first k entries)
            # If the provided K is smaller than k, take available; else take first k.
            provided_k = gate_weights_per_token.shape[1]
            take_k = min(k, provided_k)
            chosen_vals = values[:take_k]
            chosen_w = gate_weights_per_token[token][:take_k].to(dtype=values.dtype)
            # If we needed more than provided_k (rare/invalid), pad with ones
            if take_k < k:
                pad_count = k - take_k
                pad_w = torch.ones((pad_count,), device=device, dtype=values.dtype)
                chosen_w = torch.cat([chosen_w, pad_w], dim=0)
                chosen_vals = torch.cat([chosen_vals, values[take_k:take_k+pad_count]], dim=0)
        else:
            # No explicit weights, fall back to uniform for available candidates
            chosen_vals = values[:k]
            chosen_w = torch.ones((k,), device=device, dtype=values.dtype)

        wsum = float(chosen_w.sum().item())
        if wsum == 0.0:
            # if all weights are zero, skip contribution
            continue
        chosen_w = chosen_w / wsum
        out[token] = (chosen_vals * chosen_w.unsqueeze(-1)).sum(dim=0)

    return out
