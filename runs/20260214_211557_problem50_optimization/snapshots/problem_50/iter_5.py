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
    Correctness-first implementation of MoE combine semantics.

    Safety changes from prior attempt:
    - Removed @torch.no_grad() decorator to avoid environment-dependent import/serialization issues.
    - Avoid per-element .item() calls on device tensors inside loops; convert index/weight lists to CPU Python lists once.
    - Ensure indices used for index_add_ are tensors placed on the same device as the output.

    Semantics (kept consistent):
    - Only entries where dst_ranks == my_pe (assumed 0) are considered local receipts.
    - top_k == 1: sum contributions per target index; if per-candidate weights provided scale before summing.
    - top_k > 1: select up to top_k candidates per token, weight-normalize, and sum.
    """
    # Basic validation
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    if not isinstance(dst_ranks, torch.Tensor) or not isinstance(dst_indices, torch.Tensor):
        raise TypeError("dst_ranks and dst_indices must be torch.Tensors")

    device = x.device
    dtype = x.dtype

    # Fixed PE for this single-process correctness implementation
    my_pe = 0

    # Ensure indices are integer tensors on some device for comparisons
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
        local_recv_idx_list = []
        total_recv = 0
    else:
        local_recv_out = x_local[mask].contiguous()  # on device
        # For safer iteration and to avoid per-element .item() on device tensors, convert indices to CPU python list
        local_recv_idx = dst_indices[mask].contiguous().to(device='cpu')
        local_recv_idx_list = local_recv_idx.tolist()
        total_recv = int(len(local_recv_idx_list))

    # Prepare output on same device
    out = torch.zeros((out_len, D_out), device=device, dtype=dtype)

    # Prepare gate weight representations (as python lists or a per-token 2D tensor)
    local_gate_weights_list = None  # Python list of floats aligned with local_recv_idx_list if provided per-candidate
    gate_weights_per_token = None   # 2D tensor on device with shape (out_len, K) if provided

    if gate_weights is not None:
        gw = gate_weights
        if not isinstance(gw, torch.Tensor):
            raise TypeError("gate_weights must be a torch.Tensor if provided")
        # Keep per-token 2D weights on device to avoid CPU/GPU confusion; per-candidate weights will be moved to CPU list
        if gw.ndim == 1:
            gw = gw.to(device=device)
            if gw.numel() == T_local:
                # per-input: select those destined here and convert to python list
                if T_local == 0 or total_recv == 0:
                    local_gate_weights_list = []
                else:
                    local_gw = gw[mask].contiguous().to(device='cpu')
                    local_gate_weights_list = [float(v) for v in local_gw.tolist()]
            elif gw.numel() == total_recv:
                # Already per-received-entry ordering
                local_gate_weights_list = [float(v) for v in gw.to(device='cpu').tolist()]
            else:
                raise ValueError(
                    f"1D gate_weights length must equal T_local ({T_local}) or number of received rows ({total_recv}), got {int(gw.numel())}"
                )
        elif gw.ndim == 2:
            # per-token per-k
            if gw.shape[0] != out_len or gw.shape[1] < 1:
                raise ValueError("gate_weights 2D must have shape (out_len, K) with K >= 1")
            gate_weights_per_token = gw.to(device=device, dtype=dtype)
        else:
            raise ValueError("gate_weights must be either 1D (per-candidate) or 2D (per-token per-k)")

    if top_k > 1 and gate_weights is None:
        raise ValueError("gate_weights must be provided when top_k > 1")

    # Fast path: no local received rows
    if total_recv == 0:
        return out

    # Handle top_k == 1 with efficient index_add_ when possible
    if top_k == 1:
        if local_gate_weights_list is None:
            # Create index tensor on correct device and call index_add_
            idx_tensor = torch.tensor(local_recv_idx_list, dtype=torch.long, device=device)
            out.index_add_(0, idx_tensor, local_recv_out)
        else:
            w = torch.tensor(local_gate_weights_list, dtype=local_recv_out.dtype, device=device)
            src = local_recv_out * w.unsqueeze(-1)
            idx_tensor = torch.tensor(local_recv_idx_list, dtype=torch.long, device=device)
            out.index_add_(0, idx_tensor, src)
        return out

    # top_k > 1: gather per-token candidate lists
    per_token_vals = {}  # token -> list of tensors (vectors on device)
    per_token_weights = {}  # token -> list of float weights (python floats) when available

    # Populate lists (iterate in Python over cpu index list to avoid device .item() calls)
    for i in range(total_recv):
        token = local_recv_idx_list[i]
        if token is None:
            continue
        if not (0 <= token < out_len):
            continue
        if token not in per_token_vals:
            per_token_vals[token] = []
            per_token_weights[token] = []
        per_token_vals[token].append(local_recv_out[i])
        if local_gate_weights_list is not None:
            per_token_weights[token].append(float(local_gate_weights_list[i]))

    # Combine per token
    for token, vec_list in per_token_vals.items():
        c = len(vec_list)
        if c == 0:
            continue
        k = min(c, top_k)

        # Stack candidate vectors on device
        try:
            values = torch.stack(vec_list, dim=0)  # [c, D_out]
        except Exception:
            # Safety fallback: convert each to contiguous first
            values = torch.stack([v.contiguous() for v in vec_list], dim=0)

        if len(per_token_weights[token]) > 0:
            # per-candidate weights available: select top-k by weight
            weights_tensor = torch.tensor(per_token_weights[token], dtype=values.dtype, device=device)
            if k == c:
                chosen_vals = values
                chosen_w = weights_tensor
            else:
                topk_vals, topk_idx = torch.topk(weights_tensor, k=k, largest=True, sorted=False)
                chosen_vals = values[topk_idx]
                chosen_w = topk_vals
        elif gate_weights_per_token is not None:
            # use provided per-token per-k weights (take first k entries)
            provided_k = gate_weights_per_token.shape[1]
            take_k = min(k, provided_k)
            chosen_vals = values[:take_k]
            chosen_w = gate_weights_per_token[token][:take_k].to(dtype=values.dtype)
            if take_k < k:
                # pad with ones to reach k (and use subsequent candidate vectors)
                pad_count = k - take_k
                pad_w = torch.ones((pad_count,), device=device, dtype=values.dtype)
                chosen_w = torch.cat([chosen_w, pad_w], dim=0)
                chosen_vals = torch.cat([chosen_vals, values[take_k:take_k+pad_count]], dim=0)
        else:
            # No explicit weights, uniform
            chosen_vals = values[:k]
            chosen_w = torch.ones((k,), device=device, dtype=values.dtype)

        wsum = float(chosen_w.sum().item())
        if wsum == 0.0:
            continue
        chosen_w = chosen_w / wsum
        contrib = (chosen_vals * chosen_w.unsqueeze(-1)).sum(dim=0)
        out[token] = contrib

    return out
