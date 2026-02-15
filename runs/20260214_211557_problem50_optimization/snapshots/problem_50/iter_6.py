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

    Key points:
    - Works entirely with PyTorch tensors on the input device where possible.
    - Avoids constructing large Python lists of tensors; groups candidates using tensor ops.
    - Supports gate_weights as 1D (per-input or per-received) or 2D (per-token per-k).
    - For top_k == 1, uses index_add_ to sum contributions per output index.
    - For top_k > 1, gathers candidates per-token and selects/top-normalizes as required.
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

    # Use a single-process correctness implementation: assume this PE is 0
    my_pe = 0

    # Normalize index/rank tensors onto the same device (long dtype for indices)
    dst_ranks = dst_ranks.to(device=device, dtype=torch.long)
    dst_indices = dst_indices.to(device=device, dtype=torch.long)

    # Ensure x is 2D
    x_local = x.contiguous()
    if x_local.ndim != 2:
        raise ValueError("x must be 2D tensor of shape [T_local, D_out]")
    T_local, D_out = x_local.shape

    # Mask entries destined to this PE
    if T_local == 0:
        mask = torch.zeros((0,), dtype=torch.bool, device=device)
    else:
        mask = (dst_ranks == my_pe)

    # Select local received rows and corresponding indices
    if mask.numel() == 0 or not mask.any():
        # Nothing received locally
        out = torch.zeros((out_len, D_out), device=device, dtype=dtype)
        return out

    local_recv_out = x_local[mask]  # [R, D_out]
    local_recv_idx = dst_indices[mask].to(device=device, dtype=torch.long)  # [R]
    R = local_recv_out.shape[0]

    # Prepare output
    out = torch.zeros((out_len, D_out), device=device, dtype=dtype)

    # Parse gate_weights
    gate_per_candidate = None  # 1D tensor of length R aligned with local_recv_out
    gate_per_token = None  # 2D tensor [out_len, K]

    if gate_weights is not None:
        if not isinstance(gate_weights, torch.Tensor):
            raise TypeError("gate_weights must be a torch.Tensor if provided")
        gw = gate_weights
        if gw.ndim == 1:
            # Could be per-input (T_local) or per-received (R)
            if gw.numel() == T_local:
                # per-input: select those destined here
                gw_masked = gw.to(device=device).contiguous()
                gate_per_candidate = gw_masked[mask].to(dtype=dtype)
            elif gw.numel() == R:
                gate_per_candidate = gw.to(device=device, dtype=dtype).contiguous()
            else:
                raise ValueError(
                    f"1D gate_weights length must equal T_local ({T_local}) or number of received rows ({R}), got {int(gw.numel())}"
                )
        elif gw.ndim == 2:
            # per-token per-k
            if gw.shape[0] != out_len or gw.shape[1] < 1:
                raise ValueError("gate_weights 2D must have shape (out_len, K) with K >= 1")
            gate_per_token = gw.to(device=device, dtype=dtype).contiguous()
        else:
            raise ValueError("gate_weights must be either 1D (per-candidate) or 2D (per-token per-k)")

    if top_k > 1 and gate_weights is None:
        raise ValueError("gate_weights must be provided when top_k > 1")

    # Fast path: top_k == 1
    if top_k == 1:
        # If gate_per_token is provided (2D), use its first weight per token as per-token scalar
        if gate_per_token is not None:
            # gather per-candidate weights using the token index
            # gate_per_token[:, 0] is shape [out_len]
            per_token_first = gate_per_token[:, 0].to(device=device, dtype=dtype)
            gate_per_candidate = per_token_first[local_recv_idx]

        if gate_per_candidate is None:
            # Simple sum of candidate vectors into out by index
            # index_add_ sums rows with duplicate indices
            out.index_add_(0, local_recv_idx, local_recv_out)
        else:
            # Multiply each row by its scalar weight then add
            if gate_per_candidate.numel() != R:
                raise RuntimeError("Internal: gate_per_candidate length mismatch")
            src = local_recv_out * gate_per_candidate.unsqueeze(-1)
            out.index_add_(0, local_recv_idx, src)
        return out

    # top_k > 1: group candidates per token using tensor ops
    # Find unique tokens among local_recv_idx
    unique_tokens = torch.unique(local_recv_idx)

    # For each token, gather candidates and combine
    for token_t in unique_tokens:
        token = int(token_t.item())  # small scalar conversion
        if not (0 <= token < out_len):
            continue
        sel_mask = (local_recv_idx == token_t)  # [R]
        vals = local_recv_out[sel_mask]  # [c, D_out]
        c = vals.shape[0]
        if c == 0:
            continue
        k = min(c, top_k)

        # Determine weights for these candidates
        if gate_per_candidate is not None:
            w = gate_per_candidate[sel_mask]  # [c]
            if k == c:
                chosen_vals = vals
                chosen_w = w
            else:
                # pick top-k candidates by weight
                topk_vals, topk_idx = torch.topk(w, k=k, largest=True, sorted=False)
                chosen_vals = vals[topk_idx]
                chosen_w = topk_vals
        elif gate_per_token is not None:
            provided_k = gate_per_token.shape[1]
            take_k = min(k, provided_k)
            # take first take_k weights and corresponding first take_k candidates (if available)
            token_ws = gate_per_token[token][:take_k].to(device=device, dtype=dtype)
            chosen_vals = vals[:take_k]
            chosen_w = token_ws
            if take_k < k:
                pad_count = k - take_k
                # pad with ones and take next candidate vectors
                pad_w = torch.ones((pad_count,), device=device, dtype=dtype)
                chosen_w = torch.cat([chosen_w, pad_w], dim=0)
                # append the next candidate vectors (if available)
                chosen_vals = torch.cat([chosen_vals, vals[take_k:take_k + pad_count]], dim=0)
        else:
            # Uniform weights across the first k candidates
            chosen_vals = vals[:k]
            chosen_w = torch.ones((k,), device=device, dtype=dtype)

        # Normalize weights
        wsum = float(chosen_w.sum().item())
        if wsum == 0.0:
            continue
        chosen_w = chosen_w / wsum
        contrib = (chosen_vals * chosen_w.unsqueeze(-1)).sum(dim=0)
        out[token] = contrib

    return out
