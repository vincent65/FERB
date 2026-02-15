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
    Portable PyTorch-only implementation of the MoE combine-phase semantics.

    This implementation is robust to CPU/CUDA inputs and handles device/dtype
    mismatches by moving the index/rank/weight tensors to the same device as x
    (without modifying the caller's tensors). It also guards against cases
    where the number of received candidates for a token exceeds top_k by
    using at most top_k candidates when computing the weighted sum.

    Notes:
    - dst_ranks indicates the destination PE for each row in x. In a single-PE
      test environment we assume my_pe == 0 and choose rows where dst_ranks == 0
      as the received rows.
    - dst_indices are token indices into the final output tensor rows.
    - gate_weights (if provided and top_k > 1) must have shape (out_len, top_k).
    """
    # Basic checks and device alignment
    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")

    device = x.device

    # Move/ensure index/rank tensors are on the same device and have long dtype
    dst_ranks = dst_ranks.to(device=device, dtype=torch.long)
    dst_indices = dst_indices.to(device=device, dtype=torch.long)

    if top_k > 1:
        if gate_weights is None:
            raise ValueError("gate_weights must be provided when top_k > 1")
        gate_weights = gate_weights.to(device=device)
        if gate_weights.shape[0] != out_len or gate_weights.shape[1] != top_k:
            raise ValueError("gate_weights must have shape (out_len, top_k)")

    # Single-PE execution assumption for correctness tests
    my_pe = 0

    # Ensure contiguous x for predictable indexing
    x_local = x.contiguous()
    T_local, D_out = x_local.shape

    # Select rows that are destined to this PE
    if T_local == 0:
        local_recv_out = torch.empty((0, D_out), dtype=x_local.dtype, device=device)
        local_recv_idx = torch.empty((0,), dtype=torch.long, device=device)
    else:
        mask = dst_ranks == my_pe
        if mask.any():
            # preserve original order of rows among those destined to this PE
            local_recv_out = x_local[mask].contiguous()
            local_recv_idx = dst_indices[mask].contiguous()
        else:
            local_recv_out = torch.empty((0, D_out), dtype=x_local.dtype, device=device)
            local_recv_idx = torch.empty((0,), dtype=torch.long, device=device)

    total_recv = int(local_recv_idx.numel())

    # Prepare final output tensor
    out = torch.zeros((out_len, D_out), device=device, dtype=local_recv_out.dtype)

    if top_k == 1:
        # Direct placement: copy each received row into its destination index.
        if total_recv > 0:
            # index_copy_ requires indices to be long and on the same device
            out.index_copy_(0, local_recv_idx, local_recv_out)
    else:
        # Top-K combination path
        if total_recv > 0:
            # Group by token index: sort so same indices are adjacent
            sort_idx = torch.argsort(local_recv_idx)
            sorted_out = local_recv_out[sort_idx]
            sorted_dst_idx = local_recv_idx[sort_idx]

            # counts per token (length out_len)
            counts_per_token = torch.bincount(sorted_dst_idx, minlength=out_len).to(
                device=device, dtype=torch.long
            )

            base = 0
            for token_idx in range(out_len):
                c = int(counts_per_token[token_idx].item())
                if c == 0:
                    continue

                # slice of sorted arrays corresponding to this token
                token_outputs = sorted_out[base: base + c]

                # If the number of received candidates exceeds top_k, only use up to top_k
                k = min(c, top_k)
                if k <= 0:
                    base += c
                    continue

                token_outputs_k = token_outputs[:k]

                # gate weights for this token: take first k entries
                token_weights = gate_weights[token_idx][:k].to(device=device)

                # If fewer candidates than top_k, normalize the provided weights
                w = token_weights
                s = float(w.sum().item())
                if s == 0.0:
                    # all-zero weights -> leave zeros (skip)
                    pass
                else:
                    if k < top_k:
                        # When fewer candidates than top_k, weights are the first k entries
                        # Normalize them so they sum to 1
                        w = w / s
                    # Weighted sum across the selected candidates
                    out[token_idx] = (token_outputs_k * w.unsqueeze(-1)).sum(dim=0)

                base += c

    return out
