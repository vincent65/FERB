import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    X_hat: torch.Tensor,  # Input: normalized activations on *this rank*
                          # Shape: [B, H]  (B = local tokens/microbatch on this GPU, H = hidden dim)
                          # Dtype: float16/float32/bfloat16 (typical)
                          # Device: CUDA device for this rank, contiguous
    dY: torch.Tensor,     # Input: upstream gradient w.r.t. LayerNorm output on *this rank*
                          # Shape: [B, H] (must match X_hat)
                          # Dtype/Device: same as X_hat
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    LayerNorm backward param-grad aggregation (Data Parallel, NCCL AllReduce baseline).

    Math (per feature h):
        d_beta_local[h]  = sum_b dY[b, h]
        d_gamma_local[h] = sum_b dY[b, h] * X_hat[b, h]     (element-wise multiply, then sum over tokens)

    Distributed (data parallel):
        Each rank computes local partial grads from its local shard of tokens (B rows).
        Then we AllReduce(SUM) across ranks to obtain the *global* gradients over all tokens.

    Returns:
        (d_gamma, d_beta) where each is shape [H] and is the GLOBAL (all-ranks summed) gradient.

    Preconditions:
        - torch.distributed initialized with NCCL backend
        - All ranks call this with the same H (and typically same B, but B can differ if you handle padding)
        - X_hat and dY are contiguous CUDA tensors on the current rank's device
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert X_hat.is_cuda and dY.is_cuda, "Inputs must be CUDA tensors"
    assert X_hat.is_contiguous() and dY.is_contiguous(), "Inputs must be contiguous"
    assert X_hat.shape == dY.shape, "X_hat and dY must have the same shape [B, H]"

    B, H = X_hat.shape

    # Local partial grads (no communication yet): [H]
    # d_beta_local[h] = sum over tokens b of dY[b,h]
    d_beta = dY.sum(dim=0)

    # d_gamma_local[h] = sum over tokens b of dY[b,h] * X_hat[b,h]
    # (element-wise multiply, then sum over tokens)
    d_gamma = (dY * X_hat).sum(dim=0)

    # NCCL AllReduce: sum local partials across ranks -> global grads on every rank (in-place)
    dist.all_reduce(d_beta,  op=dist.ReduceOp.SUM)
    dist.all_reduce(d_gamma, op=dist.ReduceOp.SUM)

    return d_gamma, d_beta