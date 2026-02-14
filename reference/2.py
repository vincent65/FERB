import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    tensor: torch.Tensor,  # Input tensor: Must be a contiguous CUDA tensor on the current rank's device.
                           # Shape: Arbitrary (any number of dimensions, e.g., [M, N], [B, C, H, W], etc.)
                           # Dtype: Any numeric dtype (float32, float16, int32, etc.)
                           # Device: Must be on the CUDA device corresponding to this rank (torch.cuda.current_device())
                           # All ranks must provide tensors of identical shape and dtype.
) -> torch.Tensor:
    """
    Simple NCCL all-gather over all ranks.
    
    Performs an all-gather collective operation that stacks the input tensor from each rank
    along the leading dimension. All ranks must participate and provide tensors of
    identical shape and dtype.
    
    Preconditions:
        - torch.distributed must be initialized with NCCL backend
        - Input tensor must be on the current CUDA device (torch.cuda.current_device())
        - All ranks must call this function with tensors of the same shape and dtype
    
    Returns:
        A new tensor with shape [world_size, ...input.shape...] where the first dimension is
        the rank (0..world_size-1), and each slice is the input from the corresponding rank.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    input_shape = tensor.shape
    world_size = dist.get_world_size()
    out_shape = (world_size,) + input_shape
    out = torch.empty(out_shape, dtype=tensor.dtype, device=tensor.device)
    dist.all_gather_into_tensor(out, tensor)
    return out
