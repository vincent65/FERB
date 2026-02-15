import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem


@triton.jit
def copy_kernel(
    output_ptr,
    input_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Simple copy kernel - copies data from input to output buffer.
    """
    pid = tl.program_id(0)
    
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    
    val = tl.load(input_ptr + idx, mask=mask)
    tl.store(output_ptr + idx, val, mask=mask)


class PyTorchStreamWrapper:
    """Wrapper to convert PyTorch stream to CUDA Python stream protocol."""
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream
    
    def __cuda_stream__(self):
        return (0, self.pt_stream.cuda_stream)


@torch.no_grad()
def solution(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """
    Broadcast using Triton kernels with NVSHMEM.
    
    The source rank sends its tensor to all other ranks.
    All ranks end up with identical output (the source rank's data).
    
    Args:
        tensor: Input CUDA tensor. Shape: Arbitrary.
        src: Source rank that broadcasts its tensor (default: 0)
        
    Returns:
        Tensor of the same shape containing the data from the source rank.
        All ranks will have identical output.
    """
    assert tensor.is_cuda, "Input must be a CUDA tensor"
    assert tensor.dtype == torch.float32, "Only float32 is supported for now"
    
    # Get PE info
    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()
    
    assert 0 <= src < n_pes, f"Invalid source rank {src}, must be in [0, {n_pes})"
    
    # Get input shape info
    original_shape = tensor.shape
    n_elements = tensor.numel()
    
    # Flatten input tensor
    input_flat = tensor.flatten().contiguous()
    
    # Allocate symmetric memory - this will hold the broadcast data
    # All PEs allocate this
    output_sym = nvshmem.tensor(n_elements, dtype=torch.float32)
    
    # Get CUDA stream
    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)
    
    # Calculate grid size for Triton kernel
    BLOCK_SIZE = 512
    grid_size = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Step 1: Source PE writes its data to its own symmetric buffer
    # This ensures the data is in symmetric memory before broadcasting
    if my_pe == src:
        output_sym[:] = input_flat
    
    # Synchronize to ensure source has written its data
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)
    
    # Step 2: ALL PEs read from the source PE's symmetric buffer
    # This is a "pull" model - everyone reads from source instead of source pushing to everyone
    # This ensures all PEs do work and reach barriers at similar times
    
    # Get a view of the source PE's symmetric buffer
    src_view = nvshmem.get_peer_tensor(output_sym, src)
    
    # All PEs (including source) copy from source's buffer to their own
    # For source PE, this is a no-op copy from itself to itself, but keeps execution symmetric
    copy_kernel[grid_size](
        output_sym,
        src_view,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Synchronize to ensure all copies are complete
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)
    
    # Copy result from symmetric memory to output tensor
    result = output_sym.clone().detach().reshape(original_shape)
    
    # Clean up
    nvshmem.free_tensor(output_sym)
    
    return result
