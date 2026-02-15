import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem


@triton.jit
def broadcast_kernel(
    output_ptr,
    input_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Broadcast kernel - copies data from input to output buffer.
    Used by source PE to write to all PEs' symmetric buffers.
    """
    pid = tl.program_id(0)
    
    # Each thread handles a block of elements
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    
    # Load from input (source PE's data)
    val = tl.load(input_ptr + idx, mask=mask)
    
    # Store to output (could be local or remote symmetric buffer)
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
    
    # Allocate symmetric memory for the broadcast result
    output_sym = nvshmem.tensor((n_elements,), dtype=torch.float32)
    
    # Get CUDA stream
    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)
    
    # Calculate grid size for Triton kernel
    BLOCK_SIZE = 512
    grid_size = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Synchronize before starting
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)
    
    # Only the source PE performs the broadcast
    if my_pe == src:
        # Source PE writes to all PEs' symmetric buffers (including its own)
        for dest_pe in range(n_pes):
            # Get peer tensor view of the destination PE's symmetric buffer
            dest_view = nvshmem.get_peer_tensor(output_sym, dest_pe)
            
            # Launch Triton kernel to copy our data to the destination PE
            broadcast_kernel[grid_size](
                dest_view,
                input_flat,
                n_elements=n_elements,
                BLOCK_SIZE=BLOCK_SIZE
            )
    
    # Synchronize to ensure broadcast is complete
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)
    
    # Copy result from symmetric memory to output tensor
    result = output_sym.clone().detach().reshape(original_shape)
    
    # Clean up
    nvshmem.free_tensor(output_sym)
    
    return result
