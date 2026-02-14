"""
Triton All-Gather Solution using NVSHMEM

This solution uses Triton kernels with NVSHMEM for P2P communication.
Each rank contributes its local tensor, and after the all-gather,
every rank has a copy of all tensors concatenated along a new first dimension.
"""

import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem

# No global state needed - worker handles initialization


@triton.jit
def allgather_kernel(
    output_ptr,      # Pointer to output buffer (symmetric memory)
    input_ptr,       # Pointer to local input data
    n_elements: tl.constexpr,  # Number of elements per rank
    my_pe: tl.constexpr,      # My PE rank
    n_pes: tl.constexpr,      # Total number of PEs
    BLOCK_SIZE: tl.constexpr
):
    """
    All-gather kernel using Triton.
    
    Each PE copies its local data to all other PEs' segments in the output buffer.
    Memory layout: output[pe_idx * n_elements + element_idx]
    """
    pid = tl.program_id(0)
    
    # Each thread handles one element
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    
    # Load local input data
    local_val = tl.load(input_ptr + idx, mask=mask)
    
    # Write to our own segment (no remote access needed for local copy)
    my_offset = my_pe * n_elements
    tl.store(output_ptr + my_offset + idx, local_val, mask=mask)


@triton.jit
def allgather_remote_kernel_with_offset(
    remote_ptr,      # Pointer to remote PE's full buffer
    local_ptr,       # Pointer to local input data
    dest_offset,     # Offset in the remote buffer where we write (runtime value)
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Copy local data to a remote PE's memory segment at a specific offset.
    This kernel is launched for each remote PE.
    """
    pid = tl.program_id(0)
    
    # Each thread handles one element
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    
    # Load local input data
    local_val = tl.load(local_ptr + idx, mask=mask)
    
    # Store to remote memory at the correct offset
    # dest_offset is a runtime value, so we add it to the pointer
    tl.store(remote_ptr + dest_offset + idx, local_val, mask=mask)


class PyTorchStreamWrapper:
    """Wrapper to convert PyTorch stream to CUDA Python stream protocol."""
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream
        self.handle = pt_stream.cuda_stream
    
    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)


# init_triton and finalize_triton are now handled by worker_triton.py


def solution(tensor: torch.Tensor, stream=None) -> torch.Tensor:
    """
    All-gather using Triton kernels with NVSHMEM.
    
    Each PE contributes its local tensor, and after the all-gather,
    every PE has a copy of all tensors concatenated along a new first dimension.
    
    Args:
        tensor: Input CUDA tensor on current PE's device.
               Shape: [M, N] (or any shape)
        stream: Not used (kept for API compatibility)
        
    Returns:
        Tensor with shape [world_size, *input.shape] where each slice [i] 
        contains the input from PE i.
    """
    assert tensor.is_cuda, "Input must be a CUDA tensor"
    assert tensor.dtype == torch.float32, "Only float32 is supported for now"
    
    # Get PE info directly from nvshmem (initialized by worker)
    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()
    
    # Get input shape info
    input_shape = tensor.shape
    n_elements = tensor.numel()
    
    # Output shape: [world_size, *input_shape]
    out_shape = (n_pes,) + input_shape
    total_elements = n_pes * n_elements
    
    # Allocate symmetric memory for output
    # Use nvshmem.tensor() for symmetric allocation
    output_sym = nvshmem.tensor((total_elements,), dtype=torch.float32)
    
    # Flatten input tensor
    input_flat = tensor.flatten().contiguous()
    
    # Get CUDA stream
    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)
    
    # Calculate grid size for Triton kernel (must be a tuple)
    BLOCK_SIZE = 256
    grid_size = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Copy local data to our own segment in symmetric buffer
    my_offset = my_pe * n_elements
    output_sym[my_offset:my_offset + n_elements] = input_flat
    
    # Synchronize before starting remote copies
    torch.cuda.synchronize()
    
    # Barrier to ensure all PEs have written their local data
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)
    
    # Now copy our local data to all other PEs' segments using Triton kernels
    # For each remote PE, get a view of their memory segment and copy our data there
    for dest_pe in range(n_pes):
        if dest_pe == my_pe:
            continue  # Skip ourselves
        
        # Get peer tensor view of the entire symmetric buffer
        # Then we'll use an offset in the Triton kernel to write to the correct segment
        dest_view_full = nvshmem.get_peer_tensor(output_sym, dest_pe)
        
        # Calculate destination offset in the remote PE's buffer
        dest_offset = my_pe * n_elements
        
        # Launch Triton kernel to copy our local data to remote PE
        # We pass the full buffer and the offset, and the kernel will write at the correct location
        allgather_remote_kernel_with_offset[grid_size](
            dest_view_full,
            input_flat,
            dest_offset,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    # Synchronize all remote copies
    torch.cuda.synchronize()
    
    # Barrier to ensure all copies are complete
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)
    
    # Copy result from symmetric memory to output tensor
    output = output_sym.clone().detach()
    
    # Free symmetric memory
    nvshmem.free_tensor(output_sym)
    
    return output.reshape(out_shape)

