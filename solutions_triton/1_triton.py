"""
Triton All-Reduce (SUM) Solution using NVSHMEM

This solution uses Triton kernels with NVSHMEM for P2P atomics.
Each rank contributes its local tensor to the symmetric buffers of all other ranks.
"""

import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem

@triton.jit
def allreduce_sum_kernel(
    output_ptr,      # Pointer to the output buffer (symmetric memory)
    input_ptr,       # Pointer to local input data
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    All-reduce kernel using Triton.
    
    This kernel performs an atomic addition into the output_ptr.
    In a distributed context, output_ptr will be a peer-tensor view.
    """
    pid = tl.program_id(0)
    
    # Each thread handles a block of elements
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load local input data
    local_val = tl.load(input_ptr + offsets, mask=mask)
    
    # Perform atomic addition into the output buffer
    # Note: tl.atomic_add is used to ensure thread-safety and PE-safety 
    # when multiple PEs write to the same location.
    tl.atomic_add(output_ptr + offsets, local_val, mask=mask)

class PyTorchStreamWrapper:
    """Wrapper to convert PyTorch stream to CUDA Python stream protocol."""
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream
    
    def __cuda_stream__(self):
        return (0, self.pt_stream.cuda_stream)

def solution(tensor: torch.Tensor) -> torch.Tensor:
    """
    All-reduce (SUM) using Triton kernels with NVSHMEM.
    
    Args:
        tensor: Input CUDA tensor. Shape: Arbitrary.
        
    Returns:
        Tensor of the same shape containing the element-wise sum across all ranks.
    """
    assert tensor.is_cuda, "Input must be a CUDA tensor"
    assert tensor.dtype == torch.float32, "Only float32 is supported for atomic_add in this kernel"
    
    # 1. Get PE info
    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()
    
    # 2. Setup shapes and buffers
    original_shape = tensor.shape
    n_elements = tensor.numel()
    input_flat = tensor.flatten().contiguous()
    
    # Allocate symmetric memory for the reduction result
    # We initialize it to zero so we can accumulate via atomic_add
    output_sym = nvshmem.tensor(n_elements, dtype=torch.float32)
    output_sym.zero_()
    
    # Wrap stream for NVSHMEM barriers
    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)
    
    # 3. Define Grid
    BLOCK_SIZE = 512
    grid = lambda META: (triton.cdiv(n_elements, META['BLOCK_SIZE']),)
    
    # 4. Synchronize before starting (ensure all PEs have zeroed their output_sym)
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)
    
    # 5. Distributed Accumulation
    # Each PE adds its local data to EVERY other PE's output_sym buffer
    for target_pe in range(n_pes):
        # Get a pointer to the symmetric buffer on the target PE
        # For the local PE, this just returns the local pointer
        target_view = nvshmem.get_peer_tensor(output_sym, target_pe)
        
        # Launch Triton kernel to perform atomic addition on the target PE
        allreduce_sum_kernel[grid](
            target_view,
            input_flat,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE
        )
    
    # 6. Synchronize
    # This ensures all remote atomic additions from all PEs have landed
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)
    
    # 7. Finalize result
    # Clone the result out of symmetric memory and reshape
    result = output_sym.clone().detach().reshape(original_shape)
    
    # Clean up
    nvshmem.free_tensor(output_sym)
    
    return result