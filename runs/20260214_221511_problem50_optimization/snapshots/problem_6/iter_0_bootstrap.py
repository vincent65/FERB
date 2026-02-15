import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem


@triton.jit
def gather_copy_kernel(
    remote_ptr,      # Pointer to destination PE's symmetric output buffer
    local_ptr,       # Pointer to local input data (flattened)
    dest_offset,     # Offset (in elements) within remote buffer where this rank writes
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    val = tl.load(local_ptr + idx, mask=mask)
    tl.store(remote_ptr + dest_offset + idx, val, mask=mask)


class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream
        self.handle = pt_stream.cuda_stream

    def __cuda_stream__(self):
        stream_id = self.pt_stream.cuda_stream
        return (0, stream_id)


@torch.no_grad()
def solution(tensor: torch.Tensor, dst: int = 0) -> torch.Tensor:
    """
    NVSHMEM/Triton Gather: All ranks send their tensor to the destination rank.
    On dst: returns tensor of shape [world_size, *chunk_shape]. Others: return input tensor unchanged.
    Preconditions:
      - nvshmem has been initialized externally
      - tensor is a contiguous CUDA tensor, identical shape/dtype across ranks
    """
    assert tensor.is_cuda, "Input must be a CUDA tensor"
    assert tensor.is_contiguous(), "Input tensor must be contiguous"

    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()
    assert 0 <= dst < n_pes, "dst must be a valid rank"

    chunk_shape = tensor.shape
    chunk_elems = tensor.numel()
    dtype = tensor.dtype

    # Symmetric output buffer sized for dst to receive all ranks' chunks (flattened)
    total_elems = n_pes * chunk_elems
    out_sym = nvshmem.tensor((total_elems,), dtype=dtype)

    local_flat = tensor.flatten().contiguous()

    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)

    # Ensure all PEs have allocated their symmetric buffers before remote access
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Remote pointer to dst's buffer (local if my_pe == dst)
    dst_view_full = nvshmem.get_peer_tensor(out_sym, dst)

    # Launch Triton kernel to copy our local chunk into dst's buffer at our slot
    if chunk_elems > 0:
        BLOCK_SIZE = 256
        grid = (triton.cdiv(chunk_elems, BLOCK_SIZE),)
        dest_offset = my_pe * chunk_elems
        gather_copy_kernel[grid](
            dst_view_full,
            local_flat,
            dest_offset,
            n_elements=chunk_elems,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Ensure all remote writes complete
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    if my_pe == dst:
        # Materialize result and reshape
        out_flat = out_sym.clone().detach()
        result = out_flat.reshape((n_pes,) + tuple(chunk_shape))
        nvshmem.free_tensor(out_sym)
        return result
    else:
        # Other ranks return input unchanged
        nvshmem.free_tensor(out_sym)
        return tensor
