import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem


@triton.jit
def copy_flat_kernel(dst_ptr, src_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    vals = tl.load(src_ptr + idx, mask=mask)
    tl.store(dst_ptr + idx, vals, mask=mask)


class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        return (0, self.pt_stream.cuda_stream)


@torch.no_grad()
def solution(
    tensor: torch.Tensor,
    src: int = 0,
) -> torch.Tensor:
    """
    NVSHMEM/Triton-based Scatter.

    One designated source rank (src) holds a tensor stacked along dim 0 with world_size chunks.
    Each rank receives its corresponding chunk into a symmetric receive buffer, and returns it.

    Preconditions:
      - NVSHMEM is initialized externally
      - Input tensor is CUDA-contiguous on the current device
      - On src: tensor.shape == [world_size, *chunk_shape]
      - On others: tensor.shape == [*chunk_shape]
    """
    assert tensor.is_cuda, "Input tensor must be a CUDA tensor"

    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()
    assert 0 <= src < n_pes, "src must be a valid rank"

    # Determine chunk shape/numel and dtype
    if my_pe == src:
        assert tensor.dim() >= 1 and tensor.shape[0] == n_pes, "Source tensor must have world_size chunks"
        chunk_shape = tuple(tensor.shape[1:])
        chunk_numel = int(torch.tensor(chunk_shape).prod().item()) if len(chunk_shape) > 0 else 1
        src_tensor = tensor.contiguous().view(n_pes, -1)
    else:
        chunk_shape = tuple(tensor.shape)
        chunk_numel = tensor.numel() if tensor.numel() > 0 else 1
        src_tensor = None  # not used on non-src

    dtype = tensor.dtype

    # Allocate symmetric receive buffer (1D flat)
    recv_sym = nvshmem.tensor((chunk_numel,), dtype=dtype)

    # Stream wrapper for NVSHMEM barriers
    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)

    # Ensure all PEs have allocated symmetric buffers
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Launch copies from src to each rank's symmetric buffer
    if my_pe == src:
        BLOCK_SIZE = 1024
        grid = lambda META: (triton.cdiv(chunk_numel, META['BLOCK_SIZE']),)
        for dest_pe in range(n_pes):
            dest_view = nvshmem.get_peer_tensor(recv_sym, dest_pe)
            src_slice = src_tensor[dest_pe]
            copy_flat_kernel[grid](
                dest_view,
                src_slice,
                n_elements=chunk_numel,
                BLOCK_SIZE=BLOCK_SIZE,
            )

    # Synchronize to ensure data movement complete before reading recv_sym
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Materialize output on local rank and free symmetric memory
    out = recv_sym.clone().reshape(chunk_shape)
    nvshmem.free_tensor(recv_sym)

    return out
