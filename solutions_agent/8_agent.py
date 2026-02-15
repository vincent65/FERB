import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem


@triton.jit
def alltoall_send_kernel(
    remote_ptr,      # Pointer to destination PE's symmetric buffer (flattened [n_pes, n_elems])
    src_ptr,         # Pointer to local source row (flattened [n_elems])
    dest_offset,     # Runtime offset in elements within remote_ptr to write to (my_pe * n_elems)
    n_elems,         # runtime number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elems

    # Provide alignment/coalescing hints
    tl.assume_ptr_alignment(src_ptr, 16)
    tl.assume_ptr_alignment(remote_ptr, 16)

    vals = tl.load(src_ptr + offs, mask=mask, other=0)
    tl.store(remote_ptr + dest_offset + offs, vals, mask=mask)


class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream
    def __cuda_stream__(self):
        return (0, self.pt_stream.cuda_stream)


@torch.no_grad()
def solution(tensor: torch.Tensor) -> torch.Tensor:
    """
    NVSHMEM/Triton All-to-All: Each rank sends different data to every other rank.

    Input:
        tensor: CUDA tensor on current device with shape [world_size, *chunk_shape].
                tensor[i] is the data to send to rank i.
    Output:
        CUDA tensor of the same shape where output[i] contains the data that rank i sent to this rank.
    """
    assert tensor.is_cuda, "Input must be a CUDA tensor"

    # NVSHMEM world info
    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()

    assert tensor.shape[0] == n_pes, f"First dimension ({tensor.shape[0]}) must equal world_size ({n_pes})"

    # Ensure contiguous and flatten chunk dims
    tensor = tensor.contiguous()
    chunk_shape = tuple(tensor.shape[1:])
    send_2d = tensor.view(n_pes, -1)
    n_elems = send_2d.shape[1]

    # Trivial fast paths
    if n_pes == 1:
        # With a single PE, the output equals the input
        return tensor.clone()
    if n_elems == 0:
        return torch.empty_like(tensor)

    # Allocate symmetric output buffer: [n_pes, n_elems]
    out_sym = nvshmem.tensor((n_pes, n_elems), dtype=tensor.dtype)

    # Host-side world barrier to ensure symmetric allocations are globally visible before device work
    nvshmem.barrier(nvshmem.Teams.WORLD)

    # Use current CUDA stream for all operations
    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)

    # Triton launch configuration tuned for bandwidth-bound copies
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elems, BLOCK_SIZE),)
    num_warps = 8
    num_stages = 2

    dest_row_offset = my_pe * n_elems

    # Launch one kernel per destination on the current stream
    for dest_pe in range(n_pes):
        remote_full = nvshmem.get_peer_tensor(out_sym, dest_pe)
        src_row = send_2d[dest_pe]
        alltoall_send_kernel[grid](
            remote_full,           # remote_ptr
            src_row,               # src_ptr
            dest_row_offset,       # dest_offset (elements)
            n_elems,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
            num_stages=num_stages,
        )

    # Device-side world barrier to ensure all PEs completed their writes before reading results
    nvshmem.barrier(nvshmem.Teams.WORLD, stream_wrapper)

    # Materialize result locally and reshape back
    out = out_sym.view((n_pes,) + chunk_shape).clone()

    # Free symmetric memory
    nvshmem.free_tensor(out_sym)

    return out
