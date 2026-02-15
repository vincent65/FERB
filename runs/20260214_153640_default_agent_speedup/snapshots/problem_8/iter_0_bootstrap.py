import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem


@triton.jit
def alltoall_send_kernel(
    remote_ptr,      # Pointer to destination PE's symmetric buffer (flattened [n_pes, n_elems])
    src_ptr,         # Pointer to local source row (flattened [n_elems])
    dest_offset,     # Runtime offset in elements within remote_ptr to write to (my_pe * n_elems)
    n_elems: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elems

    vals = tl.load(src_ptr + offs, mask=mask)
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
    chunk_shape = tensor.shape[1:]
    send_2d = tensor.view(n_pes, -1)
    n_elems = send_2d.shape[1]

    # Allocate symmetric output buffer: [n_pes, n_elems]
    out_sym = nvshmem.tensor((n_pes, n_elems), dtype=tensor.dtype)

    # Stream wrapper for NVSHMEM barriers
    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)

    # Synchronize before starting transfers to ensure symmetric buffers are ready
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Triton launch configuration
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elems, BLOCK_SIZE),)

    # For each destination PE, send the corresponding local row to that PE's output buffer
    # On destination PE 'dest', we write into row 'my_pe'
    dest_row_offset = my_pe * n_elems

    for dest_pe in range(n_pes):
        # Remote view of destination PE's symmetric buffer
        dest_view_full = nvshmem.get_peer_tensor(out_sym, dest_pe)
        # Local source row to send to dest_pe
        src_row = send_2d[dest_pe]

        # Launch kernel to copy src_row into destination's buffer at row 'my_pe'
        alltoall_send_kernel[grid](
            dest_view_full,  # remote_ptr
            src_row,         # src_ptr
            dest_row_offset, # dest_offset (elements)
            n_elems=n_elems,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Ensure all remote writes complete
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Materialize result locally and reshape back
    out = out_sym.clone().view((n_pes,) + chunk_shape)

    # Free symmetric memory
    nvshmem.free_tensor(out_sym)

    return out
