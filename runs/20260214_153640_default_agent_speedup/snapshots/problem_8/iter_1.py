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

    # Hints to help the compiler generate wide, coalesced transactions
    tl.multiple_of(offs, 16)

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
    chunk_shape = tuple(tensor.shape[1:])
    send_2d = tensor.view(n_pes, -1)
    n_elems = send_2d.shape[1]

    # Fast-exit for empty payloads
    if n_elems == 0:
        return torch.empty_like(tensor)

    # Allocate symmetric output buffer: [n_pes, n_elems]
    out_sym = nvshmem.tensor((n_pes, n_elems), dtype=tensor.dtype)

    # Stream wrapper for NVSHMEM barriers
    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)

    # Ensure symmetric allocations are globally visible before transfers
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Triton launch configuration tuned for bandwidth-bound copies
    # Larger tile and more warps improve copy throughput on modern GPUs
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elems, BLOCK_SIZE),)
    num_warps = 8
    num_stages = 2

    # Prepare remote views once to avoid per-iteration control-plane overhead
    peer_views = [nvshmem.get_peer_tensor(out_sym, pe) for pe in range(n_pes)]

    # We'll overlap remote writes across multiple CUDA streams
    max_concurrency = min(n_pes, 4)  # 2-4 generally saturates NVLink/PCIe without oversubscription
    streams = [torch.cuda.Stream() for _ in range(max_concurrency)]

    dest_row_offset = my_pe * n_elems

    # Dispatch one kernel per destination on a small set of streams in round-robin
    for dest_pe in range(n_pes):
        s = streams[dest_pe % max_concurrency]
        remote_full = peer_views[dest_pe]
        src_row = send_2d[dest_pe]
        with torch.cuda.stream(s):
            alltoall_send_kernel[grid](
                remote_full,           # remote_ptr
                src_row,               # src_ptr
                dest_row_offset,       # dest_offset (elements)
                n_elems=n_elems,
                BLOCK_SIZE=BLOCK_SIZE,
                num_warps=num_warps,
                num_stages=num_stages,
            )

    # Ensure all enqueued sends on auxiliary streams are done before the barrier
    for s in streams:
        s.synchronize()

    # Global barrier to ensure all PEs completed their writes before reading results
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Materialize result locally and reshape back
    out = out_sym.clone().view((n_pes,) + chunk_shape)

    # Free symmetric memory
    nvshmem.free_tensor(out_sym)

    return out
