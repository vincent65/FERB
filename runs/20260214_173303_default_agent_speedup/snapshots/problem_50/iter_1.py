import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem


@triton.jit
def _memcopy_kernel(
    dst_ptr,           # pointer to destination flat buffer
    src_ptr,           # pointer to source flat buffer
    n_elems,           # runtime number of elements to copy (scalar)
    dst_offset,        # runtime element offset applied to dst_ptr
    src_offset,        # runtime element offset applied to src_ptr
    BLOCK: tl.constexpr  # compile-time block size
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    # load from source (flattened view) and store into destination (flattened view)
    src = tl.load(src_ptr + src_offset + offs, mask=mask)
    tl.store(dst_ptr + dst_offset + offs, src, mask=mask)


@triton.jit
def _memcopy_kernel_int64(
    dst_ptr,
    src_ptr,
    n_elems,
    dst_offset,
    src_offset,
    BLOCK: tl.constexpr
):
    # specialized for int64/long (indices and counts)
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n_elems
    src = tl.load(src_ptr + src_offset + offs, mask=mask)
    tl.store(dst_ptr + dst_offset + offs, src, mask=mask)


class PyTorchStreamWrapper:
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        # return a tuple (device_index, stream_handle) expected by some NVSHMEM bindings
        return (0, int(self.pt_stream.cuda_stream))


@torch.no_grad()
def solution(
    x: torch.Tensor,  # [T_local, D_out]
    dst_ranks: torch.Tensor,  # [T_local], long
    dst_indices: torch.Tensor,  # [T_local], long
    out_len: int,
    top_k: int = 1,
    gate_weights: torch.Tensor | None = None,
    capacity_factor: float | None = None,
) -> torch.Tensor:
    """
    NVSHMEM+Triton optimized combine phase for MoE AllToAll routing.

    This function mirrors the original algorithm but fixes Triton kernel signatures and
    the kernel callsites so runtime values (n_elems and offsets) are passed as runtime
    arguments while the BLOCK size remains a compile-time constant.
    """
    # Basic checks
    assert dst_ranks.dtype == torch.long and dst_indices.dtype == torch.long
    assert x.is_cuda and dst_ranks.is_cuda and dst_indices.is_cuda
    if top_k > 1:
        assert gate_weights is not None and gate_weights.is_cuda
        assert gate_weights.shape == (out_len, top_k)

    my_pe = nvshmem.my_pe()
    n_pes = nvshmem.n_pes()
    device = x.device

    T_local, D_out = x.shape

    # Ensure contiguous
    x_local = x.contiguous()
    dst_ranks = dst_ranks.contiguous()
    dst_indices = dst_indices.contiguous()

    # 1) compute send_counts per destination rank
    send_counts = torch.bincount(dst_ranks, minlength=n_pes).to(device=device, dtype=torch.long)

    # 2) allocate symmetric counts matrix (n_pes x n_pes) and publish our row
    counts_all = nvshmem.tensor((n_pes, n_pes), dtype=torch.long)
    counts_all.zero_()
    # write our row
    counts_all[my_pe] = send_counts

    # Barrier so everyone has written their row
    torch.cuda.synchronize()
    stream_wrapper = PyTorchStreamWrapper(torch.cuda.current_stream())
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # 3) compute recv counts for this rank (how many items we will receive from each source)
    recv_counts_sources = counts_all[:, my_pe].to(device=device, dtype=torch.long).contiguous()
    total_recv = int(recv_counts_sources.sum().item())

    # Allocate symmetric recv buffers for outputs and indices
    recv_out_sym = nvshmem.tensor((total_recv, D_out), dtype=x.dtype)
    recv_idx_sym = nvshmem.tensor((total_recv,), dtype=torch.long)
    recv_out_sym.zero_()
    recv_idx_sym.zero_()

    # 4) Pack local outputs by destination rank (stable sort by dst_ranks)
    if T_local > 0:
        perm = torch.argsort(dst_ranks, stable=True)
        packed_out = x_local.index_select(0, perm).contiguous()
        packed_idx = dst_indices.index_select(0, perm).contiguous()
    else:
        packed_out = torch.empty((0, D_out), dtype=x.dtype, device=device)
        packed_idx = torch.empty((0,), dtype=torch.long, device=device)

    # compute local send offsets per destination
    send_prefix = torch.zeros_like(send_counts)
    if n_pes > 0:
        send_prefix[1:] = torch.cumsum(send_counts[:-1], dim=0)

    # Bring counts_all locally for offset computations
    counts_all_local = counts_all.to(device=device)

    # 5) For each destination r with send_count>0, compute dest_offset for this source (my_pe)
    BLOCK = 1024  # compile-time block size for Triton kernel

    for dest in range(n_pes):
        sc = int(send_counts[dest].item())
        if sc == 0:
            continue
        # source-local start index in packed arrays (in rows)
        src_start_row = int(send_prefix[dest].item())
        src_n_rows = sc

        # compute destination offset rows within dest's recv buffer
        if my_pe == 0:
            dest_offset_rows = 0
        else:
            dest_offset_rows = int(counts_all_local[:my_pe, dest].sum().item())

        # Get peer view of the destination's recv_out_sym and recv_idx_sym
        dest_recv_out = nvshmem.get_peer_tensor(recv_out_sym, dest)
        dest_recv_idx = nvshmem.get_peer_tensor(recv_idx_sym, dest)

        # Flattened element counts for outputs (float elements)
        n_elems_out = src_n_rows * D_out
        src_elem_offset = src_start_row * D_out
        dst_elem_offset = dest_offset_rows * D_out

        # Launch Triton kernel to copy packed_out[src_start_row:src_start_row+src_n_rows]
        # -> dest_recv_out[dst_offset:dst_offset+src_n_rows]
        if n_elems_out > 0:
            grid = (triton.cdiv(n_elems_out, BLOCK),)
            # pass runtime args as normal python ints; BLOCK is compile-time
            _memcopy_kernel[grid](
                dest_recv_out,
                packed_out,
                n_elems_out,
                dst_elem_offset,
                src_elem_offset,
                BLOCK,
            )

        # Copy indices (1D int64): src_n_rows elements
        if src_n_rows > 0:
            grid_idx = (triton.cdiv(src_n_rows, BLOCK),)
            _memcopy_kernel_int64[grid_idx](
                dest_recv_idx,
                packed_idx,
                src_n_rows,
                dest_offset_rows,
                src_start_row,
                BLOCK,
            )

    # 6) Synchronize and barrier to ensure all remote writes completed
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # 7) Now recv_out_sym and recv_idx_sym contain the rows destined to this rank
    local_recv_out = recv_out_sym
    local_recv_idx = recv_idx_sym

    # Build final output tensor
    out = torch.zeros((out_len, D_out), device=device, dtype=local_recv_out.dtype)

    if top_k == 1:
        # Direct placement
        if total_recv > 0:
            out.index_copy_(0, local_recv_idx, local_recv_out)
    else:
        # Top-K combination
        if total_recv > 0:
            sort_idx = torch.argsort(local_recv_idx)
            sorted_out = local_recv_out[sort_idx]
            sorted_dst_idx = local_recv_idx[sort_idx]
            counts_per_token = torch.bincount(local_recv_idx, minlength=out_len).to(device=device, dtype=torch.long)

            for token_idx in range(out_len):
                c = int(counts_per_token[token_idx].item())
                if c == 0:
                    continue
                mask = sorted_dst_idx == token_idx
                token_outputs = sorted_out[mask]
                token_weights = gate_weights[token_idx][:c].to(device=device)
                if c == top_k:
                    out[token_idx] = (token_outputs * token_weights.unsqueeze(-1)).sum(dim=0)
                else:
                    w = token_weights
                    s = w.sum()
                    if s.item() == 0:
                        continue
                    w = w / s
                    out[token_idx] = (token_outputs * w.unsqueeze(-1)).sum(dim=0)

    # 8) cleanup symmetric buffers
    nvshmem.free_tensor(counts_all)
    nvshmem.free_tensor(recv_out_sym)
    nvshmem.free_tensor(recv_idx_sym)

    return out
