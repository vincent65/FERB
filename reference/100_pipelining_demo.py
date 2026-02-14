# Pipelined / Chunked MoE Top-1 (overlapped compute + combine)
# Same semantics as 101 (AllToAll dispatch + compute + combine), but chunks the input
# tokens and overlaps the combine AllToAll of chunk i with the dispatch+compute of chunk i+1.
# Parameterized: num_chunks controls pipeline depth.

import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    x: torch.Tensor,                  # Original tokens on this rank. Shape: [T, D]. Contiguous CUDA.
                                      # All ranks must have the SAME T.
    expert_assignments: torch.Tensor,  # Per-token expert assignment. Shape: [T], dtype long.
                                      # Values in [0, world_size).
    w1: torch.Tensor,                  # This rank's expert first weight. Shape: [D_hidden, D].
    w2: torch.Tensor,                  # This rank's expert second weight. Shape: [D, D_hidden].
    num_chunks: int = 2,              # Number of chunks to pipeline. 1 = no overlap (same as 101).
) -> torch.Tensor:                    # Returns: [T, D].
    """
    Pipelined MoE forward (top-1) with chunked overlap.

    Tokens are split into `num_chunks` independent sub-batches. For each chunk:
        1. DISPATCH  (sync)  — AllToAll count exchange + payload.
        2. COMPUTE   (sync)  — Expert MLP on received tokens.
        3. COMBINE   (async) — AllToAll results back (async_op=True).

    The combine of chunk i runs in the background while chunk i+1's dispatch + compute
    execute, hiding communication latency behind computation.

    Timeline (num_chunks=3):
        chunk 0: dispatch → compute → combine(async) ──────────────┐
        chunk 1:                       dispatch → compute → combine(async) ──┐
        chunk 2:                                            dispatch → compute → combine(async)
                                                                              └── wait
    Preconditions:
        - torch.distributed initialized with NCCL backend.
        - One expert per rank. All ranks have the same T.
        - T must be divisible by num_chunks (or torch.chunk handles remainder).
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    T, D = x.shape

    # Chunk input tokens and their assignments
    x_chunks = torch.chunk(x.contiguous(), num_chunks, dim=0)
    assign_chunks = torch.chunk(expert_assignments, num_chunks, dim=0)

    out_chunks = []         # collected results (in dispatch-permuted order, need inv_perm)
    pending_work = None     # async handle for previous chunk's combine AllToAll
    pending_result = None   # buffer that pending_work is writing into
    pending_inv_perm = None # inverse permutation to restore order for the pending chunk

    for x_chunk, assign_chunk in zip(x_chunks, assign_chunks):
        chunk_T = x_chunk.shape[0]

        # ────────────────── DISPATCH (synchronous) ──────────────────
        send_counts = torch.bincount(assign_chunk, minlength=world_size).to(
            device=x.device, dtype=torch.long
        )
        recv_counts = torch.empty(world_size, dtype=torch.long, device=x.device)
        dist.all_to_all_single(recv_counts, send_counts)

        perm = torch.argsort(assign_chunk)
        packed_x = x_chunk.index_select(0, perm).contiguous()

        total_recv = int(recv_counts.sum().item())
        recv_x = torch.empty((total_recv, D), dtype=x.dtype, device=x.device)
        dist.all_to_all_single(
            recv_x, packed_x,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
        )

        # ────────────────── COMPUTE (synchronous) ───────────────────
        h = torch.matmul(recv_x, w1.t())
        h = torch.nn.functional.gelu(h)
        computed = torch.matmul(h, w2.t())   # [total_recv, D]

        # ──────── WAIT for previous chunk's combine ─────────────────
        if pending_work is not None:
            pending_work.wait()
            out_chunks.append(pending_result.index_select(0, pending_inv_perm))

        # ────────────────── COMBINE (async) ─────────────────────────
        result_buf = torch.empty((chunk_T, D), dtype=x.dtype, device=x.device)
        pending_work = dist.all_to_all_single(
            result_buf, computed,
            output_split_sizes=send_counts.tolist(),
            input_split_sizes=recv_counts.tolist(),
            async_op=True,
        )
        pending_result = result_buf
        pending_inv_perm = torch.argsort(perm)

    # Wait for the last chunk's combine
    if pending_work is not None:
        pending_work.wait()
        out_chunks.append(pending_result.index_select(0, pending_inv_perm))

    out = torch.cat(out_chunks, dim=0)
    return out
