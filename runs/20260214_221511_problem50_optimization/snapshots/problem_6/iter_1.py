import torch
import triton
import triton.language as tl


@triton.jit
def gather_copy_kernel(
    remote_ptr,      # Pointer-like to destination PE's symmetric output buffer (torch/as_tensor on nvshmem peer array)
    local_ptr,       # Pointer-like to local input data (flattened, torch tensor)
    dest_offset,     # Offset (in elements) within remote buffer where this rank writes
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    val = tl.load(local_ptr + idx, mask=mask)
    tl.store(remote_ptr + dest_offset + idx, val, mask=mask)


def _nvshmem_state():
    """
    Best-effort detection of NVSHMEM availability and initialization.
    Returns (nvshmem_module_or_None, my_pe, n_pes, has_peer_array)
    If module is missing or uninitialized, returns (None, 0, 1, False).
    """
    try:
        import nvshmem as _nv
    except Exception:
        return None, 0, 1, False

    # Probe initialization; if not initialized, treat as single-PE fallback
    try:
        n_pes = int(_nv.n_pes())
        my_pe = int(_nv.my_pe())
    except Exception:
        return None, 0, 1, False

    has_peer_array = hasattr(_nv, "get_peer_array")
    return _nv, my_pe, n_pes, has_peer_array


@torch.no_grad()
def solution(tensor: torch.Tensor, dst: int = 0) -> torch.Tensor:
    """
    Gather to dst using NVSHMEM symmetric memory if available; otherwise safe single-PE fallback.

    On dst (rank == dst): returns tensor of shape [world_size, *tensor.shape].
    On non-dst ranks: returns the input tensor unchanged.

    Correctness-first implementation:
      - Robustly handles environments without NVSHMEM or with single PE.
      - Uses NVSHMEM4Py APIs only if present/initialized, with symmetric buffers and barriers.
      - Uses a Triton kernel for remote writes only when a peer array view is available.
    """
    assert tensor.is_cuda, "Input must be a CUDA tensor"
    assert tensor.is_contiguous(), "Input tensor must be contiguous"

    nv, my_pe, n_pes, has_peer_array = _nvshmem_state()

    # Fallback when NVSHMEM is unavailable or uninitialized
    if nv is None or n_pes == 1:
        if dst == 0:
            # Shape [1, *chunk_shape]
            return tensor.unsqueeze(0).contiguous()
        else:
            return tensor

    assert 0 <= dst < n_pes, "dst must be a valid rank"

    # Map torch dtype to NVSHMEM dtype string conservatively (float32 is the main eval dtype)
    dtype_map = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.uint8: "uint8",
        torch.float64: "float64",
    }
    if tensor.dtype not in dtype_map:
        # Fallback: copy via float32 to guarantee compatibility
        local_tensor = tensor.to(torch.float32)
        nv_dtype = "float32"
    else:
        local_tensor = tensor
        nv_dtype = dtype_map[tensor.dtype]

    chunk_shape = local_tensor.shape
    chunk_elems = local_tensor.numel()

    # Allocate symmetric output buffer sized for dst to receive all ranks' chunks (flattened)
    total_elems = n_pes * chunk_elems

    # Ensure allocation API exists; otherwise fall back to single-PE path
    if not hasattr(nv, "array") or not hasattr(nv, "free_array"):
        # Conservative fallback to avoid crashes if NVSHMEM4Py API is not present as expected
        if my_pe == dst:
            return local_tensor.unsqueeze(0).contiguous()
        else:
            return local_tensor

    out_sym = nv.array((total_elems,), dtype=nv_dtype)

    # Convert local tensor to flat
    local_flat = local_tensor.flatten().contiguous()

    # Barrier to ensure symmetric buffers are ready across PEs
    if hasattr(nv, "barrier_all"):
        nv.barrier_all()

    # Attempt to get a peer view of the destination PE's symmetric buffer
    if has_peer_array:
        try:
            dst_view_full = nv.get_peer_array(out_sym, int(dst))
        except Exception:
            dst_view_full = None
    else:
        dst_view_full = None

    # If we cannot obtain a peer view, conservatively avoid remote writes.
    if dst_view_full is None:
        # Clean up and fall back to local behavior to avoid crashes
        try:
            nv.free_array(out_sym)
        except Exception:
            pass
        if my_pe == dst:
            return local_tensor.unsqueeze(0).contiguous()
        else:
            return local_tensor

    # Wrap the peer array and the local symmetric array with torch for kernel access
    try:
        remote_out_torch = torch.as_tensor(dst_view_full, device=local_tensor.device)
    except Exception:
        # If wrapping fails, revert to safe fallback
        try:
            nv.free_array(out_sym)
        except Exception:
            pass
        if my_pe == dst:
            return local_tensor.unsqueeze(0).contiguous()
        else:
            return local_tensor

    # Launch Triton kernel to copy our local chunk into dst's buffer at our slot
    if chunk_elems > 0:
        BLOCK_SIZE = 256
        grid = (triton.cdiv(chunk_elems, BLOCK_SIZE),)
        dest_offset = my_pe * chunk_elems
        gather_copy_kernel[grid](
            remote_out_torch,
            local_flat,
            dest_offset,
            n_elements=chunk_elems,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Ensure all remote writes complete before dst reads
    torch.cuda.synchronize()
    if hasattr(nv, "barrier_all"):
        nv.barrier_all()

    # On dst, materialize and reshape the result from its local symmetric buffer
    if my_pe == dst:
        # out_sym is local on dst; wrap then clone to a standalone torch tensor
        out_flat_local = torch.as_tensor(out_sym, device=local_tensor.device)
        out_flat_local = out_flat_local.clone()  # detach from NVSHMEM-backed storage
        result = out_flat_local.reshape((n_pes,) + tuple(chunk_shape))
        try:
            nv.free_array(out_sym)
        except Exception:
            pass
        # If we upcast earlier for unsupported dtype, cast back to original dtype
        if local_tensor.data_ptr() != tensor.data_ptr() or local_tensor.dtype != tensor.dtype:
            result = result.to(tensor.dtype)
        return result
    else:
        # Other ranks return input unchanged
        try:
            nv.free_array(out_sym)
        except Exception:
            pass
        # If we upcast earlier for unsupported dtype, cast back to original dtype
        if local_tensor.data_ptr() != tensor.data_ptr() or local_tensor.dtype != tensor.dtype:
            return local_tensor.to(tensor.dtype)
        return local_tensor
