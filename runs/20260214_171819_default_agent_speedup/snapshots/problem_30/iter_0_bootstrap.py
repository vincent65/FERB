import torch
import triton
import triton.language as tl
import nvshmem.core as nvshmem


@triton.jit
def _allreduce_atomic_add_kernel(
    output_ptr,  # pointer to symmetric output buffer (float32)
    input_ptr,   # pointer to local input buffer (float32)
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    vals = tl.load(input_ptr + offsets, mask=mask)
    # atomic add into the symmetric buffer
    tl.atomic_add(output_ptr + offsets, vals, mask=mask)


class PyTorchStreamWrapper:
    """Wrap a torch CUDA stream for nvshmem barriers."""
    def __init__(self, pt_stream):
        self.pt_stream = pt_stream

    def __cuda_stream__(self):
        # return (device_index, cuda_stream) as expected by NVSHMEM Python binding
        return (0, self.pt_stream.cuda_stream)


@torch.no_grad()
def solution(x, w_in, w_out) -> torch.Tensor:
    """
    Tensor-parallel MLP (column-sharded W_in, row-sharded W_out) implemented
    with local GEMMs and an NVSHMEM-based all-reduce implemented via Triton
    atomic-add kernels into symmetric memory.

    Args:
        x:      [N, D]  -- local tokens on this rank (CUDA tensor)
        w_in:   [D_intermediate_per_rank, D] -- column shard of W_in (CUDA tensor)
        w_out:  [D_out_per_rank, D_intermediate_per_rank] -- row shard of W_out (CUDA tensor)

    Returns:
        Tensor of shape [N, D_out_per_rank] containing the elementwise sum across
        ranks of the local partial results (i.e., the same result as torch.distributed.all_reduce SUM).
    """
    # Basic checks
    assert x.is_cuda and w_in.is_cuda and w_out.is_cuda, "All inputs must be CUDA tensors"

    # Local computation (per reference)
    # h = x @ w_in.t()
    # gelu
    # partial_sum = h @ w_out.t()
    # Keep native precision for local matmuls, but accumulation via NVSHMEM uses float32 atomics.

    # Ensure contiguous inputs for performance
    x = x.contiguous()
    w_in = w_in.contiguous()
    w_out = w_out.contiguous()

    # Local compute
    h = torch.matmul(x, w_in.t())
    h = torch.nn.functional.gelu(h)
    partial = torch.matmul(h, w_out.t())  # shape: [N, D_out_per_rank]

    orig_dtype = partial.dtype
    device = partial.device

    # NVSHMEM atomic add supports float32 reliably. Cast to float32 for accumulation if needed.
    if partial.dtype != torch.float32:
        partial_acc = partial.to(torch.float32).contiguous()
    else:
        partial_acc = partial.contiguous()

    n_elements = partial_acc.numel()

    # NVSHMEM symmetric allocation (float32)
    # Each rank allocates a symmetric buffer of the same size and zeros it.
    out_sym = nvshmem.tensor((n_elements,), dtype=torch.float32)
    out_sym.zero_()

    # Prepare stream wrapper for NVSHMEM barrier
    cuda_stream = torch.cuda.current_stream()
    stream_wrapper = PyTorchStreamWrapper(cuda_stream)

    # Synchronize and barrier to ensure everyone has zeroed their symmetric buffer
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Triton kernel launch params
    BLOCK_SIZE = 512
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Each rank atomically adds its local partial into every PE's symmetric buffer
    n_pes = nvshmem.n_pes()

    # Flatten the input for the kernel (float32)
    input_flat = partial_acc.view(-1)

    for pe in range(n_pes):
        target_view = nvshmem.get_peer_tensor(out_sym, pe)  # view into symmetric buffer on PE `pe`
        # Launch Triton kernel to atomically add our contribution into the remote symmetric buffer
        _allreduce_atomic_add_kernel[grid](
            target_view,
            input_flat,
            n_elements=n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

    # Ensure all atomic adds finished on device and across PEs
    torch.cuda.synchronize()
    nvshmem.barrier(nvshmem.Teams.TEAM_NODE, stream_wrapper)

    # Read back the result from symmetric memory
    result_flat = out_sym.clone().detach()  # float32
    result = result_flat.view(partial.shape)

    # Cast back to original dtype if needed
    if orig_dtype != torch.float32:
        result = result.to(orig_dtype)

    # Cleanup
    nvshmem.free_tensor(out_sym)

    return result
