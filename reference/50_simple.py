# Fused chunked MoE + AllGather for high throughput. When there are enough tokens, chunk them and overlap AllGather of chunk 1 while chunk 2 is still computing.


## THIS IS UNSHUFFLED MoE: IN REALITY YOU NEED TO DO SOME SHUFFLING BEFORE THE ALLGATHER

import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    x: torch.Tensor,  # Tokens that this RANK has gotten already (via MoE routing).
                      # Shape: [Seq_Len_Tokens, Token_Dimension]
                      # All ranks must have the same shape (pad if needed) for all_gather_into_tensor.
    weights: torch.Tensor,      # Expert's weights (ASSUMING EXPERT IS SIMPLE MLP)
    num_chunks: int = 2,  # SUB-BATCH for Case 2 pipelining; unused in Case 1.
) -> torch.Tensor:
    """
    Mixture of Experts Forward with AllGather (weights not sharded).

    Tokens are already routed. Each rank runs its local expert on the tokens routed to it, then we AllGather so every rank has all experts' outputs.

    NAIVE: 
        Given Expert's tokens TOK = [T x D].
        Compute MLP(TOK) 
        allgather(tok). Now every GPU has every other GPU's computed TOK.
    """
    assert dist.is_initialized(), "torch.distributed must be initialized"
    assert x.is_cuda and x.is_contiguous(), "x must be contiguous CUDA tensor"
    assert weights.is_cuda and weights.is_contiguous(), "weights must be contiguous CUDA tensor"


    # EVERY EXPERT DOES ITS OWN COMPUTATION
    # Case 1: Compute entire local expert output, then one AllGather (no pipelining).
    # One linear layer: x [T, D] @ weights.T [D, D_out] -> [T, D_out]
    x = x.contiguous()
    local_out = torch.matmul(x, weights.t())  # expert output for this rank's tokens
    world_size = dist.get_world_size()

   

    # THEN IT SHARES WITH ALL OTHER EXPERTS IN SOME WAY
    out_shape = (world_size,) + local_out.shape   # Buffer so this GPU can hold all ranks' outputs: [world_size, T, D_out]
    out = torch.empty(out_shape, dtype=local_out.dtype, device=local_out.device)

    # TODO: IN REALITY THIS WOULD BE AN ALLTOALL WITH SOME SHUFFLING
    dist.all_gather_into_tensor(out, local_out)     # every rank calls this and so everything gets routed right
    return out



# ## PIPELINED: overlap AllGather of chunk i with compute of chunk i+1

# ## This only works in NCCL with multiple CUDA streams....

# @torch.no_grad()
# def solution(
#     x: torch.Tensor,  # Tokens that this RANK has gotten already (via MoE routing).
#                       # Shape: [Seq_Len_Tokens, Token_Dimension]
#     weights: torch.Tensor,  # Expert weights [D_out, D]. All ranks same shape.
#     num_chunks: int = 2,
# ) -> torch.Tensor:
#     """
#     Pipelined MoE + AllGather: chunk tokens, compute MLP(chunk_i), all_gather(chunk_i)
#     with overlap — wait for previous all_gather only when we need the buffer; next chunk
#     compute runs while previous all_gather is in flight (async_op=True).

#     PIPELINED:
#         Given Expert's tokens TOK = [T x D].
#         Chunk into TOK_1, TOK_2 = [T/2 x D].
#         for TOK_i:
#             Compute MLP(TOK_i) 
#             immediately allgather(TOK_i)
#     """
#     assert dist.is_initialized(), "torch.distributed must be initialized"
#     assert x.is_cuda and x.is_contiguous(), "x must be contiguous CUDA tensor"
#     assert weights.is_cuda and weights.is_contiguous(), "weights must be contiguous CUDA tensor"

#     world_size = dist.get_world_size()
#     T, D = x.shape
#     D_out = weights.shape[0]
#     out = torch.empty((world_size, T, D_out), dtype=x.dtype, device=x.device)

#     chunked_xs = torch.chunk(x, num_chunks, dim=0)
    
#     # Create a separate stream for communication
#     comm_stream = torch.cuda.Stream()
    
#     token_offset = 0
#     handles = []

#     for i, chunk in enumerate(chunked_xs):
#         # 1. COMPUTE: Runs on the DEFAULT stream
#         local_out = torch.matmul(chunk, weights.t())
#         chunk_T = local_out.shape[0]
#         out_slice = out[:, token_offset : token_offset + chunk_T, :]

#         # 2. COORDINATE: Tell the comm_stream to wait until matmul is done
#         # This is a GPU-side sync; no CPU overhead
#         comm_stream.wait_stream(torch.cuda.current_stream())

#         # 3. COMMUNICATE: Switch to the comm_stream for the All-Gather
#         with torch.cuda.stream(comm_stream):
#             handle = dist.all_gather_into_tensor(out_slice, local_out, async_op=True)
#             handles.append(handle)

#         # IMPORTANT: The loop now goes to next iteration.
#         # The Default Stream starts the NEXT matmul while comm_stream is busy.
#         token_offset += chunk_T

#     # Final synchronization
#     for h in handles:
#         h.wait()
    
#     # Make sure the default stream doesn't move on until comm_stream is finished
#     torch.cuda.current_stream().wait_stream(comm_stream)

#     return out


