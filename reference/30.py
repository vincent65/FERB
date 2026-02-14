# Tensor Parallel
# Column-sharded weights_in, row-sharded weights_out

import torch
import torch.distributed as dist

@torch.no_grad()
def solution(
    x,      # All tokens assigned to this rank [N, D]
    w_in,   # Column shard [D_intermediate / World_Size, D]
    w_out,  # Row shard [D_out, D_intermediate / World_Size]
) -> torch.Tensor:
    """

    Want to simulate MLP layer:  ( Gelu( x @ W_in ) )  @  W_out

    Here, we shard W_in by COLUMN [ W_1i  W_2i ].

    We shard W_out by ROW [ W_1o ]
                          [ W_2o ]


    So each expert can independently matmul by W_in. Then each of those matmul'd parts can matmul with W_out, and allreduce to get the right answer.
    """

    # each expert does these parts:
    h = torch.matmul(x, w_in.t())
    h = torch.nn.functional.gelu(h)
    partial_sum = torch.matmul(h, w_out.t())

    # then we AllReduce the row-sharded partials
    dist.all_reduce(partial_sum, op=dist.ReduceOp.SUM)
    return partial_sum