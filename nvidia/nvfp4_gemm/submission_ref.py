#!POPCORN leaderboard nvfp4_gemm
#!POPCORN gpu NVIDIA

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    # a:   [M, K, 1],                     natural shape [1, M, K]
    # b:   [N, K, 1],                     natural shape [1, N, K] - only the 1st row is used
    # sfa: [32, 4, M/128, 4, rest_k, 1],  natural shape [1, M/128, rest_k, 32, 4, 4]
    # sfb: [32, 4, N/128, 4, rest_k, 1],  natural shape [1, N/128, rest_k, 32, 4, 4]
    # c:   [M, N, 1],                     natural shape [1, M, N]
    a, b, _, _, sfa, sfb, c_ref = data
    torch._scaled_mm(
        a[..., 0],
        b[..., 0].T,
        sfa.permute(5, 2, 4, 0, 1, 3).view(-1),
        sfb.permute(5, 2, 4, 0, 1, 3).view(-1),
        out=c_ref[..., 0],
    )
    return c_ref
