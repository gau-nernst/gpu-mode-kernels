#!POPCORN leaderboard nvfp4_group_gemm
#!POPCORN gpu B200

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    abc_list, _, sf_list, _ = data
    return [
        torch._scaled_mm(
            a[..., 0],
            b[..., 0].T,
            sfa.permute(5, 2, 4, 0, 1, 3).view(-1),
            sfb.permute(5, 2, 4, 0, 1, 3).view(-1),
            out_dtype=torch.float16,
        ).unsqueeze(-1)
        for (a, b, _), (sfa, sfb) in zip(abc_list, sf_list)
    ]
