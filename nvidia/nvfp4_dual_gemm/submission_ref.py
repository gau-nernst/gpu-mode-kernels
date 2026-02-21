#!POPCORN leaderboard nvfp4_dual_gemm
#!POPCORN gpu NVIDIA

import torch
import torch.nn.functional as F
from task import input_t, output_t


@torch.compile(dynamic=False)
def custom_kernel(data: input_t) -> output_t:
    a, b1, b2, _, _, _, sfa, sfb1, sfb2, _ = data

    out1 = torch._scaled_mm(
        a[..., 0],
        b1[..., 0].T,
        sfa.permute(5, 2, 4, 0, 1, 3).view(-1),
        sfb1.permute(5, 2, 4, 0, 1, 3).view(-1),
        out_dtype=torch.float32,
    )
    out2 = torch._scaled_mm(
        a[..., 0],
        b2[..., 0].T,
        sfa.permute(5, 2, 4, 0, 1, 3).view(-1),
        sfb2.permute(5, 2, 4, 0, 1, 3).view(-1),
        out_dtype=torch.float32,
    )
    out = F.silu(out1) * out2
    return out.half().unsqueeze(-1)
