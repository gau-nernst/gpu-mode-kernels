#!POPCORN leaderboard nvfp4_gemv

from pathlib import Path

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    # a:   [  M, K, L],                   natural shape [L,   M, K]
    # b:   [128, K, L],                   natural shape [L, 128, K] - only the 1st row is used
    # sfa: [32, 4, rest_m, 4, rest_k, L], natural shape [L, rest_m, rest_k, 32, 4, 4]
    # sfb: [32, 4,      1, 4, rest_k, L], natural shape [L,      1, rest_k, 32, 4, 4]
    # c:   [  M, 1, L],                   natural shape [L, M, 1]
    a, b, _, _, sfa, sfb, c_ref = data

    M = a.shape[0]
    N = 128
    K = a.shape[1] * 2
    L = a.shape[2]

    a = a.permute(2, 0, 1)  # [L, M, K/2]
    b = b.permute(2, 0, 1)  # [L, 128, K/2]
    sfa = sfa.permute(5, 2, 4, 0, 1, 3).view(L, M, -1)
    sfb = sfb.permute(5, 2, 4, 0, 1, 3).view(L, N, -1)

    big_c = c_ref.new_empty(L, N, M).transpose(1, 2)  # (L, M, 128), M-major

    for l_idx in range(L):
        torch._scaled_mm(
            a[l_idx],
            b[l_idx].transpose(0, 1),
            sfa[l_idx],
            sfb[l_idx],
            out_dtype=torch.float16,
            out=big_c[l_idx],
        )

    if False:
        path = Path(f"profile_data/{M=}_{K=}_{L=}.json.gz")
        if not path.exists():
            a = a.clone()
            b = b.clone()
            sfa = sfa.clone()
            sfb = sfb.clone()
            big_c = c_ref.new_empty(L, 128, M).transpose(1, 2)  # (L, M, 128), M-major

            with torch.profiler.profile() as prof:
                for l_idx in range(L):
                    torch._scaled_mm(
                        a[l_idx],
                        b[l_idx].transpose(0, 1),
                        sfa[l_idx],
                        sfb[l_idx],
                        out_dtype=torch.float16,
                        out=big_c[l_idx],
                    )
            path.parent.mkdir(exist_ok=True)
            prof.export_chrome_trace(str(path))

    return big_c[..., :1].permute(1, 2, 0)  # convert to [L, M, 1]
