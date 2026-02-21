#!POPCORN leaderboard nvfp4_gemv

import torch
from task import input_t, output_t
from torch import Tensor

FP4E2M1_LUT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    device="cuda",
    dtype=torch.float32,
)


def dequant(x: Tensor, scales: Tensor):
    x = x.view(torch.uint8)
    x = torch.stack([x & 0xF, x >> 4], dim=-1).flatten(-2)
    x = FP4E2M1_LUT[x.view(-1).long()].view(x.shape)
    scales = scales.repeat_interleave(16, dim=2)
    return x.float() * scales.float()


def custom_kernel(data: input_t) -> output_t:
    # a:   [  M, K, L],                   natural shape [L,   M, K]
    # b:   [128, K, L],                   natural shape [L, 128, K] - only the 1st row is used
    # sfa: [32, 4, rest_m, 4, rest_k, L], natural shape [L, rest_m, rest_k, 32, 4, 4]
    # sfb: [32, 4,      1, 4, rest_k, L], natural shape [L,      1, rest_k, 32, 4, 4]
    # c:   [  M, 1, L],                   natural shape [L, M, 1]
    a, b, _, _, sfa, sfb, c = data

    a = a.permute(2, 0, 1)  # [L,   M, K/2]
    b = b.permute(2, 0, 1)  # [L, 128, K/2]
    sfa = sfa.permute(5, 2, 1, 0, 4, 3).reshape(*a.shape[:2], -1)
    sfb = sfb.permute(5, 2, 1, 0, 4, 3).reshape(*b.shape[:2], -1)

    a_f32 = dequant(a, sfa)
    b_f32 = dequant(b[:, :1], sfb[:, :1])

    out = torch.bmm(a_f32, b_f32.transpose(-1, -2)).half()
    return out.permute(1, 2, 0)
