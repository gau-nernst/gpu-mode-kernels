#!POPCORN leaderboard nvfp4_gemv

from pathlib import Path

import torch
import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def fp4x2_to_fp16x2(x):
    """f16x2 packed in a 32-bit register"""
    return tl.inline_asm_elementwise(
        asm=r"""
{
    .reg .b8 tmp<4>;
    mov.b32 {tmp0, tmp1, tmp2, tmp3}, $4;  // unpack 32-bit reg to 4x fp4x2
    cvt.rn.f16x2.e2m1x2 $0, tmp0;  // PTX only supports FP4->FP16
    cvt.rn.f16x2.e2m1x2 $1, tmp1;
    cvt.rn.f16x2.e2m1x2 $2, tmp2;
    cvt.rn.f16x2.e2m1x2 $3, tmp3;
}
""",
        constraints="=r,=r,=r,=r,r",
        args=[x],
        dtype=tl.int32,
        is_pure=True,
        pack=4,
    )


@triton.jit
def fp16x2_to_fp32x2(x):
    """Unpack fp16x2 and convert to fp32"""
    return tl.inline_asm_elementwise(
        asm=r"""
{
    .reg .b16 tmp<2>;
    mov.b32 {tmp0, tmp1}, $2;  // unpack
    cvt.f32.f16 $0, tmp0;
    cvt.f32.f16 $1, tmp1;
}
""",
        constraints="=f,=f,r",
        args=[x],
        dtype=(tl.float32, tl.float32),
        is_pure=True,
        pack=1,
    )


# to simplify the math, we use K and BLOCK_K to count fp4x2
@triton.jit
def kernel(
    A_ptr,  # [L,   M, K]
    B_ptr,  # [L, 128, K]
    SFA_ptr,  # [L,   M, K/8]
    SFB_ptr,  # [L, 128, K/8]
    C_ptr,  # [L, M]
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tl.static_assert(M % BLOCK_M == 0)
    tl.static_assert(K % BLOCK_K == 0)

    pid = tl.program_id(0)
    batch_id = tl.program_id(1)

    A_ptr += batch_id * M * K
    B_ptr += batch_id * 128 * K
    SFA_ptr += batch_id * M * (K // 8)
    SFB_ptr += batch_id * 128 * (K // 8)
    C_ptr += batch_id * M

    offs_m = (pid * BLOCK_M) + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    sf_offs_k = tl.arange(0, BLOCK_K // 8)

    A_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]  # (BLOCK_M, BLOCK_K)
    B_ptrs = B_ptr + offs_k[None, :]  # (1, BLOCK_K)
    SFA_ptrs = SFA_ptr + offs_m[:, None] * (K // 8) + sf_offs_k[None, :]  # (BLOCK_M, BLOCK_K / 8)
    SFB_ptrs = SFB_ptr + sf_offs_k[None, :]  # (1, BLOCK_K / 8)

    acc = tl.zeros((BLOCK_M, BLOCK_K // 8), dtype=tl.float32)

    num_iters = K // BLOCK_K
    for _ in range(num_iters):
        A = tl.load(A_ptrs, eviction_policy="evict_first")
        B = tl.load(B_ptrs, eviction_policy="evict_last")
        SFA = tl.load(SFA_ptrs, eviction_policy="evict_first")
        SFB = tl.load(SFB_ptrs, eviction_policy="evict_last")

        A_ptrs += BLOCK_K
        B_ptrs += BLOCK_K
        SFA_ptrs += BLOCK_K // 8
        SFB_ptrs += BLOCK_K // 8

        A_fp16x2 = fp4x2_to_fp16x2(A.reshape(BLOCK_M, BLOCK_K // 8, 8))
        B_fp16x2 = fp4x2_to_fp16x2(B.reshape(1, BLOCK_K // 8, 8))

        A_lo, A_hi = fp16x2_to_fp32x2(A_fp16x2)
        B_lo, B_hi = fp16x2_to_fp32x2(B_fp16x2)

        partial_acc = tl.sum(A_lo * B_lo + A_hi * B_hi, axis=2)  # (BLOCK_M, BLOCK_K / 8)
        acc += partial_acc * SFA.to(tl.float32) * SFB.to(tl.float32)

    acc = tl.sum(acc, axis=1)  # (BLOCK_M,)

    offs_m = (pid * BLOCK_M) + tl.arange(0, BLOCK_M)
    C_ptrs = C_ptr + offs_m
    tl.store(C_ptrs, acc)


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa, sfb, _, _, c_ref = data
    M, K, L = a.shape

    a = a.view(torch.int8)
    b = b.view(torch.int8)

    BLOCK_M = 8
    BLOCK_K = 128
    num_blocks = M // BLOCK_M
    kernel[(num_blocks, L)](a, b, sfa, sfb, c_ref, M, K, BLOCK_M, BLOCK_K)

    if False:
        path = Path(f"profile_data/{M=}_K={K * 2}_{L=}.json.gz")
        if not path.exists():
            a.new_zeros(int(1e8), dtype=torch.uint8)  # 100 MB

            with torch.profiler.profile(with_stack=True) as prof:
                kernel[(num_blocks, L)](a, b, sfa, sfb, c_ref, M, K, BLOCK_M, BLOCK_K)

            path.parent.mkdir(exist_ok=True)
            prof.export_chrome_trace(str(path))

    return c_ref
