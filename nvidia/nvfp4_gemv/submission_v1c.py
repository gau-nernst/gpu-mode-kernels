#!POPCORN leaderboard nvfp4_gemv

from pathlib import Path

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

__device__
void fp4x8_to_fp32x2x4(int in, int64_t *out) {
  int tmp[4];
  asm volatile(
    "{\n"
    ".reg .b8 tmp0, tmp1, tmp2, tmp3;\n"
    "mov.b32 {tmp0, tmp1, tmp2, tmp3}, %4; // unpack 32-bit register to 4x fp4x2\n"
    "cvt.rn.f16x2.e2m1x2 %0, tmp0; // PTX only supports FP4->FP16\n"
    "cvt.rn.f16x2.e2m1x2 %1, tmp1;\n"
    "cvt.rn.f16x2.e2m1x2 %2, tmp2;\n"
    "cvt.rn.f16x2.e2m1x2 %3, tmp3;\n"
    "}\n"
    : "=r"(tmp[0]), "=r"(tmp[1]), "=r"(tmp[2]), "=r"(tmp[3])
    : "r"(in)
  );

  for (int i = 0; i < 4; i++)
    asm volatile(
      "{\n"
      ".reg .b16 b16_0, b16_1;\n"
      ".reg .b32 f32_0, f32_1;\n"
      "mov.b32 {b16_0, b16_1}, %1;  // unpack\n"
      "cvt.f32.f16 f32_0, b16_0;\n"
      "cvt.f32.f16 f32_1, b16_1;\n"
      "mov.b64 %0, {f32_0, f32_1};  // pack\n"
      "}\n"
      : "=l"(out[i])
      : "r"(tmp[i])
    );
}

// to make our calculations simple, let's treat fp4x2 as a unit.
// hence, K = number of fp4x2 elements, and 8 elements share
// the same scale.
template <int THREAD_K>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE)
void kernel(
  const char   *A_ptr,  // [L,   M, K]
  const char   *B_ptr,  // [L, 128, K]
  const char *SFA_ptr,  // [L,   M, K/8]
  const char *SFB_ptr,  // [L, 128, K/8]
        half   *C_ptr,  // [L,   M]
  int L, int M, int K
) {
  // to ensure coalesced access, we need at least 8 threads per row (16B x 8 = 128B)
  // each thread reads 16B, which covers 2 scaled groups. hence, we only need within
  // thread reduction during the main loop.
  static_assert(THREAD_K >= 8);
  static_assert(THREAD_K <= TB_SIZE);
  constexpr int BLOCK_M = TB_SIZE / THREAD_K;  // 32
  constexpr int BLOCK_K = THREAD_K * 16;  // 128

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int batch_id = blockIdx.y;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  const int off_m = (bid * BLOCK_M) + (tid / THREAD_K);
  const int off_k = (tid % THREAD_K) * 16;  // each thread reads 16 fp4x2 values at a time

  A_ptr += (batch_id *   M * K) + (off_m * K) + off_k;
  B_ptr += (batch_id * 128 * K)               + off_k;

  SFA_ptr += (batch_id *   M * (K / 8)) + (off_m * (K / 8)) + (off_k / 8);
  SFB_ptr += (batch_id * 128 * (K / 8))                     + (off_k / 8);

  float acc = 0.0f;

  const int num_iters = K / BLOCK_K;

  for (int iter_k = 0; iter_k < num_iters; iter_k ++) {
    int A_fp4x8[4], B_fp4x8[4];

    // cache-streaming for A, since we only read it once
    asm volatile("ld.global.v4.u32.cs {%0, %1, %2, %3}, [%4];\n"
                : "=r"(A_fp4x8[0]), "=r"(A_fp4x8[1]), "=r"(A_fp4x8[2]), "=r"(A_fp4x8[3])
                : "l"(A_ptr));

    reinterpret_cast<int4 *>(B_fp4x8)[0] = reinterpret_cast<const int4 *>(B_ptr)[0];
    auto SFA_fp32x2 = static_cast<float2>(reinterpret_cast<const __nv_fp8x2_e4m3 *>(SFA_ptr)[0]);
    auto SFB_fp32x2 = static_cast<float2>(reinterpret_cast<const __nv_fp8x2_e4m3 *>(SFB_ptr)[0]);

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K;
    SFA_ptr += BLOCK_K / 8;
    SFB_ptr += BLOCK_K / 8;

    for (int group_id = 0; group_id < 2; group_id++) {
      // unpack to FP32
      int64_t A_fp32x2[8], B_fp32x2[8];
      fp4x8_to_fp32x2x4(A_fp4x8[group_id * 2 + 0], A_fp32x2);
      fp4x8_to_fp32x2x4(A_fp4x8[group_id * 2 + 1], A_fp32x2 + 4);
      fp4x8_to_fp32x2x4(B_fp4x8[group_id * 2 + 0], B_fp32x2);
      fp4x8_to_fp32x2x4(B_fp4x8[group_id * 2 + 1], B_fp32x2 + 4);

      // FMA. manually unroll the 1st iteration
      int64_t sub_acc;
      asm volatile("mul.rn.f32x2 %0, %1, %2;\n" : "=l"(sub_acc) : "l"(A_fp32x2[0]), "l"(B_fp32x2[0]));
      for (int i = 1; i < 8; i++)
        asm volatile("fma.rn.f32x2 %0, %1, %2, %0;\n" : "+l"(sub_acc) : "l"(A_fp32x2[i]), "l"(B_fp32x2[i]));

      float tmp[2];
      std::memcpy(tmp, &sub_acc, sizeof(sub_acc));

      float sfa = reinterpret_cast<float *>(&SFA_fp32x2)[group_id];
      float sfb = reinterpret_cast<float *>(&SFB_fp32x2)[group_id];
      acc += (tmp[0] + tmp[1]) * sfa * sfb;
    }
  }

  // threadblock reduction
  if constexpr (THREAD_K > WARP_SIZE) {
    __shared__ float smem[TB_SIZE];
    smem[tid] = acc;
    __syncthreads();

    for (int stride = THREAD_K / 2; stride >= WARP_SIZE; stride /= 2) {
      if ((tid % THREAD_K) < stride) {
        acc += smem[tid + stride];
        smem[tid] = acc;
      }
      __syncthreads();
    }
  }

  // warp reduction
  constexpr int start_stride = std::min(THREAD_K, WARP_SIZE) / 2;
  for (int stride = start_stride; stride > 0; stride /= 2)
    acc += __shfl_down_sync(0xFFFF'FFFF, acc, stride);

  if (tid % THREAD_K == 0)
    C_ptr[batch_id * M + off_m] = __float2half(acc);
}

void gemv(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& SFA,
  const at::Tensor& SFB,
        at::Tensor& C
) {
  const int M = A.size(0);
  const int K = A.size(1);
  const int L = A.size(2);

  auto A_ptr = reinterpret_cast<const char *>(A.data_ptr());
  auto B_ptr = reinterpret_cast<const char *>(B.data_ptr());
  auto SFA_ptr = reinterpret_cast<const char *>(SFA.data_ptr());
  auto SFB_ptr = reinterpret_cast<const char *>(SFB.data_ptr());
  auto C_ptr = reinterpret_cast<half *>(C.data_ptr());

  auto stream = at::cuda::getCurrentCUDAStream();

#define launch(THREAD_K) { \
  int BLOCK_M = TB_SIZE / THREAD_K; \
  dim3 grid(M / BLOCK_M, L); \
  kernel<THREAD_K><<<grid, TB_SIZE, 0, stream>>>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L, M, K); \
}

  if (false) {}
  else if (K % (128 * 16) == 0) launch(128)  // benchmark.0
  else if (K % (32 * 16) == 0) launch(32)    // benchmark.1 and benchmark.2
  else launch(8)                             // the rest

#undef launch
}

TORCH_LIBRARY(my_module, m) {
  m.def("gemv(Tensor A, Tensor B, Tensor SFA, Tensor SFB, Tensor(a!) C) -> ()");
  m.impl("gemv", &gemv);
}
"""

load_inline(
    "gemv_c0",
    cpp_sources="",
    cuda_sources=CUDA_SRC,
    verbose=True,
    is_python_module=False,
    no_implicit_headers=True,
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_100a,code=sm_100a",
        "-gencode=arch=compute_120a,code=sm_120a",
        "-lineinfo",
        "-Xptxas=-v",
    ],
)


def custom_kernel(data: input_t) -> output_t:
    # a:   [  M, K, L],                   natural shape [L,   M, K]
    # b:   [128, K, L],                   natural shape [L, 128, K] - only the 1st row is used
    # sfa: [32, 4, rest_m, 4, rest_k, L], natural shape [L, rest_m, rest_k, 32, 4, 4]
    # sfb: [32, 4,      1, 4, rest_k, L], natural shape [L,      1, rest_k, 32, 4, 4]
    # c:   [  M, 1, L],                   natural shape [L, M, 1]
    a, b, sfa, sfb, _, _, c_ref = data
    torch.ops.my_module.gemv(a, b, sfa, sfb, c_ref)

    if False:
        M, K, L = a.shape
        path = Path(f"profile_data/{M=}_K={K * 2}_{L=}.json.gz")
        if not path.exists():
            a.new_zeros(int(1e8), dtype=torch.uint8)  # 100 MB

            with torch.profiler.profile() as prof:
                torch.ops.my_module.gemv(a, b, sfa, sfb, c_ref)

            path.parent.mkdir(exist_ok=True)
            prof.export_chrome_trace(str(path))

    return c_ref
