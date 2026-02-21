#!POPCORN leaderboard nvfp4_gemv

from pathlib import Path

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int THREAD_K = 8;
constexpr int BLOCK_M = TB_SIZE / THREAD_K;  // 32
constexpr int BLOCK_K = THREAD_K * 16;  // 128

struct __align__(16) fp4x32 { __nv_fp4x2_e2m1 x[16]; };

// to make our calculations simple, let's treat fp4x2 as a unit.
// hence, K = number of fp4x2 elements, and 8 elements share
// the same scale.
__global__
void kernel(
  const __nv_fp4x2_e2m1 *A_ptr,  // [L,   M, K]
  const __nv_fp4x2_e2m1 *B_ptr,  // [L, 128, K]
  const __nv_fp8_e4m3 *SFA_ptr,  // [L,   M, K/8]
  const __nv_fp8_e4m3 *SFB_ptr,  // [L, 128, K/8]
        half          *C_ptr,    // [L,   M]
  int L, int M, int K
) {
  // to ensure coalesced access, we need at least 8 threads per row (16B x 8 = 128B)
  // each thread reads 16B, which covers 2 scaled groups. hence, we only need within
  // thread reduction during the main loop.

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
    auto A_fp4x32 = reinterpret_cast<const fp4x32 *>(A_ptr)[0];
    auto B_fp4x32 = reinterpret_cast<const fp4x32 *>(B_ptr)[0];
    auto SFA_fp32x2 = static_cast<float2>(reinterpret_cast<const __nv_fp8x2_e4m3 *>(SFA_ptr)[0]);
    auto SFB_fp32x2 = static_cast<float2>(reinterpret_cast<const __nv_fp8x2_e4m3 *>(SFB_ptr)[0]);

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K;
    SFA_ptr += BLOCK_K / 8;
    SFB_ptr += BLOCK_K / 8;

    for (int group_id = 0; group_id < 2; group_id++) {
      float sub_acc = 0.0f;

      for (int i = 0; i < 8; i++) {
        // __nv_cvt_fp4x2_to_halfraw2() didn't compile to the desired PTX instruction...
        __half2 A_fp16x2 = __nv_cvt_fp4x2_to_halfraw2(A_fp4x32.x[group_id * 8 + i].__x, __NV_E2M1);
        __half2 B_fp16x2 = __nv_cvt_fp4x2_to_halfraw2(B_fp4x32.x[group_id * 8 + i].__x, __NV_E2M1);
        sub_acc += __half2float(A_fp16x2.x) * __half2float(B_fp16x2.x)
                 + __half2float(A_fp16x2.y) * __half2float(B_fp16x2.y);
      }

      float sfa = reinterpret_cast<float *>(&SFA_fp32x2)[group_id];
      float sfb = reinterpret_cast<float *>(&SFB_fp32x2)[group_id];
      acc += sub_acc * sfa * sfb;
    }
  }

  // inter-thread reduction
  for (int stride = THREAD_K / 2; stride > 0; stride /= 2)
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

  auto A_ptr = reinterpret_cast<const __nv_fp4x2_e2m1 *>(A.data_ptr());
  auto B_ptr = reinterpret_cast<const __nv_fp4x2_e2m1 *>(B.data_ptr());
  auto SFA_ptr = reinterpret_cast<const __nv_fp8_e4m3 *>(SFA.data_ptr());
  auto SFB_ptr = reinterpret_cast<const __nv_fp8_e4m3 *>(SFB.data_ptr());
  auto C_ptr = reinterpret_cast<half *>(C.data_ptr());

  dim3 grid(M / BLOCK_M, L);
  auto stream = at::cuda::getCurrentCUDAStream();

  kernel<<<grid, TB_SIZE, 0, stream>>>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L, M, K);
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
        # "-gencode=arch=compute_120a,code=sm_120a",  # test fails with this...
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
