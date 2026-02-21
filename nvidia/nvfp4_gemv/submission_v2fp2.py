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

__device__
void fp4x8_to_fp16x2x4(int *out, int in) {
  asm volatile(
    "{\n\t"
    ".reg .b8 tmp0, tmp1, tmp2, tmp3;\n\t"
    "mov.b32 {tmp0, tmp1, tmp2, tmp3}, %4; // unpack 32-bit register to 4x fp4x2\n\t"
    "cvt.rn.f16x2.e2m1x2 %0, tmp0;\n\t"
    "cvt.rn.f16x2.e2m1x2 %1, tmp1;\n\t"
    "cvt.rn.f16x2.e2m1x2 %2, tmp2;\n\t"
    "cvt.rn.f16x2.e2m1x2 %3, tmp3;\n\t"
    "}"
    : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
    : "r"(in)
  );
}

__device__
void fp8x2_to_fp16x2(int *out, int16_t in) {
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out[0]) : "h"(in));
}

__device__
void ldcs_i16(int16_t *dst, const void *src) {
//#define PTX_MOD ".cs"
//#define PTX_MOD ".L1::no_allocate"
//#define PTX_MOD ".cs.nc"
#define PTX_MOD ".L1::no_allocate"
  asm volatile("ld.global" PTX_MOD ".b16 %0, [%1];" : "=h"(dst[0]) : "l"(src));
#undef PTX_MOD
}

__device__
void ldca_i16(int16_t *dst, const void *src) {
//#define PTX_MOD ".ca"
//#define PTX_MOD ".L1::evict_last"
//#define PTX_MOD ".ca.nc"
#define PTX_MOD ""
  asm volatile("ld.global" PTX_MOD ".b16 %0, [%1];" : "=h"(dst[0]) : "l"(src));
#undef PTX_MOD
}

__device__
void ldcs_i32x4(int *dst, const void *src) {
//#define PTX_MOD ".cs"
//#define PTX_MOD ".L1::no_allocate"
//#define PTX_MOD ".cs.nc"
#define PTX_MOD ".L1::no_allocate"
  asm volatile("ld.global" PTX_MOD ".v4.b32 {%0, %1, %2, %3}, [%4];"
              : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
              : "l"(src));
#undef PTX_MOD
}

__device__
void ldca_i32x4(int *dst, const void *src) {
//#define PTX_MOD ".ca"
//#define PTX_MOD ".L1::evict_last"
//#define PTX_MOD ".ca.nc"
#define PTX_MOD ""
  asm volatile("ld.global" PTX_MOD ".v4.b32 {%0, %1, %2, %3}, [%4];"
              : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
              : "l"(src));
#undef PTX_MOD
}

// to make our calculations simple, let's treat fp4x2 as a unit.
// hence, K = number of fp4x2 elements, and 8 elements share
// the same scale.
template <int K, int BLOCK_M, int BLOCK_K, int NUM_WARPS>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE)
void kernel(
  const char   *A_ptr,  // [L,   M, K]
  const char   *B_ptr,  // [L, 128, K]
  const char *SFA_ptr,  // [L,   M, K/8]
  const char *SFB_ptr,  // [L, 128, K/8]
        half   *C_ptr,  // [L,   M]
  int M, int L
) {
  static_assert(BLOCK_K % 16 == 0);  // each thread reads 16 bytes
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  constexpr int SF_BLOCK_K = BLOCK_K / 8;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int batch_id = blockIdx.y;

  constexpr int num_cols = BLOCK_K / 16;  // each thread reads 16-byte at a time
  constexpr int num_rows = TB_SIZE / num_cols;
  int col_id = tid % num_cols;
  int row_id = tid / num_cols;

  int off_m = bid * BLOCK_M;
  A_ptr += batch_id * M * K + off_m * K + col_id * 16;
  B_ptr += batch_id * 128 * K + col_id * 16;
  C_ptr += batch_id * M + off_m;
  SFA_ptr += batch_id * M * (K / 8) + off_m * (K / 8) + col_id * 2;
  SFB_ptr += batch_id * 128 * (K / 8) + col_id * 2;

  // for gmem->rmem
  int A_rmem[BLOCK_M / num_rows][4];
  int B_rmem[4];
  int16_t SFA_rmem[BLOCK_M / num_rows];
  int16_t SFB_rmem;

  // for unpacking to fp16x2
  half2 A_fp16x2[BLOCK_M / num_rows][16];
  half2 B_fp16x2[16];
  half2 SFA_fp16x2[BLOCK_M / num_rows];
  half2 SFB_fp16x2;

  // for accumulation
  half2 acc[BLOCK_M / num_rows][2];
  float master_acc[BLOCK_M / num_rows] = {};

  const int num_iters = K / BLOCK_K;
  for (int iter_k = 0; iter_k < num_iters; iter_k++) {
    ldca_i16(&SFB_rmem, SFB_ptr);
    ldca_i32x4(B_rmem, B_ptr);

    // load
    for (int m = 0; m < BLOCK_M / num_rows; m++) {
      const int row = m * num_rows + row_id;
      ldcs_i16(&SFA_rmem[m], SFA_ptr + row * (K / 8));
      ldcs_i32x4(A_rmem[m], A_ptr + row * K);
    }

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K;
    SFA_ptr += SF_BLOCK_K;
    SFB_ptr += SF_BLOCK_K;

    for (int i = 0; i < 4; i++)
      fp4x8_to_fp16x2x4(reinterpret_cast<int *>(B_fp16x2 + i * 4), B_rmem[i]);
    //fp8x2_to_fp16x2(reinterpret_cast<int *>(&SFB_fp16x2), SFB_rmem);
    SFB_fp16x2 = static_cast<half2>(reinterpret_cast<__nv_fp8x2_e4m3 *>(&SFB_rmem)[0]);

    // unpack
    for (int m = 0; m < BLOCK_M / num_rows; m++) {
      for (int i = 0; i < 4; i++)
        fp4x8_to_fp16x2x4(reinterpret_cast<int *>(A_fp16x2[m] + i * 4), A_rmem[m][i]);
      //fp8x2_to_fp16x2(reinterpret_cast<int *>(&SFA_fp16x2[m]), SFA_rmem[m]);
      SFA_fp16x2[m] = static_cast<half2>(reinterpret_cast<__nv_fp8x2_e4m3 *>(&SFA_rmem[m])[0]);
      SFA_fp16x2[m] = __hmul2(SFA_fp16x2[m], SFB_fp16x2);
    }

    // compute
    for (int m = 0; m < BLOCK_M / num_rows; m++) {
      acc[m][0] = __hmul2(A_fp16x2[m][0], B_fp16x2[0]);  // 1st group
      acc[m][1] = __hmul2(A_fp16x2[m][8], B_fp16x2[8]);  // 2nd group

      for (int i = 1; i < 8; i++) {
        acc[m][0] = __hfma2(A_fp16x2[m][0 + i], B_fp16x2[0 + i], acc[m][0]);  // 1st group
        acc[m][1] = __hfma2(A_fp16x2[m][8 + i], B_fp16x2[8 + i], acc[m][1]);  // 2nd group
      }
    }

    for (int m = 0; m < BLOCK_M / num_rows; m++) {
      __half2_raw scales = SFA_fp16x2[m];
      __half_raw group0 = __hadd(acc[m][0].x, acc[m][0].y);
      __half_raw group1 = __hadd(acc[m][1].x, acc[m][1].y);
      asm volatile("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(master_acc[m]) : "h"(group0.x), "h"(scales.x));
      asm volatile("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(master_acc[m]) : "h"(group1.x), "h"(scales.y));
    }
  }

  if constexpr (num_cols > WARP_SIZE) {
    __shared__ float smem[BLOCK_M / num_rows][TB_SIZE];

    for (int m = 0; m < BLOCK_M / num_rows; m++)
      smem[m][tid] = master_acc[m];
    __syncthreads();

    for (int stride = num_cols / 2; stride >= WARP_SIZE * 2; stride /= 2) {
      if (col_id < stride)
        for (int m = 0; m < BLOCK_M / num_rows; m++) {
          master_acc[m] += smem[m][tid + stride];
          smem[m][tid] = master_acc[m];
        }
      __syncthreads();
    }

    if (col_id < WARP_SIZE)
      for (int m = 0; m < BLOCK_M / num_rows; m++)
        master_acc[m] += smem[m][tid + WARP_SIZE];
  }

  constexpr int start_stride = std::min(num_cols, WARP_SIZE) / 2;
  for (int m = 0; m < BLOCK_M / num_rows; m++)
    for (int stride = start_stride; stride > 0; stride /= 2)
      master_acc[m] += __shfl_down_sync(0xFFFF'FFFF, master_acc[m], stride);

  if (col_id == 0)
    for (int m = 0; m < BLOCK_M / num_rows; m++)
      C_ptr[m * num_rows + row_id] = __float2half(master_acc[m]);
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

#define launch(K_, BLOCK_M, BLOCK_K, NUM_WARPS) \
  else if (K == K_) { \
    dim3 grid(M / BLOCK_M, L); \
    auto this_kernel = kernel<K_, BLOCK_M, BLOCK_K, NUM_WARPS>; \
    this_kernel<<<grid, NUM_WARPS * WARP_SIZE>>>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, M, L); \
  }

  if (false) {}
  launch(8192, 8, 1024, 4)  // benchmark.0
  launch(3584, 8,  512, 4)  // benchmark.1
  launch(1024, 8,  512, 4)  // benchmark.2
  // the rest
  launch( 128, 32, 128, 4)
  launch( 256, 32, 128, 4)
  launch( 768, 32, 128, 4)
  launch(1536, 32, 128, 4)
  launch(2048, 32, 128, 4)
  launch(2304, 32, 128, 4)

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
        # "-gencode=arch=compute_120a,code=sm_120a",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--relocatable-device-code=false",
        # "-lineinfo",
        # "-Xptxas=-v",
        # "--keep",
        # "--keep-dir",
        # f"{Path(__file__).parent}/tmp",
    ],
)
gemv = torch.ops.my_module.gemv


def custom_kernel(data: input_t) -> output_t:
    a, b, sfa, sfb, _, _, c_ref = data
    gemv(a, b, sfa, sfb, c_ref)
    return c_ref
