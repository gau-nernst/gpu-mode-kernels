#!POPCORN leaderboard nvfp4_gemv

import gzip
import json
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
void ldcs_i16(int16_t *dst, const void *src) {
  asm volatile("ld.global.L1::no_allocate.b16 %0, [%1];" : "=h"(dst[0]) : "l"(src));
}

__device__
void ldca_i16(int16_t *dst, const void *src) {
  asm volatile("ld.global.L1::evict_last.b16 %0, [%1];" : "=h"(dst[0]) : "l"(src));
}


__device__
void ldcs_i16x2(int16_t *dst, const void *src) {
  asm volatile("ld.global.L1::no_allocate.v2.b16 {%0, %1}, [%2];\n" : "=h"(dst[0]), "=h"(dst[1]) : "l"(src));
}

__device__
void ldca_i16x2(int16_t *dst, const void *src) {
  asm volatile("ld.global.L1::evict_last.v2.b16 {%0, %1}, [%2];\n" : "=h"(dst[0]), "=h"(dst[1]) : "l"(src));
}

__device__
void ldcs_i32x4(int *dst, const void *src) {
  asm volatile("ld.global.L1::no_allocate.v4.b32 {%0, %1, %2, %3}, [%4];"
              : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
              : "l"(src));
}

__device__
void ldca_i32x4(int *dst, const void *src) {
  asm volatile("ld.global.L1::evict_last.v4.b32 {%0, %1, %2, %3}, [%4];"
              : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
              : "l"(src));
}

__device__
void ldcs_i32x8(int *dst, const void *src) {
  asm volatile("ld.global.L1::no_allocate.L2::evict_first.v8.b32 "
              "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
              : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]),
                "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7])
              : "l"(src));
}

__device__
void ldca_i32x8(int *dst, const void *src) {
  asm volatile("ld.global.L1::evict_last.L2::evict_last.v8.b32 "
              "{%0, %1, %2, %3, %4, %5, %6, %7}, [%8];\n"
              : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]),
                "=r"(dst[4]), "=r"(dst[5]), "=r"(dst[6]), "=r"(dst[7])
              : "l"(src));
}

// to make our calculations simple, let's treat fp4x2 as a unit.
// hence, K = number of fp4x2 elements, and 8 elements share
// the same scale.
template <int BLOCK_M, int BLOCK_K, int K, int NUM_WARPS, int CP_SIZE>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE)
void kernel(
  const char   *A_ptr,  // [L,   M, K]
  const char   *B_ptr,  // [L, 128, K]
  const char *SFA_ptr,  // [L,   M, K/8]
  const char *SFB_ptr,  // [L, 128, K/8]
        half   *C_ptr,  // [L,   M]
  int L, int M
) {
  static_assert(BLOCK_K % CP_SIZE == 0);  // each thread reads 16 bytes
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  constexpr int SF_BLOCK_K = BLOCK_K / 8;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int batch_id = blockIdx.y;

  constexpr int num_cols = BLOCK_K / CP_SIZE;  // each thread reads 16-byte at a time
  static_assert(num_cols <= TB_SIZE);
  constexpr int num_rows = TB_SIZE / num_cols;

  const int t_col = tid % num_cols;
  const int t_row = tid / num_cols;

  {
    int off_m = bid * BLOCK_M;
    int off_k = t_col * CP_SIZE;
    A_ptr += batch_id * M * K + off_m * K + off_k;
    B_ptr += batch_id * 128 * K + off_k;
    C_ptr += batch_id * M + off_m;
    SFA_ptr += (batch_id * M * K + off_m * K + off_k) / 8;
    SFB_ptr += (batch_id * 128 * K + off_k) / 8;
  }

  // load A & B
  int A_rmem[BLOCK_M / num_rows][CP_SIZE / 4];
  int B_rmem[CP_SIZE / 4];
  int16_t SFA_rmem[BLOCK_M / num_rows][CP_SIZE / 16];
  int16_t SFB_rmem[CP_SIZE / 16];

  half2 A_fp16x2[BLOCK_M / num_rows][CP_SIZE / 16][16];
  half2 B_fp16x2[CP_SIZE / 16][16];
  half2 SFA_fp16x2[BLOCK_M / num_rows][CP_SIZE / 16];
  half2 SFB_fp16x2[CP_SIZE / 16];

  half2 acc[BLOCK_M / num_rows][CP_SIZE / 16][2];
  float master_acc[BLOCK_M / num_rows] = {};

  const int num_iters = K / BLOCK_K;
  for (int iter_k = 0; iter_k < num_iters; iter_k++) {
    // load
    if constexpr (CP_SIZE == 16) {
      ldca_i16(SFB_rmem, SFB_ptr);
      ldca_i32x4(B_rmem, B_ptr);
      for (int m = 0; m < BLOCK_M / num_rows; m++) {
        const int row = m * num_rows + t_row;
        ldcs_i16(SFA_rmem[m], SFA_ptr + row * K / 8);
        ldcs_i32x4(A_rmem[m], A_ptr + row * K);
      }
    }
    else if constexpr (CP_SIZE == 32) {
      ldca_i16x2(SFB_rmem, SFB_ptr);
      ldca_i32x8(B_rmem, B_ptr);
      for (int m = 0; m < BLOCK_M / num_rows; m++) {
        const int row = m * num_rows + t_row;
        ldcs_i16x2(SFA_rmem[m], SFA_ptr + row * K / 8);
        ldcs_i32x8(A_rmem[m], A_ptr + row * K);
      }
    }

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K;
    SFA_ptr += SF_BLOCK_K;
    SFB_ptr += SF_BLOCK_K;

    // unpack B
    for (int i = 0; i < CP_SIZE / 16; i++) {
      SFB_fp16x2[i] = static_cast<half2>(reinterpret_cast<__nv_fp8x2_e4m3 *>(&SFB_rmem)[i]);
      for (int j = 0; j < 4; j++)
        fp4x8_to_fp16x2x4(reinterpret_cast<int *>(&B_fp16x2[i][j * 4]), B_rmem[i * 4 + j]);
    }

    // unpack A
    for (int m = 0; m < BLOCK_M / num_rows; m++)
      for (int i = 0; i < CP_SIZE / 16; i++) {
        SFA_fp16x2[m][i] = static_cast<half2>(reinterpret_cast<__nv_fp8x2_e4m3 *>(&SFA_rmem[m])[i]);
        for (int j = 0; j < 4; j++)
          fp4x8_to_fp16x2x4(reinterpret_cast<int *>(&A_fp16x2[m][i][j * 4]), A_rmem[m][i * 4 + j]);
        SFA_fp16x2[m][i] = __hmul2(SFA_fp16x2[m][i], SFB_fp16x2[i]);  // pre-multiply scale
      }

    // compute
    for (int m = 0; m < BLOCK_M / num_rows; m++)
      for (int i = 0; i < CP_SIZE / 16; i++) {
      acc[m][i][0] = __hmul2(A_fp16x2[m][i][0], B_fp16x2[i][0]);
      acc[m][i][1] = __hmul2(A_fp16x2[m][i][8], B_fp16x2[i][8]);
      for (int j = 1; j < 8; j++) {
        acc[m][i][0] = __hfma2(A_fp16x2[m][i][0 + j], B_fp16x2[i][0 + j], acc[m][i][0]);
        acc[m][i][1] = __hfma2(A_fp16x2[m][i][8 + j], B_fp16x2[i][8 + j], acc[m][i][1]);
      }
    }

    for (int m = 0; m < BLOCK_M / num_rows; m++)
      for (int i = 0; i < CP_SIZE / 16; i++) {
        __half2_raw scales = SFA_fp16x2[m][i];
        __half_raw group0 = __hadd(acc[m][i][0].x, acc[m][i][0].y);
        __half_raw group1 = __hadd(acc[m][i][1].x, acc[m][i][1].y);
        asm volatile("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(master_acc[m]) : "h"(group0.x), "h"(scales.x));
        asm volatile("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(master_acc[m]) : "h"(group1.x), "h"(scales.y));
      }
  }

  if constexpr (NUM_WARPS % 2 == 0) {
    if constexpr (num_cols > WARP_SIZE) {
      __shared__ float smem[BLOCK_M / num_rows][TB_SIZE];

      for (int m = 0; m < BLOCK_M / num_rows; m++)
        smem[m][tid] = master_acc[m];
      __syncthreads();

      for (int stride = num_cols / 2; stride >= WARP_SIZE * 2; stride /= 2) {
        if (t_col < stride)
          for (int m = 0; m < BLOCK_M / num_rows; m++) {
            master_acc[m] += smem[m][tid + stride];
            smem[m][tid] = master_acc[m];
          }
        __syncthreads();
      }

      if (t_col < WARP_SIZE)
        for (int m = 0; m < BLOCK_M / num_rows; m++)
          master_acc[m] += smem[m][tid + WARP_SIZE];
    }

    constexpr int start_stride = std::min(num_cols, WARP_SIZE) / 2;
    for (int stride = start_stride; stride > 0; stride /= 2)
      for (int m = 0; m < BLOCK_M / num_rows; m++)
        master_acc[m] += __shfl_down_sync(0xFFFF'FFFF, master_acc[m], stride);

    if (t_col == 0)
      for (int m = 0; m < BLOCK_M / num_rows; m++)
        C_ptr[m * num_rows + t_row] = __float2half(master_acc[m]);
  }
  else {
    // this is for benchmark.2, when NUM_WARPS = 7
    __shared__ float smem[BLOCK_M / num_rows][(NUM_WARPS - 1) * WARP_SIZE];

    const int warp_id = tid / WARP_SIZE;
    if (warp_id > 0)
      for (int m = 0; m < BLOCK_M / num_rows; m++)
        smem[m][tid - WARP_SIZE] = master_acc[m];
    __syncthreads();

    if (warp_id == 0) {
      for (int w = 0; w < NUM_WARPS - 1; w++)
        for (int m = 0; m < BLOCK_M; m++)
          master_acc[m] += smem[m][tid + w * WARP_SIZE];

      constexpr int start_stride = std::min(num_cols, WARP_SIZE) / 2;
      for (int stride = start_stride; stride > 0; stride /= 2)
        for (int m = 0; m < BLOCK_M / num_rows; m++)
          master_acc[m] += __shfl_down_sync(0xFFFF'FFFF, master_acc[m], stride);

      if (t_col == 0)
        for (int m = 0; m < BLOCK_M / num_rows; m++)
          C_ptr[m * num_rows + t_row] = __float2half(master_acc[m]);
    }
  }
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

#define launch(K_, BLOCK_M, BLOCK_K, NUM_WARPS, CP_SIZE) \
  else if (K == K_) { \
    dim3 grid(M / BLOCK_M, L); \
    auto this_kernel = kernel<BLOCK_M, BLOCK_K, K_, NUM_WARPS, CP_SIZE>; \
    this_kernel<<<grid, NUM_WARPS * WARP_SIZE>>>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L, M); \
  }

  if (false) {}
  launch(8192, 1, 8192, 8, 32)   // benchmark.0
  launch(3584, 2, 3584, 7, 16)   // benchmark.1
  //launch(3584, 8,  512, 4, 16)   // benchmark.1 - without using 7 warps LMAO
  launch(1024, 8,  512, 4, 16)   // benchmark.2
  // the rest
  launch( 128, 32, 128, 4, 16)
  launch( 256, 32, 128, 4, 16)
  launch( 768, 32, 128, 4, 16)
  launch(1536, 32, 128, 4, 16)
  launch(2048, 32, 128, 4, 16)
  launch(2304, 32, 128, 4, 16)

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
