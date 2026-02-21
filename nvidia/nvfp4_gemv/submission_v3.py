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

constexpr int MMA_M = 16;
constexpr int MMA_N = 8;
constexpr int MMA_K = 16;

// each warp handles 16 rows
constexpr int BLOCK_M = MMA_M * NUM_WARPS;

__device__
int fp4x2_to_fp16x2(char in) {
  int out;
  short in_i16 = in;  // a bit wasteful
  asm volatile(
    "{\n"
    ".reg .b8 tmp;\n"
    "cvt.s8.s16 tmp, %1;\n"
    "cvt.rn.f16x2.e2m1x2 %0, tmp;\n"
    "}\n"
    : "=r"(out)
    : "h"(in_i16)
  );
  return out;
}

// NOTE: ignore C
__device__
void mma_fp16(int *A, int *B, float *D) {
  float zero = 0.0f;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
              "{%0, %1, %2, %3}, "
              "{%4, %5, %6, %7}, "
              "{%8, %9}, "
              "{%10, %11, %12, %13};"
              : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
              : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
                "r"(B[0]), "r"(B[1]),
                "f"(zero), "f"(zero), "f"(zero), "f"(zero));
}

template <int HEIGHT, int WIDTH, int TB_SIZE, typename T>
__device__
void cp_async_2d(int dst, const T *src, int src_stride, int tid) {
  auto load = [&](int idx) {
    const int row = idx / WIDTH;
    const int col = idx % WIDTH;

    const int dst_addr = dst + idx * sizeof(T);
    const T *src_addr = src + (row * src_stride + col);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(dst_addr), "l"(src_addr));
  };

  constexpr int num_elems = 16 / sizeof(T);
  constexpr int num_iters = HEIGHT * WIDTH / (TB_SIZE * num_elems);

  for (int iter = 0; iter < num_iters; iter++)
    load((iter * TB_SIZE + tid) * num_elems);

  // handle the case when tile size is not divisible by threadblock size
  if constexpr ((HEIGHT * WIDTH) % (TB_SIZE * num_elems) != 0) {
    const int idx = (num_iters * TB_SIZE + tid) * num_elems;
    if (idx < HEIGHT * WIDTH)
      load(idx);
  }
}

// to make our calculations simple, let's treat fp4x2 as a unit.
// hence, K = number of fp4x2 elements, and 8 elements share
// the same scale.
template <int BLOCK_K, int NUM_STAGES = 1>
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
  constexpr int SF_BLOCK_K = BLOCK_K / 8;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int batch_id = blockIdx.y;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  const int off_m = bid * BLOCK_M;

  A_ptr += (batch_id *   M * K) + (off_m * K);
  B_ptr += (batch_id * 128 * K);

  SFA_ptr += (batch_id *   M * (K / 8)) + (off_m * (K / 8));
  SFB_ptr += (batch_id * 128 * (K / 8));

  // set up smem
  extern __shared__ char smem[];
  const int smem_u32 = static_cast<int>(__cvta_generic_to_shared(smem));
  constexpr int TOTAL_SMEM = (BLOCK_M * BLOCK_K) + (BLOCK_K) + (BLOCK_M * SF_BLOCK_K) + SF_BLOCK_K;

  char   *A_smem = smem;
  char   *B_smem =   A_smem + BLOCK_M * BLOCK_K;
  char *SFA_smem =   B_smem + BLOCK_K;
  char *SFB_smem = SFA_smem + BLOCK_M * SF_BLOCK_K;

  // to be used for smem->rmem load
  char   *A_smem_ld =   A_smem + (warp_id * MMA_M + (lane_id / 4)) *    BLOCK_K + (lane_id % 4);
  char   *B_smem_ld =   B_smem                                                  + (lane_id % 4);
  char *SFA_smem_ld = SFA_smem + (warp_id * MMA_M + (lane_id / 4)) * SF_BLOCK_K;
  char *SFB_smem_ld = SFB_smem;

  float2 acc2 = {};

  auto load = [&](int iter_k) {
    const int buffer  = smem_u32 + (iter_k % NUM_STAGES) * TOTAL_SMEM;
    const int A_buf   = buffer;
    const int B_buf   = A_buf + BLOCK_M * BLOCK_K;
    const int SFA_buf = B_buf + BLOCK_K;
    const int SFB_buf = SFA_buf + BLOCK_M * SF_BLOCK_K;

    cp_async_2d<BLOCK_M,    BLOCK_K, TB_SIZE>(  A_buf,   A_ptr,     K, tid);
    cp_async_2d<      1,    BLOCK_K, TB_SIZE>(  B_buf,   B_ptr,     K, tid);
    cp_async_2d<BLOCK_M, SF_BLOCK_K, TB_SIZE>(SFA_buf, SFA_ptr, K / 8, tid);
    cp_async_2d<      1, SF_BLOCK_K, TB_SIZE>(SFB_buf, SFB_ptr, K / 8, tid);

    asm volatile("cp.async.commit_group;\n");

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K;
    SFA_ptr += SF_BLOCK_K;
    SFB_ptr += SF_BLOCK_K;
  };

  auto compute = [&](int iter_k) {
    int buf_offset = (iter_k % NUM_STAGES) * TOTAL_SMEM;

    constexpr int num_mma_k = BLOCK_K * 2 / MMA_K;  // times 2 because BLOCK_K is counting fp4x2 elems
    int A_rmem[num_mma_k][4];
    int B_rmem[num_mma_k][2];
    float sub_acc[num_mma_k][4];

    // each MMA is M16N8K16
    // A: M16K16 is M16K16. width = 16 FP4 = 8 bytes
    // bank conflicts...
    for (int mma_k = 0; mma_k < num_mma_k; mma_k++) {
      A_rmem[mma_k][0] = fp4x2_to_fp16x2(A_smem_ld[buf_offset + 0 * BLOCK_K + mma_k * 8 + 0]);
      A_rmem[mma_k][1] = fp4x2_to_fp16x2(A_smem_ld[buf_offset + 8 * BLOCK_K + mma_k * 8 + 0]);
      A_rmem[mma_k][2] = fp4x2_to_fp16x2(A_smem_ld[buf_offset + 0 * BLOCK_K + mma_k * 8 + 4]);
      A_rmem[mma_k][3] = fp4x2_to_fp16x2(A_smem_ld[buf_offset + 8 * BLOCK_K + mma_k * 8 + 4]);
    }

    // we only need the first 4 lanes, but do it for everyone anyway
    // we will discard unwanted results later
    for (int mma_k = 0; mma_k < num_mma_k; mma_k++) {
      B_rmem[mma_k][0] = fp4x2_to_fp16x2(B_smem_ld[buf_offset + mma_k * 8 + 0]);
      B_rmem[mma_k][1] = fp4x2_to_fp16x2(B_smem_ld[buf_offset + mma_k * 8 + 4]);
    }

    for (int mma_k = 0; mma_k < num_mma_k; mma_k++)
      mma_fp16(A_rmem[mma_k], B_rmem[mma_k], sub_acc[mma_k]);

    for (int mma_k = 0; mma_k < num_mma_k; mma_k++) {
      float sfa0 = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 *>(SFA_smem_ld)[buf_offset + 0 * SF_BLOCK_K + mma_k]);
      float sfa1 = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 *>(SFA_smem_ld)[buf_offset + 8 * SF_BLOCK_K + mma_k]);
      float sfb  = static_cast<float>(reinterpret_cast<__nv_fp8_e4m3 *>(SFB_smem_ld)[buf_offset                  + mma_k]);

      // only keep the first 2 columns
      acc2.x += sub_acc[mma_k][0] * sfa0 * sfb;
      acc2.y += sub_acc[mma_k][2] * sfa1 * sfb;
    }
  };

  for (int iter_k = 0; iter_k < NUM_STAGES - 1; iter_k++)
    load(iter_k);

  const int num_iters = K / BLOCK_K;
  for (int iter_k = 0; iter_k < num_iters - (NUM_STAGES - 1); iter_k++) {
    // gmem -> smem
    load(iter_k + NUM_STAGES - 1);

    asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 1));
    __syncthreads();  // memory barrier

    compute(iter_k);
    __syncthreads();  // make sure finish using the buffer for the next prefetch
  }

  asm volatile("cp.async.wait_all;\n");
  __syncthreads();  // memory barrier

  for (int k = 0; k < NUM_STAGES - 1; k++)
    compute(num_iters - (NUM_STAGES - 1) + k);

  // the final result is in lane0, lane4, ..., lane28
  if (lane_id % 4 == 0) {
    half2 out = __float22half2_rn(acc2);

    half *out_ptr = C_ptr + (batch_id * M) + (off_m + warp_id * MMA_M + (lane_id / 4));
    out_ptr[0] = out.x;
    out_ptr[8] = out.y;
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

  auto stream = at::cuda::getCurrentCUDAStream();

#define launch(BLOCK_K, NUM_STAGES) { \
  int TOTAL_SMEM = (BLOCK_M + 1) * (BLOCK_K + BLOCK_K / 8); \
  dim3 grid(M / BLOCK_M, L); \
  int smem_size = TOTAL_SMEM * NUM_STAGES; \
  auto this_kernel = kernel<BLOCK_K, NUM_STAGES>; \
  if (smem_size > 48'000) \
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
  this_kernel<<<grid, TB_SIZE, smem_size, stream>>>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L, M, K); \
}

  launch(256, 2);
  return;
  if (false) {}
  else if (K == 8192) launch(128, 2)  // benchmark.0
  else if (K == 3584) launch(128, 2)  // benchmark.1
  else if (K == 1024) launch(128, 2)  // benchmark.2
  else launch(128, 2)                 // the rest

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
