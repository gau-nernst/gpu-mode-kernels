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
constexpr int THREAD_M = 4;

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
template <int THREAD_K, int NUM_STAGES = 1>
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
  static_assert(THREAD_M == 4);
  static_assert(THREAD_K >= 8);
  static_assert(THREAD_K <= TB_SIZE);
  constexpr int BLOCK_M = (TB_SIZE / THREAD_K) * THREAD_M;
  constexpr int BLOCK_K = THREAD_K * 16;
  constexpr int SF_BLOCK_K = BLOCK_K / 8;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int batch_id = blockIdx.y;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  const int off_m = bid * BLOCK_M;
  const int off_k = (tid % THREAD_K) * 16;  // each thread reads 16 fp4x2 values at a time

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
  char   *A_smem_ld =   A_smem + (tid / THREAD_K) * THREAD_M *    BLOCK_K + off_k;
  char   *B_smem_ld =   B_smem                                            + off_k;
  char *SFA_smem_ld = SFA_smem + (tid / THREAD_K) * THREAD_M * SF_BLOCK_K + (off_k / 8);
  char *SFB_smem_ld = SFB_smem +                                          + (off_k / 8);

  float acc[THREAD_M] = {};

  auto load = [&](int iter_k) {
    // NOTE: since B, SFA, and SFB does not require the whole threadblock to load, we can partition it within the threadblock.
    const int buffer = smem_u32 + (iter_k % NUM_STAGES) * TOTAL_SMEM;
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
    int A_fp4x8[THREAD_M][4], B_fp4x8[4];
    float2 SFA_fp32x2[THREAD_M], SFB_fp32x2;
    int buf_offset = (iter_k % NUM_STAGES) * TOTAL_SMEM;

    for (int m = 0; m < THREAD_M; m++) {
      reinterpret_cast<int4 *>(A_fp4x8[m])[0] = reinterpret_cast<const int4 *>(A_smem_ld + buf_offset + m * BLOCK_K)[0];
      SFA_fp32x2[m] = static_cast<float2>(reinterpret_cast<const __nv_fp8x2_e4m3 *>(SFA_smem_ld + buf_offset + m * SF_BLOCK_K)[0]);
    }

    reinterpret_cast<int4 *>(B_fp4x8)[0] = reinterpret_cast<const int4 *>(B_smem_ld + buf_offset)[0];
    SFB_fp32x2 = static_cast<float2>(reinterpret_cast<const __nv_fp8x2_e4m3 *>(SFB_smem_ld + buf_offset)[0]);

    // unpack to FP32
    int64_t A_fp32x2[THREAD_M][16], B_fp32x2[16];

    for (int m = 0; m < THREAD_M; m++)
      for (int i = 0; i < 4; i++)
        fp4x8_to_fp32x2x4(A_fp4x8[m][i], A_fp32x2[m] + i * 4);

    for (int i = 0; i < 4; i++)
      fp4x8_to_fp32x2x4(B_fp4x8[i], B_fp32x2 + i * 4);

    for (int m = 0; m < THREAD_M; m++)
      for (int group_id = 0; group_id < 2; group_id++) {
        // FMA. manually unroll the 1st iteration
        int64_t sub_acc;
        asm volatile("mul.rn.f32x2 %0, %1, %2;\n"
                    : "=l"(sub_acc)
                    : "l"(A_fp32x2[m][group_id * 8]), "l"(B_fp32x2[group_id * 8]));
        for (int i = 1; i < 8; i++)
          asm volatile("fma.rn.f32x2 %0, %1, %2, %0;\n"
                      : "+l"(sub_acc)
                      : "l"(A_fp32x2[m][group_id * 8 + i]), "l"(B_fp32x2[group_id * 8 + i]));

        float tmp[2];
        std::memcpy(tmp, &sub_acc, sizeof(sub_acc));

        float sfa = reinterpret_cast<float *>(SFA_fp32x2 + m)[group_id];
        float sfb = reinterpret_cast<float *>(&SFB_fp32x2)[group_id];
        acc[m] += (tmp[0] + tmp[1]) * sfa * sfb;
      }
  };

  for (int iter_k = 0; iter_k < NUM_STAGES - 1; iter_k++)
    load(iter_k);

  const int num_iters = K / BLOCK_K;
  for (int iter_k = 0; iter_k < num_iters - (NUM_STAGES - 1); iter_k++) {
    // gmem -> smem
    load(iter_k + NUM_STAGES - 1);

    // smem -> rmem
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 1));
    __syncthreads();  // memory barrier

    compute(iter_k);
    __syncthreads();  // make sure previous compute finish using the buffer
  }

  asm volatile("cp.async.wait_all;\n");
  __syncthreads();  // memory barrier

  for (int k = 0; k < NUM_STAGES - 1; k++)
    compute(num_iters - (NUM_STAGES - 1) + k);

  // this is so cursed
  int64_t acc_fp32x2[2];  // 16-byte in total
  std::memcpy(acc_fp32x2, acc, THREAD_M * sizeof(float));

  // threadblock reduction
  if constexpr (THREAD_K > WARP_SIZE) {
    // reuse dynamic smem
    // using layout float red_smem[TB_SIZE][4]
    float *red_smem = reinterpret_cast<float *>(smem);

    // 16-byte store
    reinterpret_cast<float4 *>(red_smem)[tid] = reinterpret_cast<float4 *>(acc_fp32x2)[0];
    __syncthreads();

    for (int stride = THREAD_K / 2; stride >= WARP_SIZE; stride /= 2) {
      if ((tid % THREAD_K) < stride) {
        int64_t tmp[2];

        // 16-byte load
        reinterpret_cast<float4 *>(tmp)[0] = reinterpret_cast<float4 *>(red_smem)[tid + stride];

        asm volatile("add.rn.f32x2 %0, %0, %1;\n" : "+l"(acc_fp32x2[0]) : "l"(tmp[0]));
        asm volatile("add.rn.f32x2 %0, %0, %1;\n" : "+l"(acc_fp32x2[1]) : "l"(tmp[1]));

        // 16-byte store
        reinterpret_cast<float4 *>(red_smem)[tid] = reinterpret_cast<float4 *>(acc_fp32x2)[0];
      }
      __syncthreads();
    }
  }

  // warp reduction
  constexpr int start_stride = std::min(THREAD_K, WARP_SIZE) / 2;
  for (int stride = start_stride; stride > 0; stride /= 2) {
    int64_t tmp[2];
    tmp[0] = __shfl_down_sync(0xFFFF'FFFF, acc_fp32x2[0], stride);
    tmp[1] = __shfl_down_sync(0xFFFF'FFFF, acc_fp32x2[1], stride);
    asm volatile("add.rn.f32x2 %0, %0, %1;\n" : "+l"(acc_fp32x2[0]) : "l"(tmp[0]));
    asm volatile("add.rn.f32x2 %0, %0, %1;\n" : "+l"(acc_fp32x2[1]) : "l"(tmp[1]));
  }

  if (tid % THREAD_K == 0) {
    half2 out[2];
    out[0] = __float22half2_rn(reinterpret_cast<float2 *>(&acc_fp32x2)[0]);
    out[1] = __float22half2_rn(reinterpret_cast<float2 *>(&acc_fp32x2)[1]);

    // 8-byte store
    half *out_ptr = C_ptr + (batch_id * M) + off_m + (tid / THREAD_K) * THREAD_M;
    reinterpret_cast<int2 *>(out_ptr)[0] = reinterpret_cast<int2 *>(out)[0];
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
  constexpr int NUM_STAGES = 2;

#define launch(THREAD_K) { \
  int BLOCK_M = (TB_SIZE / THREAD_K) * THREAD_M; \
  int BLOCK_K = THREAD_K * 16; \
  int TOTAL_SMEM = (BLOCK_M + 1) * (BLOCK_K + BLOCK_K / 8); \
  dim3 grid(M / BLOCK_M, L); \
  int smem_size = TOTAL_SMEM * NUM_STAGES; \
  kernel<THREAD_K, NUM_STAGES><<<grid, TB_SIZE, smem_size, stream>>>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L, M, K); \
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
