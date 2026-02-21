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
#include <cudaTypedefs.h>
#include <chrono>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int WARP_SIZE = 32;

__device__
void fp4x8_to_fp16x2x4(int in, half2 *out) {
  int *out_i32 = reinterpret_cast<int *>(out);
  asm volatile(
    "{\n"
    ".reg .b8 tmp0, tmp1, tmp2, tmp3;\n"
    "mov.b32 {tmp0, tmp1, tmp2, tmp3}, %4; // unpack 32-bit register to 4x fp4x2\n"
    "cvt.rn.f16x2.e2m1x2 %0, tmp0; // PTX only supports FP4->FP16\n"
    "cvt.rn.f16x2.e2m1x2 %1, tmp1;\n"
    "cvt.rn.f16x2.e2m1x2 %2, tmp2;\n"
    "cvt.rn.f16x2.e2m1x2 %3, tmp3;\n"
    "}\n"
    : "=r"(out_i32[0]), "=r"(out_i32[1]), "=r"(out_i32[2]), "=r"(out_i32[3])
    : "r"(in)
  );
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/arch/cluster_sm90.hpp#L180
__device__
uint32_t elect_one_sync() {
  uint32_t pred = 0;
  uint32_t laneid = 0;
  asm volatile(
    "{\n"
    ".reg .b32 %%rx;\n"
    ".reg .pred %%px;\n"
    "     elect.sync %%rx|%%px, %2;\n"
    "@%%px mov.s32 %1, 1;\n"
    "     mov.s32 %0, %%rx;\n"
    "}\n"
    : "+r"(laneid), "+r"(pred)
    : "r"(0xFFFFFFFF));
  return pred;
}

__device__
void cp_async_bulk(int dst, const void *src, int size, int mbar_addr) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n"
              :: "r"(dst), "l"(src), "r"(size), "r"(mbar_addr));
}

__device__
void cp_async_bulk_policy(int dst, const void *src, int size, int mbar_addr, int64_t policy) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;\n"
              :: "r"(dst), "l"(src), "r"(size), "r"(mbar_addr), "l"(policy));
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cutlass/arch/barrier.h#L408
__device__
void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;
  asm volatile(
    "{\n"
    ".reg .pred P1;\n"
    "LAB_WAIT:\n"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n"
    "@P1     bra.uni DONE;\n"
    "bra.uni LAB_WAIT;\n"
    "DONE:\n"
    "}\n"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks)
  );
}

// trick to get 128-byte aligned shared memory
typedef struct __align__(128) {} Aligned128B;

__device__ inline int64_t globaltimer() {
  int64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t) :: "memory");
  return t;
}

struct Profiler {
  int64_t *data_ptr_;
  int sm_id_;
  int cnt_;

  __device__
  void init(int64_t *data_ptr, int bid) {
    data_ptr_ = data_ptr + bid * (1 + 1000 * 4);  // each block got 1000 entries
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(sm_id_));
    cnt_ = 0;
  }

  __device__
  void start(int tag) {
    data_ptr_[1 + cnt_ * 4 + 0] = sm_id_;
    data_ptr_[1 + cnt_ * 4 + 1] = tag;
    data_ptr_[1 + cnt_ * 4 + 2] = globaltimer();
  }

  __device__
  void stop() {
    data_ptr_[1 + cnt_ * 4 + 3] = globaltimer() - data_ptr_[1 + cnt_ * 4 + 2];
    cnt_ += 1;
  }

  __device__
  void flush() {
    data_ptr_[0] = cnt_;
  }
};

// to make our calculations simple, let's treat fp4x2 as a unit.
// hence, K = number of fp4x2 elements, and 8 elements share
// the same scale.
template <int BLOCK_M, int BLOCK_K, int NUM_WARPS, int NUM_STAGES, bool DO_PROFILE>
__global__
__launch_bounds__((1 + NUM_WARPS) * WARP_SIZE)
void kernel(
  const char   *A_ptr,  // [L,   M, K]
  const char   *B_ptr,  // [L, 128, K]
  const char *SFA_ptr,  // [L,   M, K/8]
  const char *SFB_ptr,  // [L, 128, K/8]
        half   *C_ptr,  // [L,   M]
  int L, int M, int K,
  int64_t *profiler_ptr
) {
  static_assert(BLOCK_K % 16 == 0);  // each thread reads 16 bytes
  static_assert(BLOCK_M % NUM_WARPS == 0);
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;  // only includes compute warps
  constexpr int WARP_M = BLOCK_M / NUM_WARPS;
  constexpr int SF_BLOCK_K = BLOCK_K / 8;

  // we organize threads in a warp as [THREAD_M][THREAD_K]
  // hence, each thread holds A[WARP_M / THREAD_M][BLOCK_K / THREAD_K] and B[BLOCK_K / THREAD_K]
  constexpr int THREAD_K = std::min(BLOCK_K / 16, WARP_SIZE);  // how many threads in a warp are required to read a row of BLOCK_K
  constexpr int THREAD_M = WARP_SIZE / THREAD_K;
  static_assert(WARP_M % THREAD_M == 0);

  const int tid = threadIdx.x;
  const int start_bid = blockIdx.y;
  const int batch_id = blockIdx.x;
  const int num_bids = gridDim.y;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  // DMA warp and FMA warp
  Profiler profiler;
  if constexpr (DO_PROFILE) if (lane_id == 0)
    profiler.init(profiler_ptr, (batch_id * num_bids + start_bid) * (1 + NUM_WARPS) + warp_id);

  auto prof_start = [&](int tag) {
    if constexpr (DO_PROFILE) if (lane_id == 0)
      profiler.start(tag);
  };
  auto prof_stop = [&]() {
    if constexpr (DO_PROFILE) if (lane_id == 0)
      profiler.stop();
  };

  // set up smem
  extern __shared__ Aligned128B smem[];

  constexpr int   A_size = BLOCK_M *    BLOCK_K;
  constexpr int   B_size =       1 *    BLOCK_K;
  constexpr int SFA_size = BLOCK_M * SF_BLOCK_K;
  constexpr int SFB_size =       1 * SF_BLOCK_K;

  char   *A_smem = reinterpret_cast<char *>(smem);
  char   *B_smem =   A_smem +   A_size * NUM_STAGES;
  char *SFA_smem =   B_smem +   B_size * NUM_STAGES;
  char *SFB_smem = SFA_smem + SFA_size * NUM_STAGES;

  // set up mbarrier
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t dma_mbars[NUM_STAGES], fma_mbars[NUM_STAGES];
  int dma_mbar_addrs[NUM_STAGES], fma_mbar_addrs[NUM_STAGES];
  for (int i = 0; i < NUM_STAGES; i++) {
    dma_mbar_addrs[i] = static_cast<int>(__cvta_generic_to_shared(dma_mbars + i));
    fma_mbar_addrs[i] = static_cast<int>(__cvta_generic_to_shared(fma_mbars + i));
  }

  if (warp_id == 0 && elect_one_sync()) {
    // for DMA, only 1 thread issues.
    // for FMA, all threads (among compute warps).
    for (int i = 0; i < NUM_STAGES; i++) {
      asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(dma_mbar_addrs[i]), "r"(1));
      asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(fma_mbar_addrs[i]), "r"(TB_SIZE));
    }

    // initialized mbarrier visible to TMA engine (async proxy)
    asm volatile("fence.mbarrier_init.release.cluster;\n");
  }
  __syncthreads();  // initialized mbarrier visible to all threads.

  // https://github.com/deepseek-ai/DeepGEMM/blob/v2.1.1/deep_gemm/include/deep_gemm/impls/sm100_bf16_gemm.cuh#L151
  // this is called at the END of the k-loop
  int phase = 0;
  int stage_id = 0;
  auto advance_pipeline = [&]() {
    stage_id = (stage_id + 1) % NUM_STAGES;
    phase ^= stage_id == 0;  // flip phase parity once we have cycled through all stages
  };

  // DMA warp. gmem->smem
  if (warp_id == 0 && elect_one_sync()) {
    int64_t evict_first;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0;\n" : "=l"(evict_first));

    const int   A_smem_u32 = static_cast<int>(__cvta_generic_to_shared(  A_smem));
    const int   B_smem_u32 = static_cast<int>(__cvta_generic_to_shared(  B_smem));
    const int SFA_smem_u32 = static_cast<int>(__cvta_generic_to_shared(SFA_smem));
    const int SFB_smem_u32 = static_cast<int>(__cvta_generic_to_shared(SFB_smem));

    // persistent kernel
    for (int bid = start_bid; bid < M / BLOCK_M; bid += num_bids) {
      // gmem offsets
      const int off_m = bid * BLOCK_M;
      const char *A_gmem = A_ptr + (batch_id *   M * K) + (off_m * K);
      const char *B_gmem = B_ptr + (batch_id * 128 * K);
      const char *SFA_gmem = SFA_ptr + (batch_id *   M * (K / 8)) + (off_m * (K / 8));
      const char *SFB_gmem = SFB_ptr + (batch_id * 128 * (K / 8));

      const int num_iters = K / BLOCK_K;
      for (int iter_k = 0; iter_k < num_iters; iter_k++, advance_pipeline()) {
        prof_start(TAG_WAIT_COMPUTE);
        // wait for compute warps to finish
        // fma_mbar is initially available. so we init phase for DMA warp with 1.
        // -> the first mbar_wait is immediate.
        mbarrier_wait(fma_mbar_addrs[stage_id], phase ^ 1);
        prof_stop();

        prof_start(TAG_LOAD);
        const int mbar_addr = dma_mbar_addrs[stage_id];

        for (int row = 0; row < BLOCK_M; row++) {
          cp_async_bulk_policy(  A_smem_u32 + stage_id *   A_size + row *    BLOCK_K,   A_gmem + row *       K,    BLOCK_K, mbar_addr, evict_first);
          cp_async_bulk_policy(SFA_smem_u32 + stage_id * SFA_size + row * SF_BLOCK_K, SFA_gmem + row * (K / 8), SF_BLOCK_K, mbar_addr, evict_first);
        }

        cp_async_bulk(  B_smem_u32 + stage_id *   B_size,   B_gmem,    BLOCK_K, mbar_addr);
        cp_async_bulk(SFB_smem_u32 + stage_id * SFB_size, SFB_gmem, SF_BLOCK_K, mbar_addr);

        // signal done
        constexpr int cp_size = A_size + SFA_size + B_size + SFB_size;
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                    :: "r"(mbar_addr), "r"(cp_size) : "memory");

        A_gmem += BLOCK_K;
        B_gmem += BLOCK_K;
        SFA_gmem += SF_BLOCK_K;
        SFB_gmem += SF_BLOCK_K;
        prof_stop();
      }
    }
  }
  // compute warps
  else if (warp_id > 0) {
    // remove DMA warp
    const int warp_id_ = warp_id - 1;
    const int tid_ = tid - WARP_SIZE;

    // to be used for smem->rmem load
    const int off_m = (warp_id_ * WARP_M) + (lane_id / THREAD_K);
    const int off_k = (lane_id % THREAD_K) * 16;  // each thread reads 16 fp4x2 values at a time

    A_smem += off_m * BLOCK_K + off_k;
    B_smem += off_k;
    SFA_smem += off_m * SF_BLOCK_K + (off_k / 8);
    SFB_smem += (off_k / 8);

    for (int bid = start_bid; bid < M / BLOCK_M; bid += num_bids) {
      float acc[WARP_M / THREAD_M] = {};

      const int num_iters = K / BLOCK_K;
      for (int iter_k = 0; iter_k < num_iters; iter_k++, advance_pipeline()) {
        prof_start(TAG_WAIT_LOAD);
        // basically a spin lock. acquire semantics
        mbarrier_wait(dma_mbar_addrs[stage_id], phase);
        prof_stop();

        prof_start(TAG_COMPUTE);

        // smem -> rmem
        int4 A_fp4x32[WARP_M / THREAD_M][BLOCK_K / (THREAD_K * 16)];
        int4 B_fp4x32[BLOCK_K / (THREAD_K * 16)];
        __nv_fp8x2_e4m3 SFA_fp8x2[WARP_M / THREAD_M][BLOCK_K / (THREAD_K * 16)];
        __nv_fp8x2_e4m3 SFB_fp8x2[BLOCK_K / (THREAD_K * 16)];

        for (int k = 0; k < BLOCK_K / (THREAD_K * 16); k++) {
          int col = k * THREAD_K * 16;
          B_fp4x32[k] = reinterpret_cast<int4 *>(B_smem + stage_id * B_size + col)[0];
          SFB_fp8x2[k] = reinterpret_cast<__nv_fp8x2_e4m3 *>(SFB_smem + stage_id * SFB_size + col / 8)[0];
        }

        for (int m = 0; m < WARP_M / THREAD_M; m++)
          for (int k = 0; k < BLOCK_K / (THREAD_K * 16); k++) {
            int row = m * THREAD_M;
            int col = k * THREAD_K * 16;
            A_fp4x32[m][k] = reinterpret_cast<int4 *>(A_smem + stage_id * A_size + row * BLOCK_K + col)[0];
            SFA_fp8x2[m][k] = reinterpret_cast<__nv_fp8x2_e4m3 *>(SFA_smem + stage_id * SFA_size + row * SF_BLOCK_K + col / 8)[0];
          }

        // unpack to fp16x2
        half2 A_fp16x2[WARP_M / THREAD_M][BLOCK_K / (THREAD_K * 16)][16];
        half2 B_fp16x2[BLOCK_K / (THREAD_K * 16)][16];
        half2 SFA_fp16x2[WARP_M / THREAD_M][BLOCK_K / (THREAD_K * 16)];
        half2 SFB_fp16x2[BLOCK_K / (THREAD_K * 16)];

        for (int k = 0; k < BLOCK_K / (THREAD_K * 16); k++) {
          fp4x8_to_fp16x2x4(B_fp4x32[k].x, B_fp16x2[k]);
          fp4x8_to_fp16x2x4(B_fp4x32[k].y, B_fp16x2[k] + 4);
          fp4x8_to_fp16x2x4(B_fp4x32[k].z, B_fp16x2[k] + 8);
          fp4x8_to_fp16x2x4(B_fp4x32[k].w, B_fp16x2[k] + 12);
          SFB_fp16x2[k] = static_cast<half2>(SFB_fp8x2[k]);
        }

        for (int m = 0; m < WARP_M / THREAD_M; m++)
          for (int k = 0; k < BLOCK_K / (THREAD_K * 16); k++) {
            fp4x8_to_fp16x2x4(A_fp4x32[m][k].x, A_fp16x2[m][k]);
            fp4x8_to_fp16x2x4(A_fp4x32[m][k].y, A_fp16x2[m][k] + 4);
            fp4x8_to_fp16x2x4(A_fp4x32[m][k].z, A_fp16x2[m][k] + 8);
            fp4x8_to_fp16x2x4(A_fp4x32[m][k].w, A_fp16x2[m][k] + 12);
            SFA_fp16x2[m][k] = static_cast<half2>(SFA_fp8x2[m][k]);
          }

        // compute
        // for each inner-most iteration, we handle 32 elems along K-dim.
        // this consists of 2 groups (each has 16 elems).
        half2 sub_acc[WARP_M / THREAD_M][BLOCK_K / (THREAD_K * 16)][2];

        for (int m = 0; m < WARP_M / THREAD_M; m++)
          for (int k = 0; k < BLOCK_K / (THREAD_K * 16); k++)
            for (int group_id = 0; group_id < 2; group_id++) {
              // FMA. manually unroll the 1st iteration
              sub_acc[m][k][group_id] = __hmul2(A_fp16x2[m][k][group_id * 8], B_fp16x2[k][group_id * 8]);
              for (int i = 1; i < 8; i++)
                sub_acc[m][k][group_id] = __hfma2(A_fp16x2[m][k][group_id * 8 + i],
                                                  B_fp16x2[k][group_id * 8 + i],
                                                  sub_acc[m][k][group_id]);
            }

        for (int m = 0; m < WARP_M / THREAD_M; m++)
          for (int k = 0; k < BLOCK_K / (THREAD_K * 16); k++) {
            half2 tmp;
            tmp.x = __hadd(sub_acc[m][k][0].x, sub_acc[m][k][0].y);  // 1st group
            tmp.y = __hadd(sub_acc[m][k][1].x, sub_acc[m][k][1].y);  // 2nd groupd

            // scale 2 groups in parallel
            tmp = __hmul2(tmp, SFA_fp16x2[m][k]);
            tmp = __hmul2(tmp, SFB_fp16x2[k]);

            // master accumulation in FP32
            acc[m] += __half2float(tmp.x) + __half2float(tmp.y);
          }

        // signal done to DMA warp
        asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];\n" :: "r"(fma_mbar_addrs[stage_id]));

        prof_stop();
      }

      prof_start(TAG_EPILOGUE);

      // warp reduction
      // do we need an epilogue warp?
      for (int stride = THREAD_K / 2; stride > 0; stride /= 2)
        for (int i = 0; i < WARP_M / THREAD_M; i++)
          acc[i] += __shfl_down_sync(0xFFFF'FFFF, acc[i], stride);

      half *C_gmem = C_ptr + (batch_id * M) + (bid * BLOCK_M) + (warp_id_ * WARP_M) + (lane_id / THREAD_K);

      if (lane_id % THREAD_K == 0) {
        for (int i = 0; i < WARP_M / THREAD_M; i++)
          C_gmem[i * THREAD_M] = __float2half(acc[i]);
      }

      prof_stop();
    }
  }

  if constexpr (DO_PROFILE) if (lane_id == 0)
    profiler.flush();
}

void gemv(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& SFA,
  const at::Tensor& SFB,
        at::Tensor& C,
        at::Tensor& profile_data
) {
  const int M = A.size(0);
  const int K = A.size(1);
  const int L = A.size(2);

  auto A_ptr = reinterpret_cast<const char *>(A.data_ptr());
  auto B_ptr = reinterpret_cast<const char *>(B.data_ptr());
  auto SFA_ptr = reinterpret_cast<const char *>(SFA.data_ptr());
  auto SFB_ptr = reinterpret_cast<const char *>(SFB.data_ptr());
  auto C_ptr = reinterpret_cast<half *>(C.data_ptr());
  auto *profile_ptr = profile_data.data_ptr<int64_t>();

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr bool DO_PROFILE = AA_DO_PROFILE;  // AA_DO_PROFILE is a define

#define launch(NUM_BIDS, BLOCK_M, BLOCK_K, NUM_WARPS, NUM_STAGES) { \
  int TOTAL_SMEM = (BLOCK_M + 1) * (BLOCK_K + BLOCK_K / 8); \
  dim3 grid(L, NUM_BIDS); \
  int smem_size = TOTAL_SMEM * NUM_STAGES; \
  auto this_kernel = kernel<BLOCK_M, BLOCK_K, NUM_WARPS, NUM_STAGES, DO_PROFILE>; \
  if (smem_size > 48'000) \
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
  this_kernel<<<grid, (1 + NUM_WARPS) * WARP_SIZE, smem_size, stream>>>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L, M, K, profile_ptr); \
}

  if (false) {}
  else if (K == 8192) launch(224, 8, 2048, 4, 5)  // benchmark.0
  else if (K == 3584) launch(128, 8, 512, 2, 3)  // benchmark.1
  else if (K == 1024) launch(112, 16, 1024, 4, 2)  // benchmark.2
  else launch(128, 32, 128, 4, 2)                   // the rest

#undef launch
}

TORCH_LIBRARY(my_module, m) {
  m.def("gemv(Tensor A, Tensor B, Tensor SFA, Tensor SFB, Tensor(a!) C, Tensor(b!) profile_data) -> ()");
  m.impl("gemv", &gemv);
}
"""

DO_PROFILE = False
TAGS = [
    "SETUP",
    "LOAD",
    "COMPUTE",
    "WAIT_LOAD",
    "WAIT_COMPUTE",
    "EPILOGUE",
]

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
        f"-DAA_DO_PROFILE={str(DO_PROFILE).lower()}",
        *[f"-DTAG_{tag}={i}" for i, tag in enumerate(TAGS)],
    ],
    extra_ldflags=[
        "-lcuda",
    ],
)


if DO_PROFILE:
    PROFILE_DATA = torch.zeros(10_000, 1 + 1000 * 4, dtype=torch.int64, device="cuda")
else:
    PROFILE_DATA = torch.zeros(1, dtype=torch.int64, device="cuda")


def custom_kernel(data: input_t) -> output_t:
    # a:   [  M, K, L],                   natural shape [L,   M, K]
    # b:   [128, K, L],                   natural shape [L, 128, K] - only the 1st row is used
    # sfa: [32, 4, rest_m, 4, rest_k, L], natural shape [L, rest_m, rest_k, 32, 4, 4]
    # sfb: [32, 4,      1, 4, rest_k, L], natural shape [L,      1, rest_k, 32, 4, 4]
    # c:   [  M, 1, L],                   natural shape [L, M, 1]
    a, b, sfa, sfb, _, _, c_ref = data
    torch.ops.my_module.gemv(a, b, sfa, sfb, c_ref, PROFILE_DATA)

    if DO_PROFILE:
        M, K, L = a.shape
        path = Path(f"profile_data/trace_{M=}_K={K * 2}_{L=}.json.gz")

        if not path.exists():
            PROFILE_DATA.zero_()
            torch.cuda.synchronize()

            torch.ops.my_module.gemv(a, b, sfa, sfb, c_ref, PROFILE_DATA)
            torch.cuda.synchronize()

            events = []

            profile_data = PROFILE_DATA.tolist()
            for bid, data in enumerate(profile_data):
                cnt = data[0]
                if cnt == 0:
                    break

                for i in range(cnt):
                    sm_id, tag, start, duration = data[1 + i * 4 : 1 + (i + 1) * 4]
                    events.append(dict(name=TAGS[tag], ph="X", ts=start, dur=duration, pid=sm_id, tid=sm_id + bid))

            offset = min([evt["ts"] for evt in events])
            for evt in events:
                evt["ts"] -= offset

            path.parent.mkdir(exist_ok=True)
            trace = dict(traceEvents=events)
            gzip.open(path, "w").write(json.dumps(trace).encode("utf-8"))

    if False:
        M, K, L = a.shape
        path = Path(f"profile_data/{M=}_K={K * 2}_{L=}.json.gz")
        if not path.exists():
            a.new_zeros(int(1e8), dtype=torch.uint8)  # 100 MB

            with torch.profiler.profile(with_stack=True) as prof:
                torch.ops.my_module.gemv(a, b, sfa, sfb, c_ref)

            path.parent.mkdir(exist_ok=True)
            prof.export_chrome_trace(str(path))

    return c_ref
