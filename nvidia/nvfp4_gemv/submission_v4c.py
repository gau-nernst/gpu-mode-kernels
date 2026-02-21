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
constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

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
uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{\n"
    ".reg .pred %%px;\n"
    "elect.sync _|%%px, %1;\n"
    "@%%px mov.s32 %0, 1;\n"
    "}\n"
    : "+r"(pred)
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
  asm volatile(
    "{\n"
    ".reg .pred P1;\n"
    "LAB_WAIT:\n"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n"
    "@P1     bra.uni DONE;\n"
    "bra.uni LAB_WAIT;\n"
    "DONE:\n"
    "}\n"
    :: "r"(mbar_addr), "r"(phase)
  );
}

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
template <int BLOCK_M, int BLOCK_K, int NUM_STAGES, bool DO_PROFILE>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE)
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
  constexpr int SF_BLOCK_K = BLOCK_K / 8;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int num_bids = gridDim.x;
  const int batch_id = blockIdx.y;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  Profiler profiler;
  if constexpr (DO_PROFILE) if (lane_id == 0)
    profiler.init(profiler_ptr, (batch_id * num_bids + bid) * NUM_WARPS + warp_id);

  auto prof_start = [&](int tag) {
    if constexpr (DO_PROFILE) if (lane_id == 0)
      profiler.start(tag);
  };
  auto prof_stop = [&]() {
    if constexpr (DO_PROFILE) if (lane_id == 0)
      profiler.stop();
  };

  prof_start(TAG_SETUP);

  const int off_m = bid * BLOCK_M;
  A_ptr += (batch_id * M * K) + off_m * K;
  B_ptr += (batch_id * 128 * K);
  C_ptr += (batch_id * M) + off_m;
  SFA_ptr += (batch_id * M * (K / 8)) + off_m * (K / 8);
  SFB_ptr += (batch_id * 128 * (K / 8));

  int64_t evict_first;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0;\n" : "=l"(evict_first));

  // set up smem
  extern __shared__ char smem[];
  const int smem_u32 = static_cast<int>(__cvta_generic_to_shared(smem));

  constexpr int A_size = BLOCK_M * BLOCK_K;
  constexpr int B_size = BLOCK_K;
  constexpr int SFA_size = BLOCK_M * SF_BLOCK_K;
  constexpr int SFB_size = SF_BLOCK_K;
  constexpr int TOTAL_SIZE = A_size + B_size + SFA_size + SFB_size;

  constexpr int num_cols = BLOCK_K / 16;  // each thread reads 16-byte at a time
  constexpr int TB_WIDTH = std::min(num_cols, TB_SIZE);
  //constexpr int TB_WIDTH = std::min(num_cols, WARP_SIZE);  // no reduction via smem required
  constexpr int TB_HEIGHT = TB_SIZE / TB_WIDTH;

  // set up rmem
  // for smem->rmem
  int4 A_rmem[BLOCK_M / TB_HEIGHT][num_cols / TB_WIDTH];
  int4 B_rmem[num_cols / TB_WIDTH];
  char2 SFA_rmem[BLOCK_M / TB_HEIGHT][num_cols / TB_WIDTH];
  char2 SFB_rmem[num_cols / TB_WIDTH];

  // for unpacking to fp16x2
  half2 A_fp16x2[BLOCK_M / TB_HEIGHT][num_cols / TB_WIDTH][16];
  half2 B_fp16x2[num_cols / TB_WIDTH][16];
  half2 SFA_fp16x2[BLOCK_M / TB_HEIGHT][num_cols / TB_WIDTH];
  half2 SFB_fp16x2[num_cols / TB_WIDTH];

  // for accumulation
  half2 acc[BLOCK_M / TB_HEIGHT][num_cols / TB_WIDTH][2];
  float master_acc[BLOCK_M / TB_HEIGHT] = {};

  // set up mbarrier
  int phase = 0;

  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES];
  int mbar_addrs[NUM_STAGES];
  for (int i = 0; i < NUM_STAGES; i++)
    mbar_addrs[i] = static_cast<int>(__cvta_generic_to_shared(mbars + i));

  if (warp_id == 0 && elect_sync()) {
    // only 1 thread issues TMA, hence we set expected arrival count = 1 for TMA mbarrier
    for (int i = 0; i < NUM_STAGES; i++)
      asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mbar_addrs[i]), "r"(1));

    // initialized mbarrier visible to TMA engine (async proxy)
    asm volatile("fence.mbarrier_init.release.cluster;\n");
  }
  __syncthreads();  // initialized mbarrier visible to all threads.
  prof_stop();

  auto load = [&](int iter_k) {
    if (warp_id == 0 && elect_sync()) {
      const int stage_id = iter_k % NUM_STAGES;
      const int mbar_addr = mbar_addrs[stage_id];

      int A_buf = smem_u32 + stage_id * TOTAL_SIZE;
      int B_buf = A_buf + A_size;
      int SFA_buf = B_buf + B_size;
      int SFB_buf = SFA_buf + SFA_size;

      for (int row = 0; row < BLOCK_M; row++) {
        cp_async_bulk_policy(A_buf + row * BLOCK_K, A_ptr + row * K, BLOCK_K, mbar_addr, evict_first);
        cp_async_bulk_policy(SFA_buf + row * SF_BLOCK_K, SFA_ptr + row * (K / 8), SF_BLOCK_K, mbar_addr, evict_first);
      }
      cp_async_bulk(B_buf, B_ptr, BLOCK_K, mbar_addr);
      cp_async_bulk(SFB_buf, SFB_ptr, SF_BLOCK_K, mbar_addr);

      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                  :: "r"(mbar_addr), "r"(TOTAL_SIZE) : "memory");

      A_ptr += BLOCK_K;
      B_ptr += BLOCK_K;
      SFA_ptr += SF_BLOCK_K;
      SFB_ptr += SF_BLOCK_K;
    }
  };

  auto compute = [&](int iter_k) {
    const int stage_id = iter_k % NUM_STAGES;

    prof_start(TAG_WAIT_LOAD);
    mbarrier_wait(mbar_addrs[stage_id], phase);
    prof_stop();

    // flip the phase once we have cycled through all stages
    if (stage_id == NUM_STAGES - 1)
      phase ^= 1;

    prof_start(TAG_COMPUTE);

    // smem -> rmem
    char *A_smem = smem + stage_id * TOTAL_SIZE;
    char *B_smem = A_smem + A_size;
    char *SFA_smem = B_smem + B_size;
    char *SFB_smem = SFA_smem + SFA_size;

    for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++)
      for (int k = 0; k < num_cols / TB_WIDTH; k++) {
        const int row = m * TB_HEIGHT + (tid / TB_WIDTH);
        const int col = k * TB_WIDTH + (tid % TB_WIDTH);
        A_rmem[m][k] = reinterpret_cast<const int4 *>(A_smem + row * BLOCK_K + (col * 16))[0];
        SFA_rmem[m][k] = reinterpret_cast<const char2 *>(SFA_smem + row * SF_BLOCK_K + (col * 2))[0];
      }

    for (int k = 0; k < num_cols / TB_WIDTH; k++) {
      const int col = k * TB_WIDTH + (tid % TB_WIDTH);
      B_rmem[k] = reinterpret_cast<const int4 *>(B_smem + col * 16)[0];
      SFB_rmem[k] = reinterpret_cast<const char2 *>(SFB_smem + col * 2)[0];
    }

    // unpack

    for (int k = 0; k < num_cols / TB_WIDTH; k++) {
      fp4x8_to_fp16x2x4(B_rmem[k].x, B_fp16x2[k]);
      fp4x8_to_fp16x2x4(B_rmem[k].y, B_fp16x2[k] + 4);
      fp4x8_to_fp16x2x4(B_rmem[k].z, B_fp16x2[k] + 8);
      fp4x8_to_fp16x2x4(B_rmem[k].w, B_fp16x2[k] + 12);
      SFB_fp16x2[k] = static_cast<half2>(reinterpret_cast<__nv_fp8x2_e4m3 *>(SFB_rmem)[k]);
    }

    for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++)
      for (int k = 0; k < num_cols / TB_WIDTH; k++) {
        fp4x8_to_fp16x2x4(A_rmem[m][k].x, A_fp16x2[m][k]);
        fp4x8_to_fp16x2x4(A_rmem[m][k].y, A_fp16x2[m][k] + 4);
        fp4x8_to_fp16x2x4(A_rmem[m][k].z, A_fp16x2[m][k] + 8);
        fp4x8_to_fp16x2x4(A_rmem[m][k].w, A_fp16x2[m][k] + 12);
        SFA_fp16x2[m][k] = static_cast<half2>(reinterpret_cast<__nv_fp8x2_e4m3 *>(SFA_rmem[m])[k]);
        SFA_fp16x2[m][k] = __hmul2(SFA_fp16x2[m][k], SFB_fp16x2[k]);
      }

    // compute
    for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++)
      for (int k = 0; k < num_cols / TB_WIDTH; k++) {
        acc[m][k][0] = __hmul2(A_fp16x2[m][k][0], B_fp16x2[k][0]);  // 1st group
        acc[m][k][1] = __hmul2(A_fp16x2[m][k][8], B_fp16x2[k][8]);  // 2nd group

        for (int i = 1; i < 8; i++) {
          acc[m][k][0] = __hfma2(A_fp16x2[m][k][0 + i], B_fp16x2[k][0 + i], acc[m][k][0]);  // 1st group
          acc[m][k][1] = __hfma2(A_fp16x2[m][k][8 + i], B_fp16x2[k][8 + i], acc[m][k][1]);  // 2nd group
        }
      }

    for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++)
      for (int k = 0; k < num_cols / TB_WIDTH; k++) {
        __half2_raw scales = SFA_fp16x2[m][k];
        __half_raw group0 = __hadd(acc[m][k][0].x, acc[m][k][0].y);
        __half_raw group1 = __hadd(acc[m][k][1].x, acc[m][k][1].y);
        asm volatile("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(master_acc[m]) : "h"(group0.x), "h"(scales.x));
        asm volatile("fma.rn.f32.f16 %0, %1, %2, %0;" : "+f"(master_acc[m]) : "h"(group1.x), "h"(scales.y));
      }

    prof_stop();
  };

  for (int iter_k = 0; iter_k < NUM_STAGES - 1; iter_k++)
    load(iter_k);

  const int num_iters = K / BLOCK_K;
  for (int iter_k = 0; iter_k < num_iters - (NUM_STAGES - 1); iter_k++) {
    load(iter_k + NUM_STAGES - 1);  // gmem -> smem
    compute(iter_k);
    __syncthreads();  // make sure finish using the buffer for the next prefetch
  }

  for (int k = 0; k < NUM_STAGES - 1; k++)
    compute(num_iters - (NUM_STAGES - 1) + k);

  prof_start(TAG_EPILOGUE);

  if constexpr (TB_WIDTH > WARP_SIZE) {
    // red_smem[BLOCK_M / TB_HEIGHT][TB_SIZE]
    float *red_smem = reinterpret_cast<float *>(smem);
    __syncthreads();  // make sure previous compute finish before writing to smem

    for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++)
      red_smem[m * TB_SIZE + tid] = master_acc[m];
    __syncthreads();

    for (int stride = TB_WIDTH / 2; stride >= WARP_SIZE; stride /= 2) {
      if ((tid % TB_WIDTH) < stride)
        for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++) {
          master_acc[m] += red_smem[m * TB_SIZE + (tid + stride)];
          red_smem[m *TB_SIZE + tid] = master_acc[m];
        }
      __syncthreads();
    }
  }

  constexpr int start_stride = std::min(TB_WIDTH, WARP_SIZE) / 2;
  for (int stride = start_stride; stride > 0; stride /= 2) {
    for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++)
      master_acc[m] += __shfl_down_sync(0xFFFF'FFFF, master_acc[m], stride);
  }

  if (tid % TB_WIDTH == 0) {
    for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++) {
      const int row = m * TB_HEIGHT + (tid / TB_WIDTH);
      C_ptr[row] = __float2half(master_acc[m]);
    }
  }

  prof_stop();
  if constexpr (DO_PROFILE) if (tid == 0)
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

  constexpr bool DO_PROFILE = AA_DO_PROFILE;  // AA_DO_PROFILE is a define

#define launch(BLOCK_M, BLOCK_K, NUM_STAGES) { \
  int TOTAL_SMEM = (BLOCK_M + 1) * (BLOCK_K + BLOCK_K / 8); \
  dim3 grid(M / BLOCK_M, L); \
  int smem_size = TOTAL_SMEM * NUM_STAGES; \
  auto this_kernel = kernel<BLOCK_M, BLOCK_K, NUM_STAGES, DO_PROFILE>; \
  if (smem_size > 48'000) \
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
  this_kernel<<<grid, TB_SIZE, smem_size>>>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L, M, K, profile_ptr); \
}

  if (false) {}
  else if (K == 8192) launch(8, 1024, 2)  // benchmark.0
  else if (K == 3584) launch(8,  512, 2)  // benchmark.1
  else if (K == 1024) launch(8,  512, 2)  // benchmark.2
  else launch(32, 128, 2)                 // the rest

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
    "WAIT_LOAD",
    "COMPUTE",
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
