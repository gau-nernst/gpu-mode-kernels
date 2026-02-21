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
void fp8x2_to_fp16x2(int *out, int16_t in) {
  asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(out[0]) : "h"(in));
}

__device__
void ldcs_i16(int16_t *dst, const void *src) {
//#define PTX_MOD ".cs"
//#define PTX_MOD ".L1::no_allocate"
//#define PTX_MOD ".cs.nc"
#define PTX_MOD ".nc.L1::no_allocate"
  asm volatile("ld.global" PTX_MOD ".b16 %0, [%1];" : "=h"(dst[0]) : "l"(src));
#undef PTX_MOD
}

__device__
void ldca_i16(int16_t *dst, const void *src) {
//#define PTX_MOD ".ca"
//#define PTX_MOD ".L1::evict_last"
//#define PTX_MOD ".ca.nc"
#define PTX_MOD ".nc.L1::evict_last"
  asm volatile("ld.global" PTX_MOD ".b16 %0, [%1];" : "=h"(dst[0]) : "l"(src));
#undef PTX_MOD
}

__device__
void ldcs_i32x4(int *dst, const void *src) {
//#define PTX_MOD ".cs"
//#define PTX_MOD ".L1::no_allocate"
//#define PTX_MOD ".cs.nc"
#define PTX_MOD ".nc.L1::no_allocate"
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
#define PTX_MOD ".nc.L1::evict_last"
  asm volatile("ld.global" PTX_MOD ".v4.b32 {%0, %1, %2, %3}, [%4];"
              : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
              : "l"(src));
#undef PTX_MOD
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
    data_ptr_ = data_ptr + bid * (1 + NUM_ENTRIES * 4);
    asm volatile("mov.u32 %0, %smid;" : "=r"(sm_id_));
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
template <int BLOCK_M, int BLOCK_K, int TB_WIDTH, int NUM_WARPS, bool DO_PROFILE>
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
  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  constexpr int SF_BLOCK_K = BLOCK_K / 8;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int batch_id = blockIdx.y;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  int off_m = bid * BLOCK_M;
  A_ptr += (batch_id * M * K) + off_m * K;
  B_ptr += (batch_id * 128 * K);
  C_ptr += (batch_id * M) + off_m;
  SFA_ptr += (batch_id *   M * (K / 8)) + off_m * (K / 8);
  SFB_ptr += (batch_id * 128 * (K / 8));

  constexpr int num_cols = BLOCK_K / 16;  // each thread reads 16-byte at a time
  constexpr int TB_HEIGHT = TB_SIZE / TB_WIDTH;

  // for gmem->rmem
  int A_rmem[BLOCK_M / TB_HEIGHT][num_cols / TB_WIDTH][4];
  int B_rmem[num_cols / TB_WIDTH][4];
  int16_t SFA_rmem[BLOCK_M / TB_HEIGHT][num_cols / TB_WIDTH];
  int16_t SFB_rmem[num_cols / TB_WIDTH];

  // for unpacking to fp16x2
  half2 A_fp16x2[BLOCK_M / TB_HEIGHT][num_cols / TB_WIDTH][16];
  half2 B_fp16x2[num_cols / TB_WIDTH][16];
  half2 SFA_fp16x2[BLOCK_M / TB_HEIGHT][num_cols / TB_WIDTH];
  half2 SFB_fp16x2[num_cols / TB_WIDTH];

  // for accumulation
  half2 acc[BLOCK_M / TB_HEIGHT][num_cols / TB_WIDTH][2];
  float master_acc[BLOCK_M / TB_HEIGHT] = {};

  auto gmem_to_rmem = [&]() {
    for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++)
      for (int k = 0; k < num_cols / TB_WIDTH; k++) {
        const int row = m * TB_HEIGHT + (tid / TB_WIDTH);
        const int col = k * TB_WIDTH + (tid % TB_WIDTH);
        ldcs_i32x4(A_rmem[m][k], A_ptr + row * K + (col * 16));
        ldcs_i16(SFA_rmem[m] + k, SFA_ptr + row * (K / 8) + (col * 2));
      }

    for (int k = 0; k < num_cols / TB_WIDTH; k++) {
      const int col = k * TB_WIDTH + (tid % TB_WIDTH);
      ldca_i32x4(B_rmem[k], B_ptr + (col * 16));
      ldca_i16(SFB_rmem + k, SFB_ptr + (col * 2));
    }
  };

  auto unpack = [&]() {
    for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++)
      for (int k = 0; k < num_cols / TB_WIDTH; k++) {
        for (int i = 0; i < 4; i++)
          fp4x8_to_fp16x2x4(reinterpret_cast<int *>(A_fp16x2[m][k] + i * 4), A_rmem[m][k][i]);
        fp8x2_to_fp16x2(reinterpret_cast<int *>(&SFA_fp16x2[m][k]), SFA_rmem[m][k]);
        //SFA_fp16x2[m][k] = static_cast<half2>(reinterpret_cast<__nv_fp8x2_e4m3 *>(SFA_rmem[m])[k]);
      }

    for (int k = 0; k < num_cols / TB_WIDTH; k++) {
      for (int i = 0; i < 4; i++)
        fp4x8_to_fp16x2x4(reinterpret_cast<int *>(B_fp16x2[k] + i * 4), B_rmem[k][i]);
      fp8x2_to_fp16x2(reinterpret_cast<int *>(&SFB_fp16x2[k]), SFB_rmem[k]);
      //SFB_fp16x2[k] = static_cast<half2>(reinterpret_cast<__nv_fp8x2_e4m3 *>(SFB_rmem)[k]);
    }
  };

  auto compute = [&]() {
    for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++)
      for (int k = 0; k < num_cols / TB_WIDTH; k++)
        SFA_fp16x2[m][k] = __hmul2(SFA_fp16x2[m][k], SFB_fp16x2[k]);

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
  };

  const int num_iters = K / BLOCK_K;
  for (int iter_k = 0; iter_k < num_iters; iter_k++) {
    asm volatile("//start of main loop");
    gmem_to_rmem();
    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K;
    SFA_ptr += SF_BLOCK_K;
    SFB_ptr += SF_BLOCK_K;
    unpack();
    compute();
  }

  auto final_epilogue = [&]() {
    constexpr int start_stride = std::min(TB_WIDTH, WARP_SIZE) / 2;
    for (int stride = start_stride; stride > 0; stride /= 2) {
      for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++) {
        float tmp = __shfl_down_sync(0xFFFF'FFFF, master_acc[m], stride);
        master_acc[m] += tmp;
      }
    }

    if (tid % TB_WIDTH == 0) {
      for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++) {
        const int row = m * TB_HEIGHT + (tid / TB_WIDTH);
        C_ptr[row] = __float2half(master_acc[m]);
      }
    }
  };

  // benchmark.0
  // don't think this is faster in a meaningful way, but just for the lolz.
  if constexpr (TB_WIDTH == WARP_SIZE * 2) {
    __shared__ float smem[BLOCK_M / TB_HEIGHT][NUM_WARPS / 2][WARP_SIZE];

    if (warp_id % 2 == 1)
      for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++)
        smem[m][warp_id / 2][lane_id] = master_acc[m];
    __syncthreads();

    if (warp_id % 2 == 0) {
      for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++)
        master_acc[m] += smem[m][warp_id / 2][lane_id];
      final_epilogue();
    }
  }
  else {
    if constexpr (TB_WIDTH > WARP_SIZE) {
      __shared__ float smem[BLOCK_M / TB_HEIGHT][TB_SIZE];

      for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++)
        smem[m][tid] = master_acc[m];
      __syncthreads();

      for (int stride = TB_WIDTH / 2; stride >= WARP_SIZE; stride /= 2) {
        if ((tid % TB_WIDTH) < stride) {
          for (int m = 0; m < BLOCK_M / TB_HEIGHT; m++) {
            float tmp = smem[m][tid + stride];
            master_acc[m] += tmp;
            smem[m][tid] = master_acc[m];
          }
        }
        __syncthreads();
      }
    }

    final_epilogue();
  }
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

#define launch(BLOCK_M, BLOCK_K, TB_WIDTH, NUM_WARPS) { \
  dim3 grid(M / BLOCK_M, L); \
  auto this_kernel = kernel<BLOCK_M, BLOCK_K, TB_WIDTH, NUM_WARPS, DO_PROFILE>; \
  this_kernel<<<grid, NUM_WARPS * WARP_SIZE>>>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L, M, K, profile_ptr); \
}

  if (false) {}
  else if (K == 8192) launch(8, 1024, 64, 4)   // benchmark.0
  else if (K == 3584) launch(8, 512, 32, 4)    // benchmark.1
  else if (K == 1024) launch(8, 512, 32, 4)    // benchmark.2
  else launch(32, 128, 8, 4)                  // the rest

#undef launch
}

TORCH_LIBRARY(my_module, m) {
  m.def("gemv(Tensor A, Tensor B, Tensor SFA, Tensor SFB, Tensor(a!) C, Tensor(b!) profiler) -> ()");
  m.impl("gemv", &gemv);
}
"""

DO_PROFILE = False
NUM_ENTRIES = 1000
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
        # "--keep",
        # "--keep-dir",
        # f"{Path(__file__).parent}/tmp",
        f"-DAA_DO_PROFILE={str(DO_PROFILE).lower()}",
        f"-DNUM_ENTRIES={NUM_ENTRIES}",
        *[f"-DTAG_{tag}={i}" for i, tag in enumerate(TAGS)],
    ],
)
gemv = torch.ops.my_module.gemv

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
    gemv(a, b, sfa, sfb, c_ref, PROFILE_DATA)

    if DO_PROFILE:
        M, K, L = a.shape
        path = Path(f"profile_data/trace_{M=}_K={K * 2}_{L=}.json.gz")

        if not path.exists():
            PROFILE_DATA.zero_()
            torch.cuda.synchronize()

            gemv(a, b, sfa, sfb, c_ref, PROFILE_DATA)
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

            with torch.profiler.profile() as prof:
                gemv(a, b, sfa, sfb, c_ref, PROFILE_DATA)

            path.parent.mkdir(exist_ok=True)
            prof.export_chrome_trace(str(path))

    return c_ref
