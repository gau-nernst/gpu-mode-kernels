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
  // for each iteration,
  // - load: we load A[BLOCK_M][BLOCK_K] and B[BLOCK_K] from gmem to smem.
  // - compute:
  //   - each warp is responsible A[WARP_M][BLOCK_K] and B[BLOCK_K], where WARP_M = BLOCK_M / NUM_WARPS.
  //   - this is to avoid inter-warp reduction via smem.

  static_assert(BLOCK_K % 16 == 0);  // each thread reads 16 bytes
  static_assert(BLOCK_M % NUM_WARPS == 0);
  constexpr int WARP_M = BLOCK_M / NUM_WARPS;

  // we organize threads in a warp as [THREAD_M][THREAD_K]
  // hence, each thread holds A[WARP_M / THREAD_M][BLOCK_K / THREAD_K] and B[BLOCK_K / THREAD_K]
  constexpr int THREAD_K = std::min(BLOCK_K / 16, WARP_SIZE);  // how many threads in a warp are required to read a row of BLOCK_K
  constexpr int THREAD_M = WARP_SIZE / THREAD_K;
  static_assert(WARP_M % THREAD_M == 0);

  constexpr int SF_BLOCK_K = BLOCK_K / 8;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int batch_id = blockIdx.y;

  Profiler profiler;
  if constexpr (DO_PROFILE) if (tid == 0)
    profiler.init(profiler_ptr, batch_id * gridDim.x + bid);

  auto prof_start = [&](int tag) {
    if constexpr (DO_PROFILE) if (tid == 0)
      profiler.start(tag);
  };
  auto prof_stop = [&]() {
    if constexpr (DO_PROFILE) if (tid == 0)
      profiler.stop();
  };

  prof_start(TAG_SETUP);

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  // select output tile
  {
    int off_m = bid * BLOCK_M;

    A_ptr += (batch_id * M * K) + (off_m * K);
    B_ptr += (batch_id * 128 * K);
    C_ptr += (batch_id * M) + (off_m);

    SFA_ptr += (batch_id *   M * (K / 8)) + (off_m * (K / 8));
    SFB_ptr += (batch_id * 128 * (K / 8));
  }

  // set up smem
  extern __shared__ char smem[];
  const int smem_u32 = static_cast<int>(__cvta_generic_to_shared(smem));

  constexpr int A_size = BLOCK_M * BLOCK_K;
  constexpr int B_size = BLOCK_K;
  constexpr int SFA_size = BLOCK_M * SF_BLOCK_K;
  constexpr int SFB_size = SF_BLOCK_K;
  constexpr int TOTAL_SMEM = A_size + B_size + SFA_size + SFB_size;

  char *A_smem = smem;
  char *B_smem = A_smem + A_size;
  char *SFA_smem = B_smem + B_size;
  char *SFB_smem = SFA_smem + SFA_size;

  // add offsets for smem->rmem load
  {
    int off_m = (warp_id * WARP_M) + (lane_id / THREAD_K);
    int off_k = (lane_id % THREAD_K) * 16;  // each thread reads 16 bytes

    A_smem += off_m * BLOCK_K + off_k;
    B_smem += off_k;
    SFA_smem += off_m * SF_BLOCK_K + (off_k / 8);
    SFB_smem += (off_k / 8);
  }

  prof_stop();

  float acc[WARP_M / THREAD_M] = {};

  auto load = [&](int iter_k) {
    prof_start(TAG_LOAD);

    // NOTE: since B, SFA, and SFB does not require the whole threadblock to load, we can partition it within the threadblock.
    const int buffer = smem_u32 + (iter_k % NUM_STAGES) * TOTAL_SMEM;
    const int A_buf   = buffer;
    const int B_buf   = A_buf + A_size;
    const int SFA_buf = B_buf + B_size;
    const int SFB_buf = SFA_buf + SFA_size;

    cp_async_2d<BLOCK_M,    BLOCK_K, TB_SIZE>(  A_buf,   A_ptr,     K, tid);
    cp_async_2d<      1,    BLOCK_K, TB_SIZE>(  B_buf,   B_ptr,     K, tid);
    cp_async_2d<BLOCK_M, SF_BLOCK_K, TB_SIZE>(SFA_buf, SFA_ptr, K / 8, tid);
    cp_async_2d<      1, SF_BLOCK_K, TB_SIZE>(SFB_buf, SFB_ptr, K / 8, tid);

    asm volatile("cp.async.commit_group;\n");

    A_ptr += BLOCK_K;
    B_ptr += BLOCK_K;
    SFA_ptr += SF_BLOCK_K;
    SFB_ptr += SF_BLOCK_K;

    prof_stop();
  };

  auto compute = [&](int iter_k) {
    prof_start(TAG_COMPUTE);

    int buf_offset = (iter_k % NUM_STAGES) * TOTAL_SMEM;

    // smem -> rmem
    int4 A_fp4x32[WARP_M / THREAD_M][BLOCK_K / (THREAD_K * 16)];
    int4 B_fp4x32[BLOCK_K / (THREAD_K * 16)];
    __nv_fp8x2_e4m3 SFA_fp8x2[WARP_M / THREAD_M][BLOCK_K / (THREAD_K * 16)];
    __nv_fp8x2_e4m3 SFB_fp8x2[BLOCK_K / (THREAD_K * 16)];

    for (int k = 0; k < BLOCK_K / (THREAD_K * 16); k++) {
      int col = k * THREAD_K * 16;
      B_fp4x32[k] = reinterpret_cast<int4 *>(B_smem + buf_offset + col)[0];
      SFB_fp8x2[k] = reinterpret_cast<__nv_fp8x2_e4m3 *>(SFB_smem + buf_offset + col / 8)[0];
    }

    for (int m = 0; m < WARP_M / THREAD_M; m++)
      for (int k = 0; k < BLOCK_K / (THREAD_K * 16); k++) {
        int row = m * THREAD_M;
        int col = k * THREAD_K * 16;
        A_fp4x32[m][k] = reinterpret_cast<int4 *>(A_smem + buf_offset + row * BLOCK_K + col)[0];
        SFA_fp8x2[m][k] = reinterpret_cast<__nv_fp8x2_e4m3 *>(SFA_smem + buf_offset + row * SF_BLOCK_K + col / 8)[0];
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

    prof_stop();
  };

  for (int iter_k = 0; iter_k < NUM_STAGES - 1; iter_k++)
    load(iter_k);

  const int num_iters = K / BLOCK_K;
  for (int iter_k = 0; iter_k < num_iters - (NUM_STAGES - 1); iter_k++) {
    // gmem -> smem
    load(iter_k + NUM_STAGES - 1);

    prof_start(TAG_WAIT_LOAD);
    asm volatile("cp.async.wait_group %0;\n" :: "n"(NUM_STAGES - 1));
    __syncthreads();  // memory barrier
    prof_stop();

    compute(iter_k);
    __syncthreads();  // make sure finish using the buffer for the next prefetch
  }

  prof_start(TAG_WAIT_LOAD);
  asm volatile("cp.async.wait_all;\n");
  __syncthreads();  // memory barrier
  prof_stop();

  for (int k = 0; k < NUM_STAGES - 1; k++)
    compute(num_iters - (NUM_STAGES - 1) + k);

  prof_start(TAG_EPILOGUE);

  static_assert((WARP_M / THREAD_M) % 2 == 0);
  int64_t acc_fp32x2[(WARP_M / THREAD_M) / 2];
  std::memcpy(acc_fp32x2, acc, sizeof(acc_fp32x2));

  // warp reduction
  for (int stride = THREAD_K / 2; stride > 0; stride /= 2)
    for (int i = 0; i < (WARP_M / THREAD_M) / 2; i++) {
      int64_t tmp = __shfl_down_sync(0xFFFF'FFFF, acc_fp32x2[i], stride);
      asm volatile("add.rn.f32x2 %0, %0, %1;\n" : "+l"(acc_fp32x2[i]) : "l"(tmp));
    }

  // write to smem first to get coalesced write to global
  half *smem_half = reinterpret_cast<half *>(smem);
  if (lane_id % THREAD_K == 0) {
    half *smem_ptr = smem_half + (warp_id * WARP_M) + (lane_id / THREAD_K);

    for (int i = 0; i < (WARP_M / THREAD_M) / 2; i++) {
      half2 tmp = __float22half2_rn(reinterpret_cast<float2 *>(&acc_fp32x2[i])[0]);
      smem_ptr[i * (THREAD_M * 2) +        0] = tmp.x;
      smem_ptr[i * (THREAD_M * 2) + THREAD_M] = tmp.y;
    }
  }
  __syncthreads();

  if (tid * 2 < BLOCK_M) {
    int tmp = reinterpret_cast<int *>(smem_half)[tid];
    reinterpret_cast<int *>(C_ptr)[tid] = tmp;
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

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr bool DO_PROFILE = AA_DO_PROFILE;  // AA_DO_PROFILE is a define

#define launch(BLOCK_M, BLOCK_K, NUM_STAGES) { \
  int TOTAL_SMEM = (BLOCK_M + 1) * (BLOCK_K + BLOCK_K / 8); \
  dim3 grid(M / BLOCK_M, L); \
  int smem_size = TOTAL_SMEM * NUM_STAGES; \
  auto this_kernel = kernel<BLOCK_M, BLOCK_K, NUM_STAGES, DO_PROFILE>; \
  if (smem_size > 48'000) \
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
  this_kernel<<<grid, TB_SIZE, smem_size, stream>>>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, L, M, K, profile_ptr); \
}

  if (false) {}
  else if (K == 8192) launch(8, 1024, 2)   // benchmark.0
  else if (K == 3584) launch(8, 512, 2)    // benchmark.1
  else if (K == 1024) launch(16, 1024, 2)  // benchmark.2
  else launch(32, 128, 2)                  // the rest

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
        f"-DAA_DO_PROFILE={str(DO_PROFILE).lower()}",
        f"-DNUM_ENTRIES={NUM_ENTRIES}",
        *[f"-DTAG_{tag}={i}" for i, tag in enumerate(TAGS)],
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

            with torch.profiler.profile() as prof:
                torch.ops.my_module.gemv(a, b, sfa, sfb, c_ref)

            path.parent.mkdir(exist_ok=True)
            prof.export_chrome_trace(str(path))

    return c_ref
