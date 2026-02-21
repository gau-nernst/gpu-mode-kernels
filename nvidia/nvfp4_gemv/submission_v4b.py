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
void cp_async_bulk_tensor_3d(int dst, const void *tensor_map, int x, int y, int z, int mbar_addr) {
  asm volatile(
    "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
    "[%0], "                // dstMem
    "[%1, {%2, %3, %4}], "  // tensorMap and tensorCoords
    "[%5];\n"               // mbar
    :: "r"(dst), "l"(tensor_map), "r"(x), "r"(y), "r"(z), "r"(mbar_addr)
    : "memory"
  );
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
template <int THREAD_M, int THREAD_K, int NUM_STAGES = 1, bool DO_PROFILE = false>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE)
void kernel(
  const __grid_constant__ CUtensorMap   A_tensor_map,
  const __grid_constant__ CUtensorMap SFA_tensor_map,
  const char   *B_ptr,  // [L, 128, K]
  const char *SFB_ptr,  // [L, 128, K/8]
        half   *C_ptr,  // [L,   M]
  int L, int M, int K,
  int64_t *profiler_ptr
) {
  // to ensure coalesced access, we need at least 8 threads per row (16B x 8 = 128B)
  // each thread reads 16B, which covers 2 scaled groups. hence, we only need within
  // thread reduction during the main loop.
  static_assert(THREAD_M % 4 == 0);
  static_assert(THREAD_K >= 8);
  static_assert(THREAD_K <= TB_SIZE);
  constexpr int BLOCK_M = (TB_SIZE / THREAD_K) * THREAD_M;
  constexpr int BLOCK_K = THREAD_K * 16;
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

  const int off_m = bid * BLOCK_M;
  const int off_k = (tid % THREAD_K) * 16;  // each thread reads 16 fp4x2 values at a time

    B_ptr += (batch_id * 128 * K);
  SFB_ptr += (batch_id * 128 * (K / 8));

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

  // to be used for smem->rmem load
  char   *A_smem_ld =   A_smem + (tid / THREAD_K) * THREAD_M *    BLOCK_K + off_k;
  char   *B_smem_ld =   B_smem                                            + off_k;
  char *SFA_smem_ld = SFA_smem + (tid / THREAD_K) * THREAD_M * SF_BLOCK_K + (off_k / 8);
  char *SFB_smem_ld = SFB_smem +                                          + (off_k / 8);

  const int   A_smem_u32 = static_cast<int>(__cvta_generic_to_shared(  A_smem));
  const int   B_smem_u32 = static_cast<int>(__cvta_generic_to_shared(  B_smem));
  const int SFA_smem_u32 = static_cast<int>(__cvta_generic_to_shared(SFA_smem));
  const int SFB_smem_u32 = static_cast<int>(__cvta_generic_to_shared(SFB_smem));

  // set up mbarrier
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES];
  int mbar_addrs[NUM_STAGES];
  for (int i = 0; i < NUM_STAGES; i++)
    mbar_addrs[i] = static_cast<int>(__cvta_generic_to_shared(mbars + i));

  // doesn't seem to matter much
  //if (warp_id == 0 && elect_one_sync()) {
  //  asm volatile("prefetch.tensormap [%0];\n" :: "l"(  &A_tensor_map) : "memory");
  //  asm volatile("prefetch.tensormap [%0];\n" :: "l"(&SFA_tensor_map) : "memory");
  //}
  if (warp_id == 1 && elect_one_sync()) {
    // only 1 thread issues TMA, hence we set expected arrival count = 1 for TMA mbarrier
    for (int i = 0; i < NUM_STAGES; i++)
      asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mbar_addrs[i]), "r"(1));

    // initialized mbarrier visible to TMA engine (async proxy)
    asm volatile("fence.mbarrier_init.release.cluster;\n");
  }
  __syncthreads();  // initialized mbarrier visible to all threads.
  prof_stop();

  float acc[THREAD_M] = {};

  int phase = 0;

  auto load = [&](int iter_k) {
    // one thread issue TMA
    if (warp_id == 0 && elect_one_sync()) {
      const int stage_id = iter_k % NUM_STAGES;
      const int mbar_addr = mbar_addrs[stage_id];
      const int off_k = iter_k * BLOCK_K;

      // divide by 8 because we use uint64 for tensormap
      cp_async_bulk_tensor_3d(  A_smem_u32 + stage_id *   A_size,   &A_tensor_map, off_k /  8, off_m, batch_id, mbar_addr);
      cp_async_bulk_tensor_3d(SFA_smem_u32 + stage_id * SFA_size, &SFA_tensor_map, off_k / 64, off_m, batch_id, mbar_addr);

      cp_async_bulk(  B_smem_u32 + stage_id *   B_size,   B_ptr,    BLOCK_K, mbar_addr);
      cp_async_bulk(SFB_smem_u32 + stage_id * SFB_size, SFB_ptr, SF_BLOCK_K, mbar_addr);

      // increment tx-count (expect-tx) by number of bytes, and decrement arrival count (arrive-on) by 1.
      // release semantics w/ CTA scope.
      constexpr int cp_size = A_size + SFA_size + B_size + SFB_size;
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                  :: "r"(mbar_addr), "r"(cp_size) : "memory");

      B_ptr += BLOCK_K;
      SFB_ptr += SF_BLOCK_K;
    }
  };

  auto compute = [&](int iter_k) {
    const int stage_id = iter_k % NUM_STAGES;

    prof_start(TAG_WAIT_LOAD);
    // basically a spin lock. acquire semantics
    mbarrier_wait(mbar_addrs[stage_id], phase);
    prof_stop();

    // flip the phase once we have cycled through all stages
    if (stage_id == NUM_STAGES - 1)
      phase ^= 1;

    prof_start(TAG_COMPUTE);

    // smem -> rmem
    int4 A_fp4x8x4[THREAD_M], B_fp4x8x4;
    half2 SFA_fp16x2[THREAD_M], SFB_fp16x2;

    for (int m = 0; m < THREAD_M; m++) {
      A_fp4x8x4[m] = reinterpret_cast<const int4 *>(A_smem_ld + stage_id * A_size + m * BLOCK_K)[0];
      SFA_fp16x2[m] = static_cast<half2>(reinterpret_cast<const __nv_fp8x2_e4m3 *>(SFA_smem_ld + stage_id * SFA_size + m * SF_BLOCK_K)[0]);
    }

    B_fp4x8x4 = reinterpret_cast<const int4 *>(B_smem_ld + stage_id * B_size)[0];
    SFB_fp16x2 = static_cast<half2>(reinterpret_cast<const __nv_fp8x2_e4m3 *>(SFB_smem_ld + stage_id * SFB_size)[0]);

    // unpack to FP16
    half2 A_fp16x2[THREAD_M][16], B_fp16x2[16];

    for (int m = 0; m < THREAD_M; m++) {
      fp4x8_to_fp16x2x4(A_fp4x8x4[m].x, A_fp16x2[m]);
      fp4x8_to_fp16x2x4(A_fp4x8x4[m].y, A_fp16x2[m] + 4);
      fp4x8_to_fp16x2x4(A_fp4x8x4[m].z, A_fp16x2[m] + 8);
      fp4x8_to_fp16x2x4(A_fp4x8x4[m].w, A_fp16x2[m] + 12);
    }

    fp4x8_to_fp16x2x4(B_fp4x8x4.x, B_fp16x2);
    fp4x8_to_fp16x2x4(B_fp4x8x4.y, B_fp16x2 + 4);
    fp4x8_to_fp16x2x4(B_fp4x8x4.z, B_fp16x2 + 8);
    fp4x8_to_fp16x2x4(B_fp4x8x4.w, B_fp16x2 + 12);

    for (int m = 0; m < THREAD_M; m++) {
      half2 sub_acc[2];

      // compute everything in FP16
      for (int group_id = 0; group_id < 2; group_id++) {
        // FMA. manually unroll the 1st iteration
        sub_acc[group_id] = __hmul2(A_fp16x2[m][group_id * 8], B_fp16x2[group_id * 8]);
        for (int i = 1; i < 8; i++)
          sub_acc[group_id] = __hfma2(A_fp16x2[m][group_id * 8 + i],
                                      B_fp16x2[group_id * 8 + i],
                                      sub_acc[group_id]);
      }

      half2 tmp;
      tmp.x = __hadd(sub_acc[0].x, sub_acc[0].y);  // 1st group
      tmp.y = __hadd(sub_acc[1].x, sub_acc[1].y);  // 2nd group

      // scaling 2 groups in parallel
      tmp = __hmul2(tmp, SFA_fp16x2[m]);
      tmp = __hmul2(tmp, SFB_fp16x2);

      // only master accumulation in FP32
      acc[m] += __half2float(tmp.x) + __half2float(tmp.y);
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

  int64_t acc_fp32x2[THREAD_M / 2];
  std::memcpy(acc_fp32x2, acc, THREAD_M * sizeof(float));

  // threadblock reduction
  if constexpr (THREAD_K > WARP_SIZE) {
    // reuse dynamic smem
    // using layout float red_smem[THREAD_M / 4][TB_SIZE][4]
    // to avoid bank conflicts when doing 16-byte loads/stores
    float *red_smem = reinterpret_cast<float *>(smem);

    // 16-byte store
    for (int i = 0; i < THREAD_M / 4; i++)
      reinterpret_cast<float4 *>(red_smem)[i * TB_SIZE + tid] = reinterpret_cast<float4 *>(acc_fp32x2)[i];
    __syncthreads();

    for (int stride = THREAD_K / 2; stride >= WARP_SIZE; stride /= 2) {
      if ((tid % THREAD_K) < stride) {
        for (int i = 0; i < THREAD_M / 4; i++) {
          int64_t tmp[2];

          // 16-byte load
          reinterpret_cast<float4 *>(tmp)[0] = reinterpret_cast<float4 *>(red_smem)[i * TB_SIZE + (tid + stride)];

          // f32x2 math
          asm volatile("add.rn.f32x2 %0, %0, %1;\n" : "+l"(acc_fp32x2[i * 2 + 0]) : "l"(tmp[0]));
          asm volatile("add.rn.f32x2 %0, %0, %1;\n" : "+l"(acc_fp32x2[i * 2 + 1]) : "l"(tmp[1]));

          // 16-byte store
          reinterpret_cast<float4 *>(red_smem)[i * TB_SIZE + tid] = reinterpret_cast<float4 *>(acc_fp32x2)[i];
        }
      }
      __syncthreads();
    }
  }

  // warp reduction
  constexpr int start_stride = std::min(THREAD_K, WARP_SIZE) / 2;
  for (int stride = start_stride; stride > 0; stride /= 2)
    for (int i = 0; i < THREAD_M / 2; i++) {
      int64_t tmp = __shfl_down_sync(0xFFFF'FFFF, acc_fp32x2[i], stride);
      asm volatile("add.rn.f32x2 %0, %0, %1;\n" : "+l"(acc_fp32x2[i]) : "l"(tmp));
    }

  if (tid % THREAD_K == 0) {
    half2 out[THREAD_M / 2];

    for (int i = 0; i < THREAD_M / 2; i++)
      out[i] = __float22half2_rn(reinterpret_cast<float2 *>(&acc_fp32x2[i])[0]);

    half *out_ptr = C_ptr + (batch_id * M) + off_m + (tid / THREAD_K) * THREAD_M;

    if constexpr (THREAD_M == 4) {
      // 8-byte store
      reinterpret_cast<int2 *>(out_ptr)[0] = reinterpret_cast<int2 *>(out)[0];
    }
    else {
      // 16-byte store. only when THREAD_M = 8, this is coalesced.
      for (int i = 0; i < THREAD_M / 8; i++)
        reinterpret_cast<int4 *>(out_ptr)[i] = reinterpret_cast<int4 *>(out)[i];
    }
  }

  prof_stop();
  if constexpr (DO_PROFILE) if (tid == 0)
    profiler.flush();
}

void init_tensor_map(
  CUtensorMap *tensor_map,
  const char *gmem_ptr,
  uint64_t L, uint64_t GMEM_HEIGHT, uint64_t GMEM_WIDTH,
  uint32_t SMEM_HEIGHT, uint32_t SMEM_WIDTH
) {
  // this doesn't work if any of the smem size > 256.
  // hence, we treat this as uint64, then divide width by 8.
  // NOTE: GMEM/SMEM_WIDTH is for fp4x2 (8bit).
  constexpr uint32_t rank = 3;
  uint64_t size[rank] = {GMEM_WIDTH / 8, GMEM_HEIGHT, L};
  uint64_t stride[rank - 1] = {GMEM_WIDTH, GMEM_WIDTH * GMEM_HEIGHT};  // stride in bytes
  uint32_t box_size[rank] = {SMEM_WIDTH / 8, SMEM_HEIGHT, 1};
  uint32_t elem_stride[rank] = {1, 1, 1};

  auto res = cuTensorMapEncodeTiled(
    tensor_map,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_UINT64,
    rank,
    (void *)gmem_ptr,
    size,
    stride,
    box_size,
    elem_stride,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (res != CUDA_SUCCESS) {
    const char *error_msg_ptr;
    if (cuGetErrorString(res, &error_msg_ptr) != CUDA_SUCCESS)
      error_msg_ptr = "unable to get error string";
    TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
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

  auto stream = at::cuda::getCurrentCUDAStream();
  constexpr bool DO_PROFILE = AA_DO_PROFILE;  // AA_DO_PROFILE is a define

  // TODO: cache tensormap
  CUtensorMap A_tensor_map, SFA_tensor_map;

  if (false) {
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++)
      init_tensor_map(&A_tensor_map, A_ptr, L, M, K, 16, 512);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start) / 100;
    std::cout << "Elapsed time: " << duration.count() << " ns\n";
  }

#define launch(THREAD_M, THREAD_K, NUM_STAGES) { \
  int BLOCK_M = (TB_SIZE / THREAD_K) * THREAD_M; \
  int BLOCK_K = THREAD_K * 16; \
  int SF_BLOCK_K = BLOCK_K / 8; \
  int TOTAL_SMEM = (BLOCK_M + 1) * (BLOCK_K + SF_BLOCK_K); \
  dim3 grid(M / BLOCK_M, L); \
  init_tensor_map(  &A_tensor_map,   A_ptr, L, M,     K, BLOCK_M,    BLOCK_K); \
  init_tensor_map(&SFA_tensor_map, SFA_ptr, L, M, K / 8, BLOCK_M, SF_BLOCK_K); \
  int smem_size = TOTAL_SMEM * NUM_STAGES; \
  auto this_kernel = kernel<THREAD_M, THREAD_K, NUM_STAGES, DO_PROFILE>; \
  if (smem_size > 48'000) \
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
  this_kernel<<<grid, TB_SIZE, smem_size, stream>>>(A_tensor_map, SFA_tensor_map, B_ptr, SFB_ptr, C_ptr, L, M, K, profile_ptr); \
}

  if (false) {}
  else if (K == 8192) launch(4, 128, 2)  // benchmark.0
  else if (K == 3584) launch(4,  32, 2)  // benchmark.1
  else if (K == 1024) launch(4,  32, 2)  // benchmark.2
  else launch(4, 8, 2)                   // the rest

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
