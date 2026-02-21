#!POPCORN leaderboard nvfp4_gemv

from pathlib import Path

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cudaTypedefs.h>

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/CUDAContext.h>

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

__device__
void fp4x8_to_fp16x2x4(int in, int *out) {
  asm volatile(
    "{\n"
    ".reg .b8 tmp0, tmp1, tmp2, tmp3;\n"
    "mov.b32 {tmp0, tmp1, tmp2, tmp3}, %4; // unpack 32-bit register to 4x fp4x2\n"
    "cvt.rn.f16x2.e2m1x2 %0, tmp0; // PTX only supports FP4->FP16\n"
    "cvt.rn.f16x2.e2m1x2 %1, tmp1;\n"
    "cvt.rn.f16x2.e2m1x2 %2, tmp2;\n"
    "cvt.rn.f16x2.e2m1x2 %3, tmp3;\n"
    "}\n"
    : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
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

// trick to get 128-byte aligned shared memory
typedef struct __align__(128) {} Aligned128B;

// to make our calculations simple, let's treat fp4x2 as a unit.
// hence, K = number of fp4x2 elements, and 8 elements share
// the same scale.
template <int THREAD_M, int THREAD_K, int NUM_STAGES = 1>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE)
void kernel(
  const __grid_constant__ CUtensorMap   A_tensor_map,
  const __grid_constant__ CUtensorMap SFA_tensor_map,
  const char   *B_ptr,  // [L, 128, K]
  const char *SFB_ptr,  // [L, 128, K/8]
        half   *C_ptr,  // [L,   M]
  int L, int M, int K
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
  // speed is much slower if mbar is a variable instead of an array lmao
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[1];
  int mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));

  if (warp_id == 0 && elect_one_sync()) {
    // doesn't seem to matter much
    asm volatile("prefetch.tensormap [%0];\n" :: "l"(  &A_tensor_map) : "memory");
    asm volatile("prefetch.tensormap [%0];\n" :: "l"(&SFA_tensor_map) : "memory");
  }
  if (warp_id == 1 && elect_one_sync()) {
    // init mbarrier with expected arrival count = 1
    asm volatile("mbarrier.init.shared.b64 [%0], %1;\n" :: "r"(mbar_addr), "r"(1));

    // initialized mbarrier visible to TMA engine (async proxy)
    asm volatile("fence.mbarrier_init.release.cluster;\n");
  }
  __syncthreads();  // initialized mbarrier visible to all threads.

  float acc[THREAD_M] = {};

  auto load = [&](int iter_k) {
    // NOTE: since B, SFA, and SFB does not require the whole threadblock to load, we can partition it within the threadblock.
    const int stage_id = iter_k % NUM_STAGES;

    cp_async_2d<1,    BLOCK_K, TB_SIZE>(  B_smem_u32 + stage_id *   B_size,   B_ptr,     K, tid);
    cp_async_2d<1, SF_BLOCK_K, TB_SIZE>(SFB_smem_u32 + stage_id * SFB_size, SFB_ptr, K / 8, tid);

    asm volatile("cp.async.commit_group;\n");

    B_ptr += BLOCK_K;
    SFB_ptr += SF_BLOCK_K;
  };

  int phase = 0;

  auto compute = [&](int iter_k) {
    const int stage_id = iter_k % NUM_STAGES;

    // one thread issue TMA
    if (warp_id == 0 && elect_one_sync()) {
      // divide by 8 because we use uint64 for tensormap
      const int off_k = iter_k * BLOCK_K;
      cp_async_bulk_tensor_3d(  A_smem_u32 + stage_id *   A_size,   &A_tensor_map, off_k /  8, off_m, batch_id, mbar_addr);
      cp_async_bulk_tensor_3d(SFA_smem_u32 + stage_id * SFA_size, &SFA_tensor_map, off_k / 64, off_m, batch_id, mbar_addr);

      // increment tx-count (expect-tx) by number of bytes, and decrement arrival count (arrive-on) by 1.
      // release semantics w/ CTA scope.
      constexpr int cp_size = A_size + SFA_size;
      asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;\n"
                  :: "r"(mbar_addr), "r"(cp_size) : "memory");
    }

    // basically a spin lock. acquire semantics
    mbarrier_wait(mbar_addr, phase);

    // flip the phase
    phase ^= 1;

    // smem -> rmem
    int A_fp4x8[THREAD_M][4], B_fp4x8[4];
    half2 SFA_fp16x2[THREAD_M], SFB_fp16x2;

    for (int m = 0; m < THREAD_M; m++) {
      reinterpret_cast<int4 *>(A_fp4x8[m])[0] = reinterpret_cast<const int4 *>(A_smem_ld + stage_id * A_size + m * BLOCK_K)[0];
      SFA_fp16x2[m] = static_cast<half2>(reinterpret_cast<const __nv_fp8x2_e4m3 *>(SFA_smem_ld + stage_id * SFA_size + m * SF_BLOCK_K)[0]);
    }

    reinterpret_cast<int4 *>(B_fp4x8)[0] = reinterpret_cast<const int4 *>(B_smem_ld + stage_id * B_size)[0];
    SFB_fp16x2 = static_cast<half2>(reinterpret_cast<const __nv_fp8x2_e4m3 *>(SFB_smem_ld + stage_id * SFB_size)[0]);

    // unpack to FP16
    int A_fp16x2[THREAD_M][16], B_fp16x2[16];

    for (int m = 0; m < THREAD_M; m++)
      for (int i = 0; i < 4; i++)
        fp4x8_to_fp16x2x4(A_fp4x8[m][i], A_fp16x2[m] + i * 4);

    for (int i = 0; i < 4; i++)
      fp4x8_to_fp16x2x4(B_fp4x8[i], B_fp16x2 + i * 4);

    for (int m = 0; m < THREAD_M; m++) {
      int sub_acc[2];

      // compute everything in FP16
      for (int group_id = 0; group_id < 2; group_id++) {
        // FMA. manually unroll the 1st iteration
        asm volatile("mul.rn.f16x2 %0, %1, %2;\n"
                    : "=r"(sub_acc[group_id])
                    : "r"(A_fp16x2[m][group_id * 8]), "r"(B_fp16x2[group_id * 8]));
        for (int i = 1; i < 8; i++)
          asm volatile("fma.rn.f16x2 %0, %1, %2, %0;\n"
                      : "+r"(sub_acc[group_id])
                      : "r"(A_fp16x2[m][group_id * 8 + i]), "r"(B_fp16x2[group_id * 8 + i]));
      }

      half2 tmp[2];
      std::memcpy(tmp, sub_acc, sizeof(sub_acc));

      half2 tmptmp;
      tmptmp.x = __hadd(tmp[0].x, tmp[0].y);  // 1st group
      tmptmp.y = __hadd(tmp[1].x, tmp[1].y);  // 2nd group

      // scaling 2 groups in parallel
      tmptmp = __hmul2(tmptmp, SFA_fp16x2[m]);
      tmptmp = __hmul2(tmptmp, SFB_fp16x2);

      // only master accumulation in FP32
      acc[m] += __half2float(tmptmp.x) + __half2float(tmptmp.y);
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

  // TODO: cache tensormap

#define launch(THREAD_M, THREAD_K, NUM_STAGES) { \
  int BLOCK_M = (TB_SIZE / THREAD_K) * THREAD_M; \
  int BLOCK_K = THREAD_K * 16; \
  int SF_BLOCK_K = BLOCK_K / 8; \
  int TOTAL_SMEM = (BLOCK_M + 1) * (BLOCK_K + SF_BLOCK_K); \
  dim3 grid(M / BLOCK_M, L); \
  CUtensorMap A_tensor_map, SFA_tensor_map; \
  init_tensor_map(  &A_tensor_map,   A_ptr, L, M,     K, BLOCK_M,    BLOCK_K); \
  init_tensor_map(&SFA_tensor_map, SFA_ptr, L, M, K / 8, BLOCK_M, SF_BLOCK_K); \
  int smem_size = TOTAL_SMEM * NUM_STAGES; \
  auto this_kernel = kernel<THREAD_M, THREAD_K, NUM_STAGES>; \
  if (smem_size > 48'000) \
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size); \
  this_kernel<<<grid, TB_SIZE, smem_size, stream>>>(A_tensor_map, SFA_tensor_map, B_ptr, SFB_ptr, C_ptr, L, M, K); \
}

  if (false) {}
  else if (K == 8192) launch(4, 128, 2)  // benchmark.0
  else if (K == 3584) launch(4,  32, 2)  // benchmark.1
  else if (K == 1024) launch(4,  32, 2)  // benchmark.2
  else launch(4, 8, 2)                   // the rest

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
    extra_ldflags=[
        "-lcuda",
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

            with torch.profiler.profile(with_stack=True) as prof:
                torch.ops.my_module.gemv(a, b, sfa, sfb, c_ref)

            path.parent.mkdir(exist_ok=True)
            prof.export_chrome_trace(str(path))

    return c_ref
