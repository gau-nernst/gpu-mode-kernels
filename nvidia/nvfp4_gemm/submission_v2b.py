#!POPCORN leaderboard nvfp4_gemm
#!POPCORN gpu NVIDIA

import gzip
import json
from pathlib import Path

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <cudaTypedefs.h>
#include <cuda_fp16.h>

#include <torch/library.h>
#include <ATen/core/Tensor.h>

constexpr int WARP_SIZE = 32;
constexpr int NUM_WARPS = 4;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

constexpr int MMA_K = 64;  // 32 bytes

enum ProfilerTag {
  Setup = 0,
  IssueTMA,
  IssueMMA,
  WaitTMA,
  WaitMMA,
  WaitMainloop,
  WaitEpilogue,
  Epilogue,
};

__device__ inline
int64_t globaltimer() {
  int64_t t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t) :: "memory");
  return t;
}

struct Profiler {
  int64_t *data_ptr_;
  int sm_id_;
  int cnt_;

  __device__
  void init(int num_entries, int64_t *data_ptr, int bid) {
    data_ptr_ = data_ptr + bid * (1 + num_entries * 4);
    asm volatile("mov.u32 %0, %smid;\n" : "=r"(sm_id_));
    cnt_ = 0;
  }

  __device__
  void start(ProfilerTag tag) {
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

__device__ inline
constexpr uint64_t desc_encode(uint64_t x) { return (x & 0x3'FFFFULL) >> 4ULL; };

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cute/arch/cluster_sm90.hpp#L180
__device__
uint32_t elect_sync() {
  uint32_t pred = 0;
  asm volatile(
    "{\n\t"
    ".reg .pred %%px;\n\t"
    "elect.sync _|%%px, %1;\n\t"
    "@%%px mov.s32 %0, 1;\n\t"
    "}"
    : "+r"(pred)
    : "r"(0xFFFFFFFF)
  );
  return pred;
}

__device__ inline
void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

// https://github.com/NVIDIA/cutlass/blob/v4.2.1/include/cutlass/arch/barrier.h#L408
__device__
void mbarrier_wait(int mbar_addr, int phase) {
  uint32_t ticks = 0x989680;  // this is optional
  asm volatile(
    "{\n\t"
    ".reg .pred P1;\n\t"
    "LAB_WAIT:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1, %2;\n\t"
    "@P1 bra.uni DONE;\n\t"
    "bra.uni LAB_WAIT;\n\t"
    "DONE:\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks)
  );
}

__device__ inline
void tma_gmem2smem(int dst, const void *src, int size, int mbar_addr, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1], %2, [%3], %4;"
              :: "r"(dst), "l"(src), "r"(size), "r"(mbar_addr), "l"(cache_policy));
}

__device__ inline
void tma_3d_gmem2smem(int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes.cta_group::1.L2::cache_hint "
              "[%0], [%1, {%2, %3, %4}], [%5], %6;"
              :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(mbar_addr), "l"(cache_policy)
              : "memory");
}

__device__ inline
void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
  // .32x128b corresponds to (32, 16) 8-bit scale -> 1 MMA for nvfp4.
  // .warpx4 duplicates data across 32-lane groups.
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

__device__ inline
void tcgen05_mma_nvfp4(
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t i_desc,
  int scale_A_tmem,
  int scale_B_tmem,
  int enable_input_d
) {
  const int d_tmem = 0;  // assume
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"  // predicate register enable-input-d
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\n\t"
    "}"
    :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
       "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d)
  );
}

// 32x32b loads 32x1 tile for each warp
// .x32 -> 32x32 tile
// .x64 -> 32x64 tile
__device__ inline
void tcgen05_ld_32x32bx32(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31])
              : "r"((row << 16) | col));
}

__device__ inline
void tcgen05_ld_32x32bx64(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x64.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31, "
              " %32, %33, %34, %35, %36, %37, %38, %39, "
              " %40, %41, %42, %43, %44, %45, %46, %47, "
              " %48, %49, %50, %51, %52, %53, %54, %55, "
              " %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31]),
                "=f"(tmp[32]), "=f"(tmp[33]), "=f"(tmp[34]), "=f"(tmp[35]), "=f"(tmp[36]), "=f"(tmp[37]), "=f"(tmp[38]), "=f"(tmp[39]),
                "=f"(tmp[40]), "=f"(tmp[41]), "=f"(tmp[42]), "=f"(tmp[43]), "=f"(tmp[44]), "=f"(tmp[45]), "=f"(tmp[46]), "=f"(tmp[47]),
                "=f"(tmp[48]), "=f"(tmp[49]), "=f"(tmp[50]), "=f"(tmp[51]), "=f"(tmp[52]), "=f"(tmp[53]), "=f"(tmp[54]), "=f"(tmp[55]),
                "=f"(tmp[56]), "=f"(tmp[57]), "=f"(tmp[58]), "=f"(tmp[59]), "=f"(tmp[60]), "=f"(tmp[61]), "=f"(tmp[62]), "=f"(tmp[63])
              : "r"((row << 16) | col));
}

// 16x256b loads 16x8 tile for each warp
// .x4  -> 16x32 tile
// .x8  -> 16x64 tile
// .x16 -> 16x128 tile
__device__ inline
void tcgen05_ld_16x256bx4(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x4.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15}, [%16];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15])
              : "r"((row << 16) | col));
}

__device__ inline
void tcgen05_ld_16x256bx8(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x8.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31])
              : "r"((row << 16) | col));
}

__device__ inline
void tcgen05_ld_16x256bx16(float *tmp, int row, int col) {
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x16.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31, "
              " %32, %33, %34, %35, %36, %37, %38, %39, "
              " %40, %41, %42, %43, %44, %45, %46, %47, "
              " %48, %49, %50, %51, %52, %53, %54, %55, "
              " %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31]),
                "=f"(tmp[32]), "=f"(tmp[33]), "=f"(tmp[34]), "=f"(tmp[35]), "=f"(tmp[36]), "=f"(tmp[37]), "=f"(tmp[38]), "=f"(tmp[39]),
                "=f"(tmp[40]), "=f"(tmp[41]), "=f"(tmp[42]), "=f"(tmp[43]), "=f"(tmp[44]), "=f"(tmp[45]), "=f"(tmp[46]), "=f"(tmp[47]),
                "=f"(tmp[48]), "=f"(tmp[49]), "=f"(tmp[50]), "=f"(tmp[51]), "=f"(tmp[52]), "=f"(tmp[53]), "=f"(tmp[54]), "=f"(tmp[55]),
                "=f"(tmp[56]), "=f"(tmp[57]), "=f"(tmp[58]), "=f"(tmp[59]), "=f"(tmp[60]), "=f"(tmp[61]), "=f"(tmp[62]), "=f"(tmp[63])
              : "r"((row << 16) | col));
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int TRANSPOSE, int NUM_STAGES, bool DO_PROFILE>
__global__
__launch_bounds__(TB_SIZE)
void kernel(
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B_tmap,
  const char *SFA_ptr,
  const char *SFB_ptr,
  half *C_ptr,
  int M, int N, int K,
  int64_t *profiler_ptr,
  int num_entries
) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = tid / WARP_SIZE;

  const int grid_m = M / BLOCK_M;
  const int grid_n = N / BLOCK_N;
  const int bid_m = bid / grid_n;
  const int bid_n = bid % grid_n;

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  Profiler profiler;
  if constexpr (DO_PROFILE) if (elect_sync()) {
    profiler.init(num_entries, profiler_ptr, bid * NUM_WARPS + warp_id);
    profiler.start(ProfilerTag::Setup);
  }

  // set up smem
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;
  constexpr int SFA_size = 128 * BLOCK_K / 16;  // always copy 128xBLOCK_K/16
  constexpr int SFB_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size + SFA_size + SFB_size;

  // set up mbarriers and tmem
  // we have NUM_STAGES mbars for TMA
  //         NUM_STAGES mbars for MMA
  //                  1 mbar  for mainloop
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ int64_t mbars[NUM_STAGES * 2 + 1];
  const int tma_mbar_addr = static_cast<int>(__cvta_generic_to_shared(mbars));
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

  // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-mma-scale-factor-a-layout-4x
  // each MMA consumes:
  // - (128, 64) of A -> (128, 4) of SFA -> reshaped as (32, 4', 4) -> 4 tmem columns
  constexpr int SFA_tmem = BLOCK_N;
  constexpr int SFB_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < NUM_STAGES; i++) {
      // when BLOCK_N = 32, we use cp.async.ca + cp.async.mbarrier.arrive.noinc
      // -> expected arrival count is 2 instead of 1.
      mbarrier_init(tma_mbar_addr + i * 8, 1 + (BLOCK_N == 32 ? WARP_SIZE : 0));
      mbarrier_init(mma_mbar_addr + i * 8, 1);
    }
    mbarrier_init(mainloop_mbar_addr, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  else if (warp_id == 1) {
    // allocate tmem
    // tmem address should be 0, don't bother storing and reading it.
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(smem), "r"(BLOCK_N * 2));
  }
  __syncthreads();  // visible to all threads
  if constexpr (DO_PROFILE) if (elect_sync()) profiler.stop();

  // TODO: make K constexpr as well
  const int num_iters = K / BLOCK_K;

  // warp-specialization
  if (warp_id == 0) {
    // TMA warp
    int mma_phase = 1;  // init with 1, since it is initially available.

    // https://github.com/NVIDIA/cutlass/blob/v4.3.2/include/cute/arch/copy_sm90_desc.hpp#L193-L197
    uint64_t evict_first = 0x12F0000000000000;
    uint64_t evict_last = 0x14F0000000000000;

    uint64_t evict_A, evict_B;
    if constexpr (TRANSPOSE) {
      evict_A = evict_first;  // read A once
      evict_B = evict_last;
    } else {
      evict_A = evict_last;
      evict_B = evict_first;  // read B once
    }

    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
      const int stage_id = iter_k % NUM_STAGES;

      // wait MMA
      if constexpr (DO_PROFILE) profiler.start(ProfilerTag::WaitMMA);
      mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_phase);
      if constexpr (DO_PROFILE) profiler.stop();

      if constexpr (DO_PROFILE) profiler.start(ProfilerTag::IssueTMA);

      // we have gone through all stages. flip the phase
      if (stage_id == NUM_STAGES - 1)
        mma_phase ^= 1;

      const int mbar_addr = tma_mbar_addr + stage_id * 8;
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      // issue TMA
      const int off_k = iter_k * BLOCK_K;
      if (elect_sync()) {
        tma_3d_gmem2smem(A_smem, &A_tmap, 0, off_m, off_k / 256, mbar_addr, evict_A);
        tma_3d_gmem2smem(B_smem, &B_tmap, 0, off_n, off_k / 256, mbar_addr, evict_B);
      }

      // layout of SFA is [M/128, rest_k, 32, 4, 4]
      //           SFB is [N/128, rest_k, 32, 4, 4]
      const int rest_k = K / 16 / 4;
      const char *SFA_src = SFA_ptr + ((off_m / 128) * rest_k + off_k / (16 * 4)) * 512;  // 512 = 32x4x4
      const char *SFB_src = SFB_ptr + ((off_n / 128) * rest_k + off_k / (16 * 4)) * 512;
      if (elect_sync())
        tma_gmem2smem(SFA_smem, SFA_src, SFA_size, mbar_addr, evict_A);

      if constexpr (BLOCK_N == 32) {
        // use cp.async.ca so that we can issue <16-byte load
        // we maintain 16-byte alignment for the destination
        for (int k = 0; k < BLOCK_K / MMA_K; k++) {
          int smem_addr = SFB_smem + k * 512 + lane_id * 16;
          const char *gmem_addr = SFB_src + k * 512 + lane_id * 16 + (bid_n % 4) * 4;
          asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;" :: "r"(smem_addr), "l"(gmem_addr));
        }

        asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];" :: "r"(mbar_addr));
      }
      else if (elect_sync()) {
        tma_gmem2smem(SFB_smem, SFB_src, SFB_size, mbar_addr, evict_B);
      }

      // signal TMA done
      constexpr int cp_size = STAGE_SIZE - (BLOCK_N == 32 ? SFB_size : 0);
      if (elect_sync())
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                    :: "r"(mbar_addr), "r"(cp_size) : "memory");
      if constexpr (DO_PROFILE) profiler.stop();
    }
  }
  else if (warp_id == 1 && elect_sync()) {
    // MMA warp
    int tma_phase = 0;

    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    // fp4 MMA doesn't support MMA_M=64. Hence, we will use MMA_M=128 and ignore the rest.
    constexpr uint32_t i_desc = (1U << 7U)   // atype=E2M1
                              | (1U << 10U)  // btype=E2M1
                              | ((uint32_t)BLOCK_N >> 3U << 17U)  // MMA_N
                              | ((uint32_t)128 >> 7U << 27U)  // MMA_M
                              ;

    for (int iter_k = 0; iter_k < num_iters; iter_k++) {
      const int stage_id = iter_k % NUM_STAGES;

      // wait TMA
      if constexpr (DO_PROFILE) profiler.start(ProfilerTag::WaitTMA);
      mbarrier_wait(tma_mbar_addr + stage_id * 8, tma_phase);
      if constexpr (DO_PROFILE) profiler.stop();

      // we have gone through all stages. flip the phase.
      if (stage_id == NUM_STAGES - 1)
        tma_phase ^= 1;

      if constexpr (DO_PROFILE) profiler.start(ProfilerTag::IssueMMA);
      const int A_smem = smem + stage_id * STAGE_SIZE;
      const int B_smem = A_smem + A_size;
      const int SFA_smem = B_smem + B_size;
      const int SFB_smem = SFA_smem + SFA_size;

      // set up shared memory descriptors for A and B
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
      // 128-byte swizzling. LBO is implied to be 1.
      auto make_desc_AB = [](int addr) -> uint64_t {
        const int SBO = 8 * 128;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
      };
      // no swizzling
      auto make_desc_SF = [](int addr) -> uint64_t {
        const int SBO = 8 * 16;
        return desc_encode(addr) | (desc_encode(SBO) << 32ULL) | (1ULL << 46ULL);
      };

      // tcgen05.cp -> tcgen05.mma should be pipelined correctly per PTX doc
      // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions
      // cutlass issues all of smem->tmem BEFORE mma
      // https://github.com/NVIDIA/cutlass/blob/v4.3.2/include/cutlass/gemm/collective/sm100_blockscaled_mma_warpspecialized.hpp#L1013-L1016
      for (int k = 0; k < BLOCK_K / MMA_K; k++) {
        tcgen05_cp_nvfp4(SFA_tmem + k * 4, make_desc_SF(SFA_smem + k * 512));  // 4 columns, 512 bytes of 128x4 / 32x4x4
        tcgen05_cp_nvfp4(SFB_tmem + k * 4, make_desc_SF(SFB_smem + k * 512));
      }

      // k1 selects the (BLOCK_M, 256) tile.
      // k2 selects the (BLOCK_M, 64) tile, whose rows are swizzled.
      // NOTE: this doesn't work with BLOCK_N=32, since apparently tcgen05.mma requires SFB_tmem
      // to have 2-column (8-byte) alignment (looks like not documented).
      for (int k1 = 0; k1 < BLOCK_K / 256; k1++)
        for (int k2 = 0; k2 < 256 / MMA_K; k2++) {
          int k_sf = k1 * 4 + k2;  // 4 is 256 / MMA_K
          tcgen05_mma_nvfp4(
            make_desc_AB(A_smem + k1 * BLOCK_M * 128 + k2 * 32),
            make_desc_AB(B_smem + k1 * BLOCK_N * 128 + k2 * 32),
            i_desc,
            SFA_tmem + k_sf * 4 + (BLOCK_M == 64 ? (bid_m % 2) * 2 : 0),
            SFB_tmem + k_sf * 4 + (BLOCK_N == 64 ? (bid_n % 2) * 2 : 0),
            (k1 == 0 && k2 == 0) ? iter_k : 1
          );
        }

      // signal MMA done
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                  :: "r"(mma_mbar_addr + stage_id * 8) : "memory");
      if constexpr (DO_PROFILE) profiler.stop();
    }

    // signal mainloop done
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                :: "r"(mainloop_mbar_addr) : "memory");
  }
  __syncwarp();

  // wait mainloop
  if constexpr (DO_PROFILE) if (elect_sync()) profiler.start(ProfilerTag::WaitMainloop);
  mbarrier_wait(mainloop_mbar_addr, 0);
  asm volatile("tcgen05.fence::after_thread_sync;");

  if constexpr (DO_PROFILE) if (elect_sync()) {
    profiler.stop();
    profiler.start(ProfilerTag::Epilogue);
  }

  auto epilogue_M_major = [&]() {
    // C is M-major
    if constexpr (BLOCK_N < 128) {
      float tmp[BLOCK_N];
      if constexpr (BLOCK_N == 64) tcgen05_ld_32x32bx64(tmp, warp_id * 32, 0);
      if constexpr (BLOCK_N == 32) tcgen05_ld_32x32bx32(tmp, warp_id * 32, 0);
      asm volatile("tcgen05.wait::ld.sync.aligned;");

      for (int i = 0; i < BLOCK_N; i++)
        C_ptr[(off_n + i) * M + (off_m + tid)] = __float2half(tmp[i]);
    }
    else {
      // we don't use tcgen05_ld_32x32bx128 to save registers
      for (int n = 0; n < BLOCK_N / 64; n++) {
        float tmp[64];
        tcgen05_ld_32x32bx64(tmp, warp_id * 32, n * 64);
        asm volatile("tcgen05.wait::ld.sync.aligned;");

        for (int i = 0; i < 64; i++)
          C_ptr[(off_n + n * 64 + i) * M + (off_m + tid)] = __float2half(tmp[i]);
      }
    }
  };
  auto epilogue_N_major = [&]() {
    // C is N-major
    for (int m = 0; m < 32 / 16; m++) {
      float tmp[BLOCK_N / 2];
      if constexpr (BLOCK_N == 128) tcgen05_ld_16x256bx16(tmp, warp_id * 32 + m * 16, 0);
      if constexpr (BLOCK_N == 64) tcgen05_ld_16x256bx8(tmp, warp_id * 32 + m * 16, 0);
      if constexpr (BLOCK_N == 32) tcgen05_ld_16x256bx4(tmp, warp_id * 32 + m * 16, 0);
      asm volatile("tcgen05.wait::ld.sync.aligned;");

      for (int i = 0; i < BLOCK_N / 8; i++) {
        const int row = off_m + warp_id * 32 + m * 16 + lane_id / 4;
        const int col = off_n + i * 8 + (lane_id % 4) * 2;

        reinterpret_cast<half2 *>(C_ptr + (row + 0) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 0], tmp[i * 4 + 1]});
        reinterpret_cast<half2 *>(C_ptr + (row + 8) * N + col)[0] = __float22half2_rn({tmp[i * 4 + 2], tmp[i * 4 + 3]});
      }
    }
  };

  // when BLOCK_M = 128, use all 4 warps
  //      BLOCK_M = 64, only use the 1st 2 warps (maybe we can do 2-warp threadblock)
  if (BLOCK_M == 128 || warp_id < 2) {
    if constexpr (TRANSPOSE)
      epilogue_M_major();
    else
      epilogue_N_major();
  }

  __syncthreads();  // everyone is done with tmem
  if (warp_id == 0)  // deallocate tmem. tmem address should be 0.
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(BLOCK_N * 2));

  if constexpr (DO_PROFILE) if (elect_sync()) {
    profiler.stop();
    profiler.flush();
  }
}

void check_cu(CUresult err) {
  if (err == CUDA_SUCCESS) return;
  const char *error_msg_ptr;
  if (cuGetErrorString(err, &error_msg_ptr) != CUDA_SUCCESS)
    error_msg_ptr = "unable to get error string";
  TORCH_CHECK(false, "cuTensorMapEncodeTiled error: ", error_msg_ptr);
}

void check_cuda(cudaError_t err) {
  if (err == cudaSuccess) return;
  TORCH_CHECK(false, cudaGetErrorString(err));
}

void init_AB_tmap(
  CUtensorMap *tmap,
  const char *ptr,
  uint64_t global_height, uint64_t global_width,
  uint32_t shared_height, uint32_t shared_width
) {
  constexpr uint32_t rank = 3;
  uint64_t globalDim[rank]       = {256, global_height, global_width / 256};
  uint64_t globalStrides[rank-1] = {global_width / 2, 128};  // in bytes
  uint32_t boxDim[rank]          = {256, shared_height, shared_width / 256};
  uint32_t elementStrides[rank]  = {1, 1, 1};

  auto err = cuTensorMapEncodeTiled(
    tmap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B,
    rank,
    (void *)ptr,
    globalDim,
    globalStrides,
    boxDim,
    elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );
  check_cu(err);
}

template <int BLOCK_M, int BLOCK_N, int BLOCK_K, int TRANSPOSE, int NUM_STAGES, bool DO_PROFILE>
void gemm_launch(
  const char *A_ptr,
  const char *B_ptr,
  const char *SFA_ptr,
  const char *SFB_ptr,
        half *C_ptr,
  int M, int N, int K,
  int64_t *profiler_ptr,
  int num_entries
) {
  static_assert(BLOCK_K % 256 == 0);  // 128 bytes
  CUtensorMap A_tmap, B_tmap;

  // swap A/B and M/N if TRANSPOSE is enabled
  if constexpr (TRANSPOSE) {
    std::swap(A_ptr, B_ptr);
    std::swap(SFA_ptr, SFB_ptr);
    std::swap(M, N);
  }

  // TODO: create tensormap once and cache it. replace address with cuTensorMapReplaceAddress()
  init_AB_tmap(&A_tmap, A_ptr, M, K, BLOCK_M, BLOCK_K);
  init_AB_tmap(&B_tmap, B_ptr, N, K, BLOCK_N, BLOCK_K);

  int grid = (M / BLOCK_M) * (N / BLOCK_N);
  int AB_size = (BLOCK_M + BLOCK_N) * (BLOCK_K / 2);
  int SFAB_size = 128 * (BLOCK_K / 16) * 2;
  int smem_size = (AB_size + SFAB_size) * NUM_STAGES;

  auto this_kernel = kernel<BLOCK_M, BLOCK_N, BLOCK_K, TRANSPOSE, NUM_STAGES, DO_PROFILE>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  this_kernel<<<grid, TB_SIZE, smem_size>>>(A_tmap, B_tmap, SFA_ptr, SFB_ptr, C_ptr, M, N, K, profiler_ptr, num_entries);
}

at::Tensor gemm(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& SFA,
  const at::Tensor& SFB,
        at::Tensor& C
) {
  const int M = A.size(0);
  const int N = B.size(0);
  const int K = A.size(1) * 2;

  auto A_ptr   = reinterpret_cast<const char *>(A.data_ptr());
  auto B_ptr   = reinterpret_cast<const char *>(B.data_ptr());
  auto SFA_ptr = reinterpret_cast<const char *>(SFA.data_ptr());
  auto SFB_ptr = reinterpret_cast<const char *>(SFB.data_ptr());
  auto C_ptr   = reinterpret_cast<half *>(C.data_ptr());

  gemm_launch<128, 32, 256, true, 6, false>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, M, N, K, nullptr, 0);
  return C;
}

at::Tensor profile(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& SFA,
  const at::Tensor& SFB,
        at::Tensor& C,
        at::Tensor& profiler,
  int64_t num_entries
) {
  const int M = A.size(0);
  const int N = B.size(0);
  const int K = A.size(1) * 2;

  auto A_ptr   = reinterpret_cast<const char *>(A.data_ptr());
  auto B_ptr   = reinterpret_cast<const char *>(B.data_ptr());
  auto SFA_ptr = reinterpret_cast<const char *>(SFA.data_ptr());
  auto SFB_ptr = reinterpret_cast<const char *>(SFB.data_ptr());
  auto C_ptr   = reinterpret_cast<half *>(C.data_ptr());
  auto profiler_ptr = profiler.data_ptr<int64_t>();

  gemm_launch<128, 32, 256, true, 6, true>(A_ptr, B_ptr, SFA_ptr, SFB_ptr, C_ptr, M, N, K, profiler_ptr, num_entries);
  return C;
}

TORCH_LIBRARY(my_module, m) {
  m.def("gemm(Tensor A, Tensor B, Tensor SFA, Tensor SFB, Tensor(a!) C) -> Tensor");
  m.impl("gemm", &gemm);

  m.def("profile(Tensor A, Tensor B, Tensor SFA, Tensor SFB, Tensor(a!) C, Tensor(b!) profiler, int num_entries) -> Tensor");
  m.impl("profile", &profile);
}
"""

DO_PROFILE = False
# this must match what's defined in profiler.h
TAGS = [
    "SETUP",
    "ISSUE_TMA",
    "ISSUE_MMA",
    "WAIT_TMA",
    "WAIT_MMA",
    "WAIT_MAINLOOP",
    "WAIT_EPILOGUE",
    "EPILOGUE",
]

load_inline(
    "gemm",
    cpp_sources="",
    cuda_sources=CUDA_SRC,
    verbose=True,
    is_python_module=False,
    no_implicit_headers=True,
    extra_cuda_cflags=[
        "-O3",
        "-gencode=arch=compute_100a,code=sm_100a",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--relocatable-device-code=false",
        "-lineinfo",
        "-Xptxas=-v",
        # "--keep",
        # "--keep-dir",
        # f"{Path(__file__).parent}/tmp",
    ],
    extra_ldflags=["-lcuda"],
)
gemm = torch.ops.my_module.gemm
profile = torch.ops.my_module.profile

if DO_PROFILE:
    NUM_BLOCKS = 1000
    NUM_ENTRIES = 1000
else:
    NUM_BLOCKS = 1
    NUM_ENTRIES = 1
PROFILE_DATA = torch.zeros((NUM_BLOCKS, 1 + NUM_ENTRIES * 4), dtype=torch.int64, device="cuda")


def custom_kernel(data: input_t) -> output_t:
    # a:   [M, K, 1],                     natural shape [1, M, K]
    # b:   [N, K, 1],                     natural shape [1, N, K] - only the 1st row is used
    # sfa: [32, 4, M/128, 4, rest_k, 1],  natural shape [1, M/128, rest_k, 32, 4, 4], where rest_k = K/16/4
    # sfb: [32, 4, N/128, 4, rest_k, 1],  natural shape [1, N/128, rest_k, 32, 4, 4]
    # c:   [M, N, 1],                     natural shape [1, M, N]
    if not DO_PROFILE:
        return gemm(data[0], data[1], data[4], data[5], data[6])

    # warmup
    M = data[0].shape[0]
    N = data[1].shape[0]
    K = data[0].shape[1] * 2

    profile(data[0], data[1], data[4], data[5], data[6], PROFILE_DATA, NUM_ENTRIES)

    save_path = Path(f"profile_data/{M}_{N}_{K}.json.gz")
    if not save_path.exists():
        PROFILE_DATA.zero_()
        profile(data[0], data[1], data[4], data[5], data[6], PROFILE_DATA, NUM_ENTRIES)
        torch.cuda.synchronize()

        events = []
        for bid, entry in enumerate(PROFILE_DATA.tolist()):
            for i in range(entry[0]):
                sm_id, tag, start, duration = entry[1 + i * 4 : 1 + (i + 1) * 4]
                events.append(dict(name=TAGS[tag], ph="X", ts=start, dur=duration, pid=sm_id, tid=sm_id + bid))

        offset = min([evt["ts"] for evt in events])
        for evt in events:
            evt["ts"] -= offset

        trace = dict(traceEvents=events)
        save_path.parent.mkdir(exist_ok=True)
        gzip.open(save_path, "w").write(json.dumps(trace).encode("utf-8"))

    return data[6]
