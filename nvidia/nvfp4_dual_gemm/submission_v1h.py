#!POPCORN leaderboard nvfp4_dual_gemm
#!POPCORN gpu NVIDIA

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <cudaTypedefs.h>
#include <cuda_fp16.h>

#include <torch/library.h>
#include <ATen/core/Tensor.h>

constexpr int WARP_SIZE = 32;

constexpr int MMA_K = 64;  // 32 bytes

// https://github.com/NVIDIA/cutlass/blob/v4.3.2/include/cute/arch/copy_sm90_desc.hpp#L193-L197
constexpr uint64_t EVICT_NORMAL = 0x1000000000000000;
constexpr uint64_t EVICT_FIRST = 0x12F0000000000000;
constexpr uint64_t EVICT_LAST = 0x14F0000000000000;

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
    "@!P1 bra.uni LAB_WAIT;\n\t"
    "}"
    :: "r"(mbar_addr), "r"(phase), "r"(ticks)
  );
}

__device__ inline
void tma_prefetch(const void *src, int size, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
              :: "l"(src), "r"(size), "l"(cache_policy) : "memory");
}

__device__ inline
void tma_3d_prefetch(const void *tmap_ptr, int x, int y, int z, uint64_t cache_policy) {
  asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.L2::cache_hint [%0, {%1, %2, %3}], %4;"
              :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "l"(cache_policy) : "memory");
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
void tma_2d_smem2gmem(const void *tmap_ptr, int x, int y, int src) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
              :: "l"(tmap_ptr), "r"(x), "r"(y), "r"(src) : "memory");
}

__device__ inline
void tcgen05_cp_nvfp4(int taddr, uint64_t s_desc) {
  // .32x128b corresponds to (32, 16) 8-bit scale -> 1 MMA for nvfp4.
  // .warpx4 duplicates data across 32-lane groups.
  asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(taddr), "l"(s_desc));
}

struct COLLECTOR_USAGE {
  static constexpr char NONE[]      = "";
  static constexpr char A_FILL[]    = ".collector::a::fill";
  static constexpr char A_USE[]     = ".collector::a::use";
  static constexpr char A_LASTUSE[] = ".collector::a::lastuse";
  static constexpr char A_DISCARD[] = ".collector::a::discard";
};

template <const char *collector_usage = COLLECTOR_USAGE::NONE>
__device__ inline
void tcgen05_mma_nvfp4(
  int d_tmem,
  uint64_t a_desc,
  uint64_t b_desc,
  uint32_t i_desc,
  int scale_A_tmem,
  int scale_B_tmem,
  int enable_input_d
) {
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"  // predicate register enable-input-d
    "setp.ne.b32 p, %6, 0;\n\t"
    "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16%7 [%0], %1, %2, %3, [%4], [%5], p;\n\t"
    "}"
    :: "r"(d_tmem), "l"(a_desc), "l"(b_desc), "r"(i_desc),
       "r"(scale_A_tmem), "r"(scale_B_tmem), "r"(enable_input_d),
       "C"(collector_usage)
  );
}

// see https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
struct SHAPE {
  static constexpr char _32x32b[]  = ".32x32b";   // 32x1 tile for each warp
  static constexpr char _16x128b[] = ".16x128b";  // 16x4 tile
  static constexpr char _16x256b[] = ".16x256b";  // 16x8 tile
};

template <int NUM_REGS, const char *SHAPE, int NUM>
__device__ inline
void tcgen05_ld(float *tmp, int row, int col) {
  int addr = (row << 16) | col;

  if constexpr (NUM_REGS == 1)
  asm volatile("tcgen05.ld.sync.aligned%3.x%4.b32 {%0}, [%1];"
              : "=f"(tmp[0]) : "r"(addr), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 2)
  asm volatile("tcgen05.ld.sync.aligned%3.x%4.b32 {%0, %1}, [%2];"
              : "=f"(tmp[0]), "=f"(tmp[1]) : "r"(addr), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 4)
  asm volatile("tcgen05.ld.sync.aligned%5.x%6.b32 "
              "{%0, %1, %2, %3}, [%4];"
              : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3])
              : "r"(addr), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 8)
  asm volatile("tcgen05.ld.sync.aligned%9.x%10.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7}, [%8];"
              : "=f"(tmp[0]), "=f"(tmp[1]), "=f"(tmp[2]), "=f"(tmp[3]), "=f"(tmp[4]), "=f"(tmp[5]), "=f"(tmp[6]), "=f"(tmp[7])
              : "r"(addr), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 16)
  asm volatile("tcgen05.ld.sync.aligned%17.x%18.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15}, [%16];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15])
              : "r"(addr), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 32)
  asm volatile("tcgen05.ld.sync.aligned%33.x%34.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31])
              : "r"(addr), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 64)
  asm volatile("tcgen05.ld.sync.aligned%65.x%66.b32 "
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
              : "r"(addr), "C"(SHAPE), "n"(NUM));
  if constexpr (NUM_REGS == 128)
  asm volatile("tcgen05.ld.sync.aligned%129.x%130.b32 "
              "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7, "
              "  %8,  %9, %10, %11, %12, %13, %14, %15, "
              " %16, %17, %18, %19, %20, %21, %22, %23, "
              " %24, %25, %26, %27, %28, %29, %30, %31, "
              " %32, %33, %34, %35, %36, %37, %38, %39, "
              " %40, %41, %42, %43, %44, %45, %46, %47, "
              " %48, %49, %50, %51, %52, %53, %54, %55, "
              " %56, %57, %58, %59, %60, %61, %62, %63, "
              " %64, %65, %66, %67, %68, %69, %70, %71, "
              " %72, %73, %74, %75, %76, %77, %78, %79, "
              " %80, %81, %82, %83, %84, %85, %86, %87, "
              " %88, %89, %90, %91, %92, %93, %94, %95, "
              " %96, %97, %98, %99,%100,%101,%102,%103, "
              "%104,%105,%106,%107,%108,%109,%110,%111, "
              "%112,%113,%114,%115,%116,%117,%118,%119, "
              "%120,%121,%122,%123,%124,%125,%126,%127}, [%128];"
              : "=f"(tmp[ 0]), "=f"(tmp[ 1]), "=f"(tmp[ 2]), "=f"(tmp[ 3]), "=f"(tmp[ 4]), "=f"(tmp[ 5]), "=f"(tmp[ 6]), "=f"(tmp[ 7]),
                "=f"(tmp[ 8]), "=f"(tmp[ 9]), "=f"(tmp[10]), "=f"(tmp[11]), "=f"(tmp[12]), "=f"(tmp[13]), "=f"(tmp[14]), "=f"(tmp[15]),
                "=f"(tmp[16]), "=f"(tmp[17]), "=f"(tmp[18]), "=f"(tmp[19]), "=f"(tmp[20]), "=f"(tmp[21]), "=f"(tmp[22]), "=f"(tmp[23]),
                "=f"(tmp[24]), "=f"(tmp[25]), "=f"(tmp[26]), "=f"(tmp[27]), "=f"(tmp[28]), "=f"(tmp[29]), "=f"(tmp[30]), "=f"(tmp[31]),
                "=f"(tmp[32]), "=f"(tmp[33]), "=f"(tmp[34]), "=f"(tmp[35]), "=f"(tmp[36]), "=f"(tmp[37]), "=f"(tmp[38]), "=f"(tmp[39]),
                "=f"(tmp[40]), "=f"(tmp[41]), "=f"(tmp[42]), "=f"(tmp[43]), "=f"(tmp[44]), "=f"(tmp[45]), "=f"(tmp[46]), "=f"(tmp[47]),
                "=f"(tmp[48]), "=f"(tmp[49]), "=f"(tmp[50]), "=f"(tmp[51]), "=f"(tmp[52]), "=f"(tmp[53]), "=f"(tmp[54]), "=f"(tmp[55]),
                "=f"(tmp[56]), "=f"(tmp[57]), "=f"(tmp[58]), "=f"(tmp[59]), "=f"(tmp[60]), "=f"(tmp[61]), "=f"(tmp[62]), "=f"(tmp[63]),
                "=f"(tmp[64]), "=f"(tmp[65]), "=f"(tmp[66]), "=f"(tmp[67]), "=f"(tmp[68]), "=f"(tmp[69]), "=f"(tmp[70]), "=f"(tmp[71]),
                "=f"(tmp[72]), "=f"(tmp[73]), "=f"(tmp[74]), "=f"(tmp[75]), "=f"(tmp[76]), "=f"(tmp[77]), "=f"(tmp[78]), "=f"(tmp[79]),
                "=f"(tmp[80]), "=f"(tmp[81]), "=f"(tmp[82]), "=f"(tmp[83]), "=f"(tmp[84]), "=f"(tmp[85]), "=f"(tmp[86]), "=f"(tmp[87]),
                "=f"(tmp[88]), "=f"(tmp[89]), "=f"(tmp[90]), "=f"(tmp[91]), "=f"(tmp[92]), "=f"(tmp[93]), "=f"(tmp[94]), "=f"(tmp[95]),
                "=f"(tmp[96]), "=f"(tmp[97]), "=f"(tmp[98]), "=f"(tmp[99]), "=f"(tmp[100]),"=f"(tmp[101]),"=f"(tmp[102]),"=f"(tmp[103]),
                "=f"(tmp[104]),"=f"(tmp[105]),"=f"(tmp[106]),"=f"(tmp[107]),"=f"(tmp[108]),"=f"(tmp[109]),"=f"(tmp[110]),"=f"(tmp[111]),
                "=f"(tmp[112]),"=f"(tmp[113]),"=f"(tmp[114]),"=f"(tmp[115]),"=f"(tmp[116]),"=f"(tmp[117]),"=f"(tmp[118]),"=f"(tmp[119]),
                "=f"(tmp[120]),"=f"(tmp[121]),"=f"(tmp[122]),"=f"(tmp[123]),"=f"(tmp[124]),"=f"(tmp[125]),"=f"(tmp[126]),"=f"(tmp[127])
              : "r"(addr), "C"(SHAPE), "n"(NUM));
}

template <int num>
__device__ inline void
tcgen05_ld_32x32b(float *tmp, int row, int col) {
  // each 32x32b tile uses 1 register per thread
  tcgen05_ld<num, SHAPE::_32x32b, num>(tmp, row, col);
}

template <int num>
__device__ inline
void tcgen05_ld_16x128b(float *tmp, int row, int col) {
  // each 16x128b tile uses 2 registers per thread
  tcgen05_ld<num * 2, SHAPE::_16x128b, num>(tmp, row, col);
}

template <int num>
__device__ inline
void tcgen05_ld_16x256b(float *tmp, int row, int col) {
  // each 16x256b tile uses 4 registers per thread
  tcgen05_ld<num * 4, SHAPE::_16x256b, num>(tmp, row, col);
}

__device__ inline
void stmatrix_x4(int dst, int *data) {
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared::cta.b16 [%0], {%1, %2, %3, %4};"
              :: "r"(dst), "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3]));
}

constexpr int BLOCK_M = 128;
constexpr int BLOCK_K = 256;
constexpr int NUM_WARPS = 6;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

template <typename T>
__device__ __inline__
T warp_uniform(T x) { return __shfl_sync(0xFFFF'FFFF, x, 0); }

template <int K, int BLOCK_N, int NUM_STAGES>
__global__
__launch_bounds__(TB_SIZE)
void kernel_cutlass (
  const __grid_constant__ CUtensorMap A_tmap,
  const __grid_constant__ CUtensorMap B1_tmap,
  const __grid_constant__ CUtensorMap B2_tmap,
  const char *SFA_ptr,
  const char *SFB1_ptr,
  const char *SFB2_ptr,
  const __grid_constant__ CUtensorMap C_tmap,
  int M, int N
) {
  const int tid = threadIdx.x;
  const int bid_m = warp_uniform(blockIdx.y);
  const int bid_n = warp_uniform(blockIdx.x);

  const int lane_id = tid % WARP_SIZE;
  const int warp_id = warp_uniform(tid / WARP_SIZE);  // making warp uniform is important

  const int off_m = bid_m * BLOCK_M;
  const int off_n = bid_n * BLOCK_N;

  // set up smem
  extern __shared__ __align__(1024) char smem_ptr[];
  const int smem = static_cast<int>(__cvta_generic_to_shared(smem_ptr));
  constexpr int A_size = BLOCK_M * BLOCK_K / 2;
  constexpr int B_size = BLOCK_N * BLOCK_K / 2;
  constexpr int SF_size = 128 * BLOCK_K / 16;
  constexpr int STAGE_SIZE = A_size + B_size * 2 + SF_size * 3;

  // set up mbarriers and tmem
  const int tma_mbar_addr = smem + NUM_STAGES * STAGE_SIZE;
  const int mma_mbar_addr = tma_mbar_addr + NUM_STAGES * 8;
  const int mainloop_mbar_addr = mma_mbar_addr + NUM_STAGES * 8;

  // tmem layout: |  B1   |  B2   | SFA | SFB |
  //              |BLOCK_N|BLOCK_N| ... | ... |
  // each MMA consumes:
  // - (128, 64) of A -> (128, 4) of SFA -> reshaped as (32, 4', 4) -> 4 tmem columns
  constexpr int SFA_tmem = BLOCK_N * 2;
  constexpr int SFB1_tmem = SFA_tmem + 4 * (BLOCK_K / MMA_K);
  constexpr int SFB2_tmem = SFB1_tmem + 4 * (BLOCK_K / MMA_K);

  constexpr uint64_t cache_A = EVICT_NORMAL;
  constexpr uint64_t cache_B = EVICT_FIRST;

  constexpr int bar_epilogue = 2;
  constexpr int rest_k = K / 16 / 4;

  if (warp_id == 0 && elect_sync()) {
    // this is better than cp.async.bulk.prefetch.tensor
    asm volatile("prefetch.tensormap [%0];" :: "l"(&A_tmap) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&B1_tmap) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&B2_tmap) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"(&C_tmap) : "memory");

    //tma_3d_prefetch(&A_tmap, 0, off_m, 0, cache_A);
    //tma_3d_prefetch(&B1_tmap, 0, off_n, 0, cache_B);
    //tma_3d_prefetch(&B2_tmap, 0, off_n, 0, cache_B);

    constexpr int num_prefetch = 2;
    tma_prefetch(SFA_ptr + bid_m * rest_k * 512, SF_size * num_prefetch, cache_A);
    tma_prefetch(SFB1_ptr + (off_n / 128) * rest_k * 512, SF_size * num_prefetch, cache_B);
    tma_prefetch(SFB2_ptr + (off_n / 128) * rest_k * 512, SF_size * num_prefetch, cache_B);
  }
  else if (warp_id == 1 && elect_sync()) {
    // 1 thread init mbarrier
    for (int i = 0; i < NUM_STAGES * 2 + 1; i++)
      mbarrier_init(tma_mbar_addr + i * 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");  // visible to async proxy
  }
  __syncthreads();  // visible to all threads

  constexpr int num_iters = K / BLOCK_K;

  // warp-specialization
  if (warp_id == NUM_WARPS - 2) {
    // TMA warp

    if (elect_sync()) {
      int stage_id = 0;
      int mma_phase = 1;

      // 512 = 32x4x4
      const char *SFA_src = SFA_ptr + bid_m * rest_k * 512;
      const char *SFB1_src = SFB1_ptr + (off_n / 128) * rest_k * 512;
      const char *SFB2_src = SFB2_ptr + (off_n / 128) * rest_k * 512;

      // don't unroll. this is important
      #pragma unroll 1
      for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        // select tma mbar and smem
        const int mbar_addr = tma_mbar_addr + stage_id * 8;
        const int A_smem = smem + stage_id * STAGE_SIZE;
        const int B1_smem = A_smem + A_size;
        const int B2_smem = B1_smem + B_size;
        const int SFA_smem = B2_smem + B_size;
        const int SFB1_smem = SFA_smem + SF_size;
        const int SFB2_smem = SFB1_smem + SF_size;

        // wait MMA
        mbarrier_wait(mma_mbar_addr + stage_id * 8, mma_phase);

        // issue TMA
        tma_3d_gmem2smem(A_smem, &A_tmap, 0, off_m, iter_k, mbar_addr, cache_A);
        tma_3d_gmem2smem(B1_smem, &B1_tmap, 0, off_n, iter_k, mbar_addr, cache_B);
        tma_3d_gmem2smem(B2_smem, &B2_tmap, 0, off_n, iter_k, mbar_addr, cache_B);

        // 2048 = 128 x BLOCK_K / 16
        tma_gmem2smem(SFA_smem, SFA_src + iter_k * 2048, SF_size, mbar_addr, cache_A);
        tma_gmem2smem(SFB1_smem, SFB1_src + iter_k * 2048, SF_size, mbar_addr, cache_B);
        tma_gmem2smem(SFB2_smem, SFB2_src + iter_k * 2048, SF_size, mbar_addr, cache_B);

        // signal TMA done
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                    :: "r"(mbar_addr), "r"(STAGE_SIZE) : "memory");

        stage_id = (stage_id + 1) % NUM_STAGES;
        if (stage_id == 0)
          mma_phase ^= 1;
      }
    }
  }
  else if (warp_id == NUM_WARPS - 1) {
    // MMA warp
    // allocate tmem
    // must use a separate smem address, even though we don't care about the returned value
    int addr = smem + STAGE_SIZE * NUM_STAGES + (2 * NUM_STAGES + 1) * 8;
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(addr), "r"(BLOCK_N * 4));

    // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-instruction-descriptor
    constexpr uint32_t i_desc = (1U << 7U)   // atype=E2M1
                              | (1U << 10U)  // btype=E2M1
                              | ((uint32_t)BLOCK_N >> 3U << 17U)
                              | ((uint32_t)BLOCK_M >> 7U << 27U)
                              ;

    if (elect_sync()) {
      int enable_input_d = 0;
      int stage_id = 0;
      int tma_phase = 0;

      for (int iter_k = 0; iter_k < num_iters; iter_k++) {
        // select smem
        const int A_smem = smem + stage_id * STAGE_SIZE;
        const int B1_smem = A_smem + A_size;
        const int B2_smem = B1_smem + B_size;
        const int SFA_smem = B2_smem + B_size;
        const int SFB1_smem = SFA_smem + SF_size;
        const int SFB2_smem = SFB1_smem + SF_size;

        // set up shared memory descriptors
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-shared-memory-descriptor
        // no swizzling
        constexpr uint64_t SF_desc = (desc_encode(8 * 16) << 32ULL) | (1ULL << 46ULL);
        uint64_t sfa_desc = SF_desc | (SFA_smem >> 4);
        uint64_t sfb1_desc = SF_desc | (SFB1_smem >> 4);
        uint64_t sfb2_desc = SF_desc | (SFB2_smem >> 4);

        // AB_desc: 128-byte swizzling. LBO is implied to be 1.
        constexpr uint64_t AB_desc = (desc_encode(8 * 128) << 32ULL) | (1ULL << 46ULL) | (2ULL << 61ULL);
        uint64_t a_desc = AB_desc | (A_smem >> 4);
        uint64_t b1_desc = AB_desc | (B1_smem >> 4);
        uint64_t b2_desc = AB_desc | (B2_smem >> 4);

        int scale_A_tmem = SFA_tmem;
        int scale_B1_tmem = SFB1_tmem;
        int scale_B2_tmem = SFB2_tmem;

        // select the correct tmem column when BLOCK_N=64
        int sfb_offset = 0;
        if constexpr (BLOCK_N == 64)
          sfb_offset = (bid_n % 2) * 2;

        // wait TMA
        mbarrier_wait(tma_mbar_addr + stage_id * 8, tma_phase);

        // NOTE: this doesn't work with BLOCK_N=32, since apparently tcgen05.mma requires SFB_tmem
        // to have 2-column (8-byte) alignment (looks like not documented).
        for (int k = 0; k < BLOCK_K / MMA_K; k++) {
          // tcgen05.cp -> tcgen05.mma should be pipelined correctly per PTX doc
          // https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-memory-consistency-model-pipelined-instructions
          tcgen05_cp_nvfp4(scale_A_tmem, sfa_desc);
          tcgen05_cp_nvfp4(scale_B1_tmem, sfb1_desc);
          tcgen05_cp_nvfp4(scale_B2_tmem, sfb2_desc);

          tcgen05_mma_nvfp4<COLLECTOR_USAGE::A_FILL   >(      0, a_desc, b1_desc, i_desc, scale_A_tmem, scale_B1_tmem + sfb_offset, enable_input_d);
          tcgen05_mma_nvfp4<COLLECTOR_USAGE::A_LASTUSE>(BLOCK_N, a_desc, b2_desc, i_desc, scale_A_tmem, scale_B2_tmem + sfb_offset, enable_input_d);
          enable_input_d = 1;

          // go to the next 4 columns
          scale_A_tmem += 4;
          scale_B1_tmem += 4;
          scale_B2_tmem += 4;

          // go to the next 512-byte
          sfa_desc += (512 >> 4);
          sfb1_desc += (512 >> 4);
          sfb2_desc += (512 >> 4);

          // go to the next 32-byte
          a_desc += (32 >> 4);
          b1_desc += (32 >> 4);
          b2_desc += (32 >> 4);
        }

        // signal MMA done
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                    :: "r"(mma_mbar_addr + stage_id * 8) : "memory");

        stage_id = (stage_id + 1) % NUM_STAGES;
        if (stage_id == 0)
          tma_phase ^= 1;
      }

      // signal mainloop done
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];"
                  :: "r"(mainloop_mbar_addr) : "memory");
    }
  }
  else {
    // epilogue warps
    // wait mainloop
    if (warp_id == 0)
      mbarrier_wait(mainloop_mbar_addr, 0);
    asm volatile("bar.sync %0, %1;" :: "n"(bar_epilogue), "r"(4 * WARP_SIZE) : "memory");
    asm volatile("tcgen05.fence::after_thread_sync;");

    // multiplying y first is sometimes faster
    auto act = [](float x, float y) { return y * x / (1.0f + __expf(-x)); };

    // iterate 128-byte of C store (64 elems)
    for (int epi = 0; epi < BLOCK_N / 64; epi++) {
      int base_addr = smem + epi * (BLOCK_M * 64) * sizeof(half);

      float b1[64];
      float b2[64];

      int row = warp_id * 32;
      tcgen05_ld_32x32b<64>(b1, row, epi * 64);  // load 32x8 tile
      tcgen05_ld_32x32b<64>(b2, row, epi * 64 + BLOCK_N);
      asm volatile("tcgen05.wait::ld.sync.aligned;");

      half2 out[32];
      for (int i = 0; i < 32; i++) {
        float o0 = act(b1[i * 2 + 0], b2[i * 2 + 0]);
        float o1 = act(b1[i * 2 + 1], b2[i * 2 + 1]);
        out[i] = __float22half2_rn({o0, o1});
      }

      int addr = base_addr;
      addr += tid * 64 * sizeof(half); // row

      // this is faster than doing reinterpret_cast<int4 *>
      auto st_shared = [](int dst, int *data) {
        asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                    :: "r"(dst), "r"(data[0]), "r"(data[1]), "r"(data[2]), "r"(data[3]));
      };

      for (int i = 0; i < 8; i++)
        st_shared(addr + (i ^ (tid % 8)) * 8 * sizeof(half), reinterpret_cast<int *>(out + i * 4));

      //asm volatile("fence.proxy.async.shared::cta;");  // bi-directional
      asm volatile("fence.proxy.async::generic.release.sync_restrict::shared::cta.cluster;");  // uni-directional. slightly faster
      asm volatile("bar.sync %0, %1;" :: "n"(bar_epilogue), "r"(4 * WARP_SIZE) : "memory");  // wait for everyone

      if (warp_id == 0 && elect_sync()) {
        tma_2d_smem2gmem(&C_tmap, off_n + epi * 64, off_m, base_addr);
        asm volatile("cp.async.bulk.commit_group;");
      }
    }

    // we don't need a bar.sync here because we already have a bar.sync before TMA
    if (warp_id == 1) {
      // deallocate tmem. tmem address should be 0.
      asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(0), "r"(BLOCK_N * 4));
    }
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
  //check_cu(err);
}

void init_C_tmap(
  CUtensorMap *tmap,
  const half *ptr,
  uint64_t global_height, uint64_t global_width,
  uint32_t shared_height, uint32_t shared_width
) {
  constexpr uint32_t rank = 2;
  uint64_t globalDim[rank]       = {global_width, global_height};
  uint64_t globalStrides[rank-1] = {global_width * sizeof(ptr[0])};  // in bytes
  uint32_t boxDim[rank]          = {shared_width, shared_height};
  uint32_t elementStrides[rank]  = {1, 1};

  auto err = cuTensorMapEncodeTiled(
    tmap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
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
  //check_cu(err);
}

template <
  int K,
  int BLOCK_N,
  int NUM_STAGES
>
void dual_gemm_launch(
  const char *A_ptr,
  const char *B1_ptr,
  const char *B2_ptr,
  const char *SFA_ptr,
  const char *SFB1_ptr,
  const char *SFB2_ptr,
  half *C_ptr,
  int M, int N
) {
  CUtensorMap A_tmap, B1_tmap, B2_tmap, C_tmap;
  init_AB_tmap(&A_tmap, A_ptr, M, K, BLOCK_M, BLOCK_K);
  init_AB_tmap(&B1_tmap, B1_ptr, N, K, BLOCK_N, BLOCK_K);
  init_AB_tmap(&B2_tmap, B2_ptr, N, K, BLOCK_N, BLOCK_K);
  init_C_tmap(&C_tmap, C_ptr, M, N, BLOCK_M, 64);

  dim3 grid(N / BLOCK_N, M / BLOCK_M);

  int AB_size = (BLOCK_M + BLOCK_N * 2) * (BLOCK_K / 2);
  int SFAB_size = 128 * (BLOCK_K / 16) * (1 + 2);
  int mbar_size = (2 * NUM_STAGES + 1) * 8;
  int smem_size = (AB_size + SFAB_size) * NUM_STAGES + mbar_size + 4;

  // cutlass incantation (this affects ptxas)
  auto this_kernel = kernel_cutlass<K, BLOCK_N, NUM_STAGES>;
  if (smem_size > 48'000)
    cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  this_kernel<<<grid, TB_SIZE, smem_size>>>(A_tmap, B1_tmap, B2_tmap, SFA_ptr, SFB1_ptr, SFB2_ptr, C_tmap, M, N);
}

at::Tensor dual_gemm(
  const at::Tensor& A,
  const at::Tensor& B1,
  const at::Tensor& B2,
  const at::Tensor& SFA,
  const at::Tensor& SFB1,
  const at::Tensor& SFB2,
        at::Tensor& C
) {
  const int M = A.size(0);
  const int N = B1.size(0);
  const int K = A.size(1) * 2;

  auto A_ptr    = reinterpret_cast<const char *>(A.data_ptr());
  auto B1_ptr   = reinterpret_cast<const char *>(B1.data_ptr());
  auto B2_ptr   = reinterpret_cast<const char *>(B2.data_ptr());
  auto SFA_ptr  = reinterpret_cast<const char *>(SFA.data_ptr());
  auto SFB1_ptr = reinterpret_cast<const char *>(SFB1.data_ptr());
  auto SFB2_ptr = reinterpret_cast<const char *>(SFB2.data_ptr());
  auto C_ptr    = reinterpret_cast<half *>(C.data_ptr());

#define LAUNCH(K_, BLOCK_N, NUM_STAGES) \
  dual_gemm_launch<K_, BLOCK_N, NUM_STAGES>(A_ptr, B1_ptr, B2_ptr, SFA_ptr, SFB1_ptr, SFB2_ptr, C_ptr, M, N);

  if (false) {}
  else if (M == 256 && K == 7168) LAUNCH(7168,  64, 5)
  else if (M == 512 && K == 7168) LAUNCH(7168, 128, 4)
  else if (M == 256 && K == 4096) LAUNCH(4096,  64, 5)
  // the rest
  else if (K ==  256) LAUNCH( 256, 128, 4)
  else if (K ==  512) LAUNCH( 512, 128, 4)
  else if (K == 1536) LAUNCH(1536, 128, 4)
  else if (K == 2048) LAUNCH(2048, 128, 4)
  else if (K == 2304) LAUNCH(2304, 128, 4)
  else if (K == 7168) LAUNCH(7168, 128, 4)

#undef LAUNCH

  return C;
}

TORCH_LIBRARY(my_module, m) {
  m.def("dual_gemm(Tensor A, Tensor B1, Tensor B2, Tensor SFA, Tensor SFB1, Tensor SFB2, Tensor(a!) C) -> Tensor");
  m.impl("dual_gemm", &dual_gemm);
}
"""

load_inline(
    "dual_gemm",
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
dual_gemm = torch.ops.my_module.dual_gemm


def custom_kernel(data: input_t) -> output_t:
    return dual_gemm(data[0], data[1], data[2], data[6], data[7], data[8], data[9])
