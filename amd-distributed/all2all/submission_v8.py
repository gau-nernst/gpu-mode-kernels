#!POPCORN leaderboard amd-all2all
#!POPCORN gpu MI300x8

import faulthandler
import gzip
import json
import math
import os
import sys

os.environ["PYTORCH_ROCM_ARCH"] = "gfx942"

import torch
import torch.distributed as dist
import triton
from reference import MoEConfig
from task import input_t, output_t
from torch import Tensor
from torch.utils.cpp_extension import load_inline

# this will print segfault to stderr
faulthandler.enable(file=sys.stderr, all_threads=True)

cuda_src = r"""
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <cooperative_groups.h>
#include <stdio.h>

#define STRINGIFY(x) #x
#define CUDA_CHECK(call)                                                             \
  do {                                                                               \
    cudaError_t err = call;                                                          \
    TORCH_CHECK(err == cudaSuccess, STRINGIFY(call), ": ", cudaGetErrorString(err)); \
  } while (0)

constexpr int WARP_SIZE = 64;
constexpr int WORLD_SIZE = 8;

// max problem size
// use this to slice symmetric heap
constexpr int NUM_EXPERTS_    = 256;
constexpr int TOPK_           = 8;
constexpr int MAX_NUM_TOKENS_ = 256;
constexpr int DIM_            = 7168;

at::Tensor malloc_with_flags(int64_t size, int64_t flag) {
  void *ptr;
  CUDA_CHECK(hipExtMallocWithFlags(&ptr, size, flag));

  int device;
  CUDA_CHECK(cudaGetDevice(&device));

  auto options = at::TensorOptions().dtype(at::kChar).device(at::kCUDA, device);
  return torch::from_blob(ptr, {size}, [](void *ptr){ CUDA_CHECK(hipFree(ptr)); }, options);
}

// input is CUDA, but output is CPU
at::Tensor get_ipc_handle(const at::Tensor& x) {
  // IPC handle as a tensor
  auto options = at::TensorOptions().dtype(at::kChar).device(at::kCPU);
  at::Tensor h = at::empty({sizeof(cudaIpcMemHandle_t)}, options);
  auto h_ptr = reinterpret_cast<cudaIpcMemHandle_t *>(h.data_ptr());
  CUDA_CHECK(cudaIpcGetMemHandle(h_ptr, x.data_ptr()));
  return h;
}

int64_t open_ipc_handle(const at::Tensor& h) {
  void *ptr;
  auto h_ptr = reinterpret_cast<cudaIpcMemHandle_t *>(h.data_ptr());
  CUDA_CHECK(cudaIpcOpenMemHandle(&ptr, h_ptr[0], cudaIpcMemLazyEnablePeerAccess));
  return reinterpret_cast<int64_t>(ptr);
}

void close_ipc_handle(int64_t addr) {
  void *ptr = reinterpret_cast<void *>(addr);
  CUDA_CHECK(cudaIpcCloseMemHandle(ptr));
}

// cache global P2PState object in C++ to avoid Python->C++ overhead
struct P2PState {
  int rank;
  int64_t *heap_bases;
  int8_t *sym_buf;

  // profiling
  int64_t *profile;

  // dispatch workspace
  int *send_counts;

  // keep track of MoE metadata. used by both dispatch and combine
  int *moe_meta;    // [num_local_experts][max_recv_per_expert][2]
  int *moe_counts;  // [num_local_experts]

  // token buffer
  half *moe_x;      // [num_local_experts][max_recv_per_expert][DIM]
  half *y;          // [max_num_tokens][DIM]  - after combine
} P2P_STATE = {};

template <typename T>
void cuda_zeros(T **devPtr, size_t cnt) {
  CUDA_CHECK(cudaMalloc(devPtr, cnt * sizeof(T)));
  CUDA_CHECK(cudaMemset(devPtr[0], 0, cnt * sizeof(T)));
}

void p2p_state_init(
  int64_t rank,
  const at::Tensor& heap_bases
) {
  P2P_STATE.rank       = rank;
  P2P_STATE.heap_bases = heap_bases.data_ptr<int64_t>();
  P2P_STATE.sym_buf = reinterpret_cast<int8_t *>(P2P_STATE.heap_bases[rank]);

  cuda_zeros(&P2P_STATE.profile, 10'000);

  // dispatch workspace
  cuda_zeros(&P2P_STATE.send_counts, WORLD_SIZE);

  // MoE metadata
  int num_local_experts   = NUM_EXPERTS_ / WORLD_SIZE;
  int max_recv_per_expert = MAX_NUM_TOKENS_ * WORLD_SIZE;
  cuda_zeros(&P2P_STATE.moe_meta, num_local_experts * max_recv_per_expert * 2);
  cuda_zeros(&P2P_STATE.moe_counts, num_local_experts);

  // token buffer
  cuda_zeros(&P2P_STATE.moe_x, num_local_experts * max_recv_per_expert * DIM_);
  cuda_zeros(&P2P_STATE.y, MAX_NUM_TOKENS_ * DIM_);
}

void p2p_state_destroy() {
  CUDA_CHECK(cudaFree(P2P_STATE.profile));
  CUDA_CHECK(cudaFree(P2P_STATE.send_counts));
  CUDA_CHECK(cudaFree(P2P_STATE.moe_meta));
  CUDA_CHECK(cudaFree(P2P_STATE.moe_counts));
  CUDA_CHECK(cudaFree(P2P_STATE.moe_x));
  CUDA_CHECK(cudaFree(P2P_STATE.y));
}

at::Tensor p2p_state_get_profile() {
  auto options = at::TensorOptions().dtype(at::kLong).device(at::kCUDA, P2P_STATE.rank);
  return torch::from_blob(P2P_STATE.profile, {10'000}, options);
}

template <typename T>
__device__ __host__
T *translate(T *ptr, int64_t src_base, int64_t dst_base) {
  static_assert(sizeof(ptr) == sizeof(int64_t));
  const int64_t offset = reinterpret_cast<int64_t>(ptr) - src_base;
  return reinterpret_cast<T *>(dst_base + offset);
}

using i32x2 = int __attribute__((__vector_size__(2 * sizeof(int))));
using fp32x4 = float __attribute__((__vector_size__(4 * sizeof(float))));
using fp16x8 = _Float16 __attribute__((__vector_size__(8 * sizeof(_Float16))));

__device__ __host__
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__
int64_t read_realtime() {
  int64_t t;
  asm volatile("s_waitcnt vmcnt(0)\n"
               "s_memrealtime %0\n"
               "s_waitcnt lgkmcnt(0)" : "=s"(t));
  return t;
}

__device__
int profile_start(int64_t *profile) {
  int i = atomicAdd(reinterpret_cast<int*>(profile), 1);
  profile[1 + i * 4] = read_realtime();
  return i;
}

__device__
void profile_stop(int64_t *profile, int i, int tag, int pid) {
  profile[1 + i * 4 + 1] = read_realtime() - profile[1 + i * 4];
  profile[1 + i * 4 + 2] = tag;
  profile[1 + i * 4 + 3] = pid;
}

template <int SLEEP = 0, bool RESET_FLAG = true>
__device__
int spin_lock_system(int *addr) {
  int flag = 0;
  //while ((flag = __hip_atomic_load(addr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM)) == 0) {
  while ((flag = __hip_atomic_load(addr, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) == 0) {
    if constexpr (SLEEP > 0)
      __builtin_amdgcn_s_sleep(SLEEP);
  }
  asm volatile("buffer_inv sc1;  invalidate cache outside of for loop");
  if constexpr (RESET_FLAG) {
    __builtin_nontemporal_store(0, addr);
    //addr[0] = 0;
  }
  return flag;
}

template <int DIM>
__device__
void copy_token(half *dst, const half *src, int lane_id) {
  constexpr int multiplier = sizeof(fp16x8) / sizeof(half);  // 8
  constexpr int num_iters = DIM / (WARP_SIZE * multiplier);

  for (int iter = 0; iter < num_iters; iter++) {
    const int idx = (iter * WARP_SIZE + lane_id) * multiplier;
    fp16x8 data = reinterpret_cast<const fp16x8 *>(src + idx)[0];
    reinterpret_cast<fp16x8 *>(dst + idx)[0] = data;
  }

  // DIM = 2880
  if constexpr ((DIM % (WARP_SIZE * multiplier)) > 0) {
    const int start_idx = num_iters * (WARP_SIZE * multiplier) + lane_id;
    for (int idx = start_idx; idx < DIM; idx += WARP_SIZE) {
      half data = src[idx];
      dst[idx] = data;
    }
  }
}

template <int NUM_WARPS, int DIM, bool DO_SEND, bool DO_RECV, bool DO_PROFILE>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE)
void dispatch_kernel(
  // send
  const half * __restrict__ pre_x,            // [num_tokens, DIM]
  const int * __restrict__ indices,           // [num_tokens, topk]
  // shapes
  const int num_tokens,
  const int topk,
  const int num_experts,
  const int max_num_tokens,
  const P2PState p2p_state
) {
  const int local_rank      = p2p_state.rank;
  const int64_t *heap_bases = p2p_state.heap_bases;   // [WORLD_SIZE]

  // workspace
  int *send_counts    = p2p_state.send_counts;  // [WORLD_SIZE]

  // output buffers, cached in P2P_STATE
  half *post_x        = p2p_state.moe_x;       // [num_local_experts][max_recv_per_expert][DIM]
  int *post_meta      = p2p_state.moe_meta;    // [num_local_experts][max_recv_per_expert][2], src_rank and flat_pos
  int *post_counts    = p2p_state.moe_counts;  // [num_local_experts]

  // slice symmetric heap
  half *comm_x          = reinterpret_cast<half *>(p2p_state.sym_buf);                             // [WORLD_SIZE][max_num_tokens * topk][DIM]
  int *comm_meta        = reinterpret_cast<int *>(comm_x + (WORLD_SIZE * MAX_NUM_TOKENS_ * TOPK_ * DIM_));  // [WORLD_SIZE][max_num_tokens * topk][2], local_expert_id and flat_pos
  int *comm_flag        = comm_meta + (WORLD_SIZE * MAX_NUM_TOKENS_ * TOPK_ * 2);                           // [WORLD_SIZE][max_num_tokens * topk]
  int *comm_recv_counts = comm_flag + (WORLD_SIZE * MAX_NUM_TOKENS_ * TOPK_);                               // [WORLD_SIZE]
  int *comm_recv_flag   = comm_recv_counts + WORLD_SIZE;                                                    // [WORLD_SIZE]

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  const int bid = blockIdx.x;
  const int num_local_experts = num_experts / WORLD_SIZE;
  const int max_recv_per_expert = max_num_tokens * WORLD_SIZE;

  const int num_blocks = gridDim.x;

  if constexpr (DO_SEND) {
    // SEND stage
    // 1st pass - bincount using the 1st block
    // other block can start sending token, since it's independent.
    if (bid == 0) {
      int e0_id, e1_id;
      if constexpr (DO_PROFILE) if (tid == 0) e0_id = profile_start(p2p_state.profile);

      // reset post_counts buffer used in do_recv
      // since zero_() is very expensive
      if (tid < num_local_experts)
        post_counts[tid] = 0;

      // count in smem
      __shared__ int send_counts_smem[WORLD_SIZE];
      if (tid < WORLD_SIZE)
        send_counts_smem[tid] = 0;
      __syncthreads();

      // use i32x2 load since topk is always divisible by 2 -> don't need to handle the remainder
      for (int flat_pos = tid * 2; flat_pos < num_tokens * topk; flat_pos += TB_SIZE * 2) {
        //i32x2 tmp = reinterpret_cast<const i32x2 *>(indices + flat_pos)[0];  // global_expert_id x2
        int2 tmp = __ldg(reinterpret_cast<const int2 *>(indices + flat_pos));  // global_expert_id x2
        atomicAdd(send_counts_smem + (tmp.x / num_local_experts), 1);
        atomicAdd(send_counts_smem + (tmp.y / num_local_experts), 1);
      }
      __syncthreads();  // wait for all warps/threads to finish

      // send to other ranks
      if (tid < WORLD_SIZE) {
        int *dst_recv_counts = translate(comm_recv_counts, heap_bases[local_rank], heap_bases[tid]);
        dst_recv_counts[local_rank] = send_counts_smem[tid];

        int *flag_addr = translate(comm_recv_flag, heap_bases[local_rank], heap_bases[tid]);
        flag_addr += local_rank;
        __hip_atomic_store(flag_addr, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
      }
      __syncthreads();
      if constexpr (DO_PROFILE) if (tid == 0) profile_stop(p2p_state.profile, e0_id, 0, bid);

      if constexpr (DO_PROFILE) if (tid == 0) e1_id = profile_start(p2p_state.profile);
      if (tid < WORLD_SIZE)
        spin_lock_system(comm_recv_flag + tid);
      if constexpr (DO_PROFILE) if (tid == 0) profile_stop(p2p_state.profile, e1_id, 1, bid);
    }
    else {
      // 2nd pass - sending the tokens
      // each warp process 1 (flat) token
      for (int flat_pos = (bid - 1) * NUM_WARPS + warp_id;
              flat_pos < num_tokens * topk;
              flat_pos += (num_blocks - 1) * NUM_WARPS) {
        int e_id;
        if constexpr (DO_PROFILE) if (bid < 5 && tid == 0) e_id = profile_start(p2p_state.profile);

        const int src_pos = flat_pos / topk;
        const int k = flat_pos % topk;

        const int global_expert_id = indices[flat_pos];
        const int dst_rank = global_expert_id / num_local_experts;

        // atomicAdd to get position of this token in the dst_rank buffer
        // NOTE: atomicAdd returns old
        int dst_pos;
        if (lane_id == 0)  // atomic add on lane0 only
          dst_pos = atomicAdd(send_counts + dst_rank, 1);
        dst_pos = __shfl(dst_pos, 0);  // warp-broadcast

        // copy data
        half *dst_x = translate(comm_x, heap_bases[local_rank], heap_bases[dst_rank]);
        dst_x += local_rank * max_num_tokens * topk * DIM;
        copy_token<DIM>(dst_x + dst_pos * DIM, pre_x + src_pos * DIM, lane_id);

        if (lane_id == 0) {
          // write metadata
          int *dst_meta = translate(comm_meta, heap_bases[local_rank], heap_bases[dst_rank]);
          i32x2 tmp;
          tmp[0] = global_expert_id % num_local_experts;  // local_expert_id
          tmp[1] = flat_pos;
          reinterpret_cast<i32x2 *>(dst_meta + (local_rank * max_num_tokens * topk + dst_pos) * 2)[0] = tmp;
          //printf("rank %d - dispatch-send: dst_rank=%d, flat_pos=%d, pos=%d, k=%d\n", local_rank, dst_rank, flat_pos, pos, k);

          // SIGNAL done for this token
          int *flag_addr = translate(comm_flag, heap_bases[local_rank], heap_bases[dst_rank]);
          flag_addr += local_rank * max_num_tokens * topk + dst_pos;
          __hip_atomic_store(flag_addr, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }

        if constexpr (DO_PROFILE) if (bid < 5 && tid == 0) profile_stop(p2p_state.profile, e_id, 2, bid);
      }
    }
  }

  // wait for all blocks to finish
  // if send and recv are separate kernels, there is an implicit grid-sync
  if constexpr (DO_SEND && DO_RECV)
    cooperative_groups::this_grid().sync();

  if constexpr (DO_RECV) {
    // reset send_counts buffer used in do_send
    // since zero_() is very expensive
    if (bid == 0 && tid < WORLD_SIZE)
      send_counts[tid] = 0;

    // RECV stage
    // each block is assigned a src_rank based on its bid (round-robin)
    // hence, each src_rank is handled by (num_blocks / WORLD_SIZE) threadblocks
    const int src_rank = bid % WORLD_SIZE;
    const int recv_count = comm_recv_counts[src_rank];

    // each warp handles 1 token
    // divide by WORLD_SIZE due to src_rank assignment above
    for (int comm_pos = (bid / WORLD_SIZE) * NUM_WARPS + warp_id;
             comm_pos < recv_count;
             comm_pos += (num_blocks / WORLD_SIZE) * NUM_WARPS) {
      int e_id;
      if constexpr (DO_PROFILE) if (bid < 5 && tid == 0) e_id = profile_start(p2p_state.profile);

      const int offset = src_rank * max_num_tokens * topk + comm_pos;

      // wait for arrival
      if (lane_id == 0)
        spin_lock_system(comm_flag + offset);
      __builtin_amdgcn_wave_barrier(); // equivalent to __syncwarp()

      i32x2 tmp = reinterpret_cast<i32x2 *>(comm_meta + offset * 2)[0];
      int local_expert_id = tmp[0];
      int src_flat_pos = tmp[1];

      int post_pos;
      if (lane_id == 0)
        post_pos = atomicAdd(post_counts + local_expert_id, 1);
      post_pos = __shfl(post_pos, 0);

      // copy token
      copy_token<DIM>(post_x + (local_expert_id * max_recv_per_expert + post_pos) * DIM,
                      comm_x + offset * DIM,
                      lane_id);

      // copy metadata
      if (lane_id == 0) {
        i32x2 tmp2;
        tmp2[0] = src_rank;
        tmp2[1] = src_flat_pos;
        reinterpret_cast<i32x2 *>(post_meta + (local_expert_id * max_recv_per_expert + post_pos) * 2)[0] = tmp2;
        //printf("rank %d - dispatch: src_rank=%d, pos=%d, k=%d\n", local_rank, src_rank, src_flat_pos / topk, src_flat_pos % topk);
      }

      if constexpr (DO_PROFILE) if (bid < 5 && tid == 0) profile_stop(p2p_state.profile, e_id, 3, bid);
    }
  }
}

void dispatch(
  // send
  const at::Tensor& pre_x,             // [num_tokens, DIM]
  const at::Tensor& indices,           // [num_tokens, topk]
  int64_t num_experts,
  int64_t max_num_tokens,
  int64_t evt_i64
) {
  if (false) {
    TORCH_CHECK(pre_x.scalar_type() == at::kHalf);
    TORCH_CHECK(pre_x.is_contiguous());
    TORCH_CHECK(indices.is_contiguous());
  }

  const int num_tokens = pre_x.size(0);
  const int dim        = pre_x.size(1);
  const int topk       = indices.size(1);

  // pytorch requires i64, but we pass in i32
  const int num_experts_i32    = num_experts;
  const int max_num_tokens_i32 = max_num_tokens;

  auto pre_x_ptr   = reinterpret_cast<const half *>(pre_x.data_ptr());
  auto indices_ptr = indices.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(P2P_STATE.rank);

  constexpr int NUM_WARPS = 8;
  const int grid_size = 128;  // must be divisible by WORLD_SIZE
  const int tb_size = NUM_WARPS * WARP_SIZE;
  constexpr bool DO_PROFILE = false;

  void *kernel_args[] = {(void *)&pre_x_ptr, (void *)&indices_ptr,
                         (void *)&num_tokens, (void *)&topk, (void *)&num_experts_i32,
                         (void *)&max_num_tokens_i32, (void *)&P2P_STATE};

#define my_dispatch(DIM) { \
  /*CUDA_CHECK(cudaLaunchCooperativeKernel((void *)(dispatch_kernel<NUM_WARPS, DIM, true, true, DO_PROFILE>), grid_size, tb_size, kernel_args, 0, stream));*/ \
  CUDA_CHECK(cudaLaunchKernel((void *)(dispatch_kernel<NUM_WARPS, DIM, true, false, DO_PROFILE>), grid_size, tb_size, kernel_args, 0, stream)); \
  if (evt_i64 != 0) { cudaEvent_t evt = reinterpret_cast<cudaEvent_t>(evt_i64); CUDA_CHECK(cudaEventRecord(evt, stream)); } \
  CUDA_CHECK(cudaLaunchKernel((void *)(dispatch_kernel<NUM_WARPS, DIM, false, true, DO_PROFILE>), grid_size, tb_size, kernel_args, 0, stream)); \
  }

  if      (dim == 2048) my_dispatch(2048)
  else if (dim == 2880) my_dispatch(2880)
  else if (dim == 4096) my_dispatch(4096)
  else if (dim == 6144) my_dispatch(6144)
  else if (dim == 7168) my_dispatch(7168)
  else TORCH_CHECK(false, "Unsupported dim ", dim);

#undef my_dispatch
}

template <int NUM_WARPS, int DIM, int MODE, bool DO_PROFILE>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE)
void combine_kernel(
  const float *weights,  // [num_tokens][topk]
  // shapes
  const int num_tokens,
  const int topk,
  const int num_experts,
  const int max_num_tokens,
  const P2PState p2p_state
) {
  const int local_rank      = p2p_state.rank;
  const int64_t *heap_bases = p2p_state.heap_bases;   // [WORLD_SIZE]

  // input buffers, cached in P2P_STATE
  half *post_x     = p2p_state.moe_x;       // [num_local_experts][max_recv_per_token][DIM]
  int *post_meta   = p2p_state.moe_meta;    // [num_local_experts][max_recv_per_expert][2], src_rank and flat_pos
  int *post_counts = p2p_state.moe_counts;  // [num_local_experts]

  // output buffers, cached in P2P_STATE
  half *pre_x = p2p_state.y;  // [num_tokens][DIM]

  half *comm_x   = reinterpret_cast<half *>(p2p_state.sym_buf + (5ULL << 30ULL));       // [max_num_tokens][topk][DIM]
  int *comm_flag = reinterpret_cast<int *>(comm_x + (MAX_NUM_TOKENS_ * TOPK_ * DIM_));  // [max_num_tokens][topk]

  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int bid = blockIdx.x;
  const int num_blocks = gridDim.x;

  const int num_local_experts = num_experts / WORLD_SIZE;
  const int max_recv_per_token = max_num_tokens * WORLD_SIZE;

  auto do_send = [&](int bid, int num_blocks) {
    const float moe_w = 1 + local_rank;

    // SEND stage
    // each warp handle 1 token
    for (int idx = bid * NUM_WARPS + warp_id;
             idx < num_local_experts * max_recv_per_token;
             idx += num_blocks * NUM_WARPS) {
      const int local_expert_id = idx / max_recv_per_token;
      const int pos = idx % max_recv_per_token;

      if (pos >= post_counts[local_expert_id])
        continue;

      int e_id;
      if constexpr (DO_PROFILE) if (bid < 5 && tid == 0) e_id = profile_start(p2p_state.profile);

      i32x2 tmp = reinterpret_cast<const i32x2 *>(post_meta + idx * 2)[0];
      const int src_rank = tmp[0];
      const int flat_pos = tmp[1];

      half *dst_comm_x = translate(comm_x, heap_bases[local_rank], heap_bases[src_rank]);

            half *dst_token = dst_comm_x + flat_pos * DIM;
      const half *src_token = post_x + idx * DIM;

      constexpr int multiplier = sizeof(fp16x8) / sizeof(half);  // 8
      constexpr int num_iters = DIM / (WARP_SIZE * multiplier);

      // fuse simulated MoE to combine
      for (int iter = 0; iter < num_iters; iter++) {
        const int e_idx = (iter * WARP_SIZE + lane_id) * multiplier;
        fp16x8 data = reinterpret_cast<const fp16x8 *>(src_token + e_idx)[0];
        for (int i = 0; i < multiplier; i++)
          data[i] = static_cast<_Float16>(static_cast<float>(data[i]) * moe_w);
        reinterpret_cast<fp16x8 *>(dst_token + e_idx)[0] = data;
      }

      // DIM = 2880
      if constexpr ((DIM % (WARP_SIZE * multiplier)) > 0) {
        const int start_idx = num_iters * (WARP_SIZE * multiplier) + lane_id;
        for (int e_idx = start_idx; e_idx < DIM; e_idx += WARP_SIZE)
          dst_token[e_idx] = __float2half(__half2float(src_token[e_idx]) * moe_w);
      }

      // signal done at flat_pos
      if (lane_id == 0) {
        int *flag_addr = translate(comm_flag, heap_bases[local_rank], heap_bases[src_rank]);
        flag_addr += flat_pos;
        __hip_atomic_store(flag_addr, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);  // store-release
        //printf("rank %d - combine-send: src_rank=%d, pos=%d, k=%d - set flag\n", local_rank, src_rank, flat_pos / topk, flat_pos % topk);
      }

      if constexpr (DO_PROFILE) if (bid < 5 && tid == 0) profile_stop(p2p_state.profile, e_id, 4, bid);
    }
  };

  auto do_recv = [&](int bid, int num_blocks) {
    // RECV stage
    // each warp handle 1 token, accumulate in registers
    for (int pos = bid * NUM_WARPS + warp_id; pos < num_tokens; pos += num_blocks * NUM_WARPS) {
      int e_id;
      if constexpr (DO_PROFILE) if (bid < 5 && tid == 0) e_id = profile_start(p2p_state.profile);

      static_assert(DIM % WARP_SIZE == 0);
      float acc[DIM / WARP_SIZE] = {};

      for (int k = 0; k < topk; k++) {
        const int flat_pos = pos * topk + k;
        const float w = weights[flat_pos];

        // wait for arrival. also check if we are within bounds
        if (lane_id == 0) {
          //printf("rank %d - combine-recv: rank=%d, pos=%d, k=%d - waiting\n", local_rank, local_rank, pos, k);
          spin_lock_system(comm_flag + flat_pos);
          //printf("rank %d - combine-recv: rank=%d, pos=%d, k=%d - arrived\n", local_rank, local_rank, pos, k);
        }
        __builtin_amdgcn_wave_barrier(); // equivalent to __syncwarp()

        // load 8x fp16 elements from comm buffer, then accumulate to register
        constexpr int multiplier = sizeof(fp16x8) / sizeof(half);  // 8
        constexpr int num_iters2 = DIM / (WARP_SIZE * multiplier);

        for (int iter2 = 0; iter2 < num_iters2; iter2++) {
          const int idx = flat_pos * DIM + (iter2 * WARP_SIZE + lane_id) * multiplier;
          fp16x8 data = reinterpret_cast<const fp16x8 *>(comm_x + idx)[0];
          for (int i = 0; i < multiplier; i++)
            acc[iter2 * multiplier + i] += static_cast<float>(data[i]) * w;
        }

        // DIM = 2880
        if constexpr ((DIM % (WARP_SIZE * multiplier)) > 0) {
          const int start_idx = num_iters2 * (WARP_SIZE * multiplier) + lane_id;
          for (int idx = start_idx; idx < DIM; idx += WARP_SIZE) {
            half data = comm_x[flat_pos * DIM + idx];
            acc[idx / WARP_SIZE] += __half2float(data) * w;
          }
        }
      }

      // store to output
      // must match the pattern / layout we use above
      constexpr int multiplier = sizeof(fp16x8) / sizeof(half);  // 8
      constexpr int num_iters2 = DIM / (WARP_SIZE * multiplier);

      for (int iter2 = 0; iter2 < num_iters2; iter2++) {
        // pack data to 16 bytes
        fp16x8 data;
        for (int i = 0; i < multiplier; i++)
          data[i] = static_cast<_Float16>(acc[iter2 * multiplier + i]);

        const int idx = pos * DIM + (iter2 * WARP_SIZE + lane_id) * multiplier;
        reinterpret_cast<fp16x8 *>(pre_x + idx)[0] = data;
      }

      // DIM = 2880
      if constexpr ((DIM % (WARP_SIZE * multiplier)) > 0) {
        const int start_idx = num_iters2 * (WARP_SIZE * multiplier) + lane_id;
        for (int idx = start_idx; idx < DIM; idx += WARP_SIZE) {
          half data = __float2half(acc[idx / WARP_SIZE]);
          pre_x[pos * DIM + idx] = data;
        }
      }

      if constexpr (DO_PROFILE) if (bid < 5 && tid == 0) profile_stop(p2p_state.profile, e_id, 5, bid);
    }
  };

  if constexpr (MODE == 0) {
    // send only
    do_send(bid, num_blocks);
  }
  else if constexpr (MODE == 1) {
    // recv only
    do_recv(bid, num_blocks);
  }
  else if constexpr (MODE == 2) {
    // parallel
    // even blocks do send, odd blocks do recv -> overlap send and recv
    if (bid % 2 == 0) do_send(bid / 2, num_blocks / 2);
    else              do_recv(bid / 2, num_blocks / 2);
  }
  else if constexpr (MODE == 3) {
    // sequential
    do_send(bid, num_blocks);
    do_recv(bid, num_blocks);
  }
}

at::Tensor combine(
  const at::Tensor& weights,      // [num_tokens][topk]
  int64_t dim,
  int64_t num_experts,
  int64_t max_num_tokens,
  int64_t evt_i64
) {
  TORCH_CHECK(weights.is_contiguous());

  const int num_tokens = weights.size(0);
  const int topk       = weights.size(1);

  // pytorch requires i64, but we pass in i32
  const int dim_i32            = dim;
  const int num_experts_i32    = num_experts;
  const int max_num_tokens_i32 = max_num_tokens;

  auto weights_ptr = weights.data_ptr<float>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(P2P_STATE.rank);

  void *kernel_args[] = {(void *)&weights_ptr,
                         (void *)&num_tokens, (void *)&topk, (void *)&num_experts_i32,
                         (void *)&max_num_tokens_i32, (void *)&P2P_STATE};

  constexpr int NUM_WARPS = 4;
  const int grid = 304;  // this is VERY important
  const int tb_size = NUM_WARPS * WARP_SIZE;
  constexpr int mode = 2;
  constexpr bool DO_PROFILE = false;

#define my_dispatch(DIM) { \
  if constexpr (mode == 0) { \
    CUDA_CHECK(cudaLaunchKernel((void *)(combine_kernel<NUM_WARPS, DIM, 0, DO_PROFILE>), grid, tb_size, kernel_args, 0, stream));\
    if (evt_i64 != 0) { cudaEvent_t evt = reinterpret_cast<cudaEvent_t>(evt_i64); CUDA_CHECK(cudaEventRecord(evt, stream)); } \
    CUDA_CHECK(cudaLaunchKernel((void *)(combine_kernel<NUM_WARPS, DIM, 1, DO_PROFILE>), grid, tb_size, kernel_args, 0, stream));\
  } else if constexpr (mode == 1) { \
    if (evt_i64 != 0) { cudaEvent_t evt = reinterpret_cast<cudaEvent_t>(evt_i64); CUDA_CHECK(cudaEventRecord(evt, stream)); } \
    CUDA_CHECK(cudaLaunchKernel((void *)(combine_kernel<NUM_WARPS, DIM, 2, DO_PROFILE>), grid, tb_size, kernel_args, 0, stream));\
  } else if constexpr (mode == 2) { \
    if (evt_i64 != 0) { cudaEvent_t evt = reinterpret_cast<cudaEvent_t>(evt_i64); CUDA_CHECK(cudaEventRecord(evt, stream)); } \
    CUDA_CHECK(cudaLaunchKernel((void *)(combine_kernel<NUM_WARPS, DIM, 3, DO_PROFILE>), grid, tb_size, kernel_args, 0, stream));\
  } \
}

  if      (dim == 2048) my_dispatch(2048)
  else if (dim == 2880) my_dispatch(2880)
  else if (dim == 4096) my_dispatch(4096)
  else if (dim == 6144) my_dispatch(6144)
  else if (dim == 7168) my_dispatch(7168)
  else TORCH_CHECK(false, "Unsupported dim ", dim);

#undef my_dispatch

  // construct output tensor
  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, P2P_STATE.rank);
  return torch::from_blob(P2P_STATE.y, {num_tokens, dim}, options);
}

at::Tensor custom_kernel(
  const at::Tensor& x,        // [num_tokens, dim]
  const at::Tensor& indices,  // [num_tokens, topk]
  const at::Tensor& weights,  // [num_tokens, topk]
  int64_t num_experts,
  int64_t max_num_tokens
) {
  const int dim = x.size(1);
  dispatch(x, indices, num_experts, max_num_tokens, 0);
  return combine(weights, dim, num_experts, max_num_tokens, 0);
}

TORCH_LIBRARY(p2p_module, m) {
  m.def("malloc_with_flags(int size, int flag) -> Tensor");
  m.impl("malloc_with_flags", &malloc_with_flags);

  m.def("get_ipc_handle(Tensor x) -> Tensor");
  m.impl("get_ipc_handle", &get_ipc_handle);

  m.def("open_ipc_handle(Tensor handle) -> int");
  m.impl("open_ipc_handle", &open_ipc_handle);

  m.def("close_ipc_handle(int addr) -> ()");
  m.impl("close_ipc_handle", &close_ipc_handle);

  m.def("p2p_state_init(int rank, Tensor heap_bases) -> ()");
  m.impl("p2p_state_init", &p2p_state_init);

  m.def("p2p_state_destroy() -> ()");
  m.impl("p2p_state_destroy", &p2p_state_destroy);

  m.def("p2p_state_get_profile() -> Tensor");
  m.impl("p2p_state_get_profile", &p2p_state_get_profile);

  m.def("dispatch(Tensor pre_x, Tensor indices, int num_experts, int max_num_tokens, int evt) -> ()");
  m.impl("dispatch", &dispatch);

  m.def("combine(Tensor weights, int dim, int num_experts, int max_num_tokens, int evt) -> Tensor");
  m.impl("combine", &combine);

  m.def("custom_kernel(Tensor x, Tensor indices, Tensor weights, int num_experts, int max_num_tokens) -> Tensor");
  m.impl("custom_kernel", &custom_kernel);
}
"""

load_inline(
    "p2p_module",
    cpp_sources=[""],
    cuda_sources=[cuda_src],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-save-temps", "-g"],
    # extra_cuda_cflags=["-O3"],
    verbose=True,
    is_python_module=False,
    no_implicit_headers=True,
)
ops = torch.ops.p2p_module


class P2PState:
    def __init__(self, rank: int, world_size: int, size: int = 10 << 30) -> None:
        # print(f"{rank=}: Start new allocation", file=sys.stderr, flush=True)

        # allocation of new heap
        torch.cuda.set_device(rank)
        # heap = torch.empty(size, dtype=torch.int8, device="cuda")
        finegrained = 0x1
        uncached = 0x3
        heap = ops.malloc_with_flags(size, finegrained).zero_()
        assert heap.device.index == rank

        handle = ops.get_ipc_handle(heap).cuda()
        all_handles = torch.empty(world_size, 64, dtype=torch.int8, device="cuda")
        dist.all_gather_into_tensor(all_handles.view(-1), handle)
        assert (all_handles[rank] == handle).all()

        all_handles = all_handles.cpu()
        heap_bases = [heap.data_ptr() if i == rank else ops.open_ipc_handle(all_handles[i]) for i in range(world_size)]
        heap_bases = torch.tensor(heap_bases, dtype=torch.int64, device="cuda")

        self.rank = rank
        self.world_size = world_size
        self.heap = heap
        self.heap_bases = heap_bases
        self.size = size

        ops.p2p_state_init(rank, heap_bases)

        # make sure everyone finishes initialization, especially for zero_(), before proceeding
        torch.cuda.synchronize()
        dist.barrier()

    def close(self):
        # print(f"{self.rank=}: Close IPC handles", file=sys.stderr, flush=True)

        torch.cuda.set_device(self.rank)
        ops.p2p_state_destroy()
        for i, base in enumerate(self.heap_bases.tolist()):
            if i != self.rank:
                ops.close_ipc_handle(base)

    def malloc_symmetric(self, shape: tuple[int, ...], dtype: torch.dtype, alignment: int = 128) -> Tensor:
        start = triton.cdiv(self.ptr, alignment) * alignment
        end = start + math.prod(shape) * dtype.itemsize
        assert end <= self.size
        out = self.heap[start:end].view(dtype).view(shape)
        self.ptr = end
        return out

    @staticmethod
    def malloc_finegrained(shape: tuple[int, ...], dtype: torch.dtype) -> Tensor:
        size = math.prod(shape) * dtype.itemsize
        finegrained = 0x1
        uncached = 0x3
        return ops.malloc_with_flags(size, finegrained).view(dtype).view(shape)

    def dispatch(self, cfg: MoEConfig, pre_x: Tensor, indices: Tensor, evt: torch.cuda.Event | None = None):
        # print(f"{self.rank=}: dispatch")

        evt_i64 = evt.cuda_event if evt is not None else 0
        ops.dispatch(pre_x, indices, cfg.num_experts, cfg.max_num_tokens, evt_i64)

    def combine(self, cfg: MoEConfig, weights: Tensor, evt: torch.cuda.Event | None = None):
        # print(f"{self.rank=}: combine")

        evt_i64 = evt.cuda_event if evt is not None else 0
        return ops.combine(weights, cfg.hidden_dim, cfg.num_experts, cfg.max_num_tokens, evt_i64)


P2P_STATE: P2PState | None = None

original_init = dist.init_process_group
original_destroy = dist.destroy_process_group


def patched_init(*args, rank, world_size, **kwargs):
    original_init(*args, rank=rank, world_size=world_size, **kwargs)

    global P2P_STATE
    assert P2P_STATE is None
    P2P_STATE = P2PState(rank, world_size)


def patched_destroy():
    global P2P_STATE
    dist.barrier()
    P2P_STATE.close()
    P2P_STATE = None
    original_destroy()


dist.init_process_group = patched_init
dist.destroy_process_group = patched_destroy


def custom_kernel(data: input_t) -> output_t:
    cfg, rank_data, rank, world_size = data
    # x: (num_tokens, hidden_dim)
    # indices: (num_tokens, experts_per_token) - map token_id to list of experts
    # weights: (num_tokens, experts_per_token) -> for weighted average later

    return ops.custom_kernel(rank_data.x, rank_data.indices, rank_data.weights, cfg.num_experts, cfg.max_num_tokens)
    y = ops.custom_kernel(rank_data.x, rank_data.indices, rank_data.weights, cfg.num_experts, cfg.max_num_tokens)

    # if False:
    #     e = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    #     e[1].record()  # run this to initialize cuda event
    #     e[3].record()
    #     ops.p2p_state_get_profile()[0] = 0

    #     torch.zeros(1 << 30, dtype=torch.int8, device="cuda")  # clear L2 cache
    #     torch.cuda.synchronize()
    #     dist.barrier()
    #     e[0].record()
    #     P2P_STATE.dispatch(cfg, rank_data.x, rank_data.indices, e[1])
    #     e[2].record()
    #     P2P_STATE.combine(cfg, rank_data.weights, e[3])
    #     e[4].record()

    #     torch.cuda.synchronize()
    #     dist.barrier()
    #     names = ["dispatch-send", "dispatch-recv", "combine-send", "combine-recv"]
    #     for i in range(world_size):
    #         if i == rank:
    #             print(f"{rank=}: ", end="")
    #             for j, name in enumerate(names):
    #                 duration = e[j].elapsed_time(e[j + 1]) * 1e3
    #                 print(f"{name}={duration:.2f}, ", end="")
    #             total_time = e[0].elapsed_time(e[-1]) * 1e3
    #             print(f"{total_time=:.2f}")
    #         dist.barrier()

    #     event_tags = [
    #         "(dispatch) count send tokens",
    #         "(dispatch) wait recv tokens",
    #         "(dispatch) send",
    #         "(dispatch) recv",
    #         "(combine) send",
    #         "(combine) recv",
    #     ]
    #     data = ops.p2p_state_get_profile().tolist()
    #     num_events = data[0]

    #     # normalize the earliest timestamp to 0
    #     minval = min(data[1 + i * 4] for i in range(num_events))

    #     events = []
    #     for i in range(num_events):
    #         ts, dur, tag, tid = data[1 + i * 4 : 1 + (i + 1) * 4]
    #         ts -= minval
    #         name = event_tags[tag]
    #         events.append(dict(name=name, ph="X", ts=ts, dur=dur, pid=rank, tid=rank + tid))

    #     trace = dict(traceEvents=events)
    #     gzip.open(f"trace{rank}.json.gz", "w").write(json.dumps(trace).encode("utf-8"))
    #     dist.barrier()
    #     if rank == 0:
    #         events = sum(
    #             [json.loads(gzip.open(f"trace{i}.json.gz", "r").read())["traceEvents"] for i in range(world_size)], []
    #         )
    #         trace = dict(traceEvents=events)
    #         gzip.open("trace.json.gz", "w").write(json.dumps(trace).encode("utf-8"))

    # if False:
    #     torch.cuda.synchronize()
    #     dist.barrier()
    #     with torch.profiler.profile(with_stack=False) as prof:
    #         ops.custom_kernel(rank_data.x, rank_data.indices, rank_data.weights, cfg.num_experts, cfg.max_num_tokens)
    #         torch.cuda.synchronize()
    #         dist.barrier()
    #     prof.export_chrome_trace(f"a2a_rank{rank}.json.gz")
    #     raise

    return y
