#!POPCORN leaderboard amd-all2all
#!POPCORN gpu MI300x8

import faulthandler
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

template <typename T>
__device__ __host__
T *translate(T *ptr, int64_t src_base, int64_t dst_base) {
  static_assert(sizeof(ptr) == sizeof(int64_t));
  const int64_t offset = reinterpret_cast<int64_t>(ptr) - src_base;
  return reinterpret_cast<T *>(dst_base + offset);
}

constexpr int WARP_SIZE = 64;
constexpr int WORLD_SIZE = 8;

using i32x2 = int __attribute__((__vector_size__(2 * sizeof(int))));
using fp32x4 = float __attribute__((__vector_size__(4 * sizeof(float))));
using fp16x8 = _Float16 __attribute__((__vector_size__(8 * sizeof(_Float16))));

__device__ __host__
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

__device__
int spin_lock_system(int *addr) {
  int flag = 0;
  while ((flag = __hip_atomic_load(addr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM)) == 0) {}
  __builtin_nontemporal_store(0, addr);  // reset
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

template <int NUM_WARPS, int DIM, bool DO_SEND, bool DO_RECV>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE)
void dispatch_kernel(
  // send
  const half *pre_x,                    // [num_tokens, DIM]
  const int *indices,                   // [num_tokens, topk]
        int *counts_per_global_expert,  // [NUM_EXPERTS]
  // comms
        half *comm_x,                   // symmetric, [WORLD_SIZE][num_local_experts][max_num_tokens][DIM]
        int *comm_meta,                 // symmetric, [WORLD_SIZE][num_local_experts][max_num_tokens], flat_pos
        int *comm_counts,               // symmetric, [WORLD_SIZE][num_local_experts]
  // recv
        half *post_x,                   // [num_local_experts][max_recv_per_expert][DIM]
        int *post_meta,                 // [num_local_experts][max_recv_per_expert][2], src_rank and flat_pos
        int *post_counts,               // [num_local_experts]
  // shapes
  const int num_tokens,
  const int topk,
  const int num_experts,
  const int max_num_tokens,
  const int local_rank,
  const int64_t *heap_bases
) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;
  const int bid = blockIdx.x;
  const int num_blocks = gridDim.x;
  const int num_local_experts = num_experts / WORLD_SIZE;
  const int max_recv_per_expert = max_num_tokens * WORLD_SIZE;

  if constexpr (DO_SEND) {
    // SEND stage
    // each warp process 1 (flat) token
    for (int flat_pos = bid * NUM_WARPS + warp_id;
             flat_pos < num_tokens * topk;
             flat_pos += num_blocks * NUM_WARPS) {
      const int pos = flat_pos / topk;
      const int k = flat_pos % topk;

      const int global_expert_id = indices[flat_pos];
      const int dst_rank = global_expert_id / num_local_experts;
      const int dst_local_expert_id = global_expert_id % num_local_experts;

      // atomicAdd to get position of this token for src_rank->global_expert_id
      // NOTE: atomicAdd returns old
      int dst_pos;
      if (lane_id == 0)  // atomic add on lane0 only
        dst_pos = atomicAdd(counts_per_global_expert + global_expert_id, 1);
      dst_pos = __shfl(dst_pos, 0);  // warp-broadcast

      // copy data
      half *dst_comm_x = translate(comm_x, heap_bases[local_rank], heap_bases[dst_rank]);
      dst_comm_x += (local_rank * num_local_experts + dst_local_expert_id) * max_num_tokens * DIM;
      copy_token<DIM>(dst_comm_x + dst_pos * DIM, pre_x + pos * DIM, lane_id);

      // write metadata
      if (lane_id == 0) {
        int *dst_comm_meta = translate(comm_meta, heap_bases[local_rank], heap_bases[dst_rank]);
        dst_comm_meta += (local_rank * num_local_experts + dst_local_expert_id) * max_num_tokens;
        //printf("rank %d - dispatch-send: dst_rank=%d, flat_pos=%d, pos=%d, k=%d\n", local_rank, dst_rank, flat_pos, pos, k);
        __builtin_nontemporal_store(flat_pos, dst_comm_meta + dst_pos);
      }
    }

    // SIGNAL stage
    // each thread handles 1 global expert
    cooperative_groups::this_grid().sync();  // wait for all blocks to finish
    for (int global_expert_id = bid * TB_SIZE + tid;
             global_expert_id < num_experts;
             global_expert_id += num_blocks * TB_SIZE) {
      const int dst_rank = global_expert_id / num_local_experts;
      const int dst_local_expert_id = global_expert_id % num_local_experts;

      // signal with counts+1, so that >0 means signalled
      int *dst_comm_counts = translate(comm_counts, heap_bases[local_rank], heap_bases[dst_rank]);
      int *flag_addr = dst_comm_counts + local_rank * num_local_experts + dst_local_expert_id;

      // https://github.com/ROCm/rocSHMEM/blob/rocm-6.4.4/src/atomic.hpp
      const int counts = counts_per_global_expert[global_expert_id];
      const int flag = counts + 1;
      __hip_atomic_store(flag_addr, flag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);  // store-release

      // reset to restore invariance, since torch.zeros() is expensive
      counts_per_global_expert[global_expert_id] = 0;
    }
  }

  if constexpr (DO_RECV) {
    // RECV stage
    // each block processes 1 expert
    for (int expert_id = bid;
             expert_id < num_experts;
             expert_id += num_blocks) {
      int src_rank = expert_id / num_local_experts;
      int local_expert_id = expert_id % num_local_experts;

      __shared__ int num_recv_smem, start_pos_smem;

      // wait for signal flag on 1st thread
      if (tid == 0) {
        // load-acquire, spin
        //printf("rank %d - dispatch-recv: src_rank=%d, local_expert_id=%d - waiting\n", local_rank, src_rank, local_expert_id);
        int flag = spin_lock_system(comm_counts + expert_id);
        //printf("rank %d - dispatch-recv: src_rank=%d, local_expert_id=%d - arrived\n", local_rank, src_rank, local_expert_id);

        int num_recv = flag - 1;  // -1 to compensate

        // atomicAdd to find the start position of src_rank within the local_expert_id group
        int start_pos = atomicAdd(post_counts + local_expert_id, num_recv);

        // broadcast
        num_recv_smem = num_recv;
        start_pos_smem = start_pos;
      }

      __syncthreads();
      int num_recv = num_recv_smem;
      int start_pos = start_pos_smem;

      if (num_recv == 0)
        continue;

      // each warp handles 1 token
      half *this_comm_x = comm_x + expert_id * max_num_tokens * DIM;
      half *this_post_x = post_x + (local_expert_id * max_recv_per_expert + start_pos) * DIM;
      for (int pos = warp_id; pos < num_recv; pos += NUM_WARPS)
        copy_token<DIM>(this_post_x + pos * DIM, this_comm_x + pos * DIM, lane_id);

      // each thread handles 1 token
      int *this_comm_meta = comm_meta + expert_id * max_num_tokens;
      int *this_post_meta = post_meta + (local_expert_id * max_recv_per_expert + start_pos) * 2;
      for (int pos = tid; pos < num_recv; pos += TB_SIZE) {
        i32x2 tmp;
        tmp[0] = src_rank;
        tmp[1] = this_comm_meta[pos];
        reinterpret_cast<i32x2 *>(this_post_meta + pos * 2)[0] = tmp;
        //printf("rank %d - dispatch: src_rank=%d, pos=%d, k=%d\n", local_rank, src_rank, tmp[1] / topk, tmp[1] % topk);
      }
    }
  }
}

void dispatch(
  // send
  const at::Tensor& pre_x,                     // [num_tokens, DIM]
  const at::Tensor& indices,                   // [num_tokens, topk]
        at::Tensor& counts_per_global_expert,  // [num_experts]
  // comms
        at::Tensor& comm_x,                    // [WORLD_SIZE][num_local_experts][max_num_tokens][DIM]
        at::Tensor& comm_meta,                 // [WORLD_SIZE][num_local_experts][max_num_tokens]
        at::Tensor& comm_counts,               // [WORLD_SIZE][num_local_experts]
  // recv
        at::Tensor& post_x,                    // [num_local_experts][max_recv_per_expert][DIM]
        at::Tensor& post_meta,                 // [num_local_experts][max_recv_per_expert][2], src_rank and flat_pos
        at::Tensor& post_counts,               // [num_local_experts]
  int64_t num_tokens,
  int64_t dim,
  int64_t topk,
  int64_t num_experts,
  int64_t max_num_tokens,
  int64_t local_rank,
  const at::Tensor& heap_bases
) {
  TORCH_CHECK(pre_x.scalar_type() == at::kHalf);
  TORCH_CHECK(comm_x.scalar_type() == at::kHalf);
  TORCH_CHECK(post_x.scalar_type() == at::kHalf);

  TORCH_CHECK(pre_x.is_contiguous());
  TORCH_CHECK(indices.is_contiguous());
  TORCH_CHECK(counts_per_global_expert.is_contiguous());
  //
  TORCH_CHECK(comm_x.is_contiguous());
  TORCH_CHECK(comm_meta.is_contiguous());
  TORCH_CHECK(comm_counts.is_contiguous());
  //
  TORCH_CHECK(post_x.is_contiguous());
  TORCH_CHECK(post_meta.is_contiguous());
  TORCH_CHECK(post_counts.is_contiguous());
  //
  TORCH_CHECK(heap_bases.is_contiguous());

  // pytorch requires i64, but we pass in i32
  const int num_tokens_i32     = num_tokens;
  const int dim_i32            = dim;
  const int topk_i32           = topk;
  const int num_experts_i32    = num_experts;
  const int max_num_tokens_i32 = max_num_tokens;
  const int local_rank_i32     = local_rank;

  auto pre_x_ptr                    = reinterpret_cast<const half *>(pre_x.data_ptr());
  auto indices_ptr                  = indices.data_ptr<int>();
  auto counts_per_global_expert_ptr = counts_per_global_expert.data_ptr<int>();
  //
  auto comm_x_ptr                   = reinterpret_cast<half *>(comm_x.data_ptr());
  auto comm_meta_ptr                = comm_meta.data_ptr<int>();
  auto comm_counts_ptr              = comm_counts.data_ptr<int>();
  //
  auto post_x_ptr                   = reinterpret_cast<half *>(post_x.data_ptr());
  auto post_meta_ptr                = post_meta.data_ptr<int>();
  auto post_counts_ptr              = post_counts.data_ptr<int>();
  //
  auto heap_bases_ptr               = heap_bases.data_ptr<int64_t>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(local_rank);

  constexpr int NUM_WARPS = 8;
  const int grid = 128;
  const int tb_size = NUM_WARPS * WARP_SIZE;

  void *kernel_args[] = {(void *)&pre_x_ptr, (void *)&indices_ptr, (void *)&counts_per_global_expert_ptr,
                         (void *)&comm_x_ptr, (void *)&comm_meta_ptr, (void *)&comm_counts_ptr,
                         (void *)&post_x_ptr, (void *)&post_meta_ptr, (void *)&post_counts_ptr,
                         (void *)&num_tokens_i32, (void *)&topk_i32, (void *)&num_experts_i32,
                         (void *)&max_num_tokens_i32, (void *)&local_rank_i32, (void *)&heap_bases_ptr};

#define my_dispatch(DIM) { \
  CUDA_CHECK(cudaLaunchCooperativeKernel((void *)(dispatch_kernel<NUM_WARPS, DIM, true, false>), grid, tb_size, kernel_args, 0, stream)); \
  CUDA_CHECK(cudaLaunchCooperativeKernel((void *)(dispatch_kernel<NUM_WARPS, DIM, false, true>), grid, tb_size, kernel_args, 0, stream)); }

  if      (dim == 2048) my_dispatch(2048)
  else if (dim == 2880) my_dispatch(2880)
  else if (dim == 4096) my_dispatch(4096)
  else if (dim == 6144) my_dispatch(6144)
  else if (dim == 7168) my_dispatch(7168)
  else TORCH_CHECK(false, "Unsupported dim ", dim);

#undef my_dispatch
}

template <int NUM_WARPS, int DIM, bool DO_SEND, bool DO_RECV>
__global__
__launch_bounds__(NUM_WARPS * WARP_SIZE)
void combine_kernel(
  // send
  const half *post_x,        // [num_local_experts][max_recv_per_token][DIM]
  const int *post_meta,      // [num_local_experts][max_recv_per_token][2], src_rank, flat_pos
  const int *post_counts,    // [num_local_experts]
  // comms
        half *comm_x,        // symmetric, [max_num_tokens][topk][DIM]
        int *comm_flag,      // symmetric, [max_num_tokens][topk]
  // recv
        half *pre_x,         // [num_tokens][DIM]
  const float *weights,      // [num_tokens][topk]
  // shapes
  const int num_tokens,
  const int topk,
  const int num_experts,
  const int max_num_tokens,
  const int local_rank,
  const int64_t *heap_bases  // [WORLD_SIZE]
) {
  const int tid = threadIdx.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  const int bid = blockIdx.x;
  const int num_blocks = gridDim.x;

  const int num_local_experts = num_experts / WORLD_SIZE;
  const int max_recv_per_token = max_num_tokens * WORLD_SIZE;

  if constexpr (DO_SEND) {
    // SEND stage
    // each warp handle 1 token
    for (int idx = bid * NUM_WARPS + warp_id;
             idx < num_local_experts * max_recv_per_token;
             idx += num_blocks * NUM_WARPS) {
      const int local_expert_id = idx / max_recv_per_token;
      const int pos = idx % max_recv_per_token;

      if (pos >= post_counts[local_expert_id])
        continue;

      i32x2 tmp = reinterpret_cast<const i32x2 *>(post_meta + idx * 2)[0];
      const int src_rank = tmp[0];
      const int flat_pos = tmp[1];

      half *dst_comm_x = translate(comm_x, heap_bases[local_rank], heap_bases[src_rank]);
      copy_token<DIM>(dst_comm_x + flat_pos * DIM, post_x + idx * DIM, lane_id);

      // signal done at flat_pos
      if (lane_id == 0) {
        int *flag_addr = translate(comm_flag, heap_bases[local_rank], heap_bases[src_rank]);
        flag_addr += flat_pos;
        __hip_atomic_store(flag_addr, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);  // store-release
        //printf("rank %d - combine-send: src_rank=%d, pos=%d, k=%d - set flag\n", local_rank, src_rank, flat_pos / topk, flat_pos % topk);
      }
    }
  }

  if constexpr (DO_RECV) {
    // RECV stage
    // each warp handle 1 token, accumulate in registers
    // hence, each block handles NUM_WARPS tokens
    // since AMD doesn't have __syncwarp(), we have to write code for a block
    for (int iter = 0;
             iter < cdiv(num_tokens, num_blocks * NUM_WARPS);
             iter++) {
      const int pos = iter * (num_blocks * NUM_WARPS) + bid * NUM_WARPS + warp_id;
      const bool in_bounds = pos < num_tokens;

      static_assert(DIM % WARP_SIZE == 0);
      float acc[DIM / WARP_SIZE] = {};

      for (int k = 0; k < topk; k++) {
        const int flat_pos = pos * topk + k;

        // wait for arrival. also check if we are within bounds
        if (in_bounds && (lane_id == 0)) {
          //printf("rank %d - combine-recv: rank=%d, pos=%d, k=%d - waiting\n", local_rank, local_rank, pos, k);
          spin_lock_system(comm_flag + flat_pos);
          //printf("rank %d - combine-recv: rank=%d, pos=%d, k=%d - arrived\n", local_rank, local_rank, pos, k);
        }

        // this must be called regardless of whether we are in-bounds of off-bounds
        __syncthreads();

        if (!in_bounds)
          continue;

        const float w = weights[flat_pos];

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

      if (!in_bounds)
        continue;

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
    }
  }
}

void combine(
  // send
  const at::Tensor& post_y,       // [num_local_experts][max_recv_per_expert][DIM]
  const at::Tensor& post_meta,    // [num_local_experts][max_recv_per_expert][2]
  const at::Tensor& post_counts,  // [num_local_experts]
  // comms
        at::Tensor& comm_y,       // [max_num_tokens][topk][DIM]
        at::Tensor& comm_flag,    // [max_num_tokens][topk]
  // recv
        at::Tensor& pre_y,        // [num_tokens][DIM]
  const at::Tensor& weights,      // [num_tokens][topk]
  int64_t num_tokens,
  int64_t dim,
  int64_t topk,
  int64_t num_experts,
  int64_t max_num_tokens,
  int64_t local_rank,
  const at::Tensor& heap_bases
) {
  TORCH_CHECK(post_y.scalar_type() == at::kHalf);
  TORCH_CHECK(comm_y.scalar_type() == at::kHalf);
  TORCH_CHECK(pre_y.scalar_type() == at::kHalf);

  TORCH_CHECK(post_y.is_contiguous());
  TORCH_CHECK(post_meta.is_contiguous());
  TORCH_CHECK(post_counts.is_contiguous());
  //
  TORCH_CHECK(comm_y.is_contiguous());
  TORCH_CHECK(comm_flag.is_contiguous());
  //
  TORCH_CHECK(pre_y.is_contiguous());
  TORCH_CHECK(weights.is_contiguous());
  //
  TORCH_CHECK(heap_bases.is_contiguous());

  // pytorch requires i64, but we pass in i32
  const int num_tokens_i32     = num_tokens;
  const int dim_i32            = dim;
  const int topk_i32           = topk;
  const int num_experts_i32    = num_experts;
  const int max_num_tokens_i32 = max_num_tokens;
  const int local_rank_i32     = local_rank;

  auto post_y_ptr      = reinterpret_cast<const half *>(post_y.data_ptr());
  auto post_meta_ptr   = post_meta.data_ptr<int>();
  auto post_counts_ptr = post_counts.data_ptr<int>();
  //
  auto comm_y_ptr      = reinterpret_cast<half *>(comm_y.data_ptr());
  auto comm_flag_ptr   = comm_flag.data_ptr<int>();
  //
  auto pre_y_ptr       = reinterpret_cast<half *>(pre_y.data_ptr());
  auto weights_ptr     = weights.data_ptr<float>();
  //
  auto heap_bases_ptr  = heap_bases.data_ptr<int64_t>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(local_rank);

  constexpr int NUM_WARPS = 8;
  const int grid = 128;
  const int tb_size = NUM_WARPS * WARP_SIZE;

  void *kernel_args[] = {(void *)&post_y_ptr, (void *)&post_meta_ptr, (void *)&post_counts_ptr,
                         (void *)&comm_y_ptr, (void *)&comm_flag_ptr,
                         (void *)&pre_y_ptr, (void *)&weights_ptr,
                         (void *)&num_tokens_i32, (void *)&topk_i32, (void *)&num_experts_i32,
                         (void *)&max_num_tokens_i32, (void *)&local_rank_i32, (void *)&heap_bases_ptr};

#define my_dispatch(DIM) { \
  CUDA_CHECK(cudaLaunchCooperativeKernel((void *)(combine_kernel<NUM_WARPS, DIM, true, false>), grid, tb_size, kernel_args, 0, stream));\
  CUDA_CHECK(cudaLaunchCooperativeKernel((void *)(combine_kernel<NUM_WARPS, DIM, false, true>), grid, tb_size, kernel_args, 0, stream)); }

  if      (dim == 2048) my_dispatch(2048)
  else if (dim == 2880) my_dispatch(2880)
  else if (dim == 4096) my_dispatch(4096)
  else if (dim == 6144) my_dispatch(6144)
  else if (dim == 7168) my_dispatch(7168)
  else TORCH_CHECK(false, "Unsupported dim ", dim);

#undef my_dispatch
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

  m.def("dispatch(Tensor pre_x, Tensor indices, Tensor(a!) counts_per_global_expert, Tensor(b!) comm_x, Tensor(c!) comm_meta, Tensor(d!) comm_counts, Tensor(e!) post_x, Tensor(f!) post_meta, Tensor(g!) post_counts, int num_tokens, int dim, int topk, int num_experts, int max_num_tokens, int local_rank, Tensor heap_bases) -> ()");
  m.impl("dispatch", &dispatch);

  m.def("combine(Tensor post_y, Tensor post_meta, Tensor post_counts, Tensor(a!) comm_x, Tensor(b!) comm_flag, Tensor(c!) pre_y, Tensor weights, int num_tokens, int dim, int topk, int num_experts, int max_num_tokens, int local_rank, Tensor heap_bases) -> ()");
  m.impl("combine", &combine);
}
"""

load_inline(
    "p2p_module",
    cpp_sources=[""],
    cuda_sources=[cuda_src],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
    is_python_module=False,
    no_implicit_headers=True,
)
ops = torch.ops.p2p_module


class P2PState:
    def __init__(self, rank: int, world_size: int, size: int = 1 << 30) -> None:
        # print(f"{rank=}: Start new allocation", file=sys.stderr, flush=True)

        # allocation of new heap
        torch.cuda.set_device(rank)
        # heap = torch.empty(size, dtype=torch.int8, device="cuda")
        finegrained = 0x1
        uncached = 0x3
        heap = ops.malloc_with_flags(size, finegrained)
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

        self.ptr = 0

        # stats for largest problem size
        num_experts = 256
        num_local_experts = num_experts // world_size
        topk = 8
        max_num_tokens = 256
        dim = 7168

        # dispatch
        self.dispatch_comm_x = self.malloc_symmetric(
            (world_size * num_local_experts * max_num_tokens * dim,), dtype=torch.float16
        )
        self.dispatch_comm_meta = self.malloc_symmetric(
            (world_size * num_local_experts * max_num_tokens,), dtype=torch.int32
        )
        self.dispatch_comm_counts = self.malloc_symmetric((world_size * num_local_experts,), dtype=torch.int32)
        self.dispatch_comm_counts.zero_()

        # allocating zeros is expensive in the hot loop
        self.dispatch_counts_per_expert = torch.zeros(num_experts, dtype=torch.int32, device="cuda")

        # combine
        self.combine_comm_x = self.malloc_symmetric((max_num_tokens * topk * dim,), dtype=torch.float16)
        self.combine_comm_flag = self.malloc_symmetric((max_num_tokens * topk,), dtype=torch.int32)
        self.combine_comm_flag.zero_()

        # make sure everyone finishes initialization, especially for zero_(), before proceeding
        dist.barrier()

    def close(self):
        # print(f"{self.rank=}: Close IPC handles", file=sys.stderr, flush=True)

        torch.cuda.set_device(self.rank)
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

    def dispatch(self, cfg: MoEConfig, pre_x: Tensor, indices: Tensor):
        # print(f"{self.rank=}: dispatch")
        self.num_tokens, dim = pre_x.shape

        num_local_experts = cfg.num_experts // self.world_size
        max_recv_per_expert = cfg.max_num_tokens * self.world_size  # everyone routes to a single expert

        post_x = pre_x.new_empty((num_local_experts, max_recv_per_expert, dim))
        post_meta = torch.empty((num_local_experts, max_recv_per_expert, 2), dtype=torch.int32, device="cuda")
        post_counts = torch.zeros((num_local_experts,), dtype=torch.int32, device="cuda")

        ops.dispatch(
            # send
            pre_x,
            indices,
            self.dispatch_counts_per_expert,
            # comms
            self.dispatch_comm_x,
            self.dispatch_comm_meta,
            self.dispatch_comm_counts,
            # recv
            post_x,
            post_meta,
            post_counts,
            # others
            self.num_tokens,
            cfg.hidden_dim,
            cfg.experts_per_token,
            cfg.num_experts,
            cfg.max_num_tokens,
            self.rank,
            self.heap_bases,
        )

        return post_x, post_meta, post_counts

    def combine(self, cfg: MoEConfig, post_y: Tensor, post_meta: Tensor, post_counts: Tensor, weights: Tensor):
        # print(f"{self.rank=}: combine")
        pre_y = post_y.new_empty(self.num_tokens, post_y.shape[-1])

        ops.combine(
            # send
            post_y,
            post_meta,
            post_counts,
            # comms
            self.combine_comm_x,
            self.combine_comm_flag,
            # recv
            pre_y,
            weights,
            # others
            self.num_tokens,
            cfg.hidden_dim,
            cfg.experts_per_token,
            cfg.num_experts,
            cfg.max_num_tokens,
            self.rank,
            self.heap_bases,
        )

        return pre_y


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

    # MoE. TODO: interleave with dispatch and combine
    post_x, post_meta, post_counts = P2P_STATE.dispatch(cfg, rank_data.x, rank_data.indices)
    post_y = post_x.to(cfg.out_dtype) * (1 + rank)
    y = P2P_STATE.combine(cfg, post_y, post_meta, post_counts, rank_data.weights)

    if False:
        torch.cuda.synchronize()
        dist.barrier()  # TODO: eliminate  barrier
        with torch.profiler.profile(with_stack=True) as prof:
            post_x, post_meta, post_counts = P2P_STATE.dispatch(cfg, rank_data.x, rank_data.indices)
            post_y = post_x.to(cfg.out_dtype) * (1 + rank)
            P2P_STATE.combine(cfg, post_y, post_meta, post_counts, rank_data.weights)
        prof.export_chrome_trace(f"a2a_rank{rank}.json.gz")
        raise

    return y
