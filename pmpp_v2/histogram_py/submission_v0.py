#!POPCORN leaderboard histogram_v2

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
constexpr int WARP_SIZE = 32;
constexpr int NUM_BINS = 256;
constexpr int NUM_WARPS = 8;
constexpr int TB_SIZE = NUM_WARPS * WARP_SIZE;

__device__ __host__
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

__align__(16)
struct u8x16 { uint8_t x[16]; };

__global__
void kernel(
  const uint8_t *data_ptr,    // (size,)
        int64_t *output_ptr,  // (256,)
        int64_t size) {

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int num_blocks = gridDim.x;
  const int warp_id = tid / WARP_SIZE;
  const int lane_id = tid % WARP_SIZE;

  __shared__ int smem_hist[NUM_WARPS * NUM_BINS];

  // init
  for (int iter_id = 0; iter_id < (NUM_WARPS * NUM_BINS / TB_SIZE); iter_id++)
    smem_hist[iter_id * TB_SIZE + tid] = 0;
  __syncthreads();

  // each thread reads 16 elems
  const u8x16 *data_u8x16_ptr = reinterpret_cast<const u8x16 *>(data_ptr);
  data_u8x16_ptr += bid * TB_SIZE + tid;

  const int wave_size = (16 * TB_SIZE * num_blocks);
  const int num_iters = size / wave_size;
  for (int iter_id = 0; iter_id < num_iters; iter_id ++) {
    const u8x16 x = data_u8x16_ptr[0];
    data_u8x16_ptr += num_blocks * TB_SIZE;

    for (int elem_id = 0; elem_id < 16; elem_id++) {
      const int val = x.x[elem_id];  // cast u8->i32
      atomicAdd(smem_hist + (warp_id * NUM_BINS + val), 1);
    }
  }

  // each thread reads 1 elem
  const int start = num_iters * wave_size + bid * TB_SIZE + tid;
  for (int i = start; i < size; i += num_blocks * TB_SIZE) {
    const int val = data_ptr[i];
    atomicAdd(smem_hist + (warp_id * NUM_BINS + val), 1);
  }

  __syncthreads();

  // combine histogram across warps
  static_assert(NUM_BINS % TB_SIZE == 0);
  for (int iter_id = 0; iter_id < NUM_BINS / TB_SIZE; iter_id++) {
    const int bin_id = iter_id * TB_SIZE + tid;
    int count = smem_hist[bin_id];  // from 1st sub-histogram

    for (int sub_id = 1; sub_id < NUM_WARPS; sub_id++)
      count += smem_hist[sub_id * NUM_BINS + bin_id];

    // total count shouldn't exceed int32...
    atomicAdd(reinterpret_cast<int *>(output_ptr + bin_id), count);
  }
}

void launch(const at::Tensor& data, at::Tensor& output) {
  output.zero_();

  const auto data_ptr = data.data_ptr<uint8_t>();
  auto output_ptr = output.data_ptr<int64_t>();
  const int64_t size = data.size(0);

  const int num_blocks = 264;
  kernel<<<num_blocks, TB_SIZE>>>(data_ptr, output_ptr, size);
}

TORCH_LIBRARY(my_module, m) {
  m.def("launch(Tensor data, Tensor(a!) output) -> ()");
  m.impl("launch", &launch);
}
"""

load_inline(
    "histogram_v0",
    cpp_sources="",
    cuda_sources=CUDA_SRC,
    verbose=True,
    is_python_module=False,
)


def custom_kernel(data: input_t) -> output_t:
    data, output = data
    torch.ops.my_module.launch(data, output)
    return output
