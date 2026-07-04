from pathlib import Path

import torch
import triton
import triton.language as tl
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

CPP_SRC = r"""
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContextLight.h>
#include <cublasLt.h>
#include <cstdint>
#include <torch/library.h>
#include <optional>

void launch_gmem_panel(
    const float* input,
    float* output,
    float* tau,
    float* v_fp32,
    void* v_fp16,
    int* flags,
    int batch,
    int rows,
    int cols,
    int n);
void launch_panelMN(
    const float* input,
    float* output,
    float* tau,
    float* v_fp32,
    void* v_fp16,
    int batch,
    int rows,
    int cols,
    int n);
void launch_build_t96_diag(
    const float* gram,
    const float* tau,
    float* output,
    int64_t tau_stride,
    int batch);
void launch_build_t128_diag(
    const float* gram,
    const float* tau,
    float* output,
    int64_t tau_stride,
    int batch);

namespace {

cublasLtMatrixLayout_t make_lt_layout(
    const at::Tensor& t,
    cudaDataType_t dtype) {
  TORCH_CHECK(t.dim() == 3);

  const int batch = static_cast<int>(t.size(0));
  const int64_t rows = t.size(1);
  const int64_t cols = t.size(2);

  cublasLtOrder_t order;
  int64_t ld;

  if (t.stride(2) == 1) {
    order = CUBLASLT_ORDER_ROW;
    ld = t.stride(1);
  } else if (t.stride(1) == 1) {
    order = CUBLASLT_ORDER_COL;
    ld = t.stride(2);
  } else {
    TORCH_CHECK(false, "tensor must be row-major or column-major, strides=", t.strides());
  }

  cublasLtMatrixLayout_t layout = nullptr;
  auto status = cublasLtMatrixLayoutCreate(&layout, dtype, rows, cols, ld);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "layout create failed: ", status);

  status = cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "set order failed: ", status);

  status = cublasLtMatrixLayoutSetAttribute(
      layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch));
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "set batch count failed: ", status);

  const int64_t batch_stride = t.stride(0);
  status = cublasLtMatrixLayoutSetAttribute(
      layout,
      CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET,
      &batch_stride,
      sizeof(batch_stride));
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "set batch stride failed: ", status);

  return layout;
}

void destroy_lt_layouts(std::initializer_list<cublasLtMatrixLayout_t> layouts) {
  for (auto layout : layouts) {
    if (layout) cublasLtMatrixLayoutDestroy(layout);
  }
}

void lt_baddbmm_out(
    const at::Tensor& input,
    const at::Tensor& left,
    const at::Tensor& right,
    at::Tensor& output,
    cudaDataType_t ab_dtype,
    cublasComputeType_t compute_type,
    float alpha,
    float beta) {
  TORCH_CHECK(input.dim() == 3);
  TORCH_CHECK(left.dim() == 3);
  TORCH_CHECK(right.dim() == 3);
  TORCH_CHECK(output.dim() == 3);

  TORCH_CHECK(left.size(0) == right.size(0));
  TORCH_CHECK(left.size(0) == input.size(0));
  TORCH_CHECK(left.size(0) == output.size(0));
  TORCH_CHECK(left.size(2) == right.size(1));

  TORCH_CHECK(input.size(1) == left.size(1));
  TORCH_CHECK(input.size(2) == right.size(2));
  TORCH_CHECK(output.size(1) == input.size(1));
  TORCH_CHECK(output.size(2) == input.size(2));

  TORCH_CHECK(input.dtype() == at::kFloat);
  TORCH_CHECK(output.dtype() == at::kFloat);

  cublasLtHandle_t handle = at::cuda::getCurrentCUDABlasLtHandle();

  cublasLtMatmulDesc_t op = nullptr;
  auto status = cublasLtMatmulDescCreate(&op, compute_type, CUDA_R_32F);
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "matmul desc create failed: ", status);

  auto a_layout = make_lt_layout(left, ab_dtype);
  auto b_layout = make_lt_layout(right, ab_dtype);
  auto c_layout = make_lt_layout(input, CUDA_R_32F);
  auto d_layout = make_lt_layout(output, CUDA_R_32F);

  status = cublasLtMatmul(
      handle,
      op,
      &alpha,
      left.data_ptr(),
      a_layout,
      right.data_ptr(),
      b_layout,
      &beta,
      input.data_ptr<float>(),
      c_layout,
      output.data_ptr<float>(),
      d_layout,
      nullptr,
      nullptr,
      0,
      0);

  destroy_lt_layouts({d_layout, c_layout, b_layout, a_layout});
  if (op) cublasLtMatmulDescDestroy(op);

  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, "cublasLtMatmul failed: ", status);
}

} // namespace

void tf32_baddbmm_out(
    const at::Tensor& input,
    const at::Tensor& left,
    const at::Tensor& right,
    at::Tensor& output,
    double beta,
    double alpha) {
  TORCH_CHECK(left.dtype() == at::kFloat);
  TORCH_CHECK(right.dtype() == at::kFloat);

  lt_baddbmm_out(
      input,
      left,
      right,
      output,
      CUDA_R_32F,
      CUBLAS_COMPUTE_32F_FAST_TF32,
      static_cast<float>(alpha),
      static_cast<float>(beta));
}

void bf16x9_baddbmm_out(
    const at::Tensor& input,
    const at::Tensor& left,
    const at::Tensor& right,
    at::Tensor& output,
    double beta,
    double alpha) {
  TORCH_CHECK(left.dtype() == at::kFloat);
  TORCH_CHECK(right.dtype() == at::kFloat);

  lt_baddbmm_out(
      input,
      left,
      right,
      output,
      CUDA_R_32F,
      CUBLAS_COMPUTE_32F_EMULATED_16BFX9,
      static_cast<float>(alpha),
      static_cast<float>(beta));
}

void fp16_baddbmm_out(
    const at::Tensor& input,
    const at::Tensor& left,
    const at::Tensor& right,
    at::Tensor& output,
    double beta,
    double alpha) {
  TORCH_CHECK(left.dtype() == at::kHalf);
  TORCH_CHECK(right.dtype() == at::kHalf);

  lt_baddbmm_out(
      input,
      left,
      right,
      output,
      CUDA_R_16F,
      CUBLAS_COMPUTE_32F,
      static_cast<float>(alpha),
      static_cast<float>(beta));
}

at::Tensor build_t128_diag(const at::Tensor& gram, const at::Tensor& tau) {
  TORCH_CHECK(gram.dim() == 3);
  TORCH_CHECK(tau.dim() == 2);
  TORCH_CHECK(gram.size(1) == 128);
  TORCH_CHECK(gram.size(2) == 128);
  TORCH_CHECK(tau.size(0) == gram.size(0));
  TORCH_CHECK(tau.size(1) >= 128);
  TORCH_CHECK(gram.dtype() == at::kFloat);
  TORCH_CHECK(tau.dtype() == at::kFloat);
  TORCH_CHECK(gram.is_cuda());
  TORCH_CHECK(tau.is_cuda());
  auto output = at::empty_like(gram);
  launch_build_t128_diag(
      gram.data_ptr<float>(),
      tau.data_ptr<float>(),
      output.data_ptr<float>(),
      tau.stride(0),
      static_cast<int>(gram.size(0)));
  return output;
}

at::Tensor build_t96_diag(const at::Tensor& gram, const at::Tensor& tau) {
  TORCH_CHECK(gram.dim() == 3);
  TORCH_CHECK(tau.dim() == 2);
  TORCH_CHECK(gram.size(1) == 96);
  TORCH_CHECK(gram.size(2) == 96);
  TORCH_CHECK(tau.size(0) == gram.size(0));
  TORCH_CHECK(tau.size(1) >= 96);
  TORCH_CHECK(gram.dtype() == at::kFloat);
  TORCH_CHECK(tau.dtype() == at::kFloat);
  TORCH_CHECK(gram.is_cuda());
  TORCH_CHECK(tau.is_cuda());
  auto output = at::empty_like(gram);
  launch_build_t96_diag(
      gram.data_ptr<float>(),
      tau.data_ptr<float>(),
      output.data_ptr<float>(),
      tau.stride(0),
      static_cast<int>(gram.size(0)));
  return output;
}

void gmem_panel(
  const at::Tensor& input,
  at::Tensor& output,
  at::Tensor& tau,
  at::Tensor& v_fp32,
  at::Tensor& v_fp16
) {
  const int batch = input.size(0);
  const int rows = input.size(1);
  const int cols = input.size(2);
  const int n = input.stride(1);
  TORCH_CHECK(input.stride(0) == n * n);
  TORCH_CHECK(output.stride(0) == n * n);
  TORCH_CHECK(output.stride(1) == n);
  TORCH_CHECK(tau.stride(0) == n);

  // TODO: reuseable flag buffer
  auto flags = at::zeros({batch, cols}, input.options().dtype(at::kInt));

  launch_gmem_panel(
    input.data_ptr<float>(), output.data_ptr<float>(),
    tau.data_ptr<float>(), v_fp32.data_ptr<float>(),
    v_fp16.data_ptr(), flags.data_ptr<int>(), batch, rows,
    cols, n);
}

void panelMN(
  const at::Tensor& input,
  at::Tensor& output,
  at::Tensor& tau,
  std::optional<at::Tensor> v_fp32,
  std::optional<at::Tensor> v_fp16
) {
  int batch = input.size(0);
  int rows = input.size(1);
  int cols = input.size(2);
  int n = input.stride(1);
  TORCH_CHECK(input.stride(0) == n * n);
  TORCH_CHECK(output.stride(0) == n * n);
  TORCH_CHECK(output.stride(1) == n);
  TORCH_CHECK(tau.stride(0) == n);

  float *v_fp32_ptr = nullptr;
  if (v_fp32.has_value()) {
    v_fp32_ptr = v_fp32->data_ptr<float>();
    TORCH_CHECK(v_fp32->stride(0) == rows * cols);
    TORCH_CHECK(v_fp32->stride(1) == 1);
    TORCH_CHECK(v_fp32->stride(2) == rows);
  }
  void *v_fp16_ptr = nullptr;
  if (v_fp16.has_value()) {
    v_fp16_ptr = v_fp16->data_ptr();
    TORCH_CHECK(v_fp16->stride(0) == rows * cols);
    TORCH_CHECK(v_fp16->stride(1) == 1);
    TORCH_CHECK(v_fp16->stride(2) == rows);
  }

  launch_panelMN(
      input.data_ptr<float>(),
      output.data_ptr<float>(),
      tau.data_ptr<float>(),
      v_fp32_ptr,
      v_fp16_ptr,
      batch, rows, cols, n);
}

TORCH_LIBRARY(codex, m) {
  m.def("gmem_panel(Tensor input, Tensor(a!) output, Tensor(b!) tau, Tensor(c!) v_fp32, Tensor(d!) v_fp16) -> ()");
  m.impl("gmem_panel", &gmem_panel);
  m.def("panelMN(Tensor input, Tensor(a!) output, Tensor(b!) tau, Tensor(c!)? v_fp32=None, Tensor(d!)? v_fp16=None) -> ()");
  m.impl("panelMN", &panelMN);
  m.def("tf32_baddbmm_out(Tensor input, Tensor left, Tensor right, Tensor(a!) output, float beta=1.0, float alpha=1.0) -> ()");
  m.impl("tf32_baddbmm_out", &tf32_baddbmm_out);
  m.def("bf16x9_baddbmm_out(Tensor input, Tensor left, Tensor right, Tensor(a!) output, float beta=1.0, float alpha=1.0) -> ()");
  m.impl("bf16x9_baddbmm_out", &bf16x9_baddbmm_out);
  m.def("fp16_baddbmm_out(Tensor input, Tensor left, Tensor right, Tensor(a!) output, float beta=1.0, float alpha=1.0) -> ()");
  m.impl("fp16_baddbmm_out", &fp16_baddbmm_out);
  m.def("build_t128_diag(Tensor gram, Tensor tau) -> Tensor");
  m.impl("build_t128_diag", &build_t128_diag);
  m.def("build_t96_diag(Tensor gram, Tensor tau) -> Tensor");
  m.impl("build_t96_diag", &build_t96_diag);
}
"""

CUDA_SRC = r"""
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cuda_runtime.h>

__device__ __host__
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

constexpr unsigned FULL_MASK = 0xffffffffu;

__device__ __forceinline__
float warp_sum(float value, int size = 32) {
  #pragma unroll
  for (int offset = size / 2; offset > 0; offset >>= 1)
    value += __shfl_xor_sync(FULL_MASK, value, offset);
  return value;
}

template <int vec>
__device__ inline
void ldg_f32(float* dst, const float* src) {
  if constexpr (vec == 4)
    asm volatile("ld.global.relaxed.cta.L1::no_allocate.v4.f32 {%0, %1, %2, %3}, [%4];"
                : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
                : "l"(src));
  if constexpr (vec == 8)
    asm volatile("ld.global.relaxed.cta.L1::no_allocate.v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
                  "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
                : "l"(src));
}

template <int vec>
__device__ inline
void stg_f32(float* dst, const float* src) {
  if constexpr (vec == 2)
    asm volatile("st.global.relaxed.cta.L1::no_allocate.v2.f32 [%0], {%1, %2};"
                :: "l"(dst),
                  "f"(src[0]), "f"(src[1]));
  if constexpr (vec == 4)
    asm volatile("st.global.relaxed.cta.L1::no_allocate.v4.f32 [%0], {%1, %2, %3, %4};"
                :: "l"(dst),
                  "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]));
  if constexpr (vec == 8)
    asm volatile("st.global.relaxed.cta.L1::no_allocate.v8.f32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "l"(dst),
                  "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]),
                  "f"(src[4]), "f"(src[5]), "f"(src[6]), "f"(src[7]));
}

__device__ inline
float sqrt_approx(float value) {
  float result;
  asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ inline
float rcp_approx(float value) {
  float result;
  asm volatile("rcp.approx.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ inline
void fma_f32x2(float* accumulator, const float* left, const float* right) {
  asm volatile(
    "{"
    ".reg .b64 a, b, c, d;\n"
    "mov.b64 c, {%0, %1};\n"
    "mov.b64 a, {%2, %3};\n"
    "mov.b64 b, {%4, %5};\n"
    "fma.rn.f32x2 d, a, b, c;\n"
    "mov.b64 {%0, %1}, d;\n"
    "}"
    : "+f"(accumulator[0]), "+f"(accumulator[1])
    : "f"(left[0]), "f"(left[1]), "f"(right[0]), "f"(right[1]));
}

template <int STRIDE>
__device__ inline
void build_t32_inverse_block(
    const float* lower,
    const float* tau,
    float* inverse,
    float* mid) {
  constexpr int K = 32;
  constexpr int HALF = 16;
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int lane = tid & 31;
  const int half = lane >> 4;
  const int sublane = lane & 15;
  const int block = warp >> 3;
  const int local_warp = warp & 7;
  const int local_col = local_warp * 2 + half;
  const int block_base = block * HALF;

  float x = 0.0f;

  // 16x16 diagonal block inverse
  #pragma unroll
  for (int solve_row = 0; solve_row < HALF; ++solve_row) {
    float partial = lower[(block_base + solve_row) * STRIDE + block_base + sublane] * x;
    partial = warp_sum(partial, HALF);
    const float diagonal = solve_row == local_col ? 1.0f : 0.0f;
    const float value =
        (diagonal - partial) * tau[block_base + solve_row];

    if (solve_row == sublane) {
      x = value;
      inverse[(block_base + solve_row) * STRIDE + block_base + local_col] = value;
    }
  }
  __syncthreads();

  // 16x16 off-diagonal block inverse
  if (tid < HALF * HALF) {
    const int row = tid >> 4;
    const int col = tid & 15;
    float accum = 0.0f;
    #pragma unroll
    for (int k = 0; k < HALF; ++k)
      accum += lower[(HALF + row) * STRIDE + k] * inverse[k * STRIDE + col];
    mid[row * HALF + col] = accum;
    asm volatile("bar.sync %0, %1;" :: "n"(1), "n"(HALF * HALF));

    accum = 0.0f;
    #pragma unroll
    for (int k = 0; k < HALF; ++k)
      accum -= inverse[(HALF + row) * STRIDE + HALF + k] * mid[k * HALF + col];
    inverse[(HALF + row) * STRIDE + col] = accum;
  }
}

template <int STRIDE>
__device__ inline
void build_t64_block(
    const float* gram,
    const float* tau,
    float* output,
    float* lower,
    float* inverse,
    float* mid) {
  constexpr int K = 64;
  constexpr int TB_SIZE = 16 * 32;
  const int tid = threadIdx.x;

  {
    const int pack = tid;
    const int row = pack / (K / 8);
    const int col = (pack - row * (K / 8)) * 8;
    float values[8];
    ldg_f32<8>(values, gram + row * STRIDE + col);
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      const int item_col = col + item;
      lower[row * K + item_col] = item_col < row ? values[item] : 0.0f;
    }
  }
  __syncthreads();

  build_t32_inverse_block<K>(lower, tau, inverse, mid);
  build_t32_inverse_block<K>(lower + (32 * K + 32), tau + 32, inverse + (32 * K + 32), mid);
  __syncthreads();

  #pragma unroll
  for (int item = 0; item < 2; ++item) {
    const int elem = item * TB_SIZE + tid;
    const int row = elem / 32;
    const int col = elem - row * 32;
    float accum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
      if (k >= col)
        accum += lower[(row + 32) * K + k] * inverse[k * K + col];
    }
    mid[row * 32 + col] = accum;
  }
  __syncthreads();

  #pragma unroll
  for (int item = 0; item < 2; ++item) {
    const int elem = item * TB_SIZE + tid;
    const int row = elem / 32;
    const int col = elem - row * 32;
    float accum = 0.0f;
    #pragma unroll
    for (int k = 0; k < 32; ++k) {
      if (k <= row)
        accum -= inverse[(row + 32) * K + (k + 32)] * mid[k * 32 + col];
    }
    inverse[(row + 32) * K + col] = accum;
  }
  __syncthreads();

  {
    const int pack = tid;
    const int row = pack / (K / 8);
    const int col = (pack - row * (K / 8)) * 8;
    float values[8];
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      const int item_col = col + item;
      values[item] = item_col <= row ? inverse[row * K + item_col] : 0.0f;
    }
    stg_f32<8>(output + row * STRIDE + col, values);
  }
}

__global__
__launch_bounds__(512, 1)
void build_t96_diag_kernel(const float* gram, const float* tau, float* output, int64_t tau_stride) {
  constexpr int K = 64;
  constexpr int N = 96;
  const int tid = threadIdx.x;
  const int batch = blockIdx.x;

  gram += batch * N * N;
  tau += batch * tau_stride;
  output += batch * N * N;

  extern __shared__ float storage[];
  float* lower = storage;
  float* inverse = lower + K * K;
  float* mid = inverse + K * K;

  build_t64_block<N>(gram, tau, output, lower, inverse, mid);

  if (tid < (K * 32 / 8)) {
    const int pack = tid;
    const int row = pack / (32 / 8);
    const int col = (pack - row * (32 / 8)) * 8;
    float zeros[8] = {};
    stg_f32<8>(output + row * N + K + col, zeros);
  }
  __syncthreads();

  if (tid < (32 * 32 / 8)) {
    const int pack = tid;
    const int row = pack / (32 / 8);
    const int col = (pack - row * (32 / 8)) * 8;
    float values[8];
    ldg_f32<8>(values, gram + (K + row) * N + K + col);
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      const int item_col = col + item;
      lower[row * K + item_col] = item_col < row ? values[item] : 0.0f;
    }
  }
  __syncthreads();

  build_t32_inverse_block<K>(lower, tau + K, inverse, mid);
  __syncthreads();

  if (tid < (32 * 32 / 8)) {
    const int pack = tid;
    const int row = pack / (32 / 8);
    const int col = (pack - row * (32 / 8)) * 8;
    float values[8];
    #pragma unroll
    for (int item = 0; item < 8; ++item) {
      const int item_col = col + item;
      values[item] = item_col <= row ? inverse[row * K + item_col] : 0.0f;
    }
    stg_f32<8>(output + (K + row) * N + K + col, values);
  }
}

void launch_build_t96_diag(
    const float* gram,
    const float* tau,
    float* output,
    int64_t tau_stride,
    int batch) {
  constexpr int K = 64;
  constexpr int smem_bytes = (2 * K * K + 32 * 32) * sizeof(float);
  build_t96_diag_kernel<<<batch, 512, smem_bytes>>>(gram, tau, output, tau_stride);
}

__global__
__launch_bounds__(512, 1)
void build_t128_diag_kernel(const float* gram, const float* tau, float* output, int64_t tau_stride) {
  constexpr int K = 64;
  constexpr int N = 128;
  const int tid = threadIdx.x;
  const int block = blockIdx.x & 1;
  const int batch = blockIdx.x >> 1;
  const int base = block * K;

  gram += batch * N * N + base * N + base;
  tau += batch * tau_stride + base;
  output += batch * N * N + base * N + base;

  extern __shared__ float storage[];
  float* lower = storage;
  float* inverse = lower + K * K;
  float* mid = inverse + K * K;

  build_t64_block<N>(gram, tau, output, lower, inverse, mid);

  if (block == 0) {
    const int pack = tid;
    const int row = pack / (K / 8);
    const int col = (pack - row * (K / 8)) * 8;
    float zeros[8] = {};
    stg_f32<8>(output + row * N + K + col, zeros);
  }
}

void launch_build_t128_diag(
    const float* gram,
    const float* tau,
    float* output,
    int64_t tau_stride,
    int batch) {
  constexpr int K = 64;
  constexpr int smem_bytes = (2 * K * K + 32 * 32) * sizeof(float);
  build_t128_diag_kernel<<<batch * 2, 512, smem_bytes>>>(gram, tau, output, tau_stride);
}

__device__ inline
int elect_sync() {
  int predicate = 0;
  asm volatile(
    "{\n\t"
    ".reg .pred p;\n\t"
    "elect.sync _|p, %1;\n\t"
    "@p mov.s32 %0, 1;\n\t"
    "}"
    : "+r"(predicate)
    : "r"(FULL_MASK));
  return predicate;
}

__device__ inline
void mbar_init(int address, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(address), "r"(count));
}

__device__ inline
void mbar_arrive(int address) {
  asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];" :: "r"(address) : "memory");
}

__device__ inline
void mbar_wait(int address, int phase) {
  constexpr int ticks = 0x989680;
  asm volatile(
    "{\n\t"
    ".reg .pred ready;\n\t"
    "mbar_wait_loop:\n\t"
    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 "
    "ready, [%0], %1, %2;\n\t"
    "@!ready bra.uni mbar_wait_loop;\n\t"
    "}"
    :: "r"(address), "r"(phase), "r"(ticks));
}

__device__ inline
void mbar_expect_tx(int address, int bytes) {
  asm volatile("mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [%0], %1;"
              :: "r"(address), "r"(bytes) : "memory");
}

__device__ inline
void tma_s2s(int dst, int src, int bytes, int mbar) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
              :: "r"(dst), "r"(src), "r"(bytes), "r"(mbar));
}

__device__ inline
void tma_s2g(void *dst, int src, int bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(dst), "r"(src), "r"(bytes));
}

__device__ inline
void st_async_f32(int destination, float value, int mbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
              :: "r"(destination), "f"(value), "r"(mbar));
}

__device__ __forceinline__
void store_release_gpu(int* address, int value) {
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(address), "r"(value) : "memory");
}

__device__ __forceinline__
int load_relaxed_gpu_no_allocate(const int* address) {
  int value;
  asm volatile("ld.global.relaxed.gpu.L1::no_allocate.u32 %0, [%1];" : "=r"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ void fence_acquire_gpu() {
  asm volatile("fence.acquire.gpu;" ::: "memory");
}

template <typename T>
__device__ inline T warp_uniform(T value) {
  return __shfl_sync(FULL_MASK, value, 0);
}

template <int ROWS, int COLS, int N>
__global__
__launch_bounds__((COLS / 8) * 32, 1)
void register_panel_kernel(const float* input, float* output, float* tau, float* v_fp32, __half* v_fp16) {
  static_assert(COLS % 8 == 0);
  static_assert(COLS <= ROWS);
  constexpr int ROW_ITEMS = (ROWS + 31) / 32;
  const int tid = threadIdx.x;
  const int batch = blockIdx.x;
  const int warp = warp_uniform(tid / 32);
  const int lane = tid & 31;

  input += batch * N * N;
  output += batch * N * N;
  tau += batch * N;

  extern __shared__ float storage[];
  float* reflectors = storage;  // [ROWS, COLS]
  float* taus = reflectors + ROWS * COLS;  // [COLS]
  const int mbars = __cvta_generic_to_shared(taus + COLS);

  if (warp == 0 && elect_sync()) {
    for (int i = 0; i < COLS; ++i) {
      mbar_init(mbars + i * 8, 32);
    }
  }
  __syncthreads();

  float columns[ROW_ITEMS][8];
  for (int item = 0; item < ROW_ITEMS; ++item) {
    const int row = item * 32 + lane;
    if (row < ROWS) {
      ldg_f32<8>(columns[item], input + row * N + warp * 8);
    } else {
      for (int i = 0; i < 8; ++i) columns[item][i] = 0.0f;
    }
  }

  for (int panel = 0; panel < warp; ++panel) {
    for (int i = 0; i < 8; ++i) {
      const int col = panel * 8 + i;
      mbar_wait(mbars + col * 8, 0);
      const float negative_tau = -taus[col];
      float v[ROW_ITEMS][2];
      for (int item = 0; item < ROW_ITEMS; ++item) {
        const int row = item * 32 + lane;
        const float value = row < ROWS ? reflectors[col * ROWS + row] : 0.0f;
        v[item][0] = value;
        v[item][1] = value;
      }
      for (int pair = 0; pair < 4; ++pair) {
        float dot[2] = {};
        for (int item = 0; item < ROW_ITEMS; ++item)
          fma_f32x2(dot, &columns[item][pair * 2], v[item]);
        dot[0] = warp_sum(dot[0]) * negative_tau;
        dot[1] = warp_sum(dot[1]) * negative_tau;
        for (int item = 0; item < ROW_ITEMS; ++item)
          fma_f32x2(&columns[item][pair * 2], v[item], dot);
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int col = warp * 8 + i;
    if constexpr (ROWS == COLS) {
      if (col == COLS - 1) {
        if (lane == 0) taus[col] = 0.0f;
        break;
      }
    }

    float tail = 0.0f;
    float x0 = 0.0f;
    for (int item = 0; item < ROW_ITEMS; ++item) {
      const int row = item * 32 + lane;
      const float x = columns[item][i];
      tail += (row > col) * x * x;
      x0 += row == col ? x : 0.0f;
    }
    tail = warp_sum(tail);
    x0 = __shfl_sync(FULL_MASK, x0, col & 31);
    const float norm = sqrt_approx(x0 * x0 + tail);
    const float beta = -copysignf(norm, x0);
    const bool has_tail = tail > 0.0f;
    const float tau_value = has_tail ? (beta - x0) * rcp_approx(beta) : 0.0f;
    const float inverse = has_tail ? 1.0f / (x0 - beta) : 0.0f;
    if (lane == 0) taus[col] = tau_value;

    float v[ROW_ITEMS];
    for (int item = 0; item < ROW_ITEMS; ++item) {
      const int row = item * 32 + lane;
      const float x = columns[item][i];
      v[item] = has_tail? (row == col) + (row > col) * (x * inverse) : 0.0f;
      const float reflected = (row < col) * x + (row == col) * beta + (row > col) * v[item];
      columns[item][i] = has_tail ? reflected : x;
      if (row < ROWS) reflectors[col * ROWS + row] = v[item];
    }
    mbar_arrive(mbars + col * 8);

    for (int trailing_col = i + 1; trailing_col < 8; ++trailing_col) {
      float dot = 0.0f;
      for (int item = 0; item < ROW_ITEMS; ++item)
        dot += columns[item][trailing_col] * v[item];
      dot = warp_sum(dot) * tau_value;
      for (int item = 0; item < ROW_ITEMS; ++item)
        columns[item][trailing_col] -= v[item] * dot;
    }
  }

  // emit V when it's not the last square QR
  if constexpr (ROWS > COLS) {
    v_fp32 += batch * ROWS * COLS;
    v_fp16 += batch * ROWS * COLS;

    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;");
    constexpr int PANEL_SIZE = 8 * ROWS;
    if (elect_sync()) {
      const int sV_fp32 = __cvta_generic_to_shared(reflectors) + warp * PANEL_SIZE * 4;
      tma_s2g(v_fp32 + warp * PANEL_SIZE, sV_fp32, PANEL_SIZE * 4);
    }

    for (int i = 0; i < cdiv(PANEL_SIZE, 32 * 4); i++) {
      const int idx = (i * 32 + lane) * 4;
      if (idx < ROWS * 8) {
        float4 tmp = reinterpret_cast<float4 *>(reflectors + (warp * PANEL_SIZE + idx))[0];
        half2 tmp2[2];
        tmp2[0] = __float22half2_rn({tmp.x, tmp.y});
        tmp2[1] = __float22half2_rn({tmp.z, tmp.w});
        stg_f32<2>(
          reinterpret_cast<float *>(v_fp16 + (warp * PANEL_SIZE + idx)),
          reinterpret_cast<float *>(tmp2));
      }
    }
  }

  if (lane < 2) {
    float tmp[4];
    reinterpret_cast<float4 *>(tmp)[0] = reinterpret_cast<float4 *>(taus)[warp * 2 + lane];
    stg_f32<4>(tau + (warp * 2 + lane) * 4, tmp);
  }
  for (int item = 0; item < ROW_ITEMS; ++item) {
    const int row = item * 32 + lane;
    if (row < ROWS)
      stg_f32<8>(output + row * N + warp * 8, columns[item]);
  }
}

template <int ROWS, int COLS, int N, int VEC_SIZE>
__global__
__cluster_dims__(2, 1, 1)
__launch_bounds__((COLS / VEC_SIZE / 2) * 32, 1)
void register_2sm_panel_kernel(const float* input, float* output, float* tau, float *v_fp32, __half *v_fp16) {
  static_assert(COLS % (VEC_SIZE * 2) == 0);
  static_assert(COLS <= ROWS);
  constexpr int ROW_ITEMS = (ROWS + 31) / 32;
  constexpr int NUM_WARPS = COLS / VEC_SIZE / 2;

  const int tid = threadIdx.x;
  const int block = blockIdx.x;
  const int warp = warp_uniform(tid / 32);
  const int lane = tid & 31;
  const int rank = block & 1;
  const int batch = block / 2;

  input += batch * N * N;
  output += batch * N * N;
  tau += batch * N;

  extern __shared__ float storage[];
  float* reflectors = storage;
  constexpr int LOCAL_COLS = COLS / 2;
  float* taus = reflectors + ROWS * LOCAL_COLS;
  const int reflector_addr = __cvta_generic_to_shared(reflectors);
  const int tau_addr = reflector_addr + ROWS * LOCAL_COLS * 4;
  const int mbars = tau_addr + COLS * 4;

  const int reflector_addr1 = reflector_addr | 0x01000000;
  const int tau_addr1 = tau_addr | 0x01000000;

  if (warp == 0 && elect_sync()) {
    for (int i = 0; i < COLS; ++i) mbar_init(mbars + i * 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  asm volatile("barrier.cluster.arrive.relaxed.aligned;");
  asm volatile("barrier.cluster.wait.acquire.aligned;");

  float columns[ROW_ITEMS][VEC_SIZE];
  for (int item = 0; item < ROW_ITEMS; ++item) {
    const int row = item * 32 + lane;
    const int col = (rank * NUM_WARPS + warp) * VEC_SIZE;
    if (row < ROWS) {
      ldg_f32<VEC_SIZE>(columns[item], input + row * N + col);
    } else {
      for (int i = 0; i < VEC_SIZE; ++i) columns[item][i] = 0.0f;
    }
  }

  // from remote reflectors
  for (int panel = 0; panel < rank * NUM_WARPS; ++panel) {
    for (int i = 0; i < VEC_SIZE; ++i) {
      const int col = panel * VEC_SIZE + i;
      if (warp == 0)
        mbar_wait(mbars + col * 8, 0);
      __syncthreads();
      const float negative_tau = -taus[col];
      float v[ROW_ITEMS][2];
      for (int item = 0; item < ROW_ITEMS; ++item) {
        const int row = item * 32 + lane;
        const float value = row < ROWS ? reflectors[col * ROWS + row] : 0.0f;
        v[item][0] = value;
        v[item][1] = value;
      }
      for (int pair = 0; pair < VEC_SIZE/2; ++pair) {
        float dot[2] = {};
        for (int item = 0; item < ROW_ITEMS; ++item)
          fma_f32x2(dot, &columns[item][pair * 2], v[item]);
        dot[0] = warp_sum(dot[0]) * negative_tau;
        dot[1] = warp_sum(dot[1]) * negative_tau;
        for (int item = 0; item < ROW_ITEMS; ++item)
          fma_f32x2(&columns[item][pair * 2], v[item], dot);
      }
    }
  }
  __syncthreads();

  // from local reflectors
  for (int panel = rank * NUM_WARPS; panel < rank * NUM_WARPS + warp; ++panel) {
    const int local_panel = panel - rank * NUM_WARPS;
    for (int i = 0; i < VEC_SIZE; ++i) {
      const int col = panel * VEC_SIZE + i;
      const int local_col = local_panel * VEC_SIZE + i;
      mbar_wait(mbars + col * 8, 0);
      const float negative_tau = -taus[col];
      float v[ROW_ITEMS][2];
      for (int item = 0; item < ROW_ITEMS; ++item) {
        const int row = item * 32 + lane;
        const float value = row < ROWS ? reflectors[local_col * ROWS + row] : 0.0f;
        v[item][0] = value;
        v[item][1] = value;
      }
      for (int pair = 0; pair < VEC_SIZE/2; ++pair) {
        float dot[2] = {};
        for (int item = 0; item < ROW_ITEMS; ++item)
          fma_f32x2(dot, &columns[item][pair * 2], v[item]);
        dot[0] = warp_sum(dot[0]) * negative_tau;
        dot[1] = warp_sum(dot[1]) * negative_tau;
        for (int item = 0; item < ROW_ITEMS; ++item)
          fma_f32x2(&columns[item][pair * 2], v[item], dot);
      }
    }
  }

  #pragma unroll
  for (int i = 0; i < VEC_SIZE; ++i) {
    const int col = (rank * NUM_WARPS + warp) * VEC_SIZE + i;
    const int local_col = warp * VEC_SIZE + i;
    if constexpr (ROWS == COLS) {
      if (col == COLS - 1) {
        if (lane == 0) taus[col] = 0.0f;
        break;
      }
    }

    float tail = 0.0f;
    float x0 = 0.0f;
    for (int item = 0; item < ROW_ITEMS; ++item) {
      const int row = item * 32 + lane;
      const float x = columns[item][i];
      tail += (row > col) * x * x;
      x0 += row == col ? x : 0.0f;
    }
    tail = warp_sum(tail);
    x0 = __shfl_sync(FULL_MASK, x0, col & 31);

    const float norm = sqrt_approx(x0 * x0 + tail);
    const float beta = -copysignf(norm, x0);
    const bool has_tail = tail > 0.0f;
    const float tau_value = has_tail ? (beta - x0) * rcp_approx(beta) : 0.0f;
    const float inverse = has_tail ? rcp_approx(x0 - beta) : 0.0f;
    if (lane == 0) taus[col] = tau_value;

    float v[ROW_ITEMS];
    for (int item = 0; item < ROW_ITEMS; ++item) {
      const int row = item * 32 + lane;
      const float x = columns[item][i];
      v[item] = has_tail ? (row == col) + (row > col) * (x * inverse) : 0.0f;
      const float reflected = (row < col) * x + (row == col) * beta + (row > col) * v[item];
      columns[item][i] = has_tail ? reflected : x;
      if (row < ROWS) reflectors[local_col * ROWS + row] = v[item];
    }

    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;");
    if (elect_sync()) {
      mbar_arrive(mbars + col * 8);
      if (rank == 0) {
        const int remote_mbar = (mbars + col * 8) | 0x01000000;
        mbar_expect_tx(remote_mbar, (ROWS + 1) * 4);
        tma_s2s(reflector_addr1 + col * ROWS * 4,
                reflector_addr + local_col * ROWS * 4,
                ROWS * 4, remote_mbar);
        st_async_f32(tau_addr1 + col * 4, tau_value, remote_mbar);
      }
    }

    for (int trailing = i + 1; trailing < VEC_SIZE; ++trailing) {
      float dot = 0.0f;
      for (int item = 0; item < ROW_ITEMS; ++item)
        dot += columns[item][trailing] * v[item];
      dot = warp_sum(dot) * tau_value;
      for (int item = 0; item < ROW_ITEMS; ++item)
        columns[item][trailing] -= v[item] * dot;
    }
  }
  const int panel_id = rank * NUM_WARPS + warp;
  const int local_panel_id = warp;

  // emit V when it's not the last square QR
  if constexpr (ROWS > COLS) {
    v_fp32 += batch * ROWS * COLS;
    v_fp16 += batch * ROWS * COLS;

    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;");
    constexpr int PANEL_SIZE = VEC_SIZE * ROWS;
    if (elect_sync()) {
      const int sV_fp32 = __cvta_generic_to_shared(reflectors) + local_panel_id * PANEL_SIZE * 4;
      tma_s2g(v_fp32 + panel_id * PANEL_SIZE, sV_fp32, PANEL_SIZE * 4);
    }

    for (int i = 0; i < cdiv(PANEL_SIZE, 32 * 4); i++) {
      const int idx = (i * 32 + lane) * 4;
      if (idx < PANEL_SIZE) {
        float4 tmp = reinterpret_cast<float4 *>(reflectors + (local_panel_id * PANEL_SIZE + idx))[0];
        half2 tmp2[2];
        tmp2[0] = __float22half2_rn({tmp.x, tmp.y});
        tmp2[1] = __float22half2_rn({tmp.z, tmp.w});
        stg_f32<2>(
          reinterpret_cast<float *>(v_fp16 + (panel_id * PANEL_SIZE + idx)),
          reinterpret_cast<float *>(tmp2));
      }
    }
  }

  const int col = panel_id * VEC_SIZE;
  if (lane < VEC_SIZE / 4) {
    float tmp[4];
    reinterpret_cast<float4 *>(tmp)[0] = reinterpret_cast<float4 *>(taus)[panel_id * (VEC_SIZE/4) + lane];
    stg_f32<4>(tau + (panel_id * (VEC_SIZE/4) + lane) * 4, tmp);
  }
  for (int item = 0; item < ROW_ITEMS; ++item) {
    const int row = item * 32 + lane;
    if (row < ROWS)
      stg_f32<VEC_SIZE>(output + row * N + col, columns[item]);
  }
}

template <int ROWS, int COLS, int WARPS, int N>
__global__
__launch_bounds__(WARPS * 32, 1)
void gmem_panel_kernel(
  const float* input,
  float* output,
  float* tau,
  float* v_fp32,
  __half* v_fp16,
  int* flags
) {
  constexpr int THREADS = WARPS * 32;
  constexpr int ITEMS = cdiv(ROWS, THREADS);
  constexpr int CTAS = COLS / 8;
  const int rank = blockIdx.x % CTAS;
  const int batch = blockIdx.x / CTAS;
  const int tid = threadIdx.x;
  const int warp = warp_uniform(tid / 32);
  const int lane = tid & 31;
  const int column_base = rank * 8;

  input += batch * N * N;
  output += batch * N * N;
  tau += batch * N;
  v_fp32 += batch * ROWS * COLS;
  v_fp16 += batch * ROWS * COLS;
  flags += batch * COLS;

  extern __shared__ __half sV_fp16[];
  __shared__ float partials[WARPS * 8];
  __shared__ float diagonal;

  float columns[ITEMS][8];
  #pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const int row = item * THREADS + tid;
    if (row < ROWS) {
      ldg_f32<8>(columns[item], input + row * N + column_base);
    } else {
      #pragma unroll
      for (int j = 0; j < 8; ++j) columns[item][j] = 0.0f;
    }
  }

  // Consume all reflectors produced by earlier CTAs.
  for (int k = 0; k < column_base; ++k) {
    if (tid == 0) {
      while (!load_relaxed_gpu_no_allocate(flags + k)) {
        __nanosleep(64);
      }
      fence_acquire_gpu();
    }
    __syncthreads();

    const float tau_k = tau[k];
    float dots[8] = {};
    float reflector[ITEMS];
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
      const int row = item * THREADS + tid;
      const float value = row < ROWS ? v_fp32[k * ROWS + row] : 0.0f;
      reflector[item] = value;
      #pragma unroll
      for (int j = 0; j < 8; ++j)
        dots[j] += value * columns[item][j];
    }
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float warp_dot = warp_sum(dots[j]);
      if (lane == 0) partials[j * WARPS + warp] = warp_dot;
    }
    __syncthreads();

    float scales[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float value = lane < WARPS ? partials[j * WARPS + lane] : 0.0f;
      scales[j] = warp_sum(value) * tau_k;
    }
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
      #pragma unroll
      for (int j = 0; j < 8; ++j)
        columns[item][j] -= reflector[item] * scales[j];
    }
  }

  // Factor this CTA's eight columns and publish each reflector to later CTAs.
  #pragma unroll
  for (int pivot = 0; pivot < 8; ++pivot) {
    const int col = column_base + pivot;
    float local_tail = 0.0f;
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
      const int row = item * THREADS + tid;
      const float x = columns[item][pivot];
      if (row > col && row < ROWS) local_tail += x * x;
      if (row == col) diagonal = x;
    }
    const float warp_tail = warp_sum(local_tail);
    if (lane == 0) partials[warp] = warp_tail;
    __syncthreads();

    const float tail_part = lane < WARPS ? partials[lane] : 0.0f;
    const float tail = warp_sum(tail_part);
    const float x0 = diagonal;
    const float norm = sqrt_approx(x0 * x0 + tail);
    const float beta = -copysignf(norm, x0);
    const bool has_tail = tail > 0.0f;
    const float tau_k = has_tail ? (beta - x0) * rcp_approx(beta) : 0.0f;
    const float inverse = has_tail ? rcp_approx(x0 - beta) : 0.0f;
    if (tid == 0) tau[col] = tau_k;

    float reflector[ITEMS];
    float dots[8] = {};
    #pragma unroll
    for (int item = 0; item < ITEMS; ++item) {
      const int row = item * THREADS + tid;
      const float x = columns[item][pivot];
      const float value = has_tail ? (row == col) + (row > col) * (x * inverse) : 0.0f;
      reflector[item] = value;
      const float reflected = (row < col) * x + (row == col) * beta + (row > col) * value;
      columns[item][pivot] = has_tail ? reflected : x;
      if (row < ROWS) {
        v_fp32[col * ROWS + row] = value;
        sV_fp16[pivot * ROWS + row] = __float2half_rn(value);
      }
      #pragma unroll
      for (int j = pivot + 1; j < 8; ++j)
        dots[j] += value * columns[item][j];
    }
    #pragma unroll
    for (int j = pivot + 1; j < 8; ++j) {
      const float warp_dot = warp_sum(dots[j]);
      if (lane == 0) partials[j * WARPS + warp] = warp_dot;
    }

    // This synchronizes both the V publication and local dot partials.
    __syncthreads();
    if (tid == 0)
      store_release_gpu(flags + col, 1);

    #pragma unroll
    for (int j = pivot + 1; j < 8; ++j) {
      const float value = lane < WARPS ? partials[j * WARPS + lane] : 0.0f;
      const float scale = warp_sum(value) * tau_k;
      #pragma unroll
      for (int item = 0; item < ITEMS; ++item)
        columns[item][j] -= reflector[item] * scale;
    }
    __syncthreads();
  }

  asm volatile("fence.proxy.async.shared::cta;");
  if (warp == 0 && elect_sync()) {
    const int src = __cvta_generic_to_shared(sV_fp16);
    tma_s2g(v_fp16 + column_base * ROWS, src, ROWS * 8 * 2);
  }

  #pragma unroll
  for (int item = 0; item < ITEMS; ++item) {
    const int row = item * THREADS + tid;
    if (row < ROWS)
      stg_f32<8>(output + row * N + column_base, columns[item]);
  }
}

void launch_gmem_panel(
  const float* input,
  float* output,
  float* tau,
  float* v_fp32,
  void* v_fp16_raw,
  int* flags,
  int batch,
  int rows,
  int cols,
  int n
) {
  __half* v_fp16 = static_cast<__half*>(v_fp16_raw);

#define LAUNCH_GMEM_PANEL(N, ROWS, COLS) \
if (n == N && rows == ROWS && cols == COLS) { \
  constexpr int CTAS = COLS / 8; \
  /* each thread holds 8 items */ \
  constexpr int WARPS = cdiv(ROWS, 32 * 8); \
  const int smem_bytes = (ROWS * 8) * 2; \
  auto this_kernel = gmem_panel_kernel<ROWS, COLS, WARPS, N>; \
  cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes); \
  this_kernel<<<batch * CTAS, WARPS * 32, smem_bytes>>>(input, output, tau, v_fp32, v_fp16, flags); \
  return; \
}
  // QR2048
  LAUNCH_GMEM_PANEL(2048, 2048, 128)
  LAUNCH_GMEM_PANEL(2048, 1920, 128)
  LAUNCH_GMEM_PANEL(2048, 1792, 128)
  LAUNCH_GMEM_PANEL(2048, 1664, 128)
  LAUNCH_GMEM_PANEL(2048, 1536, 128)
  LAUNCH_GMEM_PANEL(2048, 1408, 128)
  LAUNCH_GMEM_PANEL(2048, 1280, 128)
  LAUNCH_GMEM_PANEL(2048, 1152, 128)
  LAUNCH_GMEM_PANEL(2048, 1024, 128)
  LAUNCH_GMEM_PANEL(2048,  896, 128)
  LAUNCH_GMEM_PANEL(2048,  768, 128)
  // QR4096
  LAUNCH_GMEM_PANEL(4096, 4096, 128)
  LAUNCH_GMEM_PANEL(4096, 3968, 128)
  LAUNCH_GMEM_PANEL(4096, 3840, 128)
  LAUNCH_GMEM_PANEL(4096, 3712, 128)
  LAUNCH_GMEM_PANEL(4096, 3584, 128)
  LAUNCH_GMEM_PANEL(4096, 3456, 128)
  LAUNCH_GMEM_PANEL(4096, 3328, 128)
  LAUNCH_GMEM_PANEL(4096, 3200, 128)
  LAUNCH_GMEM_PANEL(4096, 3072, 128)
  LAUNCH_GMEM_PANEL(4096, 2944, 128)
  LAUNCH_GMEM_PANEL(4096, 2816, 128)
  LAUNCH_GMEM_PANEL(4096, 2688, 128)
  LAUNCH_GMEM_PANEL(4096, 2560, 128)
  LAUNCH_GMEM_PANEL(4096, 2432, 128)
  LAUNCH_GMEM_PANEL(4096, 2304, 128)
  LAUNCH_GMEM_PANEL(4096, 2176, 128)
  LAUNCH_GMEM_PANEL(4096, 2048, 128)
  LAUNCH_GMEM_PANEL(4096, 1920, 128)
  LAUNCH_GMEM_PANEL(4096, 1792, 128)
  LAUNCH_GMEM_PANEL(4096, 1664, 128)
  LAUNCH_GMEM_PANEL(4096, 1536, 128)
  LAUNCH_GMEM_PANEL(4096, 1408, 128)
  LAUNCH_GMEM_PANEL(4096, 1280, 128)
  LAUNCH_GMEM_PANEL(4096, 1152, 128)
  LAUNCH_GMEM_PANEL(4096, 1024, 128)
  LAUNCH_GMEM_PANEL(4096,  896, 128)
  LAUNCH_GMEM_PANEL(4096,  768, 128)
#undef LAUNCH_GMEM_PANEL
}

void launch_panelMN(
  const float* input,
  float* output,
  float* tau,
  float* v_fp32,
  void* v_fp16_raw,
  int batch,
  int rows,
  int cols,
  int n
) {
  __half* v_fp16 = static_cast<__half*>(v_fp16_raw);

#define LAUNCH_REGISTER_PANEL(N, ROWS, COLS) \
if (n == N && rows == ROWS && cols == COLS) { \
  constexpr int smem_bytes = (ROWS * COLS + COLS) * 4 + COLS * 8; \
  auto this_kernel = register_panel_kernel<ROWS, COLS, N>; \
  cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes); \
  this_kernel<<<batch, (COLS / 8) * 32, smem_bytes>>>(input, output, tau, v_fp32, v_fp16); \
  return; \
}
#define LAUNCH_REGISTER_2SM_PANEL(N, ROWS, COLS, VEC_SIZE) \
if (n == N && rows == ROWS && cols == COLS) { \
  constexpr int smem_bytes = (ROWS * (COLS / 2) + COLS) * 4 + COLS * 8; \
  constexpr int threads = (COLS / VEC_SIZE / 2) * 32; \
  auto this_kernel = register_2sm_panel_kernel<ROWS, COLS, N, VEC_SIZE>; \
  cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes); \
  this_kernel<<<batch * 2, threads, smem_bytes>>>(input, output, tau, v_fp32, v_fp16); \
  return; \
}

  // QR32
  LAUNCH_REGISTER_PANEL(32, 32, 32)
  // QR176
  LAUNCH_REGISTER_2SM_PANEL(176, 176, 176, 8)
  // for QR352
  LAUNCH_REGISTER_2SM_PANEL(352, 352, 128, 8)
  LAUNCH_REGISTER_2SM_PANEL(352, 224, 128, 8)
  LAUNCH_REGISTER_2SM_PANEL(352, 96, 96, 8)
  // for QR512
  LAUNCH_REGISTER_PANEL(512, 512,  96)
  LAUNCH_REGISTER_PANEL(512, 416,  96)
  LAUNCH_REGISTER_PANEL(512, 320, 128)
  LAUNCH_REGISTER_PANEL(512, 192, 192)
  // for QR1024
  LAUNCH_REGISTER_2SM_PANEL(1024, 1024, 96, 4)
  LAUNCH_REGISTER_2SM_PANEL(1024, 928, 96, 4)
  LAUNCH_REGISTER_2SM_PANEL(1024, 832, 96, 4)
  LAUNCH_REGISTER_2SM_PANEL(1024, 736, 96, 4)
  LAUNCH_REGISTER_2SM_PANEL(1024, 640, 128, 8)
  LAUNCH_REGISTER_2SM_PANEL(1024, 512, 128, 8)
  LAUNCH_REGISTER_2SM_PANEL(1024, 384, 128, 8)
  LAUNCH_REGISTER_2SM_PANEL(1024, 256, 128, 8)
  LAUNCH_REGISTER_2SM_PANEL(1024, 128, 128, 8)
  // for QR2048
  LAUNCH_REGISTER_2SM_PANEL(2048, 640, 128, 8)
  LAUNCH_REGISTER_2SM_PANEL(2048, 512, 128, 8)
  LAUNCH_REGISTER_2SM_PANEL(2048, 384, 128, 8)
  LAUNCH_REGISTER_2SM_PANEL(2048, 256, 128, 8)
  LAUNCH_REGISTER_2SM_PANEL(2048, 128, 128, 8)
  // for QR4096
  LAUNCH_REGISTER_2SM_PANEL(4096, 640, 128, 8)
  LAUNCH_REGISTER_2SM_PANEL(4096, 512, 128, 8)
  LAUNCH_REGISTER_2SM_PANEL(4096, 384, 128, 8)
  LAUNCH_REGISTER_2SM_PANEL(4096, 256, 128, 8)
  LAUNCH_REGISTER_2SM_PANEL(4096, 128, 128, 8)

#undef LAUNCH_REGISTER_PANEL
#undef LAUNCH_REGISTER_2SM_PANEL
}
"""


BUILD_DIR = Path(__file__).resolve().parent / ".build"
BUILD_DIR.mkdir(exist_ok=True)
CU13_ROOT = Path(torch.__file__).resolve().parent.parent / "nvidia" / "cu13"
CU13_LIB = CU13_ROOT / "lib"

load_inline(
    "codex_ext",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    is_python_module=False,
    no_implicit_headers=True,
    extra_include_paths=[str(CU13_ROOT / "include")],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo"],
    extra_ldflags=[
        f"-L{CU13_LIB}",
        f"-Wl,-rpath,{CU13_LIB}",
        "-l:libcublas.so.13",
        "-l:libcublasLt.so.13",
    ],
    build_directory=str(BUILD_DIR),
)


@triton.jit
def _build_t_inv_kernel(
    t_inv,
    tau,
    count,
    tau_batch_stride,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    indices = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = indices < count
    cols = indices % N
    row_batch = indices // N
    rows = row_batch % N
    batch_index = row_batch // N

    tau_value = tl.load(tau + batch_index * tau_batch_stride + cols, mask=valid, other=1.0)
    value = tl.load(t_inv + indices, mask=valid, other=0.0)
    safe_tau = tl.where(tau_value == 0.0, 1.0, tau_value)
    value = tl.where(rows < cols, value, 0.0)
    value = tl.where(rows == cols, 1.0 / safe_tau, value)

    tl.store(t_inv + indices, value, mask=valid)


def build_t_inv(t_inv: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    count = t_inv.numel()
    block = 256
    _build_t_inv_kernel[(triton.cdiv(count, block),)](
        t_inv, tau, count, tau.stride(0), N=t_inv.shape[-1], BLOCK=block
    )


FP16_GRAM_MIN_ROWS = {
    512: 320,
    1024: 512,
    2048: 1024,
    4096: 2048,
}


def compact_wy_apply_transpose(
    panel: torch.Tensor,
    panel_tau: torch.Tensor,
    trailing: torch.Tensor,
    output: torch.Tensor,
    *,
    problem_size: int,
    v_f32: torch.Tensor,
    v_f16: torch.Tensor,
) -> torch.Tensor:
    # update to the trailing matrix: A = Q^T A
    #   where Q = I - V T V^T
    #   => A = A - V T^T V^T A
    # note that        T   = inv(diag(1/tau) + strictUpper(V^T @ V))
    # or equivalently, T^T = inv(diag(1/tau) + strictLower(V^T @ V))
    batch, rows, cols = panel.shape
    trail_cols = trailing.shape[2]

    vt = v_f32.transpose(-2, -1)
    if problem_size in (512, 1024, 2048, 4096) and rows >= FP16_GRAM_MIN_ROWS[problem_size]:
        gram = torch.bmm(v_f16.transpose(-2, -1), v_f16, out_dtype=torch.float32)
    else:
        gram = vt @ v_f32

    projected = panel.new_empty(batch, cols, trail_cols)
    torch.ops.codex.tf32_baddbmm_out(projected, vt, trailing, projected, beta=0.0, alpha=1.0)

    if cols == 128:
        tt = torch.ops.codex.build_t128_diag(gram, panel_tau)
        mid = gram[:, 64:, :64] @ tt[:, :64, :64]
        torch.baddbmm(tt[:, 64:, :64], tt[:, 64:, 64:], mid, beta=0.0, alpha=-1.0, out=tt[:, 64:, :64])
        transformed = tt @ projected
    elif cols == 96:
        tt = torch.ops.codex.build_t96_diag(gram, panel_tau)
        mid = gram[:, 64:, :64] @ tt[:, :64, :64]
        torch.baddbmm(tt[:, 64:, :64], tt[:, 64:, 64:], mid, beta=0.0, alpha=-1.0, out=tt[:, 64:, :64])
        transformed = tt @ projected
    else:
        build_t_inv(gram, panel_tau)
        transformed = torch.linalg.solve_triangular(gram.transpose(-2, -1), projected, upper=False)

    torch.ops.codex.fp16_baddbmm_out(trailing, v_f16, transformed.half(), output, beta=1.0, alpha=-1.0)


def smem_blocked_qr(data: torch.Tensor, widths: tuple[int, ...], *, problem_size: int) -> output_t:
    assert sum(widths) == data.shape[-1]

    batch, n, _ = data.shape
    h = torch.empty_like(data)
    tau = data.new_empty(batch, n)
    source = data
    offset = 0
    for width in widths:
        panel_input = source[:, offset:, offset : offset + width]
        panel = h[:, offset:, offset : offset + width]
        panel_tau = tau[:, offset : offset + width]

        if offset + width < n:
            # v is col-major
            v_f32 = h.new_empty(batch, width, n - offset).transpose(1, 2)
            v_f16 = h.new_empty(batch, width, n - offset, dtype=torch.float16).transpose(1, 2)
            torch.ops.codex.panelMN(panel_input, panel, panel_tau, v_f32, v_f16)

            trailing_input = source[:, offset:, offset + width :]
            trailing_output = h[:, offset:, offset + width :]
            compact_wy_apply_transpose(
                panel,
                panel_tau,
                trailing_input,
                trailing_output,
                problem_size=problem_size,
                v_f32=v_f32,
                v_f16=v_f16,
            )

        else:
            torch.ops.codex.panelMN(panel_input, panel, panel_tau)

        source = h
        offset += width

    return h, tau


def scheduled_global_qr(
    data: torch.Tensor, panel_widths: tuple[int, ...], tail_widths: tuple[int, ...] = ()
) -> output_t:
    batch, n, _ = data.shape
    assert sum(panel_widths) + sum(tail_widths) == n
    h = data.clone()
    tau = torch.empty((batch, n), device=data.device, dtype=data.dtype)

    offset = 0
    for width in panel_widths:
        panel = h[:, offset:, offset : offset + width]
        panel_tau = tau[:, offset : offset + width]
        # v is col-major
        v_f32 = h.new_empty(batch, width, n - offset).transpose(1, 2)
        v_f16 = h.new_empty(batch, width, n - offset, dtype=torch.float16).transpose(1, 2)
        torch.ops.codex.gmem_panel(panel, panel, panel_tau, v_f32, v_f16)
        if offset + width < n:
            trailing = h[:, offset:, offset + width :]
            compact_wy_apply_transpose(
                panel,
                panel_tau,
                trailing,
                trailing,
                problem_size=n,
                v_f32=v_f32,
                v_f16=v_f16,
            )
        offset += width

    for width in tail_widths:
        panel = h[:, offset:, offset : offset + width]
        panel_tau = tau[:, offset : offset + width]
        if offset + width < n:
            # v is col-major
            v_f32 = h.new_empty(batch, width, n - offset).transpose(1, 2)
            v_f16 = h.new_empty(batch, width, n - offset, dtype=torch.float16).transpose(1, 2)
            torch.ops.codex.panelMN(panel, panel, panel_tau, v_f32, v_f16)
            trailing = h[:, offset:, offset + width :]
            compact_wy_apply_transpose(
                panel,
                panel_tau,
                trailing,
                trailing,
                problem_size=n,
                v_f32=v_f32,
                v_f16=v_f16,
            )
        else:
            torch.ops.codex.panelMN(panel, panel, panel_tau)
        offset += width

    return h, tau


SHORT_BLOCK_WIDTHS = {
    352: (128, 128, 96),
    512: (96, 96, 128, 192),
    1024: (96, 96, 96, 96, 128, 128, 128, 128, 128),
}
GLOBAL_QR_SCHEDULES = {
    2048: ((128,) * 11, (128,) * 5),
    4096: ((128,) * 27, (128,) * 5),
}
ALLOW_TF32 = (352, 1024, 2048, 4096)


def custom_kernel(data: input_t) -> output_t:
    batch, n, _ = data.shape

    # pytorch TF32 is noticeably faster than our CuBLASLt TF32 wrapper...
    if n in ALLOW_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if n in (32, 176):
        h = torch.empty_like(data)
        tau = data.new_empty(batch, n)
        torch.ops.codex.panelMN(data, h, tau)

    if n in SHORT_BLOCK_WIDTHS:
        h, tau = smem_blocked_qr(data, SHORT_BLOCK_WIDTHS[n], problem_size=n)

    if n in GLOBAL_QR_SCHEDULES:
        panel_widths, tail_widths = GLOBAL_QR_SCHEDULES[n]
        h, tau = scheduled_global_qr(data, panel_widths, tail_widths)

    if n in ALLOW_TF32:
        torch.backends.cuda.matmul.allow_tf32 = False

    return h, tau
