#!POPCORN leaderboard qr_v2
#!POPCORN gpu B200

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

CPP_SRC = r"""
#include <ATen/ATen.h>
#include <torch/library.h>

void qr32_launch(const float *gA, float *gH, float *gTau, int bs);
void qr176_launch(const float *gA, float *gH, float *gTau, int bs);

void qr32(const at::Tensor& A, at::Tensor& H, at::Tensor& tau) {
  const int bs = A.size(0);
  const float *gA = A.data_ptr<float>();
        float *gH = H.data_ptr<float>();
        float *gTau = tau.data_ptr<float>();
  qr32_launch(gA, gH, gTau, bs);
}

void qr176(const at::Tensor& A, at::Tensor& H, at::Tensor& tau) {
  const int bs = A.size(0);
  const float *gA = A.data_ptr<float>();
        float *gH = H.data_ptr<float>();
        float *gTau = tau.data_ptr<float>();
  qr176_launch(gA, gH, gTau, bs);
}

TORCH_LIBRARY(linalg, m) {
  m.def("qr32(Tensor A, Tensor(a!) H, Tensor(b!) tau) -> ()", &qr32);
  m.def("qr176(Tensor A, Tensor(a!) H, Tensor(b!) tau) -> ()", &qr176);
}
"""

CUDA_SRC = r"""
__device__ inline
float warp_sum(float val, int size) {
  for (int s = size / 2; s > 0; s >>= 1)
    val += __shfl_xor_sync(0xFFFF'FFFF, val, s);
  return val;
}

__device__ inline
constexpr int next_pow2(int x) {
  if (x <= 1) return 1;
  x--;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}

__device__ inline
void ldg_f32x8(float *dst, const float *src) {
  asm volatile("ld.global.relaxed.cta.L1::no_allocate.v8.f32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
              : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
                "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
              : "l"(src));
}

__device__ inline
void stg_f32x8(float *dst, const float *src) {
  asm volatile("st.global.relaxed.cta.L1::no_allocate.v8.f32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
              :: "l"(dst)
                "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]),
                "f"(src[4]), "f"(src[5]), "f"(src[6]), "f"(src[7]));
}

__device__ inline
void mul_f32x2(float *c, const float *a, const float *b) {
  asm volatile(
    "{"
    ".reg .b64 a, b, c;\n"
    "mov.b64 a, {%2, %3};\n"
    "mov.b64 b, {%4, %5};\n"
    "mul.rn.f32x2 c, a, b;\n"
    "mov.b64 {%0, %1}, c;\n"
    "}"
    : "=f"(c[0]), "=f"(c[1])
    : "f"(a[0]), "f"(a[1]), "f"(b[0]), "f"(b[1])
  );
}

__device__ inline
void fma_f32x2(float *c, const float *a, const float *b) {
  asm volatile(
    "{"
    ".reg .b64 a, b, c, d;\n"
    "mov.b64 c, {%0, %1};\n"
    "mov.b64 a, {%2, %3};\n"
    "mov.b64 b, {%4, %5};\n"
    "fma.rn.f32x2 d, a, b, c;\n"
    "mov.b64 {%0, %1}, d;\n"
    "}"
    : "+f"(c[0]), "+f"(c[1])
    : "f"(a[0]), "f"(a[1]), "f"(b[0]), "f"(b[1])
  );
}

__device__ inline
int elect_sync() {
  int pred = 0;
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
void tma_s2g(void *dst, int src, int size) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" :: "l"(dst), "r"(src), "r"(size));
}

template <typename T>
__device__ __inline__
T warp_uniform(T x) { return __shfl_sync(0xFFFF'FFFF, x, 0); }

__global__
__launch_bounds__(32, 1)
void qr32_kernel(const float *gA, float *gH, float *gTau) {
  constexpr int N = 32;

  const int lane_id = threadIdx.x;
  const int bid = blockIdx.x;

  gA += bid * N * N;
  gH += bid * N * N;
  gTau += bid * N;

  // load directly to registers
  float rA[N];
  for (int i = 0; i < N / 8; i++)
    ldg_f32x8(rA + i * 8, gA + (lane_id * N + i * 8));

  // skip last column
  #pragma unroll
  for (int col = 0; col < N-1; col++) {
    // compute reflector
    const int num_lanes = next_pow2(N-col);
    float x = rA[col];
    float tail = warp_sum(lane_id > col ? x * x : 0.0f, num_lanes);

    float x0 = __shfl_sync(0xFFFF'FFFF, x, col);
    float norm = sqrtf(x0 * x0 + tail);
    float beta = -copysignf(norm, x0);  // Hx target. opposite sign of x0
    float tau = (beta - x0) / beta * (tail > 0.0f);

    if (lane_id == col)
      gTau[col] = tau;

    float v = (lane_id == col) + (lane_id > col) * (x / (x0 - beta));
    float new_A = (lane_id < col) * x + (lane_id == col) * beta + (lane_id > col) * v;
    rA[col] = new_A * (tail > 0.0f) + x * (tail == 0.0f);

    // update trailing columns
    #pragma unroll
    for (int col_trail = col + 1; col_trail < N; col_trail++) {
      float x = rA[col_trail];
      float y = warp_sum(x * v, num_lanes);
      rA[col_trail] = x - y * (v * tau);
    }
  }

  // last column
  if (lane_id == 0)
    gTau[N-1] = 0.0f;

  for (int i = 0; i < N / 8; i++)
    stg_f32x8(gH + (lane_id * N + i * 8), rA + i * 8);
}

void qr32_launch(const float *gA, float *gH, float *gTau, int bs) {
  qr32_kernel<<<bs, 32>>>(gA, gH, gTau);
}

template <int NUM_WARPS>
__global__
__launch_bounds__(NUM_WARPS * 32, 1)
void qr176_kernel(const float *gA, float *gH, float *gTau) {
  constexpr int N = 176;
  constexpr int TB_SIZE = NUM_WARPS * 32;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int warp_id = warp_uniform(tid / 32);
  const int lane_id = tid % 32;

  gA += bid * N * N;
  gH += bid * N * N;
  gTau += bid * N;

  extern __shared__ float smem_ptr[];
  float *sA = smem_ptr;
  float *sTau = sA + N * 192;
  const int sA_addr = __cvta_generic_to_shared(sA);

  {
    // remap 16-lane warp to make computation easier
    const int warp_id = tid / 16;
    const int lane_id = tid % 16;

    // cover [N, 128]  
    #pragma unroll
    for (int i = 0; i < N / (NUM_WARPS * 2); i++) {
      float tmp[8];
      const int row = i * (NUM_WARPS * 2) + warp_id;
      ldg_f32x8(tmp, gA + (row * N + lane_id * 8));

      for (int j = 0; j < 8; j++) {
        const int col = lane_id * 8 + j;
        sA[col * 192 + row] = tmp[j];  // transpose
      }
    }
  }
  // the remaining [N, 48]
  static_assert(TB_SIZE >= N);
  if (tid < N) {
    const int row = tid;
    for (int i = 0; i < 48 / 8; i++) {
      float tmp[8];
      ldg_f32x8(tmp, gA + (row * N + 128 + i * 8));

      for (int j = 0; j < 8; j++) {
        const int col = 128 + i * 8 + j;
        sA[col * 192 + row] = tmp[j];  // transpose
      }
    }
    float tmp[4] = {};
    for (int i = 0; i < 16 / 4; i++)
      reinterpret_cast<float4 *>(sA + (tid * 192 + N + i * 4))[0] = reinterpret_cast<float4 *>(tmp)[0];
  }
  __syncthreads();

  if (warp_id == NUM_WARPS - 1) {
    // producer
    float x[6];
    float v[6];

    auto producer = [&](int col, int row_base, int num_regs) {
      float tail = 0.0f;
      #pragma unroll
      for (int i = 0; i < num_regs; i++) {
        const int row = row_base + i * 32 + lane_id;
        x[i] = sA[col * 192 + row];
        tail += (row > col) * (x[i] * x[i]);
      }
      tail = warp_sum(tail, 32);

      float x0 = sA[col * 192 + col];
      float norm = sqrtf(x0 * x0 + tail);
      float beta = -copysignf(norm, x0);  // Hx target. opposite sign of x0
      float tau = (beta - x0) / beta * (tail > 0.0f);
      float inv = 1.0f / (x0 - beta);

      if (lane_id == 0) {
        gTau[col] = tau;
        sTau[col] = tau;
      }

      #pragma unroll
      for (int i = 0; i < num_regs; i++) {
        const int row = row_base + i * 32 + lane_id;
        v[i] = (row == col) + (row > col) * (x[i] * inv);
        sA[col * 192 + row] = (row < col) * x[i] + (row == col) * beta + (row > col) * v[i];
      }

      // publish to smem
      __syncthreads();  // this also waits for the next column to be ready
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

      if (elect_sync())
        tma_s2g(gH + col * N, sA_addr + col * 192 * 4, N * 4);

      // update next column
      const int trail = col + 1;
      for (int i = 0; i < num_regs; i++) {
        const int row = row_base + i * 32 + lane_id;
        x[i] = sA[trail * 192 + row];
      }

      float y = 0.0f;
      for (int i = 0; i < num_regs; i++)
        y += x[i] * v[i];
      y = warp_sum(y, 32) * tau;
      for (int i = 0; i < num_regs; i++) {
        const int row = row_base + i * 32 + lane_id;
        sA[trail * 192 + row] = x[i] - v[i] * y;
      }
    };

    #pragma unroll
    for (int col = 0; col < 16; col++) producer(col, 0, 6);
    for (int col = 16; col < 48; col++) producer(col, 16, 5);
    for (int col = 48; col < 80; col++) producer(col, 48, 4);
    for (int col = 80; col < 112; col++) producer(col, 80, 3);
    for (int col = 112; col < 144; col++) producer(col, 112, 2);
    for (int col = 144; col < N-1; col++) producer(col, 144, 1);

    // store last column
    __syncwarp();
    asm volatile("fence.proxy.async.shared::cta;");
    if (elect_sync()) {
      const int col = N - 1;
      tma_s2g(gH + col * N, sA_addr + col * 192 * 4, N * 4);
      gTau[col] = 0.0f;
    }
  }
  else {
    // update warps
    float x[6];
    float v[6];

    auto consumer = [&](int col, int row_base, int num_regs) {
      // wait for reflector
      __syncthreads();

      // load H column
      // convert to v
      #pragma unroll
      for (int i = 0; i < num_regs; i++) {
        const int row = row_base + i * 32 + lane_id;
        x[i] = sA[col * 192 + row];
        v[i] = (row == col) + (row > col) * x[i];
      }
      float tau = sTau[col];

      // update trailing columns
      for (int trail = col + 2 + warp_id; trail < N; trail += NUM_WARPS - 1) {
        for (int i = 0; i < num_regs; i++) {
          const int row = row_base + i * 32 + lane_id;
          x[i] = sA[trail * 192 + row];
        }

        float y = 0.0f;
        for (int i = 0; i < num_regs; i++)
          y += x[i] * v[i];
        y = warp_sum(y, 32) * tau;
        for (int i = 0; i < num_regs; i++) {
          const int row = row_base + i * 32 + lane_id;
          sA[trail * 192 + row] = x[i] - v[i] * y;
        }
      }
    };

    #pragma unroll
    for (int col = 0; col < 16; col++) consumer(col, 0, 6);
    #pragma unroll
    for (int col = 16; col < 48; col++) consumer(col, 16, 5);
    #pragma unroll
    for (int col = 48; col < 80; col++) consumer(col, 48, 4);
    #pragma unroll
    for (int col = 80; col < 112; col++) consumer(col, 80, 3);
    #pragma unroll
    for (int col = 112; col < 144; col++) consumer(col, 112, 2);
    #pragma unroll
    for (int col = 144; col < N-1; col++) consumer(col, 144, 1);
  }
}

void qr176_launch(const float *gA, float *gH, float *gTau, int bs) {
  constexpr int N = 176;
  constexpr int NUM_WARPS = 8;
  auto this_kernel = qr176_kernel<NUM_WARPS>;
  const int smem_size = (N * 192 + N) * 4;
  cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  this_kernel<<<bs, NUM_WARPS * 32, smem_size>>>(gA, gH, gTau);
}
"""

load_inline(
    "linalg",
    cpp_sources=CPP_SRC,
    cuda_sources=CUDA_SRC,
    is_python_module=False,
    no_implicit_headers=True,
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=["-O3", "-std=c++17", "-Xptxas=-v"],
)


def custom_kernel(data: input_t) -> output_t:
    bs, n, _ = data.shape

    H = torch.empty_like(data)
    tau = data.new_empty(bs, n)

    if n == 32:
        torch.ops.linalg.qr32(data, H, tau)
        return H, tau
    if n == 176:
        H = H.transpose(1, 2)
        torch.ops.linalg.qr176(data, H, tau)
        return H, tau

    return torch.geqrf(data)
