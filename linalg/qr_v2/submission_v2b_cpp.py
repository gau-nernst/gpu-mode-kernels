#!POPCORN leaderboard qr_v2
#!POPCORN gpu B200

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

CPP_SRC = r"""
#include <ATen/ATen.h>
#include <torch/library.h>

void qr32_launch(const float *gA, float *gH, float *gTau, int bs);
void qr176_launch(const float *gA, float *gH, float *gTau, float *gV, int *flags, int bs);

void qr32(const at::Tensor& A, at::Tensor& H, at::Tensor& tau) {
  const int bs = A.size(0);
  const float *gA = A.data_ptr<float>();
        float *gH = H.data_ptr<float>();
        float *gTau = tau.data_ptr<float>();
  qr32_launch(gA, gH, gTau, bs);
}

void qr176(const at::Tensor& A, at::Tensor& H, at::Tensor& tau, at::Tensor& V, at::Tensor& flags) {
  const int bs = A.size(0);
  const float *gA = A.data_ptr<float>();
        float *gH = H.data_ptr<float>();
        float *gTau = tau.data_ptr<float>();
        float *gV = V.data_ptr<float>();
  int *flags_ = flags.data_ptr<int>();
  qr176_launch(gA, gH, gTau, gV, flags_, bs);
}

TORCH_LIBRARY(linalg, m) {
  m.def("qr32(Tensor A, Tensor(a!) H, Tensor(b!) tau) -> ()", &qr32);
  m.def("qr176(Tensor A, Tensor(a!) H, Tensor(b!) tau, Tensor(c!) V, Tensor(d!) flags) -> ()", &qr176);
}
"""

CUDA_SRC = r"""
__device__ inline
float warp_sum(float val, int size) {
  for (int s = size / 2; s > 0; s >>= 1)
    val += __shfl_xor_sync(0xFFFF'FFFF, val, s);
  return val;
}

__device__ __host__
constexpr int cdiv(int a, int b) { return (a + b - 1) / b; }

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
float sqrt_fast(float x) {
  float y;
  asm volatile("sqrt.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ inline
float rcp_fast(float x) {
  float y;
  asm volatile("rcp.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
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
void add_f32x2(float *c, const float *a, const float *b) {
  asm volatile(
    "{"
    ".reg .b64 a, b, c;\n"
    "mov.b64 a, {%2, %3};\n"
    "mov.b64 b, {%4, %5};\n"
    "add.rn.f32x2 c, a, b;\n"
    "mov.b64 {%0, %1}, c;\n"
    "}"
    : "=f"(c[0]), "=f"(c[1])
    : "f"(a[0]), "f"(a[1]), "f"(b[0]), "f"(b[1])
  );
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
void mbarrier_init(int mbar_addr, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(mbar_addr), "r"(count));
}

__device__ inline
void mbarrier_arrive_release(int mbar_addr) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" :: "r"(mbar_addr) : "memory");
}

__device__ inline
void mbarrier_arrive_relaxed(int mbar_addr) {
  asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" :: "r"(mbar_addr) : "memory");
}

__device__ inline
void mbarrier_arrive_expect_tx(int mbar_addr, int size) {
  asm volatile("mbarrier.arrive.expect_tx.relaxed.cluster.shared::cluster.b64 _, [%0], %1;"
              :: "r"(mbar_addr), "r"(size) : "memory");
}

__device__
void mbarrier_wait_acquire(int mbar_addr, int phase) {
  int ticks = 0x989680;  // this is optional
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
void tma_s2g(void *dst, int src, int size) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
              :: "l"(dst), "r"(src), "r"(size));
}

__device__ inline
void tma_s2s(int dst, int src, int size, int mbar) {
  asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
              :: "r"(dst), "r"(src), "r"(size), "r"(mbar));
}

__device__ inline
void st_async_f32(int dst, float x, int mbar) {
  asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
              :: "r"(dst), "f"(x), "r"(mbar));
}

__device__ inline
void st_release_gpu(int *dst, int val) {
  asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(dst), "r"(val));
}

__device__ inline
int ld_relaxed_gpu(const int *src) {
  int val;
  asm volatile("ld.relaxed.gpu.global.L1::no_allocate.b32 %0, [%1];" : "=r"(val) : "l"(src));
  return val;
}

__device__ inline
void wait_flag(const int *flag) {
  while (ld_relaxed_gpu(flag) == 0) {
    __nanosleep(64);
  }
  asm volatile("fence.acquire.gpu;" ::: "memory");
}

template <typename T>
__device__ __inline__
T warp_uniform(T x) { return __shfl_sync(0xFFFF'FFFF, x, 0); }

__global__
__launch_bounds__(4 * 32, 1)
void qr32_kernel(const float *gA, float *gH, float *gTau) {
  constexpr int N = 32;

  const int tid = threadIdx.x;
  const int bid = blockIdx.x;
  const int warp_id = warp_uniform(tid / 32);
  const int lane_id = tid % 32;

  gA += bid * N * N;
  gH += bid * N * N;
  gTau += bid * N;

  extern __shared__ float smem_ptr[];
  float *sV = smem_ptr;
  float *sTau = sV + N * N;
  const int mbar_base = __cvta_generic_to_shared(sTau + N);

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < N; i++)
      mbarrier_init(mbar_base + i * 8, 32);
  }
  __syncthreads();

  // load directly to registers
  float x[8];
  ldg_f32x8(x, gA + (lane_id * N + warp_id * 8));

  for (int panel = 0; panel < warp_id; panel++) {
    for (int i = 0; i < 8; i++) {
      // wait for reflector
      const int col = panel * 8 + i;
      mbarrier_wait_acquire(mbar_base + col * 8, 0);

      float neg_tau = -sTau[col];  // negate
      float v[2];  // duplicate for f32x2 math later
      v[0] = v[1] = sV[col * N + lane_id];

      // update warp-private columns
      // f32x2 math: updates 2 columns at once
      for (int j = 0; j < 4; j++) {
        float y[2];
        mul_f32x2(y, &x[j*2], v);
        y[0] = warp_sum(y[0], 32) * neg_tau;
        y[1] = warp_sum(y[1], 32) * neg_tau;
        fma_f32x2(&x[j*2], v, y);
      }
    }
  }

  // warp i's turn to compute the reflector
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const int col = warp_id * 8 + i;

    // special case for last column. don't modify x
    if (col == N - 1) {
      if (lane_id == 0)
        sTau[col] = 0.0f;
      break;
    }

    float tail = warp_sum(lane_id > col ? x[i] * x[i] : 0.0f, 32);
    float x0 = __shfl_sync(0xFFFF'FFFF, x[i], col);
    float norm = sqrt_fast(x0 * x0 + tail);
    float beta = -copysignf(norm, x0);  // Hx target. opposite sign of x0
    float tau = (beta - x0) * rcp_fast(beta) * (tail > 0.0f);

    if (lane_id == 0)
      sTau[col] = tau;

    // using div here is faster than rcp_fast?
    float v = (lane_id == col) + (lane_id > col) * (x[i] / (x0 - beta));
    float new_x = (lane_id < col) * x[i] + (lane_id == col) * beta + (lane_id > col) * v;
    x[i] = new_x * (tail > 0.0f) + x[i] * (tail == 0.0f);
    sV[col * N + lane_id] = v;
    mbarrier_arrive_release(mbar_base + col * 8);

    // update trailing columns
    #pragma unroll
    for (int j = i + 1; j < 8; j++) {
      float y = warp_sum(x[j] * v, 32);
      x[j] -= y * (v * tau);
    }
  }
  if (lane_id == 0)
    stg_f32x8(gTau + warp_id * 8, sTau + warp_id * 8);
  stg_f32x8(gH + (lane_id * N + warp_id * 8), x);
}

void qr32_launch(const float *gA, float *gH, float *gTau, int bs) {
  constexpr int N = 32;
  const int smem_size = (N * N + N) * 4 + N * 8;
  cudaFuncSetAttribute(qr32_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  qr32_kernel<<<bs, 4 * 32, smem_size>>>(gA, gH, gTau);
}

// 2-SM. even split
template <int N, int NUM_WARPS>
__global__
__launch_bounds__(NUM_WARPS * 32, 1)
void qr_2sm_kernel(const float *gA, float *gH, float *gTau, float *gV, int *flags) {
  const int tid = threadIdx.x;
  const int rank = blockIdx.x;
  const int batch_id = blockIdx.y;
  const int warp_id = warp_uniform(tid / 32);
  const int lane_id = tid % 32;

  gA += batch_id * N * N;
  gH += batch_id * N * N;
  gTau += batch_id * N;
  gV += batch_id * N * N;
  flags += batch_id * N;

  // rank0: only reads from smem
  // rank1: for the 1st half, reads from gmem.
  //        for the 2nd half, reads from smem

  // smem only needs to hold half of total columns
  extern __shared__ float smem_ptr[];
  float *sV = smem_ptr;
  float *sTau = sV + N * (N/2);
  const int sV_addr = __cvta_generic_to_shared(sV);
  const int sTau_addr = sV_addr + N * (N/2) * 4;
  const int mbar_base = sTau_addr + (N/2) * 4;

  if (warp_id == 0 && elect_sync()) {
    for (int i = 0; i < N/2; i++)
      mbarrier_init(mbar_base + i * 8, 1);
  }
  __syncthreads();
  
  // each warp holds [N,8] panel
  constexpr int NUM_REGS = cdiv(N, 32);
  float x[NUM_REGS][8];
  for (int i = 0; i < NUM_REGS; i++) {
    const int row = i * 32 + lane_id;
    const int col = rank * (N/2) + warp_id * 8;
    if (row < N)
      ldg_f32x8(x[i], gA + (row * N + col));
    else
      memset(x[i], 0, 8 * 4);
  }

  // rank1: data from remote rank in gmem
  if (rank == 1) {
    for (int panel = 0; panel < (N/2) / 8; panel++) {
      if (tid == 0)
        wait_flag(flags + panel);
      __syncthreads();

      float v[NUM_REGS][8];
      float tau[8];
      ldg_f32x8(tau, gTau + panel * 8);
      for (int i = 0; i < NUM_REGS; i++) {
        const int row = i * 32 + lane_id;
        const int col = panel * 8;
        if (row < N)
          ldg_f32x8(v[i], gH + (row * N + col));
        else
          memset(v[i], 0, 8 * 4);
      }

      for (int i = 0; i < 8; i++) {
        // wait for reflector
        const int col = panel * 8 + i;

        for (int j = 0; j < NUM_REGS; j++) {
          const int row = j * 32 + lane_id;
          v[j][i] = (row == col) + (row > col) * v[j][i];
        }

        // update warp-private columns
        for (int j = 0; j < 8; j++) {
          float y = 0.0f;
          for (int k = 0; k < NUM_REGS; k++)
            y += x[k][j] * v[k][i];
          y = warp_sum(y, 32) * tau[i];
          for (int k = 0; k < NUM_REGS; k++)
            x[k][j] -= v[k][i] * y;
        }
      }
    }
  }

  // both ranks: data from local rank in smem
  for (int panel = 0; panel < warp_id; panel++) {
    for (int i = 0; i < 8; i++) {
      // wait for reflector
      const int local_col = panel * 8 + i;
      mbarrier_wait_acquire(mbar_base + local_col * 8, 0);

      float neg_tau = -sTau[local_col];  // negate
      float v[NUM_REGS][2];  // duplicate for f32x2 math later
      for (int j = 0; j < NUM_REGS; j++) {
        const int row = j * 32 + lane_id;
        v[j][0] = row < N ? sV[local_col * N + row] : 0.0f;
        v[j][1] = v[j][0];
      }

      // update warp-private columns
      // f32x2 math: updates 2 columns at once
      for (int j = 0; j < 4; j++) {
        float y[2] = {};
        for (int k = 0; k < NUM_REGS; k++)
          fma_f32x2(y, &x[k][j*2], v[k]);
        y[0] = warp_sum(y[0], 32) * neg_tau;  // TODO: use f32x2 math here too
        y[1] = warp_sum(y[1], 32) * neg_tau;
        for (int k = 0; k < NUM_REGS; k++)
          fma_f32x2(&x[k][j*2], v[k], y);
      }
    }
  }

  // warp i's turn to compute the reflector
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    const int local_col = warp_id * 8 + i;
    const int col = rank * (N/2) + local_col;

    // special case for last column. don't modify x
    if (col == N - 1) {
      if (lane_id == 0)
        sTau[local_col] = 0.0f;
      break;
    }

    float tail = 0.0f;
    float x0 = 0.0f;
    for (int j = 0; j < NUM_REGS; j++) {
      const int row = j * 32 + lane_id;
      tail += (row > col) * x[j][i] * x[j][i];
      x0 += row == col ? x[j][i] : 0.0f;
    }
    tail = warp_sum(tail, 32);
    x0 = __shfl_sync(0xFFFF'FFFF, x0, col % 32);

    float norm = sqrt_fast(x0 * x0 + tail);
    float beta = -copysignf(norm, x0);  // Hx target. opposite sign of x0
    float tau = (beta - x0) * rcp_fast(beta) * (tail > 0.0f);
    float inv = tail > 0.0f ? rcp_fast(x0 - beta) : 0.0f;

    if (lane_id == 0)
      sTau[local_col] = tau;

    float v[NUM_REGS];
    for (int j = 0; j < NUM_REGS; j++) {
      const int row = j * 32 + lane_id;
      v[j] = (row == col) + (row > col) * (x[j][i] * inv);
      float new_x = (row < col) * x[j][i] + (row == col) * beta + (row > col) * v[j];
      x[j][i] = tail > 0.0f ? new_x : x[j][i];

      if (row < N)
        sV[local_col * N + row] = v[j];
    }
    // publish reflector. no wait
    __syncwarp();
    if (elect_sync()) {
      mbarrier_arrive_release(mbar_base + local_col * 8);
    }

    // update warp-private columns
    for (int j = i + 1; j < 8; j++) {
      float y = 0.0f;
      for (int k = 0; k < NUM_REGS; k++)
        y += x[k][j] * v[k];
      y = warp_sum(y, 32) * tau;
      for (int k = 0; k < NUM_REGS; k++)
        x[k][j] -= v[k] * y;
    }
  }

  // store result
  {
    const int local_col = warp_id * 8;
    const int col = rank * (N/2) + local_col;
    if (lane_id == 0)
      stg_f32x8(gTau + col, sTau + local_col);

    for (int i = 0; i < NUM_REGS; i++) {
      const int row = i * 32 + lane_id;
      if (row < N)
        stg_f32x8(gH + (row * N + col), x[i]);
    }
    __syncwarp();
    if (lane_id == 0)
      st_release_gpu(flags + rank * (N/2) / 8 + warp_id, 1);
  }
}

void qr176_launch(const float *gA, float *gH, float *gTau, float *gV, int *flags, int bs) {
  constexpr int N = 176;
  constexpr int NUM_WARPS = (N / 8) / 2;
  const int smem_size = (N + 1) * (N/2) * 4 + (N/2) * 8;
  auto this_kernel = qr_2sm_kernel<N, NUM_WARPS>;
  cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  dim3 grid(2, bs);
  this_kernel<<<grid, NUM_WARPS * 32, smem_size>>>(gA, gH, gTau, gV, flags);
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
        V = torch.empty_like(data)
        flags = torch.zeros(bs, n, dtype=torch.int32, device="cuda")
        torch.ops.linalg.qr176(data, H, tau, V, flags)
        return H, tau

    return torch.geqrf(data)
