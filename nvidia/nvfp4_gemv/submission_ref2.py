#!POPCORN leaderboard nvfp4_gemv

from pathlib import Path

import torch
from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

# https://github.com/NVIDIA/cutlass/blob/v4.2.1/examples/72_blackwell_narrow_precision_gemm/72b_blackwell_nvfp4_nvfp4_gemm.cu
CUDA_SRC = r"""
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/packed_stride.hpp"

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/CUDAUtils.h>
#include <ATen/cuda/CUDAContext.h>

#define STRINGIFY(x) #x
#define CUTLASS_CHECK(call) \
  do {                      \
    auto status = call;     \
    TORCH_CHECK(status == cutlass::Status::kSuccess, STRINGIFY(call), ": ", status, " - ", cutlassGetStatusString(status)); \
  } while (0)

using namespace cute;

using ElementAB  = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC   = cutlass::half_t;
using ElementAcc = float;

constexpr int AlignmentAB = 128 / 4;  // 32
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;  // 8

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;

using ArchTag       = cutlass::arch::Sm100;
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;

// Kernel Perf config
using MmaTileShape = Shape<_128,_128,_256>;
using ClusterShape = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
  ArchTag, OperatorClass,
  MmaTileShape, ClusterShape,
  cutlass::epilogue::collective::EpilogueTileAuto,
  ElementAcc, ElementAcc,
  ElementC, LayoutCTag, AlignmentC,
  ElementC, LayoutCTag, AlignmentC,
  cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
  ArchTag, OperatorClass,
  ElementAB, LayoutATag, AlignmentAB,
  ElementAB, LayoutBTag, AlignmentAB,
  ElementAcc,
  MmaTileShape, ClusterShape,
  cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
  cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
  Shape<int, int, int, int>,
  CollectiveMainloop,
  CollectiveEpilogue,
  void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

void gemv(
  const at::Tensor& A,
  const at::Tensor& B,
  const at::Tensor& SFA,
  const at::Tensor& SFB,
        at::Tensor& C
) {
  const int M = A.size(0);
  const int N = 128;
  const int K = A.size(1) * 2;
  const int L = A.size(2);

  using ABType = typename ElementAB::DataType;
  using SFType = typename ElementAB::ScaleFactorType;

  auto stride_A = cutlass::make_cute_packed_stride(typename GemmKernel::StrideA{}, {M, K, L});
  auto stride_B = cutlass::make_cute_packed_stride(typename GemmKernel::StrideB{}, {N, K, L});
  auto stride_C = cutlass::make_cute_packed_stride(typename GemmKernel::StrideC{}, {M, N, L});

  using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(M, N, K, L));
  auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(M, N, K, L));

  auto *A_ptr   = reinterpret_cast<const ABType *>(A.data_ptr());
  auto *B_ptr   = reinterpret_cast<const ABType *>(B.data_ptr());
  auto *SFA_ptr = reinterpret_cast<const SFType *>(SFA.data_ptr());
  auto *SFB_ptr = reinterpret_cast<const SFType *>(SFB.data_ptr());
  auto *C_ptr   = reinterpret_cast<ElementC *>(C.data_ptr());

  typename Gemm::Arguments arguments{
    cutlass::gemm::GemmUniversalMode::kGemm,
    {M, N, K, L},
    {
      A_ptr, stride_A,
      B_ptr, stride_B,
      SFA_ptr, layout_SFA,
      SFB_ptr, layout_SFB,
    },
    {
      {1.0f, 0.0f},  // alpha and beta
      C_ptr, stride_C,
      C_ptr, stride_C,
    }
  };

  Gemm gemm;
  //CUTLASS_CHECK(gemm.can_implement(arguments));

  //long workspace_size = Gemm::get_workspace_size(arguments);
  //at::Tensor workspace = at::empty({workspace_size}, A.options().dtype(at::kByte));
  auto stream = at::cuda::getCurrentCUDAStream();

  //CUTLASS_CHECK(gemm.initialize(arguments, workspace.data_ptr(), stream));
  CUTLASS_CHECK(gemm.initialize(arguments, 0, stream));
  CUTLASS_CHECK(gemm.run(stream));
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
    ],
)


def custom_kernel(data: input_t) -> output_t:
    # a:   [  M, K, L],                   natural shape [L,   M, K]
    # b:   [128, K, L],                   natural shape [L, 128, K] - only the 1st row is used
    # sfa: [32, 4, rest_m, 4, rest_k, L], natural shape [L, rest_m, rest_k, 32, 4, 4]
    # sfb: [32, 4,      1, 4, rest_k, L], natural shape [L,      1, rest_k, 32, 4, 4]
    # c:   [  M, 1, L],                   natural shape [L, M, 1]
    a, b, _, _, sfa, sfb, c_ref = data

    M = a.shape[0]
    N = 128
    K = a.shape[1] * 2
    L = a.shape[2]

    big_c = c_ref.new_empty(L, M, N)
    torch.ops.my_module.gemv(a, b, sfa, sfb, big_c)

    if False:
        path = Path(f"profile_data/{M=}_{K=}_{L=}.json.gz")
        if not path.exists():
            a.new_zeros(int(1e8), dtype=torch.uint8)  # 100 MB

            with torch.profiler.profile() as prof:
                torch.ops.my_module.gemv(a, b, sfa, sfb, big_c)

            path.parent.mkdir(exist_ok=True)
            prof.export_chrome_trace(str(path))

    return big_c[..., :1].permute(1, 2, 0)  # convert to [M, 1, L]
