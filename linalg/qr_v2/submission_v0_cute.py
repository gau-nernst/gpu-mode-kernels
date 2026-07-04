#!POPCORN leaderboard qr_v2
#!POPCORN gpu B200

import functools

import cutlass
import torch
from cutlass import Float32, cute
from cutlass._mlir.dialects import llvm
from cutlass.cute.runtime import make_fake_tensor
from cutlass.cutlass_dsl import dsl_user_op
from task import input_t, output_t


@dsl_user_op
def copysign(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    out = llvm.inline_asm(
        Float32.mlir_type,
        [a.ir_value(loc=loc, ip=ip), b.ir_value(loc=loc, ip=ip)],
        "copysign.f32  $0, $1, $2;",
        "=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(out)


class Kernel32:
    num_warps = 16
    n = 32

    @cute.jit
    def __call__(self, gA: cute.Tensor, gH: cute.Tensor, gTau: cute.Tensor):
        bs, _, _ = gA.shape
        grid = (bs, 1, 1)
        block = (32 * self.num_warps, 1, 1)
        self.kernel(gA, gH, gTau).launch(grid=grid, block=block)

    @cute.kernel
    def kernel(self, gA: cute.Tensor, gH: cute.Tensor, gTau: cute.Tensor):
        bid, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        warp_id = cute.arch.make_warp_uniform(tid // 32)
        lane_id = tid % 32

        n = self.n
        num_warps = self.num_warps

        smem = cutlass.utils.SmemAllocator()

        # sA_layout = cute.make_layout((n, n))  # col-major
        sA_layout = cute.make_layout((n, n), stride=(n + 1, 1))  # padding
        sA = smem.allocate_tensor(Float32, sA_layout)
        sV = smem.allocate_tensor(Float32, cute.make_layout((n, 2)))
        sT = smem.allocate_array(Float32, 2)

        # load gmem->smem
        for i in cutlass.range_constexpr(n // num_warps):
            row = i * num_warps + warp_id
            sA[row, lane_id] = gA[bid, row, lane_id]
        cute.arch.sync_threads()

        if tid == 0:
            gTau[bid, n - 1] = 0.0

        # peeled 1st column. compute reflector
        if warp_id == 0:
            x = sA[lane_id, 0] if lane_id > 0 else Float32(0.0)
            x0 = sA[0, 0]
            tail = cute.arch.warp_reduction_sum(x * x)

            tau = Float32(0.0)
            if tail > 0.0:
                norm = cute.math.sqrt(x0 * x0 + tail, fastmath=True)
                beta = -copysign(x0, norm)  # opposite sign of x0
                tau = (beta - x0) / beta
                v = x / (x0 - beta)
                sA[lane_id, 0] = beta if lane_id == 0 else v
                sV[lane_id, 0] = 1.0 if lane_id == 0 else v

            if lane_id == 0:
                sT[0] = tau
                gTau[bid, 0] = tau

        cute.arch.sync_threads()

        # skip last column
        for col in cutlass.range_constexpr(1, n - 1):
            # read the previous v and tau
            v = sV[lane_id, (col - 1) % 2]
            tau = sT[(col - 1) % 2]

            if warp_id == 0:
                # warp0: update current column and compute reflector
                x = sA[lane_id, col]
                y = cute.arch.warp_reduction_sum(x * v)
                x -= y * tau * v
                sA[lane_id, col] = x

                # compute reflector
                x_tail = x if lane_id > col else Float32(0.0)
                x0 = cute.arch.shuffle_sync(x, col)
                tail = cute.arch.warp_reduction_sum(x_tail * x_tail)

                tau = Float32(0.0)
                if tail > 0.0:
                    norm = cute.math.sqrt(x0 * x0 + tail, fastmath=True)
                    beta = -copysign(x0, norm)  # opposite sign of x0
                    tau = (beta - x0) / beta
                    v = x / (x0 - beta)

                    if lane_id < col:
                        sV[lane_id, col % 2] = 0.0
                    elif lane_id == col:
                        sV[lane_id, col % 2] = 1.0
                        sA[lane_id, col] = beta
                    else:
                        sV[lane_id, col % 2] = v
                        sA[lane_id, col] = v

                if lane_id == 0:
                    sT[col % 2] = tau
                    gTau[bid, col] = tau

            else:
                # other warps: update remaining columns
                for col_other in cutlass.range(col + 1 + (warp_id - 1), n, num_warps - 1, unroll=1):
                    x = sA[lane_id, col_other]
                    y = cute.arch.warp_reduction_sum(x * v)
                    sA[lane_id, col_other] = x - tau * y * v

            cute.arch.sync_threads()

        # update last column
        if warp_id == 0:
            # read the previous v and tau
            col = n - 1
            v = sV[lane_id, (col - 1) % 2]
            tau = sT[(col - 1) % 2]

            x = sA[lane_id, col]
            y = cute.arch.warp_reduction_sum(x * v)
            x -= y * tau * v
            sA[lane_id, col] = x

        cute.arch.sync_threads()

        # store smem->gmem
        for i in cutlass.range_constexpr(n // num_warps):
            row = i * num_warps + warp_id
            gH[bid, row, lane_id] = sA[row, lane_id]

    @functools.cache
    @staticmethod
    def compile():
        bs = cute.sym_int()
        n = Kernel32.n
        gA = make_fake_tensor(Float32, (bs, n, n), (n * n, n, 1), assumed_align=16)
        gH = make_fake_tensor(Float32, (bs, n, n), (n * n, n, 1), assumed_align=16)
        gTau = make_fake_tensor(Float32, (bs, n), (n, 1), assumed_align=16)
        return cute.compile(Kernel32(), gA, gH, gTau)


def custom_kernel(data: input_t) -> output_t:
    bs, n, _ = data.shape

    if n == 32:
        H = torch.empty_like(data)
        taus = data.new_empty(bs, n)
        Kernel32.compile()(data, H, taus)
        return H, taus

    return torch.geqrf(data)
