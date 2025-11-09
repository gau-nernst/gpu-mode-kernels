#!POPCORN leaderboard histogram_v2

import triton
import triton.language as tl
from task import input_t, output_t


@triton.jit
def kernel(
    data_ptr,  # (size,)
    output_ptr,  # (256,)
    size,
    BLOCK_SIZE: tl.constexpr,
    NUM_BINS: tl.constexpr = 256,
):
    pid = tl.program_id(0)
    num_pids = tl.num_programs(0)

    acc = tl.zeros((NUM_BINS,), dtype=tl.int32)

    num_iters = tl.cdiv(size, BLOCK_SIZE * num_pids)
    for iter_id in range(num_iters):
        offs = iter_id * (num_pids * BLOCK_SIZE) + (pid * BLOCK_SIZE) + tl.arange(0, BLOCK_SIZE)
        mask = offs < size
        data = tl.load(data_ptr + offs, mask, other=0).to(tl.int32)  # tl.histogram() doesn't work with uint8
        acc += tl.histogram(data, NUM_BINS)  # old triton doesn't have mask for histogram

    # NOTE: output_ptr is i64 type
    tl.atomic_add(output_ptr + tl.arange(0, NUM_BINS), acc)

    # compensation since we use 0 for masked elements
    if pid == 0:
        compensate = size - num_iters * BLOCK_SIZE * num_pids
        tl.atomic_add(output_ptr, compensate)


def custom_kernel(data: input_t) -> output_t:
    data, output = data

    BLOCK_SIZE = 2048
    num_blocks = 264
    output.zero_()
    kernel[(num_blocks,)](data, output, data.shape[0], BLOCK_SIZE)

    return output
