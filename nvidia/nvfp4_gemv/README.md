# NVFP4 GEMV - Worklog

- `submission_ref.py` using built-in `torch._scaled_mm()`, which uses Cutlass `cutlass3x_sm100_bstensorop_s256x192x64gemm_block_scaled_ue4m3xf4_ue4m3xf4_f32_f16_f16_256x192x256_0_tnn_align32_o_vs16_2sm_bias_f16_relu`. This has Python-loop overheads for L>1.
- `submission_ref2.py` instantiates Cutlass kernel directly to support batched cases.
- `submission_t0.py` is a Triton-based version using `cvt.rn.f16x2.e2m1x2` PTX. Perf is not great.
- `submission_v0.py` re-implements the same idea, but in CUDA, using CUDA intrinsics. The compiler is a bit wonky.
- `submission_v1.py` implements v0 again with PTX (similar to Triton version).
- `submission_v1b.py` adds `.cs` (cache streaming) to A global loads.
- `submission_v1c.py` uses `fma.rn.f32x2`. ncu shows that stall math pipe throttle > stall long scoreboard -> CUDA core is too slow, we need tensor cores.
- `submission_v2.py` coarsens along M by 4. This reduces stall math pipe (stall long scoreboard is the highest now), but reduces occupancy -> slower than v1c.
- `submission_v2b.py` add pipelining with `cp.async`. Without thread coarsening in v2, this is actually slower than v1c.
- `submission_v2c.py` increases `THREAD_M` coarsening to 8. Doesn't seem to speed up much. Stall math pipe is still the biggest bottleneck -> we need to move on to tensor cores.
- `submission_v2d.py` uses `f16x2` instead of `f32x2`. Alleviate stall math pipe. Stall long scoreboard is back at the top -> we need TMA. `cp.async.cg` + `cache-policy` results in illegal instruction. See https://github.com/meta-pytorch/applied-ai/pull/32/files.
- `submission_v2f.py` don't use `cp.async`. Use `__ldcs` to prevent cache pollution.
- `submission_v2g.py` double buffering w/ rmem.
- `submission_v3.py` uses FP16 `mma.m16n8k16`. Bank conflicts LOL.
- `submission_v3b.py` unpacks FP4 to FP16, then store back to smem. Still a lot of bank conflicts...
- `submission_v4.py` adds basic TMA for A and SFA, no pipelining.
- `submission_v4b.py` adds pipelining for TMA. Use `cp.async.bulk` for B and SFB
- `submission_v4c.py` uses `cp.async.bulk` for A and SFA as well, instead of `cp.async.bulk.tensor`.
- `submission_v4d.py` warp specialization.
- `submission_v4e.py` persistent kernel.

## Leaderboard results

| (M,N,K)    | (7168,16384,1) | (4096,7168,8) | (7168,2048,4) | Geomean |
|------------|----------------|---------------|---------------|---------|
| SOL        | 8.58           | 17.17         | 4.30          | 8.59    |
| scaled_mm  | 31.8           | 135           | 44.2          | 57.46   |
| cutlass    | 32.5           | 40            | 18            | 28.60   |
| v1         | 29.7           | 50.9          | 18.5          | 30.35   |
| v1b        | 28.8           | 49.3          | 18.4          | 29.67   |
| v1c        | 28.7           | 47.5          | 18.3          | 29.22   |
| v2b        | 24.6           | 39.4          | 16.4          | 25.14   |
| v2c        | 24.5           | 39.3          | 16.6          | 25.19   |
| v2d        | 20.5           | 34.9          | 14.3          | 21.71   |
| v2e        | 20.7           | 34.9          | 14.3          | 21.78   |
| v2f        | 19.3           | 30.7          | 12.4          | 19.44   |
| v2fp       | 19.2           | 30            | 12.4          | 19.26   |
| 2fp2       | 20.6           | 28.7          | 12.5          | 19.48   |
| v2fw       | 19.2           | 30.8          | 12.4          | 19.43   |
| v2h        | 18.4           | 28            | 12.6          | 18.65   |
| v4c        | 20.3           | 31.2          | 14.4          | 20.89   |
| v4e        | 22.5           | 31.7          | 15.4          | 22.23   |
