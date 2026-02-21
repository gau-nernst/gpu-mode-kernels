# NVFP4 GEMM - Worklog

- `submission_v0.py`: basic tcgen05 kernel in PTX. Add cache hint.
- `submission_v0b.py`: in epilogue, replace `.32x32b` with `.16x256b`
- `submission_v0c.py`: swap A and B, so that M > N = 128.
- `submission_v1.py`: SPLIT-K with FP32 atomic add. The result stays in FP32 -> probably not valid
- `submission_v1b.py`: separate reduction kernel.
- `submission_v1c.py`: FP32 atomic add in gmem, cluster sync, then FP32->FP16 cast.
- `submission_v1d.py`: FP32 store to gmem, cluster sync, final reduction + FP32->FP16 cast.
- `submission_v1e.py`: cluster sync, FP32 store to smem, cluster sync, final reduction + FP32->FP16 cast. Slower v1d, probably because there are 2 cluster syncs.
- `submission_v2.py`: enable BLOCK_M=64 and BLOCK_N=64
- `submission_v2b.py`: enable BLOCK_N=32 via `cp.async.ca` + `cp.async.mbarrier.arrive`. Only faster for benchmark.1 (due to limited no. of threadblocks).
- `submission_v3.py`: separate epilogue warps. Use named barrier (`bar.sync`) for epilogue.
- `submission_v3b.py`: unroll the first `NUM_STAGES` of TMA issues without waiting for MMA. Compute mbarrier phase from `iter_k` instead of flipping it.
- `submission_v4.py`: combine v1 and v3b -> SPLIT-K and BLOCK_N=64 support.

## Leaderboard results

| (M,N,K)        | (128,7168,16384) | (128,4096,7168) | (128,7168,2048) | Geomean |
|----------------|------------------|-----------------|-----------------|---------|
| SOL            | 8.994            | 2.354           | 1.333           | 3.04    |
| CuteDSL_v0     | 96.6             | 51.6            | 36.9            | 56.87   |
| scaled_mm      | 22.4             | 12.5            | 8.49            | 13.35   |
| v0             | 23.8             | 12.3            | 8.3             | 13.44   |
| v0b            | 23.4             | 12.4            | 8.28            | 13.39   |
| v0c            | 22.6             | 12.4            | 8.25            | 13.22   |
| v1             | 18.7             | 8.47            | 8.25            | 10.93   |
| v1b            | 21.8             | 12.4            | 11.9            | 14.76   |
| v2             | 20               | 10.3            | 7.74            | 11.68   |
| v2b            | 33.9             | 10.9            | 10.3            | 15.61   |
| v3             | 19               | 10.3            | 7.28            | 11.25   |
| v3b            | 18.8             | 10.3            | 6.78            | 10.95   |
| v4             | 18.2             | 8.3             | 6.63            | 10.01   |
| mix (4+3b)     | 18.3             | 8.29            | 6.54            | 9.97    |
