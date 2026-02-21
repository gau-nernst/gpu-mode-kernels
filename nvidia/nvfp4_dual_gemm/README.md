# NVFP4 Dual GEMM - Worklog

- `submission_v1.py`: read A tile and B1+B2 tile. Do A @ B1 and A @ B2 in a single threadblock. Epilogue computes gating.
- `submission_v1b.py`: use activation stationary to avoid reloading A between MMA with B1 and B2.
- `submission_v1c.py`: use smaller `tcgen05.ld` (I think it helps because it pipelines CUDA core computation with `tcgen05.ld` + `st.global`).
- `submission_v1d.py`: add `cp.async.bulk.prefetch` for SFA/SFB1/SFB2.
- `submission_v1e.py`: make `bid` and `warp_id` warp uniform (via `__shfl_sync()`). Carefully place `elect_sync()`. Check SASS to avoid `R2UR`.
- `submission_v1f.py`: interleave `tcgen05.cp` and `tcgen05.mma`.
- `submission_v1g.py`: allocate TMEM inside MMA warp -> TMA warp doesn't need to wait for TMEM alloc. Simplify asm generation `tcgen05.ld` (this actually gives a good boost for `BLOCK_N=64` case). Add `cutlass` to kernel name (this also gives a good boost).
- `submission_v1h.py`: use TMA for C store.
- `submission_v1i.py`: TMA multicast (WIP)
- `submission_v2.py`: 2-SM MMA. We still issue 1 MMA for each A @ B1 and A @ B2 separately.
- `submission_v2b.py`: add MCAST_B=2 for M=512, so that we don't have reuse of B across CTAs. Doesn't help.
- `submission_v3.py`: Threadblock cluster, where CTA0 does A @ B1, and CTA1 does A @ B2. In epilogue, they exchange MMA result using `st.async`.

Notes:
- Using tensor map for SF is slower
- TMA for C store is still slower than rmem->gmem store. Maybe I'm doing something wrong.
- Tried peek mbarrier, like CuteDSL, but only see slowdown. Maybe I'm doing something wrong.

TODO:
- Multicast. We can do multicast in M and N modes. When doing multicast, each CTA rank issues TMA for a portion of the tile. Need to wait for relevant CTAs to finish before loading the next tile. If we multicast B, one CTA can do B1, the other CTA does B2.

## Leaderboard results

Unit is us.

| (M,N,K)          | (256,4096,7168) | (512,4096,7168) | (256,3072,4096) | (512,3072,7168) | Geomean |
|------------------|-----------------|-----------------|-----------------|-----------------|---------|
| SOL              | 4.708           | 8.714           | 2.125           | 6.535           | 4.886   |
| ref              | 40.6            | 43.2            | 32.9            | 41.4            | 39.31   |
| ref (compiled)   | 32.4            | 33.2            | 24.7            | 32.5            | 30.48   |
| v1               | 16.5            | 18.6            | 12.3            | 18.5            | 16.26   |
| v1b              | 15.6            | 18.5            | 11.0            | 18.5            | 15.57   |
| v1c              | 14.7            | 18.5            | 10.3            | 17.5            | 14.88   |
| v1d              | 14.6            | 18.5            | 10.3            | 16.7            | 14.68   |
| v1e              | 14.4            | 18.5            | 10.3            | 16.6            | 14.61   |
| v1f              | 14.4            | 18.2            | 10.3            | 16.5            | 14.53   |
| v1g              | 14.4            | 16.6            | 10.3            | 16.4            | 14.18   |
| v2               | 12.4            | 16.4            | 9.43            | 16.3            | 13.30   |
