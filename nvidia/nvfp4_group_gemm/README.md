# NVFP4 Group GEMM - Worklog

- `submission_v1.py`: first implementation. Pack kernel arguments to a struct to make it more convenient.
- `submission_v1b.py`: 32-byte global store
- `submission_v1c.py`: don't launch excessive threadblocks.
- `submission_v2.py`: persistent kernel. Using only 128 SMs instead of 148 SMs
- `submission_v2b.py`: support `BLOCK_N=256`, following overlapping 48 tmem columns trick from https://www.youtube.com/watch?v=XzN8EtgEulU
- `submission_v2c.py`: add support for raster along M dim. Only faster for benchmark.1.
- `submission_v2d.py`: 2-SM MMA. Need to use tensor map for SFA and SFB.
- `submission_v2e.py`: from v2c. Sort groups by descending M.
- `submission_v3.py`: TMA store for C. Optimize for benchmark.2 and benchmark.3 (1 wave, persistent kernel not needed, C smem can overlap with AB smem).
- `submission_v3b.py`: uses `16x256b` and `stmatrix`. each epilogue warp issues its own TMA.
- `submission_v4.py`: BLOCK_M=64 and BLOCK_M=256 to reduce quantization effects on M. However, since MMA_M is required to be 128, swap A and B in MMA. Slower LMAO
- `submission_v5.py`: 2-SM MMA with swapped A/B in MMA, so that MMA_M side is large and is a multiple of 128/256.
- `submission_v5b.py`: support `BLOCK_N=256`. This becomes slower, maybe because exposed epilogue?
- `submission_v5c.py`: from v5. persistent kernel with CLC. Slower than static schedule.
- `submission_v5d.py`: from v5. support BLOCK_K=512. Not faster. Maybe need to tune the MMA loop again.
- `submission_v5e.py`: from v5. unroll the first NUM_STAGES TMA stages. swap A/B order in smem. mcast for SFA. unroll the last epilogue wave. use `st.relaxed.cta.global.L1::no_allocate.v8.b32` for gmem stores. switch statement for `compute_bid()`.
- `submission_v5f.py`: use more registers in epilogue. doesn't seem to be faster.
- `submission_v5g.py`: don't issue TMA for A if out of bounds.
- `submission_v6.py`: support `BLOCK_M=256` i.e. `MMA_N=256`.
- `submission_v6b.py`: when `BLOCK_M=256`, issue 2x `MMA_N=128` MMA instead of issuing `MMA_N=256`.
- `submission_v6c.py`: when `BLOCK_M=256`, signal and wait for each `MMA_N=128` separately to overlap epilogue with MMA.

Ideas:
- Mcast A and/or B
- Idle threadblocks can prefetch data to L2?
- Improve `compute_bid`
- Persistent threadblock should stay within a group?
- Use more warps to do epilogue?
- benchmark.1: NUM_STAGES=k_iters -> we can hold all of A or B in smem, then move to the next tile.

## Benchmark results

| (G,N,K) | (8,4096,7168) | (8,7168,2048) | (2,3072,4096) | (2,4096,1536) | Geomean |
|---------|---------------|---------------|---------------|---------------|---------|
| SOL     | 18.833        | 10.667        | 2.406         | 1.525         | 5.211   |
| v1      | 37.8          | 33            | 10.8          | 8.44          | 18.36   |
| v1b     | 37.30         | 31.10         | 10.60         | 8.41          | 17.93   |
| v1c     | 37.70         | 31.10         | 10.50         | 8.41          | 17.94   |
| v2      | 32.5          | 22.9          | 10.4          | 8.4           | 15.97   |
| v2b     | 32.9          | 21.4          | 10.5          | 8.4           | 15.79   |
| v2c     | 32.1          | 20.8          | 10.4          | 8.4           | 15.54   |
| v2d     | 32.4          | 21.5          | 10.4          | 8.4           | 15.71   |
| v2e     | 31.4          | 20.8          | 10.4          | 8.39          | 15.45   |
| v4      | 39.5          | 21.2          | 12.1          | 8.41          | 17.09   |
| v5      | 29            | 20.8          | 10.3          | 8.4           | 15.11   |
| v5c     | 31            | 20.9          | 10.4          | 8.4           | 15.42   |
| v5e     | 28.4          | 20.6          | 8.71          | 6.54          | 13.51   |
| v5f     | 28.3          | 20.5          | 8.78          | 6.55          | 13.52   |
| v5g     | 28.5          | 20.3          | 8.51          | 6.49          | 13.37   |
| v6      | 27.9          | 19.7          | 8.5           | 6.43          | 13.17   |
| v6b     | 27.9          | 19.4          | 8.48          | 6.43          | 13.11   |
| v6c     | 28.7          | 18.8          | 8.53          | 6.55          | 13.18   |

M list
- 80, 176, 128, 72, 64, 248, 96, 160
- 40, 76, 168, 72, 164, 148, 196, 160
- 192, 320
- 128, 384
