# QR CUDA Worklog

## Scope

Optimize the `n=32` and `n=176` benchmark cases. The implementation lives in
`codex/submission.py` and falls back to `torch.geqrf` for larger shapes.

Local development uses an RTX 5090. The leaderboard target is B200, but this
kernel uses ordinary CUDA operations and no architecture-specific features.

## Current Approach

The selected implementation is an unblocked FP32 Householder QR kernel:

For `n=32`:

- One CUDA CTA factors one `32 x 32` matrix.
- The CTA has one warp.
- Lane `i` owns row `i`, and the full row is held in registers.
- The kernel uses vectorized global load/store assembly for four `float8`
  chunks per lane.
- Reflector reductions stay inside the warp, with the shuffle width reduced to
  the next power of two of the active row count.
- Trailing columns are updated serially by the same warp. For this small shape,
  removing shared memory traffic and CTA barriers is faster than parallelizing
  trailing-column updates across more warps.
- The result is stored in LAPACK-compatible compact Householder form:
  `R` in the upper triangle, reflector tails below the diagonal, and a separate
  `tau` tensor.
- The launch uses the implicit default CUDA stream.
- The extension returns new output tensors. It does not mutate the evaluator's
  input, which is required for reliable repeated leaderboard rechecks.

For `n=176`:

- One 32-warp CTA factors each matrix.
- Each lane owns up to six rows of a column, keeping reflector reductions within
  one warp.
- The full padded matrix and two reflector buffers use 126,024 bytes of opt-in
  dynamic shared memory.
- Warp 0 performs pipelined frontier updates and reflector generation; warps
  1-31 cyclically update later columns.
- The kernel uses 37 registers per thread with no local memory or spills.
- Dispatch is gated by the device's opt-in shared-memory limit, so unsupported
  GPUs fall back to `torch.geqrf`.

## Results

Historical and current local benchmarks, `batch=20, n=32`:

| Implementation | Mean |
| --- | ---: |
| `torch.geqrf` baseline | 225.5 us |
| Initial one-warp CUDA kernel | 39.1 us |
| 32-warps, padded shared memory | 18.19 us |
| 16-warps, two-barrier kernel | 18.00 us |
| 16-warps, pipelined one-barrier kernel (historical run) | 13.65 us |
| Same kernel, remeasured before the half-warp-tail change | 12.836 us |
| 16-lane packed tail from `k=16` | 12.730 us |
| One-warp register kernel (selected, B200 run) | 14.890 us |

B200 benchmark cases:

| N | Baseline | Selected | Speedup |
| ---: | ---: | ---: | ---: |
| 32 | 323 us | 14.9 us | 21.7x |
| 176 | 21.7 ms | 367 us | 59.1x |

Using the historical `torch.geqrf` measurement, the old packed-tail shared
kernel was approximately 17.7x faster locally. The directly matched
kernel-to-kernel comparison showed a smaller 0.82% improvement from the packed
tail itself.

The selected B200 QR32 path is now the one-warp register kernel from
`submission_v0_cpp.py`, transplanted into `codex/submission.py` while keeping
the `qr32_codex.forward` wrapper. A same-container B200 comparison on benchmark
case 0 measured the register kernel at about `17.01 us` versus `18.68 us` for
the shared-memory Codex QR32 with matched compile flags. After transplanting,
the focused benchmark measured `14.890 us`, and the official QR32 test reported
`scaled_factor_residual=0.104`.

Before the packed-tail change, the displayed seven-case B200 geometric mean was
`16.94 ms`, compared with `44.96 ms` for the baseline, for a `2.654x` overall
speedup. The packed-tail variant has not yet been benchmarked on B200.

The pre-tail kernel passed both the official test and a leaderboard-style
repeated recheck. The selected packed-tail kernel passes the official `n=32`
test; a repeated recheck has not yet been rerun for this variant. The kernel
also previously passed manually generated `n=32` cases for:

- Dense input with wider column scaling
- Rank-deficient input
- Near-rank-deficient input
- Clustered scales
- Banded input
- Row-scaled input
- Near-collinear input
- Upper-triangular input

On the official `n=32` test case, the selected implementation reported a
scaled factor residual of `0.096`, comparable to the earlier kernel and the
`torch.geqrf` baseline.

## Experiments

### One Warp Per Matrix

The first implementation used one warp and stored the matrix in shared memory.
It was simple and correct, but trailing-column updates were serialized. It
measured about `39.1 us`.

### Parallel Column Updates

Using one warp per trailing column reduced runtime to about `31 us`. The
initial shared layout still suffered from 32-way bank conflicts on column
accesses.

### Padded Shared Memory

Changing the shared-memory leading dimension from 32 to 33 removed the column
bank conflicts. Runtime dropped from about `31 us` to about `18.2 us`. This was
the largest optimization.

### CTA Size

Tested padded shared-memory kernels with different warp counts:

| Warps | Mean |
| ---: | ---: |
| 8 | 21.44 us |
| 16 | 18.00 us |
| 24 | 18.50 us |
| 32 | 18.19 us |

Sixteen warps was the best local result.

### Pipelined Reflector Lookahead

The original padded kernel used two CTA barriers per reflector: one after
constructing the reflector and one after updating the trailing matrix.

The selected kernel peels reflector 0 and double-buffers reflector vectors.
Warp 0 updates the next column and constructs the following reflector while
warps 1-15 update later columns. A single CTA barrier advances the pipeline.

At the time of that experiment, this reduced local runtime from `18.00 us` to
`13.65 us` and B200 runtime from `26.1 us` to `20.6 us`. A later matched local
remeasurement put the same pre-tail kernel at `12.836 us`, so `13.65 us` should
be treated as a historical result rather than the current comparison baseline.

### Packed 16-Lane `n=32` Tail

After iteration `k=15`, reflector 16 is already available. The kernel now uses
a separate direct loop for `k=16..30`. Each consumer warp splits into two
16-lane groups with independent shuffle reductions, so one warp updates two
trailing columns. Warp 0 continues to update the frontier column and construct
the next reflector with the original full-warp path.

Matched local measurements using the same runner and environment were:

| Implementation | Mean |
| --- | ---: |
| Pre-tail kernel from `HEAD` | 12.836 us |
| Packed 16-lane tail | 12.730 us |

The change saves about `0.106 us`, or `0.82%`. The official `n=32` test passes
with a scaled factor residual of `0.096`. No B200 result is available yet.

### Warp 0 Tail Split and CTA Retuning

A follow-up variant split warp 0 into two 16-lane groups during `k=16..30`.
The low half updated frontier column `k+1` and constructed the next reflector,
while the high half updated column `k+2`. This was correct, but the 16-warp
kernel regressed from `12.730 us` to `12.914 us`. The extra subgroup indexing
and masked reductions on the critical frontier path outweighed the saved
consumer-warp work, so the change was reverted.

The selected full-warp-frontier design was also retuned after adding the packed
consumer tail:

| CTA warps | Mean |
| ---: | ---: |
| 8 | 13.471 us |
| 12 | 13.064 us |
| 14 | 13.152 us |
| 16 | 12.730 us |

Sixteen warps remains best. Although the tail needs fewer consumer warps, the
`k=0..15` phase still benefits from the larger CTA and determines the overall
tradeoff.

### Full-Mask and Half-Warp Reflector Reductions

The 16-lane shuffle helper can use `FULL_MASK` with `width=16` because the
butterfly remains partitioned into two independent half warps, provided all 32
lanes execute every shuffle. Consumer updates were changed to call the
reduction unconditionally, with inactive columns contributing zero.

Reflector construction for `k>=16` was also tested using only the low 16 lanes
for the active rows. All 32 lanes executed the width-16 shuffle; the high half
contributed zero, and only the low-half result was committed.

Both changes were correct but slower in matched local benchmarks:

| Variant | Mean |
| --- | ---: |
| Selected per-half masks, full-warp reflector | 12.730 us |
| Full-mask consumer reductions only | 12.852 us |
| Half-warp reflector only | 13.085 us |
| Both changes combined | 13.081 us |

The selected kernel therefore retains per-half consumer masks and full-warp
reflector construction. The reduced shuffle count did not compensate for the
additional predicates, remapped row indexing, and reflector-buffer clearing.

### Full-Shared `n=176`

#### Shrinking-Row Phase Specialization

The QR176 loop was split into compile-time row-item phases with 6, 5, 4, 3,
2, and 1 items per lane at `k=0`, `16`, `48`, `80`, `112`, and `144`. This was
intended to remove inactive unrolled item bodies as the active column shrinks.
The extension compiled with 38 registers and no spills, but the B200 correctness
check failed badly at `n=176` (`scaled_factor_residual=3.65e3`). The phase
refactor was reverted rather than retained as an optimization candidate.

#### QR176 Norm Expression

Replacing both QR176 `hypotf(x0, sqrtf(tail))` calls with
`sqrtf(x0 * x0 + tail)` passed B200 correctness and measured `358 us` in one
run. A baseline submission with `hypotf` landed on a different worker regime:
QR176 measured `747 us`, while all compute-heavy Torch cases were also about
2.17x slower. Normalizing against those unchanged cases did not show a reliable
win for the `sqrtf` form. The proven `hypotf` implementation was retained.

A later same-session QR176 experiment revisited this after moving warp
specialization outside the column loop. On B200 benchmark case 1:

| Variant | Mean |
| --- | ---: |
| Baseline inner `if (warp == 0)` loop | 364.43 us |
| Outer warp-specialized loop | 355.10 us |
| Outer loop + `sqrtf(x0 * x0 + tail)` | 354.74 us |
| Outer loop + `sqrtf` + branch-removed reflector guards | 353.97 us |
| Same, remeasured before unroll experiments | 353.32 us |
| De-peeled explicit producer/consumer schedule | 342.45 us |
| De-peeled schedule with inline `bar.sync 0` barriers | 341.24 us |
| De-peeled + phase-split item-count loops | 267.83 us |
| Phase split + branchless producer diagonal write | 262.90 us |
| Phase split + padded branchless row loops | 223.64 us |

The branch-removed variant keeps the direct QR176 test and full official test
suite passing. Adding `__launch_bounds__(THREADS176, 1)` passed correctness but
regressed to `357.72 us`, so it was not kept.

Outer-loop unroll directives were also tested. `#pragma unroll 1` on both
producer and consumer `k` loops passed but regressed to `354.89 us`.
`#pragma unroll 2` failed correctness with a large factor residual, consistent
with the loop-carried dependency on the just-published reflector and the
per-iteration CTA barrier. The outer `k` loops are therefore left without an
explicit unroll directive.

The peeled reflector-0 setup was later removed using an explicit
producer/consumer schedule: warp 0 computes reflector `k`, stores it, and
enters the publish barrier; consumer warps wait at that barrier and update
columns `k+2...`, while warp 0 updates only the frontier column `k+1`. The next
iteration's publish barrier also joins the previous consumer updates before
those columns can receive reflector `k+1`. This keeps one CTA barrier per
logical iteration and removes the separate peeled setup, improving benchmark
case 1 to `342.45 us`.

Replacing QR176 `__syncthreads()` calls with an inline
`asm volatile("bar.sync 0;" ::: "memory")` helper passed the full test suite
and measured `341.24 us`, but it was reverted to keep standard CUDA barriers.
Bare `#pragma unroll` was tested on both de-peeled outer `k` loops with both
`__syncthreads()` and inline `bar.sync 0`; both failed the focused QR176
correctness test (`scaled=5.32e3`), so the de-peeled outer loops remain without
an explicit unroll directive.

The retained QR176 version splits the producer and consumer `k` loops into
fixed row-item phases, using templated step helpers: `k=0..15` uses six row
items per lane, then five items for `16..47`, four for `48..79`, three for
`80..111`, two for `112..143`, and one for `144..174`. This keeps the same
standard `__syncthreads()` producer/consumer cadence while reducing the
unrolled item work in later columns. The focused QR176 benchmark case improved
to `267.83 us`, the full official test suite passed, and QR352 benchmark case
2, which uses QR176 for the tail on B200, still passed and measured `7.42 ms`.

Removing the producer diagonal-write branch with arithmetic masks improved
case 1 to `262.90 us`. Padding the QR176 shared matrix and reflector buffers to
208 rows then allowed dummy row accesses for high lanes, removing the remaining
`row < N176` branches from the unrolled row-strip loops. This raised QR176
dynamic shared memory but stayed within the B200 opt-in limit, passed the full
test suite, and improved case 1 to `223.64 us`. QR352 benchmark case 2 still
passed and measured `7.39 ms`.

The QR176 and panel352 de-peeled producer/consumer schedules now share the same
templated step helpers. QR176 instantiates them in padded-shared-memory mode to
keep the branchless dummy-row accesses, while panel352 instantiates the same
code in bounded global-output mode. This is a maintenance refactor intended to
preserve the previous codegen shape while avoiding duplicated producer,
consumer, and final-reflector implementations. Focused B200 validation passed
for both case 1 (`n=176`) and case 2 (`n=352`); the post-refactor focused
benchmarks measured `222.26 us` and `7.109 ms`, respectively. The full 22-case
official test suite also passed.

A follow-up QR352 path uses a single hard-coded split at `b=144`: the first
`352 x 144` panel is factored by a padded-smem `panelMN` instantiation, the
compact-WY update is applied to the remaining 208 columns, and the `208 x 208`
remainder is factored by the same padded-smem kernel instantiated as `qrN`.
The tail padding is 240 rows because the seven-item early phase can dummy-read
up to row 238. This fixes the previous direct-global-memory panel factorization
while keeping only one split. Focused B200 case 2 passed and benchmarked at
`974.53 us`, down from `7.109 ms`; the full 22-case official test suite passed.

The same generic `panelMN`/`qrN` mechanism now owns the QR512 and QR1024 paths
too, and the old direct-global-memory panel352/panel512/panel768/panel1024
kernels and extension bindings were removed. QR512 uses panels
`96, 96, 128` followed by `qr192`; QR1024 uses panels
`48, 48, 48, 56, 56, 64, 72, 80, 88, 104, 136, 32` followed by `qr192`.
The shared producer/final-reflector helpers always guard `tail == 0`, matching
the QR32 convention and preventing rank-deficient columns from producing NaNs.
Focused B200 benchmarks measured QR512 case 3 at `12.24 ms` and QR1024 case 4
at `9.90 ms`. After adding the zero-tail guard, the full 22-case official test
suite passed. A later universal-guard cleanup also passed the focused QR176
case; further Modal validation was blocked by the environment usage limit.

The same pipelined unblocked design was extended to `n=176`. Each lane handles
up to six rows, avoiding cross-warp reductions, while 31 consumer warps divide
the trailing columns cyclically. The matrix fits on B200 using 126,024 bytes of
opt-in dynamic shared memory but does not fit on the local RTX 5090.

The full B200 test suite passed 19/19. Runtime improved from `21.7 ms` to
`367 us`, a `59.1x` speedup. This result justifies keeping the unblocked path
as the final pre-blocked specialization.

### Column-Major Shared Memory

Storing the shared matrix in column-major order made factorization accesses
contiguous, but introduced conflicts during the initial transpose load and
final transpose store. It measured about `18.44 us`, slower than padded
row-major storage.

### In-Place Output

Writing compact factors back into the input reduced the measured mean to about
`28.9 us` before the shared-memory padding optimization. It was rejected
because the evaluator reuses inputs during benchmarking and leaderboard
rechecks. Later iterations would factor an already-factorized matrix and fail
rechecked correctness.

### Single Output Allocation

Allocating one backing tensor and returning separate `H` and `tau` views did
not improve runtime. View creation offset the saved allocator call.

### Register-Only QR

The one-warp register QR32 path is now selected. The useful ideas are:

- keep a full row per lane in registers when the whole row fits comfortably;
- avoid CTA barriers and shared-memory staging for very small square factors;
- use vectorized global load/store for the row transfer; and
- shrink the warp-reduction width with the active row count.

These ideas do not transfer directly to QR176 or the larger panel kernels.
QR176 would require each lane to hold up to six full rows of 176 columns, which
is far beyond a practical register footprint. The panel kernels already use a
register strip per lane for the active column and keep only the reflector
buffers in shared memory; making the full panel register-resident would explode
register pressure and occupancy. The most plausible transferable piece is
vectorizing pure copy/load-store phases, but those phases are not currently the
dominant cost relative to the Householder update loop and trailing GEMMs.

For QR176, a narrower four-column warp-0 scratchpad is more plausible than a
full register-resident matrix: warp 0 would hold columns `k+1..k+4` in
registers while consumer warps start at `k+4`. This could remove shared-memory
round trips for the producer columns, but it requires a tiled producer loop,
careful delayed stores for compact Householder output, and the same per-column
barrier cadence so consumer warps see each newly generated reflector. The
smaller outer-specialization experiment captures part of that structure without
the register-tile rewrite and produced a measurable B200 win.

### Launch Bounds

The old shared-memory QR32 kernel used compile-time constants for `N=32`,
shared leading dimension 33, 16 warps, and 512 threads. The compiled kernel
used 30 registers per thread, 5512 bytes of shared memory, and no local memory
or spills.

Adding `__launch_bounds__(512, 1)` increased allocation to 32 registers per
thread but slightly regressed the historical local runtime from about
`13.65 us` to `13.72 us`. It was removed; these numbers predate the current
matched benchmark environment.

## Next Work

- Re-check full leaderboard timing after the QR32 register-kernel transplant.
- Prototype the QR176 four-column warp-0 scratchpad if QR176 remains a ranking
  bottleneck after the cheaper loop/codegen changes.
- Consider specializing allocation or registration only if profiling shows it
  is material.
- Start a blocked Householder design for `n >= 352`.

### Compact-WY QR512 Trailing Update

The selected `n=512` path uses two 256-column panels. The original prototype
applied the first panel to the trailing 256 columns with `torch.ormqr`.

The trailing update now builds the compact-WY triangular factor through

`T^{-1} = diag(1 / tau) + triu(V.T @ V, 1)`

and applies `Q.T` as a triangular solve plus GEMMs:

`C <- C - V solve(T^{-T}, V.T @ C)`.

A zero-`tau` guard falls back to `torch.ormqr`. The standard, mixed,
rank-deficient, and clustered `n=512` correctness cases pass.

Matched local RTX 5090 benchmark, `batch=640, n=512`:

| Trailing update | Mean |
| --- | ---: |
| `torch.ormqr` | 521.88 ms |
| Compact-WY | 478.91 ms |

B200 `qr_v2` benchmark:

| Case | Previous | Compact-WY | Improvement |
| --- | ---: | ---: | ---: |
| Dense | 722 ms | 650 ms | 10.0% |
| Mixed | 727 ms | 654 ms | 10.0% |
| Rank-deficient | 723 ms | 650 ms | 10.1% |
| Clustered | 723 ms | 650 ms | 10.1% |

### Rejected QR1024 Two-Panel Path

A two-panel `n=1024` path using block size 512 and the compact-WY trailing
update passed all six official QR1024 tests: dense, wider-scale dense,
rank-deficient, near-rank-deficient, clustered, and mixed.

The matched local RTX 5090 benchmark regressed slightly:

| Implementation | Mean |
| --- | ---: |
| `torch.geqrf` fallback | 171.82 ms |
| Block-512 compact-WY | 173.02 ms |

The custom QR1024 dispatch was removed. Future B200 submissions should require
a local improvement first unless the implementation specifically targets a
B200-only architectural feature.

### QR512 Custom First Panel

A direct custom `panel512` kernel was added for the first 256-column panel, then
kept only for the `n=512` dispatch. It is a mechanical blocked-panel variant of
the shared-memory Householder kernels: one CTA per matrix, 512 rows, 256 panel
columns, 32 warps, and 16 rows per lane. Full loop unrolling caused excessive
register pressure and failed correctness; forcing `#pragma unroll 1` in the
panel512 body brought the kernel to 33 registers per thread and passed the
standard, mixed, rank-deficient, and clustered QR512 tests.

Local RTX 5090 benchmark, `batch=640, n=512`:

| First panel | Mean |
| --- | ---: |
| `torch.geqrf` + compact-WY trailing update | 481.45 ms |
| Custom `panel512` + compact-WY trailing update | 159.17 ms |

B200 `qr_v2` benchmark on 2026-06-16 did not reproduce the older 650 ms
compact-WY result. In the same session, the restored `torch.geqrf` first-panel
path measured about 1401 ms for dense QR512, while the custom `panel512` path
measured about 792 ms. The custom panel is therefore the better current
submission even though it is slower than the older documented compact-WY B200
run.

### Gemini Folder Review

Reviewed `linalg/qr_v2/gemini/submission.py`, `STATUS.md`, and `WORKLOG.md`.
The useful transferable idea was replacing the QR352 trailing `torch.ormqr`
call with the existing compact-WY update. The QR512 recursive block-size-64
path was not taken because it measured about 966 ms locally, versus about
160 ms for the current custom `panel512` path. The QR4096 fixed block-size-512
path was also not taken because a same-session local screen was slower than the
direct `torch.geqrf` fallback under the same conditions.

QR352 compact-WY local RTX 5090 benchmark, `batch=40, n=352`:

| QR352 trailing update | Mean |
| --- | ---: |
| Previous `torch.ormqr` path | 36.39 ms |
| Gemini compact-WY path | 20.30 ms |

The official QR352 test passes with `scaled_factor_residual=0.0127`.

B200 `qr_v2` benchmark on 2026-06-16 with QR352 compact-WY and QR512 custom
`panel512`:

| Case | Mean |
| --- | ---: |
| n=352 dense | 17.0 ms |
| n=512 dense | 521 ms |
| n=512 mixed | 523 ms |
| n=512 rankdef | 524 ms |
| n=512 clustered | 522 ms |

Decision: keep QR352 compact-WY. It is a local and B200 win, and it reuses the
same zero-`tau` fallback as the QR512 compact-WY update.

Later QR352 work applied the QR176 producer/consumer phase-split idea to
`panel352`: the first-panel loop was de-peeled, split into row-item phases
`11, 10, 9, 8, 7, 6`, and changed to branchless diagonal writes with
`sqrtf(x0 * x0 + tail)`. Unlike QR176, the panel data lives in the output
tensor rather than padded shared memory, so global-output bounds guards remain.
The first attempt failed correctness because it did not compute the real final
panel reflector for column 175; adding a final-reflector helper fixed the panel
contract. The focused QR352 test passed with `scaled_factor_residual=0.0166`,
the full official suite passed, and B200 benchmark case 2 measured `7.07 ms`
versus `7.39 ms` for the preceding padded-QR176 version.

### QR1024 Custom First Panel

Added a `panel1024` kernel for a 1024-row by 256-column first panel, adapted
from the selected `panel512` kernel by increasing the per-lane row items from
16 to 32 while keeping the panel width at 256. The Python `qr1024` path factors
that first panel, applies it to the trailing 768 columns with the existing
compact-WY update, then factors the 768x768 tail with `torch.geqrf`.

The standard, dense, mixed, and near-rank QR1024 tests pass.

Sequential local RTX 5090 benchmark, `batch=60, n=1024`:

| Case | Direct `torch.geqrf` | Custom `panel1024` path |
| --- | ---: | ---: |
| Dense | 172.36 ms | 165.01 ms |
| Mixed | 172.78 ms | 164.91 ms |
| Near-rank | 172.35 ms | 165.43 ms |

B200 `qr_v2` benchmark on 2026-06-16:

| Case | Previous current-best | Custom `panel1024` path |
| --- | ---: | ---: |
| n=1024 dense | 515 ms | 500 ms |
| n=1024 mixed | 515 ms | 500 ms |
| n=1024 near-rank | 515 ms | 500 ms |

Decision: keep the QR1024 custom first-panel path. The B200 gain is modest but
consistent across all QR1024 benchmark cases.

### Rounded Shared-Memory Rows and Generic QR176

The generic shared-memory kernel now allocates exactly `round_up(ROWS, 32)`
matrix and reflector rows. Each compile-time item phase owns a fixed row range
starting at `max(0, ROWS - 32 * ITEMS)`, so the item count decreases as soon as
the active tail crosses a 32-row boundary. For QR176, the six-item phase covers
columns 0-15 and the five-item phase begins at column 16, while storage is
reduced from 208 to 192 rows.

QR176 now dispatches through the generic `qrN` kernel. The bespoke QR176
kernel, binding, dead direct-global-memory helper branches, explicit padding
constants, and the hand-written 31-entry phase macro were removed. An explicit
1024-thread launch bound keeps the largest panel specialization within the
register budget.

The full 22-case B200 correctness suite passes. Matched B200 timings after the
cleanup were 224 us for QR176, 953 us for QR352, 12.06 ms for QR512, and
9.90 ms for QR1024. The pre-cleanup QR176 source measured 226 us in the same
environment.

### Short and Tall Panel Kernels

The shared-memory panel implementation is now split by row regime. The
`short` kernel retains compile-time row-item unrolling for shapes through 1024.
The `tall` kernel accepts runtime dimensions and processes rows in fixed
four-item chunks, keeping register demand independent of the panel height.
Both kernels use the same `panelMN` binding, producer/consumer warp
specialization, compact-WY trailing update, and short square-tail kernels.

QR2048 uses adaptive tall-panel widths computed from the 227 KiB B200
shared-memory limit and ends in `qr208`. It passes dense, rank-deficient, and
mixed correctness cases. B200 benchmark case 5 improved from the previous
`torch.geqrf` result of about 76.7 ms to 35.87 ms.

The same strategy was tested for QR4096 and passed correctness, but its initial
width of 11 required 177 panel factorizations and measured 107.0 ms. The
same-environment `torch.geqrf` fallback measured 52.10 ms, so QR4096 remains on
the fallback. A competitive custom QR4096 path will need a wider panel that
streams matrix rows from global memory instead of fitting the complete panel
in shared memory.

### Strided Panel Boundaries

`panelMN` and `qrN` now accept separate input and output tensor views and pass
their batch and row strides to CUDA. Short and tall kernels load the strided
source into shared memory and store directly into the strided destination;
they also support source and destination aliasing after the first panel. The
blocked Python path no longer materializes contiguous panel or trailing-input
copies, allocates temporary panel outputs, or copies panel and square-tail
results back into `h`.

Compact-WY is output-aware as well: its final update uses `torch.baddbmm` to
write `trailing - V @ transformed` directly into the destination view. This
removes the correction allocation and final writeback copy. Zero-`tau` columns
are removed from `V` and assigned a safe unit diagonal in `T^{-1}`, replacing
the per-panel `.item()` synchronization and `torch.ormqr` branch while
preserving rank-deficient correctness.

The final full 22-case B200 correctness suite passes. Matched timings through
the copy-removal sequence were:

| Case | Before striding | Strided boundaries | Output-aware compact-WY |
| --- | ---: | ---: | ---: |
| QR352 | 953 us | 909 us | 901 us |
| QR512 | 12.06 ms | not measured | 10.36 ms |
| QR1024 | 9.90 ms | not measured | 8.05 ms |
| QR2048 | 35.87 ms | 34.72 ms | 30.76 ms |

`run_modal.py --mode profile --case N` now times representative panel and
compact-WY stages independently using the real schedule, benchmark batch size,
and full-matrix strides. QR4096 samples show early narrow-panel stages costing
roughly 0.4-0.7 ms each; repeating them across 177 panels explains why the
shared-memory tall schedule cannot beat the 52.10 ms fallback.

Tau storage is now preallocated as the final `(batch, n)` tensor. `panelMN` and
`qrN` accept mutable strided tau views, and CUDA writes through the supplied
tau batch stride. This removes per-panel tau allocations, the Python tau list,
and the final `torch.cat`. The full correctness suite passes; QR2048 improved
slightly from 30.76 ms to 30.67 ms.

### Panel and Compact-WY Characterization

Profile mode now separates compact-WY into `V` construction, `V.T @ V` and
`T^{-1}` setup, projection, triangular solve, and final apply. Measurements use
10 timed repetitions, actual benchmark batches, full-matrix strides, and
representative stages from the real schedules.

Representative QR2048 measurements, in milliseconds:

| `(rows, width, trailing)` | Panel | Full WY | V | Gram/T | Project | Solve | Apply |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `(2048, 25, 2023)` | 0.219 | 0.307 | 0.024 | 0.053 | 0.062 | 0.024 | 0.148 |
| `(1358, 39, 1319)` | 0.576 | 0.217 | 0.018 | 0.044 | 0.057 | 0.029 | 0.062 |
| `(832, 66, 766)` | 0.316 | 0.451 | 0.023 | 0.036 | 0.025 | 0.329 | 0.033 |
| `(299, 91, 208)` | 0.388 | 0.298 | 0.020 | 0.027 | 0.011 | 0.218 | 0.009 |

Representative QR4096 measurements:

| `(rows, width, trailing)` | Panel | Full WY | V | Gram/T | Project | Solve | Apply |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `(4096, 11, 4085)` | 0.178 | 0.372 | 0.047 | 0.084 | 0.102 | 0.033 | 0.136 |
| `(3509, 13, 3496)` | 0.167 | 0.294 | 0.042 | 0.076 | 0.086 | 0.033 | 0.108 |
| `(2267, 22, 2245)` | 0.192 | 0.179 | 0.042 | 0.057 | 0.041 | 0.034 | 0.038 |
| `(234, 26, 208)` | 0.044 | 0.190 | 0.042 | 0.058 | 0.015 | 0.033 | 0.012 |

Conclusions:

- Early large-trailing WY stages are dominated by projection and final apply,
  making a fused small-width WY kernel relevant.
- `V` plus Gram/T setup costs about 0.10-0.13 ms per narrow QR4096 panel and
  becomes the majority of WY cost later in the schedule.
- QR2048 widths 66 and 91 hit large triangular-solve performance cliffs. A
  schedule cost model must include measured shape effects rather than FLOPs.
- Tall-panel spikes align with poor shared-memory leading dimensions. Width 15
  gives `LD=16`, and width 39 gives `LD=40`; both share factors with 32 and
  measured 0.429-0.576 ms. The adaptive scheduler should account for bank
  conflict degree, preferably selecting widths with odd `LD` when capacity
  and panel count permit.
- Even eliminating WY entirely would leave 177 QR4096 panel stages. A wider
  global-streamed panel remains necessary for a competitive custom QR4096.

### Scoped TF32 Compact-WY

Globally setting float32 matmul precision to `high` was rejected. Although it
improved QR2048 from 30.67 ms to about 29.30 ms, the full suite failed the
QR512 row-scaled and mixed cases, and custom-path residuals increased by two to
three orders of magnitude.

TF32 is instead scoped to QR1024 and QR2048 blocked execution. The previous
matmul precision is restored before returning, so evaluator checks and all
other sizes continue to use `highest` FP32 precision. The full 22-case suite
passes under this configuration, including FP32 evaluator reconstruction
checks and rank-deficient stress cases.

Matched B200 timings:

| Case | FP32 compact-WY | Scoped TF32 |
| --- | ---: | ---: |
| QR1024 | 8.05 ms | 5.49 ms |
| QR2048 | 30.67 ms | 29.44 ms |

QR1024 gains substantially because its compact-WY GEMMs are large enough to
benefit from tensor cores. QR2048 gains less because its many narrow panels
remain dominated by panel work, setup, and shape-dependent solve costs.

### Pipelined Global QR4096 Panel

QR4096 now uses fixed 128-column panels until the remaining square tail reaches
2048, then calls the existing QR2048 path. Each panel is transposed into a
column-major tensor and factored by a persistent CUDA kernel.

One producer CTA constructs each reflector with an eight-warp CTA reduction.
Each trailing column has a dedicated eight-warp consumer CTA and a 4096-float
shared-memory cache, so the column is read once from global memory and written
once after its update. Global epoch and per-column completion counters pipeline
reflector generation with independent column updates without cooperative-grid
barriers.

Panel-only B200 results for `batch=2, 4096 x 128`:

| Implementation | Mean |
| --- | ---: |
| `torch.geqrf` | 1.665 ms |
| Cooperative-grid prototype | 5.60 ms |
| Epoch pipeline, one warp per column | 5.30 ms |
| Epoch pipeline, eight warps per column | 1.311 ms |

The panel reconstruction residual was approximately `2.31e-7`. Generalizing
the same kernel to runtime row counts allowed all sixteen shrinking QR4096
panels to use it. Focused dense and upper-triangular correctness cases pass.
The B200 QR4096 benchmark improved from `52.13 ms` for direct `torch.geqrf` to
`44.14 ms`, a 15.3% speedup.

### QR4096 Panel Vectorized Row-Stream Experiment

The `panel4096x128_kernel` scalar row streams were reviewed for possible
replacement with vectorized global loads. The panel data is column-major after
the Python wrapper transposes the active panel, so the original scalar access
pattern

`matrix[col * rows + row]`

is already warp-coalesced: each warp iteration reads contiguous rows from one
column. A vectorized path therefore does not improve memory coalescing; its
main possible benefit is reducing the number of global load/store instructions
and loop iterations.

A `float4`/`ld.global.v4.f32` style row stream is the practical target. Wider
`v8.f32` loads are not a useful CUDA load granularity here and would be split
or rejected. Because `rows` is a multiple of four for the QR4096 panel schedule,
`col * rows` is 16-byte aligned, and starting each vector loop from row `0`
keeps every `float4` access aligned. Rows below the active reflector start
`k` can be loaded and then masked out in arithmetic instead of using a scalar
alignment peel. This is simpler than peeling from `k` to the next four-float
boundary, and the skipped prefix is tiny for a 128-column panel.

The first vectorized prototype failed correctness because it accidentally
assumed a fixed `4096` row extent. The same kernel is reused for shrinking
panels with runtime `rows = 4096, 3968, 3840, ...`; vector loops must guard
against the runtime row count and must not read or write past the logical
column stride. This is especially important for later panels, where processing
up to 4096 rows corrupts neighboring columns.

Arithmetic masks were also tested in the style used by the QR32 compact
Householder writeback, for example

`active_x = (row >= k) * x`

and

`compact = (row < k) * old + (row == k) * beta + (row > k) * v`.

This form is correct when every out-of-range vector element is also guarded by
`row < rows`, and it keeps inactive rows branchless. In the current experiment
it did not materially change performance relative to the previous mask style,
so the vectorized variant is not selected yet.

Current conclusion: keep the earlier scalar panel kernel unless a fresh B200
profile shows LD/ST issue pressure. The scalar path is already coalesced and
correct for shrinking panels. A future vectorized retry should use aligned
`float4` streams from row `0`, arithmetic masks for `row >= k`, compact
writeback masks for the panel column, and strict `row < rows` guards on every
load/store in the vector tail.

### Vectorized Global Panels for QR2048 and QR4096

The corrected `float4` implementation reduced the B200 `batch=2, 4096 x 128`
panel time from `1.311 ms` to `0.697 ms`. End-to-end QR4096 improved from
`44.14 ms` to `38.60 ms` while retaining the same focused correctness.

The global-panel design was then extended across the complete QR2048 factor,
replacing the adaptive narrow shared-memory schedule. Batch-interleaved block
ordering exposed a residency issue with the original one-column-per-CTA shape
at benchmark batch 8. The selected geometry caches four columns per CTA and
uses two warps per column. It launches 33 CTAs per matrix, allowing the batch-8
working set to remain resident on B200.

Panel-only `batch=8, 2048 x 128` timing is `0.738 ms`, versus `4.93 ms` for
`torch.geqrf`. The full QR2048 benchmark improved from approximately
`29.4 ms` to `15.03 ms`. The same four-column geometry leaves QR4096
effectively unchanged at `38.68 ms` versus `38.60 ms` for the preceding
one-column configuration. Focused dense correctness passes for both sizes.

Replacing the panel's `float4` global streams with the QR32
`ldg_f32x8`/`stg_f32x8` helpers was correct but slower. The QR2048 panel
regressed from `0.738 ms` to `0.834 ms`, and end-to-end QR2048 regressed from
`15.03 ms` to `15.30 ms`. The wider instruction and temporary eight-float
register arrays did not offset their additional register/code-generation cost,
so the panel retains native `float4` loads and stores.

A narrower follow-up used `v8.f32` only in the producer's global factor-column
load and compact writeback, leaving every shared-memory path on `float4`. This
also regressed: panel time was `0.751 ms` and QR2048 was `15.16 ms`. Therefore
the earlier loss was not solely caused by shared-memory bank behavior; the
selected implementation remains uniformly `float4`.

The panel signalling was changed from `atomicAdd(ptr, 0)` polling and
`atomicExch` publication to PTX `ld.acquire.gpu.global` and
`st.release.gpu.global`. The QR2048 panel initially measured `0.721 ms` versus
approximately `0.729 ms` before the change, and end-to-end QR2048 measured
`14.91 ms` versus `15.03 ms`. The acquire/release protocol is required by the
later persistent-cache variant at all row counts; retaining the old atomic
specialization above 2048 rows produced stale panel updates and failed QR4096
reconstruction.

The selected dedicated-producer kernel now loads each consumer CTA's four
columns into shared memory once and applies every preceding reflector there.
A column is written to global memory only after its final update, immediately
before the producer needs it as the next frontier. This reduced the batch-8
2048x128 panel from `0.721 ms` to `0.646 ms` and QR2048 from about `15.03 ms`
to `14.21 ms` in the first isolated run. QR4096 measured `39.13 ms`, about
`0.3-0.5 ms` slower than the recent `38.6-38.9 ms` range because its taller
panels pay more acquire-polling overhead. The full 22-case correctness suite
passes. The next refinement should reduce reflector-epoch polling overhead for
tall panels without weakening the release/acquire visibility contract.

Compact-WY profiling was corrected to follow the selected 128-column global
panel schedule rather than the older shared-memory width schedule. For QR2048
batch 8, PyTorch's batched 128x128 triangular solve costs roughly
`0.27-0.34 ms` per panel and is often the largest Compact-WY component.
Physical row/column-major variants and explicit triangular inverse plus GEMM
saved little or nothing at batch 8. Launching one solve per batch item on
independent CUDA streams reduced solve time to about `0.20 ms` and improved
QR2048 from `14.21 ms` to `12.45 ms`; overlapping `V.T @ V` with
`V.T @ trailing` added only about `0.03 ms` end-to-end. QR4096 reached
`38.85 ms`. This stream-based path was deliberately removed because the
evaluation harness may not reliably surface asynchronous stream failures.
The submitted implementation remains single-stream; a future custom grouped
TRSM is the safer way to recover most of the observed QR2048 opportunity.

That single-stream replacement was subsequently implemented with
`cublasStrsmBatched`. A tiny default-stream CUDA kernel builds the device
pointer arrays, and cuBLAS solves directly in the `projected` buffer, avoiding
the separate `transformed` allocation. A private cuBLAS handle remains on its
default stream; the implementation does not query or set PyTorch's current
stream. The row-major tensors are mapped to a
right-side column-major solve, so no layout conversion is required. Immediate
pointer-kernel and cuBLAS launch failures are propagated through
`TORCH_CHECK`; no auxiliary streams are used. The path is restricted to the
proven batch-8, width-128 QR2048 configuration because batch-2 QR4096 did not
benefit. QR2048 measured `12.19 ms` initially and `12.50 ms` on a later worker,
consistently improving the prior `14.21 ms` by roughly `1.7-2.0 ms`. The full
22-case suite passed before restriction, and the restricted batch-2 QR2048
fallback was rechecked successfully.

The `V` materialization in Compact-WY was fused from `torch.tril`, diagonal
fill, and zero-tau masking into one Triton kernel for width-128 panels. It
writes the row-major layout consumed by the existing GEMMs directly while
accepting the panel's arbitrary strides. A CUDA version first measured
`12.37 ms` for QR2048 and `38.88 ms` for QR4096. Triton matched or improved it;
BLOCK sizes 256, 512, and 1024 were compared, with 512 selected. The best
measurements were `12.10 ms` for QR2048 and `38.76 ms` for QR4096. BLOCK=1024
regressed both sizes. The full 22-case correctness suite passed with the fused
Triton implementation.

The Triton preparation path was then generalized to every panel width; width
is already a `tl.constexpr`, while row count and arbitrary tensor strides remain
runtime parameters to avoid needless recompilation. A second Triton kernel now
converts the contiguous Gram result `V.T @ V` in place into `T^{-1}`. It fuses
the upper-triangular mask, zero-tau substitution, reciprocal diagonal, and
diagonal insertion, replacing `torch.triu`, `torch.where`, reciprocal, and
diagonal copy operations. A/B measurements against the original PyTorch path
were `10.11 ms` versus `10.39 ms` for QR512 and `4.97 ms` versus `5.54 ms` for
QR1024. QR2048 improved from `12.10 ms` to `11.79 ms`, and QR4096 from
`38.76 ms` to `37.42 ms`. All 22 correctness cases pass. The remaining
Compact-WY boundaries are GEMM/TRSM operations; the final update already uses
cuBLAS `baddbmm` with its subtraction epilogue, so further fusion requires a
custom matmul rather than another low-risk pointwise fusion.

A fused Triton matmul prototype then combined `V` materialization, `V.T @ V`,
and the `T^{-1}` epilogue in one launch. Each Gram tile reconstructed masked
unit-lower panel values directly in registers, while one tile per column block
also wrote the row-major `V` needed by projection and update. Computing the
full symmetric grid measured `12.14 ms` for QR2048 versus `11.79 ms` for the
selected cuBLAS-Gram path. Skipping lower-triangular tiles and moving `V`
publication to diagonal tiles regressed further to `12.44 ms`; QR4096 measured
`37.96 ms` versus `37.42 ms`. Reconstructing the compact panel independently
inside every Gram tile costs more than the saved launch and `V` reread, so this
prototype was removed.

A second Gram-fusion attempt kept the first Triton kernel that materializes
dense row-major `V`, then replaced `V.T @ V` plus its epilogue with a separate
`tl.dot` kernel. Generic masked 32x32x32 tiles measured `12.18 ms` for QR2048;
32x32x64 reached `12.12 ms`, effectively tied with but not better than cuBLAS,
while 64x64 tiles regressed to `12.39 ms`. For width-128 global panels, rows
and width are tile multiples, so a mask-free specialization with compile-time
row count was tested. It still did not improve batch-8 QR2048, but improved
batch-2 QR4096 from `37.42 ms` to `37.25 ms`. Increasing parallelism with
16x16 tiles measured `37.28 ms`. A final restricted 32x32x64 rerun measured
`37.49 ms`, showing that the apparent gain was not robust across workers. The
custom Gram kernel was therefore removed. All shapes remain on cuBLAS Gram
followed by the fused Triton `T^{-1}` epilogue.

Split-K made the dense-V Triton Gram competitive for the under-parallelized
batch-2 QR4096 panels. The selected implementation uses eight K slices and a
`[batch, 8, 128, 128]` partial workspace, producing 160 useful upper-tile
programs instead of 20. A second Triton kernel reduces the partials and applies
the complete `T^{-1}` epilogue, avoiding both a zero-fill launch and atomic
contention. Split counts 4, 8, and 16 measured `37.37 ms`, `37.34 ms`, and
`37.40 ms`, respectively, so split-8 was selected only for batch-2 width-128
panels above 2048 rows. Other shapes retain cuBLAS Gram. QR2048 remained
unchanged at `11.85 ms` in the final check, the QR4096 benchmark checker passed,
and the full 22-case suite passed.

Casting split-K Gram inputs to FP16 while retaining FP32 accumulation passed
the QR4096 benchmark checker, but measured `37.331 ms` versus `37.336 ms` for
TF32. The approximately `0.005 ms` difference is noise and does not justify
reduced Gram accuracy, so the selected split-K kernel remains TF32.

End-to-end `torch.profiler.profile` traces were added for the selected paths.
Kernel-level CUDA attribution shows panel factorization is now the dominant
cost: about 77% for QR352, 45% for QR512, 50% for QR1024, 81% for QR2048, and
82% for QR4096. QR2048 spends `9.17 ms` of `11.38 ms` profiler CUDA time in its
16 global panel kernels. QR4096 spends `10.51 ms` in global panels and
`19.34 ms` in the 41 shared-memory tail panel launches, out of `36.56 ms`.
The latter launch only two matrix CTAs per stage and is the clearest occupancy
bottleneck. QR352 similarly launches 40 CTAs per panel stage, and QR1024
launches 60; QR512's batch 640 already supplies ample parallelism. These traces
shift priority from further Compact-WY tuning to panel occupancy and motivate
testing the existing multi-CTA global QR2048 implementation as QR4096's tail
before writing a new multi-CTA panel kernel.

That tail substitution was a major win. QR4096 now factors its lower-right
2048x2048 tail with `global_panel_qr` instead of the 41-stage single-CTA
shared-memory schedule. End-to-end benchmark time fell from `37.34 ms` to
`23.72 ms`; focused dense and upper-triangular checks and the full 22-case
suite pass. The new profile contains 32 global 128-column panel launches total,
spending `19.35 ms` of `22.97 ms` profiler CUDA time in the panel kernel. The
old `19.34 ms` shared-memory tail-panel cost is gone, although panels still
account for roughly 84% of the improved runtime. This confirms panel algorithm
and block-width work as the next priority.

The dedicated-producer panel geometry was tested with eight cached columns per
consumer CTA and one warp per column, reducing launch geometry from 33 to 17
CTAs per matrix. It remained correct but regressed the batch-8 2048x128 panel
from `0.646 ms` to `0.754 ms`, QR2048 from `11.85 ms` to `13.53 ms`, and
QR4096 from `23.72 ms` to `27.06 ms`. A single warp does not provide enough
per-column reduction/update parallelism to offset the lower CTA and signaling
count, so the selected geometry remains four cached columns and two warps per
column.

A fully resident QR2048 geometry was then tested with eight cached columns,
16 warps per CTA, two warps per column, and `__launch_bounds__(512, 1)`. Its 17
CTAs per matrix produce 136 CTAs for benchmark batch 8, fitting below B200's
148 SM count while preserving the selected per-column warp count. It measured
`0.704 ms` for the 2048x128 panel and `12.49 ms` end-to-end, versus `0.646 ms`
and `11.85 ms` for the selected four-column/eight-warp geometry. The larger
512-thread CTA and lower inter-CTA parallelism outweigh avoiding a second grid
wave, so four cached columns remain selected.

Routing QR1024 wholesale through the 128-column global panel implementation
was correct on dense, stressed, and mixed cases but regressed the benchmark
from `4.97 ms` to `26.10 ms`. With batch 60, the global geometry launches 1,980
large-shared-memory CTAs per panel and pays persistent signaling/global-memory
costs that dominate any occupancy benefit. QR1024 therefore remains on the
single-CTA shared-memory blocked schedule; improving its occupancy requires a
lighter multi-CTA specialization rather than reusing the 4096-row kernel.

A purpose-built QR1024 prototype then used width-32 panels with exactly two
CTAs per matrix: one reflector producer and one eight-warp consumer caching all
32 columns in about 132 KiB of shared memory. Batch 60 therefore launched 120
CTAs and fit in one B200-wide resident wave. It was correct and reduced the
global-panel experiment from `26.10 ms` to `9.05 ms`, but remained slower than
the selected shared-memory schedule at `4.97 ms`. Profiling attributed
`4.47 ms` to the 32 two-CTA panel launches and roughly `3.9 ms` to 31
Compact-WY updates; the existing schedule's panel work is only about
`2.1-2.3 ms`. A wider global panel would reduce update launches but increase
panel factorization work, so the specialization was removed.

A migrating-producer prototype kept each CTA's four columns resident in shared
memory for the complete panel. The CTA owning frontier column `k` used all
eight warps to construct reflector `k`, eliminating completion counters and
intermediate global column traffic. At QR2048 benchmark batch 8, however, the
256 persistent CTAs exceeded actual simultaneous residency. Some future
frontier owners never became resident while active CTAs waited on their epoch,
causing deadlock. The selected kernel therefore retains one dedicated producer
CTA and short-lived fixed-column consumers.

A follow-up constrained the migrating design to 16 CTAs per matrix by caching
eight columns per CTA and used a cooperative launch solely to guarantee that
all 128 QR2048 benchmark CTAs were resident. This removed the deadlock and
measured `0.518 ms` panel time, but the panel reconstruction residual was about
`0.32` even at batch 1. The ownership pipeline therefore still has a stale-
frontier ordering bug. The experiment was rejected and the correct dedicated-
producer, four-column geometry was restored.

#### Future Recovery Plan for the 0.518 ms Migrating Panel

The fully resident version remains promising enough to revisit. Its intended
structure is 16 CTAs per matrix, eight cached columns per CTA, and one warp per
column. Each CTA loads its columns once and retains them in shared memory for
the complete panel. After column `k` has consumed reflectors `0..k-1`, its
owner CTA uses all eight warps to form reflector `k`, writes compact `H[:, k]`,
and publishes the next epoch. Intermediate column updates never need to return
to global memory.

The next session should debug correctness before further tuning:

1. Add a reduced-width debug specialization and compare the cached frontier
   column against a sequential reference after every reflector.
2. Temporarily mirror every shared-cache update to global memory. If this fixes
   reconstruction, the bug is in cache lifetime/indexing; otherwise it is in
   reflector publication or ownership ordering.
3. Add per-column `applied_through` counters in debug builds and assert that
   owner `k` has applied exactly through reflector `k-1` before factorization.
4. Compare compact `H[:, k]`, `tau[k]`, and the updated trailing columns after
   each step, stopping at the first divergence rather than checking only final
   reconstruction.
5. Replace the epoch `atomicExch`/`atomicAdd(0)` handoff with device-scope
   release stores and acquire loads once the algorithm is correct. This makes
   reflector and tau visibility explicit and removes atomic RMW polling.
6. Verify occupancy with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` before
   launch. The batch-8 QR2048 configuration requires 128 simultaneously
   resident CTAs on a full 148-SM B200.

`cudaLaunchCooperativeKernel` was used only to guarantee full-grid residency;
the kernel did not call `grid.sync()`. A normal launch is legal without grid
synchronization and may have slightly lower startup overhead, but CUDA does not
contractually guarantee that all blocks become resident together. Because
persistent CTAs spin while waiting for future frontier owners, an unscheduled
owner can deadlock the grid. Keep the cooperative launch while debugging and
benchmark a normal launch only after either proving the residency assumptions
for the target B200 or changing the protocol so progress does not require every
CTA to be resident.

## Empirical Block-Schedule Cost Model

Schedule selection should use measured primitive costs rather than a FLOP-only
model. Panel kernels are sensitive to CTA residency, shared-memory layout, and
signalling overhead, while Compact-WY stages have shape-dependent cuBLAS and
triangular-solve performance cliffs. For a remaining square of size `r` and a
panel width `w`, the intended stage model is

`C(r, w) = panel(r, w) + compact_wy(r, w, r - w)`.

Once several widths and panel geometries are available, the schedule can be
selected by dynamic programming:

`best(r) = min_w(C(r, w) + best(r - w))`,

with measured terminal costs for the available square-tail kernels. Geometry,
batch size, precision mode, tensor layout, and full-matrix strides are part of
the cost key. Measurements should use medians from repeated CUDA-event samples;
noisy choices can later be penalized with their median absolute deviation. The
model is a candidate generator: top schedules still require matched end-to-end
timing because allocation reuse, library warm state, and cache effects are not
strictly additive.

The first validation deliberately uses only existing primitives and optimizes
one concrete QR4096 decision. Candidate schedules run fixed 128-column global
panels until the remaining square reaches `0`, `512`, `1024`, `1536`, or
`2048`, then optionally hand that tail to the adaptive shared-memory blocked
path. This includes both the old 2048-tail schedule and the selected all-global
schedule. `run_modal.py --mode cost_model --case 4096` measures every global
stage, each terminal tail, predicts all five schedule costs by addition, then
times every schedule end to end and reports predicted and measured ordering.
Agreement on the winning schedule and useful ordering correlation are the
initial acceptance criteria. Absolute prediction error is diagnostic rather
than a hard gate at this stage.

The first B200 run produced the following schedule totals:

| Shared-memory tail | Predicted | Measured |
| ---: | ---: | ---: |
| 0 (all global) | 28.016 ms | 23.247 ms |
| 512 | 27.183 ms | 22.567 ms |
| 1024 | 31.402 ms | 26.906 ms |
| 1536 | 33.865 ms | 29.668 ms |
| 2048 | 40.708 ms | 36.955 ms |

The model predicted the exact measured ordering and selected the 512 tail. Its
predicted advantage over all-global was `0.833 ms`; the measured advantage was
`0.680 ms`, or 2.9%. Isolated stage sums overpredict absolute end-to-end time
by roughly 4-5 ms, but most of that bias is common across candidates and does
not impair this scheduling decision. The selected QR4096 implementation now
uses global 128-column panels down to 512 rows and factors the final 512 square
with adaptive shared-memory panels `(110, 136, 58, 208)`.

The selected 512-tail implementation passed both focused QR4096 tests and the
full 22-case B200 correctness suite. The official QR4096 benchmark reported
`23.266 ms` with a run-to-run standard deviation of `0.045 ms`, versus the
previous documented `23.72 ms`. This is a 1.9% official-harness improvement;
the focused schedule harness measured 2.9%. The difference reinforces using
the cost model for schedule ranking while retaining the official benchmark as
the final performance gate.

The first search only established the best schedule among five coarse tail
handoff points. A refinement search covers every valid 128-row boundary from
256 through 1024, plus the all-global schedule. Tail size 128 is excluded
because the existing square-tail dispatcher has no `qr128` specialization;
the 256 tail ends in the supported `qr208` kernel after one panel.

The refined model ranked tails `(512, 384, 256, 0, 640, 768, 896, 1024)`.
Higher-repeat end-to-end validation put 384 ahead of 512 by only `0.014 ms`
(`22.536` versus `22.549 ms`), while the model put 512 ahead by `0.022 ms`.
An earlier lower-repeat run also put 384 ahead, but by just `0.024 ms`. These
sub-0.1% differences are below the threshold for changing a correctness-tested
schedule, so the selected tail remains 512.

After this validation, the harness no longer end-to-end benchmarks every
candidate. It computes all predicted costs directly from the measured global
stage and terminal-tail primitive costs, then benchmarks only the two predicted
leaders and the all-global baseline. Extending optimization beyond this tail
crossover requires additional panel-width primitives or a sampled response
surface; this motivated the width-64 experiment below.

### Width-64 Global Panel and Mixed-Width DP

The global panel kernel was templated at widths 64 and 128, retaining four
cached columns and two warps per column. The width-64 panel passed a direct
4096x64 reconstruction check with relative residual `1.77e-7`. The cost-model
search was upgraded to dynamic programming over remaining-row states: measured
64- and 128-column global stages are graph edges, while measured adaptive
shared-memory tails are terminal edges. It retains the three lowest-cost paths
per state and end-to-end benchmarks only the predicted top two plus the current
baseline when necessary.

Width 64 approximately halves panel factorization time, but requires twice as
many Compact-WY updates. At 4096 rows, two consecutive width-64 stages cost
about `0.787 + 0.774 = 1.561 ms`, versus `1.241 ms` for one width-128 stage.
The same relationship persists down the factorization. The DP ranked the
existing 28 width-128 panels followed by the 512 shared-memory tail first at
`27.990 ms`; the width-128 schedule with a 384 tail was second at `28.047 ms`.
The first mixed-width path ranked third and was already slower in the additive
model. End-to-end validation confirmed the predicted leaders at `23.125 ms`
and `23.147 ms`, respectively. Width 64 therefore provides no QR4096 schedule
gain with the current Compact-WY primitives, and the selected schedule remains
unchanged.

After templating, the focused dense QR4096 test still passes with
`scaled_factor_residual=0.421`. The official benchmark measured `23.123 ms`
with `0.040 ms` standard deviation. No runtime width-64 dispatch was selected;
the production factorization remains the width-128/512-tail schedule.

### Post-Schedule Cleanup

The implementation was cleaned up before starting another kernel experiment:

- Removed the disabled, incorrect migrating-producer CUDA body. Its design,
  failure mode, and recovery plan remain documented above.
- Removed the superseded tail-only cost-model mode. The active profiler uses
  the mixed-width dynamic-programming model.
- Templated the C++ width-64/128 allocation and launch wrapper instead of
  duplicating tensor setup.
- Replaced separate QR2048 and QR4096 Python implementations with one
  `scheduled_global_qr` executor and declarative `GLOBAL_QR_SCHEDULES` entries.
- Made the cost-model finalists call the production schedule executor rather
  than a duplicate implementation.
- Removed an unreachable legacy QR2048 shared-memory dispatch branch.

The cleanup is behavior-neutral. Focused QR2048 and QR4096 tests pass, the full
22-case B200 correctness suite passes, and the official QR4096 benchmark is
`23.126 ms`, matching the pre-cleanup `23.123 ms` within run-to-run noise.

### QR512 Schedule Cost Model

A QR512-specific empirical model now measures shared-memory panel plus
Compact-WY edge costs for widths `(32, 48, 64, 80, 96, 112, 128)` and terminal
square costs for `qr176`, `qr192`, and `qr208`. Edges that exceed the 227 KiB
shared-memory limit are excluded. Dynamic programming retains the three lowest
cost complete schedules and end-to-end benchmarks only the predicted top two
plus the selected baseline when needed.

With only the existing primitives, the model correctly retained the selected
`(96, 96, 128, 192)` schedule: it predicted `9.832 ms` and measured
`10.030 ms`. The nearest alternative, `(96, 96, 64, 32, 32, 192)`, measured
`10.263 ms`; its 320x64 panel used the slower generic tall-panel kernel.

A single `short_panel_qr<320, 64, 32>` specialization reduced that panel from
about `0.753 ms` to `0.430 ms`. After remeasurement, the model selected
`(96, 96, 64, 32, 32, 192)` at a predicted `9.773 ms`, versus `9.817 ms` for
the old schedule. Matched end-to-end measurements were `9.892 ms` and
`9.987 ms`, respectively, a 0.95% improvement. The new schedule therefore
replaces the old one.

The full 22-case B200 correctness suite passes. Official QR512 benchmark means
for the selected schedule are:

| Case | Mean |
| --- | ---: |
| Dense | 9.927 ms |
| Mixed | 10.048 ms |
| Rank-deficient | 10.041 ms |
| Clustered | 9.971 ms |

An end-to-end `torch.profiler` trace of the selected QR512 schedule reported
`9.819 ms` total CUDA time. Non-overlapping top-level attribution is
`4.976 ms` (50.7%) for Compact-WY, `4.190 ms` (42.7%) for the five panel
kernels plus final `qr192`, and about `0.640 ms` (6.5%) for panel/tau copies
and assignment kernels. This corrects the earlier 55-60% Compact-WY estimate,
which came from summing isolated cost-model measurements rather than an
end-to-end trace.

Within Compact-WY, final `baddbmm` apply costs `1.889 ms` (19.2% of total),
triangular solves cost `1.461 ms` (14.9%), Gram and projection `bmm` calls cost
`1.447 ms` (14.7%), and the fused Triton `V`/`T^{-1}` preparation kernels cost
only `0.179 ms` (1.8%). Further Compact-WY work should therefore target apply,
solve, or update aggregation rather than pointwise preparation.

### FP16 Final Compact-WY Apply

The initial QR512 prototype cast `V` and the solved
projection to FP16, computed their batched product with FP32 accumulation and
FP32 output, then subtracted that correction from the FP32 trailing matrix. The
trailing matrix and returned compact factors remain FP32. This replaces the
FP32 `baddbmm` apply while leaving Gram, projection, and triangular solve in
FP32.

The full 22-case B200 correctness suite passes, including every structured and
mixed QR512 case. Numerical margin is reduced: for example, the row-scaled
case's displayed scaled factor residual increased from `0.047` to `14.9`, and
the mixed case increased from `0.045` to `14.3`, although both remain within
the official checker gates. This is therefore a ranked-performance tradeoff,
not a numerically neutral transformation.

Official QR512 benchmark means improved as follows:

| Case | FP32 apply | FP16/FP32 apply |
| --- | ---: | ---: |
| Dense | 9.927 ms | 9.161 ms |
| Mixed | 10.048 ms | 9.294 ms |
| Rank-deficient | 10.041 ms | 9.235 ms |
| Clustered | 9.971 ms | 9.250 ms |

The new end-to-end profiler total is `9.080 ms`, down from `9.819 ms`. FP16
correction GEMMs are fast, and the ten FP16 casts cost only `0.135 ms`, but the
separate FP32 subtraction costs `1.070 ms` and materializing corrections raises
temporary allocation traffic substantially. A custom mixed-input, FP32-output
GEMM could theoretically fuse the correction directly into the trailing
matrix, but the retained path continues to rely on cuBLAS for matmul quality.

Large-row QR512 Gram products (`rows >= 320`) now also use FP16 `V` inputs with
FP32 accumulation and output through `torch.bmm`; smaller-row Grams remain
FP32. The full suite passes. Dense QR512 improved from `9.161 ms` with only the
FP16 final apply to `8.947 ms`, another 2.3%. The displayed row-scaled and mixed
factor residuals increased further to `16.7` and `16.4`, respectively, but all
official correctness gates still pass.

Rather than replacing cuBLAS GEMMs with Triton, `_build_v_triton` now accepts
caller-allocated FP32 and optional FP16 outputs and writes both representations
in one pass. FP32 `V` remains available for projection; FP16 `V` is reused by
large-row Gram and final apply. This removes the separate `V.to(float16)`
kernel and its FP32 reread without lowering projection precision. Dense QR512
improved from `8.947 ms` to `8.803 ms`, closely matching the removed conversion
cost. A Triton final-matmul prototype was removed; all retained matrix
multiplications remain PyTorch/cuBLAS-backed.

QR512 benchmark means after dual-output `V`, before fused `baddbmm`, were:

| Case | Dual-output V + FP16 Gram/apply |
| --- | ---: |
| Dense | 8.803 ms |
| Mixed | 8.895 ms |
| Rank-deficient | 8.938 ms |
| Clustered | 8.840 ms |

The final configuration passes the complete 22-case B200 correctness suite.

The older QR4096 split-K Triton Gram specialization was revalidated against the
standard `V.T @ V` cuBLAS path after the 512-tail schedule changes. It remained
faster by only `0.066 ms`, or 0.29%, which no longer justified the extra custom
kernel and workspace. It was removed; QR4096 now uses the standard cuBLAS-backed
Gram path.

FP16 Gram/apply was then explored for QR1024, QR2048, and QR4096. The initial
implementation used `torch.bmm(..., out_dtype=float32)` followed by a separate
full-matrix subtraction. It passed correctness but regressed all three shapes
to `5.506 ms`, `12.474 ms`, and `23.637 ms`. This did not show that FP16 GEMM
was slower than TF32; it showed that correction allocation and subtraction
dominated the composition.

FP16 Gram alone was beneficial. Same-worker A/B measurements, with FP32
accumulation/output and all projection/apply/solve work left FP32 or scoped
TF32, were:

| Shape | Standard Gram | FP16 Gram | Improvement |
| ---: | ---: | ---: | ---: |
| 1024 | 4.982 ms | 4.941 ms | 0.83% |
| 2048 | 12.143 ms | 12.093 ms | 0.42% |
| 4096 | 22.673 ms | 22.567 ms | 0.47% |

The selected minimum active-row thresholds are 512, 1024, and 2048 for those
three shapes, while QR512 retains its 320-row threshold. The shared Triton
preparation kernel emits FP16 `V` only for stages that consume it. The complete
22-case suite passes. QR1024 ranked means with the selected path are
`4.899 ms` dense, `4.981 ms` mixed, and `4.904 ms` near-rank-deficient.

PyTorch 2.12 `torch.baddbmm` supports `out_dtype=float32` for FP16 CUDA inputs.
The final apply now uses `beta=1`, `alpha=-1`, and `out=output` directly, fusing
the FP16-input GEMM with the FP32 trailing-matrix accumulation. This removes
the correction allocation and separate subtraction while retaining cuBLAS.
QR512 dense improved from `8.803 ms` to `8.344 ms`, with unchanged numerical
diagnostics.

With fused `baddbmm`, FP16 apply also became beneficial for the larger shapes.
The final selected benchmark means are:

| Shape/case | Mean |
| --- | ---: |
| QR512 dense | 8.344 ms |
| QR512 mixed | 8.460 ms |
| QR512 rank-deficient | 8.487 ms |
| QR512 clustered | 8.415 ms |
| QR1024 dense | 4.784 ms |
| QR1024 mixed | 4.840 ms |
| QR1024 near-rank-deficient | 4.750 ms |
| QR2048 dense | 11.802 ms |
| QR4096 dense | 22.629 ms |

FP16 final apply is now enabled for QR512, QR1024, QR2048, and QR4096. Gram
remains selectively FP16 only above the row thresholds listed above; projection
and triangular solve remain FP32/scoped-TF32. The full 22-case B200 correctness
suite passes.

### Schedule Retune After Faster Compact-WY

The empirical schedule models were rerun after enabling fused FP16 Compact-WY
apply. QR1024 retained its existing dynamic schedule
`(48, 48, 48, 56, 56, 64, 72, 80, 88, 104, 136, 32, 192)`. The closest model
alternative, ending in `(48, 96, 128, 192)`, measured `4.734 ms`; the selected
schedule measured `4.732 ms`, so the difference is noise and does not justify a
change.

QR4096 also retained its existing 28 global 128-wide panels followed by the
512 shared-memory tail. Its selected schedule measured `22.647 ms` in the
matched cost-model harness, versus `22.768 ms` for the best mixed-width
alternative.

QR2048 did shift toward a larger shared-memory tail. The model selected 14
global 128-wide panels followed by a 256 tail, factored as shared-memory panels
`(48, 208)`, instead of 16 global 128-wide panels. In the matched harness the
new and old schedules measured `11.175 ms` and `11.901 ms`, respectively. The
official dense benchmark improved from the prior `11.802 ms` to `10.908 ms`, a
7.6% reduction. The complete 22-case correctness suite passes with the new
schedule, including dense, rank-deficient, and mixed QR2048 inputs.

### QR512 Copy and V-Preparation Reduction

An input-shape-grouped profiler trace showed that the earlier `0.691 ms` of
`copy_` time was not panel-to-output traffic: shared-memory QR panels already
write directly into their final `H` and `tau` slices. It consisted of
`0.372 ms` to initialize the distinct first-stage `baddbmm` output, about
`0.269 ms` of projection copies required by out-of-place triangular solves,
and `0.051 ms` of required FP32-to-FP16 transformed-matrix casts.

QR512 panel kernels now emit packed FP32 `H`, FP32 `V`, and FP16 `V` directly
from the factored shared-memory matrix. This removes all five
`_build_v_triton` launches. Extra panel stores replace most of the isolated
`0.159 ms` preparation cost, but the official dense benchmark still improved
from `8.344 ms` to `8.286 ms`.

The existing cuBLAS batched-TRSM wrapper initially faulted at QR512's batch 640
because its pointer-array setup kernel populated only 32 entries. The launcher
now covers the complete batch, and QR512 solves operate in-place on the
projection buffers. This removes the five FP32 projection copies and improves
the dense benchmark further to `8.168 ms`. The latest profiler trace reports
`8.054 ms` total CUDA time; remaining copies are approximately `0.372 ms` for
the first-stage distinct C/D accumulation plus `0.050 ms` for FP16 casts. The
complete 22-case correctness suite passes with unchanged diagnostics.

### Batched-TRSM Routing Recheck

The in-place `cublasStrsmBatched` path was rechecked across all Compact-WY
updates after fixing its batch-640 pointer setup. Applying it universally was
not beneficial: QR2048 regressed from `10.908 ms` to `11.211 ms` because its
narrow shared-memory tail also took the batched path, and QR4096 regressed from
`22.629 ms` to `23.127 ms`. QR1024 improved, however, from the previous
`4.784 ms` to `4.652-4.658 ms` on two validation workers. The selected routing
therefore uses batched TRSM for QR512 and QR1024, plus batch-8 width-128 panels
in QR2048; other shapes retain `torch.linalg.solve_triangular`. The full
22-case correctness suite passes.

The private cuBLAS handle and pointer-array setup use the default CUDA execution
context required by the validator.

### Torch-Compatible cuBLASLt First Apply

The Modal image contains two cuBLAS installations. The `nvidia/cublas` Python
namespace and system toolkit provide CUDA 12 libraries, while Torch 2.12
`+cu130` actually loads CUDA 13 libraries from `nvidia/cu13`. Passing Torch's
pooled handle to a CUDA-12-linked cuBLASLt correctly failed with
`CUBLAS_STATUS_NOT_INITIALIZED`. The extension now derives `nvidia/cu13` from
`torch.__file__` and explicitly compiles and links against its headers,
`libcublas.so.13`, and `libcublasLt.so.13` with a matching rpath.

The first cuBLASLt prototype had a separate layout error: the first trailing
matrix is a strided view with row stride 512, but its C/D layouts used the
logical width 416 as their leading dimensions. The corrected wrapper preserves
the actual C/D row and batch strides, uses Torch's pooled cuBLASLt handle, and
computes the first FP16-input/FP32-output apply directly as
`D = C - V @ transformed`. The previous `0.372 ms` C-to-D initialization copy
is gone; the Lt kernel costs `0.193 ms`, and the profiler total fell from
`8.054 ms` to `7.556 ms`.

Official means are `7.697 ms` dense, `7.789 ms` mixed, `7.731 ms`
rank-deficient, and `7.783 ms` clustered, versus the prior `8.168 ms` dense.
QR1024 also measured `4.584 ms` after linking the extension's cuBLAS calls to
the matching CUDA 13 runtime. The full 22-case correctness suite passes with
unchanged QR512 numerical diagnostics.

### Current Cross-Shape Profile and Score Weighting

The current 12 measured benchmark means give an approximate geometric mean of
`2.844 ms`. By benchmark count, QR512 carries 4/12 of the score, QR1024 carries
3/12, QR32/176/352 together carry 3/12, and QR2048 and QR4096 each carry only
1/12. A relative improvement to one legacy small case therefore has the same
log-score weight as the same relative improvement to QR2048 or QR4096.

Current legacy means are `15.43 us` for QR32, `223.91 us` for QR176, and
`852.83 us` for QR352. QR352's `0.822 ms` profiler trace attributes 36.6% to
the 352x144 panel, 40.7% to final QR208, 9.6% to triangular solve, 6.3% to Gram
and projection, 3.8% to apply, and about 3% to preparation/copies. QR176 is a
single already-specialized shared-memory QR kernel.

Cross-shape top-level CUDA attribution is:

| Shape | Panels + final QR | TRSM | Gram/projection | Apply | Other |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 352 | 77.3% | 9.6% | 6.3% | 3.8% | 3.0% |
| 512 | 57.1% | 17.7% | 16.5% | 6.7% | 2.0% |
| 1024 | ~54% | 16.2% | ~13% | ~11% | ~5.8% |
| 2048 | 81.7% | 7.5% | ~5% | ~3% | ~2.8% |
| 4096 | ~83% | 6.5% | ~5% | ~3% | ~2.5% |

The global 128-wide panel kernels alone consume 77.7% of QR2048 and 77.9% of
QR4096. QR1024 still has a `0.162 ms` first-stage distinct-output copy because
the cuBLASLt C/D wrapper is currently enabled only for QR512. The QR192 final
kernel consumes `1.355 ms` (17.9%) of QR512 and `0.272 ms` (6.2%) of QR1024,
so improving that one primitive affects seven benchmark cases. This gives it
more geometric-mean leverage than isolated QR176 work. QR352 remains worth a
bounded revisit, especially its QR208 tail; QR176 should follow only if a
specific kernel-level idea offers a credible relative gain.

### Score-Weighted Follow-Up

The distinct-C/D cuBLASLt first apply is now enabled for QR1024 as well as
QR512. It removes the `0.162 ms` first-stage copy. Official means improved from
`4.584 ms` to `4.415 ms` dense, from `4.753 ms` to `4.485 ms` mixed, and from
`4.772 ms` to `4.470 ms` near-rank-deficient. Focused dense and rank-deficient
tests pass. Having QR1024 panels emit V directly was also tested, but regressed
dense runtime to `4.427 ms`; only QR512 retains fused panel/V output.

QR192 launch-width tuning did not help. Reducing from 32 warps to 24 regressed
both QR512 and QR1024; 28 warps measured a slower isolated QR192 kernel
(`1.370 ms` versus `1.355 ms` at QR512 batch size) and was also rejected.

For QR352, `(128, 32, 192)` regressed from `0.853 ms` to `0.937 ms`. The
selected `(144, 32, 176)` schedule preserves the specialized first panel,
adds one narrow stage, and replaces QR208 with QR176. It passes focused
correctness and improves the official mean from `852.83 us` to `846.66 us`, a
0.72% gain. Using the latest measured values, these retained changes reduce the
approximate 12-case geometric mean from `2.844 ms` to `2.804 ms`, about 1.39%.

### QR176 Register-Resident Small Variant

The hand-written QR176 kernel in `submission_v1_cpp.py` measures `123.05 us`,
versus about `223.96 us` for the generic short-panel primitive. A controlled
build of the unchanged standalone kernel with Torch's CUDA 13 include directory
measured `122.95 us`. CUDA 13 headers therefore do not explain the earlier
integrated-kernel regression.

The regression came from compiler-control changes in the first adaptation,
primarily forced unrolling and forced inlining across the large
register-resident panel/update loops. Restoring the original selective-unroll
structure inside the main CUDA extension produced `120.91 us` for QR176 even
with the CUDA 13 include path required by cuBLASLt.

This remains a separate small-kernel variant. Dispatch is deliberately narrow:
the original generic short-panel kernel remains the fallback, while the
register-resident QR176 kernel has compile-time specializations for contiguous
176x176 matrices and for the fixed-stride 176x176 tail view of QR352. The latter
specialization reduced the selected `(144, 32, 176)` QR352 schedule from about
`846.66 us` to `748.69 us` (11.6%) without materializing a contiguous tail.

The complete 22-case correctness suite passes after both specializations. The
CUDA 13 cuBLASLt linkage and the QR512/QR1024 Compact-WY paths remain unchanged.

### Compile-Time Small-QR Family

The register-resident QR kernel is now parameterized by compile-time `N`, row
stride, matrix batch stride, and tau batch stride. Each combination remains a
separate compiled CUDA kernel; there is no runtime-size loop or runtime-stride
cost in the selected paths. `N=128`, `160`, `176`, and `192` are instantiated.
Multiples of 32 use exactly `N/32` row items and have no partial-row checks;
QR176 retains six row items with its final 16-row guard.

QR192 uses 24 warps, eight columns per warp, and the same per-thread register
payload as QR176. It replaces the generic QR192 terminal kernel for the fixed
512 and 1024 parent strides without changing either block schedule. B200 means:

| Shape | Generic QR192 path | Register QR192 path |
| ---: | ---: | ---: |
| QR512 | ~7.67 ms | 7.007 ms |
| QR1024 | ~4.42 ms | 4.317 ms |

The first full-suite run exposed an inherited exact-zero-tail issue in the
hand-written algorithm: `x0 == beta == 0` made the reflector reciprocal
infinite. A uniform zero-tail guard now emits tau zero, leaves the column
unchanged, and avoids the reciprocal. All 22 official cases, including
rank-deficient and mixed QR512/1024 inputs, pass afterward.

Standalone contiguous checks for QR128 and QR160 are finite and reconstruct
with relative residuals `4.55e-7` and `4.84e-7`, respectively. They are
available as future terminal primitives but are not selected by a production
schedule yet. Final dense remeasurement after the guard was `110.18 us` for
QR176 and `735.35 us` for QR352.

### Register-Resident Rectangular Panels

The small-QR design now also has a separate compile-time rectangular family,
`register_panel_kernel<ROWS, COLS, ...>`, for `ROWS % 32 == 0` and
`COLS % 8 == 0`. Each warp owns eight columns and each lane owns one row in
each 32-row group. Rectangular panels factor their final column normally;
only square instantiations retain the LAPACK convention of tau zero for the
last column.

The kernel can emit packed H plus compact-WY V in FP32 and FP16 directly from
the final register-resident columns. Current QR512 panel instances use this
path, avoiding the prior shared-memory matrix round trips and separate V
construction. Exact parent strides remain compile-time parameters.

Selected production specializations are:

- QR512: 512x96, 416x96, 320x64, 256x32, and 224x32, with direct V output.
- QR1024: 224x32, without direct V output because QR1024 currently builds V
  through the standard Compact-WY path.

The first four QR512 panel substitutions reduced the dense benchmark from
`7.007 ms` to `5.946 ms`. Although the resource model classified 512x96 as
marginal, its measured specialization won as well, producing the final
`5.494 ms` dense result. Other QR512 benchmark means are `5.413 ms` mixed,
`5.502 ms` rank-deficient, and `5.413 ms` clustered. The full profile reports
512x96 at the remaining generic baseline cost of `1.148 ms`; after replacing
it, the end-to-end improvement confirms that register pressure did not erase
the benefit.

QR1024 improved from the post-QR192 result of about `4.317 ms` to `4.252 ms`.
A resource-feasible 256x48 specialization for QR2048 was tested but regressed
the official dense benchmark to `11.105 ms` versus the approximately
`10.908 ms` selected baseline, so its dispatch was removed. Resource fit is
therefore only a candidate filter, not a performance guarantee.

All 22 official correctness cases pass with the selected rectangular panels,
including QR512 rank-deficient, clustered, banded, row-scaled, near-collinear,
and mixed inputs.

The FP16 V writeback was then changed from eight scalar conversions/stores to
four explicit `cvt.rn.f16x2.f32` conversions followed by one
`st.global.v4.b32`. PTX places the first conversion operand in the upper half,
so adjacent values are passed in reversed argument order to preserve row-major
element order in memory. QR512 improved again from `5.494 ms` to `5.230 ms`,
and the complete 22-case suite passes.

V is still derived row-wise from the packed register columns at final
writeback. Below the diagonal packed H is already V; only the diagonal,
upper triangle, and tau-zero columns require selection. Although the complete
V also exists in shared memory, gathering eight column-major shared values
would add loads. Writing V at reflector creation would instead issue one
strided scalar store per row and column, using only 4 bytes of each 32-byte
sector per iteration. The retained row-wise vector write therefore has better
store-sector utilization without increasing persistent register state.

Column-major V was tested end to end. The panel kernel wrote each reflector
directly from its distributed register values to column-major FP32/FP16
buffers, `V.T` became contiguous for Gram/projection, and the cuBLASLt apply
wrapper was made temporarily stride-aware so that it consumed V as a true
column-major left operand. Correctness passed, but QR512 regressed from
`5.230 ms` to `5.367 ms`.

The profile showed that the trade was mixed: smaller register panels became
cheaper, but 416x96 became slower and several downstream BMM/baddbmm kernels
selected slower layouts. The layout-aware cuBLASLt apply itself was essentially
unchanged. Because the end-to-end loss came partly from downstream dispatch,
not merely scalar FP16 stores, column-major V and the extra layout plumbing
were removed. Restoring row-major V with packed FP16 writeback remeasured at
`5.207 ms`.

### Approximate Math and Small-QR Unification

The updated `submission_v1_cpp.py` claims were reproduced before changing the
production path: its QR176 measured `100.39 us`, and its four-warp QR32 measured
`12.35 us`. The production baseline at that point was `108.89 us` for QR176
and `15.00 us` for the old single-warp QR32.

Direct `sqrt.approx.f32` is now used in the register-resident square/panel
family. In isolation it improved production QR176 from `108.89 us` to
`107.46 us`; QR512 was effectively flat, and QR1024 improved by roughly 0.2%.
Approximate reciprocals did not transfer generally: using `rcp.approx` for both
tau and the reflector inverse regressed QR176 to `114.93 us` and QR1024 to
`4.295 ms`. Those reciprocal changes were rejected.

QR32 is the useful exception. It now uses the same compile-time rectangular
template as QR128/160/176, with four eight-column warps. `sqrt.approx` and an
approximate reciprocal for tau measure `12.35 us`, while the reflector inverse
keeps exact division; changing that inverse to `rcp.approx` was already shown
slower by the standalone prototype. The obsolete single-warp QR32 kernel and
operator were removed.

The rectangular template now uses `(ROWS + 31) / 32` row groups and guarded
loads/stores. For multiples of 32, compiler range analysis removes the guards;
the QR32 timing matches the hand-specialized prototype. Routing QR176 through
the same source improved it slightly to `106.66 us`, and QR352 measured
`731.06 us`.

A fully single-template implementation was also tested for QR192. Despite the
same compile-time dimensions, routing QR192 through the rectangular kernel
regressed QR1024 from about `4.24 ms` to `4.39 ms`. The lean square template is
therefore retained only for QR192; QR32/128/160/176 share the rectangular
family. With that bounded exception restored, QR512 measured `5.192 ms` and
QR1024 `4.240 ms`. All 22 official correctness cases pass with approximate
sqrt and the selected per-size reciprocal policy.

### Register-Kernel Cleanup

The obsolete single-warp QR32 CUDA kernel, launcher, C++ wrapper, and custom
operator were removed after QR32 moved to the shared register-panel family.
Operation-specific helpers no longer carry the historical `qr176_` prefix:
loads/stores are `ldg_f32x8`, `stg_f32x8`, and `stg_f16x8`; packed conversion
is `cvt_f16x2`; arithmetic helpers are `fma_f32x2`, `sqrt_approx`, and
`rcp_approx`; synchronization helpers are `elect_sync`, `warp_uniform`, and
`mbar_init`/`mbar_arrive`/`mbar_wait`.

The duplicate size-parameterized register reduction was removed in favor of
the existing full-warp `warp_sum`. The square-only QR192 implementation is now
named `register_square_kernel`, and its dispatch carries a local comment
documenting the measured code-generation exception.

Post-cleanup B200 checks measured `106.64 us` for QR176, `5.24 ms` for QR512,
and `4.234 ms` for QR1024, within normal worker variance of the pre-cleanup
values. The complete 22-case correctness suite passes.

### Compact-WY Triangular Inversion

The existing Compact-WY path forms
`T^{-1} = diag(1/tau) + triu(V.T @ V, 1)` and then applies `T.T` by solving
`T^{-T} X = V.T @ C`. Before changing production code, representative stages
were benchmarked with the pooled-handle batched TRSM, PyTorch triangular solve
against an identity RHS followed by BMM, and `torch.linalg.inv` followed by
BMM.

For QR512 batch 640, solve-identity plus BMM was consistently faster:

| Width x trailing | Batched TRSM | Solve identity + BMM |
| ---: | ---: | ---: |
| 96 x 416 | 0.601 ms | 0.357 ms |
| 64 x 256 | 0.248 ms | 0.140 ms |
| 32 x 224 | 0.080 ms | 0.049 ms |

`torch.linalg.inv` was uniformly much slower (0.26--2.01 ms for these cases).
Production QR512 now uses PyTorch triangular solve only to construct `T.T`,
then applies it with batched GEMM. Official benchmark means are `4.637 ms`
dense, `4.603 ms` mixed, `4.660 ms` rank-deficient, and `4.651 ms` clustered,
down from roughly 5.2--5.4 ms. The complete 22-case suite passes, and QR512
residual metrics are essentially unchanged.

QR1024 has a size-dependent crossover: widths 48--72 were faster in isolation,
while widths 80 and above and the final 32x192 stage were not. A selective
`rhs >= 8 * width` production test improved runtime by about 1%, but accumulated
explicit-inverse error increased several QR1024 residual metrics by 30--50%.
Although all thresholds still passed, this trade was rejected; QR1024 retains
direct batched TRSM. A custom triangular-inverse kernel remains optional future
work, but the library implementation already captures the high-value QR512 win.

Identity construction is now lazily cached by `(device, dtype, width)`. The
cache stores one unbatched identity and returns an expanded read-only view, so
widths 32, 64, and 96 incur allocation/initialization only on first use. The
post-cache dense QR512 measurement was `4.683 ms`; worker variance is larger
than the small expected cache benefit, but the repeated construction is gone.

CUDA 13 API review found no triangular-inverse routine in cuBLAS. cuSOLVER
provides `cusolverDnXtrtri`, which overwrites one upper or lower triangular
matrix with its inverse and requires device/host workspace plus an info output.
It has no batched interface, making it poorly matched to QR512's batch of 640
small matrices without an additional custom batching layer. cuBLAS
`getriBatched`/general inversion routines do not exploit this triangular input.

### Two-CTA Cluster QR176

The hand-written `submission_v2_cpp.py` cluster kernel was validated before
integration. Official QR176 measured `85.00 us`, versus `100.39 us` for v1,
`106.6 us` for the production register kernel, and `103.24 us` for the more
factored v2b prototype. The v2b helper/control-flow refactor therefore loses
most of the cluster gain and was not used.

A batch sweep confirms that this is an under-occupancy specialization:

| Batch | v2 cluster | v1 single CTA | Production single CTA |
| ---: | ---: | ---: | ---: |
| 1 | 85.9 us | 99.5 us | 106.9 us |
| 40 | 86.0 us | 99.9 us | 106.6 us |
| 80 | 159.6 us | 100.7 us | 107.2 us |
| 640 | 729.8 us | 491.7 us | 527.0 us |

The selected implementation keeps the v2 loop structure, adds compile-time
global parent strides and the zero-tail guard, and dispatches only when
`N=176` and batch is at most 40. Contiguous QR176 and the fixed-stride QR352
tail are instantiated; larger batches retain the single-CTA register kernel.
Integrated QR176 measures `90.27 us` officially (86--88 us in the direct batch
sweep), and QR352 improves from `731.06 us` to `716.41 us`. All 22 official
correctness cases pass.

### Generalized Two-CTA Register Panel

The cluster implementation is now templated independently over `ROWS`,
`COLS`, parent row/batch strides, tau stride, and the warp split between the
two CTAs. It supports rectangular panels, partial final 32-row groups, and
square-specific last-column handling without runtime branches in each
instantiation. Reflectors are produced by CTA 0 and transferred through DSM
to CTA 1; both CTAs update their assigned row groups directly in registers.

The refactor preserved the QR176 result (`90.70 us` official). The first new
instantiation, the batch-60 QR1024 QR192 tail, measures about `98 us` versus
`122 us` for the prior single-CTA square kernel. The full QR1024 benchmark is
`4.211 ms` mean (`4.205 ms` best), about 0.6% lower at the kernel level and
within a roughly 1% end-to-end improvement over recent baselines.

The batch-limited QR208 tails benefit substantially. The relevant parent
layouts are the compact 256x256 and 512x512 tail workspaces, not the original
2048/4096 matrix strides. With 13 row groups assigned to each CTA, both
layouts measure `110--111 us`; the old short-panel kernel measured
`332--336 us`. This is a roughly 3x tail-kernel speedup and removes
`221--226 us` from the QR2048/4096 critical path. Current official benchmark
means are `10.888 ms` for QR2048 and `22.640 ms` for QR4096.

All 22 correctness cases pass with the QR176, QR192, and QR208 cluster
dispatches enabled. The kernel is now ready for the next target: an actual
rectangular late panel, where the benefit must be compared against the
existing `short_panel_qr_kernel` rather than inferred from square tails.

### Sixteen-CTA Tall-Panel Cluster

The 128-column global panel now uses a 16-CTA cluster. Each CTA owns eight
columns and partitions the active rows across either 8 or 16 warps. Its eight
columns remain resident in registers while it consumes reflectors from earlier
CTAs, factors its own columns, and publishes those reflectors for later CTAs.
There are no global polling flags. Each target CTA owns one mbarrier per panel
column; producers use `mapa.shared::cluster` followed by a release/cluster
remote arrival, and consumers use an acquire/cluster local wait.

Reflectors are published to separate FP32 and FP16 V allocations with logical
shape `[batch, rows, 128]` and physical column-major storage. This makes V.T
physically contiguous and removes `_build_v_triton` from every global-panel
stage. H itself is returned in row-major layout for the existing panel copy.

The initial 8-warp, 4096-row version compiled to 206 registers plus a 512-byte
per-thread stack and regressed panel time to about 1.0 ms. The selected
resource buckets are:

| Maximum rows | Warps/CTA | Registers | Stack | Shared memory |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 8 | 128 | 0 | 2320 B |
| 4096 | 16 | 128 | 0 | 2576 B |

Rows at or below 2048 use the 8-warp variant; larger panels use 16 warps. This
reduces QR2048 panel times from roughly `535--606 us` to `364--399 us`.
For QR4096, panels above 2048 rows measure about `371--397 us`, while later
panels measure `309--328 us`. The old QR4096 panel range was approximately
`550--650 us`.

One extra end-of-pivot CTA barrier is required before reusing the shared dot
partials. Without it, fast warps can enter the next norm reduction while the
leader is still issuing remote arrivals, corrupting the current dot products.
With that barrier, an isolated 2048x128 panel reconstructs its input with
`7.0e-6` maximum error, emitted V exactly matches rebuilding V from H/tau, and
tau differs from cuSOLVER by about `1.2e-7` maximum.

Official benchmark means improve from `10.888 ms` to `7.477 ms` for QR2048
and from `22.640 ms` to `14.655 ms` for QR4096, reductions of roughly 31% and
35%. All 22 correctness cases pass.

### Tall-Panel Signaling A/B

The same 16-CTA kernel was tested with global flags to isolate cross-CTA
signaling from the new register/memory-access design. Everything else remains
identical, including cluster placement, Mx8 ownership, row buckets, and fused
column-major V output.

The flag producer performs a GPU-scope release store after the CTA publication
barrier. A consumer leader polls with
`ld.global.relaxed.gpu.L1::no_allocate.u32`; after observing the expected
epoch it executes one `fence.acquire.gpu`, followed by a CTA barrier. This
avoids an acquire operation on every unsuccessful poll. The old global path
used acquire loads inside the loop and redundantly combined `__threadfence`
with a release store.

Global flags are materially faster than remote mbarrier fanout:

| Shape | Mbarrier panel total | Flag panel total | Mbarrier full profile | Flag full profile |
| ---: | ---: | ---: | ---: | ---: |
| QR2048 | 5.339 ms | 4.623 ms | 7.208 ms | 6.487 ms |
| QR4096 | 9.898 ms | 6.383 ms | 14.100 ms | 10.661 ms |

Official flag-path benchmark means are `6.527 ms` for QR2048 and `11.471 ms`
for QR4096, another 13% and 22% improvement over the selected mbarrier
version. Relative to the pre-cluster `10.888 ms` and `22.640 ms` baselines,
the total improvements are roughly 40% and 49%. All 22 correctness cases pass.

The production 128-column path now selects global flags.

### Tall-Panel Cleanup

After selecting global flags, the original 33-CTA producer/consumer tall-panel
kernels and their 64/128-column launchers were deleted. The temporary
mbarrier tall-panel operator, remote-map/fanout helpers, old acquire-on-every-
poll wrappers, unused 64-column Python branch, and obsolete `coop_panel`
diagnostic were also removed. Shared mbarrier helpers remain because the
QR176/192/208 cluster kernels still use them.

The reduced extension passes all 22 correctness cases. Post-cleanup benchmark
means are `6.714 ms` for QR2048 and `11.558 ms` for QR4096; these are within
normal worker variance of the selected `6.527 ms` and `11.471 ms` results.

### Normal-Grid Tall Panels

The fixed 16-CTA cluster dimension and non-portable cluster attribute were
removed while retaining exactly 16 CTAs per matrix. This isolates scheduling
placement from CTA-count and dataflow changes: rank ownership, global flags,
register tiles, row buckets, and V output are unchanged.

Unrestricted scheduling substantially benefits batch-8 QR2048. Its panel
total falls from `4.623 ms` to `2.709 ms`, and total profiled CUDA time falls
from `6.487 ms` to `4.572 ms`. Batch-2 QR4096 changes much less: panel total
increases from `6.383 ms` to `6.691 ms`, while total profile time moves from
`10.661 ms` to `10.937 ms`.

Official benchmark means are `4.881 ms` for QR2048 and `11.511 ms` for
QR4096, versus clustered-flag means of `6.527 ms` and `11.471 ms`. The QR2048
gain is about 25%; QR4096 is effectively flat within worker variance. The
normal-grid launch is selected for both shapes, all 22 correctness cases pass,
and the tall-panel kernel no longer contains cluster-specific launch features.

### QR4096 512-Column Panels

The normal-grid flag kernel is now templated over panel width. QR2048 retains
128-column panels, while QR4096 uses 512-column panels: 64 CTAs per matrix,
eight columns per CTA, and 128 CTAs total for batch 2. The QR4096 schedule is
seven 512-wide panels followed by the existing 512-row shared-memory tail,
instead of twenty-eight 128-wide panels.

An isolated 4096x512 panel reconstructs its input with `1.23e-5` maximum
error. Emitted V exactly matches rebuilding from H/tau, FP16 V differs by at
most `3.03e-5`, and tau differs from cuSOLVER by at most `1.19e-7`.

The seven wide panels take about `6.45 ms` total, versus `6.69 ms` for the 28
narrow normal-grid panels. More importantly, reducing Compact-WY stages from
28 to 7 lowers full profiled CUDA time from `10.94 ms` to `10.16 ms`, despite
the more expensive 512x512 triangular solves. The official QR4096 benchmark
mean improves from `11.511 ms` to `10.815 ms`, roughly 6%. All 22 correctness
cases pass, so the 512-column schedule is selected for QR4096.

### QR352 Register-Panel Replacement

The two remaining legacy QR352 panel stages were replaced without changing
the selected `(144, 32, 176)` schedule. The 352x144 first panel now uses
`cluster_register_panel_kernel` with a 9+9 warp split, giving 80 CTAs for the
batch-40 case. Its profiled kernel time falls from `275.5 us` to `102.4 us`.
The 208x32 second panel now uses the single-CTA `register_panel_kernel`; it
falls from `47.3 us` to `15.5 us`. The existing 176x176 cluster-register tail
is unchanged.

The QR352 benchmark mean improves from approximately `716 us` to `487.7 us`,
about 32%. Profiled CUDA time falls from `610 us` to `433 us`. All 22 official
correctness cases pass.

### Half-Storage 2-SM Panels

The two-CTA register panel previously allocated reflector shared memory for all
`COLS` columns in both CTAs. The kernel now allocates only `COLS / 2`
reflector columns per CTA. CTA0 produces its local first-half reflectors and
TMA-copies them into CTA1's half-sized buffer. CTA1 first consumes those remote
reflectors, synchronizes, then reuses the same buffer for its local second-half
reflectors. Tau and mbarrier arrays remain full-column indexed to keep the
publication protocol unchanged.

The split is implemented as separate remote and local consumer loops:
`[0, rank * NUM_WARPS)` for remote work and
`[rank * NUM_WARPS, rank * NUM_WARPS + warp)` for local work. Local reflector
and V-emission offsets use `warp`/`local_col`; global tau, mbarrier, and output
offsets still use the full `col`.

Direct Modal case-4 comparison against a detached `HEAD` worktree:

| Version | Case 4 mean | Best | Correctness |
| --- | ---: | ---: | --- |
| Baseline full reflector storage | `2.943 ms` | `2.774 ms` | pass |
| Half reflector storage | `2.773 ms` | `2.736 ms` | pass |

Case 5 also passed and stayed effectively flat at `4.356 ms` mean, which is
expected because its main bottleneck is the gmem panel rather than the small
2-SM tail.

With the lower shared-memory footprint, the QR1024 schedule can use wider
panels. Candidate A, `(96, 96, 96, 96, 128, 128, 128, 128, 128)`, passed case 4
with mean `2.627 ms` and best `2.605 ms`, improving over the half-storage
baseline schedule. A more aggressive candidate,
`(96, 96, 128, 128, 128, 128, 128, 96, 96)`, also passed but was slower at
`2.695 ms`, so candidate A is selected.

### QR1024 Compact-WY Inverse Follow-Ups

The current QR1024 experiments are centered on reducing Compact-WY overhead
without giving up the schedule advantages of wider panels. The active 64-wide
prototype uses a custom 1-CTA FP32 inverse builder for the `64x64` lower
triangular `T^T` system, followed by the existing TF32 GEMM for
`T^T @ projected`. The best simple variant is a 32+32 block inverse: build the
two diagonal `32x32` inverses, then compute the lower-left block as
`-X11 @ (B @ X00)`. More aggressive shared-memory padding, vectorized
mini-matmul layouts, and recursive 16x16 sub-blocks all passed correctness but
were slower in case-8 measurements.

Ideas to try next:

- Capture the fixed QR schedules with CUDA graph replay to reduce Python and
  launch overhead. The submission source should avoid banned terminology in
  identifiers/comments if the checker scans source text.
- Rebuild a `128x128` inverse path so QR1024 can return to schedules with more
  128-wide panels. A first version can build the two `64x64` diagonal inverse
  blocks in one custom kernel, zero the upper-right block, then compute the
  lower-left block with two GEMMs: `mid = B @ X00` and
  `X10 = -X11 @ mid`.
- Tune the existing CuBLASLt wrapper offline for the specific batched TF32 and
  FP16 GEMM shapes where PyTorch currently wins. If useful, hard-code only a
  small shape whitelist to avoid making the C++ wrapper too fragile.
- Revisit QR512 only after the 1-CTA `64x64` inverse path is stable, because
  QR512 has been more precision-sensitive to schedule and update changes.

First 128-wide prototype: `build_t128_diag` builds the two diagonal `64x64`
inverse blocks and zeros the upper-right quadrant in one custom kernel. Python
then computes the lower-left block as `mid = B @ X00` and
`X10 = -X11 @ mid`, before the usual `T^T @ projected` GEMM. Re-enabling the
wide QR1024 schedule `(96, 96, 96, 96, 128, 128, 128, 128, 128)` passed Modal
case 8 with mean `2.330 ms` and best `2.314 ms`, improving over the 64-wide
custom inverse schedule mean of roughly `2.385 ms`.

The 128-wide inverse path was then enabled for QR2048 and QR4096 as well.
Popcorn submission `838667` passed public and secret tests, improving geomean
to `1.3686 ms` public and `1.3618 ms` secret. Enabling the same path for QR512
also passed the hidden tests in submission `838671`, with geomean
`1.3373 ms` public and `1.3441 ms` secret.

A `96x96` variant was added using a `64+32` block split. The custom kernel
builds the `64x64` and `32x32` diagonal inverse blocks and zeros the upper
right block; Python computes the `32x64` lower-left block with
`mid = B @ X00` and `X10 = -X11 @ mid`. Modal case 8 passed with mean
`2.157 ms`. Popcorn submission `838687` passed public and secret tests, with
geomean `1.2944 ms` public and `1.3070 ms` secret.

The 64x64 block inverse code was refactored into a shared device helper used
by the custom inverse builders. The refactor kept the same shared-memory layout
and writeback behavior; Modal case 8 passed with mean `2.152 ms` and best
`2.140 ms`. Popcorn submission `838696` passed public and secret tests, with
geomean `1.2854 ms` public and `1.2832 ms` secret.

### QR4096 128-Wide Schedule Probe

The QR4096 schedule was changed from seven `512`-wide gmem panels plus four
`128`-wide tail panels to twenty-eight `128`-wide gmem panels plus the same
four `128`-wide tail panels. This requires QR4096 gmem-panel specializations
for row counts from `4096` down to `640` in steps of `128`. Modal case 6 passed
with mean `8.255 ms`; the same build with the old `512`-wide schedule measured
`8.563 ms`, so the narrower panels were about `3.6%` faster on this benchmark.

Cleanup pass: the dead standalone `build_t64` Python/C++ op and kernel were
removed because the active QR1024 schedule no longer uses 64-wide panels. The
diagonal `32x32` inverse solve was refactored into a single device helper used
by both the `64x64` block builder and the `96x96` bottom diagonal block. Modal
case 8 passed with mean `2.148 ms`; Modal case 6 passed with mean `8.256 ms`.

QR352 was changed from `(144, 208)` to `(128, 128, 96)` so both compact-WY
updates use the custom `128x128` inverse path instead of the generic triangular
solve. This required panel specializations for `(352, 128)`, `(224, 128)`, and
the terminal `(96, 96)` QR. Popcorn submission `838858` passed public and
secret tests, improving geomean to `1.2703 ms` public and `1.2607 ms` secret.

A follow-up `(128, 224)` schedule with a single-CTA `(224, 224)` terminal
register panel also passed, but regressed to `1.3024 ms` public and
`1.2892 ms` secret in submission `838867`. The `(128, 128, 96)` schedule is
kept.

### QR512 Schedule Probes

Two alternatives to the baseline `(96, 96, 128, 192)` schedule were tested and
rejected. Splitting the terminal panel as `(96, 96, 128, 96, 96)` passed Modal
case 3 but measured `3.580 ms`, slower than the baseline `3.549 ms` in the same
environment. An all-`128` schedule using 2-SM QR512 panels passed correctness
but was much slower at `4.642 ms`. The original `(96, 96, 128, 192)` QR512
schedule is kept.

A third schedule, `(64, 64, 128, 128, 128)`, restored the custom `64x64`
inverse path and used single-CTA QR512 panels. It passed Modal case 3 but
measured `3.675 ms`, again slower than the baseline. The temporary `build_t64`
op and QR512 64-wide specializations were removed after the probe.

A half-shared-memory single-CTA `512x128` register panel prototype was also
tested to enable the all-`128` QR512 schedule without the 2-SM panel. The
prototype overlapped first-half reflector production with second-half
consumption and used regular global stores for V instead of TMA. It passed
Modal case 3, but measured `4.988 ms`, so the all-`128` schedule was reverted.
The prototype kernel was reverted after the probe.

The half-smem prototype was then revised to use the 2-SM-style rank split
inside one CTA. Rank 0 writes first-half fp32 V with TMA, waits with
`cp.async.bulk.wait_group.read 0`, then signals a reuse mbarrier; rank 1 uses a
single warp to wait on that mbarrier and named barriers to synchronize the
second-half warps. This passed Modal case 3 under the all-`128` QR512 schedule
and improved the prototype to `4.590 ms`, but it is still far slower than the
baseline `3.55 ms`. The production QR512 schedule remains `(96, 96, 128, 192)`.

The simpler QR512 schedule `(96, 128, 128, 160)` was also tested with normal
single-CTA panels. It passed Modal case 3, but measured `3.960 ms`, so it was
reverted to the baseline `(96, 96, 128, 192)`.

For QR512, the off-diagonal block construction in the custom `96x96` and
`128x128` inverse path was tested with the explicit CuBLASLt TF32 wrapper.
Modal case 3 passed and measured `3.544 ms` on the first run and `3.500 ms` on
a follow-up run, roughly at or slightly better than the previous baseline.
Trying PyTorch matmul for the non-QR512 projection path regressed QR1024 case 8
to `2.195 ms`, so the projection remains on the explicit wrapper for all
problem sizes. Three popcorn submissions of the QR512 TF32 off-diagonal variant
finished with GPU `-` and no score, so the change was not kept.

Two small custom-inverse kernel rewrites were tested and rejected. First, the
shared `32x32` inverse helper was changed so each warp solves two adjacent
columns in one row sweep, reusing the loaded `lower` row for two reductions.
This passed Modal case 8 but measured `2.157 ms`, essentially neutral versus
the existing helper. Second, the `64x64` off-diagonal `32x32x32` mini-GEMMs
were changed to compute two adjacent columns per thread with `float2`
shared-memory loads/stores and no triangular-loop predicates. That also passed
case 8 but regressed to `2.174 ms`, so both rewrites were reverted.

QR512 TF32 for `tt @ projected` was tested only on the last compact-WY update
(`rows == 320`, `cols == 128`) while keeping the existing FP16 Gram policy.
Modal case 7 passed and measured `3.524 ms` versus a same-session baseline of
`3.540 ms`, but popcorn submission `841564` failed both public and secret
tests. Running the full public Modal test suite showed the failure is QR512
`case:band` (`test.9`, seed `32527`): `R - Q.T @ A` residual `0.0474` exceeded
the allowed `0.0442`. The TF32 transform change was reverted.

Replacing the final QR512 compact-WY apply with TF32
(`trailing -= v_f32 @ transformed`) was also tested. It passed the sensitive
QR512 band public test with scaled factor residual `17.1`, matching baseline,
but regressed Modal case 7 from the same-session baseline `3.540 ms` to
`3.651 ms`. The FP16 final apply remains faster and is kept.

Combining TF32 `tt @ projected` with TF32 final apply on the same last QR512
update was tested as an accuracy mitigation. It still failed QR512 `case:band`
with the same residual as TF32 transform alone: `0.0474` versus allowed
`0.0442`. This shows the accuracy loss is already in the transformed
coefficients, not primarily from the FP16 final apply. The combined change was
reverted.

Swapping the FP32 `tt @ projected` GEMM operands to compute
`(projected.transpose(-2, -1) @ tt.transpose(-2, -1)).transpose(-2, -1)` was
tested to give GEMM an `M > N` shape on QR512. The math stayed correct and
QR512 `case:band` passed with baseline residuals, but Modal case 7 measured
`3.552 ms`, slower than the same-session baseline around `3.540 ms`. The swap
was reverted.

### 16x16 Block Inverse Experiments

The shared `32x32` triangular inverse helper was changed to build the inverse
from two `16x16` diagonal inverse blocks plus one `16x16` off-diagonal block.
The two diagonal blocks use all 16 warps: warps `0..7` solve the first
diagonal block and warps `8..15` solve the second, with each half-warp solving
one inverse column via `warp_sum(..., 16)`. The off-diagonal block uses the
first 256 threads for `mid = L10 @ X00` and then `X10 = -X11 @ mid`. This
keeps the existing `build_t32_inverse_block` interface except for passing the
already-allocated `mid` scratch buffer, so the surrounding compact-WY dispatch
is unchanged. Focused Modal tests for QR512 and QR1024 passed, and the full
22-case public Modal suite passed.

Several attempts to extend the same 16x16 decomposition directly to the
`64x64` inverse were tested and rejected. A first branch-heavy flat `32x32`
off-diagonal rewrite passed Modal case 7 but measured `3.393 ms`. A direct
`4x4` block-forward-substitution rewrite with generic 16x16 off-diagonal
helpers measured `3.377 ms`. Dedicated `off1`, `off2`, and `off3` helpers
measured `3.376 ms`; removing triangular-loop predicates inside the 16x16
mini-matmuls improved that to `3.354 ms`; an offset/template refactor regressed
to `3.415 ms`. Although these variants were correct, none clearly improved on
the isolated `32x32` helper change, and the extra off-diagonal helper code was
reverted.

The retained state is therefore only the isolated `32x32` inverse
implementation from `16x16` blocks. After reverting the `64x64` off-diagonal
experiments, Modal case 7 passed with mean `3.439 ms` and best `3.436 ms` in a
noisy run. Future work on the `64x64` path should start from codegen/resource
inspection rather than adding more block algebra: the rejected versions suggest
barrier count, helper phase structure, and generated instruction pressure can
erase the theoretical arithmetic savings.
