# Hopper Simple Protocol L2 Prefetch

This branch adds a Hopper-only L2 prefetch experiment to the SIMPLE protocol path.
The goal is to test whether chunked intra-slice L2 prefetch can reduce global
load miss penalty inside `reduceCopy` without introducing the synchronization
overheads of TMA/shared-memory staging.

## What Was Implemented

### Runtime knobs

Five runtime knobs were added to `ncclKernelComm` and are populated during host
device-comm setup:

- `simpleL2PrefetchEnable`
- `simpleL2PrefetchMaxBytes`
- `simpleL2PrefetchChunkBytes`
- `simpleL2PrefetchAheadChunks`
- `simpleL2PrefetchMode`

The corresponding environment variables are:

- `NCCL_SIMPLE_L2_PREFETCH`
  - `0`: disabled
  - nonzero: enabled
  - default: `0`
- `NCCL_SIMPLE_L2_PREFETCH_MAX_BYTES`
  - maximum bytes of future local/user-buffer lookahead kept prefetched ahead of the current slice
  - default: `512*1024`
  - negative values are clamped to `0`
  - value is aligned down to `16B` before being copied to device state
- `NCCL_SIMPLE_L2_PREFETCH_CHUNK_BYTES`
  - current-slice subchunk size used by the prefetch/reduce pipeline
  - default: `128*1024`
  - negative values are clamped to `0`
  - value is aligned down to `16B` before being copied to device state
- `NCCL_SIMPLE_L2_PREFETCH_AHEAD_CHUNKS`
  - current-slice prefetch lookahead window
  - default: `1`
  - values below `1` are clamped to `1`
  - values above `4` are clamped to `4`
- `NCCL_SIMPLE_L2_PREFETCH_MODE`
  - `0`: prefetch both recv and local sources
  - `1`: prefetch recv sources only
  - `2`: prefetch local sources only
  - default: `0`
  - values are clamped into `[0, 2]`

### Device-side behavior

The prefetch is issued in the SIMPLE worker loop in `genericOp()` after:

1. `waitPeer(...)`
2. `subBarrier()`
3. optional `ncclNetDeviceUnpack(...)`
4. optional unpack `subBarrier()`

and then uses a chunked `reduceCopy(...)` loop inside the current slice.

No completion wait, no extra barrier, and no correctness dependency is introduced.
The instruction is used strictly as a performance hint:

```ptx
cp.async.bulk.prefetch.L2.global [srcMem], size;
```

### Prefetch region

The implementation now uses source-aware rules:

- for **recv/FIFO-backed sources**, it stays within the **current valid slice**
- for **local/user-buffer sources**, it also prefetches a **future contiguous
  window beyond the current slice boundary**

Inside the current slice, it splits the work into internal subchunks. With
`NCCL_SIMPLE_L2_PREFETCH_AHEAD_CHUNKS=N`, it first warms chunk `0`, primes
chunks `1..N`, and then, before processing subchunk `i`, issues L2 prefetches
for subchunk `i+N+1`. The prefetch issue itself is distributed across the first
worker warp instead of a single `tid == 0` issuer. In parallel, local sources
maintain a rolling future-slice prefetch window:

- `workBytes = workSize * sizeof(T)`
- `chunkBytes = simpleL2PrefetchChunkBytes`
- future local lookahead budget is capped by `simpleL2PrefetchMaxBytes`
- pipelining is skipped unless the current slice is large enough to form at
  least two `64KB` chunks
- each prefetched chunk start is aligned up to `16B`
- each prefetched chunk end is aligned down to `16B`
- each issued prefetch size is therefore a `16B` multiple

This split is deliberate. Prefetching a future recv FIFO slice could touch lines
that are not yet producer-visible and create a stale/coherence risk. Current-slice
recv prefetch avoids that hazard because `waitPeer()` has already established
that the current slice is valid. Local/user-buffer sources do not have that
producer-visibility problem, so they can be prefetched more aggressively.

### Scope limitations in this first version

The optimization is deliberately conservative:

- enabled only for `sm90+`
- compiled/issued only when `CUDART_VERSION >= 12010`
- only the first worker warp issues the prefetch
- only non-null source pointers are prefetched
- current-slice prefetch can be restricted to recv-only, local-only, or both
- future-slice prefetch is still limited to local/user-buffer sources

This keeps recv-side accesses within the safe current-slice validity window while
still creating a deeper inter-slice pipeline for local sources.

## Files Changed

- `src/include/device.h`
  - adds device runtime state
- `src/init.cc`
  - adds env vars and copies knob values into `ncclKernelComm`
- `src/device/prims_simple.h`
  - adds the Hopper PTX prefetch helper and replaces the single `reduceCopy`
    call with a chunked prefetch/reduce pipeline

## Verification Performed

Local checks completed in this workspace:

- `git diff --check`
  - passed
- source inspection of host/device control flow and insertion point
  - completed

Full local compile was attempted but could not be completed on this machine because
the environment does not provide CUDA headers (`cuda_runtime.h` was missing during
`make`). The implementation is therefore prepared for server-side build and run
verification on a CUDA-capable Hopper system.

## Server Validation Flow

Use the provided SLURM script:

- `scripts/simple_l2_prefetch_sweep.sbatch`

Its default SLURM header and environment bootstrap intentionally mirror the H100
benchmark script you referenced:

- account: `gts-dmahajan7-paid`
- queue: `inferno`
- allocation: `1 node`, `4x H100`, `32G mem / GPU`
- shell bootstrap: `source ~/.bashrc`
- conda env: `ccoverlap`
- code generation default: `-gencode=arch=compute_90,code=sm_90`
- benchmark launch: `srun --ntasks=1`

The script does the following:

1. builds a vanilla NCCL tree and this experiment tree
2. builds `nccl-tests` if needed
3. forces `NCCL_PROTO=Simple`
4. runs a full grid:
   - `vanilla_baseline`
   - `experiment_off`
   - `experiment_prefetch_mode_<mode>_chunk_<chunk>k_max_<max>k_ahead_<ahead>`
   - default mode sweep: `both,local,recv`
   - default chunk sweep: `64KB,128KB,256KB`
   - default future-window sweep: `128KB,256KB,512KB`
   - default lookahead sweep: `1,2,4`
5. writes logs and comparison summaries under `results/simple_l2_prefetch/<jobid>/`

If you already have an `all_reduce_perf` binary elsewhere, you can skip the
`nccl-tests` source-tree requirement by exporting `NCCL_TEST_BIN=/path/to/all_reduce_perf`
when submitting the job.

By default the script expects the untouched NCCL tree at:

- `../nccl`

If your vanilla tree lives elsewhere, pass:

- `VANILLA_REPO=/path/to/vanilla/nccl`

The experiment repo defaults to the repo that contains the script.

Default sweep controls:

- `PREFETCH_MODES_CSV=both,local,recv`
- `PREFETCH_CHUNK_BYTES_CSV=65536,131072,262144`
- `PREFETCH_MAX_BYTES_CSV=131072,262144,524288`
- `PREFETCH_AHEAD_CHUNKS_CSV=1,2,4`

### Example

```bash
git pull
sbatch scripts/simple_l2_prefetch_sweep.sbatch
```

If your cluster requires explicit partition/account selection:

```bash
sbatch -p <partition> -A <account> scripts/simple_l2_prefetch_sweep.sbatch
```

Useful overrides:

```bash
sbatch \
  --gpus-per-node=8 \
  --time=00:30:00 \
  --export=ALL,VANILLA_REPO=/path/to/vanilla/nccl,NCCL_TESTS_HOME=/path/to/nccl-tests,NCCL_TEST_GPUS=8,TEST_B=8,TEST_E=1G,TEST_F=2 \
  scripts/simple_l2_prefetch_sweep.sbatch
```

For a cheap compile-only validation before running the full sweep:

```bash
sbatch \
  -t 00:20:00 \
  --export=ALL,BUILD_ONLY=1,VANILLA_REPO=/path/to/vanilla/nccl,NCCL_TESTS_HOME=/path/to/nccl-tests \
  scripts/simple_l2_prefetch_sweep.sbatch
```

The script also writes:

- `comparison_oop_busbw.tsv`
- `comparison_oop_busbw.md`
- `simple_l2_prefetch_summary.xlsx`

These files compare out-of-place bus bandwidth by message size across vanilla,
experiment-off, and every prefetched sweep point in the generated grid.

The Excel workbook contains separate sheets for the run summary, ranked average
results, out-of-place bus bandwidth, out-of-place percent deltas versus vanilla,
out-of-place time, in-place bus bandwidth, best-by-size, and metadata. It is
generated with Python standard
library code only, so it does not require `pandas` or `openpyxl`.

## Expected Measurement Signals

If the hint is effective, the likely signals are:

- lower global load miss penalty inside `reduceCopy`
- higher L2 hit rate
- lower DRAM read pressure
- lower slice time for the affected SIMPLE sub-ops, especially multi-source
  reduce paths

The optimization does **not** reduce store traffic. It is a load-side latency
hiding experiment.
