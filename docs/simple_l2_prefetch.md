# Hopper Simple Protocol L2 Prefetch

This branch adds a Hopper-only L2 prefetch experiment to the SIMPLE protocol path.
The goal is to test whether warming the tail of the current SIMPLE slice in L2 can
reduce global load miss penalty inside `reduceCopy` without introducing the
synchronization overheads of TMA/shared-memory staging.

## What Was Implemented

### Runtime knobs

Two runtime knobs were added to `ncclKernelComm` and are populated during host
device-comm setup:

- `simpleL2PrefetchEnable`
- `simpleL2PrefetchMaxBytes`

The corresponding environment variables are:

- `NCCL_SIMPLE_L2_PREFETCH`
  - `0`: disabled
  - nonzero: enabled
  - default: `0`
- `NCCL_SIMPLE_L2_PREFETCH_MAX_BYTES`
  - maximum bytes prefetched from the current slice tail
  - default: `512*1024`
  - negative values are clamped to `0`
  - value is aligned down to `16B` before being copied to device state

### Device-side behavior

The prefetch is issued in the SIMPLE worker loop in `genericOp()` after:

1. `waitPeer(...)`
2. `subBarrier()`
3. optional `ncclNetDeviceUnpack(...)`
4. optional unpack `subBarrier()`

and immediately before `reduceCopy(...)`.

No completion wait, no extra barrier, and no correctness dependency is introduced.
The instruction is used strictly as a performance hint:

```ptx
cp.async.bulk.prefetch.L2.global [srcMem], size;
```

### Prefetch region

The implementation intentionally does **not** prefetch a future slice.
Instead, it only prefetches the tail of the **current** valid source slice:

- `workBytes = workSize * sizeof(T)`
- `prefetchBytes = min(workBytes / 2, simpleL2PrefetchMaxBytes)`
- slices below `64KB` are skipped
- prefetches below `64KB` are skipped
- start address is aligned up to `16B`
- end address is aligned down to `16B`
- size is therefore always a `16B` multiple

This choice is deliberate. Prefetching a future recv FIFO slice could touch lines
that are not yet producer-visible and create a stale/coherence risk. Current-slice
tail prefetch avoids that hazard because `waitPeer()` has already established that
the current slice is valid.

### Scope limitations in this first version

The optimization is deliberately conservative:

- enabled only for `sm90+`
- compiled/issued only when `CUDART_VERSION >= 12010`
- only `tid == 0` issues the prefetch
- only when `src != nullptr`
- only when the SIMPLE op has exactly one source (`nSrcs == 1`)
- no attempt was made to generalize to multi-source all-reduce paths yet

This means the implementation is targeted at the safe single-source subset first,
which is appropriate for a fast validation pass.

## Files Changed

- `src/include/device.h`
  - adds device runtime state
- `src/init.cc`
  - adds env vars and copies knob values into `ncclKernelComm`
- `src/device/prims_simple.h`
  - adds the Hopper PTX prefetch helper and inserts the call before `reduceCopy`

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
4. runs:
   - `vanilla_baseline`
   - `experiment_off`
   - `experiment_prefetch_128k`
   - `experiment_prefetch_256k`
   - `experiment_prefetch_512k`
5. writes logs and comparison summaries under `results/simple_l2_prefetch/<jobid>/`

If you already have an `all_reduce_perf` binary elsewhere, you can skip the
`nccl-tests` source-tree requirement by exporting `NCCL_TEST_BIN=/path/to/all_reduce_perf`
when submitting the job.

By default the script expects the untouched NCCL tree at:

- `../nccl`

If your vanilla tree lives elsewhere, pass:

- `VANILLA_REPO=/path/to/vanilla/nccl`

The experiment repo defaults to the repo that contains the script.

Default sweep points:

- baseline: `NCCL_SIMPLE_L2_PREFETCH=0`
- `NCCL_SIMPLE_L2_PREFETCH=1 NCCL_SIMPLE_L2_PREFETCH_MAX_BYTES=131072`
- `NCCL_SIMPLE_L2_PREFETCH=1 NCCL_SIMPLE_L2_PREFETCH_MAX_BYTES=262144`
- `NCCL_SIMPLE_L2_PREFETCH=1 NCCL_SIMPLE_L2_PREFETCH_MAX_BYTES=524288`

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

The script also writes:

- `comparison_oop_busbw.tsv`
- `comparison_oop_busbw.md`

These files compare out-of-place bus bandwidth by message size across vanilla,
experiment-off, and the three prefetch sweep points.

## Expected Measurement Signals

If the hint is effective, the likely signals are:

- lower global load miss penalty inside `reduceCopy`
- higher L2 hit rate
- lower DRAM read pressure
- lower slice time for the affected SIMPLE single-source sub-ops

The optimization does **not** reduce store traffic. It is a load-side latency
hiding experiment.
