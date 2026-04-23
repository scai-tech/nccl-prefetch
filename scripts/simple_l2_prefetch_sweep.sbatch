#!/bin/bash
# Submit from the repository root:
#   sbatch scripts/simple_l2_prefetch_sweep.sbatch
#
# This script follows the same H100 cluster conventions as the reference
# benchmark sbatch the user provided.

#SBATCH -J nccl-simple-l2-prefetch
#SBATCH -A gts-dmahajan7-paid
#SBATCH -q inferno
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:H100:4
#SBATCH --mem-per-gpu=32G
#SBATCH -t 8:00:00
#SBATCH -o slurm-%x-%j.out

set -eo pipefail

cd "${SLURM_SUBMIT_DIR}"

echo "job_id=${SLURM_JOB_ID}"
echo "node=$(hostname)"
echo "submit_dir=${SLURM_SUBMIT_DIR}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"

set +u
if [[ -f "${HOME}/.bashrc" ]]; then
  source "${HOME}/.bashrc"
fi
if command -v conda >/dev/null 2>&1; then
  CONDA_ENV_NAME="${CONDA_ENV_NAME:-ccoverlap}"
  conda activate "${CONDA_ENV_NAME}" || {
    echo "Failed to activate conda env: ${CONDA_ENV_NAME}" >&2
    exit 1
  }
fi
set -u

nvidia-smi

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR}}"
EXPERIMENT_REPO="${EXPERIMENT_REPO:-$REPO_ROOT}"
VANILLA_REPO="${VANILLA_REPO:-$(cd "$REPO_ROOT/.." && pwd)/nccl}"
RESULT_ROOT="${RESULT_ROOT:-$REPO_ROOT/results/simple_l2_prefetch/${SLURM_JOB_ID:-manual}}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
JOBS="${JOBS:-${SLURM_CPUS_ON_NODE:-8}}"
MAKE_JOBS="${MAKE_JOBS:-$JOBS}"
NVCC_GENCODE="${NVCC_GENCODE:--gencode=arch=compute_90,code=sm_90}"

NCCL_TESTS_HOME="${NCCL_TESTS_HOME:-$REPO_ROOT/../nccl-tests}"
NCCL_TEST_BIN="${NCCL_TEST_BIN:-$NCCL_TESTS_HOME/build/all_reduce_perf}"
NCCL_TEST_BUILD_REPO="${NCCL_TEST_BUILD_REPO:-$EXPERIMENT_REPO}"
if [[ -n "${SLURM_GPUS_ON_NODE:-}" && "${SLURM_GPUS_ON_NODE}" =~ ^[0-9]+$ ]]; then
  DEFAULT_TEST_GPUS="${SLURM_GPUS_ON_NODE}"
else
  DEFAULT_TEST_GPUS="4"
fi
NCCL_TEST_GPUS="${NCCL_TEST_GPUS:-$DEFAULT_TEST_GPUS}"

TEST_B="${TEST_B:-8}"
TEST_E="${TEST_E:-1G}"
TEST_F="${TEST_F:-2}"
TEST_WARMUP_ITERS="${TEST_WARMUP_ITERS:-100}"
TEST_ITERS="${TEST_ITERS:-3000}"
NCCL_TEST_ARGS="${NCCL_TEST_ARGS:-}"

NCCL_PROTO_FORCE="${NCCL_PROTO_FORCE:-Simple}"
NCCL_ALGO_FORCE="${NCCL_ALGO_FORCE:-Ring}"
SRUN_PREFIX="${SRUN_PREFIX:-srun --ntasks=1}"
BUILD_ONLY="${BUILD_ONLY:-0}"

mkdir -p "$RESULT_ROOT"

echo "REPO_ROOT=$REPO_ROOT"
echo "EXPERIMENT_REPO=$EXPERIMENT_REPO"
echo "VANILLA_REPO=$VANILLA_REPO"
echo "RESULT_ROOT=$RESULT_ROOT"
echo "CUDA_HOME=$CUDA_HOME"
echo "JOBS=$JOBS"
echo "MAKE_JOBS=$MAKE_JOBS"
echo "NVCC_GENCODE=$NVCC_GENCODE"
echo "NCCL_TESTS_HOME=$NCCL_TESTS_HOME"
echo "NCCL_TEST_BIN=$NCCL_TEST_BIN"
echo "NCCL_TEST_BUILD_REPO=$NCCL_TEST_BUILD_REPO"
echo "NCCL_TEST_GPUS=$NCCL_TEST_GPUS"
echo "TEST_B=$TEST_B TEST_E=$TEST_E TEST_F=$TEST_F"
echo "TEST_WARMUP_ITERS=$TEST_WARMUP_ITERS TEST_ITERS=$TEST_ITERS"
echo "NCCL_PROTO_FORCE=$NCCL_PROTO_FORCE NCCL_ALGO_FORCE=$NCCL_ALGO_FORCE"
echo "NCCL_TEST_ARGS=$NCCL_TEST_ARGS"
echo "SRUN_PREFIX=$SRUN_PREFIX"
echo "BUILD_ONLY=$BUILD_ONLY"

require_repo() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "$label repo not found: $path" >&2
    exit 1
  fi
  if [[ ! -f "$path/Makefile" || ! -d "$path/src" ]]; then
    echo "$label repo does not look like an NCCL tree: $path" >&2
    exit 1
  fi
}

require_command() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Required command not found: $cmd" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

verify_nccl_build() {
  local repo="$1"
  local label="$2"
  if [[ -f "$repo/build/lib/libnccl.so" ]]; then
    return
  fi
  if [[ -f "$repo/build/lib/libnccl_static.a" ]]; then
    return
  fi
  echo "$label build completed but no NCCL library was found under $repo/build/lib" >&2
  exit 1
}

build_nccl() {
  local repo="$1"
  local label="$2"
  echo "[build] $label: $repo"
  make -C "$repo" -j"$MAKE_JOBS" src.build CUDA_HOME="$CUDA_HOME" NVCC_GENCODE="$NVCC_GENCODE"
  verify_nccl_build "$repo" "$label"
}

build_nccl_tests_if_needed() {
  if [[ -x "$NCCL_TEST_BIN" ]]; then
    return
  fi
  if [[ ! -d "$NCCL_TESTS_HOME" ]]; then
    echo "nccl-tests tree not found: $NCCL_TESTS_HOME" >&2
    echo "Set NCCL_TESTS_HOME or NCCL_TEST_BIN before submitting the job." >&2
    exit 1
  fi
  echo "[build] nccl-tests: $NCCL_TESTS_HOME"
  make -C "$NCCL_TESTS_HOME" -j"$MAKE_JOBS" CUDA_HOME="$CUDA_HOME" NCCL_HOME="$NCCL_TEST_BUILD_REPO" MPI=0 NVCC_GENCODE="$NVCC_GENCODE"
  if [[ ! -x "$NCCL_TEST_BIN" ]]; then
    echo "all_reduce_perf not found after build: $NCCL_TEST_BIN" >&2
    exit 1
  fi
}

run_case() {
  local repo="$1"
  local case_name="$2"
  local enable="$3"
  local max_bytes="$4"
  local ahead_chunks="${5:-1}"
  local log_file="$RESULT_ROOT/${case_name}.log"

  echo
  echo "[run] $case_name"
  echo "  repo=$repo"
  echo "  NCCL_SIMPLE_L2_PREFETCH=$enable"
  echo "  NCCL_SIMPLE_L2_PREFETCH_MAX_BYTES=$max_bytes"
  echo "  NCCL_SIMPLE_L2_PREFETCH_AHEAD_CHUNKS=$ahead_chunks"
  echo "  log=$log_file"

  (
    set -x
    export LD_LIBRARY_PATH="$repo/build/lib:${LD_LIBRARY_PATH:-}"
    export NCCL_PROTO="$NCCL_PROTO_FORCE"
    export NCCL_ALGO="$NCCL_ALGO_FORCE"
    export NCCL_SIMPLE_L2_PREFETCH="$enable"
    export NCCL_SIMPLE_L2_PREFETCH_MAX_BYTES="$max_bytes"
    export NCCL_SIMPLE_L2_PREFETCH_AHEAD_CHUNKS="$ahead_chunks"
    ${SRUN_PREFIX} "$NCCL_TEST_BIN" \
      -b "$TEST_B" \
      -e "$TEST_E" \
      -f "$TEST_F" \
      -g "$NCCL_TEST_GPUS" \
      -w "$TEST_WARMUP_ITERS" \
      -n "$TEST_ITERS" \
      $NCCL_TEST_ARGS
  ) 2>&1 | tee "$log_file"
}

extract_busbw_table() {
  local log_file="$1"
  local out_file="$2"
  awk '
    /^[[:space:]]*[0-9]+[[:space:]]+/ {
      if (NF >= 12) {
        print $1 "\t" $8 "\t" $12
      }
    }
  ' "$log_file" > "$out_file"
}

generate_comparison_tables() {
  local vanilla_tsv="$RESULT_ROOT/vanilla_baseline.tsv"
  local exp_off_tsv="$RESULT_ROOT/experiment_off.tsv"
  local a1_128_tsv="$RESULT_ROOT/experiment_prefetch_a1_128k.tsv"
  local a1_256_tsv="$RESULT_ROOT/experiment_prefetch_a1_256k.tsv"
  local a1_512_tsv="$RESULT_ROOT/experiment_prefetch_a1_512k.tsv"
  local a2_128_tsv="$RESULT_ROOT/experiment_prefetch_a2_128k.tsv"
  local a2_256_tsv="$RESULT_ROOT/experiment_prefetch_a2_256k.tsv"
  local a2_512_tsv="$RESULT_ROOT/experiment_prefetch_a2_512k.tsv"
  local summary_tsv="$RESULT_ROOT/comparison_oop_busbw.tsv"
  local summary_md="$RESULT_ROOT/comparison_oop_busbw.md"

  extract_busbw_table "$RESULT_ROOT/vanilla_baseline.log" "$vanilla_tsv"
  extract_busbw_table "$RESULT_ROOT/experiment_off.log" "$exp_off_tsv"
  extract_busbw_table "$RESULT_ROOT/experiment_prefetch_a1_128k.log" "$a1_128_tsv"
  extract_busbw_table "$RESULT_ROOT/experiment_prefetch_a1_256k.log" "$a1_256_tsv"
  extract_busbw_table "$RESULT_ROOT/experiment_prefetch_a1_512k.log" "$a1_512_tsv"
  extract_busbw_table "$RESULT_ROOT/experiment_prefetch_a2_128k.log" "$a2_128_tsv"
  extract_busbw_table "$RESULT_ROOT/experiment_prefetch_a2_256k.log" "$a2_256_tsv"
  extract_busbw_table "$RESULT_ROOT/experiment_prefetch_a2_512k.log" "$a2_512_tsv"

  awk '
    function pct(newv, oldv) {
      if (oldv == 0 || oldv == "") return "nan"
      return sprintf("%.2f", 100.0*(newv-oldv)/oldv)
    }
    FILENAME==ARGV[1] {van[$1]=$2; sizes[$1]=1; next}
    FILENAME==ARGV[2] {off[$1]=$2; sizes[$1]=1; next}
    FILENAME==ARGV[3] {a1_128[$1]=$2; sizes[$1]=1; next}
    FILENAME==ARGV[4] {a1_256[$1]=$2; sizes[$1]=1; next}
    FILENAME==ARGV[5] {a1_512[$1]=$2; sizes[$1]=1; next}
    FILENAME==ARGV[6] {a2_128[$1]=$2; sizes[$1]=1; next}
    FILENAME==ARGV[7] {a2_256[$1]=$2; sizes[$1]=1; next}
    FILENAME==ARGV[8] {a2_512[$1]=$2; sizes[$1]=1; next}
    END {
      print "size_bytes\tvanilla_oop_busbw_gbps\texperiment_off_oop_busbw_gbps\tprefetch_a1_128k_oop_busbw_gbps\tprefetch_a1_256k_oop_busbw_gbps\tprefetch_a1_512k_oop_busbw_gbps\tprefetch_a2_128k_oop_busbw_gbps\tprefetch_a2_256k_oop_busbw_gbps\tprefetch_a2_512k_oop_busbw_gbps\toff_vs_vanilla_pct\tprefetch_a1_128k_vs_vanilla_pct\tprefetch_a1_256k_vs_vanilla_pct\tprefetch_a1_512k_vs_vanilla_pct\tprefetch_a2_128k_vs_vanilla_pct\tprefetch_a2_256k_vs_vanilla_pct\tprefetch_a2_512k_vs_vanilla_pct"
      for (s in sizes) {
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", \
          s, van[s], off[s], a1_128[s], a1_256[s], a1_512[s], a2_128[s], a2_256[s], a2_512[s], \
          pct(off[s], van[s]), pct(a1_128[s], van[s]), pct(a1_256[s], van[s]), pct(a1_512[s], van[s]), pct(a2_128[s], van[s]), pct(a2_256[s], van[s]), pct(a2_512[s], van[s])
      }
    }
  ' "$vanilla_tsv" "$exp_off_tsv" "$a1_128_tsv" "$a1_256_tsv" "$a1_512_tsv" "$a2_128_tsv" "$a2_256_tsv" "$a2_512_tsv" > "$summary_tsv.tmp"
  {
    head -n 1 "$summary_tsv.tmp"
    tail -n +2 "$summary_tsv.tmp" | sort -n
  } > "$summary_tsv"
  rm -f "$summary_tsv.tmp"

  {
    echo "# SIMPLE L2 Prefetch Comparison (Out-of-Place BusBW)"
    echo
    echo "- vanilla baseline log: \`vanilla_baseline.log\`"
    echo "- experiment off log: \`experiment_off.log\`"
    echo "- experiment on logs: \`experiment_prefetch_a1_*.log\`, \`experiment_prefetch_a2_*.log\`"
    echo "- table source: \`comparison_oop_busbw.tsv\`"
    echo
    echo "| size_bytes | vanilla | exp_off | a1_128 | a1_256 | a1_512 | a2_128 | a2_256 | a2_512 | off_% | a1_128_% | a1_256_% | a1_512_% | a2_128_% | a2_256_% | a2_512_% |"
    echo "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    awk -F'\t' 'NR>1 {printf "| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |\n", $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16}' "$summary_tsv"
  } > "$summary_md"
}

require_repo "$EXPERIMENT_REPO" "Experiment"
require_repo "$VANILLA_REPO" "Vanilla"
require_command make
require_command awk
require_command nvidia-smi
require_command srun

if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
  echo "nvcc=$CUDA_HOME/bin/nvcc"
elif command -v nvcc >/dev/null 2>&1; then
  echo "nvcc=$(command -v nvcc)"
else
  echo "nvcc not found in CUDA_HOME/bin or PATH" >&2
  exit 1
fi

if [[ "$(cd "$EXPERIMENT_REPO" && pwd)" == "$(cd "$VANILLA_REPO" && pwd)" ]]; then
  echo "VANILLA_REPO and EXPERIMENT_REPO must point to different trees." >&2
  exit 1
fi

build_nccl "$VANILLA_REPO" "vanilla"
build_nccl "$EXPERIMENT_REPO" "experiment"
build_nccl_tests_if_needed
require_file "$NCCL_TEST_BIN" "all_reduce_perf"

if [[ "$BUILD_ONLY" == "1" ]]; then
  echo
  echo "Build-only mode complete."
  echo "vanilla and experiment NCCL builds succeeded."
  echo "nccl-tests binary: $NCCL_TEST_BIN"
  exit 0
fi

run_case "$VANILLA_REPO" vanilla_baseline 0 0 1
run_case "$EXPERIMENT_REPO" experiment_off 0 0 1
run_case "$EXPERIMENT_REPO" experiment_prefetch_a1_128k 1 131072 1
run_case "$EXPERIMENT_REPO" experiment_prefetch_a1_256k 1 262144 1
run_case "$EXPERIMENT_REPO" experiment_prefetch_a1_512k 1 524288 1
run_case "$EXPERIMENT_REPO" experiment_prefetch_a2_128k 1 131072 2
run_case "$EXPERIMENT_REPO" experiment_prefetch_a2_256k 1 262144 2
run_case "$EXPERIMENT_REPO" experiment_prefetch_a2_512k 1 524288 2

generate_comparison_tables

echo
echo "Comparison complete."
echo "Logs and summaries are in $RESULT_ROOT"
echo "Summary TSV: $RESULT_ROOT/comparison_oop_busbw.tsv"
echo "Summary MD : $RESULT_ROOT/comparison_oop_busbw.md"
