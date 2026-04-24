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
PREFETCH_MODES_CSV="${PREFETCH_MODES_CSV:-both,local,recv}"
PREFETCH_CHUNK_BYTES_CSV="${PREFETCH_CHUNK_BYTES_CSV:-65536,131072,262144}"
PREFETCH_MAX_BYTES_CSV="${PREFETCH_MAX_BYTES_CSV:-131072,262144,524288}"
PREFETCH_AHEAD_CHUNKS_CSV="${PREFETCH_AHEAD_CHUNKS_CSV:-1,2,4}"

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
echo "PREFETCH_MODES_CSV=$PREFETCH_MODES_CSV"
echo "PREFETCH_CHUNK_BYTES_CSV=$PREFETCH_CHUNK_BYTES_CSV"
echo "PREFETCH_MAX_BYTES_CSV=$PREFETCH_MAX_BYTES_CSV"
echo "PREFETCH_AHEAD_CHUNKS_CSV=$PREFETCH_AHEAD_CHUNKS_CSV"
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
  local chunk_bytes="${6:-0}"
  local mode_value="${7:-0}"
  local log_file="$RESULT_ROOT/${case_name}.log"

  echo
  echo "[run] $case_name"
  echo "  repo=$repo"
  echo "  NCCL_SIMPLE_L2_PREFETCH=$enable"
  echo "  NCCL_SIMPLE_L2_PREFETCH_MAX_BYTES=$max_bytes"
  echo "  NCCL_SIMPLE_L2_PREFETCH_AHEAD_CHUNKS=$ahead_chunks"
  echo "  NCCL_SIMPLE_L2_PREFETCH_CHUNK_BYTES=$chunk_bytes"
  echo "  NCCL_SIMPLE_L2_PREFETCH_MODE=$mode_value"
  echo "  log=$log_file"

  (
    set -x
    export LD_LIBRARY_PATH="$repo/build/lib:${LD_LIBRARY_PATH:-}"
    export NCCL_PROTO="$NCCL_PROTO_FORCE"
    export NCCL_ALGO="$NCCL_ALGO_FORCE"
    export NCCL_SIMPLE_L2_PREFETCH="$enable"
    export NCCL_SIMPLE_L2_PREFETCH_MAX_BYTES="$max_bytes"
    export NCCL_SIMPLE_L2_PREFETCH_AHEAD_CHUNKS="$ahead_chunks"
    export NCCL_SIMPLE_L2_PREFETCH_CHUNK_BYTES="$chunk_bytes"
    export NCCL_SIMPLE_L2_PREFETCH_MODE="$mode_value"
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

split_csv() {
  local csv="$1"
  local -n out_ref="$2"
  IFS=',' read -r -a out_ref <<< "$csv"
}

mode_to_env() {
  case "$1" in
    both) echo 0 ;;
    recv) echo 1 ;;
    local) echo 2 ;;
    *)
      echo "Unsupported prefetch mode: $1" >&2
      exit 1
      ;;
  esac
}

generate_reports() {
  local xlsx_path="$RESULT_ROOT/simple_l2_prefetch_summary.xlsx"
  local tsv_path="$RESULT_ROOT/comparison_oop_busbw.tsv"
  local md_path="$RESULT_ROOT/comparison_oop_busbw.md"
  if command -v python3 >/dev/null 2>&1; then
    echo "[summary] writing reports under: $RESULT_ROOT"
    python3 "$REPO_ROOT/scripts/simple_l2_prefetch_make_xlsx.py" "$RESULT_ROOT" -o "$xlsx_path" --tsv "$tsv_path" --md "$md_path"
  else
    echo "python3 not found; skipping report generation." >&2
  fi
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

split_csv "$PREFETCH_MODES_CSV" PREFETCH_MODES
split_csv "$PREFETCH_CHUNK_BYTES_CSV" PREFETCH_CHUNK_BYTES
split_csv "$PREFETCH_MAX_BYTES_CSV" PREFETCH_MAX_BYTES
split_csv "$PREFETCH_AHEAD_CHUNKS_CSV" PREFETCH_AHEAD_CHUNKS

run_case "$VANILLA_REPO" vanilla_baseline 0 0 1 0 0
run_case "$EXPERIMENT_REPO" experiment_off 0 0 1 0 0

for mode_label in "${PREFETCH_MODES[@]}"; do
  mode_value="$(mode_to_env "$mode_label")"
  for chunk_bytes in "${PREFETCH_CHUNK_BYTES[@]}"; do
    for max_bytes in "${PREFETCH_MAX_BYTES[@]}"; do
      for ahead_chunks in "${PREFETCH_AHEAD_CHUNKS[@]}"; do
        case_name="experiment_prefetch_mode_${mode_label}_chunk_$((chunk_bytes/1024))k_max_$((max_bytes/1024))k_ahead_${ahead_chunks}"
        run_case "$EXPERIMENT_REPO" "$case_name" 1 "$max_bytes" "$ahead_chunks" "$chunk_bytes" "$mode_value"
      done
    done
  done
done

generate_reports

echo
echo "Comparison complete."
echo "Logs and summaries are in $RESULT_ROOT"
echo "Summary TSV: $RESULT_ROOT/comparison_oop_busbw.tsv"
echo "Summary MD : $RESULT_ROOT/comparison_oop_busbw.md"
echo "Summary XLSX: $RESULT_ROOT/simple_l2_prefetch_summary.xlsx"
