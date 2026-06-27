#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}/Amazon VG"

export CCFCREC_DEVICE="${CCFCREC_DEVICE:-cuda}"
export CCFCREC_CUDA_DEVICE="${CCFCREC_CUDA_DEVICE:-0}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-43}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-43}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PREFETCH_FACTOR="${PREFETCH_FACTOR:-4}"
SAVE_BATCH_TIME="${SAVE_BATCH_TIME:-300}"
RESULT_ROOT="${RESULT_ROOT:-/hy-tmp/ccfcrec_result}"
VALIDATE_BATCH_SIZE="${VALIDATE_BATCH_SIZE:-512}"
NEGATIVE_SAMPLING_MODE="${NEGATIVE_SAMPLING_MODE:-fast_uniform}"
NEGATIVE_SAMPLING_CACHE_SIZE="${NEGATIVE_SAMPLING_CACHE_SIZE:-512}"

"${PYTHON_BIN}" model.py \
  --seed "${SEED}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --pin_memory \
  --persistent_workers \
  --prefetch_factor "${PREFETCH_FACTOR}" \
  --save_batch_time "${SAVE_BATCH_TIME}" \
  --result_root "${RESULT_ROOT}" \
  --validate_batch_size "${VALIDATE_BATCH_SIZE}" \
  --negative_sampling_mode "${NEGATIVE_SAMPLING_MODE}" \
  --negative_sampling_cache_size "${NEGATIVE_SAMPLING_CACHE_SIZE}" \
  "$@"
