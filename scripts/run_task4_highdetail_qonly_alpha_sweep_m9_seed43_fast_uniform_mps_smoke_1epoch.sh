#!/usr/bin/env bash
set -euo pipefail

# Experiment-specific local Mac launcher for:
# Task4-revise-4 M9 high-detail q-only alpha sweep, seed43,
# workers8, fast_uniform, mps, 1epoch smoke.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v caffeinate >/dev/null 2>&1 && [[ "${TASK4_UNDER_CAFFEINATE:-0}" != "1" ]]; then
  export TASK4_UNDER_CAFFEINATE=1
  exec caffeinate -dimsu "$0" "$@"
fi

EXPERIMENT_ID="task4_highdetail_qonly_alpha_sweep_m9_smoke"
EXPERIMENT_DESIGN_NOTE="2026-07-08 095929 CCFCRec Amazon-VG Task4-revise-4 M9 q-only alpha sweep 设计.md"
TASK4_PROFILE_DEFAULT="/Users/luojiaqiang/Documents/Obsidian Vault/科研/CCFCRec对比学习思路/temp_202607_实验文件记录/temp_20260706/2026-07-06 004222 task4-pre3-train-safe-hard-proxy/task4_train_safe_hard_proxy_profile.csv"

RUN_STAMP="${RUN_STAMP:-$(date '+%Y-%m-%d_%H%M%S')}"
SEED="${SEED:-43}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NEGATIVE_SAMPLING_MODE="${NEGATIVE_SAMPLING_MODE:-fast_uniform}"
EPOCH="${EPOCH:-1}"
SAVE_BATCH_TIME="${SAVE_BATCH_TIME:-300}"
TASK4_ALPHA_LIST="${TASK4_ALPHA_LIST:-0.75 1.0}"
TASK4_PROFILE="${TASK4_PROFILE:-${TASK4_PROFILE_DEFAULT}}"

export CCFCREC_DEVICE="${CCFCREC_DEVICE:-mps}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/ccfcrec-py3.11/bin/python}"

DEFAULT_RESULT_ROOT="/Volumes/MyPassport/CCFCRec对比学习思路硬盘/实验记录硬盘/ccfcrec_result/${RUN_STAMP}_${EXPERIMENT_ID}_seed${SEED}_workers${NUM_WORKERS}_${NEGATIVE_SAMPLING_MODE}_${CCFCREC_DEVICE}_${EPOCH}epoch"
if [[ -z "${TASK4_KEEP_RESULT_ROOT:-}" ]]; then
  export RESULT_ROOT="${DEFAULT_RESULT_ROOT}"
else
  export RESULT_ROOT="${RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}"
fi

mkdir -p "${RESULT_ROOT}/logs"
printf '%s\n' "${RESULT_ROOT}" > "${REPO_ROOT}/task4_highdetail_qonly_alpha_sweep_m9_smoke_latest_result_root.txt"

{
  printf 'EXPERIMENT_ID=%s\n' "${EXPERIMENT_ID}"
  printf 'EXPERIMENT_DESIGN_NOTE=%s\n' "${EXPERIMENT_DESIGN_NOTE}"
  printf 'LAUNCHER_SCRIPT=%s\n' "${BASH_SOURCE[0]}"
  printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
  printf 'TASK4_PROFILE=%s\n' "${TASK4_PROFILE}"
  printf 'TASK4_ALPHA_LIST=%s\n' "${TASK4_ALPHA_LIST}"
  printf 'PYTHON_BIN=%s\n' "${PYTHON_BIN}"
  printf 'CCFCREC_DEVICE=%s\n' "${CCFCREC_DEVICE}"
  printf 'SEED=%s\n' "${SEED}"
  printf 'PYTHONHASHSEED=%s\n' "${PYTHONHASHSEED}"
  printf 'NUM_WORKERS=%s\n' "${NUM_WORKERS}"
  printf 'BATCH_SIZE=%s\n' "${BATCH_SIZE}"
  printf 'NEGATIVE_SAMPLING_MODE=%s\n' "${NEGATIVE_SAMPLING_MODE}"
  printf 'EPOCH=%s\n' "${EPOCH}"
  printf 'SAVE_BATCH_TIME=%s\n' "${SAVE_BATCH_TIME}"
  printf 'METHOD_RUNS=M9a075_real M9a075_shuffle M9a100_real M9a100_shuffle\n'
} > "${RESULT_ROOT}/launcher_manifest.env"

cat "${RESULT_ROOT}/launcher_manifest.env" | tee -a "${RESULT_ROOT}/logs/master.log"

run_one() {
  local run_label="$1"
  local variant="$2"
  local alpha="$3"
  shift 3
  local log_file="${RESULT_ROOT}/logs/${run_label}_${variant}_alpha${alpha}.log"
  echo "START ${run_label} ${variant} alpha=${alpha} $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${RESULT_ROOT}/logs/master.log"
  bash scripts/train_amazon_vg_cuda.sh \
    --method_variant "${variant}" \
    --task4_profile_path "${TASK4_PROFILE}" \
    --task4_loss_alpha "${alpha}" \
    --epoch "${EPOCH}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${BATCH_SIZE}" \
    --negative_sampling_mode "${NEGATIVE_SAMPLING_MODE}" \
    --save_batch_time "${SAVE_BATCH_TIME}" \
    "$@" \
    > "${log_file}" 2>&1
  echo "END ${run_label} ${variant} alpha=${alpha} $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${RESULT_ROOT}/logs/master.log"
}

alpha_label() {
  case "$1" in
    0.75) printf '075' ;;
    1.0|1.00) printf '100' ;;
    *) printf '%s' "${1/./}" ;;
  esac
}

for alpha in ${TASK4_ALPHA_LIST}; do
  label="$(alpha_label "${alpha}")"
  run_one "M9a${label}_real" task4_highdetail_trainhard_weight "${alpha}" --task4_disable_self_contrast_weight
  run_one "M9a${label}_shuffle" task4_highdetail_trainhard_shuffle_weight "${alpha}" --task4_disable_self_contrast_weight --task4_shuffle_seed "${SEED}"
done

echo "ALL_DONE $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${RESULT_ROOT}/logs/master.log"
