#!/usr/bin/env bash
set -euo pipefail

# Four sequential E4 follow-ups. The launcher never reads test metrics and
# rejects any formal protocol other than the fixed 100-epoch configuration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v caffeinate >/dev/null 2>&1 && [[ "${M11R3_UNDER_CAFFEINATE:-0}" != "1" ]]; then
  export M11R3_UNDER_CAFFEINATE=1
  exec caffeinate -dimsu "$0" "$@"
fi

EPOCH="${EPOCH:-100}"
NEGATIVE_SAMPLING_MODE="${NEGATIVE_SAMPLING_MODE:-fast_uniform}"
if [[ "${EPOCH}" != "100" ]]; then
  echo "M11-R3 requires EPOCH=100; received EPOCH=${EPOCH}" >&2
  exit 1
fi
if [[ "${NEGATIVE_SAMPLING_MODE}" != "fast_uniform" ]]; then
  echo "M11-R3 requires NEGATIVE_SAMPLING_MODE=fast_uniform" >&2
  exit 1
fi

if [[ -z "${TASK4_PROFILE:-}" ]]; then
  DESIGN_POINTER="${REPO_ROOT}/m11r2_seven_run_design_latest_output_dir.txt"
  if [[ ! -f "${DESIGN_POINTER}" ]]; then
    echo "M11-R2 clean profile pointer not found: ${DESIGN_POINTER}" >&2
    exit 1
  fi
  DESIGN_OUTPUT_DIR="$(sed -n '1p' "${DESIGN_POINTER}")"
  TASK4_PROFILE="${DESIGN_OUTPUT_DIR}/m11r2_seven_run_profile.csv"
fi
if [[ ! -f "${TASK4_PROFILE}" ]]; then
  echo "M11-R3 clean training profile not found: ${TASK4_PROFILE}" >&2
  exit 1
fi

EXPERIMENT_ID="${EXPERIMENT_ID:-m11r3_four_e4_followups_100epoch}"
RUN_STAMP="${RUN_STAMP:-$(date '+%Y-%m-%d_%H%M%S')}"
SEED="${SEED:-43}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
SAVE_BATCH_TIME="${SAVE_BATCH_TIME:-300}"
M11_FEATURE_DIM="${M11_FEATURE_DIM:-16}"
M11_RESIDUAL_MAX_RATIO="${M11_RESIDUAL_MAX_RATIO:-0.15}"
M11_NEIGHBOR_LOSS_WEIGHT="${M11_NEIGHBOR_LOSS_WEIGHT:-0.1}"
M11_NEIGHBOR_TEMPERATURE="${M11_NEIGHBOR_TEMPERATURE:-0.25}"
M11_FILM_STRENGTH="${M11_FILM_STRENGTH:-0.1}"
DRY_RUN="${DRY_RUN:-0}"

export CCFCREC_DEVICE="${CCFCREC_DEVICE:-mps}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/ccfcrec-py3.11/bin/python}"

"${PYTHON_BIN}" - "${TASK4_PROFILE}" <<'PY'
import sys
import pandas as pd

profile = pd.read_csv(sys.argv[1], dtype={"raw_asin": str}, low_memory=False)
required = {
    "raw_asin", "split", "s_cat_v3", "RSP_score",
    "category_neighbor_mismatch_proxy_score", "support_tail_proxy_score",
    "m11_target_score", "m11r1_full_target_flag", "m11r1_full_target_loss_score",
}
forbidden = {
    "hr@20", "ndcg@20", "baseline_ndcg@20", "baseline_hr@20",
    "baseline_margin_proxy", "baseline_best_target_rank",
    "eval_baseline_hard_flag", "delta_ndcg@20", "delta_hr@20",
}
missing = sorted(required - set(profile.columns))
present_forbidden = sorted(forbidden & set(profile.columns))
if missing:
    raise SystemExit(f"M11-R3 profile missing required recommendation-time columns: {missing}")
if present_forbidden:
    raise SystemExit(f"M11-R3 profile contains forbidden evaluation columns: {present_forbidden}")
if set(profile["split"].astype(str)) != {"train", "validate", "test"}:
    raise SystemExit("M11-R3 profile must cover train/validate/test item identities")
PY

DEFAULT_RESULT_ROOT="/Volumes/MyPassport/CCFCRec对比学习思路硬盘/实验记录硬盘/ccfcrec_result/${RUN_STAMP}_${EXPERIMENT_ID}_seed${SEED}_workers${NUM_WORKERS}_${NEGATIVE_SAMPLING_MODE}_${CCFCREC_DEVICE}_${EPOCH}epoch"
export RESULT_ROOT="${RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}"
mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/status"
printf '%s\n' "${RESULT_ROOT}" > "${REPO_ROOT}/m11r3_four_e4_followups_latest_result_root.txt"

STATUS_FILE="${RESULT_ROOT}/status.tsv"
MASTER_LOG="${RESULT_ROOT}/logs/master.log"
TOTAL_RUNS=4

{
  printf 'EXPERIMENT_ID=%s\n' "${EXPERIMENT_ID}"
  printf 'EVIDENCE_CLASSIFICATION=development_validation_exploration\n'
  printf 'TARGET_DEFINITION=unchanged_full_m11_high_acat_low_rsp_neighbor_support\n'
  printf 'TRAINING_INPUT_USES_VALIDATION_ITEM_METRICS=false\n'
  printf 'TRAINING_INPUT_USES_TEST_ITEM_METRICS=false\n'
  printf 'RUN_COUNT=%s\n' "${TOTAL_RUNS}"
  printf 'EPOCHS_PER_RUN=%s\n' "${EPOCH}"
  printf 'METHOD_RUNS=M11R3E1_dual_residual M11R3E2_norm_capped M11R3E3_neighbor_transfer M11R3E4_target_film\n'
  printf 'TASK4_PROFILE=%s\n' "${TASK4_PROFILE}"
  printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
  printf 'M11_FEATURE_DIM=%s\n' "${M11_FEATURE_DIM}"
  printf 'M11_RESIDUAL_MAX_RATIO=%s\n' "${M11_RESIDUAL_MAX_RATIO}"
  printf 'M11_NEIGHBOR_LOSS_WEIGHT=%s\n' "${M11_NEIGHBOR_LOSS_WEIGHT}"
  printf 'M11_NEIGHBOR_TEMPERATURE=%s\n' "${M11_NEIGHBOR_TEMPERATURE}"
  printf 'M11_FILM_STRENGTH=%s\n' "${M11_FILM_STRENGTH}"
  printf 'PYTHON_BIN=%s\n' "${PYTHON_BIN}"
  printf 'CCFCREC_DEVICE=%s\n' "${CCFCREC_DEVICE}"
  printf 'SEED=%s\n' "${SEED}"
  printf 'NUM_WORKERS=%s\n' "${NUM_WORKERS}"
  printf 'BATCH_SIZE=%s\n' "${BATCH_SIZE}"
  printf 'NEGATIVE_SAMPLING_MODE=%s\n' "${NEGATIVE_SAMPLING_MODE}"
  printf 'DRY_RUN=%s\n' "${DRY_RUN}"
} > "${RESULT_ROOT}/launcher_manifest.env"

printf 'run_index\trun_label\tmethod_variant\tstate\tstarted_at\tended_at\n' > "${STATUS_FILE}"
tee -a "${MASTER_LOG}" < "${RESULT_ROOT}/launcher_manifest.env"

record_status() {
  local run_index="$1"
  local run_label="$2"
  local variant="$3"
  local state="$4"
  local started_at="$5"
  local ended_at="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_index}" "${run_label}" "${variant}" "${state}" "${started_at}" "${ended_at}" \
    >> "${STATUS_FILE}"
}

run_one() {
  local run_index="$1"
  local run_label="$2"
  local variant="$3"
  local log_file="${RESULT_ROOT}/logs/${run_index}_${run_label}_${variant}.log"
  local done_file="${RESULT_ROOT}/status/${run_index}_${run_label}.done"
  local started_at
  started_at="$(date '+%Y-%m-%d %H:%M:%S')"

  if [[ -f "${done_file}" ]]; then
    echo "SKIP [${run_index}/${TOTAL_RUNS}] ${run_label} already completed" | tee -a "${MASTER_LOG}"
    record_status "${run_index}" "${run_label}" "${variant}" "skipped_completed" "${started_at}" "${started_at}"
    return
  fi

  local command=(
    bash scripts/train_amazon_vg_cuda.sh
    --method_variant "${variant}"
    --task4_profile_path "${TASK4_PROFILE}"
    --m11r2_feature_dim "${M11_FEATURE_DIM}"
    --m11r3_residual_max_ratio "${M11_RESIDUAL_MAX_RATIO}"
    --m11r3_neighbor_loss_weight "${M11_NEIGHBOR_LOSS_WEIGHT}"
    --m11r3_neighbor_temperature "${M11_NEIGHBOR_TEMPERATURE}"
    --m11r3_film_strength "${M11_FILM_STRENGTH}"
    --epoch "${EPOCH}"
    --num_workers "${NUM_WORKERS}"
    --batch_size "${BATCH_SIZE}"
    --negative_sampling_mode "${NEGATIVE_SAMPLING_MODE}"
    --save_batch_time "${SAVE_BATCH_TIME}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN [${run_index}/${TOTAL_RUNS}] ${run_label} ${variant} ${started_at}" | tee -a "${MASTER_LOG}"
    printf '%q ' "${command[@]}" | tee "${log_file}"
    printf '\n' | tee -a "${log_file}"
    record_status "${run_index}" "${run_label}" "${variant}" "dry_run" "${started_at}" "${started_at}"
    return
  fi

  echo "START [${run_index}/${TOTAL_RUNS}] ${run_label} ${variant} ${started_at}" | tee -a "${MASTER_LOG}"
  record_status "${run_index}" "${run_label}" "${variant}" "running" "${started_at}" ""
  if ! "${command[@]}" > "${log_file}" 2>&1; then
    local failed_at
    failed_at="$(date '+%Y-%m-%d %H:%M:%S')"
    record_status "${run_index}" "${run_label}" "${variant}" "failed" "${started_at}" "${failed_at}"
    echo "FAIL [${run_index}/${TOTAL_RUNS}] ${run_label} ${variant} ${failed_at}" | tee -a "${MASTER_LOG}"
    return 1
  fi

  local ended_at
  ended_at="$(date '+%Y-%m-%d %H:%M:%S')"
  touch "${done_file}"
  record_status "${run_index}" "${run_label}" "${variant}" "completed" "${started_at}" "${ended_at}"
  echo "END [${run_index}/${TOTAL_RUNS}] ${run_label} ${variant} ${ended_at}" | tee -a "${MASTER_LOG}"
}

run_one 1 M11R3E1_dual_residual m11r3_dual_residual
run_one 2 M11R3E2_norm_capped m11r3_norm_capped_residual
run_one 3 M11R3E3_neighbor_transfer m11r3_neighbor_transfer
run_one 4 M11R3E4_target_film m11r3_target_film

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN_DONE [${TOTAL_RUNS}/${TOTAL_RUNS}] $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${MASTER_LOG}"
else
  echo "ALL_DONE [${TOTAL_RUNS}/${TOTAL_RUNS}] $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${MASTER_LOG}"
fi
