#!/usr/bin/env bash
set -euo pipefail

# Seven sequential M11-R2 runs: four full-target performance mechanisms,
# the original M11 carrier, and two exposure-matched controls.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v caffeinate >/dev/null 2>&1 && [[ "${M11R2_UNDER_CAFFEINATE:-0}" != "1" ]]; then
  export M11R2_UNDER_CAFFEINATE=1
  exec caffeinate -dimsu "$0" "$@"
fi

DESIGN_POINTER="${REPO_ROOT}/m11r2_seven_run_design_latest_output_dir.txt"
if [[ ! -f "${DESIGN_POINTER}" ]]; then
  echo "M11-R2 design pointer not found: ${DESIGN_POINTER}" >&2
  echo "Run validata/analyze_amazon_vg_m11r2_seven_run_design.py first." >&2
  exit 1
fi

DESIGN_OUTPUT_DIR="$(sed -n '1p' "${DESIGN_POINTER}")"
TASK4_PROFILE_DEFAULT="${DESIGN_OUTPUT_DIR}/m11r2_seven_run_profile.csv"
ROUTE_DECISION="${DESIGN_OUTPUT_DIR}/m11r2_route_decision.json"
EXPERIMENT_DESIGN="${DESIGN_OUTPUT_DIR}/m11r2_experiment_design.csv"
for required_file in "${TASK4_PROFILE_DEFAULT}" "${ROUTE_DECISION}" "${EXPERIMENT_DESIGN}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "M11-R2 design artifact missing: ${required_file}" >&2
    exit 1
  fi
done
if ! grep -q '"route": "m11r2_seven_run_100epoch_design_ready"' "${ROUTE_DECISION}"; then
  echo "M11-R2 offline design gate is not ready: ${ROUTE_DECISION}" >&2
  exit 1
fi

EXPERIMENT_ID="${EXPERIMENT_ID:-m11r2_seven_experiments_100epoch}"
RUN_STAMP="${RUN_STAMP:-$(date '+%Y-%m-%d_%H%M%S')}"
SEED="${SEED:-43}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NEGATIVE_SAMPLING_MODE="${NEGATIVE_SAMPLING_MODE:-fast_uniform}"
EPOCH="${EPOCH:-100}"
SAVE_BATCH_TIME="${SAVE_BATCH_TIME:-300}"
TASK4_PROFILE="${TASK4_PROFILE:-${TASK4_PROFILE_DEFAULT}}"
M11R2_QBPR_ALPHA="${M11R2_QBPR_ALPHA:-0.75}"
M11R2_FOCAL_GAMMA="${M11R2_FOCAL_GAMMA:-2.0}"
M11R2_FOCAL_TEMPERATURE="${M11R2_FOCAL_TEMPERATURE:-1.0}"
M11R2_CURRICULUM_WARMUP_EPOCHS="${M11R2_CURRICULUM_WARMUP_EPOCHS:-20}"
M11R2_FEATURE_DIM="${M11R2_FEATURE_DIM:-16}"
M11R2_COMPETITOR_ALPHA="${M11R2_COMPETITOR_ALPHA:-0.25}"
M11R2_COMPETITOR_MARGIN="${M11R2_COMPETITOR_MARGIN:-0.1}"
M11R2_COMPETITOR_K="${M11R2_COMPETITOR_K:-20}"

if [[ "${EPOCH}" != "100" ]]; then
  echo "M11-R2 requires EPOCH=100; received EPOCH=${EPOCH}" >&2
  exit 1
fi
if [[ "${NEGATIVE_SAMPLING_MODE}" != "fast_uniform" ]]; then
  echo "M11-R2 requires NEGATIVE_SAMPLING_MODE=fast_uniform" >&2
  exit 1
fi

export CCFCREC_DEVICE="${CCFCREC_DEVICE:-mps}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/ccfcrec-py3.11/bin/python}"

DEFAULT_RESULT_ROOT="/Volumes/MyPassport/CCFCRec对比学习思路硬盘/实验记录硬盘/ccfcrec_result/${RUN_STAMP}_${EXPERIMENT_ID}_seed${SEED}_workers${NUM_WORKERS}_${NEGATIVE_SAMPLING_MODE}_${CCFCREC_DEVICE}_${EPOCH}epoch"
export RESULT_ROOT="${RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}"
mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/status"
printf '%s\n' "${RESULT_ROOT}" > "${REPO_ROOT}/m11r2_seven_experiments_latest_result_root.txt"

STATUS_FILE="${RESULT_ROOT}/status.tsv"
MASTER_LOG="${RESULT_ROOT}/logs/master.log"
TOTAL_RUNS=7

{
  printf 'EXPERIMENT_ID=%s\n' "${EXPERIMENT_ID}"
  printf 'EVIDENCE_CLASSIFICATION=exploratory_only\n'
  printf 'TARGET_DEFINITION=unchanged_full_m11_high_acat_low_rsp_neighbor_support\n'
  printf 'TARGET_TRAIN_ITEMS=4261\n'
  printf 'TARGET_TRAIN_INTERACTIONS=58280\n'
  printf 'TARGET_TRAIN_INTERACTION_SHARE=0.175085469829\n'
  printf 'TRAINING_INPUT_USES_EVAL_ITEM_METRICS=false\n'
  printf 'RUN_COUNT=%s\n' "${TOTAL_RUNS}"
  printf 'EPOCHS_PER_RUN=%s\n' "${EPOCH}"
  printf 'LAUNCHER_SCRIPT=%s\n' "${BASH_SOURCE[0]}"
  printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
  printf 'DESIGN_OUTPUT_DIR=%s\n' "${DESIGN_OUTPUT_DIR}"
  printf 'EXPERIMENT_DESIGN=%s\n' "${EXPERIMENT_DESIGN}"
  printf 'TASK4_PROFILE=%s\n' "${TASK4_PROFILE}"
  printf 'ROUTE_DECISION=%s\n' "${ROUTE_DECISION}"
  printf 'METHOD_RUNS=M11R2E1_qbpr_score M11R2E2_focal_qbpr M11R2E3_curriculum_qbpr M11R2E4_feature_fusion M11R2C1_original_m11 M11R2C2_popmatch M11R2C3_lowacat\n'
  printf 'M11R2_QBPR_ALPHA=%s\n' "${M11R2_QBPR_ALPHA}"
  printf 'M11R2_FOCAL_GAMMA=%s\n' "${M11R2_FOCAL_GAMMA}"
  printf 'M11R2_FOCAL_TEMPERATURE=%s\n' "${M11R2_FOCAL_TEMPERATURE}"
  printf 'M11R2_CURRICULUM_WARMUP_EPOCHS=%s\n' "${M11R2_CURRICULUM_WARMUP_EPOCHS}"
  printf 'M11R2_FEATURE_DIM=%s\n' "${M11R2_FEATURE_DIM}"
  printf 'M11R2_COMPETITOR_ALPHA=%s\n' "${M11R2_COMPETITOR_ALPHA}"
  printf 'M11R2_COMPETITOR_MARGIN=%s\n' "${M11R2_COMPETITOR_MARGIN}"
  printf 'M11R2_COMPETITOR_K=%s\n' "${M11R2_COMPETITOR_K}"
  printf 'PYTHON_BIN=%s\n' "${PYTHON_BIN}"
  printf 'CCFCREC_DEVICE=%s\n' "${CCFCREC_DEVICE}"
  printf 'SEED=%s\n' "${SEED}"
  printf 'PYTHONHASHSEED=%s\n' "${PYTHONHASHSEED}"
  printf 'NUM_WORKERS=%s\n' "${NUM_WORKERS}"
  printf 'BATCH_SIZE=%s\n' "${BATCH_SIZE}"
  printf 'NEGATIVE_SAMPLING_MODE=%s\n' "${NEGATIVE_SAMPLING_MODE}"
  printf 'SAVE_BATCH_TIME=%s\n' "${SAVE_BATCH_TIME}"
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
  shift 3
  local log_file="${RESULT_ROOT}/logs/${run_index}_${run_label}_${variant}.log"
  local done_file="${RESULT_ROOT}/status/${run_index}_${run_label}.done"
  local started_at
  started_at="$(date '+%Y-%m-%d %H:%M:%S')"
  if [[ -f "${done_file}" ]]; then
    echo "SKIP [${run_index}/${TOTAL_RUNS}] ${run_label} already completed" | tee -a "${MASTER_LOG}"
    record_status "${run_index}" "${run_label}" "${variant}" "skipped_completed" "${started_at}" "${started_at}"
    return
  fi

  echo "START [${run_index}/${TOTAL_RUNS}] ${run_label} ${variant} ${started_at}" | tee -a "${MASTER_LOG}"
  record_status "${run_index}" "${run_label}" "${variant}" "running" "${started_at}" ""
  if ! bash scripts/train_amazon_vg_cuda.sh \
    --method_variant "${variant}" \
    --task4_profile_path "${TASK4_PROFILE}" \
    --task4_loss_alpha "${M11R2_QBPR_ALPHA}" \
    --task4_competitor_alpha "${M11R2_COMPETITOR_ALPHA}" \
    --task4_competitor_margin "${M11R2_COMPETITOR_MARGIN}" \
    --task4_competitor_k "${M11R2_COMPETITOR_K}" \
    --m11r2_focal_gamma "${M11R2_FOCAL_GAMMA}" \
    --m11r2_focal_temperature "${M11R2_FOCAL_TEMPERATURE}" \
    --m11r2_curriculum_warmup_epochs "${M11R2_CURRICULUM_WARMUP_EPOCHS}" \
    --m11r2_feature_dim "${M11R2_FEATURE_DIM}" \
    --epoch "${EPOCH}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${BATCH_SIZE}" \
    --negative_sampling_mode "${NEGATIVE_SAMPLING_MODE}" \
    --save_batch_time "${SAVE_BATCH_TIME}" \
    "$@" \
    > "${log_file}" 2>&1; then
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

run_one 1 M11R2E1_qbpr_score m11r2_qbpr_score_weight --task4_disable_self_contrast_weight
run_one 2 M11R2E2_focal_qbpr m11r2_qbpr_focal
run_one 3 M11R2E3_curriculum_qbpr m11r2_qbpr_curriculum --task4_disable_self_contrast_weight
run_one 4 M11R2E4_feature_fusion m11r2_target_feature_fusion
run_one 5 M11R2C1_original_m11 m11r1_full_target_competitor_pair
run_one 6 M11R2C2_popmatch m11r1_popmatch_competitor_pair_control
run_one 7 M11R2C3_lowacat m11r1_lowacat_competitor_pair_control

echo "ALL_DONE [${TOTAL_RUNS}/${TOTAL_RUNS}] $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${MASTER_LOG}"
