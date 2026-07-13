#!/usr/bin/env bash
set -euo pipefail

# M11-R1 full-target 100-epoch exploratory training. This launcher does not
# make the historically test-informed target provenance confirmatory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v caffeinate >/dev/null 2>&1 && [[ "${M11R1_UNDER_CAFFEINATE:-0}" != "1" ]]; then
  export M11R1_UNDER_CAFFEINATE=1
  exec caffeinate -dimsu "$0" "$@"
fi

AUDIT_POINTER="${REPO_ROOT}/m11r1_full_target_control_audit_latest_output_dir.txt"
if [[ ! -f "${AUDIT_POINTER}" ]]; then
  echo "M11-R1 audit pointer not found: ${AUDIT_POINTER}" >&2
  echo "Run validata/analyze_amazon_vg_m11r1_full_target_exposure_control_audit.py first." >&2
  exit 1
fi

M11R1_AUDIT_OUTPUT_DIR="$(sed -n '1p' "${AUDIT_POINTER}")"
TASK4_PROFILE_DEFAULT="${M11R1_AUDIT_OUTPUT_DIR}/m11r1_full_target_exposure_matched_profile.csv"
ROUTE_DECISION="${M11R1_AUDIT_OUTPUT_DIR}/m11r1_route_decision.json"
if [[ ! -f "${TASK4_PROFILE_DEFAULT}" || ! -f "${ROUTE_DECISION}" ]]; then
  echo "M11-R1 audited profile or route decision is missing in ${M11R1_AUDIT_OUTPUT_DIR}" >&2
  exit 1
fi
if ! grep -q '"route": "m11r1_full_target_exposure_matched_controls_ready_for_100epoch"' "${ROUTE_DECISION}"; then
  echo "M11-R1 offline gate is not ready: ${ROUTE_DECISION}" >&2
  exit 1
fi

EXPERIMENT_ID="${EXPERIMENT_ID:-m11r1_full_target_exposure_controls_exploratory_100epoch}"
RUN_STAMP="${RUN_STAMP:-$(date '+%Y-%m-%d_%H%M%S')}"
SEED="${SEED:-43}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
NEGATIVE_SAMPLING_MODE="${NEGATIVE_SAMPLING_MODE:-fast_uniform}"
EPOCH="${EPOCH:-100}"
SAVE_BATCH_TIME="${SAVE_BATCH_TIME:-300}"
M11R1_COMPETITOR_ALPHA="${M11R1_COMPETITOR_ALPHA:-0.25}"
M11R1_COMPETITOR_MARGIN="${M11R1_COMPETITOR_MARGIN:-0.1}"
M11R1_COMPETITOR_K="${M11R1_COMPETITOR_K:-20}"
TASK4_PROFILE="${TASK4_PROFILE:-${TASK4_PROFILE_DEFAULT}}"

if [[ "${EPOCH}" != "100" ]]; then
  echo "M11-R1 requires EPOCH=100; received EPOCH=${EPOCH}" >&2
  exit 1
fi

export CCFCREC_DEVICE="${CCFCREC_DEVICE:-mps}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/ccfcrec-py3.11/bin/python}"

DEFAULT_RESULT_ROOT="/Volumes/MyPassport/CCFCRec对比学习思路硬盘/实验记录硬盘/ccfcrec_result/${RUN_STAMP}_${EXPERIMENT_ID}_seed${SEED}_workers${NUM_WORKERS}_${NEGATIVE_SAMPLING_MODE}_${CCFCREC_DEVICE}_${EPOCH}epoch"
export RESULT_ROOT="${RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}"
mkdir -p "${RESULT_ROOT}/logs"
printf '%s\n' "${RESULT_ROOT}" > "${REPO_ROOT}/m11r1_full_target_exposure_controls_latest_result_root.txt"

{
  printf 'EXPERIMENT_ID=%s\n' "${EXPERIMENT_ID}"
  printf 'EVIDENCE_CLASSIFICATION=exploratory_only\n'
  printf 'HISTORICAL_TEST_INFORMED_PROVENANCE=true\n'
  printf 'LAUNCHER_SCRIPT=%s\n' "${BASH_SOURCE[0]}"
  printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
  printf 'M11R1_AUDIT_OUTPUT_DIR=%s\n' "${M11R1_AUDIT_OUTPUT_DIR}"
  printf 'TASK4_PROFILE=%s\n' "${TASK4_PROFILE}"
  printf 'ROUTE_DECISION=%s\n' "${ROUTE_DECISION}"
  printf 'M11R1_COMPETITOR_ALPHA=%s\n' "${M11R1_COMPETITOR_ALPHA}"
  printf 'M11R1_COMPETITOR_MARGIN=%s\n' "${M11R1_COMPETITOR_MARGIN}"
  printf 'M11R1_COMPETITOR_K=%s\n' "${M11R1_COMPETITOR_K}"
  printf 'CONTROL_SCOPE=full_target_popmatch_lowacat\n'
  printf 'METHOD_RUNS=M11R1_real M11R1_popmatch M11R1_lowacat\n'
  printf 'PYTHON_BIN=%s\n' "${PYTHON_BIN}"
  printf 'CCFCREC_DEVICE=%s\n' "${CCFCREC_DEVICE}"
  printf 'SEED=%s\n' "${SEED}"
  printf 'PYTHONHASHSEED=%s\n' "${PYTHONHASHSEED}"
  printf 'NUM_WORKERS=%s\n' "${NUM_WORKERS}"
  printf 'BATCH_SIZE=%s\n' "${BATCH_SIZE}"
  printf 'NEGATIVE_SAMPLING_MODE=%s\n' "${NEGATIVE_SAMPLING_MODE}"
  printf 'EPOCH=%s\n' "${EPOCH}"
  printf 'SAVE_BATCH_TIME=%s\n' "${SAVE_BATCH_TIME}"
} > "${RESULT_ROOT}/launcher_manifest.env"

tee -a "${RESULT_ROOT}/logs/master.log" < "${RESULT_ROOT}/launcher_manifest.env"

run_one() {
  local run_label="$1"
  local variant="$2"
  local log_file="${RESULT_ROOT}/logs/${run_label}_${variant}.log"
  echo "START ${run_label} ${variant} $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${RESULT_ROOT}/logs/master.log"
  bash scripts/train_amazon_vg_cuda.sh \
    --method_variant "${variant}" \
    --task4_profile_path "${TASK4_PROFILE}" \
    --task4_competitor_alpha "${M11R1_COMPETITOR_ALPHA}" \
    --task4_competitor_margin "${M11R1_COMPETITOR_MARGIN}" \
    --task4_competitor_k "${M11R1_COMPETITOR_K}" \
    --epoch "${EPOCH}" \
    --num_workers "${NUM_WORKERS}" \
    --batch_size "${BATCH_SIZE}" \
    --negative_sampling_mode "${NEGATIVE_SAMPLING_MODE}" \
    --save_batch_time "${SAVE_BATCH_TIME}" \
    > "${log_file}" 2>&1
  echo "END ${run_label} ${variant} $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${RESULT_ROOT}/logs/master.log"
}

run_one M11R1_real m11r1_full_target_competitor_pair
run_one M11R1_popmatch m11r1_popmatch_competitor_pair_control
run_one M11R1_lowacat m11r1_lowacat_competitor_pair_control

echo "ALL_DONE $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${RESULT_ROOT}/logs/master.log"
