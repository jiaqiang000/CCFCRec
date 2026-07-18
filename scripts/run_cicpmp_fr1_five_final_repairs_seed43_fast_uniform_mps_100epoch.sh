#!/usr/bin/env bash
set -euo pipefail

# CICP-MP-FR1 is the single, pre-registered final repair validation.
# It contains one scalar hidden-residual reference, three non-residual performance
# mechanisms, and one parameter-matched semantic shuffle control.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v caffeinate >/dev/null 2>&1 && [[ "${CICPMP_FR1_UNDER_CAFFEINATE:-0}" != "1" ]]; then
  export CICPMP_FR1_UNDER_CAFFEINATE=1
  exec caffeinate -dimsu "$0" "$@"
fi

EPOCH="${EPOCH:-100}"
SEED="${SEED:-43}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
SAVE_BATCH_TIME="${SAVE_BATCH_TIME:-300}"
NEGATIVE_SAMPLING_MODE="${NEGATIVE_SAMPLING_MODE:-fast_uniform}"
DRY_RUN="${DRY_RUN:-0}"
export CCFCREC_DEVICE="${CCFCREC_DEVICE:-mps}"

[[ "${EPOCH}" == "100" ]] || { echo "CICP-MP-FR1 requires EPOCH=100" >&2; exit 1; }
[[ "${SEED}" == "43" ]] || { echo "CICP-MP-FR1 requires SEED=43" >&2; exit 1; }
[[ "${NUM_WORKERS}" == "8" ]] || { echo "CICP-MP-FR1 requires NUM_WORKERS=8" >&2; exit 1; }
[[ "${BATCH_SIZE}" == "1024" ]] || { echo "CICP-MP-FR1 requires BATCH_SIZE=1024" >&2; exit 1; }
[[ "${SAVE_BATCH_TIME}" == "300" ]] || { echo "CICP-MP-FR1 requires SAVE_BATCH_TIME=300" >&2; exit 1; }
[[ "${NEGATIVE_SAMPLING_MODE}" == "fast_uniform" ]] || {
  echo "CICP-MP-FR1 requires NEGATIVE_SAMPLING_MODE=fast_uniform" >&2
  exit 1
}
[[ "${CCFCREC_DEVICE}" == "mps" ]] || {
  echo "This local CICP-MP-FR1 launcher requires CCFCREC_DEVICE=mps" >&2
  exit 1
}

DEFAULT_SCALAR_SOURCE="${REPO_ROOT}/../temp_202607_实验文件记录/temp_20260713/2026-07-13 121409 cicp-train-only-signal-audit-v1_1/cicp_item_profile.csv"
DEFAULT_MP_SOURCE="${REPO_ROOT}/../temp_202607_实验文件记录/temp_20260717/2026-07-17 120710 cicp-mp-v1-train-only-offline-audit/cicp_mp_v1_item_profile.csv"
SCALAR_SOURCE_PROFILE="${SCALAR_SOURCE_PROFILE:-${DEFAULT_SCALAR_SOURCE}}"
MP_SOURCE_PROFILE="${MP_SOURCE_PROFILE:-${DEFAULT_MP_SOURCE}}"
[[ -f "${SCALAR_SOURCE_PROFILE}" ]] || {
  echo "CICP-MP-FR1 scalar source profile not found: ${SCALAR_SOURCE_PROFILE}" >&2
  exit 1
}
[[ -f "${MP_SOURCE_PROFILE}" ]] || {
  echo "CICP-MP-FR1 MP source profile not found: ${MP_SOURCE_PROFILE}" >&2
  exit 1
}

BLOCK_DIM="${CICPMP_FR1_BLOCK_DIM:-8}"
RESIDUAL_MAX_RATIO="${CICPMP_FR1_RESIDUAL_MAX_RATIO:-0.15}"
BASE_WEIGHT_DECAY="${BASE_WEIGHT_DECAY:-0.1}"
METHOD_WEIGHT_DECAY="${CICPMP_FR1_METHOD_WEIGHT_DECAY:-0.0}"
[[ "${BLOCK_DIM}" == "8" ]] || { echo "CICP-MP-FR1 requires BLOCK_DIM=8" >&2; exit 1; }
[[ "${RESIDUAL_MAX_RATIO}" == "0.15" ]] || { echo "CICP-MP-FR1 requires RESIDUAL_MAX_RATIO=0.15" >&2; exit 1; }
[[ "${BASE_WEIGHT_DECAY}" == "0.1" ]] || { echo "CICP-MP-FR1 requires BASE_WEIGHT_DECAY=0.1" >&2; exit 1; }
[[ "${METHOD_WEIGHT_DECAY}" == "0.0" ]] || { echo "CICP-MP-FR1 requires METHOD_WEIGHT_DECAY=0.0" >&2; exit 1; }

export PYTHONHASHSEED="${SEED}"
export PYTHONUNBUFFERED=1
export PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/ccfcrec-py3.11/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || {
  echo "CICP-MP-FR1 Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
}

EXPERIMENT_ID="${EXPERIMENT_ID:-cicpmp_fr1_five_final_repairs_100epoch}"
RUN_STAMP="${RUN_STAMP:-$(date '+%Y-%m-%d_%H%M%S')}"
DEFAULT_RESULT_ROOT="/Volumes/MyPassport/CCFCRec对比学习思路硬盘/实验记录硬盘/ccfcrec_result/${RUN_STAMP}_${EXPERIMENT_ID}_seed${SEED}_workers${NUM_WORKERS}_${NEGATIVE_SAMPLING_MODE}_${CCFCREC_DEVICE}_${EPOCH}epoch"
export RESULT_ROOT="${RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}"
mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/status" "${RESULT_ROOT}/protocol" "${RESULT_ROOT}/runs"

LATEST_POINTER="${CICPMP_FR1_LATEST_POINTER:-${REPO_ROOT}/cicpmp_fr1_five_final_repairs_latest_result_root.txt}"
printf '%s\n' "${RESULT_ROOT}" > "${LATEST_POINTER}"

SCALAR_PROFILE="${RESULT_ROOT}/protocol/cicpmp_fr1_scalar_train_validate_profile.csv"
MP_PROFILE="${RESULT_ROOT}/protocol/cicpmp_fr1_mp_train_standardized_profile.csv"
SHUFFLE_PROFILE="${RESULT_ROOT}/protocol/cicpmp_fr1_mp_train_standardized_shuffle_profile.csv"
PROFILE_AUDIT="${RESULT_ROOT}/protocol/cicpmp_fr1_profile_audit.json"
prepare_command=(
  "${PYTHON_BIN}" scripts/prepare_cicpmp_fr1_profiles.py
  --scalar-source "${SCALAR_SOURCE_PROFILE}"
  --mp-source "${MP_SOURCE_PROFILE}"
  --scalar-output "${SCALAR_PROFILE}"
  --mp-output "${MP_PROFILE}"
  --shuffle-output "${SHUFFLE_PROFILE}"
  --audit-output "${PROFILE_AUDIT}"
  --seed "${SEED}"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  prepare_command+=(--dry-run)
fi
"${prepare_command[@]}"

INITIALIZATION_AUDIT="${RESULT_ROOT}/protocol/cicpmp_fr1_initialization_audit.json"
"${PYTHON_BIN}" scripts/audit_cicpmp_fr1_protocol.py \
  --output "${INITIALIZATION_AUDIT}" \
  --seed "${SEED}"

STATUS_FILE="${RESULT_ROOT}/status.tsv"
MASTER_LOG="${RESULT_ROOT}/logs/master.log"
ACTUAL_RUN_AUDIT="${RESULT_ROOT}/protocol/cicpmp_fr1_actual_run_config_audit.tsv"
ACTUAL_COMMON_HASH_FILE="${RESULT_ROOT}/protocol/cicpmp_fr1_actual_common_parameter_sha256.txt"
TOTAL_RUNS=5

{
  printf 'EXPERIMENT_ID=%s\n' "${EXPERIMENT_ID}"
  printf 'EXPERIMENT_DESIGN_NOTE=2026-07-19 014100 CCFCRec Amazon-VG CICP-MP-FR1一次性修复验证实验设计.md\n'
  printf 'DESIGN_SOURCE_NOTE=2026-07-19 014100 CCFCRec Amazon-VG CICP-MP-FR1实验设计来源.md\n'
  printf 'PROTOCOL_VERSION=cicpmp_fr1_five_final_repairs_v1\n'
  printf 'EVIDENCE_CLASSIFICATION=development_validation_exploration\n'
  printf 'RUN_COUNT=%s\n' "${TOTAL_RUNS}"
  printf 'EPOCHS_PER_RUN=%s\n' "${EPOCH}"
  printf 'METHOD_RUNS=CICP-MP-FR1-E1-SRR CICP-MP-FR1-E2-MFM CICP-MP-FR1-E3-CER CICP-MP-FR1-E4-CMA CICP-MP-FR1-E5-MFS\n'
  printf 'E4_STYLE_HIDDEN_RESIDUAL_BRANCH_COUNT=1\n'
  printf 'E4_STYLE_HIDDEN_RESIDUAL_BRANCH=CICP-MP-FR1-E1-SRR\n'
  printf 'ACTIVATION_SCHEDULE=none\n'
  printf 'INITIAL_EFFECT=exact_zero\n'
  printf 'BASE_WEIGHT_DECAY=%s\n' "${BASE_WEIGHT_DECAY}"
  printf 'METHOD_WEIGHT_DECAY=%s\n' "${METHOD_WEIGHT_DECAY}"
  printf 'FIXED_RELIABILITY_MULTIPLICATION=false\n'
  printf 'OFFLINE_AUXILIARY_TARGET=false\n'
  printf 'MP_STANDARDIZATION_FIT_SPLIT=train\n'
  printf 'MP_STANDARDIZATION_APPLIED_SPLITS=train_validate\n'
  printf 'MP_SEMANTIC_BLOCK_COUNT=4\n'
  printf 'SEMANTIC_CONTROL=whole_23d_row_shuffle_within_split\n'
  printf 'SCALAR_SOURCE_PROFILE=%s\n' "${SCALAR_SOURCE_PROFILE}"
  printf 'MP_SOURCE_PROFILE=%s\n' "${MP_SOURCE_PROFILE}"
  printf 'SCALAR_PROFILE=%s\n' "${SCALAR_PROFILE}"
  printf 'MP_PROFILE=%s\n' "${MP_PROFILE}"
  printf 'SHUFFLE_PROFILE=%s\n' "${SHUFFLE_PROFILE}"
  printf 'PROFILE_AUDIT=%s\n' "${PROFILE_AUDIT}"
  printf 'INITIALIZATION_AUDIT=%s\n' "${INITIALIZATION_AUDIT}"
  printf 'ACTUAL_RUN_CONFIG_AUDIT=%s\n' "${ACTUAL_RUN_AUDIT}"
  printf 'TRAINING_INPUT_USES_VALIDATION_ITEM_METRICS=false\n'
  printf 'TRAINING_INPUT_USES_TEST_ITEM_METRICS=false\n'
  printf 'TEST_METRICS_READ_OR_GENERATED=false\n'
  printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
  printf 'PYTHON_BIN=%s\n' "${PYTHON_BIN}"
  printf 'CCFCREC_DEVICE=%s\n' "${CCFCREC_DEVICE}"
  printf 'SEED=%s\n' "${SEED}"
  printf 'NUM_WORKERS=%s\n' "${NUM_WORKERS}"
  printf 'BATCH_SIZE=%s\n' "${BATCH_SIZE}"
  printf 'SAVE_BATCH_TIME=%s\n' "${SAVE_BATCH_TIME}"
  printf 'NEGATIVE_SAMPLING_MODE=%s\n' "${NEGATIVE_SAMPLING_MODE}"
  printf 'DRY_RUN=%s\n' "${DRY_RUN}"
} > "${RESULT_ROOT}/launcher_manifest.env"

printf 'run_index\trun_label\tmethod_variant\tstate\tstarted_at\tended_at\n' > "${STATUS_FILE}"
printf 'run_index\trun_label\tmethod_variant\tcommon_parameter_sha256\tcommon_parameters\tmethod_specific_parameters\ttotal_parameters\trun_config\n' > "${ACTUAL_RUN_AUDIT}"
tee -a "${MASTER_LOG}" < "${RESULT_ROOT}/launcher_manifest.env"

record_status() {
  local run_index="$1" run_label="$2" variant="$3" state="$4" started_at="$5" ended_at="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_index}" "${run_label}" "${variant}" "${state}" "${started_at}" "${ended_at}" \
    >> "${STATUS_FILE}"
}

verify_actual_run_config() {
  local run_index="$1" run_label="$2" variant="$3" profile_kind="$4" branch_root="$5"
  local run_config audit_fields actual_hash common_count method_count total_count
  run_config="$(find "${branch_root}" -type f -name run_config.json -print | sort | tail -n 1)"
  [[ -n "${run_config}" && -f "${run_config}" ]] || {
    echo "CICP-MP-FR1 run_config.json not found for ${run_label}" >&2
    return 1
  }
  audit_fields="$("${PYTHON_BIN}" - "${run_config}" "${variant}" "${profile_kind}" "${SCALAR_PROFILE}" "${MP_PROFILE}" "${SHUFFLE_PROFILE}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
variant, profile_kind = sys.argv[2:4]
scalar_profile, mp_profile, shuffle_profile = sys.argv[4:7]
config = json.loads(config_path.read_text(encoding="utf-8"))
if config.get("method_variant") != variant:
    raise SystemExit(f"method mismatch: {config.get('method_variant')} != {variant}")
if config.get("cicpmp_fr1_activation_schedule") != "none":
    raise SystemExit("unexpected activation schedule")
if config.get("cicpmp_fr1_initial_effect") != "exact_zero":
    raise SystemExit("initial effect is not exact zero")
if config.get("cicpmp_fr1_uses_offline_auxiliary_target") is not False:
    raise SystemExit("offline auxiliary target must be disabled")
if config.get("cicpmp_fr1_uses_fixed_reliability_multiplication") is not False:
    raise SystemExit("fixed reliability multiplication must be disabled")
expected_residual = variant == "cicpmp_fr1_scalar_residual_reference"
if config.get("cicpmp_fr1_e4_style_hidden_residual") is not expected_residual:
    raise SystemExit("E4-style hidden residual ownership mismatch")
groups = config.get("optimizer_parameter_groups", {})
if groups.get("uses_method_specific_group") is not True:
    raise SystemExit("method-specific optimizer group is missing")
if float(groups.get("base_weight_decay", -1)) != 0.1:
    raise SystemExit("base weight decay mismatch")
if float(groups.get("method_weight_decay", -1)) != 0.0:
    raise SystemExit("method weight decay mismatch")
if profile_kind == "scalar":
    if config.get("cicp_profile_path") != scalar_profile:
        raise SystemExit("scalar profile mismatch")
    if config.get("cicp_mp_feature_input_width") != 0:
        raise SystemExit("scalar reference unexpectedly uses MP input")
else:
    expected_mp = shuffle_profile if profile_kind == "shuffle" else mp_profile
    if config.get("cicp_mp_profile_path") != expected_mp:
        raise SystemExit("MP profile mismatch")
    if config.get("cicp_mp_feature_input_width") != 23:
        raise SystemExit("MP input width mismatch")
if config.get("training_input_uses_validation_item_metrics") is not False:
    raise SystemExit("validation item metrics safety flag failed")
if config.get("training_input_uses_test_item_metrics") is not False:
    raise SystemExit("test item metrics safety flag failed")
digest = str(config.get("ccfcrec_common_parameter_sha256", ""))
counts = config.get("parameter_count", {})
if len(digest) != 64:
    raise SystemExit("invalid common parameter hash")
if int(counts["common"]) + int(counts["method_specific"]) != int(counts["total"]):
    raise SystemExit("parameter count mismatch")
print(digest, counts["common"], counts["method_specific"], counts["total"], sep="\t")
PY
)"
  IFS=$'\t' read -r actual_hash common_count method_count total_count <<< "${audit_fields}"
  [[ "${actual_hash}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "CICP-MP-FR1 failed to parse common parameter hash for ${run_label}" >&2
    return 1
  }
  if [[ -f "${ACTUAL_COMMON_HASH_FILE}" ]]; then
    local expected_hash
    expected_hash="$(sed -n '1p' "${ACTUAL_COMMON_HASH_FILE}")"
    [[ "${actual_hash}" == "${expected_hash}" ]] || {
      echo "CICP-MP-FR1 actual common initialization mismatch for ${run_label}" >&2
      return 1
    }
  else
    printf '%s\n' "${actual_hash}" > "${ACTUAL_COMMON_HASH_FILE}"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_index}" "${run_label}" "${variant}" "${actual_hash}" \
    "${common_count}" "${method_count}" "${total_count}" "${run_config}" \
    >> "${ACTUAL_RUN_AUDIT}"
}

run_one() {
  local run_index="$1" run_label="$2" variant="$3" profile_kind="$4"
  local log_file="${RESULT_ROOT}/logs/${run_index}_${run_label}_${variant}.log"
  local done_file="${RESULT_ROOT}/status/${run_index}_${run_label}.done"
  local branch_root="${RESULT_ROOT}/runs/${run_index}_${run_label}"
  local started_at
  started_at="$(date '+%Y-%m-%d %H:%M:%S')"
  mkdir -p "${branch_root}"

  if [[ -f "${done_file}" ]]; then
    record_status "${run_index}" "${run_label}" "${variant}" "skipped_completed" "${started_at}" "${started_at}"
    return
  fi

  local command=(
    bash scripts/train_amazon_vg_cuda.sh
    --method_variant "${variant}"
    --weight_decay "${BASE_WEIGHT_DECAY}"
    --cicpmp_fr1_block_dim "${BLOCK_DIM}"
    --cicpmp_fr1_residual_max_ratio "${RESIDUAL_MAX_RATIO}"
    --cicpmp_fr1_method_weight_decay "${METHOD_WEIGHT_DECAY}"
    --epoch "${EPOCH}"
    --seed "${SEED}"
    --num_workers "${NUM_WORKERS}"
    --batch_size "${BATCH_SIZE}"
    --negative_sampling_mode "${NEGATIVE_SAMPLING_MODE}"
    --save_batch_time "${SAVE_BATCH_TIME}"
  )
  if [[ "${profile_kind}" == "scalar" ]]; then
    command+=(--cicp_profile_path "${SCALAR_PROFILE}")
  elif [[ "${profile_kind}" == "shuffle" ]]; then
    command+=(--cicp_mp_profile_path "${SHUFFLE_PROFILE}")
  else
    command+=(--cicp_mp_profile_path "${MP_PROFILE}")
  fi

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN [${run_index}/${TOTAL_RUNS}] ${run_label} ${variant}" | tee -a "${MASTER_LOG}"
    printf 'RESULT_ROOT=%q ' "${branch_root}" > "${log_file}"
    printf '%q ' "${command[@]}" >> "${log_file}"
    printf '\n' >> "${log_file}"
    record_status "${run_index}" "${run_label}" "${variant}" "dry_run" "${started_at}" "${started_at}"
    return
  fi

  echo "START [${run_index}/${TOTAL_RUNS}] ${run_label} ${variant} ${started_at}" | tee -a "${MASTER_LOG}"
  record_status "${run_index}" "${run_label}" "${variant}" "running" "${started_at}" ""
  if ! RESULT_ROOT="${branch_root}" "${command[@]}" > "${log_file}" 2>&1; then
    local failed_at
    failed_at="$(date '+%Y-%m-%d %H:%M:%S')"
    record_status "${run_index}" "${run_label}" "${variant}" "failed" "${started_at}" "${failed_at}"
    echo "FAIL [${run_index}/${TOTAL_RUNS}] ${run_label} ${variant} ${failed_at}" | tee -a "${MASTER_LOG}"
    return 1
  fi
  verify_actual_run_config "${run_index}" "${run_label}" "${variant}" "${profile_kind}" "${branch_root}"
  local ended_at
  ended_at="$(date '+%Y-%m-%d %H:%M:%S')"
  touch "${done_file}"
  record_status "${run_index}" "${run_label}" "${variant}" "completed" "${started_at}" "${ended_at}"
  echo "END [${run_index}/${TOTAL_RUNS}] ${run_label} ${variant} ${ended_at}" | tee -a "${MASTER_LOG}"
}

run_one 1 CICP-MP-FR1-E1-SRR cicpmp_fr1_scalar_residual_reference scalar
run_one 2 CICP-MP-FR1-E2-MFM cicpmp_fr1_modality_film real
run_one 3 CICP-MP-FR1-E3-CER cicpmp_fr1_content_expert_routing real
run_one 4 CICP-MP-FR1-E4-CMA cicpmp_fr1_cross_modal_attention real
run_one 5 CICP-MP-FR1-E5-MFS cicpmp_fr1_modality_film_shuffle shuffle

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN_DONE [${TOTAL_RUNS}/${TOTAL_RUNS}] $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${MASTER_LOG}"
else
  echo "ALL_DONE [${TOTAL_RUNS}/${TOTAL_RUNS}] $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${MASTER_LOG}"
fi
