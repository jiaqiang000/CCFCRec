#!/usr/bin/env bash
set -euo pipefail

# Four sequential M11-R4 performance-first runs. Validation outcomes are used
# only after training for method selection; they never enter this training profile.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v caffeinate >/dev/null 2>&1 && [[ "${M11R4_UNDER_CAFFEINATE:-0}" != "1" ]]; then
  export M11R4_UNDER_CAFFEINATE=1
  exec caffeinate -dimsu "$0" "$@"
fi

EPOCH="${EPOCH:-100}"
NEGATIVE_SAMPLING_MODE="${NEGATIVE_SAMPLING_MODE:-fast_uniform}"
export CCFCREC_DEVICE="${CCFCREC_DEVICE:-mps}"
if [[ "${EPOCH}" != "100" ]]; then
  echo "M11-R4 requires EPOCH=100; received EPOCH=${EPOCH}" >&2
  exit 1
fi
if [[ "${NEGATIVE_SAMPLING_MODE}" != "fast_uniform" ]]; then
  echo "M11-R4 requires NEGATIVE_SAMPLING_MODE=fast_uniform" >&2
  exit 1
fi
if [[ "${CCFCREC_DEVICE}" != "mps" ]]; then
  echo "This local M11-R4 launcher requires CCFCREC_DEVICE=mps" >&2
  exit 1
fi

if [[ -z "${SOURCE_PROFILE:-}" ]]; then
  DESIGN_POINTER="${REPO_ROOT}/m11r2_seven_run_design_latest_output_dir.txt"
  if [[ ! -f "${DESIGN_POINTER}" ]]; then
    echo "M11-R2 clean profile pointer not found: ${DESIGN_POINTER}" >&2
    exit 1
  fi
  DESIGN_OUTPUT_DIR="$(sed -n '1p' "${DESIGN_POINTER}")"
  SOURCE_PROFILE="${DESIGN_OUTPUT_DIR}/m11r2_seven_run_profile.csv"
fi
if [[ ! -f "${SOURCE_PROFILE}" ]]; then
  echo "M11-R4 source profile not found: ${SOURCE_PROFILE}" >&2
  exit 1
fi

EXPERIMENT_ID="${EXPERIMENT_ID:-m11r4_four_performance_first_100epoch}"
RUN_STAMP="${RUN_STAMP:-$(date '+%Y-%m-%d_%H%M%S')}"
SEED="${SEED:-43}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
SAVE_BATCH_TIME="${SAVE_BATCH_TIME:-300}"
M11_FEATURE_DIM="${M11_FEATURE_DIM:-16}"
M11R4_EXPERT_FILM_STRENGTH="${M11R4_EXPERT_FILM_STRENGTH:-0.20}"
M11R4_FUSION_STRENGTH="${M11R4_FUSION_STRENGTH:-0.25}"
M11R4_RELATION_LOSS_WEIGHT="${M11R4_RELATION_LOSS_WEIGHT:-0.05}"
M11R4_FOCAL_ALPHA="${M11R4_FOCAL_ALPHA:-1.50}"
M11R4_FOCAL_GAMMA="${M11R4_FOCAL_GAMMA:-2.0}"
M11R4_FOCAL_TEMPERATURE="${M11R4_FOCAL_TEMPERATURE:-0.50}"
M11R4_FOCAL_FLOOR="${M11R4_FOCAL_FLOOR:-0.35}"
DRY_RUN="${DRY_RUN:-0}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/ccfcrec-py3.11/bin/python}"

DEFAULT_RESULT_ROOT="/Volumes/MyPassport/CCFCRec对比学习思路硬盘/实验记录硬盘/ccfcrec_result/${RUN_STAMP}_${EXPERIMENT_ID}_seed${SEED}_workers${NUM_WORKERS}_${NEGATIVE_SAMPLING_MODE}_${CCFCREC_DEVICE}_${EPOCH}epoch"
export RESULT_ROOT="${RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}"
mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/status" "${RESULT_ROOT}/protocol"
printf '%s\n' "${RESULT_ROOT}" > "${REPO_ROOT}/m11r4_four_performance_first_latest_result_root.txt"

TRAINING_PROFILE="${RESULT_ROOT}/protocol/m11r4_train_validate_only_profile.csv"
PROFILE_AUDIT="${RESULT_ROOT}/protocol/m11r4_profile_audit.json"
"${PYTHON_BIN}" - "${SOURCE_PROFILE}" "${TRAINING_PROFILE}" "${PROFILE_AUDIT}" <<'PY'
import json
import sys
from pathlib import Path

import pandas as pd

source_path, output_path, audit_path = map(Path, sys.argv[1:])
profile = pd.read_csv(source_path, dtype={"raw_asin": str}, low_memory=False)
required = {
    "raw_asin",
    "split",
    "s_cat_v3",
    "RSP_score",
    "category_neighbor_mismatch_proxy_score",
    "support_tail_proxy_score",
    "m11_target_score",
    "m11r1_full_target_flag",
    "m11r1_full_target_loss_score",
}
forbidden = {
    "hr@5",
    "hr@10",
    "hr@20",
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "baseline_hr@20",
    "baseline_ndcg@20",
    "baseline_margin_proxy",
    "baseline_best_target_rank",
    "best_target_rank",
    "eval_baseline_hard_flag",
    "delta_hr@20",
    "delta_ndcg@20",
}
normalized_columns = {str(column).strip().lower() for column in profile.columns}
missing = sorted(required - set(profile.columns))
present_forbidden = sorted(forbidden & normalized_columns)
if missing:
    raise SystemExit(f"M11-R4 profile missing required recommendation-time columns: {missing}")
if present_forbidden:
    raise SystemExit(f"M11-R4 profile contains forbidden evaluation-result columns: {present_forbidden}")
if profile["raw_asin"].isna().any() or profile["raw_asin"].duplicated().any():
    raise SystemExit("M11-R4 profile raw_asin values must be non-null and unique")

work = profile.loc[profile["split"].astype(str).isin(["train", "validate"])].copy()
split_counts = work["split"].astype(str).value_counts().to_dict()
if set(split_counts) != {"train", "validate"}:
    raise SystemExit(f"M11-R4 training profile must contain train and validate only: {split_counts}")
score = pd.to_numeric(work["m11_target_score"], errors="coerce")
if score.isna().any() or not score.between(0.0, 1.0).all():
    raise SystemExit("M11-R4 m11_target_score must be finite and in [0,1]")
target_flag = work["m11r1_full_target_flag"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
target_counts = target_flag.groupby(work["split"].astype(str)).sum().astype(int).to_dict()
if any(count <= 0 for count in target_counts.values()):
    raise SystemExit(f"M11-R4 target identity must be present in both retained splits: {target_counts}")

work.to_csv(output_path, index=False)
audit = {
    "source_profile": str(source_path),
    "training_profile": str(output_path),
    "source_row_count": int(len(profile)),
    "training_profile_row_count": int(len(work)),
    "retained_split_counts": {str(key): int(value) for key, value in split_counts.items()},
    "retained_target_counts": {str(key): int(value) for key, value in target_counts.items()},
    "test_rows_passed_to_training": 0,
    "validation_item_outcomes_passed_to_training": False,
    "test_item_outcomes_read_or_generated": False,
    "forbidden_evaluation_columns": sorted(forbidden),
    "present_forbidden_evaluation_columns": present_forbidden,
}
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
PY

STATUS_FILE="${RESULT_ROOT}/status.tsv"
MASTER_LOG="${RESULT_ROOT}/logs/master.log"
TOTAL_RUNS=4

{
  printf 'EXPERIMENT_ID=%s\n' "${EXPERIMENT_ID}"
  printf 'EVIDENCE_CLASSIFICATION=development_validation_exploration\n'
  printf 'FORECAST_TYPE=mechanism_scenario_not_observed_result\n'
  printf 'FORECAST_E1_OVERALL_NDCG_PCT=3.248\n'
  printf 'FORECAST_E2_OVERALL_NDCG_PCT=3.183\n'
  printf 'FORECAST_E3_OVERALL_NDCG_PCT=3.133\n'
  printf 'FORECAST_E4_OVERALL_NDCG_PCT=3.254\n'
  printf 'TARGET_DEFINITION=unchanged_m11_high_acat_low_rsp_neighbor_support_identity_and_score\n'
  printf 'CONTINUOUS_EXTENSION_SOURCE=m11_target_score_recommendation_time_structural_signal\n'
  printf 'TRAINING_PROFILE_EXCLUDES_TEST_ROWS=true\n'
  printf 'TRAINING_INPUT_USES_VALIDATION_ITEM_METRICS=false\n'
  printf 'TRAINING_INPUT_USES_TEST_ITEM_METRICS=false\n'
  printf 'TEST_METRICS_READ_OR_GENERATED=false\n'
  printf 'RUN_COUNT=%s\n' "${TOTAL_RUNS}"
  printf 'EPOCHS_PER_RUN=%s\n' "${EPOCH}"
  printf 'METHOD_RUNS=M11R4E1_protected_experts M11R4E2_continuous_fusion M11R4E3_relational_alignment M11R4E4_continuous_focal\n'
  printf 'SOURCE_PROFILE=%s\n' "${SOURCE_PROFILE}"
  printf 'TRAINING_PROFILE=%s\n' "${TRAINING_PROFILE}"
  printf 'PROFILE_AUDIT=%s\n' "${PROFILE_AUDIT}"
  printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
  printf 'M11_FEATURE_DIM=%s\n' "${M11_FEATURE_DIM}"
  printf 'M11R4_EXPERT_FILM_STRENGTH=%s\n' "${M11R4_EXPERT_FILM_STRENGTH}"
  printf 'M11R4_FUSION_STRENGTH=%s\n' "${M11R4_FUSION_STRENGTH}"
  printf 'M11R4_RELATION_LOSS_WEIGHT=%s\n' "${M11R4_RELATION_LOSS_WEIGHT}"
  printf 'M11R4_FOCAL_ALPHA=%s\n' "${M11R4_FOCAL_ALPHA}"
  printf 'M11R4_FOCAL_GAMMA=%s\n' "${M11R4_FOCAL_GAMMA}"
  printf 'M11R4_FOCAL_TEMPERATURE=%s\n' "${M11R4_FOCAL_TEMPERATURE}"
  printf 'M11R4_FOCAL_FLOOR=%s\n' "${M11R4_FOCAL_FLOOR}"
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
    --task4_profile_path "${TRAINING_PROFILE}"
    --m11r2_feature_dim "${M11_FEATURE_DIM}"
    --m11r4_expert_film_strength "${M11R4_EXPERT_FILM_STRENGTH}"
    --m11r4_fusion_strength "${M11R4_FUSION_STRENGTH}"
    --m11r4_relation_loss_weight "${M11R4_RELATION_LOSS_WEIGHT}"
    --m11r4_focal_alpha "${M11R4_FOCAL_ALPHA}"
    --m11r4_focal_gamma "${M11R4_FOCAL_GAMMA}"
    --m11r4_focal_temperature "${M11R4_FOCAL_TEMPERATURE}"
    --m11r4_focal_floor "${M11R4_FOCAL_FLOOR}"
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

run_one 1 M11R4E1_protected_experts m11r4_protected_experts
run_one 2 M11R4E2_continuous_fusion m11r4_continuous_fusion
run_one 3 M11R4E3_relational_alignment m11r4_relational_alignment
run_one 4 M11R4E4_continuous_focal m11r4_continuous_focal

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN_DONE [${TOTAL_RUNS}/${TOTAL_RUNS}] $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${MASTER_LOG}"
else
  echo "ALL_DONE [${TOTAL_RUNS}/${TOTAL_RUNS}] $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${MASTER_LOG}"
fi
