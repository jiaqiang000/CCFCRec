#!/usr/bin/env bash
set -euo pipefail

# CICP-R2 uses six named embedding-generation mechanisms. Only CICP-R2-E1-CDR
# adds an E4-style residual after the first generator layer.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v caffeinate >/dev/null 2>&1 && [[ "${CICPR2_UNDER_CAFFEINATE:-0}" != "1" ]]; then
  export CICPR2_UNDER_CAFFEINATE=1
  exec caffeinate -dimsu "$0" "$@"
fi

EPOCH="${EPOCH:-100}"
SEED="${SEED:-43}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
SAVE_BATCH_TIME="${SAVE_BATCH_TIME:-300}"
NEGATIVE_SAMPLING_MODE="${NEGATIVE_SAMPLING_MODE:-fast_uniform}"
export CCFCREC_DEVICE="${CCFCREC_DEVICE:-mps}"

[[ "${EPOCH}" == "100" ]] || { echo "CICP-R2 requires EPOCH=100" >&2; exit 1; }
[[ "${SEED}" == "43" ]] || { echo "CICP-R2 requires SEED=43" >&2; exit 1; }
[[ "${NUM_WORKERS}" == "8" ]] || { echo "CICP-R2 requires NUM_WORKERS=8" >&2; exit 1; }
[[ "${BATCH_SIZE}" == "1024" ]] || { echo "CICP-R2 requires BATCH_SIZE=1024" >&2; exit 1; }
[[ "${SAVE_BATCH_TIME}" == "300" ]] || { echo "CICP-R2 requires SAVE_BATCH_TIME=300" >&2; exit 1; }
[[ "${NEGATIVE_SAMPLING_MODE}" == "fast_uniform" ]] || {
  echo "CICP-R2 requires NEGATIVE_SAMPLING_MODE=fast_uniform" >&2
  exit 1
}
[[ "${CCFCREC_DEVICE}" == "mps" ]] || {
  echo "This local CICP-R2 launcher requires CCFCREC_DEVICE=mps" >&2
  exit 1
}

DEFAULT_SOURCE_PROFILE="${REPO_ROOT}/../temp_202607_实验文件记录/temp_20260713/2026-07-13 121409 cicp-train-only-signal-audit-v1_1/cicp_item_profile.csv"
SOURCE_PROFILE="${SOURCE_PROFILE:-${DEFAULT_SOURCE_PROFILE}}"
[[ -f "${SOURCE_PROFILE}" ]] || {
  echo "CICP-R2 source profile not found: ${SOURCE_PROFILE}" >&2
  exit 1
}

EXPERIMENT_ID="${EXPERIMENT_ID:-cicpr2_six_embedding_generation_100epoch}"
RUN_STAMP="${RUN_STAMP:-$(date '+%Y-%m-%d_%H%M%S')}"
DRY_RUN="${DRY_RUN:-0}"

CICPR2_RESIDUAL_MAX_RATIO="${CICPR2_RESIDUAL_MAX_RATIO:-0.15}"
CICPR2_INCREMENT_STRENGTH="${CICPR2_INCREMENT_STRENGTH:-0.50}"
CICPR2_CROSS_ATTENTION_STRENGTH="${CICPR2_CROSS_ATTENTION_STRENGTH:-0.50}"
CICPR2_CROSS_ATTENTION_TEMPERATURE="${CICPR2_CROSS_ATTENTION_TEMPERATURE:-0.25}"
CICPR2_DISTILLATION_WEIGHT="${CICPR2_DISTILLATION_WEIGHT:-0.05}"
CICPR2_ORDINAL_WEIGHT="${CICPR2_ORDINAL_WEIGHT:-0.05}"
CICPR2_ORDINAL_MARGIN="${CICPR2_ORDINAL_MARGIN:-0.02}"
CICPR2_CATEGORY_DROPOUT_MAX="${CICPR2_CATEGORY_DROPOUT_MAX:-0.50}"

export PYTHONHASHSEED="${PYTHONHASHSEED:-${SEED}}"
export PYTHONUNBUFFERED=1
export PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/ccfcrec-py3.11/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || {
  echo "CICP-R2 Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
}

DEFAULT_RESULT_ROOT="/Volumes/MyPassport/CCFCRec对比学习思路硬盘/实验记录硬盘/ccfcrec_result/${RUN_STAMP}_${EXPERIMENT_ID}_seed${SEED}_workers${NUM_WORKERS}_${NEGATIVE_SAMPLING_MODE}_${CCFCREC_DEVICE}_${EPOCH}epoch"
export RESULT_ROOT="${RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}"
mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/status" "${RESULT_ROOT}/protocol" "${RESULT_ROOT}/runs"
LATEST_POINTER="${CICPR2_LATEST_POINTER:-${REPO_ROOT}/cicpr2_six_embedding_generation_latest_result_root.txt}"
printf '%s\n' "${RESULT_ROOT}" > "${LATEST_POINTER}"

TRAINING_PROFILE="${RESULT_ROOT}/protocol/cicpr2_train_validate_score_only_profile.csv"
PROFILE_AUDIT="${RESULT_ROOT}/protocol/cicpr2_profile_audit.json"
"${PYTHON_BIN}" - "${SOURCE_PROFILE}" "${TRAINING_PROFILE}" "${PROFILE_AUDIT}" "${DRY_RUN}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

source_path, output_path, audit_path = map(Path, sys.argv[1:4])
dry_run = sys.argv[4] == "1"
profile = pd.read_csv(source_path, dtype={"raw_asin": str}, low_memory=False)
required = {"raw_asin", "split", "cicp_score"}
forbidden = {
    "hr@5", "hr@10", "hr@20", "ndcg@5", "ndcg@10", "ndcg@20",
    "baseline_hr@20", "baseline_ndcg@20", "baseline_margin_proxy",
    "baseline_best_target_rank", "best_target_rank", "eval_baseline_hard_flag",
    "delta_hr@20", "delta_ndcg@20",
}
normalized_columns = {str(column).strip().lower() for column in profile.columns}
missing = sorted(required - set(profile.columns))
present_forbidden = sorted(forbidden & normalized_columns)
if missing:
    raise SystemExit(f"CICP-R2 profile missing required columns: {missing}")
if present_forbidden:
    raise SystemExit(
        f"CICP-R2 profile contains forbidden evaluation-result columns: {present_forbidden}"
    )
if profile["raw_asin"].isna().any() or profile["raw_asin"].duplicated().any():
    raise SystemExit("CICP-R2 profile raw_asin values must be non-null and unique")

split = profile["split"].astype(str)
work = profile.loc[
    split.isin(["train", "validate"]),
    ["raw_asin", "split", "cicp_score"],
].copy()
split_counts = work["split"].astype(str).value_counts().to_dict()
if set(split_counts) != {"train", "validate"}:
    raise SystemExit(f"CICP-R2 profile must retain train and validate only: {split_counts}")
if not dry_run and split_counts != {"train": 24726, "validate": 5298}:
    raise SystemExit(f"CICP-R2 formal profile has unexpected split counts: {split_counts}")
score = pd.to_numeric(work["cicp_score"], errors="coerce")
if score.isna().any() or not np.isfinite(score.to_numpy(dtype=float)).all():
    raise SystemExit("CICP-R2 cicp_score must be finite")
if not score.between(0.0, 1.0).all():
    raise SystemExit("CICP-R2 cicp_score must be in [0,1]")
work["cicp_score"] = score
work.to_csv(output_path, index=False)

audit = {
    "protocol": "cicpr2_six_embedding_generation_v1",
    "source_profile": str(source_path),
    "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "training_profile": str(output_path),
    "source_row_count": int(len(profile)),
    "training_profile_row_count": int(len(work)),
    "retained_columns": list(work.columns),
    "retained_split_counts": {str(key): int(value) for key, value in split_counts.items()},
    "cicp_score_min": float(score.min()),
    "cicp_score_max": float(score.max()),
    "cicp_score_mean": float(score.mean()),
    "independent_cicp_information_dimensions": 1,
    "derived_model_basis_width": 3,
    "test_rows_passed_to_training": 0,
    "validation_item_outcomes_passed_to_training": False,
    "test_item_outcomes_read_or_generated": False,
    "m11_target_columns_passed_to_training": False,
    "forbidden_evaluation_columns": sorted(forbidden),
    "present_forbidden_evaluation_columns": present_forbidden,
    "dry_run": dry_run,
}
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
PY

STATUS_FILE="${RESULT_ROOT}/status.tsv"
MASTER_LOG="${RESULT_ROOT}/logs/master.log"
TOTAL_RUNS=6

{
  printf 'EXPERIMENT_ID=%s\n' "${EXPERIMENT_ID}"
  printf 'PROTOCOL_VERSION=cicpr2_six_embedding_generation_v1\n'
  printf 'EVIDENCE_CLASSIFICATION=development_validation_exploration\n'
  printf 'FROZEN_SIGNAL=cicp_score\n'
  printf 'FROZEN_SIGNAL_INDEPENDENT_DIMENSIONS=1\n'
  printf 'CICP_SCORE_IS_CALIBRATED_PROBABILITY=false\n'
  printf 'E4_STYLE_RESIDUAL_BRANCH_COUNT=1\n'
  printf 'E4_STYLE_RESIDUAL_BRANCH=CICP-R2-E1-CDR\n'
  printf 'TRAINING_PROFILE_EXCLUDES_TEST_ROWS=true\n'
  printf 'TRAINING_INPUT_USES_VALIDATION_ITEM_METRICS=false\n'
  printf 'TRAINING_INPUT_USES_TEST_ITEM_METRICS=false\n'
  printf 'TEST_METRICS_READ_OR_GENERATED=false\n'
  printf 'RUN_COUNT=%s\n' "${TOTAL_RUNS}"
  printf 'EPOCHS_PER_RUN=%s\n' "${EPOCH}"
  printf 'METHOD_RUNS=CICP-R2-E1-CDR CICP-R2-E2-CID CICP-R2-E3-CMA CICP-R2-E4-SD CICP-R2-E5-OCS CICP-R2-E6-RCD\n'
  printf 'SOURCE_PROFILE=%s\n' "${SOURCE_PROFILE}"
  printf 'TRAINING_PROFILE=%s\n' "${TRAINING_PROFILE}"
  printf 'PROFILE_AUDIT=%s\n' "${PROFILE_AUDIT}"
  printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
  printf 'CICPR2_RESIDUAL_MAX_RATIO=%s\n' "${CICPR2_RESIDUAL_MAX_RATIO}"
  printf 'CICPR2_INCREMENT_STRENGTH=%s\n' "${CICPR2_INCREMENT_STRENGTH}"
  printf 'CICPR2_CROSS_ATTENTION_STRENGTH=%s\n' "${CICPR2_CROSS_ATTENTION_STRENGTH}"
  printf 'CICPR2_CROSS_ATTENTION_TEMPERATURE=%s\n' "${CICPR2_CROSS_ATTENTION_TEMPERATURE}"
  printf 'CICPR2_DISTILLATION_WEIGHT=%s\n' "${CICPR2_DISTILLATION_WEIGHT}"
  printf 'CICPR2_ORDINAL_WEIGHT=%s\n' "${CICPR2_ORDINAL_WEIGHT}"
  printf 'CICPR2_ORDINAL_MARGIN=%s\n' "${CICPR2_ORDINAL_MARGIN}"
  printf 'CICPR2_CATEGORY_DROPOUT_MAX=%s\n' "${CICPR2_CATEGORY_DROPOUT_MAX}"
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
tee -a "${MASTER_LOG}" < "${RESULT_ROOT}/launcher_manifest.env"

record_status() {
  local run_index="$1" run_label="$2" variant="$3" state="$4" started_at="$5" ended_at="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_index}" "${run_label}" "${variant}" "${state}" "${started_at}" "${ended_at}" \
    >> "${STATUS_FILE}"
}

run_one() {
  local run_index="$1" run_label="$2" variant="$3"
  local log_file="${RESULT_ROOT}/logs/${run_index}_${run_label}_${variant}.log"
  local done_file="${RESULT_ROOT}/status/${run_index}_${run_label}.done"
  local branch_root="${RESULT_ROOT}/runs/${run_index}_${run_label}"
  local started_at
  started_at="$(date '+%Y-%m-%d %H:%M:%S')"
  mkdir -p "${branch_root}"

  if [[ -f "${done_file}" ]]; then
    echo "SKIP [${run_index}/${TOTAL_RUNS}] ${run_label} already completed" | tee -a "${MASTER_LOG}"
    record_status "${run_index}" "${run_label}" "${variant}" "skipped_completed" "${started_at}" "${started_at}"
    return
  fi

  local command=(
    bash scripts/train_amazon_vg_cuda.sh
    --method_variant "${variant}"
    --cicp_profile_path "${TRAINING_PROFILE}"
    --cicpr2_residual_max_ratio "${CICPR2_RESIDUAL_MAX_RATIO}"
    --cicpr2_increment_strength "${CICPR2_INCREMENT_STRENGTH}"
    --cicpr2_cross_attention_strength "${CICPR2_CROSS_ATTENTION_STRENGTH}"
    --cicpr2_cross_attention_temperature "${CICPR2_CROSS_ATTENTION_TEMPERATURE}"
    --cicpr2_distillation_weight "${CICPR2_DISTILLATION_WEIGHT}"
    --cicpr2_ordinal_weight "${CICPR2_ORDINAL_WEIGHT}"
    --cicpr2_ordinal_margin "${CICPR2_ORDINAL_MARGIN}"
    --cicpr2_category_dropout_max "${CICPR2_CATEGORY_DROPOUT_MAX}"
    --epoch "${EPOCH}"
    --num_workers "${NUM_WORKERS}"
    --batch_size "${BATCH_SIZE}"
    --negative_sampling_mode "${NEGATIVE_SAMPLING_MODE}"
    --save_batch_time "${SAVE_BATCH_TIME}"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DRY_RUN [${run_index}/${TOTAL_RUNS}] ${run_label} ${variant} ${started_at}" | tee -a "${MASTER_LOG}"
    printf 'RESULT_ROOT=%q ' "${branch_root}" | tee "${log_file}"
    printf '%q ' "${command[@]}" | tee -a "${log_file}"
    printf '\n' | tee -a "${log_file}"
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

  local ended_at
  ended_at="$(date '+%Y-%m-%d %H:%M:%S')"
  touch "${done_file}"
  record_status "${run_index}" "${run_label}" "${variant}" "completed" "${started_at}" "${ended_at}"
  echo "END [${run_index}/${TOTAL_RUNS}] ${run_label} ${variant} ${ended_at}" | tee -a "${MASTER_LOG}"
}

run_one 1 CICP-R2-E1-CDR cicpr2_content_direction_residual
run_one 2 CICP-R2-E2-CID cicpr2_category_increment_gate
run_one 3 CICP-R2-E3-CMA cicpr2_cross_modal_attention
run_one 4 CICP-R2-E4-SD cicpr2_score_distillation
run_one 5 CICP-R2-E5-OCS cicpr2_ordinal_counterfactual
run_one 6 CICP-R2-E6-RCD cicpr2_reliability_dropout

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN_DONE [${TOTAL_RUNS}/${TOTAL_RUNS}] $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${MASTER_LOG}"
else
  echo "ALL_DONE [${TOTAL_RUNS}/${TOTAL_RUNS}] $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${MASTER_LOG}"
fi
