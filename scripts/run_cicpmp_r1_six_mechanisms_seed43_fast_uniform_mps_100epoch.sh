#!/usr/bin/env bash
set -euo pipefail

# CICP-MP-R1 evaluates six mechanisms over the frozen 23D CICP-MP-v1 profile.
# Only E1 maps the profile to an E4-style post-generator hidden residual.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if command -v caffeinate >/dev/null 2>&1 && [[ "${CICPMP_R1_UNDER_CAFFEINATE:-0}" != "1" ]]; then
  export CICPMP_R1_UNDER_CAFFEINATE=1
  exec caffeinate -dimsu "$0" "$@"
fi

EPOCH="${EPOCH:-100}"
SEED="${SEED:-43}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
SAVE_BATCH_TIME="${SAVE_BATCH_TIME:-300}"
NEGATIVE_SAMPLING_MODE="${NEGATIVE_SAMPLING_MODE:-fast_uniform}"
export CCFCREC_DEVICE="${CCFCREC_DEVICE:-mps}"

[[ "${EPOCH}" == "100" ]] || { echo "CICP-MP-R1 requires EPOCH=100" >&2; exit 1; }
[[ "${SEED}" == "43" ]] || { echo "CICP-MP-R1 requires SEED=43" >&2; exit 1; }
[[ "${NUM_WORKERS}" == "8" ]] || { echo "CICP-MP-R1 requires NUM_WORKERS=8" >&2; exit 1; }
[[ "${BATCH_SIZE}" == "1024" ]] || { echo "CICP-MP-R1 requires BATCH_SIZE=1024" >&2; exit 1; }
[[ "${SAVE_BATCH_TIME}" == "300" ]] || { echo "CICP-MP-R1 requires SAVE_BATCH_TIME=300" >&2; exit 1; }
[[ "${NEGATIVE_SAMPLING_MODE}" == "fast_uniform" ]] || {
  echo "CICP-MP-R1 requires NEGATIVE_SAMPLING_MODE=fast_uniform" >&2
  exit 1
}
[[ "${CCFCREC_DEVICE}" == "mps" ]] || {
  echo "This local CICP-MP-R1 launcher requires CCFCREC_DEVICE=mps" >&2
  exit 1
}

DEFAULT_SOURCE_PROFILE="${REPO_ROOT}/../temp_202607_实验文件记录/temp_20260717/2026-07-17 120710 cicp-mp-v1-train-only-offline-audit/cicp_mp_v1_item_profile.csv"
SOURCE_PROFILE="${SOURCE_PROFILE:-${DEFAULT_SOURCE_PROFILE}}"
[[ -f "${SOURCE_PROFILE}" ]] || {
  echo "CICP-MP-R1 source profile not found: ${SOURCE_PROFILE}" >&2
  exit 1
}

EXPERIMENT_ID="${EXPERIMENT_ID:-cicpmp_r1_six_mechanisms_100epoch}"
RUN_STAMP="${RUN_STAMP:-$(date '+%Y-%m-%d_%H%M%S')}"
DRY_RUN="${DRY_RUN:-0}"

CICPMP_HIDDEN_DIM="${CICPMP_HIDDEN_DIM:-32}"
CICPMP_RESIDUAL_MAX_RATIO="${CICPMP_RESIDUAL_MAX_RATIO:-0.15}"
CICPMP_RELIABILITY_SCALE="${CICPMP_RELIABILITY_SCALE:-50.0}"
CICPMP_DIRECTION_WEIGHT="${CICPMP_DIRECTION_WEIGHT:-0.05}"
CICPMP_ENTROPY_WEIGHT="${CICPMP_ENTROPY_WEIGHT:-0.02}"
CICPMP_EXPERT_STRENGTH="${CICPMP_EXPERT_STRENGTH:-0.20}"
CICPMP_COUNTERFACTUAL_WEIGHT="${CICPMP_COUNTERFACTUAL_WEIGHT:-0.05}"
CICPMP_HARD_NEGATIVE_STRENGTH="${CICPMP_HARD_NEGATIVE_STRENGTH:-0.50}"

[[ "${CICPMP_HIDDEN_DIM}" == "32" ]] || { echo "CICP-MP-R1 requires CICPMP_HIDDEN_DIM=32" >&2; exit 1; }
[[ "${CICPMP_RESIDUAL_MAX_RATIO}" == "0.15" ]] || { echo "CICP-MP-R1 requires CICPMP_RESIDUAL_MAX_RATIO=0.15" >&2; exit 1; }
[[ "${CICPMP_RELIABILITY_SCALE}" == "50.0" ]] || { echo "CICP-MP-R1 requires CICPMP_RELIABILITY_SCALE=50.0" >&2; exit 1; }
[[ "${CICPMP_DIRECTION_WEIGHT}" == "0.05" ]] || { echo "CICP-MP-R1 requires CICPMP_DIRECTION_WEIGHT=0.05" >&2; exit 1; }
[[ "${CICPMP_ENTROPY_WEIGHT}" == "0.02" ]] || { echo "CICP-MP-R1 requires CICPMP_ENTROPY_WEIGHT=0.02" >&2; exit 1; }
[[ "${CICPMP_EXPERT_STRENGTH}" == "0.20" ]] || { echo "CICP-MP-R1 requires CICPMP_EXPERT_STRENGTH=0.20" >&2; exit 1; }
[[ "${CICPMP_COUNTERFACTUAL_WEIGHT}" == "0.05" ]] || { echo "CICP-MP-R1 requires CICPMP_COUNTERFACTUAL_WEIGHT=0.05" >&2; exit 1; }
[[ "${CICPMP_HARD_NEGATIVE_STRENGTH}" == "0.50" ]] || { echo "CICP-MP-R1 requires CICPMP_HARD_NEGATIVE_STRENGTH=0.50" >&2; exit 1; }

export PYTHONHASHSEED="${SEED}"
export PYTHONUNBUFFERED=1
export PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/envs/ccfcrec-py3.11/bin/python}"
[[ -x "${PYTHON_BIN}" ]] || {
  echo "CICP-MP-R1 Python executable not found: ${PYTHON_BIN}" >&2
  exit 1
}

DEFAULT_RESULT_ROOT="/Volumes/MyPassport/CCFCRec对比学习思路硬盘/实验记录硬盘/ccfcrec_result/${RUN_STAMP}_${EXPERIMENT_ID}_seed${SEED}_workers${NUM_WORKERS}_${NEGATIVE_SAMPLING_MODE}_${CCFCREC_DEVICE}_${EPOCH}epoch"
export RESULT_ROOT="${RESULT_ROOT:-${DEFAULT_RESULT_ROOT}}"
mkdir -p "${RESULT_ROOT}/logs" "${RESULT_ROOT}/status" "${RESULT_ROOT}/protocol" "${RESULT_ROOT}/runs"
LATEST_POINTER="${CICPMP_R1_LATEST_POINTER:-${REPO_ROOT}/cicpmp_r1_six_mechanisms_latest_result_root.txt}"
printf '%s\n' "${RESULT_ROOT}" > "${LATEST_POINTER}"

TRAINING_PROFILE="${RESULT_ROOT}/protocol/cicpmp_v1_train_validate_23d_profile.csv"
PROFILE_AUDIT="${RESULT_ROOT}/protocol/cicpmp_v1_profile_audit.json"
"${PYTHON_BIN}" - "${SOURCE_PROFILE}" "${TRAINING_PROFILE}" "${PROFILE_AUDIT}" "${DRY_RUN}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

source_path, output_path, audit_path = map(Path, sys.argv[1:4])
dry_run = sys.argv[4] == "1"
direction = [f"mp_direction16_{index:02d}" for index in range(16)]
features = [
    "mp_raw_predicted_increment",
    "mp_category_semantic_increment_prediction",
    "mp_category_total_increment_prediction",
    "mp_category_attribution_positive_share_prediction",
    "mp_category_attribution_entropy_prediction",
    "mp_fold_prediction_uncertainty",
    "mp_hgb_ridge_disagreement",
    *direction,
]
required = {"raw_asin", "split", *features}
forbidden = {
    "hr@5", "hr@10", "hr@20", "ndcg@5", "ndcg@10", "ndcg@20",
    "baseline_hr@20", "baseline_ndcg@20", "baseline_margin_proxy",
    "baseline_best_target_rank", "best_target_rank", "eval_baseline_hard_flag",
    "delta_hr@20", "delta_ndcg@20",
}
profile = pd.read_csv(source_path, dtype={"raw_asin": str}, low_memory=False)
normalized_columns = {str(column).strip().lower() for column in profile.columns}
missing = sorted(required - set(profile.columns))
present_forbidden = sorted(forbidden & normalized_columns)
if missing:
    raise SystemExit(f"CICP-MP-R1 profile missing required columns: {missing}")
if present_forbidden:
    raise SystemExit(
        "CICP-MP-R1 profile contains forbidden evaluation-result columns: "
        f"{present_forbidden}"
    )
if profile["raw_asin"].isna().any() or profile["raw_asin"].duplicated().any():
    raise SystemExit("CICP-MP-R1 raw_asin values must be non-null and unique")

work = profile.loc[
    profile["split"].astype(str).isin(["train", "validate"]),
    ["raw_asin", "split", *features],
].copy()
split_counts = work["split"].astype(str).value_counts().to_dict()
if set(split_counts) != {"train", "validate"}:
    raise SystemExit(
        f"CICP-MP-R1 profile must retain train and validate only: {split_counts}"
    )
if not dry_run and split_counts != {"train": 24726, "validate": 5298}:
    raise SystemExit(
        f"CICP-MP-R1 formal profile has unexpected split counts: {split_counts}"
    )
numeric = work[features].apply(pd.to_numeric, errors="coerce")
if not np.isfinite(numeric.to_numpy(dtype=float)).all():
    raise SystemExit("CICP-MP-R1 retained 23D features must all be finite")
for column in ["mp_fold_prediction_uncertainty", "mp_hgb_ridge_disagreement"]:
    if (numeric[column] < 0.0).any():
        raise SystemExit(f"CICP-MP-R1 {column} must be non-negative")
work.loc[:, features] = numeric
work.to_csv(output_path, index=False)

audit = {
    "protocol": "cicpmp_r1_six_mechanisms_v1",
    "source_profile": str(source_path),
    "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
    "training_profile": str(output_path),
    "source_row_count": int(len(profile)),
    "training_profile_row_count": int(len(work)),
    "retained_columns": list(work.columns),
    "retained_feature_count": len(features),
    "retained_scalar_count": 7,
    "retained_direction_count": 16,
    "retained_split_counts": {str(key): int(value) for key, value in split_counts.items()},
    "test_rows_passed_to_training": 0,
    "validation_item_outcomes_passed_to_training": False,
    "test_item_outcomes_read_or_generated": False,
    "m11_target_columns_passed_to_training": False,
    "present_forbidden_evaluation_columns": present_forbidden,
    "all_retained_features_finite": True,
    "attribution_prediction_ranges": {
        column: {
            "min": float(numeric[column].min()),
            "max": float(numeric[column].max()),
            "below_zero_count": int((numeric[column] < 0.0).sum()),
            "above_one_count": int((numeric[column] > 1.0).sum()),
            "usage_policy": "preserve_in_profile_clip_at_mechanism_use",
        }
        for column in [
            "mp_category_attribution_positive_share_prediction",
            "mp_category_attribution_entropy_prediction",
        ]
    },
    "dry_run": dry_run,
}
audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
PY

INITIALIZATION_AUDIT="${RESULT_ROOT}/protocol/cicpmp_r1_common_initialization_audit.json"
ACTUAL_COMMON_HASH_FILE="${RESULT_ROOT}/protocol/cicpmp_r1_actual_common_parameter_sha256.txt"
ACTUAL_RUN_AUDIT="${RESULT_ROOT}/protocol/cicpmp_r1_actual_run_config_audit.tsv"
"${PYTHON_BIN}" - "${INITIALIZATION_AUDIT}" "${SEED}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

repo_root = Path.cwd()
sys.path.insert(0, str(repo_root / "Amazon VG"))
from model import CCFCRec, CCFCREC_COMMON_PARAMETER_NAMES, ccfcrec_common_parameter_sha256

output_path = Path(sys.argv[1])
seed = int(sys.argv[2])
variants = [
    "cicpmp_r1_reliable_residual",
    "cicpmp_r1_direction_alignment",
    "cicpmp_r1_attention_entropy",
    "cicpmp_r1_reliable_expert",
    "cicpmp_r1_counterfactual_calibration",
    "cicpmp_r1_direction_hard_negative",
]

def args_for(variant):
    return SimpleNamespace(
        method_variant=variant,
        attr_num=8,
        attr_present_dim=4,
        implicit_dim=4,
        cat_implicit_dim=4,
        user_number=7,
        item_number=5,
        pretrain=False,
        seed=seed,
        cicpmp_hidden_dim=6,
        cicpmp_residual_max_ratio=0.15,
        cicpmp_expert_strength=0.20,
    )

hashes = {}
post_init_rng_hashes = {}
for variant in variants:
    torch.manual_seed(seed)
    model = CCFCRec(args_for(variant))
    hashes[variant] = ccfcrec_common_parameter_sha256(model)
    post_init_rng_hashes[variant] = hashlib.sha256(
        torch.get_rng_state().numpy().tobytes()
    ).hexdigest()

if len(set(hashes.values())) != 1:
    raise SystemExit(f"CICP-MP-R1 common initialization hash mismatch: {hashes}")
if len(set(post_init_rng_hashes.values())) != 1:
    raise SystemExit(
        f"CICP-MP-R1 post-initialization random stream mismatch: {post_init_rng_hashes}"
    )
audit = {
    "protocol": "cicpmp_r1_common_initialization_v1",
    "seed": seed,
    "variants": variants,
    "common_parameter_names": list(CCFCREC_COMMON_PARAMETER_NAMES),
    "common_parameter_sha256_by_variant": hashes,
    "common_hash_unique_count": len(set(hashes.values())),
    "post_initialization_rng_sha256_by_variant": post_init_rng_hashes,
    "post_initialization_rng_hash_unique_count": len(set(post_init_rng_hashes.values())),
    "common_parameters_elementwise_equal": True,
    "method_specific_modules_use_isolated_rng": True,
}
output_path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
PY

STATUS_FILE="${RESULT_ROOT}/status.tsv"
MASTER_LOG="${RESULT_ROOT}/logs/master.log"
TOTAL_RUNS=6

{
  printf 'EXPERIMENT_ID=%s\n' "${EXPERIMENT_ID}"
  printf 'EXPERIMENT_DESIGN_NOTE=2026-07-18 014651 CCFCRec Amazon-VG CICP-MP-R1机制空间审查与六机制正式实验设计.md\n'
  printf 'PROTOCOL_VERSION=cicpmp_r1_six_mechanisms_v1\n'
  printf 'EVIDENCE_CLASSIFICATION=development_validation_exploration\n'
  printf 'FROZEN_SIGNAL=CICP-MP-v1_23D\n'
  printf 'FROZEN_SIGNAL_DIMENSIONS=23\n'
  printf 'FROZEN_SCALAR_DIMENSIONS=7\n'
  printf 'FROZEN_DIRECTION_DIMENSIONS=16\n'
  printf 'E4_STYLE_RESIDUAL_BRANCH_COUNT=1\n'
  printf 'E4_STYLE_RESIDUAL_BRANCH=CICP-MP-R1-E1-RRA\n'
  printf 'COMMON_INITIALIZATION_AUDIT=%s\n' "${INITIALIZATION_AUDIT}"
  printf 'PREFLIGHT_COMMON_PARAMETER_HASH_UNIQUE_COUNT=1\n'
  printf 'PREFLIGHT_POST_INITIALIZATION_RNG_HASH_UNIQUE_COUNT=1\n'
  printf 'ACTUAL_RUN_CONFIG_AUDIT=%s\n' "${ACTUAL_RUN_AUDIT}"
  printf 'TRAINING_PROFILE_EXCLUDES_TEST_ROWS=true\n'
  printf 'TRAINING_INPUT_USES_VALIDATION_ITEM_METRICS=false\n'
  printf 'TRAINING_INPUT_USES_TEST_ITEM_METRICS=false\n'
  printf 'TEST_METRICS_READ_OR_GENERATED=false\n'
  printf 'RUN_COUNT=%s\n' "${TOTAL_RUNS}"
  printf 'EPOCHS_PER_RUN=%s\n' "${EPOCH}"
  printf 'METHOD_RUNS=CICP-MP-R1-E1-RRA CICP-MP-R1-E2-DTA CICP-MP-R1-E3-AEC CICP-MP-R1-E4-RCE CICP-MP-R1-E5-CCI CICP-MP-R1-E6-DHN\n'
  printf 'SOURCE_PROFILE=%s\n' "${SOURCE_PROFILE}"
  printf 'TRAINING_PROFILE=%s\n' "${TRAINING_PROFILE}"
  printf 'PROFILE_AUDIT=%s\n' "${PROFILE_AUDIT}"
  printf 'RESULT_ROOT=%s\n' "${RESULT_ROOT}"
  printf 'CICPMP_HIDDEN_DIM=%s\n' "${CICPMP_HIDDEN_DIM}"
  printf 'CICPMP_RESIDUAL_MAX_RATIO=%s\n' "${CICPMP_RESIDUAL_MAX_RATIO}"
  printf 'CICPMP_RELIABILITY_SCALE=%s\n' "${CICPMP_RELIABILITY_SCALE}"
  printf 'CICPMP_DIRECTION_WEIGHT=%s\n' "${CICPMP_DIRECTION_WEIGHT}"
  printf 'CICPMP_ENTROPY_WEIGHT=%s\n' "${CICPMP_ENTROPY_WEIGHT}"
  printf 'CICPMP_EXPERT_STRENGTH=%s\n' "${CICPMP_EXPERT_STRENGTH}"
  printf 'CICPMP_COUNTERFACTUAL_WEIGHT=%s\n' "${CICPMP_COUNTERFACTUAL_WEIGHT}"
  printf 'CICPMP_HARD_NEGATIVE_STRENGTH=%s\n' "${CICPMP_HARD_NEGATIVE_STRENGTH}"
  printf 'PYTHON_BIN=%s\n' "${PYTHON_BIN}"
  printf 'CCFCREC_DEVICE=%s\n' "${CCFCREC_DEVICE}"
  printf 'SEED=%s\n' "${SEED}"
  printf 'NUM_WORKERS=%s\n' "${NUM_WORKERS}"
  printf 'BATCH_SIZE=%s\n' "${BATCH_SIZE}"
  printf 'SAVE_BATCH_TIME=%s\n' "${SAVE_BATCH_TIME}"
  printf 'NEGATIVE_SAMPLING_MODE=%s\n' "${NEGATIVE_SAMPLING_MODE}"
  printf 'DRY_RUN=%s\n' "${DRY_RUN}"
} > "${RESULT_ROOT}/launcher_manifest.env"

if [[ ! -f "${ACTUAL_RUN_AUDIT}" ]]; then
  printf 'run_index\trun_label\tmethod_variant\tcommon_parameter_sha256\tcommon_parameters\tmethod_specific_parameters\ttotal_parameters\trun_config\n' \
    > "${ACTUAL_RUN_AUDIT}"
fi
awk -F '\t' 'NR > 1 {
  printf "RUN_%s_ACTUAL_COMMON_PARAMETER_SHA256=%s\n", $1, $4
  printf "RUN_%s_COMMON_PARAMETER_COUNT=%s\n", $1, $5
  printf "RUN_%s_METHOD_SPECIFIC_PARAMETER_COUNT=%s\n", $1, $6
  printf "RUN_%s_TOTAL_PARAMETER_COUNT=%s\n", $1, $7
}' "${ACTUAL_RUN_AUDIT}" >> "${RESULT_ROOT}/launcher_manifest.env"

printf 'run_index\trun_label\tmethod_variant\tstate\tstarted_at\tended_at\n' > "${STATUS_FILE}"
tee -a "${MASTER_LOG}" < "${RESULT_ROOT}/launcher_manifest.env"

record_status() {
  local run_index="$1" run_label="$2" variant="$3" state="$4" started_at="$5" ended_at="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_index}" "${run_label}" "${variant}" "${state}" "${started_at}" "${ended_at}" \
    >> "${STATUS_FILE}"
}

verify_actual_run_config() {
  local run_index="$1" run_label="$2" variant="$3" branch_root="$4"
  local run_config audit_fields actual_hash common_count method_count total_count expected_hash
  run_config="$(find "${branch_root}" -type f -name run_config.json -print | sort | tail -n 1)"
  [[ -n "${run_config}" && -f "${run_config}" ]] || {
    echo "CICP-MP-R1 actual run_config.json not found for ${run_label}" >&2
    return 1
  }

  if ! audit_fields="$("${PYTHON_BIN}" - "${run_config}" "${variant}" "${SEED}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
expected_variant = sys.argv[2]
expected_seed = int(sys.argv[3])
config = json.loads(config_path.read_text(encoding="utf-8"))
if config.get("method_variant") != expected_variant:
    raise SystemExit(
        f"method_variant mismatch in {config_path}: "
        f"{config.get('method_variant')!r} != {expected_variant!r}"
    )
if int(config.get("seed", -1)) != expected_seed:
    raise SystemExit(
        f"seed mismatch in {config_path}: {config.get('seed')!r} != {expected_seed}"
    )
if config.get("training_input_uses_validation_item_metrics") is not False:
    raise SystemExit(f"validation item metrics safety flag failed in {config_path}")
if config.get("training_input_uses_test_item_metrics") is not False:
    raise SystemExit(f"test item metrics safety flag failed in {config_path}")
digest = str(config.get("ccfcrec_common_parameter_sha256", ""))
if len(digest) != 64:
    raise SystemExit(f"invalid common parameter SHA-256 in {config_path}: {digest!r}")
counts = config.get("parameter_count", {})
required_counts = ("common", "method_specific", "total")
missing = [key for key in required_counts if key not in counts]
if missing:
    raise SystemExit(f"parameter count audit missing {missing} in {config_path}")
if int(counts["common"]) + int(counts["method_specific"]) != int(counts["total"]):
    raise SystemExit(f"parameter count audit is inconsistent in {config_path}")
print(
    digest,
    int(counts["common"]),
    int(counts["method_specific"]),
    int(counts["total"]),
    sep="\t",
)
PY
  )"; then
    return 1
  fi
  IFS=$'\t' read -r actual_hash common_count method_count total_count <<< "${audit_fields}"
  [[ "${actual_hash}" =~ ^[0-9a-f]{64}$ ]] || {
    echo "CICP-MP-R1 failed to parse actual common parameter SHA-256 for ${run_label}" >&2
    return 1
  }

  if [[ -f "${ACTUAL_COMMON_HASH_FILE}" ]]; then
    expected_hash="$(sed -n '1p' "${ACTUAL_COMMON_HASH_FILE}")"
    [[ "${actual_hash}" == "${expected_hash}" ]] || {
      echo "CICP-MP-R1 actual common initialization mismatch: ${run_label} ${actual_hash} != ${expected_hash}" >&2
      return 1
    }
  else
    printf '%s\n' "${actual_hash}" > "${ACTUAL_COMMON_HASH_FILE}"
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${run_index}" "${run_label}" "${variant}" "${actual_hash}" \
    "${common_count}" "${method_count}" "${total_count}" "${run_config}" \
    >> "${ACTUAL_RUN_AUDIT}"
  {
    printf 'RUN_%s_ACTUAL_COMMON_PARAMETER_SHA256=%s\n' "${run_index}" "${actual_hash}"
    printf 'RUN_%s_COMMON_PARAMETER_COUNT=%s\n' "${run_index}" "${common_count}"
    printf 'RUN_%s_METHOD_SPECIFIC_PARAMETER_COUNT=%s\n' "${run_index}" "${method_count}"
    printf 'RUN_%s_TOTAL_PARAMETER_COUNT=%s\n' "${run_index}" "${total_count}"
  } >> "${RESULT_ROOT}/launcher_manifest.env"
  echo "AUDIT [${run_index}/${TOTAL_RUNS}] ${run_label} actual_common_sha256=${actual_hash}" \
    | tee -a "${MASTER_LOG}"
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
    --cicp_mp_profile_path "${TRAINING_PROFILE}"
    --cicpmp_hidden_dim "${CICPMP_HIDDEN_DIM}"
    --cicpmp_residual_max_ratio "${CICPMP_RESIDUAL_MAX_RATIO}"
    --cicpmp_reliability_scale "${CICPMP_RELIABILITY_SCALE}"
    --cicpmp_direction_weight "${CICPMP_DIRECTION_WEIGHT}"
    --cicpmp_entropy_weight "${CICPMP_ENTROPY_WEIGHT}"
    --cicpmp_expert_strength "${CICPMP_EXPERT_STRENGTH}"
    --cicpmp_counterfactual_weight "${CICPMP_COUNTERFACTUAL_WEIGHT}"
    --cicpmp_hard_negative_strength "${CICPMP_HARD_NEGATIVE_STRENGTH}"
    --epoch "${EPOCH}"
    --seed "${SEED}"
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

  if ! verify_actual_run_config "${run_index}" "${run_label}" "${variant}" "${branch_root}"; then
    local failed_at
    failed_at="$(date '+%Y-%m-%d %H:%M:%S')"
    record_status "${run_index}" "${run_label}" "${variant}" "failed_actual_config_audit" "${started_at}" "${failed_at}"
    echo "FAIL_AUDIT [${run_index}/${TOTAL_RUNS}] ${run_label} ${variant} ${failed_at}" | tee -a "${MASTER_LOG}"
    return 1
  fi

  local ended_at
  ended_at="$(date '+%Y-%m-%d %H:%M:%S')"
  touch "${done_file}"
  record_status "${run_index}" "${run_label}" "${variant}" "completed" "${started_at}" "${ended_at}"
  echo "END [${run_index}/${TOTAL_RUNS}] ${run_label} ${variant} ${ended_at}" | tee -a "${MASTER_LOG}"
}

run_one 1 CICP-MP-R1-E1-RRA cicpmp_r1_reliable_residual
run_one 2 CICP-MP-R1-E2-DTA cicpmp_r1_direction_alignment
run_one 3 CICP-MP-R1-E3-AEC cicpmp_r1_attention_entropy
run_one 4 CICP-MP-R1-E4-RCE cicpmp_r1_reliable_expert
run_one 5 CICP-MP-R1-E5-CCI cicpmp_r1_counterfactual_calibration
run_one 6 CICP-MP-R1-E6-DHN cicpmp_r1_direction_hard_negative

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "DRY_RUN_DONE [${TOTAL_RUNS}/${TOTAL_RUNS}] $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${MASTER_LOG}"
else
  echo "ALL_DONE [${TOTAL_RUNS}/${TOTAL_RUNS}] $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${MASTER_LOG}"
fi
