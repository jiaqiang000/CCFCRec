from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


CICP_MP_SCALAR_FEATURE_NAMES = (
    "mp_raw_predicted_increment",
    "mp_category_semantic_increment_prediction",
    "mp_category_total_increment_prediction",
    "mp_category_attribution_positive_share_prediction",
    "mp_category_attribution_entropy_prediction",
    "mp_fold_prediction_uncertainty",
    "mp_hgb_ridge_disagreement",
)
CICP_MP_DIRECTION_FEATURE_NAMES = tuple(
    f"mp_direction16_{index:02d}" for index in range(16)
)
CICP_MP_FEATURE_NAMES = CICP_MP_SCALAR_FEATURE_NAMES + CICP_MP_DIRECTION_FEATURE_NAMES
CICP_MP_FEATURE_WIDTH = len(CICP_MP_FEATURE_NAMES)
CICP_MP_SCALAR_WIDTH = len(CICP_MP_SCALAR_FEATURE_NAMES)
CICP_MP_DIRECTION_START = CICP_MP_SCALAR_WIDTH

CICP_MP_RAW_INCREMENT_INDEX = 0
CICP_MP_SEMANTIC_INCREMENT_INDEX = 1
CICP_MP_TOTAL_INCREMENT_INDEX = 2
CICP_MP_POSITIVE_SHARE_INDEX = 3
CICP_MP_ATTRIBUTION_ENTROPY_INDEX = 4
CICP_MP_UNCERTAINTY_INDEX = 5
CICP_MP_DISAGREEMENT_INDEX = 6

CICP_MP_FORBIDDEN_EVALUATION_COLUMNS = {
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


def assert_cicp_mp_profile_train_safe(profile: pd.DataFrame) -> None:
    normalized = {str(column).strip().lower() for column in profile.columns}
    forbidden = sorted(normalized & CICP_MP_FORBIDDEN_EVALUATION_COLUMNS)
    if forbidden:
        raise ValueError(
            f"CICP-MP profile contains forbidden evaluation-result columns: {forbidden}"
        )


def build_cicp_mp_feature_frame(
    profile: pd.DataFrame,
    *,
    reject_evaluation_columns: bool = True,
) -> pd.DataFrame:
    if reject_evaluation_columns:
        assert_cicp_mp_profile_train_safe(profile)
    required = {"raw_asin", *CICP_MP_FEATURE_NAMES}
    missing = sorted(required - set(profile.columns))
    if missing:
        raise ValueError(f"CICP-MP profile is missing required columns: {missing}")
    if profile["raw_asin"].isna().any() or profile["raw_asin"].duplicated().any():
        raise ValueError("CICP-MP profile raw_asin values must be non-null and unique")

    features = profile.loc[:, CICP_MP_FEATURE_NAMES].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = features.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("CICP-MP retained features must all be finite")
    standardization = profile.get("cicpmp_standardization")
    is_train_standardized = (
        standardization is not None
        and standardization.astype(str).eq("train_feature_wise_zscore_v1").all()
    )
    if not is_train_standardized:
        # 原始画像中的不确定性和模型分歧必须非负；训练集标准化后允许负值。
        for column in (
            "mp_fold_prediction_uncertainty",
            "mp_hgb_ridge_disagreement",
        ):
            if (features[column] < 0.0).any():
                raise ValueError(f"CICP-MP {column} must be non-negative")
    return features.astype("float32")


def build_cicp_mp_feature_tensor(
    profile: pd.DataFrame,
    item_serialize_dict: dict,
    *,
    item_number: int | None = None,
    reject_evaluation_columns: bool = True,
) -> torch.Tensor:
    if not item_serialize_dict:
        raise ValueError("item_serialize_dict must not be empty")
    if item_number is None:
        item_number = max(int(value) for value in item_serialize_dict.values()) + 1

    features = build_cicp_mp_feature_frame(
        profile,
        reject_evaluation_columns=reject_evaluation_columns,
    )
    profile_items = set(profile["raw_asin"].astype(str))
    missing_items = sorted(
        str(item) for item in item_serialize_dict if str(item) not in profile_items
    )
    if missing_items:
        raise ValueError(
            "CICP-MP profile is missing "
            f"{len(missing_items)} requested items; first={missing_items[:5]}"
        )

    tensor = torch.zeros((int(item_number), CICP_MP_FEATURE_WIDTH), dtype=torch.float32)
    assigned = set()
    for raw_asin, values in zip(
        profile["raw_asin"].astype(str),
        features.to_numpy(dtype=np.float32),
    ):
        serial_item = item_serialize_dict.get(raw_asin)
        if serial_item is None:
            continue
        serial_item = int(serial_item)
        if serial_item < 0 or serial_item >= int(item_number):
            raise ValueError(f"CICP-MP feature item id out of range: {serial_item}")
        tensor[serial_item] = torch.from_numpy(values)
        assigned.add(serial_item)
    if len(assigned) != len(item_serialize_dict):
        raise ValueError(
            f"CICP-MP feature assignment incomplete: {len(assigned)}/{len(item_serialize_dict)}"
        )
    return tensor


def load_cicp_mp_feature_tensor(
    profile_path: str | Path,
    item_serialize_dict: dict,
    *,
    item_number: int | None = None,
    reject_evaluation_columns: bool = True,
) -> torch.Tensor:
    profile = pd.read_csv(Path(profile_path), dtype={"raw_asin": str}, low_memory=False)
    return build_cicp_mp_feature_tensor(
        profile,
        item_serialize_dict,
        item_number=item_number,
        reject_evaluation_columns=reject_evaluation_columns,
    )
