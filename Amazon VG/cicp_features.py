from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


CICP_FEATURE_WIDTH = 3
CICP_FEATURE_NAMES = (
    "cicp_score",
    "cicp_inverse_score",
    "cicp_mid_band",
)

CICP_FORBIDDEN_EVALUATION_COLUMNS = {
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


def assert_cicp_profile_train_safe(profile: pd.DataFrame) -> None:
    normalized = {str(column).strip().lower() for column in profile.columns}
    forbidden = sorted(normalized & CICP_FORBIDDEN_EVALUATION_COLUMNS)
    if forbidden:
        raise ValueError(f"CICP profile contains forbidden evaluation-result columns: {forbidden}")


def build_cicp_feature_frame(
    profile: pd.DataFrame,
    *,
    reject_evaluation_columns: bool = True,
) -> pd.DataFrame:
    if reject_evaluation_columns:
        assert_cicp_profile_train_safe(profile)
    if "raw_asin" not in profile.columns:
        raise ValueError("CICP profile is missing raw_asin")
    if "cicp_score" not in profile.columns:
        raise ValueError("CICP profile is missing cicp_score")
    if profile["raw_asin"].isna().any() or profile["raw_asin"].duplicated().any():
        raise ValueError("CICP profile raw_asin values must be non-null and unique")

    score = pd.to_numeric(profile["cicp_score"], errors="coerce")
    if score.isna().any() or not np.isfinite(score.to_numpy(dtype=float)).all():
        raise ValueError("CICP profile cicp_score must be finite")
    if not score.between(0.0, 1.0).all():
        raise ValueError("CICP profile cicp_score must be in [0,1]")

    return pd.DataFrame(
        {
            "cicp_score": score,
            "cicp_inverse_score": 1.0 - score,
            "cicp_mid_band": 4.0 * score * (1.0 - score),
        },
        index=profile.index,
    )


def build_cicp_feature_tensor(
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

    features = build_cicp_feature_frame(
        profile,
        reject_evaluation_columns=reject_evaluation_columns,
    )
    profile_items = set(profile["raw_asin"].astype(str))
    missing_items = sorted(str(item) for item in item_serialize_dict if str(item) not in profile_items)
    if missing_items:
        preview = missing_items[:5]
        raise ValueError(
            f"CICP profile is missing {len(missing_items)} requested items; first={preview}"
        )

    tensor = torch.zeros((int(item_number), CICP_FEATURE_WIDTH), dtype=torch.float32)
    assigned = set()
    for raw_asin, values in zip(
        profile["raw_asin"].astype(str),
        features.to_numpy(dtype="float32"),
    ):
        serial_item = item_serialize_dict.get(raw_asin)
        if serial_item is None:
            continue
        serial_item = int(serial_item)
        if serial_item < 0 or serial_item >= int(item_number):
            raise ValueError(f"CICP feature item id out of range: {serial_item}")
        tensor[serial_item] = torch.from_numpy(values)
        assigned.add(serial_item)
    if len(assigned) != len(item_serialize_dict):
        raise ValueError(
            f"CICP feature assignment incomplete: {len(assigned)}/{len(item_serialize_dict)}"
        )
    return tensor


def load_cicp_feature_tensor(
    profile_path: str | Path,
    item_serialize_dict: dict,
    *,
    item_number: int | None = None,
    reject_evaluation_columns: bool = True,
) -> torch.Tensor:
    profile = pd.read_csv(Path(profile_path), dtype={"raw_asin": str}, low_memory=False)
    return build_cicp_feature_tensor(
        profile,
        item_serialize_dict,
        item_number=item_number,
        reject_evaluation_columns=reject_evaluation_columns,
    )
