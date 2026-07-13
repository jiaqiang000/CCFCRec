from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


M11_FEATURE_WIDTH = 6
M11_FEATURE_MODE_TARGET_MASKED = "target_masked"
M11_FEATURE_MODE_FULL_STRUCTURAL = "full_structural"
M11_FEATURE_MODES = {
    M11_FEATURE_MODE_TARGET_MASKED,
    M11_FEATURE_MODE_FULL_STRUCTURAL,
}

M11_FORBIDDEN_EVALUATION_COLUMNS = {
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


def assert_m11_profile_train_safe(profile: pd.DataFrame) -> None:
    normalized = {str(column).strip().lower() for column in profile.columns}
    forbidden = sorted(normalized & M11_FORBIDDEN_EVALUATION_COLUMNS)
    if forbidden:
        raise ValueError(f"M11 profile contains forbidden evaluation-result columns: {forbidden}")


def _bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _numeric_series(profile: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in profile.columns:
        return pd.Series(default, index=profile.index, dtype=float)
    return pd.to_numeric(profile[column], errors="coerce").fillna(default).clip(0.0, 1.0)


def _m11_source_feature_frame(profile: pd.DataFrame, *, prefer_unmasked_score: bool = False) -> pd.DataFrame:
    flag_column = (
        "m11r1_full_target_flag"
        if "m11r1_full_target_flag" in profile.columns
        else "m11_high_acat_low_rsp_neighbor_support_flag"
    )
    if flag_column not in profile.columns:
        raise ValueError("M11 feature profile is missing the full-target flag")

    if prefer_unmasked_score and "m11_target_score" in profile.columns:
        score_column = "m11_target_score"
    else:
        score_column = (
            "m11r1_full_target_loss_score"
            if "m11r1_full_target_loss_score" in profile.columns
            else "m11_target_score"
        )
    if score_column not in profile.columns:
        raise ValueError("M11 feature profile is missing the target score")

    return pd.DataFrame(
        {
            "target_flag": _bool_series(profile[flag_column]).astype(float),
            "target_score": _numeric_series(profile, score_column),
            "acat_score": _numeric_series(profile, "s_cat_v3"),
            "rsp_inverse": 1.0 - _numeric_series(profile, "RSP_score", default=1.0),
            "neighbor_mismatch_score": _numeric_series(profile, "category_neighbor_mismatch_proxy_score"),
            "support_tail_score": _numeric_series(profile, "support_tail_proxy_score"),
        },
        index=profile.index,
    )


def build_m11_feature_frame(
    profile: pd.DataFrame,
    *,
    feature_mode: str = M11_FEATURE_MODE_TARGET_MASKED,
    reject_evaluation_columns: bool = False,
) -> pd.DataFrame:
    if reject_evaluation_columns:
        assert_m11_profile_train_safe(profile)
    if feature_mode not in M11_FEATURE_MODES:
        raise ValueError(f"unsupported M11 feature_mode={feature_mode}")

    source = _m11_source_feature_frame(
        profile,
        prefer_unmasked_score=feature_mode == M11_FEATURE_MODE_FULL_STRUCTURAL,
    )
    if feature_mode == M11_FEATURE_MODE_FULL_STRUCTURAL:
        return source

    flags = source["target_flag"]
    return pd.DataFrame(
        {
            "target_flag": flags,
            "target_score": flags * source["target_score"],
            "acat_score": flags * source["acat_score"],
            "rsp_inverse": flags * source["rsp_inverse"],
            "neighbor_mismatch_score": flags * source["neighbor_mismatch_score"],
            "support_tail_score": flags * source["support_tail_score"],
        },
        index=profile.index,
    )


def build_m11_feature_tensor(
    profile: pd.DataFrame,
    item_serialize_dict: dict,
    *,
    item_number: int | None = None,
    feature_mode: str = M11_FEATURE_MODE_TARGET_MASKED,
    reject_evaluation_columns: bool = False,
) -> torch.Tensor:
    if "raw_asin" not in profile.columns:
        raise ValueError("M11 feature profile is missing raw_asin")
    if not item_serialize_dict:
        raise ValueError("item_serialize_dict must not be empty")
    if item_number is None:
        item_number = max(int(value) for value in item_serialize_dict.values()) + 1

    tensor = torch.zeros((int(item_number), M11_FEATURE_WIDTH), dtype=torch.float32)
    features = build_m11_feature_frame(
        profile,
        feature_mode=feature_mode,
        reject_evaluation_columns=reject_evaluation_columns,
    )
    for raw_asin, values in zip(profile["raw_asin"], features.to_numpy(dtype="float32")):
        serial_item = item_serialize_dict.get(raw_asin)
        if serial_item is None:
            continue
        serial_item = int(serial_item)
        if serial_item < 0 or serial_item >= int(item_number):
            raise ValueError(f"M11 feature item id out of range: {serial_item}")
        tensor[serial_item] = torch.from_numpy(values)
    return tensor


def load_m11_feature_tensor(
    profile_path: str | Path,
    item_serialize_dict: dict,
    *,
    item_number: int | None = None,
    feature_mode: str = M11_FEATURE_MODE_TARGET_MASKED,
    reject_evaluation_columns: bool = False,
) -> torch.Tensor:
    profile = pd.read_csv(Path(profile_path), dtype={"raw_asin": str}, low_memory=False)
    return build_m11_feature_tensor(
        profile,
        item_serialize_dict,
        item_number=item_number,
        feature_mode=feature_mode,
        reject_evaluation_columns=reject_evaluation_columns,
    )
