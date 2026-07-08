#!/usr/bin/env python3
"""
Build Amazon-VG item-level category availability variables for CCFCRec.

This script is an offline diagnostic/data-construction utility. It does not
train models and does not modify CCFCRec training entry points.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SPLIT_PRIORITY = {"train": 0, "validate": 1, "test": 2}
GROUP_LABELS = ("s_cat_weak", "s_cat_mid", "s_cat_strong")


@dataclass(frozen=True)
class BuildOutputs:
    item_csv: Path
    meta_json: Path


def now_info() -> tuple[str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def split_category_tokens(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def title_token_count(value: object) -> int:
    if not isinstance(value, str) or not value.strip():
        return 0
    return len(re.findall(r"[A-Za-z0-9]+", value))


def safe_mean(values: Iterable[float], default: float = 0.0) -> float:
    arr = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if arr.size == 0:
        return default
    return float(arr.mean())


def safe_max(values: Iterable[float], default: float = 0.0) -> float:
    arr = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if arr.size == 0:
        return default
    return float(arr.max())


def safe_min(values: Iterable[float], default: float = 0.0) -> float:
    arr = np.asarray([value for value in values if pd.notna(value)], dtype=float)
    if arr.size == 0:
        return default
    return float(arr.min())


def log_norm(value: float, max_value: float) -> float:
    if max_value <= 0:
        return 0.0
    return float(np.clip(math.log1p(max(value, 0.0)) / math.log1p(max_value), 0.0, 1.0))


def clamp01(value: float) -> float:
    if pd.isna(value):
        return 0.0
    return float(np.clip(value, 0.0, 1.0))


def normalized_entropy(values: list[float]) -> float:
    positive = np.asarray([value for value in values if value > 0], dtype=float)
    if positive.size <= 1:
        return 0.0
    probs = positive / positive.sum()
    entropy = float(-(probs * np.log(probs)).sum())
    return float(np.clip(entropy / math.log(positive.size), 0.0, 1.0))


def mean_pairwise_jaccard(sets: list[set[str]]) -> float:
    non_empty = [item_set for item_set in sets if item_set]
    if not non_empty:
        return 0.0
    if len(non_empty) == 1:
        return 1.0
    scores: list[float] = []
    for left, right in combinations(non_empty, 2):
        union_size = len(left | right)
        scores.append(len(left & right) / union_size if union_size else 0.0)
    return safe_mean(scores)


def cached_token_user_jaccard(
    tokens: list[str],
    token_to_users: dict[str, set[str]],
    cache: dict[tuple[str, str], float],
) -> float:
    unique_tokens = sorted(set(tokens))
    non_empty_tokens = [token for token in unique_tokens if token_to_users.get(token)]
    if not non_empty_tokens:
        return 0.0
    if len(non_empty_tokens) == 1:
        return 1.0

    scores: list[float] = []
    for left_token, right_token in combinations(non_empty_tokens, 2):
        key = (left_token, right_token)
        if key not in cache:
            left = token_to_users.get(left_token, set())
            right = token_to_users.get(right_token, set())
            union_size = len(left | right)
            cache[key] = len(left & right) / union_size if union_size else 0.0
        scores.append(cache[key])
    return safe_mean(scores)


def deduplicate_metadata(metadata_df: pd.DataFrame) -> pd.DataFrame:
    required = {"asin", "title", "category"}
    missing = required - set(metadata_df.columns)
    if missing:
        raise ValueError(f"metadata 缺少字段: {sorted(missing)}")

    frame = metadata_df.copy()
    frame["_original_order"] = np.arange(len(frame))
    frame["_category_count"] = frame["category"].map(lambda value: len(split_category_tokens(value)))
    frame["_title_chars"] = frame["title"].map(lambda value: len(value) if isinstance(value, str) else 0)
    frame = frame.sort_values(
        ["asin", "_category_count", "_title_chars", "_original_order"],
        ascending=[True, False, False, True],
        kind="mergesort",
    )
    frame = frame.drop_duplicates("asin", keep="first")
    return frame.drop(columns=["_category_count", "_title_chars", "_original_order"]).reset_index(drop=True)


def load_image_features(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        value = loaded.item()
        return value if isinstance(value, dict) else {}
    if isinstance(loaded, dict):
        return loaded
    return {}


def image_norm_for(raw_asin: str, image_features: dict[str, object]) -> float:
    value = image_features.get(raw_asin)
    if value is None:
        return 0.0
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr))


def split_item_map(train_df: pd.DataFrame, validate_df: pd.DataFrame, test_df: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for split_name, frame in [("train", train_df), ("validate", validate_df), ("test", test_df)]:
        if "asin" not in frame.columns:
            raise ValueError(f"{split_name} rating 缺少 asin 字段")
        for raw_asin in frame["asin"].dropna().astype(str).unique():
            current = result.get(raw_asin)
            if current is None or SPLIT_PRIORITY[split_name] < SPLIT_PRIORITY[current]:
                result[raw_asin] = split_name
    return result


def build_train_token_stats(
    metadata_by_asin: dict[str, dict[str, object]],
    train_df: pd.DataFrame,
) -> dict[str, object]:
    train_work = train_df[["asin", "reviewerID"]].dropna().copy()
    train_work["asin"] = train_work["asin"].astype(str)
    train_work["reviewerID"] = train_work["reviewerID"].astype(str)
    train_items = set(train_work["asin"].unique())
    train_item_interactions = train_work.groupby("asin").size().to_dict()
    item_to_users = train_work.groupby("asin")["reviewerID"].agg(lambda values: set(values)).to_dict()
    token_to_items: dict[str, set[str]] = {}
    token_to_users: dict[str, set[str]] = {}
    token_to_interactions: dict[str, int] = {}
    combo_count: dict[tuple[str, ...], int] = {}

    for raw_asin in train_items:
        tokens = list(metadata_by_asin.get(raw_asin, {}).get("category_tokens", []))
        combo = tuple(sorted(tokens))
        combo_count[combo] = combo_count.get(combo, 0) + 1
        users = set(item_to_users.get(raw_asin, set()))
        interactions = int(train_item_interactions.get(raw_asin, 0))
        for token in tokens:
            token_to_items.setdefault(token, set()).add(raw_asin)
            token_to_users.setdefault(token, set()).update(users)
            token_to_interactions[token] = token_to_interactions.get(token, 0) + interactions

    train_item_count = max(len(train_items), 1)
    token_df = {token: len(items) for token, items in token_to_items.items()}
    idf = {
        token: math.log((train_item_count + 1) / (df + 1)) + 1.0
        for token, df in token_df.items()
    }
    unseen_idf = math.log(train_item_count + 1) + 1.0
    generic_tokens = {
        token
        for token, df in token_df.items()
        if df / train_item_count >= 0.5
    }
    max_values = {
        "category_count": max(
            [len(metadata_by_asin.get(raw_asin, {}).get("category_tokens", [])) for raw_asin in train_items] or [1]
        ),
        "idf": max([*idf.values(), unseen_idf, 1.0]),
        "token_item_support": max([len(items) for items in token_to_items.values()] or [0]),
        "token_user_support": max([len(users) for users in token_to_users.values()] or [0]),
        "token_interaction_support": max(list(token_to_interactions.values()) or [0]),
        "combo_count": max(list(combo_count.values()) or [0]),
    }
    return {
        "train_items": train_items,
        "token_to_items": token_to_items,
        "token_to_users": token_to_users,
        "token_to_interactions": token_to_interactions,
        "token_df": token_df,
        "idf": idf,
        "unseen_idf": unseen_idf,
        "generic_tokens": generic_tokens,
        "combo_count": combo_count,
        "max_values": max_values,
    }


def compute_train_threshold_groups(frame: pd.DataFrame) -> tuple[pd.Series, dict[str, float]]:
    train_scores = frame.loc[frame["split"].eq("train"), "s_cat"].dropna()
    thresholds: dict[str, float] = {}
    if train_scores.empty:
        thresholds = {"weak_max": 1 / 3, "mid_max": 2 / 3, "fallback": True}
        ranked = frame["s_cat"].rank(method="average", pct=True)
        return ranked.map(rank_to_group), thresholds

    weak_max = float(train_scores.quantile(1 / 3))
    mid_max = float(train_scores.quantile(2 / 3))
    if math.isclose(weak_max, mid_max):
        ranked = frame["s_cat"].rank(method="average", pct=True)
        thresholds = {"weak_max": weak_max, "mid_max": mid_max, "fallback": True}
        return ranked.map(rank_to_group), thresholds

    thresholds = {"weak_max": weak_max, "mid_max": mid_max, "fallback": False}
    return frame["s_cat"].map(lambda value: score_to_group(value, weak_max, mid_max)), thresholds


def rank_to_group(rank_pct: float) -> str:
    if rank_pct <= 1 / 3:
        return GROUP_LABELS[0]
    if rank_pct <= 2 / 3:
        return GROUP_LABELS[1]
    return GROUP_LABELS[2]


def score_to_group(score: float, weak_max: float, mid_max: float) -> str:
    if score <= weak_max:
        return GROUP_LABELS[0]
    if score <= mid_max:
        return GROUP_LABELS[1]
    return GROUP_LABELS[2]


def build_category_availability(
    metadata_df: pd.DataFrame,
    train_df: pd.DataFrame,
    validate_df: pd.DataFrame,
    test_df: pd.DataFrame,
    image_features: dict[str, object] | None = None,
    fit_scope: str = "train_metadata_fit",
) -> tuple[pd.DataFrame, dict[str, object]]:
    if fit_scope != "train_metadata_fit":
        raise ValueError("Only fit_scope='train_metadata_fit' is supported for the first protocol.")

    image_features = image_features or {}
    deduped = deduplicate_metadata(metadata_df)
    metadata_by_asin: dict[str, dict[str, object]] = {}
    for _, row in deduped.iterrows():
        raw_asin = str(row["asin"])
        tokens = split_category_tokens(row.get("category"))
        metadata_by_asin[raw_asin] = {
            "title": row.get("title") if isinstance(row.get("title"), str) else "",
            "category_raw": row.get("category") if isinstance(row.get("category"), str) else "",
            "category_tokens": tokens,
        }

    item_to_split = split_item_map(train_df, validate_df, test_df)
    stats = build_train_token_stats(metadata_by_asin, train_df)
    max_values = stats["max_values"]
    max_title_tokens = max(
        [
            title_token_count(metadata_by_asin.get(raw_asin, {}).get("title", ""))
            for raw_asin, split in item_to_split.items()
            if split == "train"
        ]
        or [1]
    )
    max_image_norm = max(
        [
            image_norm_for(raw_asin, image_features)
            for raw_asin, split in item_to_split.items()
            if split == "train"
        ]
        or [0.0]
    )

    rows: list[dict[str, object]] = []
    jaccard_cache: dict[tuple[str, str], float] = {}
    sorted_items = sorted(item_to_split.items(), key=lambda item: (SPLIT_PRIORITY[item[1]], item[0]))
    for serial_item_id, (raw_asin, split_name) in enumerate(sorted_items):
        meta = metadata_by_asin.get(raw_asin, {"title": "", "category_raw": "", "category_tokens": []})
        tokens = list(meta["category_tokens"])
        category_count = len(tokens)
        token_idfs = [stats["idf"].get(token, stats["unseen_idf"]) for token in tokens]
        idf_mean = safe_mean(token_idfs)
        idf_max = safe_max(token_idfs)
        idf_mean_norm = log_norm(idf_mean, max_values["idf"])
        idf_max_norm = log_norm(idf_max, max_values["idf"])
        specific_tokens = [token for token in tokens if token not in stats["generic_tokens"]]
        generic_ratio = 1.0 - (len(specific_tokens) / category_count) if category_count else 0.0

        combo = tuple(sorted(tokens))
        combo_frequency = int(stats["combo_count"].get(combo, 0))
        combo_rarity = 1.0 - log_norm(combo_frequency, max_values["combo_count"])

        item_supports = [len(stats["token_to_items"].get(token, set())) for token in tokens]
        user_supports = [len(stats["token_to_users"].get(token, set())) for token in tokens]
        interaction_supports = [stats["token_to_interactions"].get(token, 0) for token in tokens]
        support_entropy = normalized_entropy([float(value) for value in item_supports])
        jaccard = cached_token_user_jaccard(tokens, stats["token_to_users"], jaccard_cache)

        item_support_mean = safe_mean(item_supports)
        user_support_mean = safe_mean(user_supports)
        interaction_support_mean = safe_mean(interaction_supports)
        item_support_max = safe_max(item_supports)
        interaction_support_max = safe_max(interaction_supports)
        item_support_min = safe_min(item_supports)
        user_support_min = safe_min(user_supports)
        interaction_support_min = safe_min(interaction_supports)

        item_support_score = log_norm(item_support_mean, max_values["token_item_support"])
        user_support_score = log_norm(user_support_mean, max_values["token_user_support"])
        interaction_support_score = log_norm(interaction_support_mean, max_values["token_interaction_support"])

        title_tokens = title_token_count(meta["title"])
        image_norm = image_norm_for(raw_asin, image_features)
        has_title = 1 if title_tokens > 0 else 0
        has_image = 1 if raw_asin in image_features else 0
        title_score = log_norm(title_tokens, max_title_tokens)
        image_score = log_norm(image_norm, max_image_norm)

        gran_count_score = log_norm(category_count, max_values["category_count"])
        specific_ratio = len(specific_tokens) / category_count if category_count else 0.0
        s_cat_gran = safe_mean([gran_count_score, specific_ratio, idf_mean_norm])
        disc_specificity = safe_mean([idf_mean_norm, 1.0 - generic_ratio, combo_rarity])
        s_cat_disc = disc_specificity
        collab_score = safe_mean(
            [
                item_support_score,
                user_support_score,
                interaction_support_score,
                jaccard,
                1.0 - support_entropy,
            ]
        )
        s_cat_collab = collab_score
        s_cat = safe_mean([s_cat_gran, s_cat_disc, s_cat_collab])

        row = {
            "raw_asin": raw_asin,
            "serial_item_id": serial_item_id,
            "split": split_name,
            "category_raw": meta["category_raw"],
            "category_tokens": "|".join(tokens),
            "category_count": category_count,
            "cat_count_bin": category_count_bin(category_count),
            "A_gran_token_count": category_count,
            "A_gran_specific_token_count": len(specific_tokens),
            "A_gran_specific_ratio": specific_ratio,
            "A_gran_idf_mean": idf_mean,
            "A_gran_idf_max": idf_max,
            "A_disc_token_idf_mean": idf_mean,
            "A_disc_token_idf_max": idf_max,
            "A_disc_combo_frequency_train": combo_frequency,
            "A_disc_combo_rarity": combo_rarity,
            "A_disc_generic_token_ratio": generic_ratio,
            "A_disc_specificity_score": disc_specificity,
            "A_collab_train_token_item_support_min": item_support_min,
            "A_collab_train_token_item_support_mean": item_support_mean,
            "A_collab_train_token_item_support_max": item_support_max,
            "A_collab_train_token_user_support_min": user_support_min,
            "A_collab_train_token_user_support_mean": user_support_mean,
            "A_collab_train_token_interaction_support_min": interaction_support_min,
            "A_collab_train_token_interaction_support_mean": interaction_support_mean,
            "A_collab_train_token_interaction_support_max": interaction_support_max,
            "A_collab_user_set_jaccard_mean": jaccard,
            "A_collab_support_entropy_mean": support_entropy,
            "A_collab_consistency_score": collab_score,
            "R_title_token_count": title_tokens,
            "R_has_title": has_title,
            "R_has_image_feature": has_image,
            "R_image_norm": image_norm,
            "R_metadata_richness_score": safe_mean([has_title, title_score, has_image, image_score]),
            "S_train_token_item_support_min": item_support_min,
            "S_train_token_item_support_mean": item_support_mean,
            "S_train_token_item_support_max": item_support_max,
            "S_train_token_interaction_support_min": interaction_support_min,
            "S_train_token_interaction_support_mean": interaction_support_mean,
            "S_train_token_interaction_support_max": interaction_support_max,
            "S_train_token_user_support_min": user_support_min,
            "S_train_token_user_support_mean": user_support_mean,
            "S_train_support_score": safe_mean([item_support_score, user_support_score, interaction_support_score]),
            "P_train_token_popularity_mean": interaction_support_mean,
            "P_train_token_popularity_max": interaction_support_max,
            "P_popularity_score": interaction_support_score,
            "s_cat_gran": clamp01(s_cat_gran),
            "s_cat_disc": clamp01(s_cat_disc),
            "s_cat_collab": clamp01(s_cat_collab),
            "s_cat": clamp01(s_cat),
        }
        rows.append(row)

    result = pd.DataFrame(rows)
    if not result.empty:
        groups, thresholds = compute_train_threshold_groups(result)
        result["s_cat_group"] = groups
    else:
        thresholds = {"weak_max": None, "mid_max": None, "fallback": True}
        result["s_cat_group"] = []

    created_stamp, created_at = now_info()
    meta = {
        "created_stamp": created_stamp,
        "created_at": created_at,
        "dataset": "Amazon VG",
        "fit_scope": fit_scope,
        "interaction_scope": "train_rating_only",
        "stable_key": "raw_asin",
        "row_scope": "train_validate_test_item_union",
        "metadata_dedup_policy": "prefer_more_category_tokens_then_longer_title_then_original_order",
        "generic_token_policy": "train_item_document_frequency_share >= 0.5",
        "s_cat_policy": "equal_weight_mean_of_gran_disc_collab",
        "s_cat_group_policy": "train_s_cat_tertiles_applied_to_all_splits",
        "s_cat_group_thresholds": thresholds,
        "leakage_policy": {
            "validation_test_interactions": "not_used_for_feature_construction",
            "test_labels": "not_used_for_feature_construction",
            "metadata_for_target_items": "allowed_as_side_information",
            "idf_and_group_thresholds": "fit_on_train_items_only",
        },
        "input_counts": {
            "metadata_rows": int(len(metadata_df)),
            "metadata_unique_asin_after_dedup": int(len(deduped)),
            "train_rows": int(len(train_df)),
            "validate_rows": int(len(validate_df)),
            "test_rows": int(len(test_df)),
            "output_rows": int(len(result)),
        },
        "performance_notes": {
            "cached_token_user_jaccard_pairs": int(len(jaccard_cache)),
        },
    }
    return result, meta


def category_count_bin(category_count: int) -> str:
    if category_count <= 0:
        return "cat_count_0"
    if category_count <= 3:
        return "cat_count_1_3"
    if category_count == 4:
        return "cat_count_4"
    return "cat_count_5_plus"


def write_outputs(result: pd.DataFrame, meta: dict[str, object], output_dir: Path) -> BuildOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    item_csv = output_dir / "category_availability_item.csv"
    meta_json = output_dir / "category_availability_meta.json"
    result.to_csv(item_csv, index=False)
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return BuildOutputs(item_csv=item_csv, meta_json=meta_json)


def read_rating_csv(path: Path, split_name: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"reviewerID", "asin"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{split_name} rating 缺少字段: {sorted(missing)}")
    return frame


def build_from_paths(
    metadata_path: Path,
    train_rating_path: Path,
    validate_rating_path: Path,
    test_rating_path: Path,
    image_feature_path: Path | None,
    output_dir: Path,
    fit_scope: str,
) -> BuildOutputs:
    metadata = pd.read_csv(metadata_path)
    train = read_rating_csv(train_rating_path, "train")
    validate = read_rating_csv(validate_rating_path, "validate")
    test = read_rating_csv(test_rating_path, "test")
    image_features = load_image_features(image_feature_path)
    result, meta = build_category_availability(
        metadata_df=metadata,
        train_df=train,
        validate_df=validate,
        test_df=test,
        image_features=image_features,
        fit_scope=fit_scope,
    )
    meta["input_files"] = {
        "metadata": str(metadata_path),
        "train_rating": str(train_rating_path),
        "validate_rating": str(validate_rating_path),
        "test_rating": str(test_rating_path),
        "image_feature": str(image_feature_path) if image_feature_path else None,
    }
    outputs = write_outputs(result, meta, output_dir)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amazon-vg-dir", type=Path, default=Path("Amazon VG"))
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--train-rating", type=Path)
    parser.add_argument("--validate-rating", type=Path)
    parser.add_argument("--test-rating", type=Path)
    parser.add_argument("--image-feature", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fit-scope", default="train_metadata_fit")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    amazon_vg_dir = args.amazon_vg_dir
    data_dir = amazon_vg_dir / "data"
    metadata = args.metadata or data_dir / "asin.csv"
    train_rating = args.train_rating or data_dir / "train_rating.csv"
    validate_rating = args.validate_rating or data_dir / "validate_rating.csv"
    test_rating = args.test_rating or data_dir / "test_rating.csv"
    image_feature = args.image_feature or data_dir / "img_feature.npy"
    outputs = build_from_paths(
        metadata_path=metadata,
        train_rating_path=train_rating,
        validate_rating_path=validate_rating,
        test_rating_path=test_rating,
        image_feature_path=image_feature,
        output_dir=args.output_dir,
        fit_scope=args.fit_scope,
    )
    print(f"wrote {outputs.item_csv}")
    print(f"wrote {outputs.meta_json}")


if __name__ == "__main__":
    main()
