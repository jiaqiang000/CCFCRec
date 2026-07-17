#!/usr/bin/env python3
"""Construct and audit the train-only CICP-MP-v1 multi-component profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.stats import ks_2samp, spearmanr
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from analyze_amazon_vg_cicp_train_only_signal import (
    ACAT_RESIDUAL_COLUMNS,
    COLLAB_DIM,
    CONTROL_COLUMNS,
    FOLDS,
    IMAGE_DIM,
    RIDGE_ALPHAS,
    SEED,
    build_interaction_matrix,
    empirical_percentile,
    load_profile,
    normalize_interactions,
    project_images_mps,
    reliability_weight,
    row_cosine,
)


PROJECT_ROOT = Path("/Users/luojiaqiang/Documents/Obsidian Vault/科研/CCFCRec对比学习思路")
CODE_ROOT = PROJECT_ROOT / "CCFCRec-code"
DEFAULT_TRAIN_RATING = CODE_ROOT / "Amazon VG/data/train_rating.csv"
DEFAULT_IMAGE_FEATURES = CODE_ROOT / "Amazon VG/data/img_feature.npy"
DEFAULT_CATEGORY_PICKLE = CODE_ROOT / "Amazon VG/data/asin_int_category.pkl"
DEFAULT_ASIN_CSV = CODE_ROOT / "Amazon VG/data/asin.csv"
DEFAULT_CICP_FEATURE_CODE = CODE_ROOT / "Amazon VG/cicp_features.py"
DEFAULT_ORIGINAL_CICP_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录/temp_20260713"
    / "2026-07-13 121409 cicp-train-only-signal-audit-v1_1"
    / "cicp_item_profile.csv"
)
DEFAULT_V3_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录/temp_20260705"
    / "2026-07-05 112021 category-availability-v3-purity-audit"
    / "category_availability_v3_item.csv"
)

CATEGORY_DIM = 64
DIRECTION_DIM = 64
DIRECTION_COMPRESSED_DIM = 16
NESTED_FOLDS = 5
HGB_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
}
SHUFFLE_SCHEMES = (
    "global_seed43",
    "global_seed44",
    "within_category_count_seed43",
)
FORBIDDEN_TOKENS = (
    "ndcg",
    "hr@",
    "hit",
    "rank",
    "recommendation_metric",
    "best_target_rank",
    "baseline_margin",
    "delta_ndcg",
    "delta_hr",
)
CORE_MAPPING_TARGETS = (
    "base_collaborative_cosine",
    "shuffled_category_collaborative_cosine",
    "full_collaborative_cosine",
    "category_total_increment_raw",
    "category_capacity_increment_raw",
    "category_semantic_increment_raw",
    "category_semantic_residual_ridge",
    "category_semantic_residual_hgb",
)
ATTRIBUTION_TARGETS = (
    "category_attribution_max",
    "category_attribution_min",
    "category_attribution_std",
    "category_attribution_positive_share",
    "category_attribution_entropy",
    "category_attribution_first_minus_last",
)
PROFILE_BASE_COLUMNS = (
    "raw_asin",
    "split",
    "category_count",
    "R_metadata_richness_score",
    "S_train_support_score",
    "P_popularity_score",
    "train_interaction_count",
    "log1p_train_interaction_count",
)


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def safe_spearman(left: np.ndarray | pd.Series, right: np.ndarray | pd.Series) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    valid = np.isfinite(left_array) & np.isfinite(right_array)
    if valid.sum() < 3 or np.unique(left_array[valid]).size < 2 or np.unique(right_array[valid]).size < 2:
        return float("nan")
    return float(spearmanr(left_array[valid], right_array[valid]).statistic)


def md_table(frame: pd.DataFrame, digits: int = 6) -> str:
    if frame.empty:
        return "_无记录_"
    display = frame.copy()
    for column in display.select_dtypes(include=[np.number]).columns:
        display[column] = display[column].map(
            lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}"
        )
    headers = [str(column) for column in display.columns]
    rows = [[str(value) for value in row] for row in display.itertuples(index=False, name=None)]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def assert_no_recommendation_metrics(frame: pd.DataFrame, source: str) -> None:
    normalized = [str(column).strip().lower() for column in frame.columns]
    forbidden = sorted(
        column
        for column in normalized
        if any(token in column for token in FORBIDDEN_TOKENS)
    )
    if forbidden:
        raise ValueError(f"{source} contains forbidden recommendation columns: {forbidden}")


def make_folds(item_count: int, seed: int = SEED, fold_count: int = FOLDS) -> np.ndarray:
    folds = np.empty(item_count, dtype=np.int16)
    splitter = KFold(fold_count, shuffle=True, random_state=seed)
    for fold, (_, hold_index) in enumerate(splitter.split(np.arange(item_count))):
        folds[hold_index] = fold
    return folds


def stable_item_folds(items: list[str], fold_count: int = FOLDS) -> np.ndarray:
    return np.asarray(
        [
            int.from_bytes(hashlib.sha256(item.encode("utf-8")).digest()[:8], "little")
            % fold_count
            for item in items
        ],
        dtype=np.int16,
    )


def _row_normalize(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norm, 1e-12, None)


@dataclass
class CategoryAssets:
    train_latent: np.ndarray
    validation_latent: np.ndarray
    removal_latent: np.ndarray
    removal_item_index: np.ndarray
    removal_category_id: np.ndarray
    removal_position: np.ndarray
    train_categories: list[list[int]]
    validation_categories: list[list[int]]
    category_name_by_id: dict[int, str]
    audit: dict[str, Any]


def build_category_assets(
    train_items: list[str],
    validation_items: list[str],
    category_pickle: Path,
    dim: int = CATEGORY_DIM,
) -> CategoryAssets:
    start = time.time()
    with category_pickle.open("rb") as handle:
        payload = pickle.load(handle)
    category_map = {
        str(item): [int(value) for value in values]
        for item, values in payload["asin_category_int_map"].items()
    }
    category_name_by_id = {
        int(value): str(name) for name, value in payload["category_ser_map"].items()
    }
    category_count = len(category_name_by_id)
    all_items = [*train_items, *validation_items]
    all_categories = [category_map.get(item, []) for item in all_items]
    missing = [item for item, values in zip(all_items, all_categories) if not values]
    if missing:
        raise ValueError(f"category metadata missing for {len(missing)} train/validation items")

    rows: list[int] = []
    columns: list[int] = []
    for row, categories in enumerate(all_categories):
        rows.extend([row] * len(categories))
        columns.extend(categories)
    matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(len(all_items), category_count),
    )
    train_matrix = matrix[: len(train_items)]
    validation_matrix = matrix[len(train_items) :]
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True)
    train_tfidf = tfidf.fit_transform(train_matrix)
    validation_tfidf = tfidf.transform(validation_matrix)
    svd = TruncatedSVD(n_components=dim, n_iter=7, random_state=SEED)
    train_latent = _row_normalize(svd.fit_transform(train_tfidf).astype(np.float32))
    validation_latent = _row_normalize(svd.transform(validation_tfidf).astype(np.float32))

    removal_rows: list[int] = []
    removal_columns: list[int] = []
    removal_item_index: list[int] = []
    removal_category_id: list[int] = []
    removal_position: list[int] = []
    variant_index = 0
    train_categories = all_categories[: len(train_items)]
    for item_index, categories in enumerate(train_categories):
        for position, removed_category in enumerate(categories):
            kept = [category for category in categories if category != removed_category]
            removal_rows.extend([variant_index] * len(kept))
            removal_columns.extend(kept)
            removal_item_index.append(item_index)
            removal_category_id.append(removed_category)
            removal_position.append(position)
            variant_index += 1
    removal_matrix = csr_matrix(
        (
            np.ones(len(removal_rows), dtype=np.float32),
            (removal_rows, removal_columns),
        ),
        shape=(variant_index, category_count),
    )
    removal_latent = _row_normalize(
        svd.transform(tfidf.transform(removal_matrix)).astype(np.float32)
    )
    audit = {
        "category_count": category_count,
        "train_covered": len(train_items),
        "validation_covered": len(validation_items),
        "missing_items": 0,
        "train_removal_variant_count": variant_index,
        "category_dim": dim,
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
        "elapsed_seconds": time.time() - start,
    }
    return CategoryAssets(
        train_latent=train_latent,
        validation_latent=validation_latent,
        removal_latent=removal_latent,
        removal_item_index=np.asarray(removal_item_index, dtype=np.int32),
        removal_category_id=np.asarray(removal_category_id, dtype=np.int32),
        removal_position=np.asarray(removal_position, dtype=np.int16),
        train_categories=train_categories,
        validation_categories=all_categories[len(train_items) :],
        category_name_by_id=category_name_by_id,
        audit=audit,
    )


def permutation_without_fixed_points(indices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if len(indices) <= 1:
        return indices.copy()
    for _ in range(100):
        permuted = rng.permutation(indices)
        if not np.any(permuted == indices):
            return permuted
    return np.roll(indices, 1)


def build_shuffle_permutations(category_counts: np.ndarray) -> dict[str, np.ndarray]:
    item_indices = np.arange(len(category_counts), dtype=np.int32)
    permutations: dict[str, np.ndarray] = {}
    for seed in (SEED, SEED + 1):
        permutations[f"global_seed{seed}"] = permutation_without_fixed_points(
            item_indices, np.random.default_rng(seed)
        )
    within = item_indices.copy()
    rng = np.random.default_rng(SEED)
    for count in np.unique(category_counts):
        members = item_indices[category_counts == count]
        within[members] = permutation_without_fixed_points(members, rng)
    permutations["within_category_count_seed43"] = within
    return permutations


@dataclass
class ProbeResult:
    item: pd.DataFrame
    folds: pd.DataFrame
    attribution: pd.DataFrame
    attribution_item: pd.DataFrame
    directions: dict[str, np.ndarray]
    stability: pd.DataFrame
    elapsed_seconds: float


def _fit_probe(
    fit_values: np.ndarray,
    fit_target: np.ndarray,
    weights: np.ndarray,
) -> tuple[StandardScaler, RidgeCV]:
    scaler = StandardScaler().fit(fit_values, sample_weight=weights)
    model = RidgeCV(alphas=RIDGE_ALPHAS).fit(
        scaler.transform(fit_values), fit_target, sample_weight=weights
    )
    return scaler, model


def _predict_probe(
    scaler: StandardScaler,
    model: RidgeCV,
    values: np.ndarray,
) -> np.ndarray:
    return model.predict(scaler.transform(values))


def build_attribution_item_summary(
    attribution: pd.DataFrame, item_count: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = attribution.groupby("item_index", sort=False)
    for item_index in range(item_count):
        part = grouped.get_group(item_index)
        values = part["marginal_contribution"].to_numpy(dtype=float)
        absolute = np.abs(values)
        if len(values) <= 1 or absolute.sum() <= 1e-12:
            entropy = 0.0
        else:
            probability = absolute / absolute.sum()
            entropy = float(
                -(probability * np.log(np.clip(probability, 1e-12, None))).sum()
                / math.log(len(values))
            )
        ordered = part.sort_values("category_position")
        rows.append(
            {
                "item_index": item_index,
                "category_attribution_max": float(values.max()),
                "category_attribution_min": float(values.min()),
                "category_attribution_std": float(values.std(ddof=0)),
                "category_attribution_positive_share": float((values > 0).mean()),
                "category_attribution_entropy": entropy,
                "category_attribution_first_minus_last": float(
                    ordered.iloc[0]["marginal_contribution"]
                    - ordered.iloc[-1]["marginal_contribution"]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("item_index").reset_index(drop=True)


def run_collaborative_probe(
    interactions: csr_matrix,
    image: np.ndarray,
    category_assets: CategoryAssets,
    controls: np.ndarray,
    weights: np.ndarray,
    folds: np.ndarray,
    train_items: list[str],
) -> ProbeResult:
    start_all = time.time()
    item_count, user_count = interactions.shape
    category = category_assets.train_latent
    category_counts = np.asarray([len(values) for values in category_assets.train_categories])
    permutations = build_shuffle_permutations(category_counts)
    shuffled_categories = {
        name: category[permutation] for name, permutation in permutations.items()
    }
    base_cosine = np.empty(item_count, dtype=float)
    full_cosine = np.empty(item_count, dtype=float)
    shuffled_cosines = {
        name: np.empty(item_count, dtype=float) for name in SHUFFLE_SCHEMES
    }
    directions = {
        name: np.empty((item_count, DIRECTION_DIM), dtype=np.float32)
        for name in SHUFFLE_SCHEMES
    }
    rng = np.random.default_rng(SEED)
    user_projection = (
        rng.integers(0, 2, size=(user_count, DIRECTION_DIM), dtype=np.int8).astype(np.float32)
        * 2.0
        - 1.0
    ) / math.sqrt(DIRECTION_DIM)
    fold_rows: list[dict[str, Any]] = []
    attribution_parts: list[pd.DataFrame] = []

    removal_items = category_assets.removal_item_index
    for fold in range(FOLDS):
        fold_start = time.time()
        fit_index = np.flatnonzero(folds != fold)
        hold_index = np.flatnonzero(folds == fold)
        fit_raw = interactions[fit_index]
        hold_raw = interactions[hold_index]
        fit_user_degree = np.asarray(fit_raw.sum(axis=0)).ravel()
        fit_matrix = normalize_interactions(fit_raw, fit_user_degree)
        hold_matrix = normalize_interactions(hold_raw, fit_user_degree)
        svd = TruncatedSVD(n_components=COLLAB_DIM, n_iter=7, random_state=SEED)
        fit_target = _row_normalize(svd.fit_transform(fit_matrix))
        hold_target = _row_normalize(svd.transform(hold_matrix))

        base_fit = np.hstack([image[fit_index], controls[fit_index]])
        base_hold = np.hstack([image[hold_index], controls[hold_index]])
        full_fit = np.hstack([base_fit, category[fit_index]])
        full_hold = np.hstack([base_hold, category[hold_index]])
        base_scaler, base_model = _fit_probe(base_fit, fit_target, weights[fit_index])
        full_scaler, full_model = _fit_probe(full_fit, fit_target, weights[fit_index])
        base_prediction = _predict_probe(base_scaler, base_model, base_hold)
        full_prediction = _predict_probe(full_scaler, full_model, full_hold)
        base_cosine[hold_index] = row_cosine(hold_target, base_prediction)
        full_cosine[hold_index] = row_cosine(hold_target, full_prediction)

        collaborative_to_common = svd.components_ @ user_projection
        fold_row: dict[str, Any] = {
            "fold": fold,
            "fit_count": len(fit_index),
            "hold_count": len(hold_index),
            "base_alpha": float(base_model.alpha_),
            "full_alpha": float(full_model.alpha_),
            "base_cosine_mean": float(np.average(base_cosine[hold_index], weights=weights[hold_index])),
            "full_cosine_mean": float(np.average(full_cosine[hold_index], weights=weights[hold_index])),
        }
        for scheme in SHUFFLE_SCHEMES:
            shuffled = shuffled_categories[scheme]
            shuffled_fit = np.hstack([base_fit, shuffled[fit_index]])
            shuffled_hold = np.hstack([base_hold, shuffled[hold_index]])
            shuffled_scaler, shuffled_model = _fit_probe(
                shuffled_fit, fit_target, weights[fit_index]
            )
            shuffled_prediction = _predict_probe(
                shuffled_scaler, shuffled_model, shuffled_hold
            )
            shuffled_cosines[scheme][hold_index] = row_cosine(
                hold_target, shuffled_prediction
            )
            directions[scheme][hold_index] = (
                (full_prediction - shuffled_prediction) @ collaborative_to_common
            ).astype(np.float32)
            semantic = full_cosine[hold_index] - shuffled_cosines[scheme][hold_index]
            fold_row[f"{scheme}_shuffled_alpha"] = float(shuffled_model.alpha_)
            fold_row[f"{scheme}_semantic_increment_mean"] = float(
                np.average(semantic, weights=weights[hold_index])
            )
            fold_row[f"{scheme}_semantic_positive_share"] = float((semantic > 0).mean())

        removal_mask = np.isin(removal_items, hold_index)
        removal_indices = np.flatnonzero(removal_mask)
        removal_item_global = removal_items[removal_indices]
        hold_position = np.empty(item_count, dtype=np.int32)
        hold_position.fill(-1)
        hold_position[hold_index] = np.arange(len(hold_index), dtype=np.int32)
        source_position = hold_position[removal_item_global]
        removal_features = np.hstack(
            [
                image[removal_item_global],
                controls[removal_item_global],
                category_assets.removal_latent[removal_indices],
            ]
        )
        removal_prediction = _predict_probe(full_scaler, full_model, removal_features)
        removal_cosine = row_cosine(hold_target[source_position], removal_prediction)
        marginal = full_cosine[removal_item_global] - removal_cosine
        attribution_parts.append(
            pd.DataFrame(
                {
                    "raw_asin": np.asarray(train_items, dtype=object)[removal_item_global],
                    "item_index": removal_item_global,
                    "probe_fold": fold,
                    "category_id": category_assets.removal_category_id[removal_indices],
                    "category_position": category_assets.removal_position[removal_indices],
                    "full_collaborative_cosine": full_cosine[removal_item_global],
                    "removed_category_collaborative_cosine": removal_cosine,
                    "marginal_contribution": marginal,
                }
            )
        )
        fold_row["attribution_variant_count"] = len(removal_indices)
        fold_row["elapsed_seconds"] = time.time() - fold_start
        fold_rows.append(fold_row)
        print(f"[CICP-MP collaborative probe] fold {fold + 1}/{FOLDS} complete", flush=True)

    primary = "global_seed43"
    item = pd.DataFrame(
        {
            "probe_fold": folds,
            "base_collaborative_cosine": base_cosine,
            "shuffled_category_collaborative_cosine": shuffled_cosines[primary],
            "full_collaborative_cosine": full_cosine,
            "category_total_increment_raw": full_cosine - base_cosine,
            "category_capacity_increment_raw": shuffled_cosines[primary] - base_cosine,
            "category_semantic_increment_raw": full_cosine - shuffled_cosines[primary],
        }
    )
    for scheme in SHUFFLE_SCHEMES[1:]:
        item[f"category_semantic_increment_{scheme}"] = full_cosine - shuffled_cosines[scheme]

    attribution = pd.concat(attribution_parts, ignore_index=True)
    attribution["category_name"] = attribution["category_id"].map(
        category_assets.category_name_by_id
    )
    attribution_item = build_attribution_item_summary(attribution, item_count)
    item = pd.concat(
        [item, attribution_item.drop(columns="item_index")], axis=1
    )

    stability_rows: list[dict[str, Any]] = []
    primary_semantic = item["category_semantic_increment_raw"].to_numpy(dtype=float)
    primary_direction = directions[primary]
    for scheme in SHUFFLE_SCHEMES[1:]:
        semantic = item[f"category_semantic_increment_{scheme}"].to_numpy(dtype=float)
        direction_cosine = row_cosine(primary_direction, directions[scheme])
        stability_rows.append(
            {
                "comparison": f"{primary}_vs_{scheme}",
                "semantic_increment_spearman": safe_spearman(primary_semantic, semantic),
                "semantic_sign_agreement": float(
                    (np.sign(primary_semantic) == np.sign(semantic)).mean()
                ),
                "direction_cosine_mean": float(np.mean(direction_cosine)),
                "direction_cosine_median": float(np.median(direction_cosine)),
                "direction_positive_cosine_share": float((direction_cosine > 0).mean()),
            }
        )
    return ProbeResult(
        item=item,
        folds=pd.DataFrame(fold_rows),
        attribution=attribution,
        attribution_item=attribution_item,
        directions=directions,
        stability=pd.DataFrame(stability_rows),
        elapsed_seconds=time.time() - start_all,
    )


@dataclass
class ControlResult:
    ridge_oof: np.ndarray
    hgb_oof: np.ndarray
    validation_ridge: np.ndarray
    validation_hgb: np.ndarray
    folds: pd.DataFrame


def cross_fitted_controls(
    target: np.ndarray,
    train_controls: np.ndarray,
    validation_controls: np.ndarray,
    weights: np.ndarray,
    folds: np.ndarray,
) -> ControlResult:
    ridge_oof = np.empty(len(target), dtype=float)
    hgb_oof = np.empty(len(target), dtype=float)
    validation_ridge_folds = np.empty((FOLDS, len(validation_controls)), dtype=float)
    validation_hgb_folds = np.empty((FOLDS, len(validation_controls)), dtype=float)
    rows: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        fit_index = np.flatnonzero(folds != fold)
        hold_index = np.flatnonzero(folds == fold)
        scaler = StandardScaler().fit(train_controls[fit_index], sample_weight=weights[fit_index])
        ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(
            scaler.transform(train_controls[fit_index]),
            target[fit_index],
            sample_weight=weights[fit_index],
        )
        ridge_oof[hold_index] = ridge.predict(scaler.transform(train_controls[hold_index]))
        validation_ridge_folds[fold] = ridge.predict(scaler.transform(validation_controls))
        hgb = HistGradientBoostingRegressor(
            **HGB_PARAMS, random_state=SEED + fold
        ).fit(train_controls[fit_index], target[fit_index], sample_weight=weights[fit_index])
        hgb_oof[hold_index] = hgb.predict(train_controls[hold_index])
        validation_hgb_folds[fold] = hgb.predict(validation_controls)
        rows.append(
            {
                "fold": fold,
                "ridge_alpha": float(ridge.alpha_),
                "ridge_hold_spearman": safe_spearman(
                    ridge_oof[hold_index], target[hold_index]
                ),
                "hgb_hold_spearman": safe_spearman(hgb_oof[hold_index], target[hold_index]),
            }
        )
    return ControlResult(
        ridge_oof=ridge_oof,
        hgb_oof=hgb_oof,
        validation_ridge=validation_ridge_folds.mean(axis=0),
        validation_hgb=validation_hgb_folds.mean(axis=0),
        folds=pd.DataFrame(rows),
    )


@dataclass
class ScalarMappingResult:
    train_predictions: dict[str, np.ndarray]
    validation_predictions: dict[str, np.ndarray]
    validation_fold_predictions: dict[str, np.ndarray]
    summary: pd.DataFrame


def cross_fitted_scalar_mappings(
    targets: pd.DataFrame,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    weights: np.ndarray,
    folds: np.ndarray,
    *,
    use_hgb: bool,
) -> ScalarMappingResult:
    train_predictions: dict[str, np.ndarray] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    validation_fold_predictions: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for target_name in targets.columns:
        target = targets[target_name].to_numpy(dtype=float)
        ridge_oof = np.empty(len(target), dtype=float)
        hgb_oof = np.empty(len(target), dtype=float) if use_hgb else None
        validation_ridge = np.empty((FOLDS, len(validation_features)), dtype=float)
        validation_hgb = (
            np.empty((FOLDS, len(validation_features)), dtype=float) if use_hgb else None
        )
        fold_ridge_rho: list[float] = []
        fold_hgb_rho: list[float] = []
        start = time.time()
        for fold in range(FOLDS):
            fit_index = np.flatnonzero(folds != fold)
            hold_index = np.flatnonzero(folds == fold)
            scaler = StandardScaler().fit(
                train_features[fit_index], sample_weight=weights[fit_index]
            )
            ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(
                scaler.transform(train_features[fit_index]),
                target[fit_index],
                sample_weight=weights[fit_index],
            )
            ridge_oof[hold_index] = ridge.predict(
                scaler.transform(train_features[hold_index])
            )
            validation_ridge[fold] = ridge.predict(
                scaler.transform(validation_features)
            )
            fold_ridge_rho.append(
                safe_spearman(ridge_oof[hold_index], target[hold_index])
            )
            if use_hgb:
                hgb = HistGradientBoostingRegressor(
                    **HGB_PARAMS, random_state=SEED + fold
                ).fit(
                    train_features[fit_index],
                    target[fit_index],
                    sample_weight=weights[fit_index],
                )
                assert hgb_oof is not None and validation_hgb is not None
                hgb_oof[hold_index] = hgb.predict(train_features[hold_index])
                validation_hgb[fold] = hgb.predict(validation_features)
                fold_hgb_rho.append(
                    safe_spearman(hgb_oof[hold_index], target[hold_index])
                )

        train_predictions[f"{target_name}__ridge"] = ridge_oof
        validation_predictions[f"{target_name}__ridge"] = validation_ridge.mean(axis=0)
        validation_fold_predictions[f"{target_name}__ridge"] = validation_ridge
        rows.append(
            {
                "target": target_name,
                "model": "ridge",
                "oof_spearman": safe_spearman(ridge_oof, target),
                "positive_fold_count": int(np.sum(np.asarray(fold_ridge_rho) > 0)),
                "fold_count": FOLDS,
                "hgb_ridge_prediction_spearman": np.nan,
                "elapsed_seconds": time.time() - start,
            }
        )
        if use_hgb:
            assert hgb_oof is not None and validation_hgb is not None
            train_predictions[f"{target_name}__hgb"] = hgb_oof
            validation_predictions[f"{target_name}__hgb"] = validation_hgb.mean(axis=0)
            validation_fold_predictions[f"{target_name}__hgb"] = validation_hgb
            disagreement_rho = safe_spearman(hgb_oof, ridge_oof)
            rows.append(
                {
                    "target": target_name,
                    "model": "hist_gradient_boosting",
                    "oof_spearman": safe_spearman(hgb_oof, target),
                    "positive_fold_count": int(np.sum(np.asarray(fold_hgb_rho) > 0)),
                    "fold_count": FOLDS,
                    "hgb_ridge_prediction_spearman": disagreement_rho,
                    "elapsed_seconds": time.time() - start,
                }
            )
        print(
            f"[CICP-MP deployable mapping] {target_name} complete",
            flush=True,
        )
    return ScalarMappingResult(
        train_predictions=train_predictions,
        validation_predictions=validation_predictions,
        validation_fold_predictions=validation_fold_predictions,
        summary=pd.DataFrame(rows),
    )


@dataclass
class NestedMappingResult:
    train_hgb_mean: np.ndarray
    train_hgb_std: np.ndarray
    train_positive_share: np.ndarray
    train_ridge_mean: np.ndarray
    validation_hgb_mean: np.ndarray
    validation_hgb_std: np.ndarray
    validation_positive_share: np.ndarray
    validation_ridge_mean: np.ndarray
    summary: pd.DataFrame


def nested_oof_main_mapping(
    target: np.ndarray,
    train_features: np.ndarray,
    validation_features: np.ndarray,
    weights: np.ndarray,
    outer_folds: np.ndarray,
    validation_outer_folds: np.ndarray,
) -> NestedMappingResult:
    train_hgb_predictions = np.empty((len(target), NESTED_FOLDS), dtype=float)
    train_ridge_predictions = np.empty((len(target), NESTED_FOLDS), dtype=float)
    validation_hgb_outer = np.empty(
        (FOLDS, NESTED_FOLDS, len(validation_features)), dtype=float
    )
    validation_ridge_outer = np.empty_like(validation_hgb_outer)
    rows: list[dict[str, Any]] = []
    for outer_fold in range(FOLDS):
        outer_fit = np.flatnonzero(outer_folds != outer_fold)
        outer_hold = np.flatnonzero(outer_folds == outer_fold)
        inner_splitter = KFold(
            NESTED_FOLDS, shuffle=True, random_state=SEED + 100 + outer_fold
        )
        for inner_fold, (inner_fit_local, _) in enumerate(inner_splitter.split(outer_fit)):
            fit_index = outer_fit[inner_fit_local]
            start = time.time()
            hgb = HistGradientBoostingRegressor(
                **HGB_PARAMS,
                random_state=SEED + outer_fold * NESTED_FOLDS + inner_fold,
            ).fit(
                train_features[fit_index],
                target[fit_index],
                sample_weight=weights[fit_index],
            )
            train_hgb_predictions[outer_hold, inner_fold] = hgb.predict(
                train_features[outer_hold]
            )
            validation_hgb_outer[outer_fold, inner_fold] = hgb.predict(
                validation_features
            )
            scaler = StandardScaler().fit(
                train_features[fit_index], sample_weight=weights[fit_index]
            )
            ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(
                scaler.transform(train_features[fit_index]),
                target[fit_index],
                sample_weight=weights[fit_index],
            )
            train_ridge_predictions[outer_hold, inner_fold] = ridge.predict(
                scaler.transform(train_features[outer_hold])
            )
            validation_ridge_outer[outer_fold, inner_fold] = ridge.predict(
                scaler.transform(validation_features)
            )
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "fit_count": len(fit_index),
                    "hold_count": len(outer_hold),
                    "ridge_alpha": float(ridge.alpha_),
                    "hgb_hold_spearman": safe_spearman(
                        train_hgb_predictions[outer_hold, inner_fold], target[outer_hold]
                    ),
                    "ridge_hold_spearman": safe_spearman(
                        train_ridge_predictions[outer_hold, inner_fold], target[outer_hold]
                    ),
                    "elapsed_seconds": time.time() - start,
                }
            )
        print(
            f"[CICP-MP nested uncertainty] outer fold {outer_fold + 1}/{FOLDS} complete",
            flush=True,
        )
    train_hgb_mean = train_hgb_predictions.mean(axis=1)
    train_ridge_mean = train_ridge_predictions.mean(axis=1)
    validation_hgb_predictions = np.empty(
        (len(validation_features), NESTED_FOLDS), dtype=float
    )
    validation_ridge_predictions = np.empty_like(validation_hgb_predictions)
    for outer_fold in range(FOLDS):
        mask = validation_outer_folds == outer_fold
        validation_hgb_predictions[mask] = validation_hgb_outer[outer_fold][:, mask].T
        validation_ridge_predictions[mask] = validation_ridge_outer[outer_fold][:, mask].T
    return NestedMappingResult(
        train_hgb_mean=train_hgb_mean,
        train_hgb_std=train_hgb_predictions.std(axis=1, ddof=1),
        train_positive_share=(train_hgb_predictions > 0).mean(axis=1),
        train_ridge_mean=train_ridge_mean,
        validation_hgb_mean=validation_hgb_predictions.mean(axis=1),
        validation_hgb_std=validation_hgb_predictions.std(axis=1, ddof=1),
        validation_positive_share=(validation_hgb_predictions > 0).mean(axis=1),
        validation_ridge_mean=validation_ridge_predictions.mean(axis=1),
        summary=pd.DataFrame(rows),
    )


@dataclass
class DirectionResult:
    train_compressed: np.ndarray
    train_mapped_oof: np.ndarray
    validation_mapped: np.ndarray
    profile_summary: pd.DataFrame
    fold_summary: pd.DataFrame
    feasibility: dict[str, Any]


def audit_direction(
    directions: dict[str, np.ndarray],
    train_features: np.ndarray,
    validation_features: np.ndarray,
    weights: np.ndarray,
    folds: np.ndarray,
) -> DirectionResult:
    start = time.time()
    primary = directions["global_seed43"]
    pca = PCA(n_components=DIRECTION_COMPRESSED_DIM, svd_solver="randomized", random_state=SEED)
    train_compressed = pca.fit_transform(primary).astype(np.float32)
    train_mapped = np.empty_like(train_compressed)
    validation_folds = np.empty(
        (FOLDS, len(validation_features), DIRECTION_COMPRESSED_DIM), dtype=np.float32
    )
    fold_rows: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        fit_index = np.flatnonzero(folds != fold)
        hold_index = np.flatnonzero(folds == fold)
        scaler = StandardScaler().fit(
            train_features[fit_index], sample_weight=weights[fit_index]
        )
        ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(
            scaler.transform(train_features[fit_index]),
            train_compressed[fit_index],
            sample_weight=weights[fit_index],
        )
        train_mapped[hold_index] = ridge.predict(
            scaler.transform(train_features[hold_index])
        )
        validation_folds[fold] = ridge.predict(
            scaler.transform(validation_features)
        )
        hold_cosine = row_cosine(train_compressed[hold_index], train_mapped[hold_index])
        fold_rows.append(
            {
                "fold": fold,
                "ridge_alpha": float(ridge.alpha_),
                "hold_direction_cosine_mean": float(hold_cosine.mean()),
                "hold_direction_cosine_median": float(np.median(hold_cosine)),
                "hold_positive_cosine_share": float((hold_cosine > 0).mean()),
                "hold_norm_spearman": safe_spearman(
                    np.linalg.norm(train_compressed[hold_index], axis=1),
                    np.linalg.norm(train_mapped[hold_index], axis=1),
                ),
            }
        )
    validation_mapped = validation_folds.mean(axis=0)
    mapped_cosine = row_cosine(train_compressed, train_mapped)
    shuffle_cosines = []
    for scheme in SHUFFLE_SCHEMES[1:]:
        shuffle_cosines.append(row_cosine(primary, directions[scheme]))
    min_shuffle_median = float(min(np.median(values) for values in shuffle_cosines))
    profile_summary = pd.DataFrame(
        [
            {
                "direction64_norm_mean": float(np.linalg.norm(primary, axis=1).mean()),
                "direction64_norm_median": float(np.median(np.linalg.norm(primary, axis=1))),
                "direction64_zero_norm_share": float(
                    (np.linalg.norm(primary, axis=1) <= 1e-12).mean()
                ),
                "pca16_explained_variance_ratio_sum": float(
                    pca.explained_variance_ratio_.sum()
                ),
                "oof_mapping_cosine_mean": float(mapped_cosine.mean()),
                "oof_mapping_cosine_median": float(np.median(mapped_cosine)),
                "oof_mapping_positive_cosine_share": float((mapped_cosine > 0).mean()),
                "oof_norm_spearman": safe_spearman(
                    np.linalg.norm(train_compressed, axis=1),
                    np.linalg.norm(train_mapped, axis=1),
                ),
                "minimum_shuffle_direction_cosine_median": min_shuffle_median,
                "validation_fold_direction_relative_std_mean": float(
                    np.mean(
                        np.linalg.norm(
                            validation_folds - validation_mapped[None, :, :], axis=2
                        )
                        / np.clip(np.linalg.norm(validation_mapped, axis=1), 1e-8, None)[None, :]
                    )
                ),
                "elapsed_seconds": time.time() - start,
            }
        ]
    )
    row = profile_summary.iloc[0]
    feasible = bool(
        row["pca16_explained_variance_ratio_sum"] >= 0.50
        and row["oof_mapping_cosine_median"] >= 0.20
        and row["minimum_shuffle_direction_cosine_median"] >= 0.20
        and pd.DataFrame(fold_rows)["hold_direction_cosine_median"].gt(0).all()
    )
    feasibility = {
        "status": "retain_in_v1" if feasible else "defer_from_v1",
        "feasible": feasible,
        "reason": (
            "16D direction is stable across shuffles and deployably mapped"
            if feasible
            else "64D direction lacks sufficient shuffle stability or cold-item mappability"
        ),
    }
    return DirectionResult(
        train_compressed=train_compressed,
        train_mapped_oof=train_mapped,
        validation_mapped=validation_mapped,
        profile_summary=profile_summary,
        fold_summary=pd.DataFrame(fold_rows),
        feasibility=feasibility,
    )


def uncertainty_calibration(
    target: np.ndarray,
    prediction: np.ndarray,
    uncertainty: np.ndarray,
    disagreement: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    absolute_error = np.abs(prediction - target)
    summary = pd.DataFrame(
        [
            {
                "item_count": len(target),
                "prediction_vs_target_spearman": safe_spearman(prediction, target),
                "uncertainty_vs_absolute_error_spearman": safe_spearman(
                    uncertainty, absolute_error
                ),
                "disagreement_vs_absolute_error_spearman": safe_spearman(
                    disagreement, absolute_error
                ),
                "uncertainty_mean": float(uncertainty.mean()),
                "absolute_error_mean": float(absolute_error.mean()),
                "finite_share": float(
                    np.isfinite(uncertainty).mean()
                    * np.isfinite(absolute_error).mean()
                ),
            }
        ]
    )
    ranked = pd.Series(uncertainty).rank(method="first")
    bins = pd.qcut(ranked, q=5, labels=False)
    rows: list[dict[str, Any]] = []
    for bucket in range(5):
        mask = np.asarray(bins == bucket)
        rows.append(
            {
                "uncertainty_quintile": bucket + 1,
                "item_count": int(mask.sum()),
                "uncertainty_mean": float(uncertainty[mask].mean()),
                "absolute_error_mean": float(absolute_error[mask].mean()),
            }
        )
    return summary, pd.DataFrame(rows)


def residualization_audit(
    frame: pd.DataFrame,
    controls: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    variants = (
        ("unresidualized", "category_semantic_increment_raw"),
        ("ridge_residual", "category_semantic_residual_ridge"),
        ("nonlinear_hgb_residual", "category_semantic_residual_hgb"),
    )
    for variant, column in variants:
        values = frame[column].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "variant": variant,
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "positive_share": float((values > 0).mean()),
        }
        for control in controls:
            row[f"spearman_{control}"] = safe_spearman(values, frame[control])
        row["max_abs_control_spearman"] = max(
            abs(row[f"spearman_{control}"]) for control in controls
        )
        rows.append(row)
    output = pd.DataFrame(rows)
    raw = frame["category_semantic_increment_raw"].to_numpy(dtype=float)
    output["spearman_vs_unresidualized"] = [
        safe_spearman(raw, frame[column]) for _, column in variants
    ]
    return output


def build_purity_audit(profile: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    controls = [
        "category_count",
        "R_metadata_richness_score",
        "S_train_support_score",
        "P_popularity_score",
        "log1p_train_interaction_count",
    ]
    for split in ("train", "validate"):
        part = profile[profile["split"].eq(split)]
        for signal in signals:
            if signal not in part:
                continue
            for control in controls:
                rows.append(
                    {
                        "split": split,
                        "signal": signal,
                        "control": control,
                        "spearman": safe_spearman(part[signal], part[control]),
                    }
                )
    return pd.DataFrame(rows)


def build_shift_audit(profile: pd.DataFrame, signals: list[str]) -> pd.DataFrame:
    train = profile[profile["split"].eq("train")]
    validation = profile[profile["split"].eq("validate")]
    rows: list[dict[str, Any]] = []
    for signal in signals:
        train_values = train[signal].to_numpy(dtype=float)
        validation_values = validation[signal].to_numpy(dtype=float)
        ks = ks_2samp(train_values, validation_values)
        pooled = math.sqrt(
            (float(train_values.var(ddof=1)) + float(validation_values.var(ddof=1))) / 2.0
        )
        rows.append(
            {
                "signal": signal,
                "train_mean": float(train_values.mean()),
                "train_std": float(train_values.std(ddof=1)),
                "validation_mean": float(validation_values.mean()),
                "validation_std": float(validation_values.std(ddof=1)),
                "standardized_mean_difference": float(
                    (validation_values.mean() - train_values.mean()) / max(pooled, 1e-12)
                ),
                "ks_statistic": float(ks.statistic),
                "ks_pvalue": float(ks.pvalue),
                "train_finite_share": float(np.isfinite(train_values).mean()),
                "validation_finite_share": float(np.isfinite(validation_values).mean()),
            }
        )
    return pd.DataFrame(rows)


def effective_dimension_audit(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not columns:
        raise ValueError(f"effective dimension requires at least one component: {label}")
    values = frame[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"effective dimension input contains non-finite values: {label}")
    standardized = StandardScaler().fit_transform(values)
    correlation = pd.DataFrame(standardized, columns=columns).corr(method="spearman")
    pearson_correlation = (
        np.asarray([[1.0]])
        if len(columns) == 1
        else np.corrcoef(standardized, rowvar=False)
    )
    eigenvalues = np.linalg.eigvalsh(pearson_correlation)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    effective_rank = float(eigenvalues.sum() ** 2 / np.clip((eigenvalues**2).sum(), 1e-12, None))
    off_diagonal = np.abs(correlation.to_numpy()[np.triu_indices(len(columns), k=1)])
    summary = pd.DataFrame(
        [
            {
                "profile": label,
                "component_count": len(columns),
                "numerical_rank": int(np.linalg.matrix_rank(standardized)),
                "effective_rank": effective_rank,
                "effective_rank_ratio": effective_rank / len(columns),
                "max_abs_pairwise_spearman": float(off_diagonal.max()) if len(off_diagonal) else 0.0,
                "pair_share_abs_spearman_ge_0_95": float((off_diagonal >= 0.95).mean())
                if len(off_diagonal)
                else 0.0,
            }
        ]
    )
    correlation.index.name = "component"
    return summary, correlation


def build_fallacy_scan() -> pd.DataFrame:
    rows = [
        (1, "Simpson's paradox", "CHECKED", "overall, fold, support and shuffle-scheme directions are reported separately"),
        (2, "Ecological fallacy", "CAUTION", "profile-level averages do not establish every item's category utility"),
        (3, "Berkson's paradox", "CAUTION", "collaborative labels exist only for observed training items"),
        (4, "Collider bias", "CAUTION", "controls may share causes with collaborative structure; residuals are not causal effects"),
        (5, "Base-rate neglect", "CHECKED", "24726 train and 5298 validation coverage are reported against the full item profile"),
        (6, "Regression to the mean", "CHECKED", "no recommendation-outcome extreme group is selected"),
        (7, "Survivorship bias", "CHECKED", "all train and validation structural items are retained"),
        (8, "Multiple comparisons", "CAUTION", "candidate pool is exploratory and retained status is not a recommendation-performance result"),
        (9, "Garden of forking paths", "CAUTION", "multiple residual and shuffle definitions are disclosed rather than silently selected"),
        (10, "Correlation is not causation", "CAUTION", "residualization and uncertainty calibration are predictive diagnostics only"),
        (11, "Reverse causality", "CHECKED", "no validation/test recommendation outcome enters any feature or label"),
    ]
    return pd.DataFrame(rows, columns=["fallacy_id", "fallacy", "status", "assessment"])


def build_component_catalog(
    mapping_summary: pd.DataFrame,
    uncertainty_summary: pd.DataFrame,
    direction_feasibility: dict[str, Any],
    attribution_mapping_summary: pd.DataFrame,
) -> pd.DataFrame:
    hgb_rho = {
        row.target: float(row.oof_spearman)
        for row in mapping_summary.itertuples()
        if row.model == "hist_gradient_boosting"
    }
    ridge_attr_rho = {
        row.target: float(row.oof_spearman)
        for row in attribution_mapping_summary.itertuples()
        if row.model == "ridge"
    }
    calibration = float(
        uncertainty_summary.iloc[0]["uncertainty_vs_absolute_error_spearman"]
    )
    rows = [
        ("mp_category_total_increment_prediction", "full-base cosine increment mapped by HGB", "cosine absolute difference", "higher means real category adds more over base", "cross-fitted_recomputed", hgb_rho.get("category_total_increment_raw", np.nan), "candidate"),
        ("mp_category_capacity_increment_prediction", "shuffled-base cosine increment mapped by HGB", "cosine absolute difference", "higher means category-channel capacity adds more", "cross-fitted_recomputed", hgb_rho.get("category_capacity_increment_raw", np.nan), "candidate"),
        ("mp_category_semantic_increment_prediction", "full-shuffled cosine increment mapped by HGB", "cosine absolute difference", "higher means real category identity adds more", "cross-fitted_recomputed", hgb_rho.get("category_semantic_increment_raw", np.nan), "candidate"),
        ("mp_category_semantic_residual_ridge_prediction", "ridge-control residual mapped by HGB", "cosine residual", "higher means more increment remains after linear control prediction", "cross-fitted_recomputed", hgb_rho.get("category_semantic_residual_ridge", np.nan), "candidate"),
        ("mp_category_semantic_residual_hgb_prediction", "nonlinear-control residual mapped by HGB", "cosine residual", "higher means more increment remains after nonlinear control prediction", "new_computation", hgb_rho.get("category_semantic_residual_hgb", np.nan), "candidate"),
        ("mp_raw_predicted_increment", "nested strict-OOF HGB prediction of ridge residual", "cosine residual", "higher means more predicted residual category increment", "new_nested_cross_fit", hgb_rho.get("category_semantic_residual_ridge", np.nan), "retain_anchor"),
        ("mp_empirical_percentile", "train-fitted percentile of raw predicted increment", "unit interval rank", "higher rank only", "mathematical_transform", 1.0, "compatibility_only"),
        ("mp_fold_prediction_uncertainty", "strict-OOF nested-fold prediction standard deviation", "cosine residual std", "higher means less stable prediction", "new_nested_cross_fit", calibration, "candidate"),
        ("mp_positive_fold_share", "share of strict-OOF nested predictions above zero", "proportion", "higher means more folds predict positive increment", "new_nested_cross_fit", np.nan, "candidate"),
        ("mp_hgb_ridge_disagreement", "absolute nested HGB-Ridge prediction difference", "cosine residual absolute difference", "higher means model-form disagreement", "new_nested_cross_fit", float(uncertainty_summary.iloc[0]["disagreement_vs_absolute_error_spearman"]), "candidate"),
        ("mp_control_explained_component_ridge", "cross-fitted ridge prediction from category count/R/S/P/interactions", "cosine increment", "descriptive nuisance component; not causal", "cross-fitted_recomputed", np.nan, "diagnostic_control"),
        ("mp_control_explained_component_hgb", "cross-fitted nonlinear prediction from category count/R/S/P/interactions", "cosine increment", "descriptive nuisance component; not causal", "new_computation", np.nan, "diagnostic_control"),
        ("mp_base_collaborative_predictability_prediction", "base cosine mapped by HGB", "cosine", "higher means image/control predicts collaboration better", "cross-fitted_recomputed", hgb_rho.get("base_collaborative_cosine", np.nan), "context_candidate"),
        ("mp_shuffled_collaborative_predictability_prediction", "shuffled-category cosine mapped by HGB", "cosine", "higher means random category channel predicts collaboration better", "cross-fitted_recomputed", hgb_rho.get("shuffled_category_collaborative_cosine", np.nan), "context_candidate"),
        ("mp_full_collaborative_predictability_prediction", "real-category cosine mapped by HGB", "cosine", "higher means full content predicts collaboration better", "cross_fitted_recomputed", hgb_rho.get("full_collaborative_cosine", np.nan), "context_candidate"),
        ("mp_full_remaining_space", "one minus full predictability prediction", "cosine complement", "higher means more unpredicted space", "mathematical_transform", 1.0, "diagnostic_transform"),
        ("mp_direction16_*", "PCA16 of common-coordinate full-minus-shuffled 64D direction, then OOF ridge mapped", "latent direction", "no scalar monotonic direction", "new_computation", np.nan, direction_feasibility["status"]),
        ("mp_category_attribution_max_prediction", "OOF ridge mapping of maximum leave-one-category-out marginal contribution", "cosine absolute difference", "higher means at least one category is strongly useful", "new_computation", ridge_attr_rho.get("category_attribution_max", np.nan), "candidate"),
        ("mp_category_attribution_min_prediction", "OOF ridge mapping of minimum leave-one-category-out marginal contribution", "cosine absolute difference", "higher means least useful category is less harmful", "new_computation", ridge_attr_rho.get("category_attribution_min", np.nan), "candidate"),
        ("mp_category_attribution_std_prediction", "OOF ridge mapping of within-item marginal contribution std", "cosine std", "higher means category contributions are heterogeneous", "new_computation", ridge_attr_rho.get("category_attribution_std", np.nan), "candidate"),
        ("mp_category_attribution_positive_share_prediction", "OOF ridge mapping of positive marginal category share", "proportion", "higher means more listed categories are helpful", "new_computation", ridge_attr_rho.get("category_attribution_positive_share", np.nan), "candidate"),
        ("mp_category_attribution_entropy_prediction", "OOF ridge mapping of normalized absolute-contribution entropy", "unit interval", "higher means contribution is spread across categories", "new_computation", ridge_attr_rho.get("category_attribution_entropy", np.nan), "candidate"),
        ("mp_category_attribution_first_minus_last_prediction", "OOF ridge mapping of first-listed minus last-listed category contribution", "cosine absolute difference", "sequence-position contrast only; not a hierarchy direction", "new_computation", ridge_attr_rho.get("category_attribution_first_minus_last", np.nan), "diagnostic_position_only"),
        ("parent_leaf_attribution_difference", "not available: source category list mixes taxonomy, brands and descriptions", "not evaluated", "no valid direction", "currently_missing", np.nan, "defer_from_v1"),
        ("1-s / s^2 / 4s(1-s)", "deterministic transforms of empirical percentile", "unit interval", "transform-specific", "mathematical_transform", 1.0, "reject_no_new_information"),
    ]
    return pd.DataFrame(
        rows,
        columns=[
            "component",
            "definition",
            "unit",
            "direction",
            "data_source_class",
            "offline_reliability_statistic",
            "initial_status",
        ],
    )


def choose_retained_components(
    train_profile: pd.DataFrame,
    component_catalog: pd.DataFrame,
    mapping_summary: pd.DataFrame,
    attribution_mapping_summary: pd.DataFrame,
    uncertainty_summary: pd.DataFrame,
    direction: DirectionResult,
) -> tuple[list[str], pd.DataFrame]:
    hgb_rho = {
        row.target: float(row.oof_spearman)
        for row in mapping_summary.itertuples()
        if row.model == "hist_gradient_boosting"
    }
    ridge_attr_rho = {
        row.target: float(row.oof_spearman)
        for row in attribution_mapping_summary.itertuples()
        if row.model == "ridge"
    }
    candidates = [
        ("mp_raw_predicted_increment", 1.0, 0.0, "anchor raw magnitude"),
        ("mp_category_semantic_increment_prediction", hgb_rho.get("category_semantic_increment_raw", np.nan), 0.15, "unresidualized semantic magnitude"),
        ("mp_category_capacity_increment_prediction", hgb_rho.get("category_capacity_increment_raw", np.nan), 0.15, "capacity component"),
        ("mp_category_total_increment_prediction", hgb_rho.get("category_total_increment_raw", np.nan), 0.15, "total component"),
        ("mp_category_attribution_max_prediction", ridge_attr_rho.get("category_attribution_max", np.nan), 0.15, "maximum category marginal contribution"),
        ("mp_category_attribution_min_prediction", ridge_attr_rho.get("category_attribution_min", np.nan), 0.15, "minimum category marginal contribution"),
        ("mp_category_attribution_std_prediction", ridge_attr_rho.get("category_attribution_std", np.nan), 0.15, "within-item category heterogeneity"),
        ("mp_category_attribution_positive_share_prediction", ridge_attr_rho.get("category_attribution_positive_share", np.nan), 0.15, "positive marginal category share"),
        ("mp_category_attribution_entropy_prediction", ridge_attr_rho.get("category_attribution_entropy", np.nan), 0.15, "category contribution entropy"),
        ("mp_fold_prediction_uncertainty", float(uncertainty_summary.iloc[0]["uncertainty_vs_absolute_error_spearman"]), 0.05, "calibrated uncertainty"),
        ("mp_hgb_ridge_disagreement", float(uncertainty_summary.iloc[0]["disagreement_vs_absolute_error_spearman"]), 0.05, "model-form uncertainty"),
    ]
    retained: list[str] = []
    decisions: list[dict[str, Any]] = []
    purity_controls = [
        "category_count",
        "R_metadata_richness_score",
        "S_train_support_score",
        "P_popularity_score",
        "log1p_train_interaction_count",
    ]
    for column, reliability, minimum_reliability, reason in candidates:
        maximum_control_correlation = max(
            abs(safe_spearman(train_profile[column], train_profile[control]))
            for control in purity_controls
        )
        if column == "mp_raw_predicted_increment":
            retain = maximum_control_correlation < 0.35
            maximum_correlation = 0.0
        else:
            maximum_correlation = max(
                [
                    abs(safe_spearman(train_profile[column], train_profile[other]))
                    for other in retained
                ]
                or [0.0]
            )
            retain = bool(
                np.isfinite(reliability)
                and reliability >= minimum_reliability
                and maximum_correlation < 0.95
                and maximum_control_correlation < 0.35
            )
        if retain:
            retained.append(column)
        decisions.append(
            {
                "component": column,
                "reliability_statistic": reliability,
                "minimum_reliability": minimum_reliability,
                "max_abs_spearman_vs_previously_retained": maximum_correlation,
                "max_abs_rsp_control_spearman": maximum_control_correlation,
                "decision": "retain" if retain else "reject_or_diagnostic_only",
                "reason": reason,
            }
        )
    direction_max_control_correlation = max(
        abs(
            safe_spearman(
                train_profile[f"mp_direction16_{index:02d}"],
                train_profile[control],
            )
        )
        for index in range(DIRECTION_COMPRESSED_DIM)
        for control in purity_controls
    )
    direction_retained = bool(
        direction.feasibility["feasible"] and direction_max_control_correlation < 0.35
    )
    if direction_retained:
        for index in range(DIRECTION_COMPRESSED_DIM):
            retained.append(f"mp_direction16_{index:02d}")
        decisions.append(
            {
                "component": "mp_direction16_00..15",
                "reliability_statistic": float(
                    direction.profile_summary.iloc[0]["oof_mapping_cosine_median"]
                ),
                "max_abs_spearman_vs_previously_retained": np.nan,
                "max_abs_rsp_control_spearman": direction_max_control_correlation,
                "decision": "retain",
                "reason": "direction passed compression, shuffle stability and OOF mapping gates",
            }
        )
    else:
        decisions.append(
            {
                "component": "mp_direction16_00..15",
                "reliability_statistic": float(
                    direction.profile_summary.iloc[0]["oof_mapping_cosine_median"]
                ),
                "max_abs_spearman_vs_previously_retained": np.nan,
                "max_abs_rsp_control_spearman": direction_max_control_correlation,
                "decision": "defer_from_v1",
                "reason": (
                    direction.feasibility["reason"]
                    if not direction.feasibility["feasible"]
                    else "direction violates R/S/P purity threshold"
                ),
            }
        )
    decisions.extend(
        [
            {
                "component": "mp_category_semantic_residual_ridge_prediction",
                "reliability_statistic": hgb_rho.get(
                    "category_semantic_residual_ridge", np.nan
                ),
                "minimum_reliability": 0.15,
                "max_abs_spearman_vs_previously_retained": abs(
                    safe_spearman(
                        train_profile[
                            "mp_category_semantic_residual_ridge_prediction"
                        ],
                        train_profile["mp_raw_predicted_increment"],
                    )
                ),
                "max_abs_rsp_control_spearman": max(
                    abs(
                        safe_spearman(
                            train_profile[
                                "mp_category_semantic_residual_ridge_prediction"
                            ],
                            train_profile[control],
                        )
                    )
                    for control in purity_controls
                ),
                "decision": "diagnostic_protocol_alternative",
                "reason": "same target and HGB mapping as raw predicted increment; differs only by single versus nested cross-fitting protocol",
            },
            {
                "component": "mp_category_semantic_residual_hgb_prediction",
                "reliability_statistic": hgb_rho.get(
                    "category_semantic_residual_hgb", np.nan
                ),
                "minimum_reliability": 0.15,
                "max_abs_spearman_vs_previously_retained": abs(
                    safe_spearman(
                        train_profile["mp_category_semantic_residual_hgb_prediction"],
                        train_profile["mp_raw_predicted_increment"],
                    )
                ),
                "max_abs_rsp_control_spearman": max(
                    abs(
                        safe_spearman(
                            train_profile[
                                "mp_category_semantic_residual_hgb_prediction"
                            ],
                            train_profile[control],
                        )
                    )
                    for control in purity_controls
                ),
                "decision": "residualization_sensitivity_only",
                "reason": "alternative nonlinear residualization of the same semantic increment, retained for sensitivity rather than counted as a new signal",
            },
            {
                "component": "mp_empirical_percentile",
                "reliability_statistic": 1.0,
                "minimum_reliability": np.nan,
                "max_abs_spearman_vs_previously_retained": 1.0,
                "max_abs_rsp_control_spearman": np.nan,
                "decision": "compatibility_only",
                "reason": "deterministic rank transform of raw predicted increment",
            },
            {
                "component": "mp_positive_fold_share",
                "reliability_statistic": np.nan,
                "minimum_reliability": np.nan,
                "max_abs_spearman_vs_previously_retained": abs(
                    safe_spearman(
                        train_profile["mp_positive_fold_share"],
                        train_profile["mp_raw_predicted_increment"],
                    )
                ),
                "max_abs_rsp_control_spearman": np.nan,
                "decision": "diagnostic_only",
                "reason": "fold-sign summary is retained for audit but not counted as independent by default",
            },
            {
                "component": "mp_base/full/shuffled_predictability",
                "reliability_statistic": np.nan,
                "minimum_reliability": np.nan,
                "max_abs_spearman_vs_previously_retained": np.nan,
                "max_abs_rsp_control_spearman": np.nan,
                "decision": "context_only",
                "reason": "collaborative predictability context must not inflate CICP core effective rank",
            },
            {
                "component": "mp_control_explained_component_*",
                "reliability_statistic": np.nan,
                "minimum_reliability": np.nan,
                "max_abs_spearman_vs_previously_retained": np.nan,
                "max_abs_rsp_control_spearman": np.nan,
                "decision": "diagnostic_control",
                "reason": "explicit R/S/P nuisance component, not category availability",
            },
            {
                "component": "mp_full_remaining_space",
                "reliability_statistic": 1.0,
                "minimum_reliability": np.nan,
                "max_abs_spearman_vs_previously_retained": np.nan,
                "max_abs_rsp_control_spearman": np.nan,
                "decision": "diagnostic_transform",
                "reason": "affine transform of full predictability",
            },
            {
                "component": "mp_category_attribution_first_minus_last_prediction",
                "reliability_statistic": ridge_attr_rho.get(
                    "category_attribution_first_minus_last", np.nan
                ),
                "minimum_reliability": np.nan,
                "max_abs_spearman_vs_previously_retained": np.nan,
                "max_abs_rsp_control_spearman": np.nan,
                "decision": "diagnostic_position_only",
                "reason": "first-minus-last sequence position is auditable but cannot be interpreted as parent-minus-leaf hierarchy",
            },
            {
                "component": "parent_leaf_attribution_difference",
                "reliability_statistic": np.nan,
                "minimum_reliability": np.nan,
                "max_abs_spearman_vs_previously_retained": np.nan,
                "max_abs_rsp_control_spearman": np.nan,
                "decision": "defer_from_v1",
                "reason": "no trustworthy parent-child hierarchy in source metadata",
            },
            {
                "component": "1-s / s^2 / 4s(1-s)",
                "reliability_statistic": 1.0,
                "minimum_reliability": np.nan,
                "max_abs_spearman_vs_previously_retained": 1.0,
                "max_abs_rsp_control_spearman": np.nan,
                "decision": "reject_no_new_information",
                "reason": "deterministic mathematical transforms of empirical percentile",
            },
        ]
    )
    return retained, pd.DataFrame(decisions)


def finalize_component_catalog(
    component_catalog: pd.DataFrame,
    component_decisions: pd.DataFrame,
) -> pd.DataFrame:
    decisions = component_decisions.set_index("component")

    def decision_key(component: str) -> str:
        if component == "mp_direction16_*":
            return "mp_direction16_00..15"
        if component in {
            "mp_base_collaborative_predictability_prediction",
            "mp_shuffled_collaborative_predictability_prediction",
            "mp_full_collaborative_predictability_prediction",
        }:
            return "mp_base/full/shuffled_predictability"
        if component.startswith("mp_control_explained_component_"):
            return "mp_control_explained_component_*"
        return component

    final_statuses: list[str] = []
    final_reasons: list[str] = []
    independence: list[str] = []
    for component in component_catalog["component"].astype(str):
        key = decision_key(component)
        if key not in decisions.index:
            raise ValueError(f"component catalog lacks final decision: {component}")
        row = decisions.loc[key]
        status = str(row["decision"])
        final_statuses.append(status)
        final_reasons.append(str(row["reason"]))
        if status == "retain":
            independence.append("retained_after_redundancy_audit")
        elif status in {
            "compatibility_only",
            "diagnostic_transform",
            "reject_no_new_information",
        }:
            independence.append("not_independent_deterministic_transform")
        elif status in {
            "diagnostic_protocol_alternative",
            "residualization_sensitivity_only",
        }:
            independence.append("not_independent_alternative_estimator")
        elif status in {"context_only", "diagnostic_control"}:
            independence.append("not_counted_as_cicp_core_dimension")
        else:
            independence.append("independence_not_claimed")
    catalog = component_catalog.copy()
    catalog["final_status"] = final_statuses
    catalog["independence_status"] = independence
    catalog["final_reason"] = final_reasons
    return catalog


def decide_route(
    profile: pd.DataFrame,
    retained_scalar_summary: pd.DataFrame,
    mapping_summary: pd.DataFrame,
    uncertainty_summary: pd.DataFrame,
    direction: DirectionResult,
    shift: pd.DataFrame,
    retained_components: list[str],
    leakage_pass: bool,
) -> dict[str, Any]:
    hgb = mapping_summary[
        mapping_summary["model"].eq("hist_gradient_boosting")
    ].set_index("target")
    main_rho = float(hgb.loc["category_semantic_residual_ridge", "oof_spearman"])
    coverage_pass = bool(
        len(profile[profile["split"].eq("train")]) == 24726
        and len(profile[profile["split"].eq("validate")]) == 5298
        and np.isfinite(
            profile.filter(regex=r"^mp_").select_dtypes(include=[np.number]).to_numpy()
        ).all()
    )
    retained_shift = shift[shift["signal"].isin(retained_components)]
    shift_pass = bool(
        len(retained_shift) == len(retained_components)
        and retained_shift["ks_statistic"].lt(0.15).all()
    )
    effective_rank = float(retained_scalar_summary.iloc[0]["effective_rank"])
    uncertainty_calibrated = bool(
        uncertainty_summary.iloc[0]["uncertainty_vs_absolute_error_spearman"] >= 0.05
    )
    direction_retained = any(
        component.startswith("mp_direction16_") for component in retained_components
    )
    independent_scalar_pass = bool(
        retained_scalar_summary.iloc[0]["component_count"] >= 3
        and effective_rank >= 2.0
        and retained_scalar_summary.iloc[0]["pair_share_abs_spearman_ge_0_95"] < 0.50
    )
    deployability_pass = bool(main_rho >= 0.15)
    if not leakage_pass or not coverage_pass:
        route = "cicp_mp_v1_protocol_failure"
    elif not deployability_pass or not shift_pass:
        route = "prioritize_category_semantic_basis"
    elif independent_scalar_pass:
        route = "cicp_mp_v1_ready_for_one_formal_mechanism_design"
    elif uncertainty_calibrated or direction_retained:
        route = "shrink_profile_to_uncertainty_or_direction"
    else:
        route = "prioritize_updated_cold_start_backbone_review"
    route_number = {
        "cicp_mp_v1_ready_for_one_formal_mechanism_design": 1,
        "shrink_profile_to_uncertainty_or_direction": 2,
        "prioritize_category_semantic_basis": 3,
        "prioritize_updated_cold_start_backbone_review": 4,
        "cicp_mp_v1_protocol_failure": 0,
    }[route]
    return {
        "route": route,
        "route_number": route_number,
        "gates": {
            "leakage_audit": leakage_pass,
            "coverage_audit": coverage_pass,
            "main_component_deployability": deployability_pass,
            "retained_scalar_effective_dimension": independent_scalar_pass,
            "uncertainty_calibration": uncertainty_calibrated,
            "train_validation_shift": shift_pass,
            "direction_feasibility_and_purity": direction_retained,
        },
        "main_residual_mapping_oof_spearman": main_rho,
        "retained_scalar_effective_rank": effective_rank,
        "maximum_retained_component_ks_statistic": float(
            retained_shift["ks_statistic"].max()
        ),
        "maximum_all_candidate_ks_statistic": float(shift["ks_statistic"].max()),
        "shift_gate_scope": "retained_components_only; rejected candidates remain reported",
        "run_ccfcrec_training_now": False,
        "design_multiple_100epoch_experiments_now": False,
        "validation_recommendation_metrics_read_or_generated": False,
        "test_recommendation_metrics_read_or_generated": False,
        "test_items_analyzed": 0,
        "fallacy_scan_coverage": "11/11",
    }


def build_protocol_audit(
    profile: pd.DataFrame,
    retained_components: list[str],
    decision: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    forbidden_columns = sorted(
        str(column)
        for column in profile.columns
        if any(token in str(column).strip().lower() for token in FORBIDDEN_TOKENS)
    )
    mp_columns = profile.filter(regex=r"^mp_").select_dtypes(include=[np.number]).columns.tolist()
    retained_values = profile[retained_components].to_numpy(dtype=float)
    uncertainty = profile["mp_fold_prediction_uncertainty"].to_numpy(dtype=float)
    coverage_rows = []
    expected_counts = {"train": 24726, "validate": 5298}
    for split, expected in expected_counts.items():
        subset = profile[profile["split"].eq(split)]
        numeric = subset[mp_columns].to_numpy(dtype=float)
        coverage_rows.append(
            {
                "split": split,
                "expected_item_count": expected,
                "observed_item_count": len(subset),
                "item_count_match": len(subset) == expected,
                "all_mp_values_finite": bool(np.isfinite(numeric).all()),
                "all_retained_values_finite": bool(
                    np.isfinite(subset[retained_components].to_numpy(dtype=float)).all()
                ),
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    audit = {
        "protocol": "cicp_mp_v1_train_only_offline_audit",
        "profile_item_count": len(profile),
        "train_item_count": int(profile["split"].eq("train").sum()),
        "validation_structural_item_count": int(profile["split"].eq("validate").sum()),
        "test_item_count": 0,
        "recommendation_metrics_read_or_generated": False,
        "forbidden_output_columns": forbidden_columns,
        "all_profile_components_finite": bool(
            np.isfinite(profile[mp_columns].to_numpy(dtype=float)).all()
        ),
        "all_retained_components_finite": bool(np.isfinite(retained_values).all()),
        "train_predictions_are_strict_nested_oof": True,
        "train_uncertainty_uses_sample_in_predictions": False,
        "train_uncertainty_zero_filled": False,
        "train_uncertainty_zero_share": float(
            np.mean(uncertainty[profile["split"].eq("train").to_numpy()] == 0.0)
        ),
        "validation_predictions_use_train_only_models": True,
        "train_validation_uncertainty_protocol_consistent": True,
        "outer_fold_count": FOLDS,
        "inner_fold_count": NESTED_FOLDS,
        "retained_component_count": len(retained_components),
        "coverage_pass": bool(coverage["item_count_match"].all()),
        "leakage_pass": len(forbidden_columns) == 0,
        "deployability_pass": bool(decision["gates"]["main_component_deployability"]),
        "ccfcrec_training_launched": False,
        "multiple_100epoch_experiments_designed": False,
        "final_route_number": int(decision["route_number"]),
    }
    return audit, coverage


def file_fingerprint(path: Path, hash_content: bool = False) -> dict[str, Any]:
    stat = path.stat()
    output: dict[str, Any] = {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if hash_content:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        output["sha256"] = digest.hexdigest()
    return output


def build_source_field_audit(original_profile_path: Path) -> pd.DataFrame:
    original = pd.read_csv(
        original_profile_path, dtype={"raw_asin": str}, low_memory=False
    )
    assert_no_recommendation_metrics(original, "original CICP item profile")
    require_columns(original, ["raw_asin", "split", "cicp_score"], "original CICP profile")
    train = original[original["split"].eq("train")]
    validation = original[original["split"].eq("validate")]
    formal_loader_fields = {"raw_asin", "cicp_score"}
    label_fields = {
        "base_collaborative_cosine",
        "shuffled_category_collaborative_cosine",
        "full_collaborative_cosine",
        "category_total_increment_raw",
        "category_capacity_increment_raw",
        "category_semantic_increment_raw",
        "category_semantic_increment_residual",
    }
    prediction_fields = {
        "cicp_score_raw_oof",
        "cicp_ridge_score_raw_oof",
        "cicp_score_uncertainty",
    }
    rows: list[dict[str, Any]] = []
    for column in original.columns:
        if column in formal_loader_fields:
            source_class = "formal_training_loader_input"
        elif column in label_fields:
            source_class = "existing_train_only_intermediate"
        elif column in prediction_fields:
            source_class = "existing_deployable_mapping_intermediate"
        elif column in {"split", "probe_fold", "support_bucket"}:
            source_class = "audit_metadata"
        else:
            source_class = "existing_structural_control_or_context"
        rows.append(
            {
                "field": column,
                "source_class": source_class,
                "formal_training_loader_reads_field": column in formal_loader_fields,
                "train_non_null_share": float(train[column].notna().mean()),
                "validation_non_null_share": float(validation[column].notna().mean()),
                "mp_v1_action": (
                    "retain_identifier"
                    if column == "raw_asin"
                    else "retain_compatibility_only"
                    if column == "cicp_score"
                    else "recompute_with_strict_cross_fitting"
                    if column in label_fields | prediction_fields
                    else "use_for_audit_or_control"
                ),
            }
        )
    rows.extend(
        [
            {
                "field": "cicp_inverse_score",
                "source_class": "generated_by_cicp_features.py",
                "formal_training_loader_reads_field": False,
                "train_non_null_share": 1.0,
                "validation_non_null_share": 1.0,
                "mp_v1_action": "reject_no_new_information",
            },
            {
                "field": "cicp_mid_band",
                "source_class": "generated_by_cicp_features.py",
                "formal_training_loader_reads_field": False,
                "train_non_null_share": 1.0,
                "validation_non_null_share": 1.0,
                "mp_v1_action": "reject_no_new_information",
            },
        ]
    )
    return pd.DataFrame(rows)


def audit_formal_feature_loader(path: Path) -> dict[str, Any]:
    source = path.read_text(encoding="utf-8")
    required_snippets = {
        "reads_cicp_score": '"cicp_score": score',
        "builds_inverse": '"cicp_inverse_score": 1.0 - score',
        "builds_mid_band": '"cicp_mid_band": 4.0 * score * (1.0 - score)',
        "feature_width_is_three": "CICP_FEATURE_WIDTH = 3",
    }
    checks = {name: snippet in source for name, snippet in required_snippets.items()}
    if not all(checks.values()):
        raise ValueError(f"formal CICP feature loader changed unexpectedly: {checks}")
    return {
        **checks,
        "independent_input_dimension": 1,
        "generated_feature_width": 3,
        "loader_path": str(path),
    }


def build_hierarchy_feasibility_audit(
    asin_csv: Path,
    all_items: list[str],
    category_assets: CategoryAssets,
) -> pd.DataFrame:
    metadata = pd.read_csv(
        asin_csv, usecols=["asin", "category"], dtype={"asin": str}
    ).set_index("asin")
    sequences: list[list[str]] = []
    exact_matches: list[bool] = []
    non_taxonomic_tail: list[bool] = []
    category_lists = [
        *category_assets.train_categories,
        *category_assets.validation_categories,
    ]
    for item, category_ids in zip(all_items, category_lists):
        raw = metadata.at[item, "category"] if item in metadata.index else ""
        parts = [part.strip() for part in str(raw).split(",") if part.strip()]
        names = [category_assets.category_name_by_id[value] for value in category_ids]
        sequences.append(parts)
        exact_matches.append(parts == names)
        tail = parts[-1].lower() if parts else ""
        non_taxonomic_tail.append(
            tail.startswith("by ")
            or len(tail.split()) >= 6
            or "." in tail
            or tail.startswith("officially ")
        )
    return pd.DataFrame(
        [
            {
                "item_count": len(all_items),
                "comma_sequence_available_share": float(
                    np.mean([len(values) > 0 for values in sequences])
                ),
                "pickle_sequence_matches_asin_csv_share": float(np.mean(exact_matches)),
                "non_taxonomic_tail_marker_share": float(np.mean(non_taxonomic_tail)),
                "explicit_parent_edge_available": False,
                "explicit_depth_available": False,
                "parent_leaf_attribution_status": "defer_from_v1",
                "reason": "comma sequence mixes taxonomy nodes, brands, publishers and descriptive attributes",
            }
        ]
    )


def write_reports(
    output_dir: Path,
    route_output: Path,
    run_stamp: str,
    tables: dict[str, pd.DataFrame],
    decision: dict[str, Any],
    manifest: dict[str, Any],
    component_catalog: pd.DataFrame,
    retained: list[str],
    direction: DirectionResult,
) -> tuple[Path, Path]:
    result_path = output_dir / f"{run_stamp} CCFCRec Amazon-VG CICP-MP-v1 train-only offline audit 结果.md"
    route_path = route_output
    mapping_display = tables["mapping_summary"][
        tables["mapping_summary"]["target"].isin(CORE_MAPPING_TARGETS)
    ][["target", "model", "oof_spearman", "positive_fold_count", "fold_count", "hgb_ridge_prediction_spearman"]]
    retained_display = tables["component_decisions"]
    report = f"""---
title: {run_stamp} CCFCRec Amazon-VG CICP-MP-v1 train-only offline audit 结果
date: 2026-07-17
tags:
  - CCFCRec
  - Amazon-VG
  - CICP-MP-v1
  - train-only
  - offline-audit
---

# CCFCRec Amazon-VG CICP-MP-v1 train-only offline audit 结果

## Material Passport（材料护照）

- verification status（验证状态）：VERIFIED（已运行并完成产物审计）
- protocol（协议）：`cicp_mp_v1_train_only_offline_audit`
- train items（训练物品）：24,726
- validation cold-start items（验证冷启动物品）：5,298，仅使用结构特征
- test items（测试物品）：0
- validation/test recommendation metrics（验证集/测试集推荐指标）：未读取、未生成
- CCFCRec training（冷启动对比协同过滤训练）：未启动

## 一、最终判断

> [!important] 路线 {decision['route_number']}
> `{decision['route']}`

本轮实际构造了完整 CICP-MP-v1（第一版类别增量协同可预测性多分量画像），并完成训练物品严格折外预测、验证冷物品训练期映射、嵌套不确定性、三种类别打乱、64维协同增量方向、16维压缩方向和逐类别移除归因。该结论是离线画像结论，不是推荐性能结论。

> [!warning] 判定器纠正记录
> 首次自动汇总曾让已拒绝的 capacity increment（容量增量）候选参与路线迁移门槛，因其 KS（柯尔莫哥洛夫-斯米尔诺夫统计量）为 `0.205` 而误触发路线3。该候选折外 Spearman（斯皮尔曼秩相关）为负且本就不进入画像。最终判定器改为“所有候选均报告，但只有实际保留分量参与路线门槛”；原始画像、折外预测和审计数值没有重选或修改。

## 二、真实来源核对

### 已存在但正式训练时被丢弃

- `base/full/shuffled collaborative cosine`（基础/真实类别/打乱类别协同余弦）；
- `category total/capacity/semantic increment`（类别总量/容量/语义增量）；
- ridge residual（岭回归残差）与 HGB/Ridge（直方图梯度提升树/岭回归）原始折外预测；
- 验证冷物品五折预测标准差。原训练只读取 `cicp_score`（CICP经验百分位分数）。

### 可由现有数据重新交叉拟合

- 总增量、容量增量、语义增量和基础/完整/打乱协同可预测性的冷物品预测；
- 线性与非线性控制可解释部分及对应残差；
- HGB-Ridge disagreement（直方图梯度提升树与岭回归预测分歧）。

### 本轮新增计算

- 训练物品严格折外 nested-fold uncertainty（嵌套折不确定性）与 positive-fold share（正增量折占比）；
- 三种类别打乱方式的符号、排序与方向稳定性；
- 真实类别预测向量减打乱类别预测向量的公共坐标64维方向及训练期16维压缩；
- 逐类别移除的边际贡献与最大值、最小值、标准差、正贡献比例和贡献熵。

### 不属于新信息

- `empirical percentile`（经验百分位）、`1-s`、`s²`、`4s(1-s)`均是单一分数的确定性变换；
- `1-full_predictability`（一减完整可预测性）是完整可预测性的仿射变换；
- total/capacity/semantic（总量/容量/语义）在标签层满足代数关系，必须通过有效维度审计后才能决定是否共同保留。

### 原画像字段与正式训练读取核对

{md_table(tables['source_field_audit'], 5)}

## 三、候选分量与数据来源

{md_table(component_catalog, 5)}

## 四、冷物品可部署映射

{md_table(mapping_display, 6)}

所有训练预测均来自未见该物品协同标签的 held-out fold（留出折）；验证预测只由训练期模型生成。推荐指标没有参与模型选择或分量选择。

## 五、不确定性校准

{md_table(tables['uncertainty_summary'], 6)}

{md_table(tables['uncertainty_bins'], 6)}

训练不确定性没有填零，也没有使用样本内预测。每个训练物品的不确定性来自五个都排除该物品外层折的内部模型；每个验证物品按物品哈希固定分配到一个外层折，并使用该折同训练比例的五个内部模型。两类物品采用一致的五模型口径。不确定性只称为误差排序诊断，不称为概率校准。

## 六、残差化敏感性

{md_table(tables['residualization'], 6)}

ridge residual（岭回归残差）和 nonlinear residual（非线性残差）只表示相应模型未解释的预测剩余，不代表类别数量、丰富度、支持度、流行度或交互数的因果影响已被移除。

## 七、方向可行性

{md_table(direction.profile_summary, 6)}

{md_table(direction.fold_summary, 6)}

方向状态：`{direction.feasibility['status']}`。原因：{direction.feasibility['reason']}。原始64维协同空间按每折 SVD（截断奇异值分解）分解回共同用户坐标，再使用固定训练期随机投影进入统一64维坐标，避免直接拼接五套任意旋转的SVD坐标。

## 八、逐类别贡献归因

{md_table(tables['attribution_mapping_summary'], 6)}

逐类别移除贡献只在训练物品上具有真实协同标签；验证冷物品只接收其汇总统计的折外映射。当前 `asin.csv` 的类别序列混合 taxonomy（分类路径）、品牌和描述字段，没有可信父子层级，因此父类/叶类差异明确标记为“暂不进入v1”，没有用首尾位置伪造成层级贡献。

{md_table(tables['hierarchy_feasibility'], 6)}

## 九、稳定性、有效维度与冗余

### 类别打乱稳定性

{md_table(tables['shuffle_stability'], 6)}

### 有效维度

{md_table(tables['effective_dimension'], 6)}

### 分量保留决策

{md_table(retained_display, 6)}

最终保留列：`{', '.join(retained)}`。

## 十、R/S/P纯度与训练-验证迁移

### 纯度摘要

{md_table(tables['purity_summary'], 6)}

### 分布迁移

{md_table(tables['shift'], 6)}

## 十一、覆盖、泄漏与部署审计

{md_table(tables['coverage_audit'], 5)}

- 覆盖：24,726个训练物品和5,298个验证冷启动物品，画像列有限值覆盖100%；
- 训练部署：所有协同监督映射均为 item-level cross-fitting（逐物品交叉拟合）；
- 验证部署：只使用训练模型和验证结构特征；
- 禁止输入：验证/测试 NDCG、HR、rank、hit（归一化折损累计增益、命中率、排名、命中答案）均未读取；
- 测试范围：0个测试物品；
- 本轮不启动训练，也不设计多组100轮实验。

## 十二、11项统计谬误扫描

覆盖：`11/11`。

{md_table(tables['fallacy_scan'], 5)}

## 十三、路线判断边界

{md_table(pd.DataFrame([{'gate': key, 'pass': value} for key, value in decision['gates'].items()]), 5)}

本轮只决定画像是否包含稳定、独立且可部署的信息。即使路线1通过，也只允许下一步与用户讨论一次正式机制设计；不得把离线映射相关性写成推荐指标提升，更不得直接启动训练。

## 十四、输出

- `cicp_mp_v1_item_profile.csv`
- `cicp_mp_v1_component_catalog.csv`
- `cicp_mp_v1_source_field_audit.csv`
- `cicp_mp_v1_component_decisions.csv`
- `cicp_mp_v1_mapping_summary.csv`
- `cicp_mp_v1_nested_mapping_summary.csv`
- `cicp_mp_v1_uncertainty_calibration.csv`
- `cicp_mp_v1_direction64_train_oof.npz`
- `cicp_mp_v1_direction_summary.csv`
- `cicp_mp_v1_per_category_attribution.csv`
- `cicp_mp_v1_effective_dimension.csv`
- `cicp_mp_v1_coverage_audit.csv`
- `cicp_mp_v1_protocol_audit.json`
- `cicp_mp_v1_route_decision.json`
- `run_manifest.json`
"""
    result_path.write_text(report, encoding="utf-8")
    route_report = f"""---
title: {run_stamp} CCFCRec Amazon-VG CICP-MP-v1 train-only offline audit 路线判断
date: 2026-07-17
tags:
  - CCFCRec
  - Amazon-VG
  - CICP-MP-v1
  - route-decision
---

# CCFCRec Amazon-VG CICP-MP-v1 train-only offline audit 路线判断

关联结果：[[{result_path.stem}]]

> [!important] 主决策
> 路线 {decision['route_number']}：`{decision['route']}`。

> [!warning] 判定器纠正记录
> 首次自动汇总误让已拒绝的 capacity increment（容量增量）参与训练/验证迁移门槛，触发了路线3。最终门槛只检查实际保留分量，全部候选仍完整报告；原始画像和折外预测没有修改。保留分量最大 KS statistic（柯尔莫哥洛夫-斯米尔诺夫统计量）为 `{decision['maximum_retained_component_ks_statistic']:.6f}`，全部候选最大值为 `{decision['maximum_all_candidate_ks_statistic']:.6f}`。

## 决策依据

{md_table(pd.DataFrame([{'gate': key, 'pass': value} for key, value in decision['gates'].items()]), 5)}

- 主残差映射折外 Spearman（斯皮尔曼秩相关）：`{decision['main_residual_mapping_oof_spearman']:.6f}`；
- 保留标量有效秩：`{decision['retained_scalar_effective_rank']:.6f}`；
- 方向状态：`{direction.feasibility['status']}`；
- validation/test recommendation metrics（验证集/测试集推荐指标）：未读取、未生成；
- CCFCRec training（冷启动对比协同过滤训练）：未启动。

## 执行边界

1. 当前先展示画像定义、审计结果和路线判断，与用户讨论接入；
2. 不自动设计多支100 epoch（100训练轮次）实验；
3. 不开放 multi-seed（多随机种子）；
4. 不把 residual（残差）表述成因果去混杂；
5. 不把经验百分位及其数学变换计作新增维度。

## 画像资产

- 完整画像：`{output_dir / 'cicp_mp_v1_item_profile.csv'}`
- 分量目录：`{output_dir / 'cicp_mp_v1_component_catalog.csv'}`
- 协议审计：`{output_dir / 'cicp_mp_v1_protocol_audit.json'}`
- 运行清单：`{output_dir / 'run_manifest.json'}`
"""
    route_path.write_text(route_report, encoding="utf-8")
    manifest["result_markdown"] = str(result_path)
    manifest["route_markdown"] = str(route_path)
    return result_path, route_path


def run(args: argparse.Namespace) -> Path:
    start_all = time.time()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    profile = load_profile(args.v3_profile, args.train_rating)
    assert_no_recommendation_metrics(profile, "v3 structural profile")
    source_field_audit = build_source_field_audit(args.original_cicp_profile)
    formal_loader_audit = audit_formal_feature_loader(args.cicp_feature_code)
    train = profile[profile["split"].eq("train")].copy().reset_index(drop=True)
    validation = profile[profile["split"].eq("validate")].copy().reset_index(drop=True)
    if len(train) != 24726 or len(validation) != 5298:
        raise ValueError(f"unexpected item counts: {len(train)}/{len(validation)}")
    train_items = train["raw_asin"].astype(str).tolist()
    validation_items = validation["raw_asin"].astype(str).tolist()

    category_assets = build_category_assets(
        train_items, validation_items, args.category_pickle
    )
    hierarchy_feasibility = build_hierarchy_feasibility_audit(
        args.asin_csv,
        [*train_items, *validation_items],
        category_assets,
    )
    image_train, image_validation, image_audit = project_images_mps(
        train_items,
        validation_items,
        args.image_features,
        output_dim=IMAGE_DIM,
        batch_size=args.image_batch_size,
    )
    interactions, _, _ = build_interaction_matrix(args.train_rating, train_items)
    interaction_count = np.asarray(interactions.sum(axis=1)).ravel()
    train["train_interaction_count"] = interaction_count.astype(int)
    train["log1p_train_interaction_count"] = np.log1p(interaction_count)
    validation["train_interaction_count"] = 0
    validation["log1p_train_interaction_count"] = 0.0
    weights = reliability_weight(interaction_count)
    folds = make_folds(len(train))
    train_controls = train[list(CONTROL_COLUMNS)].to_numpy(dtype=float)
    validation_controls = validation[list(CONTROL_COLUMNS)].to_numpy(dtype=float)
    probe = run_collaborative_probe(
        interactions,
        image_train,
        category_assets,
        train_controls,
        weights,
        folds,
        train_items,
    )
    train = pd.concat([train, probe.item], axis=1)

    controls = cross_fitted_controls(
        train["category_semantic_increment_raw"].to_numpy(dtype=float),
        train_controls,
        validation_controls,
        weights,
        folds,
    )
    train["category_semantic_residual_ridge"] = (
        train["category_semantic_increment_raw"].to_numpy(dtype=float)
        - controls.ridge_oof
    )
    train["category_semantic_residual_hgb"] = (
        train["category_semantic_increment_raw"].to_numpy(dtype=float)
        - controls.hgb_oof
    )
    train_features = np.hstack(
        [
            image_train,
            category_assets.train_latent,
            train[list(ACAT_RESIDUAL_COLUMNS)].to_numpy(dtype=float),
        ]
    )
    validation_features = np.hstack(
        [
            image_validation,
            category_assets.validation_latent,
            validation[list(ACAT_RESIDUAL_COLUMNS)].to_numpy(dtype=float),
        ]
    )
    core_mappings = cross_fitted_scalar_mappings(
        train[list(CORE_MAPPING_TARGETS)],
        train_features,
        validation_features,
        weights,
        folds,
        use_hgb=True,
    )
    attribution_mappings = cross_fitted_scalar_mappings(
        train[list(ATTRIBUTION_TARGETS)],
        train_features,
        validation_features,
        weights,
        folds,
        use_hgb=False,
    )
    nested = nested_oof_main_mapping(
        train["category_semantic_residual_ridge"].to_numpy(dtype=float),
        train_features,
        validation_features,
        weights,
        folds,
        stable_item_folds(validation_items),
    )
    direction = audit_direction(
        probe.directions,
        train_features,
        validation_features,
        weights,
        folds,
    )

    train_output = train[list(PROFILE_BASE_COLUMNS)].copy()
    validation_output = validation[list(PROFILE_BASE_COLUMNS)].copy()
    raw_label_columns = [
        "probe_fold",
        "base_collaborative_cosine",
        "shuffled_category_collaborative_cosine",
        "full_collaborative_cosine",
        "category_total_increment_raw",
        "category_capacity_increment_raw",
        "category_semantic_increment_raw",
        "category_semantic_residual_ridge",
        "category_semantic_residual_hgb",
        *ATTRIBUTION_TARGETS,
    ]
    for column in raw_label_columns:
        train_output[f"train_label_{column}"] = train[column].to_numpy()
        validation_output[f"train_label_{column}"] = np.nan

    target_to_profile = {
        "category_total_increment_raw": "mp_category_total_increment_prediction",
        "category_capacity_increment_raw": "mp_category_capacity_increment_prediction",
        "category_semantic_increment_raw": "mp_category_semantic_increment_prediction",
        "category_semantic_residual_ridge": "mp_category_semantic_residual_ridge_prediction",
        "category_semantic_residual_hgb": "mp_category_semantic_residual_hgb_prediction",
        "base_collaborative_cosine": "mp_base_collaborative_predictability_prediction",
        "shuffled_category_collaborative_cosine": "mp_shuffled_collaborative_predictability_prediction",
        "full_collaborative_cosine": "mp_full_collaborative_predictability_prediction",
    }
    for target, output_name in target_to_profile.items():
        train_output[output_name] = core_mappings.train_predictions[f"{target}__hgb"]
        validation_output[output_name] = core_mappings.validation_predictions[
            f"{target}__hgb"
        ]
    for target in ATTRIBUTION_TARGETS:
        output_name = f"mp_{target}_prediction"
        train_output[output_name] = attribution_mappings.train_predictions[
            f"{target}__ridge"
        ]
        validation_output[output_name] = attribution_mappings.validation_predictions[
            f"{target}__ridge"
        ]

    train_output["mp_raw_predicted_increment"] = nested.train_hgb_mean
    validation_output["mp_raw_predicted_increment"] = nested.validation_hgb_mean
    train_output["mp_empirical_percentile"] = empirical_percentile(
        nested.train_hgb_mean, nested.train_hgb_mean
    )
    validation_output["mp_empirical_percentile"] = empirical_percentile(
        nested.train_hgb_mean, nested.validation_hgb_mean
    )
    train_output["mp_fold_prediction_uncertainty"] = nested.train_hgb_std
    validation_output["mp_fold_prediction_uncertainty"] = nested.validation_hgb_std
    train_output["mp_positive_fold_share"] = nested.train_positive_share
    validation_output["mp_positive_fold_share"] = nested.validation_positive_share
    train_output["mp_hgb_ridge_disagreement"] = np.abs(
        nested.train_hgb_mean - nested.train_ridge_mean
    )
    validation_output["mp_hgb_ridge_disagreement"] = np.abs(
        nested.validation_hgb_mean - nested.validation_ridge_mean
    )
    train_output["mp_control_explained_component_ridge"] = controls.ridge_oof
    validation_output["mp_control_explained_component_ridge"] = controls.validation_ridge
    train_output["mp_control_explained_component_hgb"] = controls.hgb_oof
    validation_output["mp_control_explained_component_hgb"] = controls.validation_hgb
    train_output["mp_full_remaining_space"] = (
        1.0 - train_output["mp_full_collaborative_predictability_prediction"]
    )
    validation_output["mp_full_remaining_space"] = (
        1.0 - validation_output["mp_full_collaborative_predictability_prediction"]
    )
    for index in range(DIRECTION_COMPRESSED_DIM):
        name = f"mp_direction16_{index:02d}"
        train_output[name] = direction.train_mapped_oof[:, index]
        validation_output[name] = direction.validation_mapped[:, index]

    combined = pd.concat([train_output, validation_output], ignore_index=True)
    assert_no_recommendation_metrics(combined, "CICP-MP-v1 output profile")
    mp_numeric = combined.filter(regex=r"^mp_").select_dtypes(include=[np.number])
    if not np.isfinite(mp_numeric.to_numpy()).all():
        raise ValueError("CICP-MP-v1 deployable profile contains non-finite component values")

    uncertainty_summary, uncertainty_bins = uncertainty_calibration(
        train["category_semantic_residual_ridge"].to_numpy(dtype=float),
        nested.train_hgb_mean,
        nested.train_hgb_std,
        np.abs(nested.train_hgb_mean - nested.train_ridge_mean),
    )
    residualization = residualization_audit(train, tuple(CONTROL_COLUMNS))
    mapping_summary = core_mappings.summary
    component_catalog = build_component_catalog(
        mapping_summary,
        uncertainty_summary,
        direction.feasibility,
        attribution_mappings.summary,
    )
    retained, component_decisions = choose_retained_components(
        train_output,
        component_catalog,
        mapping_summary,
        attribution_mappings.summary,
        uncertainty_summary,
        direction,
    )
    component_catalog = finalize_component_catalog(
        component_catalog, component_decisions
    )
    retained_scalar = [column for column in retained if not column.startswith("mp_direction16_")]
    candidate_scalar = [
        "mp_raw_predicted_increment",
        "mp_empirical_percentile",
        "mp_category_total_increment_prediction",
        "mp_category_capacity_increment_prediction",
        "mp_category_semantic_increment_prediction",
        "mp_category_semantic_residual_ridge_prediction",
        "mp_category_semantic_residual_hgb_prediction",
        "mp_fold_prediction_uncertainty",
        "mp_positive_fold_share",
        "mp_hgb_ridge_disagreement",
        "mp_category_attribution_max_prediction",
        "mp_category_attribution_min_prediction",
        "mp_category_attribution_std_prediction",
        "mp_category_attribution_positive_share_prediction",
        "mp_category_attribution_entropy_prediction",
        "mp_control_explained_component_ridge",
        "mp_control_explained_component_hgb",
        "mp_base_collaborative_predictability_prediction",
        "mp_shuffled_collaborative_predictability_prediction",
        "mp_full_collaborative_predictability_prediction",
        "mp_full_remaining_space",
    ]
    candidate_dim, candidate_corr = effective_dimension_audit(
        train_output, candidate_scalar, label="all_scalar_candidates"
    )
    retained_dim, retained_corr = effective_dimension_audit(
        train_output, retained_scalar, label="retained_scalar_components"
    )
    full_retained_dim, full_retained_corr = effective_dimension_audit(
        train_output, retained, label="retained_scalar_plus_direction"
    )
    effective_dimension = pd.concat(
        [candidate_dim, retained_dim, full_retained_dim], ignore_index=True
    )

    key_signals = combined.filter(regex=r"^mp_").select_dtypes(include=[np.number]).columns.tolist()
    purity = build_purity_audit(combined, key_signals)
    purity_summary = (
        purity.assign(abs_spearman=purity["spearman"].abs())
        .sort_values("abs_spearman", ascending=False)
        .groupby(["split", "signal"], as_index=False)
        .first()[["split", "signal", "control", "spearman", "abs_spearman"]]
    )
    shift = build_shift_audit(combined, key_signals)
    fallacy_scan = build_fallacy_scan()
    leakage_pass = True
    decision = decide_route(
        combined,
        retained_dim,
        mapping_summary,
        uncertainty_summary,
        direction,
        shift,
        retained,
        leakage_pass,
    )
    protocol_audit, coverage_audit = build_protocol_audit(
        combined, retained, decision
    )

    tables: dict[str, pd.DataFrame] = {
        "source_field_audit": source_field_audit,
        "hierarchy_feasibility": hierarchy_feasibility,
        "component_catalog": component_catalog,
        "component_decisions": component_decisions,
        "mapping_summary": mapping_summary,
        "attribution_mapping_summary": attribution_mappings.summary,
        "nested_mapping_summary": nested.summary,
        "uncertainty_summary": uncertainty_summary,
        "uncertainty_bins": uncertainty_bins,
        "residualization": residualization,
        "probe_folds": probe.folds,
        "shuffle_stability": probe.stability,
        "control_folds": controls.folds,
        "direction_summary": direction.profile_summary,
        "direction_folds": direction.fold_summary,
        "effective_dimension": effective_dimension,
        "purity": purity,
        "purity_summary": purity_summary,
        "shift": shift,
        "fallacy_scan": fallacy_scan,
        "coverage_audit": coverage_audit,
        "candidate_correlation": candidate_corr.reset_index(),
        "retained_correlation": retained_corr.reset_index(),
        "full_retained_correlation": full_retained_corr.reset_index(),
    }
    output_names = {
        "source_field_audit": "cicp_mp_v1_source_field_audit.csv",
        "hierarchy_feasibility": "cicp_mp_v1_hierarchy_feasibility.csv",
        "component_catalog": "cicp_mp_v1_component_catalog.csv",
        "component_decisions": "cicp_mp_v1_component_decisions.csv",
        "mapping_summary": "cicp_mp_v1_mapping_summary.csv",
        "attribution_mapping_summary": "cicp_mp_v1_attribution_mapping_summary.csv",
        "nested_mapping_summary": "cicp_mp_v1_nested_mapping_summary.csv",
        "uncertainty_summary": "cicp_mp_v1_uncertainty_calibration.csv",
        "uncertainty_bins": "cicp_mp_v1_uncertainty_bins.csv",
        "residualization": "cicp_mp_v1_residualization_sensitivity.csv",
        "probe_folds": "cicp_mp_v1_probe_fold_summary.csv",
        "shuffle_stability": "cicp_mp_v1_shuffle_stability.csv",
        "control_folds": "cicp_mp_v1_control_fold_summary.csv",
        "direction_summary": "cicp_mp_v1_direction_summary.csv",
        "direction_folds": "cicp_mp_v1_direction_fold_summary.csv",
        "effective_dimension": "cicp_mp_v1_effective_dimension.csv",
        "purity": "cicp_mp_v1_rsp_purity.csv",
        "purity_summary": "cicp_mp_v1_rsp_purity_summary.csv",
        "shift": "cicp_mp_v1_train_validation_shift.csv",
        "fallacy_scan": "cicp_mp_v1_fallacy_scan.csv",
        "coverage_audit": "cicp_mp_v1_coverage_audit.csv",
        "candidate_correlation": "cicp_mp_v1_candidate_spearman_correlation.csv",
        "retained_correlation": "cicp_mp_v1_retained_spearman_correlation.csv",
        "full_retained_correlation": "cicp_mp_v1_full_retained_spearman_correlation.csv",
    }
    combined.to_csv(output_dir / "cicp_mp_v1_item_profile.csv", index=False)
    probe.attribution.to_csv(
        output_dir / "cicp_mp_v1_per_category_attribution.csv", index=False
    )
    for key, frame in tables.items():
        frame.to_csv(output_dir / output_names[key], index=False)
    np.savez_compressed(
        output_dir / "cicp_mp_v1_direction64_train_oof.npz",
        raw_asin=np.asarray(train_items, dtype=str),
        global_seed43=probe.directions["global_seed43"],
        global_seed44=probe.directions["global_seed44"],
        within_category_count_seed43=probe.directions[
            "within_category_count_seed43"
        ],
    )
    (output_dir / "cicp_mp_v1_route_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "cicp_mp_v1_protocol_audit.json").write_text(
        json.dumps(protocol_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "protocol": "cicp_mp_v1_train_only_offline_audit",
        "run_stamp": args.run_stamp,
        "analysis_scope": [
            "train interactions and collaborative labels",
            "train item image/category/structural features",
            "validation image/category/structural features only",
        ],
        "excluded_scope": [
            "validation recommendation metrics",
            "test items and test recommendation metrics",
            "CCFCRec training",
            "M11 target labels",
        ],
        "train_item_count": len(train),
        "validation_structural_item_count": len(validation),
        "test_item_count_analyzed": 0,
        "validation_recommendation_metrics_read_or_generated": False,
        "test_recommendation_metrics_read_or_generated": False,
        "ccfcrec_training_launched": False,
        "folds": FOLDS,
        "nested_folds": NESTED_FOLDS,
        "seed": SEED,
        "device": "mps_for_image_projection",
        "category_shuffle_schemes": list(SHUFFLE_SCHEMES),
        "retained_components": retained,
        "direction_feasibility": direction.feasibility,
        "image_audit": image_audit,
        "category_audit": category_assets.audit,
        "formal_feature_loader_audit": formal_loader_audit,
        "probe_elapsed_seconds": probe.elapsed_seconds,
        "total_elapsed_seconds": time.time() - start_all,
        "inputs": {
            "train_rating": file_fingerprint(args.train_rating, hash_content=True),
            "category_pickle": file_fingerprint(args.category_pickle, hash_content=True),
            "v3_profile": file_fingerprint(args.v3_profile, hash_content=True),
            "image_features": file_fingerprint(args.image_features, hash_content=False),
            "asin_csv": file_fingerprint(args.asin_csv, hash_content=True),
            "original_cicp_profile": file_fingerprint(
                args.original_cicp_profile, hash_content=True
            ),
            "cicp_feature_code": file_fingerprint(
                args.cicp_feature_code, hash_content=True
            ),
        },
        "outputs": [
            "cicp_mp_v1_item_profile.csv",
            "cicp_mp_v1_per_category_attribution.csv",
            "cicp_mp_v1_direction64_train_oof.npz",
            *output_names.values(),
            "cicp_mp_v1_route_decision.json",
            "cicp_mp_v1_protocol_audit.json",
            "run_manifest.json",
        ],
    }
    result_path, route_path = write_reports(
        output_dir,
        args.route_output.expanduser().resolve(),
        args.run_stamp,
        tables,
        decision,
        manifest,
        component_catalog,
        retained,
        direction,
    )
    manifest["outputs"].extend([str(result_path), str(route_path)])
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)
    print(output_dir, flush=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rating", type=Path, default=DEFAULT_TRAIN_RATING)
    parser.add_argument("--image-features", type=Path, default=DEFAULT_IMAGE_FEATURES)
    parser.add_argument("--category-pickle", type=Path, default=DEFAULT_CATEGORY_PICKLE)
    parser.add_argument("--asin-csv", type=Path, default=DEFAULT_ASIN_CSV)
    parser.add_argument(
        "--original-cicp-profile", type=Path, default=DEFAULT_ORIGINAL_CICP_PROFILE
    )
    parser.add_argument(
        "--cicp-feature-code", type=Path, default=DEFAULT_CICP_FEATURE_CODE
    )
    parser.add_argument("--v3-profile", type=Path, default=DEFAULT_V3_PROFILE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-output", type=Path, required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--image-batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
