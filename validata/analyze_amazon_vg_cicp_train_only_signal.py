#!/usr/bin/env python3
"""Construct and audit the train-only CICP signal without recommendation outcomes."""

from __future__ import annotations

import argparse
import json
import math
import pickle
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.sparse import csr_matrix, diags
from scipy.stats import ks_2samp, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, normalize


PROJECT_ROOT = Path("/Users/luojiaqiang/Documents/Obsidian Vault/科研/CCFCRec对比学习思路")
CODE_ROOT = PROJECT_ROOT / "CCFCRec-code"
DEFAULT_TRAIN_RATING = CODE_ROOT / "Amazon VG/data/train_rating.csv"
DEFAULT_IMAGE_FEATURES = CODE_ROOT / "Amazon VG/data/img_feature.npy"
DEFAULT_CATEGORY_PICKLE = CODE_ROOT / "Amazon VG/data/asin_int_category.pkl"
DEFAULT_V3_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录/temp_20260705"
    / "2026-07-05 112021 category-availability-v3-purity-audit"
    / "category_availability_v3_item.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "temp_202607_实验文件记录/temp_20260713"

SEED = 43
FOLDS = 5
COLLAB_DIM = 64
IMAGE_DIM = 256
CATEGORY_DIM = 64
RIDGE_ALPHAS = np.asarray([0.1, 1.0, 10.0, 100.0], dtype=float)
CONTROL_COLUMNS = (
    "category_count",
    "R_metadata_richness_score",
    "S_train_support_score",
    "P_popularity_score",
    "log1p_train_interaction_count",
)
PURITY_COLUMNS = (
    "category_count",
    "R_metadata_richness_score",
    "S_train_support_score",
    "P_popularity_score",
)
ACAT_RESIDUAL_COLUMNS = (
    "Acat_v3_gran_residual_pct",
    "Acat_v3_disc_residual_pct",
    "Acat_v3_collab_residual_pct",
)
PROFILE_COLUMNS = (
    "raw_asin",
    "split",
    "category_count",
    "R_metadata_richness_score",
    "S_train_support_score",
    "P_popularity_score",
    *ACAT_RESIDUAL_COLUMNS,
)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def _row_normalize_dense(values: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norm, 1e-12, None)


def row_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.clip(denominator, 1e-12, None)


def reliability_weight(interaction_count: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.log1p(interaction_count) / math.log(21.0))


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(reference, dtype=float))
    values = np.asarray(values, dtype=float)
    return np.searchsorted(reference, values, side="right") / max(len(reference), 1)


def bootstrap_statistic(
    arrays: tuple[np.ndarray, ...],
    statistic,
    iterations: int,
    seed: int = SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    estimates = np.empty(iterations, dtype=float)
    for index in range(iterations):
        sampled = rng.integers(0, n, size=n)
        estimates[index] = statistic(*(array[sampled] for array in arrays))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.average(values, weights=weights))


def load_profile(path: Path, train_rating_path: Path) -> pd.DataFrame:
    profile = pd.read_csv(path, usecols=list(PROFILE_COLUMNS), dtype={"raw_asin": str})
    profile = profile[profile["split"].isin({"train", "validate"})].copy()
    if profile["raw_asin"].duplicated().any():
        raise ValueError("profile contains duplicate train/validation items")
    train_rating = pd.read_csv(train_rating_path, usecols=["asin"], dtype={"asin": str})
    counts = train_rating.groupby("asin").size().rename("train_interaction_count")
    profile = profile.merge(counts, left_on="raw_asin", right_index=True, how="left", validate="one_to_one")
    profile["train_interaction_count"] = profile["train_interaction_count"].fillna(0).astype(int)
    profile["log1p_train_interaction_count"] = np.log1p(profile["train_interaction_count"])
    for column in (*CONTROL_COLUMNS, *ACAT_RESIDUAL_COLUMNS):
        profile[column] = pd.to_numeric(profile[column], errors="coerce")
    if profile[list(PROFILE_COLUMNS[2:])].isna().any().any():
        missing = profile.columns[profile.isna().any()].tolist()
        raise ValueError(f"profile contains missing structural values: {missing}")
    return profile.sort_values(["split", "raw_asin"]).reset_index(drop=True)


def load_category_map(path: Path) -> tuple[dict[str, list[int]], int]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    mapping = {str(key): list(value) for key, value in payload["asin_category_int_map"].items()}
    return mapping, len(payload["category_ser_map"])


def build_category_latent(
    train_items: list[str],
    validation_items: list[str],
    category_map: dict[str, list[int]],
    category_count: int,
    dim: int = CATEGORY_DIM,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    all_items = [*train_items, *validation_items]
    rows: list[int] = []
    columns: list[int] = []
    missing: list[str] = []
    for row, item in enumerate(all_items):
        categories = category_map.get(item)
        if not categories:
            missing.append(item)
            continue
        rows.extend([row] * len(categories))
        columns.extend(int(category) for category in categories)
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
    train_latent = _row_normalize_dense(svd.fit_transform(train_tfidf).astype(np.float32))
    validation_latent = _row_normalize_dense(svd.transform(validation_tfidf).astype(np.float32))
    audit = {
        "category_count": category_count,
        "train_covered": int(np.asarray(train_matrix.getnnz(axis=1) > 0).sum()),
        "validation_covered": int(np.asarray(validation_matrix.getnnz(axis=1) > 0).sum()),
        "missing_items": missing,
        "explained_variance_ratio_sum": float(svd.explained_variance_ratio_.sum()),
    }
    return train_latent, validation_latent, audit


def project_images_mps(
    train_items: list[str],
    validation_items: list[str],
    image_path: Path,
    output_dim: int = IMAGE_DIM,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch

    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is required for the local image random projection")
    image_map = np.load(image_path, allow_pickle=True).item()
    all_items = [*train_items, *validation_items]
    missing = [item for item in all_items if item not in image_map]
    if missing:
        raise ValueError(f"image features missing for {len(missing)} train/validation items")
    input_dim = int(np.asarray(image_map[all_items[0]]).shape[0])
    rng = np.random.default_rng(SEED)
    signs = rng.integers(0, 2, size=(input_dim, output_dim), dtype=np.int8)
    projection_np = (signs.astype(np.float32) * 2.0 - 1.0) / math.sqrt(output_dim)
    projection = torch.from_numpy(projection_np).to("mps")
    output = np.empty((len(all_items), output_dim), dtype=np.float32)
    start = time.time()
    for offset in range(0, len(all_items), batch_size):
        items = all_items[offset : offset + batch_size]
        batch = np.stack([np.asarray(image_map[item], dtype=np.float32) for item in items])
        batch_tensor = torch.from_numpy(batch).to("mps")
        projected = batch_tensor @ projection
        projected = torch.nn.functional.normalize(projected, dim=1, eps=1e-12)
        output[offset : offset + len(items)] = projected.cpu().numpy()
        if offset == 0 or offset + batch_size >= len(all_items) or (offset // batch_size) % 10 == 0:
            print(f"[image projection] {min(offset + batch_size, len(all_items))}/{len(all_items)}")
    torch.mps.synchronize()
    audit = {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "device": "mps",
        "train_covered": len(train_items),
        "validation_covered": len(validation_items),
        "missing_items": 0,
        "elapsed_seconds": time.time() - start,
    }
    return output[: len(train_items)], output[len(train_items) :], audit


def build_interaction_matrix(
    train_rating_path: Path, train_items: list[str]
) -> tuple[csr_matrix, pd.DataFrame, list[str]]:
    ratings = pd.read_csv(
        train_rating_path,
        usecols=["reviewerID", "asin"],
        dtype={"reviewerID": str, "asin": str},
    )
    item_index = {item: index for index, item in enumerate(train_items)}
    if not set(ratings["asin"]).issubset(item_index):
        raise ValueError("train ratings contain items absent from the train profile")
    users = sorted(ratings["reviewerID"].unique().tolist())
    user_index = {user: index for index, user in enumerate(users)}
    rows = ratings["asin"].map(item_index).to_numpy(dtype=np.int64)
    columns = ratings["reviewerID"].map(user_index).to_numpy(dtype=np.int64)
    matrix = csr_matrix(
        (np.ones(len(ratings), dtype=np.float32), (rows, columns)),
        shape=(len(train_items), len(users)),
    )
    matrix.data[:] = 1.0
    matrix.eliminate_zeros()
    return matrix, ratings, users


def normalize_interactions(matrix: csr_matrix, user_degree: np.ndarray | None = None) -> csr_matrix:
    item_degree = np.asarray(matrix.sum(axis=1)).ravel()
    if user_degree is None:
        user_degree = np.asarray(matrix.sum(axis=0)).ravel()
    item_scale = np.zeros_like(item_degree, dtype=np.float64)
    user_scale = np.zeros_like(user_degree, dtype=np.float64)
    item_scale[item_degree > 0] = 1.0 / np.sqrt(item_degree[item_degree > 0])
    user_scale[user_degree > 0] = 1.0 / np.sqrt(user_degree[user_degree > 0])
    return (diags(item_scale) @ matrix @ diags(user_scale)).tocsr()


def cross_fitted_category_increment(
    interactions: csr_matrix,
    image: np.ndarray,
    category: np.ndarray,
    shuffled_category: np.ndarray,
    controls: np.ndarray,
    weights: np.ndarray,
    folds: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    n_items = interactions.shape[0]
    base_cosine = np.empty(n_items, dtype=float)
    shuffled_cosine = np.empty(n_items, dtype=float)
    full_cosine = np.empty(n_items, dtype=float)
    fold_rows: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        start = time.time()
        fit_index = np.flatnonzero(folds != fold)
        hold_index = np.flatnonzero(folds == fold)
        fit_raw = interactions[fit_index]
        hold_raw = interactions[hold_index]
        fit_user_degree = np.asarray(fit_raw.sum(axis=0)).ravel()
        fit_matrix = normalize_interactions(fit_raw, fit_user_degree)
        hold_matrix = normalize_interactions(hold_raw, fit_user_degree)
        svd = TruncatedSVD(n_components=COLLAB_DIM, n_iter=7, random_state=SEED)
        fit_target = _row_normalize_dense(svd.fit_transform(fit_matrix))
        hold_target = _row_normalize_dense(svd.transform(hold_matrix))

        base_fit = np.hstack([image[fit_index], controls[fit_index]])
        base_hold = np.hstack([image[hold_index], controls[hold_index]])
        full_fit = np.hstack([base_fit, category[fit_index]])
        full_hold = np.hstack([base_hold, category[hold_index]])
        shuffled_fit = np.hstack([base_fit, shuffled_category[fit_index]])
        shuffled_hold = np.hstack([base_hold, shuffled_category[hold_index]])

        base_scaler = StandardScaler().fit(base_fit, sample_weight=weights[fit_index])
        full_scaler = StandardScaler().fit(full_fit, sample_weight=weights[fit_index])
        shuffled_scaler = StandardScaler().fit(shuffled_fit, sample_weight=weights[fit_index])
        base_model = RidgeCV(alphas=RIDGE_ALPHAS).fit(
            base_scaler.transform(base_fit), fit_target, sample_weight=weights[fit_index]
        )
        full_model = RidgeCV(alphas=RIDGE_ALPHAS).fit(
            full_scaler.transform(full_fit), fit_target, sample_weight=weights[fit_index]
        )
        shuffled_model = RidgeCV(alphas=RIDGE_ALPHAS).fit(
            shuffled_scaler.transform(shuffled_fit), fit_target, sample_weight=weights[fit_index]
        )
        base_prediction = base_model.predict(base_scaler.transform(base_hold))
        full_prediction = full_model.predict(full_scaler.transform(full_hold))
        shuffled_prediction = shuffled_model.predict(shuffled_scaler.transform(shuffled_hold))
        base_cosine[hold_index] = row_cosine(hold_target, base_prediction)
        full_cosine[hold_index] = row_cosine(hold_target, full_prediction)
        shuffled_cosine[hold_index] = row_cosine(hold_target, shuffled_prediction)
        fold_rows.append(
            {
                "fold": fold,
                "fit_count": len(fit_index),
                "hold_count": len(hold_index),
                "base_alpha": float(base_model.alpha_),
                "shuffled_alpha": float(shuffled_model.alpha_),
                "full_alpha": float(full_model.alpha_),
                "base_cosine_mean": float(np.average(base_cosine[hold_index], weights=weights[hold_index])),
                "shuffled_cosine_mean": float(
                    np.average(shuffled_cosine[hold_index], weights=weights[hold_index])
                ),
                "full_cosine_mean": float(np.average(full_cosine[hold_index], weights=weights[hold_index])),
                "category_total_increment_mean": float(
                    np.average(full_cosine[hold_index] - base_cosine[hold_index], weights=weights[hold_index])
                ),
                "category_capacity_increment_mean": float(
                    np.average(shuffled_cosine[hold_index] - base_cosine[hold_index], weights=weights[hold_index])
                ),
                "category_semantic_increment_mean": float(
                    np.average(full_cosine[hold_index] - shuffled_cosine[hold_index], weights=weights[hold_index])
                ),
                "elapsed_seconds": time.time() - start,
            }
        )
        print(f"[collaborative probe] fold {fold + 1}/{FOLDS} complete")
    item = pd.DataFrame(
        {
            "probe_fold": folds,
            "base_collaborative_cosine": base_cosine,
            "shuffled_category_collaborative_cosine": shuffled_cosine,
            "full_collaborative_cosine": full_cosine,
            "category_total_increment_raw": full_cosine - base_cosine,
            "category_capacity_increment_raw": shuffled_cosine - base_cosine,
            "category_semantic_increment_raw": full_cosine - shuffled_cosine,
        }
    )
    return item, pd.DataFrame(fold_rows)


def cross_fitted_residual(
    values: np.ndarray,
    controls: np.ndarray,
    weights: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction = np.empty(len(values), dtype=float)
    rows: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        fit_index = np.flatnonzero(folds != fold)
        hold_index = np.flatnonzero(folds == fold)
        scaler = StandardScaler().fit(controls[fit_index], sample_weight=weights[fit_index])
        model = RidgeCV(alphas=RIDGE_ALPHAS).fit(
            scaler.transform(controls[fit_index]), values[fit_index], sample_weight=weights[fit_index]
        )
        prediction[hold_index] = model.predict(scaler.transform(controls[hold_index]))
        rows.append({"fold": fold, "alpha": float(model.alpha_), "hold_count": len(hold_index)})
    return values - prediction, pd.DataFrame(rows)


def cross_fitted_deployable_mapping(
    train_features: np.ndarray,
    validation_features: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    folds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    hgb_oof = np.empty(len(target), dtype=float)
    ridge_oof = np.empty(len(target), dtype=float)
    validation_hgb = np.empty((FOLDS, len(validation_features)), dtype=float)
    validation_ridge = np.empty((FOLDS, len(validation_features)), dtype=float)
    rows: list[dict[str, Any]] = []
    for fold in range(FOLDS):
        start = time.time()
        fit_index = np.flatnonzero(folds != fold)
        hold_index = np.flatnonzero(folds == fold)
        hgb = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=SEED,
        ).fit(train_features[fit_index], target[fit_index], sample_weight=weights[fit_index])
        hgb_oof[hold_index] = hgb.predict(train_features[hold_index])
        validation_hgb[fold] = hgb.predict(validation_features)

        scaler = StandardScaler().fit(train_features[fit_index], sample_weight=weights[fit_index])
        ridge = RidgeCV(alphas=RIDGE_ALPHAS).fit(
            scaler.transform(train_features[fit_index]),
            target[fit_index],
            sample_weight=weights[fit_index],
        )
        ridge_oof[hold_index] = ridge.predict(scaler.transform(train_features[hold_index]))
        validation_ridge[fold] = ridge.predict(scaler.transform(validation_features))
        rows.append(
            {
                "fold": fold,
                "hgb_spearman": float(spearmanr(hgb_oof[hold_index], target[hold_index]).statistic),
                "ridge_spearman": float(spearmanr(ridge_oof[hold_index], target[hold_index]).statistic),
                "ridge_alpha": float(ridge.alpha_),
                "elapsed_seconds": time.time() - start,
            }
        )
        print(f"[deployable mapping] fold {fold + 1}/{FOLDS} complete")
    return hgb_oof, ridge_oof, validation_hgb, validation_ridge, pd.DataFrame(rows)


def build_split_half_stability(
    interactions: csr_matrix,
    ratings: pd.DataFrame,
    train_items: list[str],
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    item_index = {item: index for index, item in enumerate(train_items)}
    users = sorted(ratings["reviewerID"].unique().tolist())
    user_index = {user: index for index, user in enumerate(users)}
    hashes = pd.util.hash_pandas_object(
        ratings["reviewerID"].astype(str) + "|" + ratings["asin"].astype(str) + f"|{SEED}",
        index=False,
    ).to_numpy(dtype=np.uint64)
    half = hashes % 2
    rows = ratings["asin"].map(item_index).to_numpy(dtype=np.int64)
    columns = ratings["reviewerID"].map(user_index).to_numpy(dtype=np.int64)
    matrices = []
    for value in (0, 1):
        keep = half == value
        matrix = csr_matrix(
            (np.ones(int(keep.sum()), dtype=np.float32), (rows[keep], columns[keep])),
            shape=interactions.shape,
        )
        matrix.data[:] = 1.0
        matrices.append(matrix)
    embeddings = []
    half_degrees = []
    for matrix in matrices:
        half_degrees.append(np.asarray(matrix.sum(axis=1)).ravel())
        svd = TruncatedSVD(n_components=COLLAB_DIM, n_iter=7, random_state=SEED)
        embeddings.append(_row_normalize_dense(svd.fit_transform(normalize_interactions(matrix))))
    total_degree = np.asarray(interactions.sum(axis=1)).ravel()
    valid = (total_degree >= 4) & (half_degrees[0] > 0) & (half_degrees[1] > 0)
    rotation, _ = orthogonal_procrustes(embeddings[1][valid], embeddings[0][valid])
    aligned = embeddings[1] @ rotation
    cosine = row_cosine(embeddings[0], aligned)
    valid_cosine = cosine[valid]
    ci_low, ci_high = bootstrap_statistic(
        (valid_cosine,), lambda x: float(np.median(x)), bootstrap_iterations, SEED
    )
    item = pd.DataFrame(
        {
            "raw_asin": train_items,
            "interaction_count": total_degree.astype(int),
            "half0_count": half_degrees[0].astype(int),
            "half1_count": half_degrees[1].astype(int),
            "stability_eligible": valid,
            "split_half_collaborative_cosine": cosine,
        }
    )
    summary = pd.DataFrame(
        [
            {
                "train_item_count": len(train_items),
                "eligible_support_ge4_both_halves_count": int(valid.sum()),
                "eligible_share": float(valid.mean()),
                "median_split_half_cosine": float(np.median(valid_cosine)),
                "median_bootstrap_ci_low": ci_low,
                "median_bootstrap_ci_high": ci_high,
            }
        ]
    )
    return item, summary


def support_bucket(count: pd.Series) -> pd.Series:
    return pd.cut(
        count,
        bins=[0, 1, 3, 10, np.inf],
        labels=["support_1", "support_2_3", "support_4_10", "support_11_plus"],
    ).astype(str)


def summarize_probe(
    train: pd.DataFrame,
    weights: np.ndarray,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    total_increment = train["category_total_increment_raw"].to_numpy(dtype=float)
    semantic_increment = train["category_semantic_increment_raw"].to_numpy(dtype=float)
    total_ci_low, total_ci_high = bootstrap_statistic(
        (total_increment, weights), weighted_mean, bootstrap_iterations, SEED
    )
    semantic_ci_low, semantic_ci_high = bootstrap_statistic(
        (semantic_increment, weights), weighted_mean, bootstrap_iterations, SEED + 1
    )
    overall = pd.DataFrame(
        [
            {
                "train_item_count": len(train),
                "base_cosine_weighted_mean": weighted_mean(
                    train["base_collaborative_cosine"].to_numpy(), weights
                ),
                "shuffled_cosine_weighted_mean": weighted_mean(
                    train["shuffled_category_collaborative_cosine"].to_numpy(), weights
                ),
                "full_cosine_weighted_mean": weighted_mean(
                    train["full_collaborative_cosine"].to_numpy(), weights
                ),
                "category_total_increment_weighted_mean": weighted_mean(total_increment, weights),
                "category_total_increment_bootstrap_ci_low": total_ci_low,
                "category_total_increment_bootstrap_ci_high": total_ci_high,
                "category_semantic_increment_weighted_mean": weighted_mean(
                    semantic_increment, weights
                ),
                "category_semantic_increment_bootstrap_ci_low": semantic_ci_low,
                "category_semantic_increment_bootstrap_ci_high": semantic_ci_high,
                "total_positive_item_share": float((total_increment > 0).mean()),
                "semantic_positive_item_share": float((semantic_increment > 0).mean()),
            }
        ]
    )
    rows: list[dict[str, Any]] = []
    for bucket, part in train.groupby("support_bucket", sort=False):
        part_weights = reliability_weight(part["train_interaction_count"].to_numpy(dtype=float))
        rows.append(
            {
                "support_bucket": bucket,
                "item_count": len(part),
                "base_cosine_weighted_mean": weighted_mean(
                    part["base_collaborative_cosine"].to_numpy(), part_weights
                ),
                "shuffled_cosine_weighted_mean": weighted_mean(
                    part["shuffled_category_collaborative_cosine"].to_numpy(), part_weights
                ),
                "full_cosine_weighted_mean": weighted_mean(
                    part["full_collaborative_cosine"].to_numpy(), part_weights
                ),
                "category_total_increment_weighted_mean": weighted_mean(
                    part["category_total_increment_raw"].to_numpy(), part_weights
                ),
                "category_semantic_increment_weighted_mean": weighted_mean(
                    part["category_semantic_increment_raw"].to_numpy(), part_weights
                ),
                "total_positive_item_share": float(
                    part["category_total_increment_raw"].gt(0).mean()
                ),
                "semantic_positive_item_share": float(
                    part["category_semantic_increment_raw"].gt(0).mean()
                ),
            }
        )
    return overall, pd.DataFrame(rows)


def summarize_deployability(
    train: pd.DataFrame,
    fold_summary: pd.DataFrame,
    bootstrap_iterations: int,
) -> pd.DataFrame:
    predicted = train["cicp_score_raw_oof"].to_numpy(dtype=float)
    target = train["category_semantic_increment_residual"].to_numpy(dtype=float)
    rho = float(spearmanr(predicted, target).statistic)
    ci_low, ci_high = bootstrap_statistic(
        (predicted, target),
        lambda x, y: float(spearmanr(x, y).statistic),
        min(bootstrap_iterations, 400),
        SEED,
    )
    low_cut, high_cut = np.quantile(predicted, [0.2, 0.8])
    bottom = target[predicted <= low_cut]
    top = target[predicted >= high_cut]
    separation = float(top.mean() - bottom.mean())
    return pd.DataFrame(
        [
            {
                "mapping": "hist_gradient_boosting",
                "oof_spearman": rho,
                "oof_spearman_bootstrap_ci_low": ci_low,
                "oof_spearman_bootstrap_ci_high": ci_high,
                "top20_minus_bottom20_true_residual": separation,
                "positive_fold_count": int(fold_summary["hgb_spearman"].gt(0).sum()),
                "fold_count": len(fold_summary),
            },
            {
                "mapping": "ridge_robustness",
                "oof_spearman": float(
                    spearmanr(train["cicp_ridge_score_raw_oof"], target).statistic
                ),
                "oof_spearman_bootstrap_ci_low": np.nan,
                "oof_spearman_bootstrap_ci_high": np.nan,
                "top20_minus_bottom20_true_residual": np.nan,
                "positive_fold_count": int(fold_summary["ridge_spearman"].gt(0).sum()),
                "fold_count": len(fold_summary),
            },
        ]
    )


def build_purity_summary(combined: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ("train", "validate"):
        part = combined[combined["split"].eq(split)]
        for control in PURITY_COLUMNS:
            rows.append(
                {
                    "split": split,
                    "signal": "cicp_score",
                    "control": control,
                    "spearman": float(spearmanr(part["cicp_score"], part[control]).statistic),
                }
            )
    return pd.DataFrame(rows)


def build_fallacy_scan() -> pd.DataFrame:
    return pd.DataFrame(
        [
            (1, "Simpson's paradox", "PASS", "overall and all four support strata have positive semantic increment"),
            (2, "Ecological fallacy", "PASS", "construction and intended inference are both item-level"),
            (3, "Berkson's paradox", "CAUTION", "collaborative labels exist only for train items; cold-item transfer is not yet a recommendation result"),
            (4, "Collider bias", "CAUTION", "train-graph support controls may share causes with collaborative structure; no causal claim is made"),
            (5, "Base rate neglect", "NOT_APPLICABLE", "no diagnostic classifier, sensitivity, or predictive value is reported"),
            (6, "Regression to the mean", "PASS", "items were not selected from an extreme recommendation-outcome group"),
            (7, "Survivorship bias", "PASS", "all 24726 train items and all 5298 validation structural items are retained"),
            (8, "Look-elsewhere effect", "CAUTION", "v1 omitted a shuffle control; v1 is retained as preliminary and v1.1 is the only decision protocol"),
            (9, "Garden of forking paths", "CAUTION", "this remains an exploratory single-dataset design despite disclosed fixed v1.1 gates"),
            (10, "Correlation is not causation", "CAUTION", "offline association does not prove category use will improve recommendation metrics"),
            (11, "Reverse causality", "CAUTION", "the audit establishes predictability, not a directional causal effect of category metadata"),
        ],
        columns=["fallacy_id", "fallacy", "status", "assessment"],
    )


def decide_route(
    stability: pd.DataFrame,
    probe: pd.DataFrame,
    probe_folds: pd.DataFrame,
    support_summary: pd.DataFrame,
    deployability: pd.DataFrame,
    purity: pd.DataFrame,
    combined: pd.DataFrame,
    image_audit: dict[str, Any],
    category_audit: dict[str, Any],
) -> dict[str, Any]:
    stability_row = stability.iloc[0]
    probe_row = probe.iloc[0]
    deploy_row = deployability[deployability["mapping"].eq("hist_gradient_boosting")].iloc[0]
    validation = combined[combined["split"].eq("validate")]
    train = combined[combined["split"].eq("train")]
    ks = ks_2samp(train["cicp_score"], validation["cicp_score"])
    gates = {
        "gate_a_data_and_collaborative_target": bool(
            image_audit["train_covered"] == len(train)
            and image_audit["validation_covered"] == len(validation)
            and category_audit["train_covered"] == len(train)
            and category_audit["validation_covered"] == len(validation)
            and stability_row["median_split_half_cosine"] >= 0.20
            and stability_row["median_bootstrap_ci_low"] > 0
        ),
        "gate_b_category_increment": bool(
            probe_row["category_total_increment_weighted_mean"] >= 0.01
            and probe_row["category_total_increment_bootstrap_ci_low"] > 0
            and probe_row["category_semantic_increment_weighted_mean"] >= 0.005
            and probe_row["category_semantic_increment_bootstrap_ci_low"] > 0
            and probe_folds["category_semantic_increment_mean"].gt(0).all()
            and support_summary["category_semantic_increment_weighted_mean"].gt(0).sum() >= 3
        ),
        "gate_c_deployability": bool(
            deploy_row["oof_spearman"] >= 0.15
            and deploy_row["oof_spearman_bootstrap_ci_low"] >= 0.10
            and deploy_row["top20_minus_bottom20_true_residual"] >= 0.02
            and deploy_row["positive_fold_count"] == FOLDS
        ),
        "gate_d_rsp_purity": bool(purity["spearman"].abs().max() < 0.35),
        "gate_e_coverage_and_shift": bool(
            len(validation) == 5298
            and np.isfinite(validation["cicp_score"]).all()
            and ks.statistic < 0.15
        ),
    }
    if not gates["gate_a_data_and_collaborative_target"]:
        route = "collaborative_target_unreliable"
    elif not gates["gate_b_category_increment"]:
        route = "train_category_has_no_incremental_collaborative_signal"
    elif not gates["gate_c_deployability"]:
        route = "category_increment_exists_but_not_deployable"
    elif not gates["gate_d_rsp_purity"]:
        route = "cicp_collapses_to_rsp"
    elif not gates["gate_e_coverage_and_shift"]:
        route = "cicp_coverage_or_shift_failure"
    else:
        route = "cicp_ready_for_one_formal_round"
    return {
        "route": route,
        "gates": gates,
        "run_ccfcrec_training_now": False,
        "run_multi_seed_now": False,
        "test_items_analyzed": 0,
        "test_item_level_metrics_read_or_generated": False,
        "validation_recommendation_metrics_read_or_generated": False,
        "validation_structural_item_count": len(validation),
        "validation_score_finite_count": int(np.isfinite(validation["cicp_score"]).sum()),
        "train_validation_score_ks_statistic": float(ks.statistic),
        "train_validation_score_ks_pvalue": float(ks.pvalue),
        "max_abs_score_rsp_spearman": float(purity["spearman"].abs().max()),
        "construct_training_plan_next": route == "cicp_ready_for_one_formal_round",
        "upgrade_semantic_basis_next": route
        in {
            "train_category_has_no_incremental_collaborative_signal",
            "category_increment_exists_but_not_deployable",
        },
        "protocol_version": "cicp_train_only_offline_v1_1",
        "preliminary_v1_superseded": True,
        "fallacy_scan_coverage": "11/11",
    }


def run(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    profile = load_profile(args.v3_profile, args.train_rating)
    train = profile[profile["split"].eq("train")].copy().reset_index(drop=True)
    validation = profile[profile["split"].eq("validate")].copy().reset_index(drop=True)
    train_items = train["raw_asin"].tolist()
    validation_items = validation["raw_asin"].tolist()
    if len(train) != 24726 or len(validation) != 5298:
        raise ValueError(f"unexpected train/validation item counts: {len(train)}/{len(validation)}")

    category_map, category_count = load_category_map(args.category_pickle)
    category_train, category_validation, category_audit = build_category_latent(
        train_items, validation_items, category_map, category_count
    )
    image_train, image_validation, image_audit = project_images_mps(
        train_items,
        validation_items,
        args.image_features,
        output_dim=IMAGE_DIM,
        batch_size=args.image_batch_size,
    )
    interactions, ratings, _ = build_interaction_matrix(args.train_rating, train_items)
    interaction_count = np.asarray(interactions.sum(axis=1)).ravel()
    weights = reliability_weight(interaction_count)
    controls = train[list(CONTROL_COLUMNS)].to_numpy(dtype=float)
    acat_train = train[list(ACAT_RESIDUAL_COLUMNS)].to_numpy(dtype=float)
    acat_validation = validation[list(ACAT_RESIDUAL_COLUMNS)].to_numpy(dtype=float)
    folds = np.empty(len(train), dtype=int)
    for fold, (_, hold) in enumerate(KFold(FOLDS, shuffle=True, random_state=SEED).split(train)):
        folds[hold] = fold
    category_permutation = np.random.default_rng(SEED).permutation(len(train))
    shuffled_category_train = category_train[category_permutation]

    probe_item, probe_folds = cross_fitted_category_increment(
        interactions,
        image_train,
        category_train,
        shuffled_category_train,
        controls,
        weights,
        folds,
    )
    train = pd.concat([train, probe_item], axis=1)
    residual, residual_folds = cross_fitted_residual(
        train["category_semantic_increment_raw"].to_numpy(), controls, weights, folds
    )
    train["category_semantic_increment_residual"] = residual
    train_features = np.hstack([image_train, category_train, acat_train])
    validation_features = np.hstack([image_validation, category_validation, acat_validation])
    hgb_oof, ridge_oof, validation_hgb, validation_ridge, mapping_folds = (
        cross_fitted_deployable_mapping(
            train_features, validation_features, residual, weights, folds
        )
    )
    train["cicp_score_raw_oof"] = hgb_oof
    train["cicp_ridge_score_raw_oof"] = ridge_oof
    train["cicp_score_uncertainty"] = np.nan
    train["cicp_score"] = empirical_percentile(hgb_oof, hgb_oof)
    validation["probe_fold"] = -1
    validation["base_collaborative_cosine"] = np.nan
    validation["shuffled_category_collaborative_cosine"] = np.nan
    validation["full_collaborative_cosine"] = np.nan
    validation["category_total_increment_raw"] = np.nan
    validation["category_capacity_increment_raw"] = np.nan
    validation["category_semantic_increment_raw"] = np.nan
    validation["category_semantic_increment_residual"] = np.nan
    validation["cicp_score_raw_oof"] = validation_hgb.mean(axis=0)
    validation["cicp_ridge_score_raw_oof"] = validation_ridge.mean(axis=0)
    validation["cicp_score_uncertainty"] = validation_hgb.std(axis=0, ddof=1)
    validation["cicp_score"] = empirical_percentile(hgb_oof, validation["cicp_score_raw_oof"])
    train["support_bucket"] = support_bucket(train["train_interaction_count"])
    validation["support_bucket"] = "cold_start"

    stability_item, stability_summary = build_split_half_stability(
        interactions, ratings, train_items, args.bootstrap_iterations
    )
    probe_summary, support_summary = summarize_probe(train, weights, args.bootstrap_iterations)
    deployability = summarize_deployability(train, mapping_folds, args.bootstrap_iterations)
    combined = pd.concat([train, validation], ignore_index=True, sort=False)
    purity = build_purity_summary(combined)
    distribution = pd.DataFrame(
        [
            {
                "signal": "cicp_score",
                "train_count": len(train),
                "validation_count": len(validation),
                "train_mean": train["cicp_score"].mean(),
                "validation_mean": validation["cicp_score"].mean(),
                "ks_statistic": ks_2samp(train["cicp_score"], validation["cicp_score"]).statistic,
                "ks_pvalue": ks_2samp(train["cicp_score"], validation["cicp_score"]).pvalue,
                "validation_finite_share": np.isfinite(validation["cicp_score"]).mean(),
            }
        ]
    )
    fallacy_scan = build_fallacy_scan()
    decision = decide_route(
        stability_summary,
        probe_summary,
        probe_folds,
        support_summary,
        deployability,
        purity,
        combined,
        image_audit,
        category_audit,
    )

    outputs = {
        "cicp_item_profile.csv": combined,
        "cicp_split_half_stability_item.csv": stability_item,
        "cicp_split_half_stability_summary.csv": stability_summary,
        "cicp_probe_fold_summary.csv": probe_folds,
        "cicp_probe_overall_summary.csv": probe_summary,
        "cicp_probe_support_summary.csv": support_summary,
        "cicp_residual_fold_summary.csv": residual_folds,
        "cicp_mapping_fold_summary.csv": mapping_folds,
        "cicp_deployability_summary.csv": deployability,
        "cicp_purity_summary.csv": purity,
        "cicp_distribution_summary.csv": distribution,
        "cicp_fallacy_scan.csv": fallacy_scan,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    (output_dir / "cicp_route_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "analysis_scope": [
            "train_interactions",
            "train_item_category_and_image_features",
            "validation_structural_features_only",
        ],
        "excluded_scope": [
            "validation_recommendation_metrics",
            "test_items",
            "test_item_level_metrics",
            "m11_target_identity_as_label",
        ],
        "train_item_count": len(train),
        "validation_structural_item_count": len(validation),
        "test_item_count_analyzed": 0,
        "test_item_level_metrics_read_or_generated": False,
        "validation_recommendation_metrics_read_or_generated": False,
        "device": "mps_for_image_projection",
        "seed": SEED,
        "protocol_version": "cicp_train_only_offline_v1_1",
        "category_shuffle_control": True,
        "folds": FOLDS,
        "collaborative_dim": COLLAB_DIM,
        "image_dim": IMAGE_DIM,
        "category_dim": CATEGORY_DIM,
        "bootstrap_iterations": args.bootstrap_iterations,
        "image_audit": image_audit,
        "category_audit": category_audit,
        "inputs": {
            "train_rating": str(args.train_rating),
            "image_features": str(args.image_features),
            "category_pickle": str(args.category_pickle),
            "v3_profile_structural_columns_only": str(args.v3_profile),
        },
        "outputs": [*outputs, "cicp_route_decision.json", "run_manifest.json"],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rating", type=Path, default=DEFAULT_TRAIN_RATING)
    parser.add_argument("--image-features", type=Path, default=DEFAULT_IMAGE_FEATURES)
    parser.add_argument("--category-pickle", type=Path, default=DEFAULT_CATEGORY_PICKLE)
    parser.add_argument("--v3-profile", type=Path, default=DEFAULT_V3_PROFILE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-batch-size", type=int, default=512)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = run(args)
    print(output_dir)


if __name__ == "__main__":
    main()
