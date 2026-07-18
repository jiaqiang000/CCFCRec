#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "Amazon VG"))

from cicp_features import CICP_FORBIDDEN_EVALUATION_COLUMNS  # noqa: E402
from cicp_mp_features import (  # noqa: E402
    CICP_MP_FEATURE_NAMES,
    CICP_MP_FORBIDDEN_EVALUATION_COLUMNS,
)


STANDARDIZATION_MARKER = "train_feature_wise_zscore_v1"
FORMAL_SPLIT_COUNTS = {"train": 24726, "validate": 5298}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_columns(frame: pd.DataFrame) -> set[str]:
    return {str(column).strip().lower() for column in frame.columns}


def validate_identity(frame: pd.DataFrame, label: str) -> None:
    if "raw_asin" not in frame.columns or "split" not in frame.columns:
        raise ValueError(f"{label} requires raw_asin and split")
    if frame["raw_asin"].isna().any() or frame["raw_asin"].duplicated().any():
        raise ValueError(f"{label} raw_asin values must be non-null and unique")


def retain_train_validate(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    work = frame.loc[
        frame["split"].astype(str).isin(FORMAL_SPLIT_COUNTS),
        columns,
    ].copy().reset_index(drop=True)
    work["raw_asin"] = work["raw_asin"].astype(str)
    return work


def split_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in frame["split"].astype(str).value_counts().items()
    }


def validate_counts(frame: pd.DataFrame, dry_run: bool, label: str) -> dict[str, int]:
    counts = split_counts(frame)
    if set(counts) != set(FORMAL_SPLIT_COUNTS):
        raise ValueError(f"{label} must retain train and validate only: {counts}")
    if not dry_run and counts != FORMAL_SPLIT_COUNTS:
        raise ValueError(f"{label} formal split counts differ: {counts}")
    return counts


def derangement(size: int, rng: np.random.Generator) -> np.ndarray:
    if size < 2:
        raise ValueError("semantic shuffle requires at least two rows per split")
    for _ in range(1000):
        order = rng.permutation(size)
        if not np.equal(order, np.arange(size)).any():
            return order
    raise RuntimeError(f"failed to construct a derangement for {size} rows")


def prepare(args: argparse.Namespace) -> dict:
    scalar_source = pd.read_csv(
        args.scalar_source,
        dtype={"raw_asin": str},
        low_memory=False,
    )
    mp_source = pd.read_csv(
        args.mp_source,
        dtype={"raw_asin": str},
        low_memory=False,
    )
    validate_identity(scalar_source, "scalar source")
    validate_identity(mp_source, "CICP-MP source")

    scalar_forbidden = sorted(
        normalized_columns(scalar_source) & CICP_FORBIDDEN_EVALUATION_COLUMNS
    )
    mp_forbidden = sorted(
        normalized_columns(mp_source) & CICP_MP_FORBIDDEN_EVALUATION_COLUMNS
    )
    if scalar_forbidden or mp_forbidden:
        raise ValueError(
            "source profile contains forbidden evaluation columns: "
            f"scalar={scalar_forbidden}, mp={mp_forbidden}"
        )
    if "cicp_score" not in scalar_source.columns:
        raise ValueError("scalar source is missing cicp_score")
    missing_mp = sorted(set(CICP_MP_FEATURE_NAMES) - set(mp_source.columns))
    if missing_mp:
        raise ValueError(f"CICP-MP source is missing features: {missing_mp}")

    scalar = retain_train_validate(
        scalar_source,
        ["raw_asin", "split", "cicp_score"],
    )
    scalar_counts = validate_counts(scalar, args.dry_run, "scalar profile")
    scalar_score = pd.to_numeric(scalar["cicp_score"], errors="coerce")
    if not np.isfinite(scalar_score.to_numpy(dtype=float)).all():
        raise ValueError("cicp_score must be finite")
    if not scalar_score.between(0.0, 1.0).all():
        raise ValueError("cicp_score must be in [0,1]")
    scalar["cicp_score"] = scalar_score

    mp = retain_train_validate(
        mp_source,
        ["raw_asin", "split", *CICP_MP_FEATURE_NAMES],
    )
    mp_counts = validate_counts(mp, args.dry_run, "CICP-MP profile")
    numeric = mp.loc[:, CICP_MP_FEATURE_NAMES].apply(
        pd.to_numeric,
        errors="coerce",
    )
    values = numeric.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("all CICP-MP features must be finite")

    train_mask = mp["split"].astype(str).eq("train").to_numpy()
    train_values = values[train_mask]
    means = train_values.mean(axis=0)
    standard_deviations = train_values.std(axis=0, ddof=0)
    near_constant = [
        name
        for name, value in zip(CICP_MP_FEATURE_NAMES, standard_deviations)
        if value <= 1e-12
    ]
    if near_constant:
        raise ValueError(f"CICP-MP train features have zero variance: {near_constant}")
    standardized_values = (values - means) / standard_deviations
    if not np.isfinite(standardized_values).all():
        raise ValueError("standardized CICP-MP features must be finite")

    standardized = mp.loc[:, ["raw_asin", "split"]].copy()
    standardized.loc[:, CICP_MP_FEATURE_NAMES] = standardized_values
    standardized["cicpmp_standardization"] = STANDARDIZATION_MARKER

    shuffled = standardized.copy()
    shuffle_fixed_points: dict[str, int] = {}
    shuffle_sha256: dict[str, str] = {}
    for split_index, split in enumerate(("train", "validate")):
        mask = shuffled["split"].astype(str).eq(split).to_numpy()
        indices = np.flatnonzero(mask)
        rng = np.random.default_rng(args.seed + 1009 * (split_index + 1))
        order = derangement(len(indices), rng)
        source_indices = indices[order]
        shuffled.loc[mask, CICP_MP_FEATURE_NAMES] = standardized.iloc[
            source_indices
        ].loc[:, CICP_MP_FEATURE_NAMES].to_numpy()
        shuffle_fixed_points[split] = int(np.equal(order, np.arange(len(order))).sum())
        shuffle_sha256[split] = hashlib.sha256(order.tobytes()).hexdigest()

    train_standardized = standardized.loc[
        standardized["split"].astype(str).eq("train"),
        CICP_MP_FEATURE_NAMES,
    ].to_numpy(dtype=np.float64)
    train_post_means = train_standardized.mean(axis=0)
    train_post_stds = train_standardized.std(axis=0, ddof=0)
    if float(np.abs(train_post_means).max()) > 1e-6:
        raise ValueError("train-standardized means are not approximately zero")
    if float(np.abs(train_post_stds - 1.0).max()) > 1e-6:
        raise ValueError("train-standardized deviations are not approximately one")

    args.scalar_output.parent.mkdir(parents=True, exist_ok=True)
    args.mp_output.parent.mkdir(parents=True, exist_ok=True)
    args.shuffle_output.parent.mkdir(parents=True, exist_ok=True)
    scalar.to_csv(args.scalar_output, index=False)
    standardized.to_csv(args.mp_output, index=False)
    shuffled.to_csv(args.shuffle_output, index=False)

    audit = {
        "protocol": "cicpmp_fr1_five_final_repairs_v1",
        "seed": int(args.seed),
        "dry_run": bool(args.dry_run),
        "scalar_source": str(args.scalar_source),
        "scalar_source_sha256": sha256(args.scalar_source),
        "mp_source": str(args.mp_source),
        "mp_source_sha256": sha256(args.mp_source),
        "scalar_output": str(args.scalar_output),
        "mp_output": str(args.mp_output),
        "shuffle_output": str(args.shuffle_output),
        "scalar_split_counts": scalar_counts,
        "mp_split_counts": mp_counts,
        "standardization_fit_split": "train",
        "standardization_applied_splits": ["train", "validate"],
        "standardization_marker": STANDARDIZATION_MARKER,
        "feature_count": len(CICP_MP_FEATURE_NAMES),
        "semantic_blocks": {
            "increment_magnitude": list(CICP_MP_FEATURE_NAMES[0:3]),
            "attribution": list(CICP_MP_FEATURE_NAMES[3:5]),
            "confidence": list(CICP_MP_FEATURE_NAMES[5:7]),
            "direction": list(CICP_MP_FEATURE_NAMES[7:]),
        },
        "train_feature_mean_before": dict(zip(CICP_MP_FEATURE_NAMES, means.tolist())),
        "train_feature_std_before": dict(
            zip(CICP_MP_FEATURE_NAMES, standard_deviations.tolist())
        ),
        "train_post_standardization_max_abs_mean": float(
            np.abs(train_post_means).max()
        ),
        "train_post_standardization_max_abs_std_error": float(
            np.abs(train_post_stds - 1.0).max()
        ),
        "shuffle_unit": "whole_23d_row_within_split",
        "shuffle_fixed_points": shuffle_fixed_points,
        "shuffle_permutation_sha256": shuffle_sha256,
        "test_rows_passed_to_training": 0,
        "validation_item_outcomes_passed_to_training": False,
        "test_item_outcomes_read_or_generated": False,
        "forbidden_evaluation_columns": {
            "scalar": scalar_forbidden,
            "mp": mp_forbidden,
        },
    }
    args.audit_output.write_text(
        json.dumps(audit, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar-source", type=Path, required=True)
    parser.add_argument("--mp-source", type=Path, required=True)
    parser.add_argument("--scalar-output", type=Path, required=True)
    parser.add_argument("--mp-output", type=Path, required=True)
    parser.add_argument("--shuffle-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    prepare(parse_args())
