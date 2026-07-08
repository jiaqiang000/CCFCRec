#!/usr/bin/env python3
"""
Build v2 within-control category availability variables for Amazon-VG.

V2 converts disc/collab scores into train-fitted within-control percentiles.
It keeps v1 columns for audit and writes v2 into standard s_cat/s_cat_group.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CONTROL_BIN_SPECS = {
    "category_count_bin": ("category_count", 4),
    "R_bin": ("R_metadata_richness_score", 4),
    "S_bin": ("S_train_support_score", 5),
    "P_bin": ("P_popularity_score", 5),
}
CONTROL_BIN_COLUMNS = list(CONTROL_BIN_SPECS.keys())


@dataclass(frozen=True)
class V2Outputs:
    item_csv: Path
    meta_json: Path


def now_info() -> tuple[str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def required_columns() -> set[str]:
    return {
        "raw_asin",
        "split",
        "category_count",
        "R_metadata_richness_score",
        "S_train_support_score",
        "P_popularity_score",
        "s_cat_disc",
        "s_cat_collab",
        "s_cat",
        "s_cat_group",
    }


def assert_required_columns(frame: pd.DataFrame) -> None:
    missing = required_columns() - set(frame.columns)
    if missing:
        raise ValueError(f"v1 availability 缺少字段: {sorted(missing)}")


def fit_edges(train_series: pd.Series, q: int) -> list[float] | None:
    values = pd.to_numeric(train_series, errors="coerce").dropna().to_numpy(dtype=float)
    if len(np.unique(values)) <= 1:
        return None
    edges = np.unique(np.quantile(values, np.linspace(0, 1, q + 1)))
    if len(edges) <= 2:
        return None
    edges[0] = -np.inf
    edges[-1] = np.inf
    return [float(value) for value in edges]


def apply_edges(series: pd.Series, edges: list[float] | None) -> pd.Series:
    if edges is None:
        return pd.Series([0] * len(series), index=series.index, dtype="int64")
    labels = pd.cut(
        pd.to_numeric(series, errors="coerce"),
        bins=edges,
        labels=False,
        include_lowest=True,
    )
    return labels.fillna(0).astype("int64")


def fit_control_bins(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    assert_required_columns(frame)
    train = frame[frame["split"].eq("train")]
    bins = pd.DataFrame(index=frame.index)
    edge_meta: dict[str, object] = {}
    for output_col, (source_col, q) in CONTROL_BIN_SPECS.items():
        edges = fit_edges(train[source_col], q=q)
        bins[output_col] = apply_edges(frame[source_col], edges)
        edge_meta[output_col] = {
            "source_col": source_col,
            "q": q,
            "edges": edges,
        }
    return bins, {"fit_scope": "train_control_bins", "control_bins": edge_meta}


def cell_keys(bins: pd.DataFrame) -> pd.Series:
    return pd.Series(list(map(tuple, bins[CONTROL_BIN_COLUMNS].to_numpy())), index=bins.index)


def percentile(value: float, sorted_train_values: np.ndarray) -> float:
    if sorted_train_values.size <= 1 or not np.isfinite(value):
        return 0.5
    return float(np.searchsorted(sorted_train_values, value, side="right") / sorted_train_values.size)


def percentile_against_train_cells(frame: pd.DataFrame, bins: pd.DataFrame, score_col: str) -> pd.Series:
    train_mask = frame["split"].eq("train")
    keys = cell_keys(bins)
    global_values = np.sort(
        pd.to_numeric(frame.loc[train_mask, score_col], errors="coerce").dropna().to_numpy(dtype=float)
    )
    cell_values: dict[tuple[int, ...], np.ndarray] = {}
    train_work = pd.DataFrame({"key": keys[train_mask], "score": pd.to_numeric(frame.loc[train_mask, score_col], errors="coerce")})
    for key, group in train_work.dropna().groupby("key"):
        values = np.sort(group["score"].to_numpy(dtype=float))
        if values.size:
            cell_values[key] = values

    result = []
    scores = pd.to_numeric(frame[score_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    for idx, value in enumerate(scores):
        values = cell_values.get(keys.iloc[idx], global_values)
        result.append(percentile(float(value), values))
    return pd.Series(result, index=frame.index, dtype=float)


def score_to_group(value: float, weak_max: float, mid_max: float) -> str:
    if value <= weak_max:
        return "s_cat_v2_weak"
    if value <= mid_max:
        return "s_cat_v2_mid"
    return "s_cat_v2_strong"


def build_v2_group(score: pd.Series, split: pd.Series) -> tuple[pd.Series, dict[str, float | bool]]:
    train_scores = score[split.eq("train")].dropna()
    if train_scores.empty:
        ranked = score.rank(method="average", pct=True)
        return ranked.map(lambda value: score_to_group(float(value), 1 / 3, 2 / 3)), {
            "weak_max": np.nan,
            "mid_max": np.nan,
            "fallback": True,
        }
    weak_max = float(train_scores.quantile(1 / 3))
    mid_max = float(train_scores.quantile(2 / 3))
    if np.isclose(weak_max, mid_max):
        ranked = score.rank(method="average", pct=True)
        return ranked.map(lambda value: score_to_group(float(value), 1 / 3, 2 / 3)), {
            "weak_max": weak_max,
            "mid_max": mid_max,
            "fallback": True,
        }
    return score.map(lambda value: score_to_group(float(value), weak_max, mid_max)), {
        "weak_max": weak_max,
        "mid_max": mid_max,
        "fallback": False,
    }


def build_category_availability_v2(v1: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    assert_required_columns(v1)
    result = v1.copy()
    result["s_cat_v1"] = result["s_cat"]
    result["s_cat_group_v1"] = result["s_cat_group"]
    bins, bin_meta = fit_control_bins(result)
    for column in CONTROL_BIN_COLUMNS:
        result[f"v2_{column}"] = bins[column]

    result["s_cat_v2_disc_within_control"] = percentile_against_train_cells(result, bins, "s_cat_disc")
    result["s_cat_v2_collab_within_control"] = percentile_against_train_cells(result, bins, "s_cat_collab")
    result["s_cat_v2"] = result[["s_cat_v2_disc_within_control", "s_cat_v2_collab_within_control"]].mean(axis=1)
    groups, thresholds = build_v2_group(result["s_cat_v2"], result["split"])
    result["s_cat_v2_group"] = groups
    result["s_cat"] = result["s_cat_v2"]
    result["s_cat_group"] = result["s_cat_v2_group"]

    stamp, created_at = now_info()
    meta = {
        "created_stamp": stamp,
        "created_at": created_at,
        "dataset": "Amazon VG",
        "fit_scope": "train_control_bins_and_train_cell_percentiles",
        "source_policy": "derived_from_category_availability_item_v1",
        "s_cat_policy": "v2_within_control_disc_collab_mean",
        "stable_key": "raw_asin",
        "control_columns": [source for source, _ in CONTROL_BIN_SPECS.values()],
        "v2_score_columns": [
            "s_cat_v2_disc_within_control",
            "s_cat_v2_collab_within_control",
            "s_cat_v2",
        ],
        "s_cat_group_policy": "train_s_cat_v2_tertiles_applied_to_all_splits",
        "s_cat_group_thresholds": thresholds,
        "bin_meta": bin_meta,
        "input_rows": int(len(v1)),
        "output_rows": int(len(result)),
    }
    return result, meta


def write_v2_outputs(result: pd.DataFrame, meta: dict[str, object], output_dir: Path) -> V2Outputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    item_csv = output_dir / "category_availability_v2_item.csv"
    meta_json = output_dir / "category_availability_v2_meta.json"
    result.to_csv(item_csv, index=False)
    meta_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return V2Outputs(item_csv=item_csv, meta_json=meta_json)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    v1 = pd.read_csv(args.availability)
    result, meta = build_category_availability_v2(v1)
    meta["source_path"] = str(args.availability)
    outputs = write_v2_outputs(result, meta, args.output_dir)
    print(f"wrote {outputs.item_csv}")
    print(f"wrote {outputs.meta_json}")


if __name__ == "__main__":
    main()
