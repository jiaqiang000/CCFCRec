#!/usr/bin/env python3
"""
Build and audit Amazon-VG category availability v3 for CCFCRec.

V3 reconstructs A_i^cat from category metadata and train-graph evidence, then
residualizes it against R/S/P controls. Recoverability columns are allowed only
for audit outputs, never for feature construction.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


CONTROL_COLUMNS = [
    "category_count",
    "R_metadata_richness_score",
    "S_train_support_score",
    "P_popularity_score",
]

GRAN_SOURCE_COLUMNS = [
    "A_gran_specific_ratio",
    "A_gran_idf_mean",
    "A_gran_idf_max",
]

DISC_SOURCE_COLUMNS = [
    "A_disc_combo_rarity",
    "A_disc_generic_token_ratio",
    "A_disc_specificity_score",
]

COLLAB_SOURCE_COLUMNS = [
    "A_collab_user_set_jaccard_mean",
    "A_collab_support_entropy_mean",
]

RECOVERABILITY_AUDIT_COLUMNS = [
    "hr@20",
    "ndcg@20",
    "q_norm",
    "margin_proxy",
    "margin_to_top20_cutoff",
    "best_target_rank",
    "target_competitor_gap_proxy",
    "modality_alignment_proxy",
    "calibration_proxy",
    "consensus_score",
    "proxy_ensemble_score",
    "delta_ndcg@20",
]

V3_GROUP_LABELS = ("s_cat_v3_weak", "s_cat_v3_mid", "s_cat_v3_strong")

DESIGN_NOTE = (
    "上游设计：[[2026-07-05 111332 CCFCRec Amazon-VG category availability v3 purity audit 实验设计]]"
)


@dataclass(frozen=True)
class V3Outputs:
    item_csv: Path
    component_summary_csv: Path
    correlations_csv: Path
    group_summary_csv: Path
    route_json: Path
    result_md: Path
    manifest_json: Path


def now_stamp() -> tuple[str, str]:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} 缺少字段: {missing}")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_mean(series: pd.Series) -> float:
    values = _numeric(series).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    valid = pd.concat([_numeric(left), _numeric(right)], axis=1).dropna()
    if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))


def train_percentile(series: pd.Series, train_mask: pd.Series) -> pd.Series:
    values = _numeric(series)
    train_values = values[train_mask].replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if train_values.size <= 1 or np.unique(train_values).size <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)

    sorted_train = np.sort(train_values)
    result: list[float] = []
    for value in values.to_numpy(dtype=float):
        if not np.isfinite(value):
            result.append(0.5)
        else:
            result.append(float(np.searchsorted(sorted_train, value, side="right") / sorted_train.size))
    return pd.Series(result, index=series.index, dtype=float).clip(0.0, 1.0)


def _mean_columns(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame[columns].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0.5)


def _fit_residual(
    frame: pd.DataFrame,
    y_col: str,
    controls: list[str],
    train_mask: pd.Series,
) -> tuple[pd.Series, dict[str, Any]]:
    _require_columns(frame, [y_col, *controls], f"residual model for {y_col}")

    train = frame.loc[train_mask, [y_col, *controls]].apply(pd.to_numeric, errors="coerce")
    if train.empty:
        return pd.Series(0.0, index=frame.index, dtype=float), {
            "target": y_col,
            "controls": controls,
            "fallback": "empty_train",
            "train_r2": None,
        }

    medians = train[controls].median(numeric_only=True).fillna(0.0)
    y_median = float(train[y_col].median()) if train[y_col].notna().any() else 0.0
    train_filled = train.copy()
    train_filled[controls] = train_filled[controls].fillna(medians)
    train_filled[y_col] = train_filled[y_col].fillna(y_median)

    y_train = train_filled[y_col].to_numpy(dtype=float)
    x_train = train_filled[controls].to_numpy(dtype=float)
    design_train = np.column_stack([np.ones(len(x_train)), x_train])
    beta, *_ = np.linalg.lstsq(design_train, y_train, rcond=None)

    all_controls = frame[controls].apply(pd.to_numeric, errors="coerce").fillna(medians)
    y_all = _numeric(frame[y_col]).fillna(y_median).to_numpy(dtype=float)
    design_all = np.column_stack([np.ones(len(all_controls)), all_controls.to_numpy(dtype=float)])
    prediction_all = design_all @ beta
    residual_all = y_all - prediction_all

    prediction_train = design_train @ beta
    ss_res = float(np.square(y_train - prediction_train).sum())
    ss_tot = float(np.square(y_train - y_train.mean()).sum())
    r2 = None if ss_tot <= 0 else 1.0 - ss_res / ss_tot

    meta = {
        "target": y_col,
        "controls": controls,
        "coefficients": [float(value) for value in beta],
        "train_r2": _jsonable(r2),
        "fallback": None,
    }
    return pd.Series(residual_all, index=frame.index, dtype=float), meta


def _build_group(score: pd.Series, split: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    train_scores = _numeric(score[split.eq("train")]).dropna()
    if train_scores.empty or train_scores.nunique() <= 1:
        ranked = _numeric(score).rank(method="average", pct=True)
        return ranked.map(_rank_to_group), {
            "weak_max": None,
            "mid_max": None,
            "fallback": True,
        }
    weak_max = float(train_scores.quantile(1 / 3))
    mid_max = float(train_scores.quantile(2 / 3))
    if np.isclose(weak_max, mid_max):
        ranked = _numeric(score).rank(method="average", pct=True)
        return ranked.map(_rank_to_group), {
            "weak_max": weak_max,
            "mid_max": mid_max,
            "fallback": True,
        }
    return score.map(lambda value: _score_to_group(float(value), weak_max, mid_max)), {
        "weak_max": weak_max,
        "mid_max": mid_max,
        "fallback": False,
    }


def _rank_to_group(rank_pct: float) -> str:
    if rank_pct <= 1 / 3:
        return V3_GROUP_LABELS[0]
    if rank_pct <= 2 / 3:
        return V3_GROUP_LABELS[1]
    return V3_GROUP_LABELS[2]


def _score_to_group(score: float, weak_max: float, mid_max: float) -> str:
    if score <= weak_max:
        return V3_GROUP_LABELS[0]
    if score <= mid_max:
        return V3_GROUP_LABELS[1]
    return V3_GROUP_LABELS[2]


def build_category_availability_v3(v2: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    required = [
        "raw_asin",
        "split",
        *CONTROL_COLUMNS,
        *GRAN_SOURCE_COLUMNS,
        *DISC_SOURCE_COLUMNS,
        *COLLAB_SOURCE_COLUMNS,
    ]
    _require_columns(v2, required, "v2 availability")

    result = v2.copy()
    train_mask = result["split"].eq("train")

    result["Acat_v3_gran_specific_ratio_pct"] = train_percentile(result["A_gran_specific_ratio"], train_mask)
    result["Acat_v3_gran_idf_mean_pct"] = train_percentile(result["A_gran_idf_mean"], train_mask)
    result["Acat_v3_gran_idf_max_pct"] = train_percentile(result["A_gran_idf_max"], train_mask)
    result["Acat_v3_gran_raw"] = _mean_columns(
        result,
        [
            "Acat_v3_gran_specific_ratio_pct",
            "Acat_v3_gran_idf_mean_pct",
            "Acat_v3_gran_idf_max_pct",
        ],
    )

    result["Acat_v3_disc_combo_rarity_pct"] = train_percentile(result["A_disc_combo_rarity"], train_mask)
    result["Acat_v3_disc_inverse_generic_ratio_pct"] = train_percentile(
        1.0 - _numeric(result["A_disc_generic_token_ratio"]),
        train_mask,
    )
    result["Acat_v3_disc_specificity_pct"] = train_percentile(result["A_disc_specificity_score"], train_mask)
    result["Acat_v3_disc_raw"] = _mean_columns(
        result,
        [
            "Acat_v3_disc_combo_rarity_pct",
            "Acat_v3_disc_inverse_generic_ratio_pct",
            "Acat_v3_disc_specificity_pct",
        ],
    )

    result["Acat_v3_collab_jaccard_pct"] = train_percentile(result["A_collab_user_set_jaccard_mean"], train_mask)
    result["Acat_v3_collab_inverse_entropy_pct"] = train_percentile(
        1.0 - _numeric(result["A_collab_support_entropy_mean"]),
        train_mask,
    )
    result["Acat_v3_collab_raw"] = _mean_columns(
        result,
        [
            "Acat_v3_collab_jaccard_pct",
            "Acat_v3_collab_inverse_entropy_pct",
        ],
    )

    component_models: dict[str, Any] = {}
    for component in ["gran", "disc", "collab"]:
        raw_col = f"Acat_v3_{component}_raw"
        residual_col = f"Acat_v3_{component}_residual"
        pct_col = f"Acat_v3_{component}_residual_pct"
        residual, model_meta = _fit_residual(result, raw_col, CONTROL_COLUMNS, train_mask)
        result[residual_col] = residual
        result[pct_col] = train_percentile(residual, train_mask)
        component_models[component] = model_meta

    result["s_cat_v3"] = _mean_columns(
        result,
        [
            "Acat_v3_gran_residual_pct",
            "Acat_v3_disc_residual_pct",
            "Acat_v3_collab_residual_pct",
        ],
    ).clip(0.0, 1.0)
    groups, thresholds = _build_group(result["s_cat_v3"], result["split"])
    result["s_cat_v3_group"] = groups

    stamp, created_at = now_stamp()
    meta = {
        "created_stamp": stamp,
        "created_at": created_at,
        "dataset": "Amazon VG",
        "source_policy": "derived_from_category_availability_v2_item_columns",
        "fit_scope": "train_fitted_percentiles_and_train_residual_models",
        "stable_key": "raw_asin",
        "s_cat_policy": "v3_rsp_residualized_category_evidence_mean",
        "component_policy": {
            "gran": GRAN_SOURCE_COLUMNS,
            "disc": [
                "A_disc_combo_rarity",
                "1 - A_disc_generic_token_ratio",
                "A_disc_specificity_score",
            ],
            "collab": [
                "A_collab_user_set_jaccard_mean",
                "1 - A_collab_support_entropy_mean",
            ],
        },
        "control_columns": CONTROL_COLUMNS,
        "component_models": component_models,
        "s_cat_v3_group_policy": "train_s_cat_v3_tertiles_applied_to_all_splits",
        "s_cat_v3_group_thresholds": thresholds,
        "leakage_policy": {
            "validation_test_interactions": "not_used_for_feature_construction",
            "rank_margin_ndcg_delta_columns": "not_used_for_feature_construction",
            "recoverability_columns": "not_used_for_feature_construction",
            "recoverability_columns_if_present": "audit_only",
        },
        "input_rows": int(len(v2)),
        "output_rows": int(len(result)),
    }
    return result, meta


def build_purity_correlations(frame: pd.DataFrame) -> pd.DataFrame:
    left_columns = [
        column
        for column in [
            "s_cat_v3",
            "Acat_v3_gran_raw",
            "Acat_v3_disc_raw",
            "Acat_v3_collab_raw",
            "Acat_v3_gran_residual_pct",
            "Acat_v3_disc_residual_pct",
            "Acat_v3_collab_residual_pct",
        ]
        if column in frame.columns
    ]
    right_specs = [(column, "control") for column in CONTROL_COLUMNS if column in frame.columns]
    right_specs.extend(
        (column, "recoverability")
        for column in RECOVERABILITY_AUDIT_COLUMNS
        if column in frame.columns
    )
    if "s_cat" in frame.columns:
        right_specs.append(("s_cat", "previous_acat"))
    if "s_cat_v2" in frame.columns:
        right_specs.append(("s_cat_v2", "previous_acat"))

    rows: list[dict[str, Any]] = []
    for left in left_columns:
        for right, family in right_specs:
            valid = pd.concat([_numeric(frame[left]), _numeric(frame[right])], axis=1).dropna()
            pearson = _safe_corr(frame[left], frame[right], "pearson")
            spearman = _safe_corr(frame[left], frame[right], "spearman")
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "right_family": family,
                    "n": int(len(valid)),
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_pearson": abs(pearson) if np.isfinite(pearson) else float("nan"),
                    "abs_spearman": abs(spearman) if np.isfinite(spearman) else float("nan"),
                    "warning": family == "control" and np.isfinite(spearman) and abs(spearman) >= 0.70,
                }
            )
    return pd.DataFrame(rows)


def build_component_summary(frame: pd.DataFrame, meta: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component in ["gran", "disc", "collab"]:
        raw_col = f"Acat_v3_{component}_raw"
        pct_col = f"Acat_v3_{component}_residual_pct"
        control_corrs = []
        for control in CONTROL_COLUMNS:
            if raw_col in frame.columns and control in frame.columns:
                control_corrs.append(abs(_safe_corr(frame[raw_col], frame[control], "spearman")))
        residual_corrs = []
        for control in CONTROL_COLUMNS:
            if pct_col in frame.columns and control in frame.columns:
                residual_corrs.append(abs(_safe_corr(frame[pct_col], frame[control], "spearman")))
        model_meta = meta.get("component_models", {}).get(component, {})
        rows.append(
            {
                "component": component,
                "raw_mean": _safe_mean(frame[raw_col]) if raw_col in frame.columns else float("nan"),
                "residual_pct_mean": _safe_mean(frame[pct_col]) if pct_col in frame.columns else float("nan"),
                "raw_max_abs_spearman_control": max(control_corrs) if control_corrs else float("nan"),
                "residual_max_abs_spearman_control": max(residual_corrs) if residual_corrs else float("nan"),
                "raw_control_train_r2": model_meta.get("train_r2"),
            }
        )
    return pd.DataFrame(rows)


def build_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if "s_cat_v3_group" not in frame.columns:
        raise ValueError("group summary 需要 s_cat_v3_group 字段")

    agg: dict[str, tuple[str, str]] = {"item_count": ("raw_asin", "size")}
    candidate_columns = [
        "s_cat_v3",
        "hr@20",
        "ndcg@20",
        "margin_proxy",
        "margin_to_top20_cutoff",
        "best_target_rank",
        "proxy_ensemble_score",
        "consensus_score",
        "target_competitor_gap_proxy",
    ]
    for column in candidate_columns:
        if column in frame.columns:
            safe_name = column.replace("@", "").replace(" ", "_")
            agg[f"{safe_name}_count"] = (column, "count")
            agg[f"{safe_name}_mean"] = (column, "mean")

    return frame.groupby("s_cat_v3_group", dropna=False).agg(**agg).reset_index()


def _group_value(summary: pd.DataFrame, group: str, column: str) -> float:
    if column not in summary.columns:
        return float("nan")
    row = summary[summary["s_cat_v3_group"].eq(group)]
    if row.empty:
        return float("nan")
    return float(row.iloc[0][column])


def decide_v3_route(correlations: pd.DataFrame, group_summary: pd.DataFrame) -> dict[str, Any]:
    control_rows = correlations[
        correlations.get("left", pd.Series(dtype=object)).eq("s_cat_v3")
        & correlations.get("right_family", pd.Series(dtype=object)).eq("control")
    ]
    control_abs = pd.to_numeric(control_rows.get("abs_spearman", pd.Series(dtype=float)), errors="coerce").dropna()
    max_abs_control = float(control_abs.max()) if not control_abs.empty else float("nan")

    weak_ndcg = _group_value(group_summary, "s_cat_v3_weak", "ndcg20_mean")
    strong_ndcg = _group_value(group_summary, "s_cat_v3_strong", "ndcg20_mean")
    weak_margin = _group_value(group_summary, "s_cat_v3_weak", "margin_proxy_mean")
    strong_margin = _group_value(group_summary, "s_cat_v3_strong", "margin_proxy_mean")
    weak_proxy = _group_value(group_summary, "s_cat_v3_weak", "proxy_ensemble_score_mean")
    strong_proxy = _group_value(group_summary, "s_cat_v3_strong", "proxy_ensemble_score_mean")

    ndcg_gap = strong_ndcg - weak_ndcg if np.isfinite(strong_ndcg) and np.isfinite(weak_ndcg) else float("nan")
    margin_gap = strong_margin - weak_margin if np.isfinite(strong_margin) and np.isfinite(weak_margin) else float("nan")
    proxy_gap = strong_proxy - weak_proxy if np.isfinite(strong_proxy) and np.isfinite(weak_proxy) else float("nan")

    easy_relevant = (
        (np.isfinite(ndcg_gap) and ndcg_gap >= 0.02)
        or (np.isfinite(margin_gap) and margin_gap >= 0.25)
    )
    failure_relevant = (
        (np.isfinite(ndcg_gap) and ndcg_gap <= -0.02)
        or (np.isfinite(margin_gap) and margin_gap <= -0.25)
    )
    proxy_relevant = np.isfinite(proxy_gap) and abs(proxy_gap) >= 0.10

    relevance_direction = "not_relevant"
    if easy_relevant:
        relevance_direction = "high_acat_baseline_easier"
    elif failure_relevant:
        relevance_direction = "high_acat_baseline_harder"
    elif proxy_relevant:
        relevance_direction = "proxy_only"

    if np.isfinite(max_abs_control) and max_abs_control >= 0.70:
        route = "acat_v3_collapses_to_rsp"
        reason = "s_cat_v3 与 category_count/R/S/P 的相关过高，不能声称为独立 category availability。"
    elif group_summary["s_cat_v3_group"].nunique() < 2:
        route = "acat_v3_needs_rebuild"
        reason = "s_cat_v3_group 分层不足，无法审计推荐相关性。"
    elif easy_relevant:
        route = "acat_v3_pure_baseline_easy_relevant"
        reason = "s_cat_v3 通过 R/S/P purity gate；高 A-cat 组在 baseline 上更容易。"
    elif failure_relevant:
        route = "acat_v3_pure_baseline_failure_relevant"
        reason = "s_cat_v3 通过 R/S/P purity gate；高 A-cat 组反而是 baseline failure 更强的机会组。"
    elif proxy_relevant:
        route = "acat_v3_pure_proxy_relevant"
        reason = "s_cat_v3 通过 R/S/P purity gate，但主要只和 recoverability proxy 有可见差异。"
    else:
        route = "acat_v3_pure_not_recommendation_relevant"
        reason = "s_cat_v3 与 R/S/P 可区分，但与 baseline failure/recoverability 的关系较弱。"

    return {
        "route": route,
        "reason": reason,
        "evidence": {
            "max_abs_spearman_control": _jsonable(max_abs_control),
            "strong_minus_weak_ndcg@20": _jsonable(ndcg_gap),
            "strong_minus_weak_margin_proxy": _jsonable(margin_gap),
            "strong_minus_weak_proxy_ensemble_score": _jsonable(proxy_gap),
            "relevance_direction": relevance_direction,
        },
    }


def merge_recoverability_profiles(v3: pd.DataFrame, profile_paths: list[Path]) -> pd.DataFrame:
    result = v3.copy()
    for path in profile_paths:
        if not path.exists():
            raise FileNotFoundError(path)
        profile = pd.read_csv(path)
        if "raw_asin" not in profile.columns and "asin" in profile.columns:
            profile = profile.rename(columns={"asin": "raw_asin"})
        if "raw_asin" not in profile.columns:
            raise ValueError(f"{path} 缺少 asin/raw_asin 字段")
        keep = ["raw_asin"]
        keep.extend(
            column
            for column in RECOVERABILITY_AUDIT_COLUMNS
            if column in profile.columns and column not in result.columns
        )
        if len(keep) > 1:
            result = result.merge(profile[keep].drop_duplicates("raw_asin"), on="raw_asin", how="left")
    return result


def _md_table(frame: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    if frame.empty:
        return "_无数据_"
    selected = frame[[column for column in columns if column in frame.columns]].head(max_rows)
    if selected.empty:
        return "_无可展示列_"
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in selected.iterrows():
        cells = []
        for header in headers:
            value = row[header]
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_outputs(
    result: pd.DataFrame,
    component_summary: pd.DataFrame,
    correlations: pd.DataFrame,
    group_summary: pd.DataFrame,
    decision: dict[str, Any],
    meta: dict[str, Any],
    output_dir: Path,
    input_files: dict[str, Any],
) -> V3Outputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp, run_iso = now_stamp()

    item_csv = output_dir / "category_availability_v3_item.csv"
    component_summary_csv = output_dir / "category_availability_v3_component_summary.csv"
    correlations_csv = output_dir / "category_availability_v3_correlations.csv"
    group_summary_csv = output_dir / "category_availability_v3_group_summary.csv"
    route_json = output_dir / "category_availability_v3_route_decision.json"
    result_md = output_dir / f"{stamp} CCFCRec Amazon-VG category availability v3 purity audit 结果.md"
    manifest_json = output_dir / "run_manifest.json"

    result.to_csv(item_csv, index=False)
    component_summary.to_csv(component_summary_csv, index=False)
    correlations.to_csv(correlations_csv, index=False)
    group_summary.to_csv(group_summary_csv, index=False)
    route_json.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "run_time": run_iso,
        "script": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "input_files": input_files,
        "outputs": {
            "item_csv": str(item_csv),
            "component_summary_csv": str(component_summary_csv),
            "correlations_csv": str(correlations_csv),
            "group_summary_csv": str(group_summary_csv),
            "route_json": str(route_json),
            "result_md": str(result_md),
        },
        "row_counts": {
            "item": int(len(result)),
            "component_summary": int(len(component_summary)),
            "correlations": int(len(correlations)),
            "group_summary": int(len(group_summary)),
        },
        "v3_meta": meta,
        "route_decision": decision,
        "design_note": DESIGN_NOTE,
    }
    manifest_json.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    markdown = f"""---
title: {stamp} CCFCRec Amazon-VG category availability v3 purity audit 结果
date: {stamp[:10]}
time: "{stamp[11:13]}:{stamp[13:15]}:{stamp[15:17]}"
created_at: "{stamp[:10]} {stamp[11:13]}:{stamp[13:15]}:{stamp[15:17]}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - Acat_v3
  - purity_audit
---

# {stamp} CCFCRec Amazon-VG category availability v3 purity audit 结果

## 来源说明

> [!info] 来源说明
> {DESIGN_NOTE}
> 本结果目录：`{output_dir}/`

## Material Passport

- artifact_type: experiment_result
- status: completed
- verification_status: analyzed
- stable_key: raw_asin
- row_count: {len(result)}

## 结论

route decision 为 `{decision.get("route")}`。

{decision.get("reason", "")}

## 关键证据

```json
{json.dumps(_jsonable(decision.get("evidence", {})), ensure_ascii=False, indent=2)}
```

## Component Summary

{_md_table(component_summary, list(component_summary.columns))}

## Group Summary

{_md_table(group_summary, list(group_summary.columns))}

## Correlation Warnings

{_md_table(correlations[correlations.get("warning", False).astype(bool)] if "warning" in correlations.columns else pd.DataFrame(), list(correlations.columns))}

## 产物

- `category_availability_v3_item.csv`
- `category_availability_v3_component_summary.csv`
- `category_availability_v3_correlations.csv`
- `category_availability_v3_group_summary.csv`
- `category_availability_v3_route_decision.json`
- `run_manifest.json`
"""
    result_md.write_text(markdown, encoding="utf-8")

    return V3Outputs(
        item_csv=item_csv,
        component_summary_csv=component_summary_csv,
        correlations_csv=correlations_csv,
        group_summary_csv=group_summary_csv,
        route_json=route_json,
        result_md=result_md,
        manifest_json=manifest_json,
    )


def run_v3_audit(
    availability_v2_path: Path,
    recoverability_profile_paths: list[Path],
    output_dir: Path,
) -> V3Outputs:
    v2 = pd.read_csv(availability_v2_path)
    v3, meta = build_category_availability_v3(v2)
    audit_frame = merge_recoverability_profiles(v3, recoverability_profile_paths)
    component_summary = build_component_summary(audit_frame, meta)
    correlations = build_purity_correlations(audit_frame)
    group_summary = build_group_summary(audit_frame)
    decision = decide_v3_route(correlations, group_summary)
    meta["source_path"] = str(availability_v2_path)
    meta["recoverability_profile_paths"] = [str(path) for path in recoverability_profile_paths]
    return write_outputs(
        result=audit_frame,
        component_summary=component_summary,
        correlations=correlations,
        group_summary=group_summary,
        decision=decision,
        meta=meta,
        output_dir=output_dir,
        input_files={
            "availability_v2": str(availability_v2_path),
            "recoverability_profiles": [str(path) for path in recoverability_profile_paths],
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability-v2", type=Path, required=True)
    parser.add_argument("--recoverability-profile", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_v3_audit(
        availability_v2_path=args.availability_v2,
        recoverability_profile_paths=list(args.recoverability_profile),
        output_dir=args.output_dir,
    )
    print(f"wrote {outputs.item_csv}")
    print(f"wrote {outputs.component_summary_csv}")
    print(f"wrote {outputs.correlations_csv}")
    print(f"wrote {outputs.group_summary_csv}")
    print(f"wrote {outputs.route_json}")
    print(f"wrote {outputs.result_md}")
    print(f"wrote {outputs.manifest_json}")


if __name__ == "__main__":
    main()
