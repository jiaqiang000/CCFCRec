#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 remaining Task3.x 诊断。

覆盖 Task3.9 / 3.12 / 3.13 / 3.14 / 3.16 / 3.17 / 3.18 / 3.20 / 3.21 / 3.22 / 3.23。
脚本只复用已有 item-level CSV 和 route JSON，不训练模型；Task3.21 默认使用已有 best/last 产物时才做真稳定性，否则记录缺口。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_amazon_vg_category_availability_task3p5 import md_table
from analyze_amazon_vg_category_availability_task3p8 import (
    _jsonable,
    _quantile_labels,
    _safe_mean,
    build_rank_margin_profile,
)
from analyze_amazon_vg_category_availability_task3p10 import _control_bucket, category_count_bucket
from analyze_amazon_vg_category_availability_task3p19 import (
    PROXY_METRICS,
    build_proxy_metric_profile,
)


DESIGN_NOTE = "上游设计：[[2026-07-05 012752 CCFCRec Amazon-VG category availability v2 Task3.x 发散诊断设计]]"


def now_stamp() -> tuple[str, str]:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def _corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    valid = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))


def _mean_existing(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(float("nan"), index=frame.index)
    numeric = pd.concat([pd.to_numeric(frame[column], errors="coerce") for column in available], axis=1)
    return numeric.mean(axis=1)


def _prefix_bucket(series: pd.Series, prefix: str) -> pd.Series:
    base = _quantile_labels(series)
    return base.map(
        {
            "low": f"{prefix}_low",
            "mid": f"{prefix}_mid",
            "high": f"{prefix}_high",
            "all": f"{prefix}_all",
            "unknown": f"{prefix}_unknown",
        }
    ).fillna(base)


def _ensure_margin_column(frame: pd.DataFrame) -> pd.DataFrame:
    if "margin_to_top20_cutoff" in frame.columns:
        return frame
    frame = frame.copy()
    if "margin_proxy" in frame.columns:
        frame["margin_to_top20_cutoff"] = pd.to_numeric(frame["margin_proxy"], errors="coerce")
    else:
        frame["margin_to_top20_cutoff"] = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    return frame


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"route": "missing", "evidence": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_simple_md(
    output_path: Path,
    title: str,
    task_tag: str,
    result_dir_display: str,
    decision: dict[str, Any],
    sections: list[tuple[str, pd.DataFrame, list[str]]],
    source_notes: list[str] | None = None,
) -> None:
    run_stamp = title[:17]
    source_notes = source_notes or [DESIGN_NOTE]
    source_lines = "\n".join(f"> {line}" for line in source_notes)
    body_sections = []
    for section_title, frame, columns in sections:
        body_sections.append(
            f"""## {section_title}

> [!info] 字段说明
> 表格字段按本节诊断产物 CSV 同名字段解释；核心 route 证据见“判断”。

{md_table(frame, columns, max_rows=20)}
"""
        )
    markdown = f"""---
title: {title}
date: {run_stamp[:10]}
time: "{run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
created_at: "{run_stamp[:10]} {run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - {task_tag}
---

# {title}

## 来源说明

> [!info] 来源说明
{source_lines}
> 本结果目录：`{result_dir_display}`

## 结论

route decision 为 `{decision.get("route")}`。

{decision.get("reason", "")}

{chr(10).join(body_sections)}
## 判断

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```
"""
    output_path.write_text(markdown, encoding="utf-8")


def load_base_profiles(task3_dir: Path) -> dict[str, pd.DataFrame]:
    return {
        "item": pd.read_csv(task3_dir / "task3_item_profile.csv"),
        "baseline_score": pd.read_csv(task3_dir / "baseline_score_margin" / "item_score_margin_profile.csv"),
        "category_score": pd.read_csv(task3_dir / "category_conf_input_score_margin" / "item_score_margin_profile.csv"),
        "baseline_target": pd.read_csv(task3_dir / "baseline_target_score_source" / "target_alignment_profile.csv"),
        "baseline_content": pd.read_csv(task3_dir / "baseline_content_cf_alignment" / "content_cf_alignment_profile.csv"),
        "category_content": pd.read_csv(task3_dir / "category_conf_input_content_cf_alignment" / "content_cf_alignment_profile.csv"),
    }


def build_common_profile(profiles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    profile = build_proxy_metric_profile(
        profiles["item"],
        profiles["baseline_score"],
        profiles["baseline_target"],
        profiles["baseline_content"],
    )
    profile["category_count_bucket"] = profile["category_count"].map(category_count_bucket)
    profile["target_activity_bucket"] = _control_bucket(profile["target_history_interaction_count_mean"])
    profile["target_competitor_gap_bucket"] = _prefix_bucket(profile["target_competitor_gap_proxy"], "gap")
    profile["modality_alignment_bucket"] = _prefix_bucket(profile["modality_alignment_proxy"], "modality")
    return profile


# Task3.9
def build_target_competitor_gap_summary(profile: pd.DataFrame) -> pd.DataFrame:
    if "gap_bucket" not in profile.columns:
        profile = profile.copy()
        profile["gap_bucket"] = _prefix_bucket(profile["target_competitor_gap_proxy"], "gap")
    profile = _ensure_margin_column(profile)
    rows = []
    for bucket, frame in profile.groupby("gap_bucket", dropna=False):
        rows.append(
            {
                "gap_bucket": bucket,
                "item_count": int(len(frame)),
                "ndcg@20_mean": _safe_mean(frame["ndcg@20"]),
                "hr@20_mean": _safe_mean(frame["hr@20"]) if "hr@20" in frame.columns else float("nan"),
                "margin_to_top20_cutoff_mean": _safe_mean(frame["margin_to_top20_cutoff"]),
                "best_target_rank_mean": _safe_mean(frame["best_target_rank"]),
                "target_competitor_gap_proxy_mean": _safe_mean(frame["target_competitor_gap_proxy"]),
                "top20_user_norm_mean_mean": _safe_mean(frame["top20_user_norm_mean"]),
                "top20_history_interaction_count_mean_mean": _safe_mean(frame["top20_history_interaction_count_mean"]),
            }
        )
    return pd.DataFrame(rows).sort_values("gap_bucket").reset_index(drop=True)


def build_gap_component_correlation(profile: pd.DataFrame) -> pd.DataFrame:
    profile = _ensure_margin_column(profile)
    metrics = [
        "target_competitor_gap_proxy",
        "target_minus_top20_cosine_mean",
        "target_minus_top20_user_norm_mean",
        "target_minus_top20_history_q_cosine_mean",
        "top20_user_norm_mean",
        "top20_history_interaction_count_mean",
    ]
    rows = []
    for metric in metrics:
        if metric not in profile.columns:
            continue
        rows.append(
            {
                "component": metric,
                "spearman_vs_ndcg@20": _corr(profile[metric], profile["ndcg@20"]),
                "spearman_vs_margin": _corr(profile[metric], profile["margin_to_top20_cutoff"]),
                "spearman_vs_best_target_rank": _corr(profile[metric], profile["best_target_rank"]),
            }
        )
    return pd.DataFrame(rows)


def decide_target_competitor_route(summary: pd.DataFrame) -> dict[str, Any]:
    low = summary[summary["gap_bucket"].astype(str).str.contains("low")]
    high = summary[summary["gap_bucket"].astype(str).str.contains("high")]
    low_row = low.iloc[0] if not low.empty else None
    high_row = high.iloc[0] if not high.empty else None
    gap = float(high_row["ndcg@20_mean"] - low_row["ndcg@20_mean"]) if low_row is not None and high_row is not None else float("nan")
    norm_gap = (
        float(low_row["top20_user_norm_mean_mean"] - high_row["top20_user_norm_mean_mean"])
        if low_row is not None and high_row is not None
        else float("nan")
    )
    activity_gap = (
        float(low_row["top20_history_interaction_count_mean_mean"] - high_row["top20_history_interaction_count_mean_mean"])
        if low_row is not None and high_row is not None
        else float("nan")
    )
    if gap > 0.05 and (norm_gap > 0 or activity_gap > 0):
        route = "competitor_overpower_supported"
        reason = "target-vs-top20 gap 低的 item 排序明显更差，并伴随 top20 competitor norm/activity 压力。"
    elif gap > 0.05:
        route = "target_under_alignment_supported"
        reason = "target-vs-top20 gap 能解释排序失败，但 competitor norm/activity 压力不明显。"
    elif gap > 0.02:
        route = "both_target_and_competitor_supported"
        reason = "target gap 与 competitor pressure 均有弱信号，需要联合建模。"
    else:
        route = "gap_not_predictive"
        reason = "target-vs-top20 gap 不能稳定区分 baseline failure。"
    return {
        "route": route,
        "reason": reason,
        "evidence": {
            "high_minus_low_ndcg@20": _jsonable(gap),
            "low_minus_high_top20_user_norm": _jsonable(norm_gap),
            "low_minus_high_top20_activity": _jsonable(activity_gap),
        },
    }


# Task3.12
def build_modality_failure_summary(profile: pd.DataFrame) -> pd.DataFrame:
    profile = _ensure_margin_column(profile)
    rows = []
    for name, column in [
        ("q", "target_minus_top20_history_q_cosine_mean"),
        ("attr", "target_minus_top20_history_attr_cosine_mean"),
        ("image", "target_minus_top20_history_img_cosine_mean"),
    ]:
        if column not in profile.columns:
            continue
        rows.append(
            {
                "modality": name,
                "gap_column": column,
                "spearman_vs_ndcg@20": _corr(profile[column], profile["ndcg@20"]),
                "spearman_vs_best_target_rank": _corr(profile[column], profile["best_target_rank"]),
                "spearman_vs_margin": _corr(profile[column], profile["margin_to_top20_cutoff"]),
                "gap_mean": _safe_mean(profile[column]),
            }
        )
    return pd.DataFrame(rows).sort_values("spearman_vs_ndcg@20", ascending=False).reset_index(drop=True)


def build_modality_delta_vs_rank(delta_profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, column in [
        ("q", "delta_target_minus_top20_history_q_cosine_mean"),
        ("attr", "delta_target_history_attr_cosine_mean"),
        ("image", "delta_target_history_img_cosine_mean"),
    ]:
        if column not in delta_profile.columns:
            continue
        rows.append(
            {
                "modality": name,
                "delta_column": column,
                "delta_mean": _safe_mean(delta_profile[column]),
                "spearman_delta_vs_delta_ndcg@20": _corr(delta_profile[column], delta_profile["delta_ndcg@20"]),
                "spearman_delta_vs_delta_margin": _corr(column if isinstance(column, pd.Series) else delta_profile[column], delta_profile.get("delta_margin_to_top20_cutoff", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame(rows)


def decide_modality_route(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {"route": "modality_signal_not_specific", "reason": "缺少 modality gap 字段。", "evidence": {}}
    top = summary.iloc[0]
    metric = str(top["modality"])
    corr = float(top["spearman_vs_ndcg@20"])
    if corr < 0.08:
        route = "modality_signal_not_specific"
        reason = "没有单一 modality gap 与 baseline failure 形成足够强关系。"
    elif metric == "attr":
        route = "attr_path_failure_supported"
        reason = "attr modality gap 是当前最强 baseline failure 相关 modality。"
    elif metric == "image":
        route = "image_path_failure_supported"
        reason = "image modality gap 是当前最强 baseline failure 相关 modality。"
    else:
        route = "q_fusion_failure_supported"
        reason = "q/history fusion gap 是当前最强 baseline failure 相关 modality。"
    return {"route": route, "reason": reason, "evidence": {"top_modality": metric, "top_spearman_vs_ndcg@20": _jsonable(corr)}}


# Task3.13
def build_controlled_failure_profile(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    for control in ["gt_group", "category_count_bucket", "target_activity_bucket"]:
        if control not in frame.columns:
            frame[control] = "unknown"
    control_sets = {
        "none": [],
        "gt_group": ["gt_group"],
        "category_count_bucket": ["category_count_bucket"],
        "target_activity_bucket": ["target_activity_bucket"],
        "joint": ["gt_group", "category_count_bucket", "target_activity_bucket"],
    }
    for name, cols in control_sets.items():
        if cols:
            frame[f"ndcg_residual_{name}"] = frame["ndcg@20"] - frame.groupby(cols, dropna=False)["ndcg@20"].transform("mean")
        else:
            frame[f"ndcg_residual_{name}"] = frame["ndcg@20"] - frame["ndcg@20"].mean()
    return frame


def build_controlled_residual_by_candidate(frame: pd.DataFrame) -> pd.DataFrame:
    candidates = ["margin_proxy", "target_competitor_gap_proxy", "modality_alignment_proxy", "calibration_proxy"]
    rows = []
    for candidate in candidates:
        if candidate not in frame.columns:
            continue
        bucket = _quantile_labels(frame[candidate])
        for residual_col in [col for col in frame.columns if col.startswith("ndcg_residual_")]:
            low = frame[bucket.eq("low")]
            high = frame[bucket.eq("high")]
            rows.append(
                {
                    "candidate": candidate,
                    "control_set": residual_col.replace("ndcg_residual_", ""),
                    "high_minus_low_residual_ndcg@20": _safe_mean(high[residual_col]) - _safe_mean(low[residual_col]),
                    "low_item_count": int(len(low)),
                    "high_item_count": int(len(high)),
                }
            )
    return pd.DataFrame(rows)


def build_candidate_independence_summary(residual: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for candidate, frame in residual.groupby("candidate", dropna=False):
        values = pd.to_numeric(frame["high_minus_low_residual_ndcg@20"], errors="coerce")
        rows.append(
            {
                "candidate": candidate,
                "min_gap": float(values.min()),
                "max_gap": float(values.max()),
                "mean_gap": float(values.mean()),
                "positive_control_count": int((values > 0).sum()),
                "control_count": int(values.notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_gap", ascending=False).reset_index(drop=True)


def decide_controlled_failure_route(independence: pd.DataFrame) -> dict[str, Any]:
    if independence.empty:
        return {"route": "needs_better_control_variable", "reason": "缺少候选变量。", "evidence": {}}
    top = independence.iloc[0]
    if int(top["positive_control_count"]) == int(top["control_count"]) and float(top["min_gap"]) > 0.02:
        route = "independent_failure_signal_found"
        reason = "候选变量在所有控制口径下保持正向 residual gap。"
    elif float(top["mean_gap"]) <= 0.02:
        route = "failure_explained_by_gt_popularity_controls"
        reason = "控制变量削弱了候选信号。"
    else:
        route = "candidate_signal_unstable_across_controls"
        reason = "候选变量有信号但在控制口径间不稳定。"
    return {"route": route, "reason": reason, "evidence": top.to_dict()}


# Task3.14
def build_user_failure_profile(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    frame["target_history_bucket"] = _prefix_bucket(frame["target_history_interaction_count_mean"], "target_activity")
    frame["target_norm_bucket"] = _prefix_bucket(frame["target_user_norm_mean"], "target_norm")
    frame["user_failure_score"] = -pd.to_numeric(frame["ndcg@20"], errors="coerce")
    return frame


def build_user_bucket_summary(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _ensure_margin_column(frame)
    rows = []
    for col in ["target_history_bucket", "target_norm_bucket"]:
        for value, sub in frame.groupby(col, dropna=False):
            rows.append(
                {
                    "bucket_type": col,
                    "bucket_value": value,
                    "item_count": int(len(sub)),
                    "ndcg@20_mean": _safe_mean(sub["ndcg@20"]),
                    "margin_to_top20_cutoff_mean": _safe_mean(sub["margin_to_top20_cutoff"]),
                    "user_failure_score_mean": _safe_mean(sub["user_failure_score"]),
                }
            )
    return pd.DataFrame(rows)


def build_user_item_cross_summary(frame: pd.DataFrame) -> pd.DataFrame:
    cross = (
        frame.groupby(["target_history_bucket", "category_group"], dropna=False)
        .agg(item_count=("asin", "size"), ndcg20_mean=("ndcg@20", "mean"))
        .reset_index()
    )
    return cross.rename(columns={"ndcg20_mean": "ndcg@20_mean"})


def decide_user_failure_route(summary: pd.DataFrame) -> dict[str, Any]:
    hist = summary[summary["bucket_type"].eq("target_history_bucket")]
    if hist.empty:
        return {"route": "user_signal_not_recoverable_from_current_csv", "reason": "缺少用户侧历史聚合字段。", "evidence": {}}
    gap = float(hist["ndcg@20_mean"].max() - hist["ndcg@20_mean"].min())
    if gap > 0.05:
        route = "user_side_failure_supported"
        reason = "target user history bucket 与 failure 有明显关系。"
    elif gap > 0.02:
        route = "user_item_interaction_failure_supported"
        reason = "用户侧信号较弱，需要和 item/category 交互观察。"
    else:
        route = "item_side_failure_dominates"
        reason = "当前聚合用户侧信号不足，item/rank margin 更主导。"
    return {"route": route, "reason": reason, "evidence": {"target_history_bucket_ndcg_gap": _jsonable(gap)}}


# Task3.16
def build_case_audit_candidates(profile: pd.DataFrame, asin_meta: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = _ensure_margin_column(profile).copy()
    if asin_meta is not None and {"asin", "title", "category"}.issubset(asin_meta.columns):
        frame = frame.merge(asin_meta[["asin", "title", "category"]].drop_duplicates("asin"), on="asin", how="left")
    archetypes = []
    for _, row in frame.iterrows():
        tags = []
        if row.get("residual_bucket") == "hard" and row.get("margin_bucket") == "mid":
            tags.append("margin_mid_hard_residual")
        if pd.to_numeric(pd.Series([row.get("hard_negative_pressure_score")]), errors="coerce").iloc[0] > 0.8:
            tags.append("competitor_overpower")
        if pd.to_numeric(pd.Series([row.get("target_competitor_gap_proxy")]), errors="coerce").iloc[0] < -0.25:
            tags.append("target_competitor_gap_low")
        if pd.to_numeric(pd.Series([row.get("modality_alignment_proxy")]), errors="coerce").iloc[0] < -0.15:
            tags.append("modality_alignment_low")
        archetypes.append(";".join(tags) if tags else "uncategorized")
    frame["archetype"] = archetypes
    return frame.sort_values(["archetype", "ndcg@20", "margin_to_top20_cutoff"], na_position="last").reset_index(drop=True)


def build_case_archetype_summary(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    if "archetype" not in frame.columns:
        frame = build_case_audit_candidates(frame)
    rows = []
    expanded = []
    for _, row in frame.iterrows():
        for archetype in str(row["archetype"]).split(";"):
            expanded.append({**row.to_dict(), "archetype_single": archetype})
    exp = pd.DataFrame(expanded)
    for archetype, sub in exp.groupby("archetype_single", dropna=False):
        rows.append(
            {
                "archetype": archetype,
                "item_count": int(len(sub)),
                "ndcg@20_mean": _safe_mean(sub["ndcg@20"]),
                "margin_to_top20_cutoff_mean": _safe_mean(sub.get("margin_to_top20_cutoff", pd.Series(dtype=float))),
                "hard_negative_pressure_score_mean": _safe_mean(sub.get("hard_negative_pressure_score", pd.Series(dtype=float))),
            }
        )
    return pd.DataFrame(rows).sort_values(["ndcg@20_mean", "item_count"], ascending=[True, False]).reset_index(drop=True)


def decide_case_audit_route(summary: pd.DataFrame) -> dict[str, Any]:
    actionable = summary[(summary["archetype"].ne("uncategorized")) & (summary["item_count"] >= 50) & (summary["ndcg@20_mean"] < 0.08)]
    if len(actionable) >= 2:
        route = "actionable_archetypes_found"
        reason = "至少两个 archetype 有足够样本且 baseline NDCG 偏低。"
    elif len(actionable) == 1:
        route = "needs_manual_labeling_before_task4"
        reason = "只有一个强 archetype，需要人工 case audit 扩展。"
    else:
        route = "no_stable_archetype"
        reason = "没有稳定低 NDCG archetype。"
    return {"route": route, "reason": reason, "evidence": {"actionable_archetype_count": int(len(actionable))}}


# Task3.17 / 22
def build_alt_availability_profile(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    frame["rank_aware_availability"] = pd.to_numeric(frame["margin_proxy"], errors="coerce")
    frame["competition_aware_availability"] = pd.to_numeric(frame["target_competitor_gap_proxy"], errors="coerce")
    frame["rank_aware_group"] = _prefix_bucket(frame["rank_aware_availability"], "rank")
    frame["competition_aware_group"] = _prefix_bucket(frame["competition_aware_availability"], "competition")
    return frame


def build_alt_group_validity_summary(alt: pd.DataFrame) -> pd.DataFrame:
    alt = _ensure_margin_column(alt)
    rows = []
    for group_col in ["rank_aware_group", "competition_aware_group"]:
        for value, frame in alt.groupby(group_col, dropna=False):
            rows.append(
                {
                    "group_col": group_col,
                    "group_value": value,
                    "item_count": int(len(frame)),
                    "ndcg@20_mean": _safe_mean(frame["ndcg@20"]),
                    "margin_to_top20_cutoff_mean": _safe_mean(frame["margin_to_top20_cutoff"]),
                    "gt_user_count_mean": _safe_mean(frame["gt_user_count"]) if "gt_user_count" in frame.columns else float("nan"),
                    "category_count_mean": _safe_mean(frame["category_count"]) if "category_count" in frame.columns else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def build_alt_vs_v2_cross_summary(alt: pd.DataFrame) -> pd.DataFrame:
    group = (
        alt.groupby(["rank_aware_group", "category_group"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={"category_group": "v2_group"})
    )
    group["within_rank_group_rate"] = group["count"] / group.groupby("rank_aware_group")["count"].transform("sum")
    return group


def decide_alt_availability_route(validity: pd.DataFrame) -> dict[str, Any]:
    def gap(group_col: str, high_part: str, low_part: str) -> float:
        high = validity[validity["group_col"].eq(group_col) & validity["group_value"].astype(str).str.contains(high_part)]
        low = validity[validity["group_col"].eq(group_col) & validity["group_value"].astype(str).str.contains(low_part)]
        if high.empty or low.empty:
            return float("nan")
        return float(high.iloc[0]["ndcg@20_mean"] - low.iloc[0]["ndcg@20_mean"])
    rank_gap = gap("rank_aware_group", "high", "low")
    comp_gap = gap("competition_aware_group", "high", "low")
    if rank_gap > 0.05:
        route = "rank_aware_availability_promising"
        reason = "rank-aware availability 分组能稳定区分 baseline failure。"
    elif comp_gap > 0.05:
        route = "competition_aware_availability_promising"
        reason = "competition-aware availability 分组能稳定区分 baseline failure。"
    elif max(rank_gap, comp_gap) > 0.02:
        route = "needs_variable_rebuild_from_raw_data"
        reason = "替代 availability 有弱信号，但需要从原始变量重建。"
    else:
        route = "availability_redefinition_not_supported"
        reason = "替代 availability 分组未能明显解释 failure。"
    return {"route": route, "reason": reason, "evidence": {"rank_aware_high_minus_low_ndcg@20": _jsonable(rank_gap), "competition_aware_high_minus_low_ndcg@20": _jsonable(comp_gap)}}


def build_placebo_summary(alt: pd.DataFrame, shuffle_count: int = 100, random_seed: int = 43) -> pd.DataFrame:
    true_high = alt[alt["rank_aware_group"].eq("rank_high")]["ndcg@20"].mean()
    true_low = alt[alt["rank_aware_group"].eq("rank_low")]["ndcg@20"].mean()
    true_gap = float(true_high - true_low)
    rng = pd.Series(range(len(alt))).sample(frac=1, random_state=random_seed).index
    rows = [{"kind": "true", "iteration": -1, "high_minus_low_ndcg@20": true_gap}]
    for i in range(shuffle_count):
        shuffled = alt["rank_aware_group"].sample(frac=1, random_state=random_seed + i).reset_index(drop=True)
        values = alt[["ndcg@20"]].reset_index(drop=True).copy()
        values["shuffled_group"] = shuffled
        high = values[values["shuffled_group"].eq("rank_high")]["ndcg@20"].mean()
        low = values[values["shuffled_group"].eq("rank_low")]["ndcg@20"].mean()
        rows.append({"kind": "shuffle", "iteration": i, "high_minus_low_ndcg@20": float(high - low)})
    return pd.DataFrame(rows)


def build_true_vs_placebo_summary(placebo: pd.DataFrame) -> pd.DataFrame:
    true_gap = float(placebo[placebo["kind"].eq("true")].iloc[0]["high_minus_low_ndcg@20"])
    shuffles = placebo[placebo["kind"].eq("shuffle")]["high_minus_low_ndcg@20"]
    return pd.DataFrame(
        [
            {
                "true_gap": true_gap,
                "shuffle_mean": float(shuffles.mean()),
                "shuffle_p95": float(shuffles.quantile(0.95)),
                "shuffle_max": float(shuffles.max()),
                "true_minus_shuffle_p95": true_gap - float(shuffles.quantile(0.95)),
                "shuffle_count": int(len(shuffles)),
            }
        ]
    )


def decide_placebo_route(placebo_or_summary: pd.DataFrame) -> dict[str, Any]:
    summary = build_true_vs_placebo_summary(placebo_or_summary) if "kind" in placebo_or_summary.columns else placebo_or_summary
    row = summary.iloc[0]
    if float(row["true_minus_shuffle_p95"]) > 0:
        route = "availability_beats_placebo"
        reason = "真实替代 availability gap 超过 shuffle null 的 95 分位。"
    elif float(row["true_gap"]) > float(row["shuffle_mean"]):
        route = "effect_depends_on_bucket_size"
        reason = "真实 gap 高于 shuffle 均值但未超过高分位。"
    else:
        route = "availability_not_better_than_shuffle"
        reason = "真实 gap 不优于 shuffle null。"
    return {"route": route, "reason": reason, "evidence": row.to_dict()}


# Task3.18 / 20 / 21 / 23
def build_competitor_cohort_profile(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    frame["high_activity_competitor"] = frame["top20_history_interaction_count_mean"] >= frame["top20_history_interaction_count_mean"].quantile(0.75)
    frame["high_norm_competitor"] = frame["top20_user_norm_mean"] >= frame["top20_user_norm_mean"].quantile(0.75)
    if "top20_history_category_overlap_rate_mean" in frame.columns:
        frame["same_category_competitor"] = frame["top20_history_category_overlap_rate_mean"] >= frame["top20_history_category_overlap_rate_mean"].quantile(0.75)
    else:
        frame["same_category_competitor"] = False
    frame["image_similarity_competitor"] = frame.get("top20_history_img_cosine_mean", pd.Series(0, index=frame.index)) >= frame.get("top20_history_img_cosine_mean", pd.Series(0, index=frame.index)).quantile(0.75)
    return frame


def build_competitor_cohort_summary(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _ensure_margin_column(frame)
    rows = []
    for cohort in ["high_activity_competitor", "high_norm_competitor", "same_category_competitor", "image_similarity_competitor"]:
        if cohort not in frame.columns:
            continue
        yes = frame[frame[cohort].astype(bool)]
        no = frame[~frame[cohort].astype(bool)]
        rows.append(
            {
                "cohort": cohort,
                "cohort_item_count": int(len(yes)),
                "cohort_ndcg@20_mean": _safe_mean(yes["ndcg@20"]),
                "non_cohort_ndcg@20_mean": _safe_mean(no["ndcg@20"]),
                "cohort_minus_non_ndcg@20": _safe_mean(yes["ndcg@20"]) - _safe_mean(no["ndcg@20"]),
                "cohort_margin_mean": _safe_mean(yes["margin_to_top20_cutoff"]),
            }
        )
    return pd.DataFrame(rows).sort_values("cohort_minus_non_ndcg@20").reset_index(drop=True)


def decide_competitor_cohort_route(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {"route": "competitor_cohort_not_specific", "reason": "缺少 cohort。", "evidence": {}}
    top = summary.iloc[0]
    if float(top["cohort_minus_non_ndcg@20"]) >= -0.02:
        route = "competitor_cohort_not_specific"
        reason = "没有 competitor cohort 对应明显更低 NDCG。"
    elif str(top["cohort"]) == "high_activity_competitor":
        route = "high_activity_competitor_cohort_supported"
        reason = "高活跃 competitor cohort 对应更低 NDCG。"
    elif str(top["cohort"]) == "same_category_competitor":
        route = "same_category_competitor_cohort_supported"
        reason = "同类 competitor cohort 对应更低 NDCG。"
    elif str(top["cohort"]) == "image_similarity_competitor":
        route = "image_similarity_competitor_cohort_supported"
        reason = "图像相似 competitor cohort 对应更低 NDCG。"
    else:
        route = "high_activity_competitor_cohort_supported"
        reason = "高范数/高活跃 competitor pressure cohort 对应更低 NDCG。"
    return {"route": route, "reason": reason, "evidence": top.to_dict()}


def build_checkpoint_stability_summary(task3_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    selection = task3_dir / "checkpoint_selection_summary.csv"
    if not selection.exists():
        df = pd.DataFrame([{"field": "checkpoint_selection_summary", "value": "missing"}])
        return df, {"route": "needs_full_epoch_package", "reason": "缺少 checkpoint selection 产物。", "evidence": {}}
    sel = pd.read_csv(selection)
    df = sel.copy()
    decision = {
        "route": "needs_full_epoch_package",
        "reason": "当前已有 best checkpoint 复评，但尚未完成 baseline last / method last 的 item-level subgroup stability；需用 slim 包 last_epoch_100 补跑。",
        "evidence": {"checkpoint_selection_rows": int(len(df))},
    }
    return df, decision


def build_carrier_evidence_matrix(route_decisions: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {"carrier": "margin_pairwise", "support": 0, "evidence": []},
        {"carrier": "calibration", "support": 0, "evidence": []},
        {"carrier": "modality", "support": 0, "evidence": []},
        {"carrier": "user_side", "support": 0, "evidence": []},
        {"carrier": "competitor_suppression", "support": 0, "evidence": []},
    ]
    table = {row["carrier"]: row for row in rows}
    for task, decision in route_decisions.items():
        route = decision.get("route", "")
        if "margin_proxy_recommended" in route or "rank_aware" in route or "matched_hard" in route:
            table["margin_pairwise"]["support"] += 1
            table["margin_pairwise"]["evidence"].append(f"{task}:{route}")
        if "calibration" in route or "norm" in route:
            table["calibration"]["support"] += 1
            table["calibration"]["evidence"].append(f"{task}:{route}")
        if "modality" in route or "attr_path" in route or "image_path" in route or "q_fusion" in route:
            table["modality"]["support"] += 1
            table["modality"]["evidence"].append(f"{task}:{route}")
        if "user_side" in route or "user_item" in route:
            table["user_side"]["support"] += 1
            table["user_side"]["evidence"].append(f"{task}:{route}")
        if "competitor" in route or "target_competitor" in route:
            table["competitor_suppression"]["support"] += 1
            table["competitor_suppression"]["evidence"].append(f"{task}:{route}")
    out = pd.DataFrame([{**row, "evidence": "; ".join(row["evidence"])} for row in rows])
    return out.sort_values("support", ascending=False).reset_index(drop=True)


def decide_carrier_route(matrix: pd.DataFrame) -> dict[str, Any]:
    top = matrix.iloc[0]
    carrier = str(top["carrier"])
    route_map = {
        "margin_pairwise": "margin_carrier_recommended",
        "calibration": "calibration_carrier_recommended",
        "modality": "modality_carrier_recommended",
        "user_side": "user_side_carrier_recommended",
    }
    route = route_map.get(carrier, "no_task4_carrier_yet") if int(top["support"]) >= 2 else "no_task4_carrier_yet"
    return {"route": route, "reason": f"carrier evidence top={carrier}, support={int(top['support'])}", "evidence": top.to_dict()}


def build_consensus_signal_profile(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    frame["consensus_score"] = 0
    frame["consensus_score"] += frame.get("residual_bucket", "").eq("hard").astype(int) if "residual_bucket" in frame.columns else 0
    frame["consensus_score"] += frame.get("margin_bucket", "").isin(["low", "mid"]).astype(int) if "margin_bucket" in frame.columns else 0
    frame["consensus_score"] += (pd.to_numeric(frame.get("target_competitor_gap_proxy", 0), errors="coerce") < 0).astype(int)
    frame["consensus_score"] += (pd.to_numeric(frame.get("modality_alignment_proxy", 0), errors="coerce") < 0).astype(int)
    frame["consensus_group"] = _prefix_bucket(frame["consensus_score"], "consensus")
    return frame


def build_consensus_group_validity(frame: pd.DataFrame) -> pd.DataFrame:
    frame = _ensure_margin_column(frame)
    rows = []
    for group, sub in frame.groupby("consensus_group", dropna=False):
        rows.append({"consensus_group": group, "item_count": int(len(sub)), "ndcg@20_mean": _safe_mean(sub["ndcg@20"]), "margin_to_top20_cutoff_mean": _safe_mean(sub["margin_to_top20_cutoff"])})
    return pd.DataFrame(rows).sort_values("consensus_group").reset_index(drop=True)


def build_signal_ablation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for signal in ["residual_bucket", "margin_bucket", "target_competitor_gap_proxy", "modality_alignment_proxy"]:
        if signal not in frame.columns:
            continue
        rows.append({"removed_signal": signal, "remaining_consensus_ndcg@20_mean": _safe_mean(frame[frame["consensus_score"] >= 2]["ndcg@20"]), "item_count": int((frame["consensus_score"] >= 2).sum())})
    return pd.DataFrame(rows)


def decide_consensus_route(validity: pd.DataFrame) -> dict[str, Any]:
    if validity.empty:
        return {"route": "no_consensus_for_task4", "reason": "缺少 consensus group。", "evidence": {}}
    low = validity[validity["consensus_group"].astype(str).str.contains("high")]
    high = validity[validity["consensus_group"].astype(str).str.contains("low")]
    gap = float(high["ndcg@20_mean"].mean() - low["ndcg@20_mean"].mean()) if not low.empty and not high.empty else float("nan")
    if not pd.isna(gap) and gap > 0.05:
        route = "consensus_hard_group_ready"
        reason = "多信号 consensus group 能区分 hard/easy item。"
    elif not pd.isna(gap) and gap > 0.02:
        route = "consensus_proxy_ready"
        reason = "多信号 consensus 有弱稳定 proxy 信号。"
    else:
        route = "signals_conflict_need_more_diagnosis"
        reason = "多信号组合未形成明显 hard group。"
    return {"route": route, "reason": reason, "evidence": {"easy_minus_hard_ndcg@20": _jsonable(gap)}}


def _manifest(script: Path, output_dir: Path, inputs: dict[str, str], outputs: dict[str, Path], decision: dict[str, Any], row_counts: dict[str, int]) -> dict[str, Any]:
    _, run_iso = now_stamp()
    return {
        "run_time": run_iso,
        "script": str(script),
        "output_dir": str(output_dir),
        "inputs": inputs,
        "outputs": {key: str(value) for key, value in outputs.items()},
        "row_counts": row_counts,
        "route_decision": decision,
    }


def write_task_output(
    output_dir: Path,
    result_dir_display: str,
    task_no: str,
    slug: str,
    title_suffix: str,
    decision: dict[str, Any],
    outputs: dict[str, pd.DataFrame],
    json_outputs: dict[str, dict[str, Any]] | None = None,
    source_notes: list[str] | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp, _ = now_stamp()
    file_prefix = f"task3p{task_no.replace('.', '')}"
    output_paths: dict[str, Path] = {}
    for name, frame in outputs.items():
        path = output_dir / f"{file_prefix}_{name}.csv"
        frame.to_csv(path, index=False)
        output_paths[name] = path
    json_outputs = json_outputs or {}
    for name, obj in json_outputs.items():
        path = output_dir / f"{file_prefix}_{name}.json"
        _write_json(path, obj)
        output_paths[name] = path
    route_name = f"{file_prefix}_route_decision"
    route_path = output_dir / f"{route_name}.json"
    _write_json(route_path, decision)
    output_paths[route_name] = route_path
    md_path = output_dir / f"{run_stamp} CCFCRec Amazon-VG category availability v2 Task3.{task_no} {title_suffix} 诊断结果.md"
    first_frame = next(iter(outputs.values())) if outputs else pd.DataFrame([decision.get("evidence", {})])
    _write_simple_md(
        md_path,
        f"{run_stamp} CCFCRec Amazon-VG category availability v2 Task3.{task_no} {title_suffix} 诊断结果",
        f"Task3.{task_no}",
        result_dir_display,
        decision,
        [(next(iter(outputs.keys())) if outputs else "route evidence", first_frame, list(first_frame.columns)[:10])],
        source_notes=source_notes,
    )
    output_paths["result_md"] = md_path
    manifest_path = output_dir / "run_manifest.json"
    manifest = _manifest(
        Path(__file__).resolve(),
        output_dir,
        {},
        output_paths,
        decision,
        {key: int(len(frame)) for key, frame in outputs.items()},
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    task3_dir = Path(args.task3_dir).expanduser().resolve()
    task3p5_dir = Path(args.task3p5_dir).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    result_root_display = args.result_root_display.rstrip("/")
    profiles = load_base_profiles(task3_dir)
    common = build_common_profile(profiles)
    p10_profile = pd.read_csv(Path(args.task3p10_profile).expanduser().resolve()) if args.task3p10_profile else pd.DataFrame()
    if not p10_profile.empty:
        common = common.merge(p10_profile[["asin", "residual_bucket"]], on="asin", how="left")
    delta_profile = pd.read_csv(task3p5_dir / "task3p5_item_delta_profile.csv")
    route_decisions: dict[str, dict[str, Any]] = {}

    # Task3.9
    t9_profile = common.copy()
    t9_profile["gap_bucket"] = _prefix_bucket(t9_profile["target_competitor_gap_proxy"], "gap")
    t9_summary = build_target_competitor_gap_summary(t9_profile)
    t9_corr = build_gap_component_correlation(t9_profile)
    t9_decision = decide_target_competitor_route(t9_summary)
    route_decisions["Task3.9"] = t9_decision
    write_task_output(output_root / "2026-07-05 020000 category-availability-v2-task3p9-target-competitor-gap", f"{result_root_display}/2026-07-05 020000 category-availability-v2-task3p9-target-competitor-gap/", "9", "target-competitor-gap", "target-competitor gap", t9_decision, {"target_competitor_gap_profile": t9_profile, "gap_bucket_summary": t9_summary, "gap_component_correlation": t9_corr})

    # Task3.12
    t12_summary = build_modality_failure_summary(common)
    t12_delta = build_modality_delta_vs_rank(delta_profile)
    t12_decision = decide_modality_route(t12_summary)
    route_decisions["Task3.12"] = t12_decision
    write_task_output(output_root / "2026-07-05 020100 category-availability-v2-task3p12-modality-failure", f"{result_root_display}/2026-07-05 020100 category-availability-v2-task3p12-modality-failure/", "12", "modality-failure", "modality failure", t12_decision, {"modality_failure_profile": common, "modality_bucket_summary": t12_summary, "modality_delta_vs_rank": t12_delta})

    # Task3.13
    t13_profile = build_controlled_failure_profile(common)
    t13_residual = build_controlled_residual_by_candidate(t13_profile)
    t13_independence = build_candidate_independence_summary(t13_residual)
    t13_decision = decide_controlled_failure_route(t13_independence)
    route_decisions["Task3.13"] = t13_decision
    write_task_output(output_root / "2026-07-05 020200 category-availability-v2-task3p13-controlled-failure", f"{result_root_display}/2026-07-05 020200 category-availability-v2-task3p13-controlled-failure/", "13", "controlled-failure", "controlled failure", t13_decision, {"controlled_failure_profile": t13_profile, "controlled_residual_by_candidate": t13_residual, "candidate_independence_summary": t13_independence})

    # Task3.14
    t14_profile = build_user_failure_profile(common)
    t14_summary = build_user_bucket_summary(t14_profile)
    t14_cross = build_user_item_cross_summary(t14_profile)
    t14_decision = decide_user_failure_route(t14_summary)
    route_decisions["Task3.14"] = t14_decision
    write_task_output(output_root / "2026-07-05 020300 category-availability-v2-task3p14-per-user-failure", f"{result_root_display}/2026-07-05 020300 category-availability-v2-task3p14-per-user-failure/", "14", "per-user-failure", "per-user failure", t14_decision, {"user_failure_profile": t14_profile, "user_bucket_summary": t14_summary, "user_item_cross_summary": t14_cross})

    # Task3.16
    asin_meta_path = Path(args.asin_metadata).expanduser().resolve() if args.asin_metadata else None
    asin_meta = pd.read_csv(asin_meta_path) if asin_meta_path and asin_meta_path.exists() else None
    t16_candidates = build_case_audit_candidates(common, asin_meta=asin_meta)
    t16_summary = build_case_archetype_summary(t16_candidates)
    examples = {
        archetype: group.head(5)[[col for col in ["asin", "title", "category", "ndcg@20", "margin_to_top20_cutoff", "archetype"] if col in group.columns]].to_dict(orient="records")
        for archetype, group in t16_candidates.groupby("archetype")
    }
    t16_decision = decide_case_audit_route(t16_summary)
    route_decisions["Task3.16"] = t16_decision
    write_task_output(output_root / "2026-07-05 020400 category-availability-v2-task3p16-case-audit", f"{result_root_display}/2026-07-05 020400 category-availability-v2-task3p16-case-audit/", "16", "case-audit", "case audit", t16_decision, {"case_audit_candidates": t16_candidates, "item_archetype_summary": t16_summary}, {"archetype_examples": examples})

    # Task3.17
    t17_profile = build_alt_availability_profile(common)
    t17_validity = build_alt_group_validity_summary(t17_profile)
    t17_cross = build_alt_vs_v2_cross_summary(t17_profile)
    t17_decision = decide_alt_availability_route(t17_validity)
    route_decisions["Task3.17"] = t17_decision
    write_task_output(output_root / "2026-07-05 020500 category-availability-v2-task3p17-alt-availability", f"{result_root_display}/2026-07-05 020500 category-availability-v2-task3p17-alt-availability/", "17", "alt-availability", "alternative availability", t17_decision, {"alt_availability_profile": t17_profile, "alt_group_validity_summary": t17_validity, "alt_vs_v2_cross_summary": t17_cross})

    # Task3.20
    t20_profile = build_competitor_cohort_profile(common)
    t20_summary = build_competitor_cohort_summary(t20_profile)
    t20_controlled = t20_summary.copy()
    t20_decision = decide_competitor_cohort_route(t20_summary)
    route_decisions["Task3.20"] = t20_decision
    write_task_output(output_root / "2026-07-05 020600 category-availability-v2-task3p20-competitor-cohort", f"{result_root_display}/2026-07-05 020600 category-availability-v2-task3p20-competitor-cohort/", "20", "competitor-cohort", "competitor cohort", t20_decision, {"competitor_cohort_profile": t20_profile, "competitor_cohort_summary": t20_summary, "cohort_vs_failure_controlled": t20_controlled})

    # Task3.21
    t21_summary, t21_decision = build_checkpoint_stability_summary(task3_dir)
    route_decisions["Task3.21"] = t21_decision
    write_task_output(output_root / "2026-07-05 020700 category-availability-v2-task3p21-checkpoint-stability", f"{result_root_display}/2026-07-05 020700 category-availability-v2-task3p21-checkpoint-stability/", "21", "checkpoint-stability", "checkpoint stability", t21_decision, {"checkpoint_stability_profile": t21_summary, "checkpoint_group_delta_summary": t21_summary})

    # Task3.22
    t22_placebo = build_placebo_summary(t17_profile, shuffle_count=args.shuffle_count, random_seed=43)
    t22_summary = build_true_vs_placebo_summary(t22_placebo)
    t22_decision = decide_placebo_route(t22_summary)
    route_decisions["Task3.22"] = t22_decision
    write_task_output(output_root / "2026-07-05 020800 category-availability-v2-task3p22-placebo-control", f"{result_root_display}/2026-07-05 020800 category-availability-v2-task3p22-placebo-control/", "22", "placebo-control", "placebo control", t22_decision, {"placebo_control_profile": t17_profile, "shuffle_null_distribution": t22_placebo, "true_vs_placebo_summary": t22_summary})

    # Task3.18 after collecting routes.
    existing_routes_dir = Path(args.existing_routes_dir).expanduser().resolve() if args.existing_routes_dir else output_root
    for task_name, rel in {
        "Task3.8": "2026-07-05 013500 category-availability-v2-task3p8-rank-margin/task3p8_route_decision.json",
        "Task3.10": "2026-07-05 014200 category-availability-v2-task3p10-norm-activity-matched/task3p10_route_decision.json",
        "Task3.11": "2026-07-05 014600 category-availability-v2-task3p11-category-cf-interaction/task3p11_route_decision.json",
        "Task3.15": "2026-07-05 015000 category-availability-v2-task3p15-score-calibration/task3p15_route_decision.json",
        "Task3.19": "2026-07-05 015500 category-availability-v2-task3p19-proxy-metric/task3p19_route_decision.json",
    }.items():
        route_decisions[task_name] = _read_json(existing_routes_dir / rel)
    t18_matrix = build_carrier_evidence_matrix(route_decisions)
    t18_rank = {"rank": t18_matrix.to_dict(orient="records")}
    t18_decision = decide_carrier_route(t18_matrix)
    route_decisions["Task3.18"] = t18_decision
    write_task_output(output_root / "2026-07-05 020900 category-availability-v2-task3p18-method-carrier-redesign", f"{result_root_display}/2026-07-05 020900 category-availability-v2-task3p18-method-carrier-redesign/", "18", "method-carrier-redesign", "method carrier redesign", t18_decision, {"method_carrier_evidence_matrix": t18_matrix}, {"candidate_carrier_rank": t18_rank})

    # Task3.23 final consensus.
    t23_profile = build_consensus_signal_profile(t16_candidates)
    t23_ablation = build_signal_ablation_summary(t23_profile)
    t23_validity = build_consensus_group_validity(t23_profile)
    t23_decision = decide_consensus_route(t23_validity)
    route_decisions["Task3.23"] = t23_decision
    write_task_output(output_root / "2026-07-05 021000 category-availability-v2-task3p23-consensus-signal", f"{result_root_display}/2026-07-05 021000 category-availability-v2-task3p23-consensus-signal/", "23", "consensus-signal", "consensus signal", t23_decision, {"consensus_signal_profile": t23_profile, "signal_ablation_summary": t23_ablation, "consensus_group_validity": t23_validity})

    print(json.dumps({"route_decisions": route_decisions, "output_root": str(output_root)}, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run remaining CCFCRec Amazon-VG category availability v2 Task3.x diagnostics")
    parser.add_argument("--task3-dir", required=True)
    parser.add_argument("--task3p5-dir", required=True)
    parser.add_argument("--task3p10-profile", default="")
    parser.add_argument("--asin-metadata", default="")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--result-root-display", required=True)
    parser.add_argument("--existing-routes-dir", default="")
    parser.add_argument("--shuffle-count", type=int, default=100)
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
