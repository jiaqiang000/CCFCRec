#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.5 机制锁定诊断。

脚本只复用 Task3 已生成的 item-level CSV，不重新跑模型前向，不训练模型。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


SCORE_METRICS = [
    "hr@20",
    "ndcg@20",
    "q_norm",
    "top20_user_norm_mean",
    "target_score_max",
    "margin_to_top20_cutoff",
    "best_target_rank",
    "target_user_norm_mean",
]

TARGET_METRICS = [
    "target_cosine_at_score_max",
    "target_minus_top20_cosine_mean",
    "target_minus_top20_user_norm_mean",
    "target_history_category_overlap_rate_mean",
]

CONTENT_METRICS = [
    "target_history_q_cosine_mean",
    "top20_history_q_cosine_mean",
    "target_minus_top20_history_q_cosine_mean",
    "target_history_attr_cosine_mean",
    "target_history_img_cosine_mean",
    "top20_history_interaction_count_mean",
    "target_minus_top20_history_interaction_count_mean",
]

GROUP_SUMMARY_METRICS = [
    "delta_hr@20",
    "delta_ndcg@20",
    "delta_q_norm",
    "delta_top20_user_norm_mean",
    "delta_target_score_max",
    "delta_margin_to_top20_cutoff",
    "delta_best_target_rank",
    "delta_target_cosine_at_score_max",
    "delta_target_minus_top20_cosine_mean",
    "delta_target_minus_top20_user_norm_mean",
    "delta_target_history_q_cosine_mean",
    "delta_top20_history_q_cosine_mean",
    "delta_target_minus_top20_history_q_cosine_mean",
    "delta_target_history_attr_cosine_mean",
    "delta_target_history_img_cosine_mean",
    "delta_top20_history_interaction_count_mean",
    "norm_activity_pressure_score",
]

CORRELATION_X_METRICS = [
    "delta_top20_user_norm_mean",
    "delta_top20_history_interaction_count_mean",
    "norm_activity_pressure_score",
    "delta_target_history_q_cosine_mean",
    "delta_target_minus_top20_history_q_cosine_mean",
    "delta_target_minus_top20_cosine_mean",
    "delta_margin_to_top20_cutoff",
    "delta_best_target_rank",
]


def now_stamp() -> tuple[str, str]:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def _as_jsonable(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def _safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.mean()) if numeric.notna().any() else float("nan")


def _zscore(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    std = numeric.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (numeric - numeric.mean()) / std


def _available_metrics(*frames: pd.DataFrame, metrics: list[str]) -> list[str]:
    return [metric for metric in metrics if all(metric in frame.columns for frame in frames)]


def _delta_frame(
    baseline: pd.DataFrame,
    category_conf: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    if "asin" not in baseline.columns or "asin" not in category_conf.columns:
        raise ValueError("baseline/category_conf 输入都必须包含 asin 列")
    available = _available_metrics(baseline, category_conf, metrics=metrics)
    merged = baseline[["asin", *available]].merge(
        category_conf[["asin", *available]],
        on="asin",
        how="inner",
        suffixes=("_baseline", "_category_conf"),
    )
    output = pd.DataFrame({"asin": merged["asin"]})
    for metric in available:
        base_col = f"{metric}_baseline"
        conf_col = f"{metric}_category_conf"
        output[f"baseline_{metric}"] = pd.to_numeric(merged[base_col], errors="coerce")
        output[f"category_conf_{metric}"] = pd.to_numeric(merged[conf_col], errors="coerce")
        output[f"delta_{metric}"] = output[f"category_conf_{metric}"] - output[f"baseline_{metric}"]
    return output


def build_item_delta_profile(
    item_profile: pd.DataFrame,
    baseline_score: pd.DataFrame,
    category_score: pd.DataFrame,
    baseline_target: pd.DataFrame,
    category_target: pd.DataFrame,
    baseline_content: pd.DataFrame,
    category_content: pd.DataFrame,
) -> pd.DataFrame:
    if "asin" not in item_profile.columns:
        raise ValueError("item_profile 必须包含 asin 列")
    base_columns = [
        column
        for column in [
            "asin",
            "category_group",
            "category_count",
            "gt_group",
            "gt_user_count",
            "raw_test_user_count",
            "s_cat",
            "s_cat_group",
            "s_cat_v1",
            "s_cat_group_v1",
            "s_cat_v2_disc_within_control",
            "s_cat_v2_collab_within_control",
        ]
        if column in item_profile.columns
    ]
    profile = item_profile[base_columns].drop_duplicates("asin").copy()

    for delta in [
        _delta_frame(baseline_score, category_score, SCORE_METRICS),
        _delta_frame(baseline_target, category_target, TARGET_METRICS),
        _delta_frame(baseline_content, category_content, CONTENT_METRICS),
    ]:
        profile = profile.merge(delta, on="asin", how="left")

    norm_pressure = _zscore(profile.get("delta_top20_user_norm_mean", pd.Series(index=profile.index, dtype=float)))
    activity_pressure = _zscore(
        profile.get("delta_top20_history_interaction_count_mean", pd.Series(index=profile.index, dtype=float))
    )
    profile["norm_activity_pressure_score"] = norm_pressure + activity_pressure
    return profile.sort_values("asin").reset_index(drop=True)


def build_group_mechanism_summary(profile: pd.DataFrame) -> pd.DataFrame:
    if "category_group" not in profile.columns:
        raise ValueError("profile 必须包含 category_group 列")
    rows: list[dict[str, Any]] = []
    for group, frame in profile.groupby("category_group", dropna=False):
        row: dict[str, Any] = {
            "category_group": group,
            "item_count": int(len(frame)),
            "ndcg_improved_rate": float((pd.to_numeric(frame["delta_ndcg@20"], errors="coerce") > 0).mean()),
            "ndcg_declined_rate": float((pd.to_numeric(frame["delta_ndcg@20"], errors="coerce") < 0).mean()),
        }
        for metric in GROUP_SUMMARY_METRICS:
            if metric in frame.columns:
                row[f"{metric}_mean"] = _safe_mean(frame[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("category_group").reset_index(drop=True)


def _quantile_labels(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    labels = pd.Series("unknown", index=series.index, dtype="object")
    valid = numeric.dropna().sort_values(kind="mergesort")
    if valid.empty:
        return labels
    if len(valid) == 1 or valid.nunique() == 1:
        labels.loc[valid.index] = "all"
        return labels
    label_names = ["low", "mid", "high"]
    ordered_positions = {index: pos for pos, index in enumerate(valid.index)}
    denom = max(len(valid) - 1, 1)
    for index, pos in ordered_positions.items():
        bucket_index = round((pos / denom) * 2)
        labels.loc[index] = label_names[int(bucket_index)]
    return labels


def build_weak_bucket_summary(profile: pd.DataFrame, value_col: str, bucket_col: str) -> pd.DataFrame:
    if value_col not in profile.columns:
        raise ValueError(f"profile 缺少分桶列: {value_col}")
    weak = profile[profile["category_group"].astype(str).str.contains("weak", na=False)].copy()
    if weak.empty:
        return pd.DataFrame(
            columns=[
                bucket_col,
                "item_count",
                "delta_hr@20_mean",
                "delta_ndcg@20_mean",
                "ndcg_improved_rate",
                "ndcg_declined_rate",
                "delta_top20_user_norm_mean_mean",
                "delta_top20_history_interaction_count_mean_mean",
                "delta_target_minus_top20_history_q_cosine_mean_mean",
                "delta_margin_to_top20_cutoff_mean",
            ]
        )
    weak[bucket_col] = _quantile_labels(weak[value_col])
    rows: list[dict[str, Any]] = []
    for bucket, frame in weak.groupby(bucket_col, dropna=False):
        row = {
            bucket_col: bucket,
            "item_count": int(len(frame)),
            "delta_hr@20_mean": _safe_mean(frame["delta_hr@20"]),
            "delta_ndcg@20_mean": _safe_mean(frame["delta_ndcg@20"]),
            "ndcg_improved_rate": float((pd.to_numeric(frame["delta_ndcg@20"], errors="coerce") > 0).mean()),
            "ndcg_declined_rate": float((pd.to_numeric(frame["delta_ndcg@20"], errors="coerce") < 0).mean()),
        }
        for metric in [
            "delta_top20_user_norm_mean",
            "delta_top20_history_interaction_count_mean",
            "delta_target_minus_top20_history_q_cosine_mean",
            "delta_margin_to_top20_cutoff",
        ]:
            if metric in frame.columns:
                row[f"{metric}_mean"] = _safe_mean(frame[metric])
        rows.append(row)
    order = {"low": 0, "mid": 1, "high": 2, "all": 3, "unknown": 4}
    return (
        pd.DataFrame(rows)
        .assign(_order=lambda df: df[bucket_col].map(order).fillna(99))
        .sort_values("_order")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )


def build_correlation_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("overall", profile)]
    scopes.extend((str(group), frame) for group, frame in profile.groupby("category_group", dropna=False))
    for scope, frame in scopes:
        y = pd.to_numeric(frame.get("delta_ndcg@20"), errors="coerce")
        for metric in CORRELATION_X_METRICS:
            if metric not in frame.columns:
                continue
            x = pd.to_numeric(frame[metric], errors="coerce")
            valid = pd.concat([x, y], axis=1).dropna()
            if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
                pearson = float("nan")
                spearman = float("nan")
            else:
                pearson = float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="pearson"))
                spearman = float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman"))
            rows.append(
                {
                    "scope": scope,
                    "x_metric": metric,
                    "y_metric": "delta_ndcg@20",
                    "n": int(len(valid)),
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    return pd.DataFrame(rows)


def build_route_decision(
    profile: pd.DataFrame,
    group_summary: pd.DataFrame,
    extra_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    extra_evidence = extra_evidence or {}
    weak_rows = group_summary[group_summary["category_group"].astype(str).str.contains("weak", na=False)]
    if weak_rows.empty:
        return {
            "route": "inconclusive_needs_item_case_audit",
            "reason": "未找到 v2 weak 组，无法解释 weak 组失败机制。",
            "evidence": {},
        }
    weak = weak_rows.iloc[0]
    weak_delta_ndcg = float(weak.get("delta_ndcg@20_mean", float("nan")))
    weak_delta_hr = float(weak.get("delta_hr@20_mean", float("nan")))
    weak_norm = float(weak.get("delta_top20_user_norm_mean_mean", float("nan")))
    weak_activity = float(weak.get("delta_top20_history_interaction_count_mean_mean", float("nan")))
    weak_alignment = float(weak.get("delta_target_history_q_cosine_mean_mean", float("nan")))
    weak_gap = float(weak.get("delta_target_minus_top20_history_q_cosine_mean_mean", float("nan")))
    weak_margin = float(weak.get("delta_margin_to_top20_cutoff_mean", float("nan")))
    pressure_gap = extra_evidence.get("weak_pressure_high_minus_low_ndcg")

    route = "inconclusive_needs_item_case_audit"
    reason = "聚合证据不足以锁定单一失败机制，需要 item case audit。"
    if (
        weak_delta_ndcg <= 0
        and weak_norm > 0
        and weak_activity > 0
        and pressure_gap is not None
        and pressure_gap < 0
    ):
        route = "norm_activity_bias_supported"
        reason = "weak 组 NDCG 不增，同时 top20 高范数/高活跃压力上升，且高压力桶比低压力桶的 delta NDCG 更差。"
    elif weak_delta_ndcg <= 0 and weak_alignment > 0:
        route = "alignment_not_ranking_signal"
        reason = "weak 组 alignment 有提升，但 HR/NDCG 未提升，说明当前 alignment 指标没有稳定转化成排序收益。"

    evidence = {
        "weak_delta_ndcg@20_mean": weak_delta_ndcg,
        "weak_delta_hr@20_mean": weak_delta_hr,
        "weak_delta_top20_user_norm_mean_mean": weak_norm,
        "weak_delta_top20_history_interaction_count_mean_mean": weak_activity,
        "weak_delta_target_history_q_cosine_mean_mean": weak_alignment,
        "weak_delta_target_minus_top20_history_q_cosine_mean_mean": weak_gap,
        "weak_delta_margin_to_top20_cutoff_mean": weak_margin,
        "weak_pressure_high_minus_low_ndcg": pressure_gap,
    }
    evidence.update(extra_evidence)
    return {"route": route, "reason": reason, "evidence": {k: _as_jsonable(v) for k, v in evidence.items()}}


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20) -> str:
    available = [column for column in columns if column in df.columns]
    if not available:
        return "\n"
    display = df[available].head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_numeric_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}".rstrip("0").rstrip("."))
        else:
            display[column] = display[column].fillna("").astype(str)
    header = "| " + " | ".join(display.columns) + " |"
    separator = "| " + " | ".join("---" for _ in display.columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |"
        for row in display.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _pressure_high_minus_low(bucket: pd.DataFrame, bucket_col: str) -> float | None:
    if bucket.empty or "delta_ndcg@20_mean" not in bucket.columns:
        return None
    values = bucket.set_index(bucket_col)["delta_ndcg@20_mean"].to_dict()
    if "high" in values and "low" in values:
        return float(values["high"] - values["low"])
    return None


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    source_notes: list[str],
    result_dir_display: str,
    group_summary: pd.DataFrame,
    norm_bucket: pd.DataFrame,
    alignment_bucket: pd.DataFrame,
    margin_bucket: pd.DataFrame,
    correlation_summary: pd.DataFrame,
    route_decision: dict[str, Any],
) -> None:
    source_lines = "\n".join(f"> {note}" for note in source_notes) or "> 上游设计：未提供"
    weak_corr = correlation_summary[correlation_summary["scope"].astype(str).str.contains("weak", na=False)].copy()
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.5 机制锁定诊断结果
date: {run_stamp[:10]}
time: "{run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
created_at: "{run_stamp[:10]} {run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - Task3.5
  - 机制诊断
---

# {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.5 机制锁定诊断结果

## 来源说明

> [!info] 来源说明
{source_lines}
> 本结果目录：`{result_dir_display}`

## 结论

Task3.5 的 route decision 为 `{route_decision["route"]}`。

{route_decision["reason"]}

本结果只解释 Task3 中 category_conf_input 相对 baseline 的失败机制，不判断 v2 分组是否适合作为 Task4 分层，也不判断 category_conf_input 是否是合适方法载体。

## 数据验收

> [!info] 字段说明
> `field`：验收项。
> `value`：验收项取值。

| field | value |
|---|---:|
| group_count | {len(group_summary)} |
| route | `{route_decision["route"]}` |
| weak_pressure_high_minus_low_ndcg | {route_decision["evidence"].get("weak_pressure_high_minus_low_ndcg")} |

## v2 分组机制汇总

> [!info] 字段说明
> `category_group`：v2 `s_cat_group`。
> `item_count`：该组 test item 数。
> `delta_ndcg@20_mean`：category_conf_input 减 baseline 的 item-level NDCG@20 均值。
> `delta_hr@20_mean`：category_conf_input 减 baseline 的 item-level HR@20 均值。
> `delta_top20_user_norm_mean_mean`：top20 用户范数均值变化。
> `delta_top20_history_interaction_count_mean_mean`：top20 用户历史交互数均值变化。
> `delta_target_history_q_cosine_mean_mean`：目标用户历史中心 q-cosine 变化。
> `delta_target_minus_top20_history_q_cosine_mean_mean`：目标用户与 top20 用户历史中心 q-cosine gap 变化。
> `delta_margin_to_top20_cutoff_mean`：目标用户最高分相对 top20 cutoff 的 margin 变化。

{md_table(group_summary, ["category_group", "item_count", "delta_ndcg@20_mean", "delta_hr@20_mean", "delta_top20_user_norm_mean_mean", "delta_top20_history_interaction_count_mean_mean", "delta_target_history_q_cosine_mean_mean", "delta_target_minus_top20_history_q_cosine_mean_mean", "delta_margin_to_top20_cutoff_mean"])}

## weak 组 norm/activity 分桶

> [!info] 字段说明
> `norm_activity_bucket`：weak 组内按 norm/activity pressure score 分桶。
> `item_count`：该桶 item 数。
> `delta_ndcg@20_mean`：该桶 NDCG@20 delta 均值。
> `ndcg_improved_rate`：该桶 delta NDCG@20 大于 0 的 item 比例。
> `ndcg_declined_rate`：该桶 delta NDCG@20 小于 0 的 item 比例。
> `delta_top20_user_norm_mean_mean`：top20 用户范数均值变化。
> `delta_top20_history_interaction_count_mean_mean`：top20 用户历史交互数均值变化。

{md_table(norm_bucket, ["norm_activity_bucket", "item_count", "delta_ndcg@20_mean", "delta_hr@20_mean", "ndcg_improved_rate", "ndcg_declined_rate", "delta_top20_user_norm_mean_mean", "delta_top20_history_interaction_count_mean_mean"])}

## weak 组 alignment gap 分桶

> [!info] 字段说明
> `alignment_gap_bucket`：weak 组内按 `delta_target_minus_top20_history_q_cosine_mean` 分桶。
> `item_count`：该桶 item 数。
> `delta_ndcg@20_mean`：该桶 NDCG@20 delta 均值。
> `delta_target_minus_top20_history_q_cosine_mean_mean`：目标用户与 top20 用户历史中心 q-cosine gap 变化。

{md_table(alignment_bucket, ["alignment_gap_bucket", "item_count", "delta_ndcg@20_mean", "delta_hr@20_mean", "ndcg_improved_rate", "ndcg_declined_rate", "delta_target_minus_top20_history_q_cosine_mean_mean"])}

## weak 组 margin 分桶

> [!info] 字段说明
> `margin_bucket`：weak 组内按 `delta_margin_to_top20_cutoff` 分桶。
> `item_count`：该桶 item 数。
> `delta_ndcg@20_mean`：该桶 NDCG@20 delta 均值。
> `delta_margin_to_top20_cutoff_mean`：目标用户最高分相对 top20 cutoff 的 margin 变化。

{md_table(margin_bucket, ["margin_bucket", "item_count", "delta_ndcg@20_mean", "delta_hr@20_mean", "ndcg_improved_rate", "ndcg_declined_rate", "delta_margin_to_top20_cutoff_mean"])}

## weak 组相关性

> [!info] 字段说明
> `scope`：统计范围。
> `x_metric`：解释变量。
> `y_metric`：被解释排序收益变量。
> `n`：有效样本数。
> `pearson`：Pearson 相关。
> `spearman`：Spearman 相关。

{md_table(weak_corr, ["scope", "x_metric", "y_metric", "n", "pearson", "spearman"], max_rows=30)}

## 判断

```json
{json.dumps(route_decision, ensure_ascii=False, indent=2)}
```

## 下一步

继续进入 Task3.6，检查 v2 `s_cat_group` 本身是否真的和推荐失败模式相关；Task3.5 不能替代这个判断。

## 产物

```text
task3p5_item_delta_profile.csv
task3p5_group_mechanism_summary.csv
task3p5_weak_bucket_by_norm_activity.csv
task3p5_weak_bucket_by_alignment_gap.csv
task3p5_weak_bucket_by_margin.csv
task3p5_correlation_summary.csv
task3p5_route_decision.json
run_manifest.json
```
"""
    output_path.write_text(markdown, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    task3_dir = Path(args.task3_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp, run_iso = now_stamp()

    item_profile = pd.read_csv(task3_dir / "task3_item_profile.csv")
    baseline_score = pd.read_csv(task3_dir / "baseline_score_margin" / "item_score_margin_profile.csv")
    category_score = pd.read_csv(task3_dir / "category_conf_input_score_margin" / "item_score_margin_profile.csv")
    baseline_target = pd.read_csv(task3_dir / "baseline_target_score_source" / "target_alignment_profile.csv")
    category_target = pd.read_csv(task3_dir / "category_conf_input_target_score_source" / "target_alignment_profile.csv")
    baseline_content = pd.read_csv(task3_dir / "baseline_content_cf_alignment" / "content_cf_alignment_profile.csv")
    category_content = pd.read_csv(task3_dir / "category_conf_input_content_cf_alignment" / "content_cf_alignment_profile.csv")

    profile = build_item_delta_profile(
        item_profile=item_profile,
        baseline_score=baseline_score,
        category_score=category_score,
        baseline_target=baseline_target,
        category_target=category_target,
        baseline_content=baseline_content,
        category_content=category_content,
    )
    group_summary = build_group_mechanism_summary(profile)
    norm_bucket = build_weak_bucket_summary(profile, "norm_activity_pressure_score", "norm_activity_bucket")
    alignment_bucket = build_weak_bucket_summary(
        profile,
        "delta_target_minus_top20_history_q_cosine_mean",
        "alignment_gap_bucket",
    )
    margin_bucket = build_weak_bucket_summary(profile, "delta_margin_to_top20_cutoff", "margin_bucket")
    correlation_summary = build_correlation_summary(profile)
    pressure_gap = _pressure_high_minus_low(norm_bucket, "norm_activity_bucket")
    route_decision = build_route_decision(
        profile,
        group_summary,
        {"weak_pressure_high_minus_low_ndcg": pressure_gap},
    )

    outputs = {
        "task3p5_item_delta_profile": output_dir / "task3p5_item_delta_profile.csv",
        "task3p5_group_mechanism_summary": output_dir / "task3p5_group_mechanism_summary.csv",
        "task3p5_weak_bucket_by_norm_activity": output_dir / "task3p5_weak_bucket_by_norm_activity.csv",
        "task3p5_weak_bucket_by_alignment_gap": output_dir / "task3p5_weak_bucket_by_alignment_gap.csv",
        "task3p5_weak_bucket_by_margin": output_dir / "task3p5_weak_bucket_by_margin.csv",
        "task3p5_correlation_summary": output_dir / "task3p5_correlation_summary.csv",
        "task3p5_route_decision": output_dir / "task3p5_route_decision.json",
    }
    profile.to_csv(outputs["task3p5_item_delta_profile"], index=False)
    group_summary.to_csv(outputs["task3p5_group_mechanism_summary"], index=False)
    norm_bucket.to_csv(outputs["task3p5_weak_bucket_by_norm_activity"], index=False)
    alignment_bucket.to_csv(outputs["task3p5_weak_bucket_by_alignment_gap"], index=False)
    margin_bucket.to_csv(outputs["task3p5_weak_bucket_by_margin"], index=False)
    correlation_summary.to_csv(outputs["task3p5_correlation_summary"], index=False)
    outputs["task3p5_route_decision"].write_text(
        json.dumps(route_decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result_md = output_dir / f"{run_stamp} CCFCRec Amazon-VG category availability v2 Task3.5 机制锁定诊断结果.md"
    manifest_path = output_dir / "run_manifest.json"
    result_dir_display = args.result_dir_display or str(output_dir)
    write_result_markdown(
        output_path=result_md,
        run_stamp=run_stamp,
        source_notes=args.source_note,
        result_dir_display=result_dir_display,
        group_summary=group_summary,
        norm_bucket=norm_bucket,
        alignment_bucket=alignment_bucket,
        margin_bucket=margin_bucket,
        correlation_summary=correlation_summary,
        route_decision=route_decision,
    )

    manifest = {
        "run_time": run_iso,
        "script": str(Path(__file__).resolve()),
        "task3_dir": str(task3_dir),
        "output_dir": str(output_dir),
        "inputs": {
            "task3_item_profile": str(task3_dir / "task3_item_profile.csv"),
            "baseline_score_margin": str(task3_dir / "baseline_score_margin" / "item_score_margin_profile.csv"),
            "category_conf_input_score_margin": str(task3_dir / "category_conf_input_score_margin" / "item_score_margin_profile.csv"),
            "baseline_target_score_source": str(task3_dir / "baseline_target_score_source" / "target_alignment_profile.csv"),
            "category_conf_input_target_score_source": str(task3_dir / "category_conf_input_target_score_source" / "target_alignment_profile.csv"),
            "baseline_content_cf_alignment": str(task3_dir / "baseline_content_cf_alignment" / "content_cf_alignment_profile.csv"),
            "category_conf_input_content_cf_alignment": str(task3_dir / "category_conf_input_content_cf_alignment" / "content_cf_alignment_profile.csv"),
        },
        "outputs": {key: str(value) for key, value in outputs.items()} | {
            "result_md": str(result_md),
            "run_manifest": str(manifest_path),
        },
        "row_counts": {
            "task3p5_item_delta_profile": int(len(profile)),
            "task3p5_group_mechanism_summary": int(len(group_summary)),
            "task3p5_weak_bucket_by_norm_activity": int(len(norm_bucket)),
            "task3p5_weak_bucket_by_alignment_gap": int(len(alignment_bucket)),
            "task3p5_weak_bucket_by_margin": int(len(margin_bucket)),
            "task3p5_correlation_summary": int(len(correlation_summary)),
        },
        "route_decision": route_decision,
        "notes": [
            "只复用 Task3 已生成 CSV，不训练模型，不重新跑 checkpoint 前向。",
            "Task3.5 只解释 alignment/cosine 改善未转化为 weak 组排序收益的机制。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG category availability v2 Task3.5 机制锁定诊断")
    parser.add_argument("--task3-dir", required=True, type=str, help="Task3 checkpoint 复评输出目录")
    parser.add_argument("--output-dir", required=True, type=str, help="Task3.5 输出目录")
    parser.add_argument("--source-note", action="append", default=[], help="写入结果 MD 来源说明的 Obsidian wikilink 行")
    parser.add_argument("--result-dir-display", type=str, default="", help="写入结果 MD 的相对结果目录")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
