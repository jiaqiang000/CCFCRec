#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.8 rank-margin 诊断。

脚本只复用 Task3 已生成的 baseline item-level CSV，不训练模型，不重新跑 checkpoint 前向。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_amazon_vg_category_availability_task3p5 import md_table


BASE_ITEM_COLUMNS = [
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

SCORE_COLUMNS = [
    "asin",
    "hr@20",
    "ndcg@20",
    "q_norm",
    "top1_score",
    "score_at_20",
    "score_at_21",
    "top20_score_mean",
    "top20_score_std",
    "local_gap_20_21",
    "top20_user_norm_mean",
    "target_score_max",
    "target_score_mean",
    "margin_to_top1",
    "margin_to_top20_cutoff",
    "best_target_rank",
    "best_target_rank_percentile",
    "target_hit_count_at20",
    "target_user_norm_mean",
]

TARGET_COLUMNS = [
    "asin",
    "top20_cosine_mean",
    "top20_cosine_max",
    "target_cosine_at_score_max",
    "target_cosine_max",
    "target_cosine_mean",
    "target_user_norm_at_score_max",
    "target_minus_top20_cosine_mean",
    "target_minus_top20_user_norm_mean",
    "target_history_category_overlap_rate_mean",
    "target_minus_top20_history_overlap_rate_mean",
]

CONTENT_COLUMNS = [
    "asin",
    "target_history_interaction_count_mean",
    "top20_history_interaction_count_mean",
    "target_minus_top20_history_interaction_count_mean",
    "target_history_q_cosine_mean",
    "top20_history_q_cosine_mean",
    "target_minus_top20_history_q_cosine_mean",
    "target_history_attr_cosine_mean",
    "top20_history_attr_cosine_mean",
    "target_minus_top20_history_attr_cosine_mean",
    "target_history_img_cosine_mean",
    "top20_history_img_cosine_mean",
    "target_minus_top20_history_img_cosine_mean",
]

CONTROL_SETS = [
    ("none", []),
    ("gt_group", ["gt_group"]),
    ("target_activity_bucket", ["target_activity_bucket"]),
    ("gt_group+target_activity_bucket", ["gt_group", "target_activity_bucket"]),
]


def now_stamp() -> tuple[str, str]:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
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
    denom = max(len(valid) - 1, 1)
    for pos, index in enumerate(valid.index):
        bucket_index = round((pos / denom) * 2)
        labels.loc[index] = label_names[int(bucket_index)]
    return labels


def _select_existing(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    if "asin" not in available:
        raise ValueError("输入 CSV 必须包含 asin 列")
    return frame[available].drop_duplicates("asin").copy()


def build_rank_margin_profile(
    item_profile: pd.DataFrame,
    baseline_score: pd.DataFrame,
    baseline_target: pd.DataFrame,
    baseline_content: pd.DataFrame,
) -> pd.DataFrame:
    if "asin" not in item_profile.columns:
        raise ValueError("item_profile 必须包含 asin 列")
    base_columns = [column for column in BASE_ITEM_COLUMNS if column in item_profile.columns]
    profile = item_profile[base_columns].drop_duplicates("asin").copy()
    profile = profile.merge(_select_existing(baseline_score, SCORE_COLUMNS), on="asin", how="left")
    profile = profile.merge(_select_existing(baseline_target, TARGET_COLUMNS), on="asin", how="left")
    profile = profile.merge(_select_existing(baseline_content, CONTENT_COLUMNS), on="asin", how="left")

    for column in [
        "hr@20",
        "ndcg@20",
        "margin_to_top20_cutoff",
        "best_target_rank",
        "best_target_rank_percentile",
        "top20_user_norm_mean",
        "top20_history_interaction_count_mean",
        "top20_cosine_mean",
        "target_cosine_at_score_max",
        "target_minus_top20_cosine_mean",
        "target_minus_top20_user_norm_mean",
        "target_minus_top20_history_q_cosine_mean",
        "target_history_interaction_count_mean",
    ]:
        if column in profile.columns:
            profile[column] = pd.to_numeric(profile[column], errors="coerce")

    for optional_column in ["margin_to_top20_cutoff", "target_history_interaction_count_mean", "best_target_rank"]:
        if optional_column not in profile.columns:
            profile[optional_column] = pd.NA
    profile["margin_bucket"] = _quantile_labels(profile["margin_to_top20_cutoff"])
    profile["target_activity_bucket"] = _quantile_labels(profile["target_history_interaction_count_mean"])
    rank = pd.to_numeric(profile["best_target_rank"], errors="coerce")
    profile["target_rank_near_cutoff"] = (rank > 20) & (rank <= 200)

    pressure = pd.Series(0.0, index=profile.index)
    positive_pressure = [
        "top20_user_norm_mean",
        "top20_history_interaction_count_mean",
        "top20_cosine_mean",
    ]
    negative_gap = [
        "target_minus_top20_cosine_mean",
        "target_minus_top20_user_norm_mean",
        "target_minus_top20_history_q_cosine_mean",
    ]
    for column in positive_pressure:
        if column in profile.columns:
            pressure = pressure + _zscore(profile[column])
    for column in negative_gap:
        if column in profile.columns:
            pressure = pressure - _zscore(profile[column])
    profile["hard_negative_pressure_score"] = pressure
    return profile.sort_values("asin").reset_index(drop=True)


def build_margin_bucket_summary(profile: pd.DataFrame) -> pd.DataFrame:
    if "margin_bucket" not in profile.columns:
        raise ValueError("profile 必须包含 margin_bucket 列")
    rows: list[dict[str, Any]] = []
    for bucket, frame in profile.groupby("margin_bucket", dropna=False):
        row: dict[str, Any] = {
            "margin_bucket": bucket,
            "item_count": int(len(frame)),
            "hr@20_mean": _safe_mean(frame["hr@20"]),
            "ndcg@20_mean": _safe_mean(frame["ndcg@20"]),
            "target_rank_near_cutoff_rate": float(frame["target_rank_near_cutoff"].mean()),
        }
        for metric in [
            "margin_to_top20_cutoff",
            "best_target_rank",
            "best_target_rank_percentile",
            "hard_negative_pressure_score",
            "top20_user_norm_mean",
            "top20_history_interaction_count_mean",
            "top20_cosine_mean",
            "target_cosine_at_score_max",
            "target_minus_top20_cosine_mean",
            "target_minus_top20_user_norm_mean",
            "target_minus_top20_history_q_cosine_mean",
        ]:
            if metric in frame.columns:
                row[f"{metric}_mean"] = _safe_mean(frame[metric])
        rows.append(row)
    order = {"low": 0, "mid": 1, "high": 2, "all": 3, "unknown": 4}
    return (
        pd.DataFrame(rows)
        .assign(_order=lambda df: df["margin_bucket"].map(order).fillna(99))
        .sort_values("_order")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )


def build_hard_negative_signature(profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket, frame in profile.groupby("margin_bucket", dropna=False):
        near = frame[frame["target_rank_near_cutoff"]].copy()
        rows.append(
            {
                "margin_bucket": bucket,
                "item_count": int(len(frame)),
                "near_cutoff_item_count": int(len(near)),
                "near_cutoff_rate": float(len(near) / len(frame)) if len(frame) else float("nan"),
                "ndcg@20_mean": _safe_mean(frame["ndcg@20"]),
                "near_cutoff_ndcg@20_mean": _safe_mean(near["ndcg@20"]) if not near.empty else float("nan"),
                "hard_negative_pressure_score_mean": _safe_mean(frame["hard_negative_pressure_score"]),
                "near_cutoff_hard_negative_pressure_score_mean": _safe_mean(near["hard_negative_pressure_score"])
                if not near.empty
                else float("nan"),
                "top20_user_norm_mean": _safe_mean(frame["top20_user_norm_mean"])
                if "top20_user_norm_mean" in frame.columns
                else float("nan"),
                "top20_history_interaction_count_mean": _safe_mean(frame["top20_history_interaction_count_mean"])
                if "top20_history_interaction_count_mean" in frame.columns
                else float("nan"),
                "target_minus_top20_cosine_mean": _safe_mean(frame["target_minus_top20_cosine_mean"])
                if "target_minus_top20_cosine_mean" in frame.columns
                else float("nan"),
            }
        )
    order = {"low": 0, "mid": 1, "high": 2, "all": 3, "unknown": 4}
    return (
        pd.DataFrame(rows)
        .assign(_order=lambda df: df["margin_bucket"].map(order).fillna(99))
        .sort_values("_order")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )


def build_controlled_margin_gap(profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for control_name, control_cols in CONTROL_SETS:
        missing = [column for column in control_cols if column not in profile.columns]
        if missing:
            continue
        frame = profile.copy()
        for metric in ["ndcg@20", "hr@20"]:
            if control_cols:
                frame[f"{metric}_control_mean"] = frame.groupby(control_cols, dropna=False)[metric].transform("mean")
            else:
                frame[f"{metric}_control_mean"] = pd.to_numeric(frame[metric], errors="coerce").mean()
            frame[f"{metric}_control_residual"] = frame[metric] - frame[f"{metric}_control_mean"]
        for bucket, bucket_frame in frame.groupby("margin_bucket", dropna=False):
            rows.append(
                {
                    "control_set": control_name,
                    "margin_bucket": bucket,
                    "item_count": int(len(bucket_frame)),
                    "ndcg@20_mean": _safe_mean(bucket_frame["ndcg@20"]),
                    "ndcg@20_control_residual_mean": _safe_mean(bucket_frame["ndcg@20_control_residual"]),
                    "hr@20_mean": _safe_mean(bucket_frame["hr@20"]),
                    "hr@20_control_residual_mean": _safe_mean(bucket_frame["hr@20_control_residual"]),
                }
            )
    order = {"low": 0, "mid": 1, "high": 2, "all": 3, "unknown": 4}
    return (
        pd.DataFrame(rows)
        .assign(_order=lambda df: df["margin_bucket"].map(order).fillna(99))
        .sort_values(["control_set", "_order"])
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )


def _bucket_value(frame: pd.DataFrame, bucket: str, column: str) -> float:
    rows = frame[frame["margin_bucket"].eq(bucket)]
    if rows.empty or column not in rows.columns:
        return float("nan")
    return float(rows.iloc[0][column])


def build_route_decision(bucket_summary: pd.DataFrame, controlled_gap: pd.DataFrame) -> dict[str, Any]:
    low_ndcg = _bucket_value(bucket_summary, "low", "ndcg@20_mean")
    high_ndcg = _bucket_value(bucket_summary, "high", "ndcg@20_mean")
    low_pressure = _bucket_value(bucket_summary, "low", "hard_negative_pressure_score_mean")
    high_pressure = _bucket_value(bucket_summary, "high", "hard_negative_pressure_score_mean")
    low_near = _bucket_value(bucket_summary, "low", "target_rank_near_cutoff_rate")

    controlled = controlled_gap[controlled_gap["control_set"].eq("gt_group+target_activity_bucket")]
    low_residual = _bucket_value(controlled, "low", "ndcg@20_control_residual_mean")
    high_residual = _bucket_value(controlled, "high", "ndcg@20_control_residual_mean")

    raw_gap = low_ndcg - high_ndcg
    controlled_gap_value = low_residual - high_residual
    pressure_gap = low_pressure - high_pressure

    if raw_gap < -0.02 and controlled_gap_value < -0.005 and pressure_gap > 0 and low_near >= 0.2:
        route = "rank_margin_hard_negative_supported"
        reason = "低 margin 组 baseline NDCG 明显更低，控制 gt_group 与 target activity 后仍为负，且 hard negative pressure 更高。"
    elif raw_gap < -0.02 and controlled_gap_value >= -0.003:
        route = "margin_signal_confounded_by_gt_count"
        reason = "低 margin 组原始 NDCG 更低，但控制 gt_group 与 target activity 后 gap 明显减弱或反转。"
    elif raw_gap >= -0.005:
        route = "margin_not_failure_driver"
        reason = "低 margin 组没有表现出稳定更低的 baseline NDCG。"
    else:
        route = "inconclusive_needs_case_audit"
        reason = "margin 与 failure 有一定关系，但 hard negative 或控制后证据不足，需要 case audit。"

    evidence = {
        "low_ndcg@20_mean": low_ndcg,
        "high_ndcg@20_mean": high_ndcg,
        "low_minus_high_ndcg@20": raw_gap,
        "controlled_low_minus_high_residual_ndcg@20": controlled_gap_value,
        "low_hard_negative_pressure_score_mean": low_pressure,
        "high_hard_negative_pressure_score_mean": high_pressure,
        "low_minus_high_hard_negative_pressure": pressure_gap,
        "low_target_rank_near_cutoff_rate": low_near,
    }
    return {"route": route, "reason": reason, "evidence": {key: _jsonable(value) for key, value in evidence.items()}}


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    source_notes: list[str],
    result_dir_display: str,
    bucket_summary: pd.DataFrame,
    hard_negative_signature: pd.DataFrame,
    controlled_gap: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    source_lines = "\n".join(f"> {note}" for note in source_notes) or "> 上游设计：未提供"
    controlled_focus = controlled_gap[
        controlled_gap["control_set"].isin(["none", "gt_group+target_activity_bucket"])
    ].copy()
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.8 rank-margin 诊断结果
date: {run_stamp[:10]}
time: "{run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
created_at: "{run_stamp[:10]} {run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - Task3.8
  - rank_margin
---

# {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.8 rank-margin 诊断结果

## 来源说明

> [!info] 来源说明
{source_lines}
> 本结果目录：`{result_dir_display}`

## 结论

Task3.8 的 route decision 为 `{decision["route"]}`。

{decision["reason"]}

本结果只判断 baseline 失败是否可由 rank margin / hard negative pressure 解释，不训练模型，也不直接进入 Task4。

## 数据验收

> [!info] 字段说明
> `field`：验收项。
> `value`：验收项取值。

| field | value |
|---|---:|
| margin_bucket_count | {len(bucket_summary)} |
| route | `{decision["route"]}` |
| low_minus_high_ndcg@20 | {decision["evidence"].get("low_minus_high_ndcg@20")} |
| controlled_low_minus_high_residual_ndcg@20 | {decision["evidence"].get("controlled_low_minus_high_residual_ndcg@20")} |

## margin bucket summary

> [!info] 字段说明
> `margin_bucket`：按 baseline `margin_to_top20_cutoff` 分桶。
> `item_count`：该桶 item 数。
> `ndcg@20_mean`：baseline item-level NDCG@20 均值。
> `target_rank_near_cutoff_rate`：目标用户最佳排名在 21 到 200 的 item 比例。
> `hard_negative_pressure_score_mean`：top20 高范数、高活跃、高 cosine 与 target-vs-top20 gap 合成压力均值。

{md_table(bucket_summary, ["margin_bucket", "item_count", "ndcg@20_mean", "hr@20_mean", "margin_to_top20_cutoff_mean", "best_target_rank_mean", "target_rank_near_cutoff_rate", "hard_negative_pressure_score_mean"])}

## hard negative signature

> [!info] 字段说明
> `margin_bucket`：按 baseline `margin_to_top20_cutoff` 分桶。
> `near_cutoff_item_count`：目标用户最佳排名在 21 到 200 的 item 数。
> `near_cutoff_ndcg@20_mean`：near cutoff 子集 NDCG@20 均值。
> `top20_user_norm_mean`：top20 用户范数均值。
> `top20_history_interaction_count_mean`：top20 用户历史交互数均值。
> `target_minus_top20_cosine_mean`：target 用户 cosine 相对 top20 用户 cosine 的 gap。

{md_table(hard_negative_signature, ["margin_bucket", "item_count", "near_cutoff_item_count", "near_cutoff_rate", "near_cutoff_ndcg@20_mean", "hard_negative_pressure_score_mean", "top20_user_norm_mean", "top20_history_interaction_count_mean", "target_minus_top20_cosine_mean"])}

## controlled margin gap

> [!info] 字段说明
> `control_set`：控制变量集合。
> `margin_bucket`：按 baseline `margin_to_top20_cutoff` 分桶。
> `ndcg@20_control_residual_mean`：扣除控制桶均值后的 NDCG@20 residual 均值。
> `hr@20_control_residual_mean`：扣除控制桶均值后的 HR@20 residual 均值。

{md_table(controlled_focus, ["control_set", "margin_bucket", "item_count", "ndcg@20_mean", "ndcg@20_control_residual_mean", "hr@20_control_residual_mean"], max_rows=12)}

## 判断

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

## 下一步

如果本 route 为 `rank_margin_hard_negative_supported`，继续用 Task3.9 / Task3.20 分解 target-vs-competitor 和 competitor cohort；如果控制后证据不足，转入 Task3.10 做 norm/activity matched residual。

## 产物

```text
task3p8_rank_margin_profile.csv
task3p8_margin_bucket_summary.csv
task3p8_hard_negative_signature.csv
task3p8_controlled_margin_gap.csv
task3p8_route_decision.json
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
    baseline_target = pd.read_csv(task3_dir / "baseline_target_score_source" / "target_alignment_profile.csv")
    baseline_content = pd.read_csv(task3_dir / "baseline_content_cf_alignment" / "content_cf_alignment_profile.csv")

    profile = build_rank_margin_profile(item_profile, baseline_score, baseline_target, baseline_content)
    bucket_summary = build_margin_bucket_summary(profile)
    hard_negative_signature = build_hard_negative_signature(profile)
    controlled_gap = build_controlled_margin_gap(profile)
    decision = build_route_decision(bucket_summary, controlled_gap)

    outputs = {
        "task3p8_rank_margin_profile": output_dir / "task3p8_rank_margin_profile.csv",
        "task3p8_margin_bucket_summary": output_dir / "task3p8_margin_bucket_summary.csv",
        "task3p8_hard_negative_signature": output_dir / "task3p8_hard_negative_signature.csv",
        "task3p8_controlled_margin_gap": output_dir / "task3p8_controlled_margin_gap.csv",
        "task3p8_route_decision": output_dir / "task3p8_route_decision.json",
    }
    profile.to_csv(outputs["task3p8_rank_margin_profile"], index=False)
    bucket_summary.to_csv(outputs["task3p8_margin_bucket_summary"], index=False)
    hard_negative_signature.to_csv(outputs["task3p8_hard_negative_signature"], index=False)
    controlled_gap.to_csv(outputs["task3p8_controlled_margin_gap"], index=False)
    outputs["task3p8_route_decision"].write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    result_md = output_dir / f"{run_stamp} CCFCRec Amazon-VG category availability v2 Task3.8 rank-margin 诊断结果.md"
    manifest_path = output_dir / "run_manifest.json"
    result_dir_display = args.result_dir_display or str(output_dir)
    write_result_markdown(
        output_path=result_md,
        run_stamp=run_stamp,
        source_notes=args.source_note,
        result_dir_display=result_dir_display,
        bucket_summary=bucket_summary,
        hard_negative_signature=hard_negative_signature,
        controlled_gap=controlled_gap,
        decision=decision,
    )

    manifest = {
        "run_time": run_iso,
        "script": str(Path(__file__).resolve()),
        "task3_dir": str(task3_dir),
        "output_dir": str(output_dir),
        "inputs": {
            "task3_item_profile": str(task3_dir / "task3_item_profile.csv"),
            "baseline_score_margin": str(task3_dir / "baseline_score_margin" / "item_score_margin_profile.csv"),
            "baseline_target_score_source": str(task3_dir / "baseline_target_score_source" / "target_alignment_profile.csv"),
            "baseline_content_cf_alignment": str(task3_dir / "baseline_content_cf_alignment" / "content_cf_alignment_profile.csv"),
        },
        "outputs": {key: str(value) for key, value in outputs.items()} | {
            "result_md": str(result_md),
            "run_manifest": str(manifest_path),
        },
        "row_counts": {
            "task3p8_rank_margin_profile": int(len(profile)),
            "task3p8_margin_bucket_summary": int(len(bucket_summary)),
            "task3p8_hard_negative_signature": int(len(hard_negative_signature)),
            "task3p8_controlled_margin_gap": int(len(controlled_gap)),
        },
        "route_decision": decision,
        "notes": [
            "只复用 Task3 baseline item-level CSV，不训练模型。",
            "Task3.8 判断 baseline rank margin / hard negative pressure 是否足以作为 Task4 前置入口。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG category availability v2 Task3.8 rank-margin 诊断")
    parser.add_argument("--task3-dir", required=True, type=str, help="Task3 checkpoint 复评输出目录")
    parser.add_argument("--output-dir", required=True, type=str, help="Task3.8 输出目录")
    parser.add_argument("--source-note", action="append", default=[], help="写入结果 MD 来源说明的 Obsidian wikilink 行")
    parser.add_argument("--result-dir-display", type=str, default="", help="写入结果 MD 的相对结果目录")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
