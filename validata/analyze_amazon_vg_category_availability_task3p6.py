#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.6 分组有效性诊断。

脚本只复用 Task3 baseline item-level CSV，判断 v2 s_cat_group 是否真能预测推荐失败模式。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_amazon_vg_category_availability_task3p5 import md_table


CONTROL_SETS = [
    ("none", []),
    ("gt_group", ["gt_group"]),
    ("category_count_bucket", ["category_count_bucket"]),
    ("target_activity_bucket", ["target_activity_bucket"]),
    ("gt_group+target_activity_bucket", ["gt_group", "target_activity_bucket"]),
    ("gt_group+category_count_bucket", ["gt_group", "category_count_bucket"]),
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


def category_count_bucket(value: Any) -> str:
    count = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(count):
        return "cat_unknown"
    if count <= 3:
        return "cat_weak_1_3"
    if count <= 5:
        return "cat_mid_4_5"
    return "cat_strong_6_plus"


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


def build_group_validity_profile(
    item_profile: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    baseline_content: pd.DataFrame,
) -> pd.DataFrame:
    required = {"asin", "category_group", "category_count", "gt_group", "gt_user_count"}
    missing = sorted(required.difference(item_profile.columns))
    if missing:
        raise ValueError(f"item_profile 缺少必要列: {missing}")
    if "asin" not in baseline_metrics.columns:
        raise ValueError("baseline_metrics 必须包含 asin 列")
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
    metric_columns = [column for column in ["hr@20", "ndcg@20"] if column in baseline_metrics.columns]
    metrics = baseline_metrics[["asin", *metric_columns]].copy()
    metrics = metrics.rename(columns={column: f"baseline_{column}" for column in metric_columns})
    profile = profile.merge(metrics, on="asin", how="left")

    if "target_history_interaction_count_mean" in baseline_content.columns:
        activity = baseline_content[["asin", "target_history_interaction_count_mean"]].drop_duplicates("asin")
        profile = profile.merge(activity, on="asin", how="left")
    else:
        profile["target_history_interaction_count_mean"] = pd.NA

    profile["category_count_bucket"] = profile["category_count"].map(category_count_bucket)
    profile["target_activity_bucket"] = _quantile_labels(profile["target_history_interaction_count_mean"])
    return profile.sort_values("asin").reset_index(drop=True)


def build_baseline_metric_by_v2_group(profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, frame in profile.groupby("category_group", dropna=False):
        row = {
            "category_group": group,
            "item_count": int(len(frame)),
            "baseline_hr@20_mean": _safe_mean(frame["baseline_hr@20"]),
            "baseline_ndcg@20_mean": _safe_mean(frame["baseline_ndcg@20"]),
            "baseline_ndcg@20_median": float(pd.to_numeric(frame["baseline_ndcg@20"], errors="coerce").median()),
            "gt_user_count_mean": _safe_mean(frame["gt_user_count"]) if "gt_user_count" in frame.columns else float("nan"),
            "category_count_mean": _safe_mean(frame["category_count"]) if "category_count" in frame.columns else float("nan"),
            "target_history_interaction_count_mean": _safe_mean(frame["target_history_interaction_count_mean"])
            if "target_history_interaction_count_mean" in frame.columns
            else float("nan"),
        }
        for metric in ["s_cat", "s_cat_v1", "s_cat_v2_disc_within_control", "s_cat_v2_collab_within_control"]:
            if metric in frame.columns:
                row[f"{metric}_mean"] = _safe_mean(frame[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("category_group").reset_index(drop=True)


def build_cross_distribution(profile: pd.DataFrame, other_col: str, old_group_type: str | None = None) -> pd.DataFrame:
    if other_col not in profile.columns:
        raise ValueError(f"profile 缺少交叉列: {other_col}")
    grouped = (
        profile.groupby(["category_group", other_col], dropna=False)
        .size()
        .rename("count")
        .reset_index()
        .rename(columns={other_col: "other_group"})
    )
    grouped["within_v2_group_rate"] = grouped["count"] / grouped.groupby("category_group")["count"].transform("sum")
    grouped["within_other_group_rate"] = grouped["count"] / grouped.groupby("other_group")["count"].transform("sum")
    if old_group_type is not None:
        grouped.insert(1, "old_group_type", old_group_type)
    return grouped.sort_values(["category_group", "other_group"]).reset_index(drop=True)


def build_controlled_metric_gap(profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for control_name, control_cols in CONTROL_SETS:
        missing = [column for column in control_cols if column not in profile.columns]
        if missing:
            continue
        frame = profile.copy()
        for metric in ["baseline_ndcg@20", "baseline_hr@20"]:
            if control_cols:
                frame[f"{metric}_control_mean"] = frame.groupby(control_cols, dropna=False)[metric].transform("mean")
            else:
                frame[f"{metric}_control_mean"] = pd.to_numeric(frame[metric], errors="coerce").mean()
            frame[f"{metric}_control_residual"] = frame[metric] - frame[f"{metric}_control_mean"]
        for group, group_frame in frame.groupby("category_group", dropna=False):
            rows.append(
                {
                    "control_set": control_name,
                    "category_group": group,
                    "item_count": int(len(group_frame)),
                    "baseline_ndcg@20_mean": _safe_mean(group_frame["baseline_ndcg@20"]),
                    "baseline_ndcg@20_control_residual_mean": _safe_mean(
                        group_frame["baseline_ndcg@20_control_residual"]
                    ),
                    "baseline_hr@20_mean": _safe_mean(group_frame["baseline_hr@20"]),
                    "baseline_hr@20_control_residual_mean": _safe_mean(
                        group_frame["baseline_hr@20_control_residual"]
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["control_set", "category_group"]).reset_index(drop=True)


def _group_value(summary: pd.DataFrame, group_name_part: str, column: str) -> float:
    rows = summary[summary["category_group"].astype(str).str.contains(group_name_part, na=False)]
    if rows.empty or column not in rows.columns:
        return float("nan")
    return float(rows.iloc[0][column])


def build_group_validity_summary(metric_summary: pd.DataFrame, controlled_gap: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    weak_ndcg = _group_value(metric_summary, "weak", "baseline_ndcg@20_mean")
    mid_ndcg = _group_value(metric_summary, "mid", "baseline_ndcg@20_mean")
    strong_ndcg = _group_value(metric_summary, "strong", "baseline_ndcg@20_mean")
    s_cat_corr = float(
        pd.to_numeric(profile.get("s_cat"), errors="coerce").corr(
            pd.to_numeric(profile.get("baseline_ndcg@20"), errors="coerce"),
            method="spearman",
        )
    )
    gt_corr = float(
        pd.to_numeric(profile.get("gt_user_count"), errors="coerce").corr(
            pd.to_numeric(profile.get("baseline_ndcg@20"), errors="coerce"),
            method="spearman",
        )
    )
    controlled = controlled_gap[
        controlled_gap["control_set"].eq("gt_group+target_activity_bucket")
    ]
    weak_residual = _group_value(controlled, "weak", "baseline_ndcg@20_control_residual_mean")
    strong_residual = _group_value(controlled, "strong", "baseline_ndcg@20_control_residual_mean")
    return pd.DataFrame(
        [
            {"field": "weak_baseline_ndcg@20_mean", "value": weak_ndcg},
            {"field": "mid_baseline_ndcg@20_mean", "value": mid_ndcg},
            {"field": "strong_baseline_ndcg@20_mean", "value": strong_ndcg},
            {"field": "weak_minus_strong_baseline_ndcg@20", "value": weak_ndcg - strong_ndcg},
            {"field": "weak_minus_mid_baseline_ndcg@20", "value": weak_ndcg - mid_ndcg},
            {"field": "spearman_s_cat_vs_baseline_ndcg@20", "value": s_cat_corr},
            {"field": "spearman_gt_user_count_vs_baseline_ndcg@20", "value": gt_corr},
            {"field": "controlled_weak_minus_strong_residual_ndcg@20", "value": weak_residual - strong_residual},
        ]
    )


def build_group_validity_decision(metric_summary: pd.DataFrame, controlled_gap: pd.DataFrame) -> dict[str, Any]:
    weak_ndcg = _group_value(metric_summary, "weak", "baseline_ndcg@20_mean")
    mid_ndcg = _group_value(metric_summary, "mid", "baseline_ndcg@20_mean")
    strong_ndcg = _group_value(metric_summary, "strong", "baseline_ndcg@20_mean")
    controlled = controlled_gap[controlled_gap["control_set"].eq("gt_group+target_activity_bucket")]
    weak_residual = _group_value(controlled, "weak", "baseline_ndcg@20_control_residual_mean")
    strong_residual = _group_value(controlled, "strong", "baseline_ndcg@20_control_residual_mean")
    weak_minus_strong = weak_ndcg - strong_ndcg
    weak_minus_mid = weak_ndcg - mid_ndcg
    controlled_weak_minus_strong = weak_residual - strong_residual

    if weak_minus_strong < -0.005 and controlled_weak_minus_strong < -0.003:
        route = "v2_group_predicts_failure_mode"
        reason = "v2 weak 的 baseline NDCG 明显低于 strong，且控制 gt_group 与 target activity 后残差差距仍为负。"
    elif weak_minus_strong >= 0 or controlled_weak_minus_strong >= 0:
        route = "v2_group_not_task_relevant"
        reason = "v2 weak 没有表现为 baseline 推荐失败组；原始或控制后 NDCG gap 不支持 weak=hard subgroup。"
    else:
        route = "v2_group_needs_redefinition"
        reason = "v2 group 与 baseline 失败关系不稳定，且控制后仍无法形成清晰 hard subgroup 语义。"

    evidence = {
        "weak_baseline_ndcg@20_mean": weak_ndcg,
        "mid_baseline_ndcg@20_mean": mid_ndcg,
        "strong_baseline_ndcg@20_mean": strong_ndcg,
        "weak_minus_strong_baseline_ndcg@20": weak_minus_strong,
        "weak_minus_mid_baseline_ndcg@20": weak_minus_mid,
        "controlled_weak_minus_strong_residual_ndcg@20": controlled_weak_minus_strong,
    }
    return {"route": route, "reason": reason, "evidence": {key: _jsonable(value) for key, value in evidence.items()}}


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    source_notes: list[str],
    result_dir_display: str,
    validity_summary: pd.DataFrame,
    metric_summary: pd.DataFrame,
    cross_gt: pd.DataFrame,
    cross_old: pd.DataFrame,
    controlled_gap: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    source_lines = "\n".join(f"> {note}" for note in source_notes) or "> 上游设计：未提供"
    controlled_focus = controlled_gap[
        controlled_gap["control_set"].isin(["none", "gt_group+target_activity_bucket", "gt_group+category_count_bucket"])
    ].copy()
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.6 分组有效性诊断结果
date: {run_stamp[:10]}
time: "{run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
created_at: "{run_stamp[:10]} {run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - Task3.6
  - 分组有效性
---

# {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.6 分组有效性诊断结果

## 来源说明

> [!info] 来源说明
{source_lines}
> 本结果目录：`{result_dir_display}`

## 结论

Task3.6 的 route decision 为 `{decision["route"]}`。

{decision["reason"]}

这说明本轮判断的是 v2 `s_cat_group` 是否适合作为推荐失败分层，不是在否定 category availability 变量构造本身。

## 数据验收

> [!info] 字段说明
> `field`：验收项。
> `value`：验收项取值。

| field | value |
|---|---:|
| v2_group_count | {len(metric_summary)} |
| route | `{decision["route"]}` |
| weak_minus_strong_baseline_ndcg@20 | {decision["evidence"].get("weak_minus_strong_baseline_ndcg@20")} |
| controlled_weak_minus_strong_residual_ndcg@20 | {decision["evidence"].get("controlled_weak_minus_strong_residual_ndcg@20")} |

## 分组有效性摘要

> [!info] 字段说明
> `field`：诊断项。
> `value`：诊断项取值。

{md_table(validity_summary, ["field", "value"])}

## baseline 指标 by v2 group

> [!info] 字段说明
> `category_group`：v2 `s_cat_group`。
> `item_count`：该组 test item 数。
> `baseline_ndcg@20_mean`：baseline item-level NDCG@20 均值。
> `baseline_hr@20_mean`：baseline item-level HR@20 均值。
> `gt_user_count_mean`：测试集中该 item 真实目标用户数均值。
> `category_count_mean`：原始 category_count 均值。
> `target_history_interaction_count_mean`：目标用户历史交互数均值。

{md_table(metric_summary, ["category_group", "item_count", "baseline_ndcg@20_mean", "baseline_hr@20_mean", "gt_user_count_mean", "category_count_mean", "target_history_interaction_count_mean"])}

## v2 group x gt_group

> [!info] 字段说明
> `category_group`：v2 `s_cat_group`。
> `other_group`：gt_user_count 分桶。
> `count`：交叉单元 item 数。
> `within_v2_group_rate`：在该 v2 group 内的占比。
> `within_other_group_rate`：在该 gt_group 内的占比。

{md_table(cross_gt, ["category_group", "other_group", "count", "within_v2_group_rate", "within_other_group_rate"], max_rows=18)}

## v2 group x old group

> [!info] 字段说明
> `category_group`：v2 `s_cat_group`。
> `old_group_type`：旧分组类型。
> `other_group`：旧分组取值。
> `count`：交叉单元 item 数。
> `within_v2_group_rate`：在该 v2 group 内的占比。

{md_table(cross_old, ["category_group", "old_group_type", "other_group", "count", "within_v2_group_rate"], max_rows=24)}

## 控制后性能 gap

> [!info] 字段说明
> `control_set`：控制变量集合。
> `category_group`：v2 `s_cat_group`。
> `item_count`：该组 item 数。
> `baseline_ndcg@20_mean`：baseline 原始 NDCG@20 均值。
> `baseline_ndcg@20_control_residual_mean`：扣除控制桶均值后的 NDCG@20 residual 均值。
> `baseline_hr@20_control_residual_mean`：扣除控制桶均值后的 HR@20 residual 均值。

{md_table(controlled_focus, ["control_set", "category_group", "item_count", "baseline_ndcg@20_mean", "baseline_ndcg@20_control_residual_mean", "baseline_hr@20_control_residual_mean"], max_rows=18)}

## 判断

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

## 下一步

继续进入 Task3.7，检查 category_conf_input 这个方法载体是否真的匹配 v2 category availability 关心的机制。

## 产物

```text
task3p6_group_validity_summary.csv
task3p6_v2_group_cross_with_gt_group.csv
task3p6_v2_group_cross_with_v1_old_group.csv
task3p6_baseline_metric_by_v2_group.csv
task3p6_controlled_metric_gap.csv
task3p6_group_validity_decision.json
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
    baseline_metrics = pd.read_csv(task3_dir / "baseline_item_profile.csv")
    baseline_content = pd.read_csv(task3_dir / "baseline_content_cf_alignment" / "content_cf_alignment_profile.csv")
    profile = build_group_validity_profile(item_profile, baseline_metrics, baseline_content)
    metric_summary = build_baseline_metric_by_v2_group(profile)
    cross_gt = build_cross_distribution(profile, "gt_group")
    cross_v1 = build_cross_distribution(profile, "s_cat_group_v1", old_group_type="s_cat_group_v1")
    cross_category_count = build_cross_distribution(
        profile,
        "category_count_bucket",
        old_group_type="category_count_bucket",
    )
    cross_old = pd.concat([cross_v1, cross_category_count], ignore_index=True)
    controlled_gap = build_controlled_metric_gap(profile)
    validity_summary = build_group_validity_summary(metric_summary, controlled_gap, profile)
    decision = build_group_validity_decision(metric_summary, controlled_gap)

    outputs = {
        "task3p6_group_validity_summary": output_dir / "task3p6_group_validity_summary.csv",
        "task3p6_v2_group_cross_with_gt_group": output_dir / "task3p6_v2_group_cross_with_gt_group.csv",
        "task3p6_v2_group_cross_with_v1_old_group": output_dir / "task3p6_v2_group_cross_with_v1_old_group.csv",
        "task3p6_baseline_metric_by_v2_group": output_dir / "task3p6_baseline_metric_by_v2_group.csv",
        "task3p6_controlled_metric_gap": output_dir / "task3p6_controlled_metric_gap.csv",
        "task3p6_group_validity_decision": output_dir / "task3p6_group_validity_decision.json",
    }
    validity_summary.to_csv(outputs["task3p6_group_validity_summary"], index=False)
    cross_gt.to_csv(outputs["task3p6_v2_group_cross_with_gt_group"], index=False)
    cross_old.to_csv(outputs["task3p6_v2_group_cross_with_v1_old_group"], index=False)
    metric_summary.to_csv(outputs["task3p6_baseline_metric_by_v2_group"], index=False)
    controlled_gap.to_csv(outputs["task3p6_controlled_metric_gap"], index=False)
    outputs["task3p6_group_validity_decision"].write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result_md = output_dir / f"{run_stamp} CCFCRec Amazon-VG category availability v2 Task3.6 分组有效性诊断结果.md"
    manifest_path = output_dir / "run_manifest.json"
    result_dir_display = args.result_dir_display or str(output_dir)
    write_result_markdown(
        output_path=result_md,
        run_stamp=run_stamp,
        source_notes=args.source_note,
        result_dir_display=result_dir_display,
        validity_summary=validity_summary,
        metric_summary=metric_summary,
        cross_gt=cross_gt,
        cross_old=cross_old,
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
            "baseline_item_profile": str(task3_dir / "baseline_item_profile.csv"),
            "baseline_content_cf_alignment": str(task3_dir / "baseline_content_cf_alignment" / "content_cf_alignment_profile.csv"),
        },
        "outputs": {key: str(value) for key, value in outputs.items()} | {
            "result_md": str(result_md),
            "run_manifest": str(manifest_path),
        },
        "row_counts": {
            "task3p6_group_validity_summary": int(len(validity_summary)),
            "task3p6_v2_group_cross_with_gt_group": int(len(cross_gt)),
            "task3p6_v2_group_cross_with_v1_old_group": int(len(cross_old)),
            "task3p6_baseline_metric_by_v2_group": int(len(metric_summary)),
            "task3p6_controlled_metric_gap": int(len(controlled_gap)),
        },
        "route_decision": decision,
        "notes": [
            "只复用 Task3 baseline item-level CSV，不训练模型。",
            "Task3.6 只判断 v2 group 是否推荐失败相关，不判断方法载体。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG category availability v2 Task3.6 分组有效性诊断")
    parser.add_argument("--task3-dir", required=True, type=str, help="Task3 checkpoint 复评输出目录")
    parser.add_argument("--output-dir", required=True, type=str, help="Task3.6 输出目录")
    parser.add_argument("--source-note", action="append", default=[], help="写入结果 MD 来源说明的 Obsidian wikilink 行")
    parser.add_argument("--result-dir-display", type=str, default="", help="写入结果 MD 的相对结果目录")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
