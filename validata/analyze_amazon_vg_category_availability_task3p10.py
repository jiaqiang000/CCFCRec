#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.10 norm/activity matched residual 诊断。

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
from analyze_amazon_vg_category_availability_task3p8 import (
    _jsonable,
    _quantile_labels,
    _safe_mean,
    build_rank_margin_profile,
)


CONTROL_COLUMNS = [
    "gt_group",
    "target_activity_bucket",
    "target_norm_bucket",
    "top20_norm_bucket",
]

CANDIDATE_COLUMNS = [
    "margin_bucket",
    "category_group",
    "category_count_bucket",
    "target_top20_cosine_gap_bucket",
    "target_top20_history_q_gap_bucket",
]


def now_stamp() -> tuple[str, str]:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def category_count_bucket(value: Any) -> str:
    count = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(count):
        return "cat_unknown"
    if count <= 3:
        return "cat_weak_1_3"
    if count <= 5:
        return "cat_mid_4_5"
    return "cat_strong_6_plus"


def _residual_labels(series: pd.Series) -> pd.Series:
    base = _quantile_labels(series)
    return base.map({"low": "hard", "mid": "mid", "high": "easy", "all": "all", "unknown": "unknown"}).fillna(base)


def _gap_bucket(series: pd.Series) -> pd.Series:
    base = _quantile_labels(series)
    return base.map({"low": "gap_low", "mid": "gap_mid", "high": "gap_high", "all": "gap_all", "unknown": "gap_unknown"}).fillna(base)


def _control_bucket(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.dropna()
    if len(valid) < 6 or valid.nunique() <= 1:
        labels = pd.Series("all", index=series.index, dtype="object")
        labels.loc[numeric.isna()] = "unknown"
        return labels
    return _quantile_labels(series)


def build_norm_activity_matched_profile(
    item_profile: pd.DataFrame,
    baseline_score: pd.DataFrame,
    baseline_target: pd.DataFrame,
    baseline_content: pd.DataFrame,
) -> pd.DataFrame:
    profile = build_rank_margin_profile(item_profile, baseline_score, baseline_target, baseline_content)

    profile["category_count_bucket"] = profile["category_count"].map(category_count_bucket)
    profile["target_activity_bucket"] = _control_bucket(profile["target_history_interaction_count_mean"])
    profile["target_norm_bucket"] = _control_bucket(profile["target_user_norm_mean"])
    profile["top20_norm_bucket"] = _control_bucket(profile["top20_user_norm_mean"])
    profile["top20_activity_bucket"] = _control_bucket(profile["top20_history_interaction_count_mean"])
    profile["target_top20_cosine_gap_bucket"] = _gap_bucket(profile["target_minus_top20_cosine_mean"])
    profile["target_top20_history_q_gap_bucket"] = _gap_bucket(profile["target_minus_top20_history_q_cosine_mean"])

    for column in CONTROL_COLUMNS:
        if column not in profile.columns:
            profile[column] = "unknown"
    profile["matched_control_key"] = profile[CONTROL_COLUMNS].fillna("unknown").astype(str).agg("|".join, axis=1)
    profile["matched_cell_size"] = profile.groupby("matched_control_key", dropna=False)["asin"].transform("size")

    for metric in ["ndcg@20", "hr@20"]:
        profile[f"{metric}_control_mean"] = profile.groupby("matched_control_key", dropna=False)[metric].transform("mean")
        profile[f"{metric}_control_residual"] = profile[metric] - profile[f"{metric}_control_mean"]
    profile["residual_bucket"] = _residual_labels(profile["ndcg@20_control_residual"])
    return profile.sort_values("asin").reset_index(drop=True)


def build_matched_residual_summary(profile: pd.DataFrame) -> pd.DataFrame:
    if "residual_bucket" not in profile.columns:
        raise ValueError("profile 必须包含 residual_bucket 列")
    rows: list[dict[str, Any]] = []
    for bucket, frame in profile.groupby("residual_bucket", dropna=False):
        row: dict[str, Any] = {
            "residual_bucket": bucket,
            "item_count": int(len(frame)),
            "ndcg@20_mean": _safe_mean(frame["ndcg@20"]),
            "hr@20_mean": _safe_mean(frame["hr@20"]),
            "ndcg@20_control_residual_mean": _safe_mean(frame["ndcg@20_control_residual"]),
            "hr@20_control_residual_mean": _safe_mean(frame["hr@20_control_residual"]),
            "matched_cell_size_mean": _safe_mean(frame["matched_cell_size"])
            if "matched_cell_size" in frame.columns
            else float(len(frame)),
        }
        for metric in [
            "margin_to_top20_cutoff",
            "best_target_rank",
            "hard_negative_pressure_score",
            "top20_user_norm_mean",
            "top20_history_interaction_count_mean",
            "target_minus_top20_cosine_mean",
            "target_minus_top20_history_q_cosine_mean",
        ]:
            if metric in frame.columns:
                row[f"{metric}_mean"] = _safe_mean(frame[metric])
        rows.append(row)
    order = {"hard": 0, "mid": 1, "easy": 2, "all": 3, "unknown": 4}
    return (
        pd.DataFrame(rows)
        .assign(_order=lambda df: df["residual_bucket"].map(order).fillna(99))
        .sort_values("_order")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )


def build_hard_subgroup_candidates(profile: pd.DataFrame) -> pd.DataFrame:
    hard_mask = profile["residual_bucket"].eq("hard")
    base_hard_rate = float(hard_mask.mean()) if len(profile) else float("nan")
    rows: list[dict[str, Any]] = []
    for candidate_col in CANDIDATE_COLUMNS:
        if candidate_col not in profile.columns:
            continue
        for value, frame in profile.groupby(candidate_col, dropna=False):
            hard_count = int(frame["residual_bucket"].eq("hard").sum())
            hard_rate = float(hard_count / len(frame)) if len(frame) else float("nan")
            enrichment = hard_rate / base_hard_rate if base_hard_rate and not pd.isna(base_hard_rate) else float("nan")
            rows.append(
                {
                    "candidate_col": candidate_col,
                    "candidate_value": value,
                    "item_count": int(len(frame)),
                    "hard_item_count": hard_count,
                    "hard_rate": hard_rate,
                    "base_hard_rate": base_hard_rate,
                    "hard_rate_enrichment": enrichment,
                    "ndcg@20_mean": _safe_mean(frame["ndcg@20"]),
                    "ndcg@20_control_residual_mean": _safe_mean(frame["ndcg@20_control_residual"]),
                    "margin_to_top20_cutoff_mean": _safe_mean(frame["margin_to_top20_cutoff"])
                    if "margin_to_top20_cutoff" in frame.columns
                    else float("nan"),
                    "hard_negative_pressure_score_mean": _safe_mean(frame["hard_negative_pressure_score"])
                    if "hard_negative_pressure_score" in frame.columns
                    else float("nan"),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["hard_rate_enrichment", "hard_item_count", "item_count"], ascending=[False, False, False])
        .reset_index(drop=True)
    )


def _summary_value(summary: pd.DataFrame, bucket: str, column: str) -> float:
    rows = summary[summary["residual_bucket"].eq(bucket)]
    if rows.empty or column not in rows.columns:
        return float("nan")
    return float(rows.iloc[0][column])


def build_route_decision(summary: pd.DataFrame, candidates: pd.DataFrame) -> dict[str, Any]:
    hard_count = int(_summary_value(summary, "hard", "item_count")) if not pd.isna(_summary_value(summary, "hard", "item_count")) else 0
    hard_residual = _summary_value(summary, "hard", "ndcg@20_control_residual_mean")
    easy_residual = _summary_value(summary, "easy", "ndcg@20_control_residual_mean")
    hard_minus_easy = hard_residual - easy_residual
    total_count = int(summary["item_count"].sum()) if "item_count" in summary.columns else 0
    min_candidate_count = 2 if total_count <= 10 else 50

    usable_candidates = candidates[
        (candidates["item_count"] >= min_candidate_count) & (candidates["hard_item_count"] >= max(2, min_candidate_count // 5))
    ].copy()
    top = usable_candidates.iloc[0] if not usable_candidates.empty else None
    top_enrichment = float(top["hard_rate_enrichment"]) if top is not None else float("nan")
    top_col = str(top["candidate_col"]) if top is not None else ""
    top_value = str(top["candidate_value"]) if top is not None else ""

    if hard_count < max(2, min_candidate_count):
        route = "matched_signal_too_sparse"
        reason = "匹配后 hard residual 样本过少，不能作为 Task4 子群。"
    elif pd.isna(hard_residual) or hard_residual >= -0.005:
        route = "failure_explained_by_activity_norm"
        reason = "控制 user activity / norm 后 hard residual 不明显，失败主要可由控制变量解释。"
    elif top is not None and top_enrichment >= 1.25:
        route = "matched_hard_subgroup_found"
        reason = "控制 user activity / norm 后仍存在明显 hard residual，并且有候选机制变量富集。"
    else:
        route = "inconclusive_needs_more_controls"
        reason = "控制后存在 hard residual，但没有找到足够清晰的候选机制变量富集。"

    evidence = {
        "hard_item_count": hard_count,
        "hard_ndcg@20_control_residual_mean": hard_residual,
        "easy_ndcg@20_control_residual_mean": easy_residual,
        "hard_minus_easy_residual_ndcg@20": hard_minus_easy,
        "top_candidate_col": top_col,
        "top_candidate_value": top_value,
        "top_candidate_hard_rate_enrichment": top_enrichment,
    }
    return {"route": route, "reason": reason, "evidence": {key: _jsonable(value) for key, value in evidence.items()}}


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    source_notes: list[str],
    result_dir_display: str,
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    source_lines = "\n".join(f"> {note}" for note in source_notes) or "> 上游设计：未提供"
    top_candidates = candidates.head(20).copy()
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.10 norm-activity matched 诊断结果
date: {run_stamp[:10]}
time: "{run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
created_at: "{run_stamp[:10]} {run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - Task3.10
  - matched_residual
---

# {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.10 norm-activity matched 诊断结果

## 来源说明

> [!info] 来源说明
{source_lines}
> 本结果目录：`{result_dir_display}`

## 结论

Task3.10 的 route decision 为 `{decision["route"]}`。

{decision["reason"]}

本结果只判断控制 user activity / norm 后是否仍存在可解释 hard residual 子群，不训练模型，也不直接进入 Task4。

## 数据验收

> [!info] 字段说明
> `field`：验收项。
> `value`：验收项取值。

| field | value |
|---|---:|
| residual_bucket_count | {len(summary)} |
| route | `{decision["route"]}` |
| hard_item_count | {decision["evidence"].get("hard_item_count")} |
| top_candidate | `{decision["evidence"].get("top_candidate_col")}={decision["evidence"].get("top_candidate_value")}` |

## matched residual summary

> [!info] 字段说明
> `residual_bucket`：按控制 user activity / norm 后的 NDCG residual 分桶。
> `item_count`：该桶 item 数。
> `ndcg@20_control_residual_mean`：扣除匹配控制桶均值后的 NDCG@20 residual。
> `matched_cell_size_mean`：匹配控制桶平均样本数。
> `hard_negative_pressure_score_mean`：hard negative pressure 均值。

{md_table(summary, ["residual_bucket", "item_count", "ndcg@20_mean", "ndcg@20_control_residual_mean", "hr@20_control_residual_mean", "matched_cell_size_mean", "margin_to_top20_cutoff_mean", "hard_negative_pressure_score_mean"])}

## hard subgroup candidates

> [!info] 字段说明
> `candidate_col`：候选机制变量。
> `candidate_value`：候选变量取值。
> `hard_rate`：该候选内 hard residual item 比例。
> `base_hard_rate`：全体 hard residual item 比例。
> `hard_rate_enrichment`：候选 hard rate 相对全体 hard rate 的富集倍数。

{md_table(top_candidates, ["candidate_col", "candidate_value", "item_count", "hard_item_count", "hard_rate", "base_hard_rate", "hard_rate_enrichment", "ndcg@20_control_residual_mean", "margin_to_top20_cutoff_mean"], max_rows=20)}

## 判断

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

## 下一步

如果 `matched_hard_subgroup_found` 成立，继续检查富集候选是否只是 margin proxy，必要时进入 Task3.11 / Task3.15；如果不成立，继续按第一波设计执行 category-CF interaction。

## 产物

```text
task3p10_norm_activity_matched_profile.csv
task3p10_matched_residual_summary.csv
task3p10_hard_subgroup_candidates.csv
task3p10_route_decision.json
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

    profile = build_norm_activity_matched_profile(item_profile, baseline_score, baseline_target, baseline_content)
    summary = build_matched_residual_summary(profile)
    candidates = build_hard_subgroup_candidates(profile)
    decision = build_route_decision(summary, candidates)

    outputs = {
        "task3p10_norm_activity_matched_profile": output_dir / "task3p10_norm_activity_matched_profile.csv",
        "task3p10_matched_residual_summary": output_dir / "task3p10_matched_residual_summary.csv",
        "task3p10_hard_subgroup_candidates": output_dir / "task3p10_hard_subgroup_candidates.csv",
        "task3p10_route_decision": output_dir / "task3p10_route_decision.json",
    }
    profile.to_csv(outputs["task3p10_norm_activity_matched_profile"], index=False)
    summary.to_csv(outputs["task3p10_matched_residual_summary"], index=False)
    candidates.to_csv(outputs["task3p10_hard_subgroup_candidates"], index=False)
    outputs["task3p10_route_decision"].write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    result_md = output_dir / f"{run_stamp} CCFCRec Amazon-VG category availability v2 Task3.10 norm-activity matched 诊断结果.md"
    manifest_path = output_dir / "run_manifest.json"
    result_dir_display = args.result_dir_display or str(output_dir)
    write_result_markdown(
        output_path=result_md,
        run_stamp=run_stamp,
        source_notes=args.source_note,
        result_dir_display=result_dir_display,
        summary=summary,
        candidates=candidates,
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
            "task3p10_norm_activity_matched_profile": int(len(profile)),
            "task3p10_matched_residual_summary": int(len(summary)),
            "task3p10_hard_subgroup_candidates": int(len(candidates)),
        },
        "route_decision": decision,
        "notes": [
            "只复用 Task3 baseline item-level CSV，不训练模型。",
            "Task3.10 判断控制 user activity / norm 后是否仍存在可解释 hard residual 子群。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG category availability v2 Task3.10 norm/activity matched residual 诊断")
    parser.add_argument("--task3-dir", required=True, type=str, help="Task3 checkpoint 复评输出目录")
    parser.add_argument("--output-dir", required=True, type=str, help="Task3.10 输出目录")
    parser.add_argument("--source-note", action="append", default=[], help="写入结果 MD 来源说明的 Obsidian wikilink 行")
    parser.add_argument("--result-dir-display", type=str, default="", help="写入结果 MD 的相对结果目录")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
