#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.11 category-CF interaction 诊断。

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
from analyze_amazon_vg_category_availability_task3p10 import _control_bucket, category_count_bucket


def now_stamp() -> tuple[str, str]:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def _prefixed_bucket(series: pd.Series, prefix: str) -> pd.Series:
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


def _mean_existing(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(float("nan"), index=frame.index)
    numeric = pd.concat([pd.to_numeric(frame[column], errors="coerce") for column in available], axis=1)
    return numeric.mean(axis=1)


def build_category_cf_interaction_profile(
    item_profile: pd.DataFrame,
    baseline_score: pd.DataFrame,
    baseline_target: pd.DataFrame,
    baseline_content: pd.DataFrame,
) -> pd.DataFrame:
    profile = build_rank_margin_profile(item_profile, baseline_score, baseline_target, baseline_content)

    profile["category_evidence_score"] = _mean_existing(
        profile,
        ["s_cat_v2_disc_within_control", "s_cat_v2_collab_within_control", "s_cat"],
    )
    profile["cf_evidence_score"] = _mean_existing(
        profile,
        [
            "target_history_q_cosine_mean",
            "target_minus_top20_history_q_cosine_mean",
            "target_minus_top20_cosine_mean",
        ],
    )
    profile["category_evidence_bucket"] = _prefixed_bucket(profile["category_evidence_score"], "cat")
    profile["cf_evidence_bucket"] = _prefixed_bucket(profile["cf_evidence_score"], "cf")
    profile["category_count_bucket"] = profile["category_count"].map(category_count_bucket)
    profile["target_activity_bucket"] = _control_bucket(profile["target_history_interaction_count_mean"])

    control_cols = ["gt_group", "category_count_bucket", "target_activity_bucket"]
    profile["interaction_control_key"] = profile[control_cols].fillna("unknown").astype(str).agg("|".join, axis=1)
    for metric in ["ndcg@20", "hr@20"]:
        profile[f"{metric}_control_mean"] = profile.groupby("interaction_control_key", dropna=False)[metric].transform("mean")
        profile[f"{metric}_control_residual"] = profile[metric] - profile[f"{metric}_control_mean"]
    return profile.sort_values("asin").reset_index(drop=True)


def build_interaction_grid_summary(profile: pd.DataFrame) -> pd.DataFrame:
    required = {"category_evidence_bucket", "cf_evidence_bucket"}
    missing = sorted(required.difference(profile.columns))
    if missing:
        raise ValueError(f"profile 缺少必要列: {missing}")
    rows: list[dict[str, Any]] = []
    for (cat_bucket, cf_bucket), frame in profile.groupby(["category_evidence_bucket", "cf_evidence_bucket"], dropna=False):
        rows.append(
            {
                "category_evidence_bucket": cat_bucket,
                "cf_evidence_bucket": cf_bucket,
                "item_count": int(len(frame)),
                "ndcg@20_mean": _safe_mean(frame["ndcg@20"]),
                "hr@20_mean": _safe_mean(frame["hr@20"]),
                "ndcg@20_control_residual_mean": _safe_mean(frame["ndcg@20_control_residual"]),
                "hr@20_control_residual_mean": _safe_mean(frame["hr@20_control_residual"])
                if "hr@20_control_residual" in frame.columns
                else float("nan"),
                "margin_to_top20_cutoff_mean": _safe_mean(frame["margin_to_top20_cutoff"])
                if "margin_to_top20_cutoff" in frame.columns
                else float("nan"),
                "category_evidence_score_mean": _safe_mean(frame["category_evidence_score"])
                if "category_evidence_score" in frame.columns
                else float("nan"),
                "cf_evidence_score_mean": _safe_mean(frame["cf_evidence_score"])
                if "cf_evidence_score" in frame.columns
                else float("nan"),
            }
        )
    cat_order = {"cat_low": 0, "cat_mid": 1, "cat_high": 2, "cat_all": 3, "cat_unknown": 4}
    cf_order = {"cf_low": 0, "cf_mid": 1, "cf_high": 2, "cf_all": 3, "cf_unknown": 4}
    return (
        pd.DataFrame(rows)
        .assign(
            _cat_order=lambda df: df["category_evidence_bucket"].map(cat_order).fillna(99),
            _cf_order=lambda df: df["cf_evidence_bucket"].map(cf_order).fillna(99),
        )
        .sort_values(["_cf_order", "_cat_order"])
        .drop(columns=["_cat_order", "_cf_order"])
        .reset_index(drop=True)
    )


def _grid_value(grid: pd.DataFrame, cf_bucket: str, cat_bucket: str, column: str) -> float:
    rows = grid[grid["cf_evidence_bucket"].eq(cf_bucket) & grid["category_evidence_bucket"].eq(cat_bucket)]
    if rows.empty or column not in rows.columns:
        return float("nan")
    return float(rows.iloc[0][column])


def build_within_cf_category_effect(grid: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cf_bucket in sorted(grid["cf_evidence_bucket"].dropna().unique()):
        high_residual = _grid_value(grid, cf_bucket, "cat_high", "ndcg@20_control_residual_mean")
        low_residual = _grid_value(grid, cf_bucket, "cat_low", "ndcg@20_control_residual_mean")
        high_ndcg = _grid_value(grid, cf_bucket, "cat_high", "ndcg@20_mean")
        low_ndcg = _grid_value(grid, cf_bucket, "cat_low", "ndcg@20_mean")
        rows.append(
            {
                "cf_evidence_bucket": cf_bucket,
                "cat_high_ndcg@20_mean": high_ndcg,
                "cat_low_ndcg@20_mean": low_ndcg,
                "high_minus_low_cat_ndcg@20": high_ndcg - low_ndcg,
                "cat_high_residual_ndcg@20_mean": high_residual,
                "cat_low_residual_ndcg@20_mean": low_residual,
                "high_minus_low_cat_residual_ndcg@20": high_residual - low_residual,
                "cat_high_item_count": _grid_value(grid, cf_bucket, "cat_high", "item_count"),
                "cat_low_item_count": _grid_value(grid, cf_bucket, "cat_low", "item_count"),
            }
        )
    cf_order = {"cf_low": 0, "cf_mid": 1, "cf_high": 2, "cf_all": 3, "cf_unknown": 4}
    return (
        pd.DataFrame(rows)
        .assign(_order=lambda df: df["cf_evidence_bucket"].map(cf_order).fillna(99))
        .sort_values("_order")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )


def _effect_value(effect: pd.DataFrame, cf_bucket: str, column: str) -> float:
    rows = effect[effect["cf_evidence_bucket"].eq(cf_bucket)]
    if rows.empty or column not in rows.columns:
        return float("nan")
    return float(rows.iloc[0][column])


def build_route_decision(grid: pd.DataFrame, effect: pd.DataFrame) -> dict[str, Any]:
    cf_low_effect = _effect_value(effect, "cf_low", "high_minus_low_cat_residual_ndcg@20")
    cf_mid_effect = _effect_value(effect, "cf_mid", "high_minus_low_cat_residual_ndcg@20")
    cf_high_effect = _effect_value(effect, "cf_high", "high_minus_low_cat_residual_ndcg@20")
    cf_low_count = (
        _effect_value(effect, "cf_low", "cat_high_item_count") + _effect_value(effect, "cf_low", "cat_low_item_count")
    )
    cf_high_minus_low_raw = (
        grid[grid["cf_evidence_bucket"].eq("cf_high")]["ndcg@20_mean"].mean()
        - grid[grid["cf_evidence_bucket"].eq("cf_low")]["ndcg@20_mean"].mean()
    )

    if cf_low_effect > 0.02 and (pd.isna(cf_high_effect) or cf_low_effect > cf_high_effect + 0.02):
        route = "category_helps_when_cf_weak"
        reason = "CF weak 条件下高 category evidence 相比低 category evidence 有更高控制后 NDCG residual。"
    elif cf_low_effect < -0.02:
        route = "category_conflicts_with_cf"
        reason = "CF weak 条件下高 category evidence 反而对应更低控制后 NDCG residual，存在类别证据误导风险。"
    elif abs(cf_low_effect) < 0.02 and abs(cf_mid_effect) < 0.02 and abs(cf_high_effect) < 0.02 and cf_high_minus_low_raw > 0.02:
        route = "cf_dominates_category"
        reason = "category evidence 的 within-CF 效应弱，baseline failure 主要随 CF evidence 变化。"
    else:
        route = "no_interaction_signal"
        reason = "category evidence 与 CF evidence 的交互不足以形成稳定 Task4 入口。"

    evidence = {
        "cf_low_high_minus_low_cat_residual_ndcg@20": cf_low_effect,
        "cf_mid_high_minus_low_cat_residual_ndcg@20": cf_mid_effect,
        "cf_high_high_minus_low_cat_residual_ndcg@20": cf_high_effect,
        "cf_low_cat_low_plus_high_item_count": cf_low_count,
        "cf_high_minus_low_raw_ndcg@20": cf_high_minus_low_raw,
    }
    return {"route": route, "reason": reason, "evidence": {key: _jsonable(value) for key, value in evidence.items()}}


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    source_notes: list[str],
    result_dir_display: str,
    grid: pd.DataFrame,
    effect: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    source_lines = "\n".join(f"> {note}" for note in source_notes) or "> 上游设计：未提供"
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.11 category-CF interaction 诊断结果
date: {run_stamp[:10]}
time: "{run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
created_at: "{run_stamp[:10]} {run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - Task3.11
  - category_cf_interaction
---

# {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.11 category-CF interaction 诊断结果

## 来源说明

> [!info] 来源说明
{source_lines}
> 本结果目录：`{result_dir_display}`

## 结论

Task3.11 的 route decision 为 `{decision["route"]}`。

{decision["reason"]}

本结果只判断 category evidence 与 CF evidence 的交互是否能解释 baseline failure，不训练模型，也不直接进入 Task4。

## 数据验收

> [!info] 字段说明
> `field`：验收项。
> `value`：验收项取值。

| field | value |
|---|---:|
| interaction_cell_count | {len(grid)} |
| route | `{decision["route"]}` |
| cf_low_category_effect | {decision["evidence"].get("cf_low_high_minus_low_cat_residual_ndcg@20")} |

## interaction grid

> [!info] 字段说明
> `category_evidence_bucket`：category availability evidence 分桶。
> `cf_evidence_bucket`：CF evidence 分桶。
> `ndcg@20_control_residual_mean`：控制 gt_group、category_count、target activity 后的 NDCG residual。
> `margin_to_top20_cutoff_mean`：baseline target score 相对 top20 cutoff margin。

{md_table(grid, ["cf_evidence_bucket", "category_evidence_bucket", "item_count", "ndcg@20_mean", "ndcg@20_control_residual_mean", "margin_to_top20_cutoff_mean", "category_evidence_score_mean", "cf_evidence_score_mean"], max_rows=18)}

## within-CF category effect

> [!info] 字段说明
> `cf_evidence_bucket`：CF evidence 分桶。
> `high_minus_low_cat_residual_ndcg@20`：同一 CF 分桶内 cat_high 减 cat_low 的控制后 NDCG residual。
> `cat_high_item_count`：cat_high cell 样本数。
> `cat_low_item_count`：cat_low cell 样本数。

{md_table(effect, ["cf_evidence_bucket", "high_minus_low_cat_ndcg@20", "high_minus_low_cat_residual_ndcg@20", "cat_high_item_count", "cat_low_item_count"])}

## 判断

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

## 下一步

如果 category-CF interaction 有效，后续 Task4 carrier 才考虑 conditional fusion / availability-aware gate；若无效，继续 Task3.15 做 score calibration / norm-control。

## 产物

```text
task3p11_category_cf_interaction_profile.csv
task3p11_interaction_grid_summary.csv
task3p11_within_control_interaction_effect.csv
task3p11_route_decision.json
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

    profile = build_category_cf_interaction_profile(item_profile, baseline_score, baseline_target, baseline_content)
    grid = build_interaction_grid_summary(profile)
    effect = build_within_cf_category_effect(grid)
    decision = build_route_decision(grid, effect)

    outputs = {
        "task3p11_category_cf_interaction_profile": output_dir / "task3p11_category_cf_interaction_profile.csv",
        "task3p11_interaction_grid_summary": output_dir / "task3p11_interaction_grid_summary.csv",
        "task3p11_within_control_interaction_effect": output_dir / "task3p11_within_control_interaction_effect.csv",
        "task3p11_route_decision": output_dir / "task3p11_route_decision.json",
    }
    profile.to_csv(outputs["task3p11_category_cf_interaction_profile"], index=False)
    grid.to_csv(outputs["task3p11_interaction_grid_summary"], index=False)
    effect.to_csv(outputs["task3p11_within_control_interaction_effect"], index=False)
    outputs["task3p11_route_decision"].write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    result_md = output_dir / f"{run_stamp} CCFCRec Amazon-VG category availability v2 Task3.11 category-CF interaction 诊断结果.md"
    manifest_path = output_dir / "run_manifest.json"
    result_dir_display = args.result_dir_display or str(output_dir)
    write_result_markdown(
        output_path=result_md,
        run_stamp=run_stamp,
        source_notes=args.source_note,
        result_dir_display=result_dir_display,
        grid=grid,
        effect=effect,
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
            "task3p11_category_cf_interaction_profile": int(len(profile)),
            "task3p11_interaction_grid_summary": int(len(grid)),
            "task3p11_within_control_interaction_effect": int(len(effect)),
        },
        "route_decision": decision,
        "notes": [
            "只复用 Task3 baseline item-level CSV，不训练模型。",
            "Task3.11 判断 category evidence 与 CF evidence 的交互是否能解释 baseline failure。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG category availability v2 Task3.11 category-CF interaction 诊断")
    parser.add_argument("--task3-dir", required=True, type=str, help="Task3 checkpoint 复评输出目录")
    parser.add_argument("--output-dir", required=True, type=str, help="Task3.11 输出目录")
    parser.add_argument("--source-note", action="append", default=[], help="写入结果 MD 来源说明的 Obsidian wikilink 行")
    parser.add_argument("--result-dir-display", type=str, default="", help="写入结果 MD 的相对结果目录")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
