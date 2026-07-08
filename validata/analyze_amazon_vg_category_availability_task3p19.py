#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.19 Task4 proxy metric 诊断。

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
from analyze_amazon_vg_category_availability_task3p10 import _control_bucket


PROXY_METRICS = [
    "margin_proxy",
    "target_competitor_gap_proxy",
    "modality_alignment_proxy",
    "calibration_proxy",
]


def now_stamp() -> tuple[str, str]:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def _mean_existing(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(float("nan"), index=frame.index)
    numeric = pd.concat([pd.to_numeric(frame[column], errors="coerce") for column in available], axis=1)
    return numeric.mean(axis=1)


def _corr(x: pd.Series, y: pd.Series, method: str) -> float:
    valid = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))


def build_proxy_metric_profile(
    item_profile: pd.DataFrame,
    baseline_score: pd.DataFrame,
    baseline_target: pd.DataFrame,
    baseline_content: pd.DataFrame,
) -> pd.DataFrame:
    profile = build_rank_margin_profile(item_profile, baseline_score, baseline_target, baseline_content)
    profile["target_activity_bucket"] = _control_bucket(profile["target_history_interaction_count_mean"])
    profile["margin_proxy"] = pd.to_numeric(profile["margin_to_top20_cutoff"], errors="coerce")
    profile["target_competitor_gap_proxy"] = _mean_existing(
        profile,
        ["target_minus_top20_cosine_mean", "target_minus_top20_user_norm_mean"],
    )
    profile["modality_alignment_proxy"] = _mean_existing(
        profile,
        [
            "target_minus_top20_history_q_cosine_mean",
            "target_minus_top20_history_attr_cosine_mean",
            "target_minus_top20_history_img_cosine_mean",
        ],
    )
    profile["calibration_proxy"] = -pd.to_numeric(profile["hard_negative_pressure_score"], errors="coerce")
    return profile.sort_values("asin").reset_index(drop=True)


def build_proxy_vs_rank_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for proxy in PROXY_METRICS:
        if proxy not in profile.columns:
            continue
        rows.append(
            {
                "proxy_metric": proxy,
                "n": int(pd.to_numeric(profile[proxy], errors="coerce").notna().sum()),
                "spearman_vs_ndcg@20": _corr(profile[proxy], profile["ndcg@20"], "spearman"),
                "pearson_vs_ndcg@20": _corr(profile[proxy], profile["ndcg@20"], "pearson"),
                "spearman_vs_best_target_rank": _corr(profile[proxy], profile["best_target_rank"], "spearman")
                if "best_target_rank" in profile.columns
                else float("nan"),
                "spearman_vs_margin_to_top20_cutoff": _corr(
                    profile[proxy],
                    profile["margin_to_top20_cutoff"],
                    "spearman",
                )
                if "margin_to_top20_cutoff" in profile.columns
                else float("nan"),
                "spearman_vs_q_norm": _corr(profile[proxy], profile["q_norm"], "spearman")
                if "q_norm" in profile.columns
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_proxy_controlled_stability(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    for column in ["gt_group", "target_activity_bucket"]:
        if column not in frame.columns:
            frame[column] = "unknown"
    control_cols = ["gt_group", "target_activity_bucket"]
    frame["ndcg@20_control_mean"] = frame.groupby(control_cols, dropna=False)["ndcg@20"].transform("mean")
    frame["ndcg@20_control_residual"] = frame["ndcg@20"] - frame["ndcg@20_control_mean"]

    rows: list[dict[str, Any]] = []
    for proxy in PROXY_METRICS:
        if proxy not in frame.columns:
            continue
        bucket_col = f"{proxy}_bucket"
        frame[bucket_col] = _quantile_labels(frame[proxy])
        bucket_rows: dict[str, dict[str, Any]] = {}
        for bucket, bucket_frame in frame.groupby(bucket_col, dropna=False):
            bucket_rows[str(bucket)] = {
                "item_count": int(len(bucket_frame)),
                "ndcg@20_mean": _safe_mean(bucket_frame["ndcg@20"]),
                "ndcg@20_control_residual_mean": _safe_mean(bucket_frame["ndcg@20_control_residual"]),
                "proxy_mean": _safe_mean(bucket_frame[proxy]),
            }
        low = bucket_rows.get("low", {})
        high = bucket_rows.get("high", {})
        rows.append(
            {
                "proxy_metric": proxy,
                "low_item_count": low.get("item_count"),
                "high_item_count": high.get("item_count"),
                "low_ndcg@20_mean": low.get("ndcg@20_mean"),
                "high_ndcg@20_mean": high.get("ndcg@20_mean"),
                "high_minus_low_ndcg@20": (high.get("ndcg@20_mean", float("nan")) - low.get("ndcg@20_mean", float("nan"))),
                "low_residual_ndcg@20_mean": low.get("ndcg@20_control_residual_mean"),
                "high_residual_ndcg@20_mean": high.get("ndcg@20_control_residual_mean"),
                "high_minus_low_residual_ndcg@20": (
                    high.get("ndcg@20_control_residual_mean", float("nan"))
                    - low.get("ndcg@20_control_residual_mean", float("nan"))
                ),
            }
        )
    return pd.DataFrame(rows)


def _row_for(frame: pd.DataFrame, proxy: str) -> pd.Series | None:
    rows = frame[frame["proxy_metric"].eq(proxy)]
    if rows.empty:
        return None
    return rows.iloc[0]


def build_route_decision(proxy_summary: pd.DataFrame, stability: pd.DataFrame) -> dict[str, Any]:
    def is_good(proxy: str, ndcg_threshold: float = 0.5, stability_threshold: float = 0.05) -> bool:
        summary = _row_for(proxy_summary, proxy)
        stable = _row_for(stability, proxy)
        if summary is None or stable is None:
            return False
        return (
            float(summary["spearman_vs_ndcg@20"]) >= ndcg_threshold
            and float(summary["spearman_vs_best_target_rank"]) <= -ndcg_threshold
            and float(stable["high_minus_low_residual_ndcg@20"]) >= stability_threshold
        )

    if is_good("margin_proxy", ndcg_threshold=0.6, stability_threshold=0.05):
        route = "margin_proxy_recommended"
        proxy = "margin_proxy"
        reason = "margin proxy 与 baseline NDCG / target rank 关系强，且控制后 high-low residual gap 稳定。"
    elif is_good("target_competitor_gap_proxy"):
        route = "target_competitor_gap_proxy_recommended"
        proxy = "target_competitor_gap_proxy"
        reason = "target-vs-top20 gap proxy 与排序失败关系稳定。"
    elif is_good("modality_alignment_proxy"):
        route = "modality_alignment_proxy_recommended"
        proxy = "modality_alignment_proxy"
        reason = "modality alignment proxy 与排序失败关系稳定。"
    elif is_good("calibration_proxy"):
        route = "calibration_proxy_recommended"
        proxy = "calibration_proxy"
        reason = "calibration proxy 与排序失败关系稳定。"
    else:
        route = "no_proxy_ready_for_task4"
        proxy = ""
        reason = "当前 proxy 与排序失败或控制后 residual 的关系不足以支撑 Task4。"

    summary = _row_for(proxy_summary, proxy) if proxy else None
    stable = _row_for(stability, proxy) if proxy else None
    evidence = {
        "recommended_proxy": proxy,
        "spearman_vs_ndcg@20": float(summary["spearman_vs_ndcg@20"]) if summary is not None else None,
        "spearman_vs_best_target_rank": float(summary["spearman_vs_best_target_rank"]) if summary is not None else None,
        "spearman_vs_margin_to_top20_cutoff": float(summary["spearman_vs_margin_to_top20_cutoff"])
        if summary is not None
        else None,
        "high_minus_low_residual_ndcg@20": float(stable["high_minus_low_residual_ndcg@20"]) if stable is not None else None,
    }
    return {"route": route, "reason": reason, "evidence": {key: _jsonable(value) for key, value in evidence.items()}}


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    source_notes: list[str],
    result_dir_display: str,
    proxy_summary: pd.DataFrame,
    stability: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    source_lines = "\n".join(f"> {note}" for note in source_notes) or "> 上游设计：未提供"
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.19 proxy metric 诊断结果
date: {run_stamp[:10]}
time: "{run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
created_at: "{run_stamp[:10]} {run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - Task3.19
  - proxy_metric
---

# {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.19 proxy metric 诊断结果

## 来源说明

> [!info] 来源说明
{source_lines}
> 本结果目录：`{result_dir_display}`

## 结论

Task3.19 的 route decision 为 `{decision["route"]}`。

{decision["reason"]}

本结果只选择 Task4 可训练 proxy 候选，不训练模型，也不直接进入 Task4。

## 数据验收

> [!info] 字段说明
> `field`：验收项。
> `value`：验收项取值。

| field | value |
|---|---:|
| proxy_count | {len(proxy_summary)} |
| route | `{decision["route"]}` |
| recommended_proxy | `{decision["evidence"].get("recommended_proxy")}` |

## proxy vs rank summary

> [!info] 字段说明
> `proxy_metric`：候选 proxy。
> `spearman_vs_ndcg@20`：proxy 与 baseline NDCG@20 的 Spearman。
> `spearman_vs_best_target_rank`：proxy 与最佳目标用户 rank 的 Spearman，负值表示 proxy 越高 rank 越靠前。
> `spearman_vs_q_norm`：proxy 与 q_norm 的 Spearman，用于检查 norm 绑定风险。

{md_table(proxy_summary, ["proxy_metric", "n", "spearman_vs_ndcg@20", "pearson_vs_ndcg@20", "spearman_vs_best_target_rank", "spearman_vs_margin_to_top20_cutoff", "spearman_vs_q_norm"])}

## proxy controlled stability

> [!info] 字段说明
> `proxy_metric`：候选 proxy。
> `high_minus_low_residual_ndcg@20`：控制 gt_group 与 target activity 后，高 proxy 桶减低 proxy 桶的 NDCG residual。
> `high_minus_low_ndcg@20`：高 proxy 桶减低 proxy 桶的原始 NDCG。

{md_table(stability, ["proxy_metric", "low_item_count", "high_item_count", "high_minus_low_ndcg@20", "high_minus_low_residual_ndcg@20"])}

## 判断

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

## 下一步

如果 `margin_proxy_recommended` 成立，Task4 的候选载体应优先考虑 rank-margin / pairwise margin，并结合 Task3.15 的 norm-control 约束设计负控。

## 产物

```text
task3p19_proxy_metric_profile.csv
task3p19_proxy_vs_rank_summary.csv
task3p19_proxy_controlled_stability.csv
task3p19_route_decision.json
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

    profile = build_proxy_metric_profile(item_profile, baseline_score, baseline_target, baseline_content)
    proxy_summary = build_proxy_vs_rank_summary(profile)
    stability = build_proxy_controlled_stability(profile)
    decision = build_route_decision(proxy_summary, stability)

    outputs = {
        "task3p19_proxy_metric_profile": output_dir / "task3p19_proxy_metric_profile.csv",
        "task3p19_proxy_vs_rank_summary": output_dir / "task3p19_proxy_vs_rank_summary.csv",
        "task3p19_proxy_controlled_stability": output_dir / "task3p19_proxy_controlled_stability.csv",
        "task3p19_route_decision": output_dir / "task3p19_route_decision.json",
    }
    profile.to_csv(outputs["task3p19_proxy_metric_profile"], index=False)
    proxy_summary.to_csv(outputs["task3p19_proxy_vs_rank_summary"], index=False)
    stability.to_csv(outputs["task3p19_proxy_controlled_stability"], index=False)
    outputs["task3p19_route_decision"].write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    result_md = output_dir / f"{run_stamp} CCFCRec Amazon-VG category availability v2 Task3.19 proxy metric 诊断结果.md"
    manifest_path = output_dir / "run_manifest.json"
    result_dir_display = args.result_dir_display or str(output_dir)
    write_result_markdown(
        output_path=result_md,
        run_stamp=run_stamp,
        source_notes=args.source_note,
        result_dir_display=result_dir_display,
        proxy_summary=proxy_summary,
        stability=stability,
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
            "task3p19_proxy_metric_profile": int(len(profile)),
            "task3p19_proxy_vs_rank_summary": int(len(proxy_summary)),
            "task3p19_proxy_controlled_stability": int(len(stability)),
        },
        "route_decision": decision,
        "notes": [
            "只复用 Task3 baseline item-level CSV，不训练模型。",
            "Task3.19 判断哪个离线 proxy 适合承接 Task4 可训练目标。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG category availability v2 Task3.19 proxy metric 诊断")
    parser.add_argument("--task3-dir", required=True, type=str, help="Task3 checkpoint 复评输出目录")
    parser.add_argument("--output-dir", required=True, type=str, help="Task3.19 输出目录")
    parser.add_argument("--source-note", action="append", default=[], help="写入结果 MD 来源说明的 Obsidian wikilink 行")
    parser.add_argument("--result-dir-display", type=str, default="", help="写入结果 MD 的相对结果目录")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
