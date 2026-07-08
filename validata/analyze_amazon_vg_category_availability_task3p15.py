#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.15 score calibration / norm-control 诊断。

脚本复用 Task3.5 item-level delta profile，判断 category_conf_input 的变化是否主要是 norm/scale artifact。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_amazon_vg_category_availability_task3p5 import md_table
from analyze_amazon_vg_category_availability_task3p8 import _jsonable, _quantile_labels, _safe_mean


DIRECTION_COLUMNS = [
    "delta_target_cosine_at_score_max",
    "delta_target_history_q_cosine_mean",
    "delta_target_minus_top20_history_q_cosine_mean",
    "delta_target_minus_top20_cosine_mean",
]


def now_stamp() -> tuple[str, str]:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def build_calibration_profile(delta_profile: pd.DataFrame) -> pd.DataFrame:
    if "asin" not in delta_profile.columns:
        raise ValueError("delta_profile 必须包含 asin 列")
    profile = delta_profile.copy()
    for column in [
        "delta_ndcg@20",
        "delta_hr@20",
        "delta_q_norm",
        "delta_top20_user_norm_mean",
        "delta_target_user_norm_at_score_max",
        "delta_margin_to_top20_cutoff",
        *DIRECTION_COLUMNS,
    ]:
        if column not in profile.columns:
            profile[column] = 0.0
        profile[column] = pd.to_numeric(profile[column], errors="coerce").fillna(0.0)

    profile["norm_shift_score"] = profile["delta_q_norm"].abs()
    direction = pd.Series(0.0, index=profile.index)
    for column in DIRECTION_COLUMNS:
        direction = direction + profile[column].abs()
    profile["direction_shift_score"] = direction
    profile["norm_direction_ratio"] = profile["norm_shift_score"] / profile["direction_shift_score"].clip(lower=1e-12)
    profile["norm_dominant"] = profile["norm_direction_ratio"] > 10.0
    profile["norm_shift_bucket"] = _quantile_labels(profile["norm_shift_score"])
    profile["top20_norm_shift_bucket"] = _quantile_labels(profile["delta_top20_user_norm_mean"])
    profile["margin_delta_bucket"] = _quantile_labels(profile["delta_margin_to_top20_cutoff"])
    return profile.sort_values("asin").reset_index(drop=True)


def build_norm_control_summary(profile: pd.DataFrame) -> pd.DataFrame:
    if "norm_shift_bucket" not in profile.columns:
        raise ValueError("profile 必须包含 norm_shift_bucket 列")
    if "delta_hr@20" not in profile.columns:
        profile = profile.copy()
        profile["delta_hr@20"] = 0.0
    rows: list[dict[str, Any]] = []
    for bucket, frame in profile.groupby("norm_shift_bucket", dropna=False):
        rows.append(
            {
                "norm_shift_bucket": bucket,
                "item_count": int(len(frame)),
                "delta_ndcg@20_mean": _safe_mean(frame["delta_ndcg@20"]),
                "delta_hr@20_mean": _safe_mean(frame["delta_hr@20"]),
                "delta_margin_to_top20_cutoff_mean": _safe_mean(frame["delta_margin_to_top20_cutoff"]),
                "delta_top20_user_norm_mean_mean": _safe_mean(frame["delta_top20_user_norm_mean"]),
                "norm_shift_score_mean": _safe_mean(frame["norm_shift_score"]),
                "direction_shift_score_mean": _safe_mean(frame["direction_shift_score"]),
                "norm_direction_ratio_mean": _safe_mean(frame["norm_direction_ratio"]),
                "norm_dominant_rate": float(frame["norm_dominant"].mean()),
            }
        )
    order = {"low": 0, "mid": 1, "high": 2, "all": 3, "unknown": 4}
    return (
        pd.DataFrame(rows)
        .assign(_order=lambda df: df["norm_shift_bucket"].map(order).fillna(99))
        .sort_values("_order")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )


def build_scale_direction_summary(profile: pd.DataFrame) -> pd.DataFrame:
    if "delta_hr@20" not in profile.columns:
        profile = profile.copy()
        profile["delta_hr@20"] = 0.0
    rows: list[dict[str, Any]] = []
    scopes = [("overall", profile)]
    if "category_group" in profile.columns:
        scopes.extend((str(group), frame) for group, frame in profile.groupby("category_group", dropna=False))
    for scope, frame in scopes:
        rows.append(
            {
                "scope": scope,
                "item_count": int(len(frame)),
                "delta_ndcg@20_mean": _safe_mean(frame["delta_ndcg@20"]),
                "delta_hr@20_mean": _safe_mean(frame["delta_hr@20"]),
                "delta_margin_to_top20_cutoff_mean": _safe_mean(frame["delta_margin_to_top20_cutoff"]),
                "delta_top20_user_norm_mean_mean": _safe_mean(frame["delta_top20_user_norm_mean"]),
                "norm_shift_score_mean": _safe_mean(frame["norm_shift_score"]),
                "direction_shift_score_mean": _safe_mean(frame["direction_shift_score"]),
                "norm_direction_ratio_mean": _safe_mean(frame["norm_direction_ratio"]),
                "norm_dominant_rate": float(frame["norm_dominant"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _summary_value(frame: pd.DataFrame, key_col: str, key_value: str, column: str) -> float:
    rows = frame[frame[key_col].eq(key_value)]
    if rows.empty or column not in rows.columns:
        return float("nan")
    return float(rows.iloc[0][column])


def build_route_decision(norm_summary: pd.DataFrame, scale_summary: pd.DataFrame) -> dict[str, Any]:
    overall_norm_rate = _summary_value(scale_summary, "scope", "overall", "norm_dominant_rate")
    overall_ratio = _summary_value(scale_summary, "scope", "overall", "norm_direction_ratio_mean")
    overall_delta = _summary_value(scale_summary, "scope", "overall", "delta_ndcg@20_mean")
    high_delta = _summary_value(norm_summary, "norm_shift_bucket", "high", "delta_ndcg@20_mean")
    high_margin = _summary_value(norm_summary, "norm_shift_bucket", "high", "delta_margin_to_top20_cutoff_mean")
    high_top20_norm = _summary_value(norm_summary, "norm_shift_bucket", "high", "delta_top20_user_norm_mean_mean")
    high_norm_rate = _summary_value(norm_summary, "norm_shift_bucket", "high", "norm_dominant_rate")
    high_ratio = _summary_value(norm_summary, "norm_shift_bucket", "high", "norm_direction_ratio_mean")
    low_delta = _summary_value(norm_summary, "norm_shift_bucket", "low", "delta_ndcg@20_mean")

    if (overall_norm_rate >= 0.75 and overall_ratio >= 10 and high_delta <= 0) or (
        high_norm_rate >= 0.75 and high_ratio >= 10 and high_delta <= 0
    ):
        route = "signal_is_scale_artifact"
        reason = "表示变化主要由 q_norm / scale shift 主导，高 norm shift 桶没有排序收益。"
    elif high_top20_norm > 0 and high_delta <= 0:
        route = "competitor_norm_calibration_needed"
        reason = "top20 competitor norm 上升且高 norm shift 桶排序不增，Task4 需要 competitor norm calibration。"
    elif low_delta > 0 and high_margin > 0:
        route = "norm_control_preserves_signal"
        reason = "低 norm shift 下仍保留正向排序或 margin 信号，可考虑 norm-controlled 方法。"
    else:
        route = "target_norm_calibration_needed"
        reason = "norm-control 后信号不清晰，需进一步区分 target norm 与 competitor norm calibration。"

    evidence = {
        "overall_norm_dominant_rate": overall_norm_rate,
        "overall_norm_direction_ratio_mean": overall_ratio,
        "overall_delta_ndcg@20_mean": overall_delta,
        "high_norm_bucket_delta_ndcg@20_mean": high_delta,
        "low_norm_bucket_delta_ndcg@20_mean": low_delta,
        "high_norm_bucket_delta_margin_to_top20_cutoff_mean": high_margin,
        "high_norm_bucket_delta_top20_user_norm_mean_mean": high_top20_norm,
        "high_norm_bucket_norm_dominant_rate": high_norm_rate,
        "high_norm_bucket_norm_direction_ratio_mean": high_ratio,
    }
    return {"route": route, "reason": reason, "evidence": {key: _jsonable(value) for key, value in evidence.items()}}


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    source_notes: list[str],
    result_dir_display: str,
    norm_summary: pd.DataFrame,
    scale_summary: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    source_lines = "\n".join(f"> {note}" for note in source_notes) or "> 上游设计：未提供"
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.15 score calibration 诊断结果
date: {run_stamp[:10]}
time: "{run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
created_at: "{run_stamp[:10]} {run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - Task3.15
  - score_calibration
---

# {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.15 score calibration 诊断结果

## 来源说明

> [!info] 来源说明
{source_lines}
> 本结果目录：`{result_dir_display}`

## 结论

Task3.15 的 route decision 为 `{decision["route"]}`。

{decision["reason"]}

本结果只判断 category_conf_input 的变化是否主要是 norm/scale artifact，不训练模型，也不直接进入 Task4。

## 数据验收

> [!info] 字段说明
> `field`：验收项。
> `value`：验收项取值。

| field | value |
|---|---:|
| norm_bucket_count | {len(norm_summary)} |
| route | `{decision["route"]}` |
| overall_norm_dominant_rate | {decision["evidence"].get("overall_norm_dominant_rate")} |
| high_norm_bucket_delta_ndcg@20_mean | {decision["evidence"].get("high_norm_bucket_delta_ndcg@20_mean")} |

## norm-control bucket summary

> [!info] 字段说明
> `norm_shift_bucket`：按 `abs(delta_q_norm)` 分桶。
> `delta_ndcg@20_mean`：category_conf_input 相对 baseline 的 NDCG@20 变化。
> `delta_margin_to_top20_cutoff_mean`：target score 相对 top20 cutoff margin 变化。
> `norm_dominant_rate`：norm shift 大于 direction shift 的 item 比例。

{md_table(norm_summary, ["norm_shift_bucket", "item_count", "delta_ndcg@20_mean", "delta_hr@20_mean", "delta_margin_to_top20_cutoff_mean", "delta_top20_user_norm_mean_mean", "norm_shift_score_mean", "direction_shift_score_mean", "norm_direction_ratio_mean", "norm_dominant_rate"])}

## scale-vs-direction summary

> [!info] 字段说明
> `scope`：统计范围。
> `norm_shift_score_mean`：`abs(delta_q_norm)` 均值。
> `direction_shift_score_mean`：direction / alignment 变化绝对值和均值。
> `norm_direction_ratio_mean`：norm shift 与 direction shift 比值均值。

{md_table(scale_summary, ["scope", "item_count", "delta_ndcg@20_mean", "delta_margin_to_top20_cutoff_mean", "delta_top20_user_norm_mean_mean", "norm_shift_score_mean", "direction_shift_score_mean", "norm_direction_ratio_mean", "norm_dominant_rate"], max_rows=10)}

## 判断

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

## 下一步

如果 `signal_is_scale_artifact` 成立，Task4 不能继续扩展 category_conf_input；后续应优先寻找 margin / proxy 指标，并在方法设计中加入 norm-control 或 calibration 负控。

## 产物

```text
task3p15_calibration_profile.csv
task3p15_norm_control_bucket_summary.csv
task3p15_scale_vs_direction_summary.csv
task3p15_route_decision.json
run_manifest.json
```
"""
    output_path.write_text(markdown, encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    task3p5_dir = Path(args.task3p5_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_stamp, run_iso = now_stamp()

    delta_profile = pd.read_csv(task3p5_dir / "task3p5_item_delta_profile.csv")
    profile = build_calibration_profile(delta_profile)
    norm_summary = build_norm_control_summary(profile)
    scale_summary = build_scale_direction_summary(profile)
    decision = build_route_decision(norm_summary, scale_summary)

    outputs = {
        "task3p15_calibration_profile": output_dir / "task3p15_calibration_profile.csv",
        "task3p15_norm_control_bucket_summary": output_dir / "task3p15_norm_control_bucket_summary.csv",
        "task3p15_scale_vs_direction_summary": output_dir / "task3p15_scale_vs_direction_summary.csv",
        "task3p15_route_decision": output_dir / "task3p15_route_decision.json",
    }
    profile.to_csv(outputs["task3p15_calibration_profile"], index=False)
    norm_summary.to_csv(outputs["task3p15_norm_control_bucket_summary"], index=False)
    scale_summary.to_csv(outputs["task3p15_scale_vs_direction_summary"], index=False)
    outputs["task3p15_route_decision"].write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")

    result_md = output_dir / f"{run_stamp} CCFCRec Amazon-VG category availability v2 Task3.15 score calibration 诊断结果.md"
    manifest_path = output_dir / "run_manifest.json"
    result_dir_display = args.result_dir_display or str(output_dir)
    write_result_markdown(
        output_path=result_md,
        run_stamp=run_stamp,
        source_notes=args.source_note,
        result_dir_display=result_dir_display,
        norm_summary=norm_summary,
        scale_summary=scale_summary,
        decision=decision,
    )

    manifest = {
        "run_time": run_iso,
        "script": str(Path(__file__).resolve()),
        "task3p5_dir": str(task3p5_dir),
        "output_dir": str(output_dir),
        "inputs": {
            "task3p5_item_delta_profile": str(task3p5_dir / "task3p5_item_delta_profile.csv"),
        },
        "outputs": {key: str(value) for key, value in outputs.items()} | {
            "result_md": str(result_md),
            "run_manifest": str(manifest_path),
        },
        "row_counts": {
            "task3p15_calibration_profile": int(len(profile)),
            "task3p15_norm_control_bucket_summary": int(len(norm_summary)),
            "task3p15_scale_vs_direction_summary": int(len(scale_summary)),
        },
        "route_decision": decision,
        "notes": [
            "只复用 Task3.5 item-level delta profile，不训练模型。",
            "Task3.15 判断 category_conf_input 的变化是否主要是 norm/scale artifact。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG category availability v2 Task3.15 score calibration 诊断")
    parser.add_argument("--task3p5-dir", required=True, type=str, help="Task3.5 输出目录")
    parser.add_argument("--output-dir", required=True, type=str, help="Task3.15 输出目录")
    parser.add_argument("--source-note", action="append", default=[], help="写入结果 MD 来源说明的 Obsidian wikilink 行")
    parser.add_argument("--result-dir-display", type=str, default="", help="写入结果 MD 的相对结果目录")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
