#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.7 方法信号匹配诊断。

脚本复用 Task3.5 item-level delta profile，判断 category_conf_input 的改变是否匹配 v2 category availability 机制。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from analyze_amazon_vg_category_availability_task3p5 import md_table


V2_COMPONENTS = [
    "s_cat",
    "s_cat_v1",
    "s_cat_v2_disc_within_control",
    "s_cat_v2_collab_within_control",
]

METHOD_DELTAS = [
    "delta_ndcg@20",
    "delta_hr@20",
    "delta_q_norm",
    "delta_target_score_max",
    "delta_margin_to_top20_cutoff",
    "delta_target_cosine_at_score_max",
    "delta_target_history_q_cosine_mean",
    "delta_target_minus_top20_history_q_cosine_mean",
    "delta_top20_user_norm_mean",
    "delta_top20_history_interaction_count_mean",
]

DIRECTION_COLUMNS = [
    "delta_target_cosine_at_score_max",
    "delta_target_history_q_cosine_mean",
    "delta_target_minus_top20_history_q_cosine_mean",
    "delta_target_minus_top20_cosine_mean",
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


def build_method_signal_profile(delta_profile: pd.DataFrame) -> pd.DataFrame:
    if "asin" not in delta_profile.columns or "category_group" not in delta_profile.columns:
        raise ValueError("delta_profile 必须包含 asin 和 category_group 列")
    profile = delta_profile.copy()
    for column in ["delta_q_norm", *DIRECTION_COLUMNS]:
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
    return profile.sort_values("asin").reset_index(drop=True)


def build_representation_shift_by_v2_group(profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, frame in profile.groupby("category_group", dropna=False):
        row: dict[str, Any] = {
            "category_group": group,
            "item_count": int(len(frame)),
            "norm_dominant_rate": float(frame["norm_dominant"].mean()),
            "norm_direction_ratio_mean": _safe_mean(frame["norm_direction_ratio"]),
            "norm_shift_score_mean": _safe_mean(frame["norm_shift_score"]),
            "direction_shift_score_mean": _safe_mean(frame["direction_shift_score"]),
        }
        for metric in METHOD_DELTAS:
            if metric in frame.columns:
                row[f"{metric}_mean"] = _safe_mean(frame[metric])
        rows.append(row)
    return pd.DataFrame(rows).sort_values("category_group").reset_index(drop=True)


def build_norm_vs_direction_shift(profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes = [("overall", profile)]
    scopes.extend((str(group), frame) for group, frame in profile.groupby("category_group", dropna=False))
    for scope, frame in scopes:
        norm_mean = _safe_mean(frame["norm_shift_score"])
        direction_mean = _safe_mean(frame["direction_shift_score"])
        rows.append(
            {
                "scope": scope,
                "item_count": int(len(frame)),
                "norm_shift_score_mean": norm_mean,
                "direction_shift_score_mean": direction_mean,
                "norm_direction_ratio_mean": norm_mean / max(direction_mean, 1e-12),
                "norm_dominant_rate": float(frame["norm_dominant"].mean()),
                "delta_ndcg@20_mean": _safe_mean(frame["delta_ndcg@20"]) if "delta_ndcg@20" in frame.columns else float("nan"),
                "delta_margin_to_top20_cutoff_mean": _safe_mean(frame["delta_margin_to_top20_cutoff"])
                if "delta_margin_to_top20_cutoff" in frame.columns
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_component_vs_method_delta(profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for component in V2_COMPONENTS:
        if component not in profile.columns:
            continue
        x = pd.to_numeric(profile[component], errors="coerce")
        for method_delta in METHOD_DELTAS:
            if method_delta not in profile.columns:
                continue
            y = pd.to_numeric(profile[method_delta], errors="coerce")
            valid = pd.concat([x, y], axis=1).dropna()
            if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
                pearson = float("nan")
                spearman = float("nan")
            else:
                pearson = float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="pearson"))
                spearman = float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman"))
            rows.append(
                {
                    "v2_component": component,
                    "method_delta": method_delta,
                    "n": int(len(valid)),
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
    return pd.DataFrame(rows)


def build_method_signal_match_summary(
    representation_shift: pd.DataFrame,
    norm_shift: pd.DataFrame,
    component_delta: pd.DataFrame,
) -> pd.DataFrame:
    weak = representation_shift[representation_shift["category_group"].astype(str).str.contains("weak", na=False)]
    strong = representation_shift[representation_shift["category_group"].astype(str).str.contains("strong", na=False)]
    overall_norm = norm_shift[norm_shift["scope"].eq("overall")]
    ndcg_corr = component_delta[component_delta["method_delta"].eq("delta_ndcg@20")].copy()
    max_abs_ndcg_corr = float(ndcg_corr["spearman"].abs().max()) if not ndcg_corr.empty else float("nan")
    rows = [
        {
            "field": "weak_delta_ndcg@20_mean",
            "value": float(weak.iloc[0].get("delta_ndcg@20_mean", float("nan"))) if not weak.empty else float("nan"),
        },
        {
            "field": "strong_delta_ndcg@20_mean",
            "value": float(strong.iloc[0].get("delta_ndcg@20_mean", float("nan"))) if not strong.empty else float("nan"),
        },
        {
            "field": "overall_norm_direction_ratio_mean",
            "value": float(overall_norm.iloc[0].get("norm_direction_ratio_mean", float("nan"))) if not overall_norm.empty else float("nan"),
        },
        {
            "field": "overall_norm_dominant_rate",
            "value": float(overall_norm.iloc[0].get("norm_dominant_rate", float("nan"))) if not overall_norm.empty else float("nan"),
        },
        {"field": "max_abs_spearman_v2_component_vs_delta_ndcg@20", "value": max_abs_ndcg_corr},
    ]
    return pd.DataFrame(rows)


def build_method_fit_decision(
    representation_shift: pd.DataFrame,
    norm_shift: pd.DataFrame,
    component_delta: pd.DataFrame,
) -> dict[str, Any]:
    overall = norm_shift[norm_shift["scope"].eq("overall")]
    overall_ratio = float(overall.iloc[0]["norm_direction_ratio_mean"]) if not overall.empty else float("nan")
    overall_norm_rate = float(overall.iloc[0]["norm_dominant_rate"]) if not overall.empty else float("nan")
    weak = representation_shift[representation_shift["category_group"].astype(str).str.contains("weak", na=False)]
    strong = representation_shift[representation_shift["category_group"].astype(str).str.contains("strong", na=False)]
    weak_delta = float(weak.iloc[0].get("delta_ndcg@20_mean", float("nan"))) if not weak.empty else float("nan")
    strong_delta = float(strong.iloc[0].get("delta_ndcg@20_mean", float("nan"))) if not strong.empty else float("nan")
    ndcg_corr = component_delta[component_delta["method_delta"].eq("delta_ndcg@20")]
    max_abs_ndcg_corr = float(ndcg_corr["spearman"].abs().max()) if not ndcg_corr.empty else float("nan")

    if overall_norm_rate >= 0.75 and overall_ratio >= 10:
        route = "method_changes_norm_not_direction"
        reason = "category_conf_input 的 item 表示变化主要体现为 q_norm/scale shift，方向性 alignment shift 相对很小。"
    elif (pd.isna(max_abs_ndcg_corr) or max_abs_ndcg_corr < 0.05) and weak_delta <= 0:
        route = "method_signal_mismatch"
        reason = "method delta 与 v2 component 的相关性弱，且 weak 组没有排序收益，说明当前载体不匹配 v2 机制。"
    else:
        route = "method_signal_matches_v2_but_no_gain"
        reason = "method delta 与 v2 component 存在一定对齐，但没有稳定转化为排序收益。"

    evidence = {
        "overall_norm_direction_ratio_mean": overall_ratio,
        "overall_norm_dominant_rate": overall_norm_rate,
        "weak_delta_ndcg@20_mean": weak_delta,
        "strong_delta_ndcg@20_mean": strong_delta,
        "max_abs_spearman_v2_component_vs_delta_ndcg@20": max_abs_ndcg_corr,
    }
    return {"route": route, "reason": reason, "evidence": {key: _jsonable(value) for key, value in evidence.items()}}


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    source_notes: list[str],
    result_dir_display: str,
    summary: pd.DataFrame,
    representation_shift: pd.DataFrame,
    norm_shift: pd.DataFrame,
    component_delta: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    source_lines = "\n".join(f"> {note}" for note in source_notes) or "> 上游设计：未提供"
    component_focus = component_delta[
        component_delta["method_delta"].isin(
            [
                "delta_ndcg@20",
                "delta_q_norm",
                "delta_target_history_q_cosine_mean",
                "delta_margin_to_top20_cutoff",
            ]
        )
    ].copy()
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.7 方法信号匹配诊断结果
date: {run_stamp[:10]}
time: "{run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
created_at: "{run_stamp[:10]} {run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - Task3.7
  - 方法载体
---

# {run_stamp} CCFCRec Amazon-VG category availability v2 Task3.7 方法信号匹配诊断结果

## 来源说明

> [!info] 来源说明
{source_lines}
> 本结果目录：`{result_dir_display}`

## 结论

Task3.7 的 route decision 为 `{decision["route"]}`。

{decision["reason"]}

这个结论只说明 category_conf_input 作为当前方法载体的问题，不直接否定 category availability 方向。

## 数据验收

> [!info] 字段说明
> `field`：验收项。
> `value`：验收项取值。

| field | value |
|---|---:|
| v2_group_count | {len(representation_shift)} |
| component_delta_rows | {len(component_delta)} |
| route | `{decision["route"]}` |
| overall_norm_dominant_rate | {decision["evidence"].get("overall_norm_dominant_rate")} |

## 方法信号摘要

> [!info] 字段说明
> `field`：诊断项。
> `value`：诊断项取值。

{md_table(summary, ["field", "value"])}

## representation shift by v2 group

> [!info] 字段说明
> `category_group`：v2 `s_cat_group`。
> `item_count`：该组 test item 数。
> `delta_ndcg@20_mean`：category_conf_input 减 baseline 的 NDCG@20 均值。
> `delta_q_norm_mean`：q_norm 变化均值。
> `delta_target_history_q_cosine_mean_mean`：目标用户历史中心 q-cosine 变化均值。
> `delta_margin_to_top20_cutoff_mean`：target score 相对 top20 cutoff margin 变化均值。
> `norm_dominant_rate`：norm shift 大于 direction shift 的 item 比例。

{md_table(representation_shift, ["category_group", "item_count", "delta_ndcg@20_mean", "delta_q_norm_mean", "delta_target_history_q_cosine_mean_mean", "delta_margin_to_top20_cutoff_mean", "norm_dominant_rate"])}

## norm vs direction shift

> [!info] 字段说明
> `scope`：统计范围。
> `item_count`：item 数。
> `norm_shift_score_mean`：`abs(delta_q_norm)` 均值。
> `direction_shift_score_mean`：direction / alignment 变化绝对值和的均值。
> `norm_direction_ratio_mean`：norm shift 与 direction shift 的比值均值。
> `norm_dominant_rate`：norm shift 主导的 item 比例。

{md_table(norm_shift, ["scope", "item_count", "norm_shift_score_mean", "direction_shift_score_mean", "norm_direction_ratio_mean", "norm_dominant_rate", "delta_ndcg@20_mean"])}

## v2 component vs method delta

> [!info] 字段说明
> `v2_component`：v2 category availability 变量或子组件。
> `method_delta`：category_conf_input 相对 baseline 的方法变化指标。
> `n`：有效样本数。
> `pearson`：Pearson 相关。
> `spearman`：Spearman 相关。

{md_table(component_focus, ["v2_component", "method_delta", "n", "pearson", "spearman"], max_rows=32)}

## 判断

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

## 下一步

综合 Task3.5 / Task3.6 / Task3.7 后再写 Task4 前置路线判断；不建议直接把原 category_conf_input 扩展成新训练实验。

## 产物

```text
task3p7_method_signal_match_summary.csv
task3p7_representation_shift_by_v2_group.csv
task3p7_norm_vs_direction_shift.csv
task3p7_v2_component_vs_method_delta.csv
task3p7_method_fit_decision.json
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
    profile = build_method_signal_profile(delta_profile)
    representation_shift = build_representation_shift_by_v2_group(profile)
    norm_shift = build_norm_vs_direction_shift(profile)
    component_delta = build_component_vs_method_delta(profile)
    summary = build_method_signal_match_summary(representation_shift, norm_shift, component_delta)
    decision = build_method_fit_decision(representation_shift, norm_shift, component_delta)

    outputs = {
        "task3p7_method_signal_match_summary": output_dir / "task3p7_method_signal_match_summary.csv",
        "task3p7_representation_shift_by_v2_group": output_dir / "task3p7_representation_shift_by_v2_group.csv",
        "task3p7_norm_vs_direction_shift": output_dir / "task3p7_norm_vs_direction_shift.csv",
        "task3p7_v2_component_vs_method_delta": output_dir / "task3p7_v2_component_vs_method_delta.csv",
        "task3p7_method_fit_decision": output_dir / "task3p7_method_fit_decision.json",
    }
    summary.to_csv(outputs["task3p7_method_signal_match_summary"], index=False)
    representation_shift.to_csv(outputs["task3p7_representation_shift_by_v2_group"], index=False)
    norm_shift.to_csv(outputs["task3p7_norm_vs_direction_shift"], index=False)
    component_delta.to_csv(outputs["task3p7_v2_component_vs_method_delta"], index=False)
    outputs["task3p7_method_fit_decision"].write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    result_md = output_dir / f"{run_stamp} CCFCRec Amazon-VG category availability v2 Task3.7 方法信号匹配诊断结果.md"
    manifest_path = output_dir / "run_manifest.json"
    result_dir_display = args.result_dir_display or str(output_dir)
    write_result_markdown(
        output_path=result_md,
        run_stamp=run_stamp,
        source_notes=args.source_note,
        result_dir_display=result_dir_display,
        summary=summary,
        representation_shift=representation_shift,
        norm_shift=norm_shift,
        component_delta=component_delta,
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
            "task3p7_method_signal_match_summary": int(len(summary)),
            "task3p7_representation_shift_by_v2_group": int(len(representation_shift)),
            "task3p7_norm_vs_direction_shift": int(len(norm_shift)),
            "task3p7_v2_component_vs_method_delta": int(len(component_delta)),
        },
        "route_decision": decision,
        "notes": [
            "只复用 Task3.5 item-level delta profile，不训练模型。",
            "Task3.7 只判断 category_conf_input 是否匹配 v2 category availability 机制。",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CCFCRec Amazon-VG category availability v2 Task3.7 方法信号匹配诊断")
    parser.add_argument("--task3p5-dir", required=True, type=str, help="Task3.5 输出目录")
    parser.add_argument("--output-dir", required=True, type=str, help="Task3.7 输出目录")
    parser.add_argument("--source-note", action="append", default=[], help="写入结果 MD 来源说明的 Obsidian wikilink 行")
    parser.add_argument("--result-dir-display", type=str, default="", help="写入结果 MD 的相对结果目录")
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
