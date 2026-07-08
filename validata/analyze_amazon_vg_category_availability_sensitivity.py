#!/usr/bin/env python3
"""
Sensitivity audit for Amazon-VG category availability variables.

This script stays in Task 1/2 territory: it audits variable definitions and
does not run checkpoint evaluation, model training, or Task 3.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from analyze_amazon_vg_category_availability import markdown_table, safe_corr


CONTROL_COLS = [
    "category_count",
    "R_metadata_richness_score",
    "S_train_support_score",
    "P_popularity_score",
]
COMPONENT_COLS = ["s_cat_gran", "s_cat_disc", "s_cat_collab"]
RAW_CANDIDATES = [
    "score_s_cat_current",
    "score_s_cat_no_disc",
    "score_s_cat_no_gran",
    "score_s_cat_no_collab",
]
RESIDUAL_CANDIDATES = [
    "score_s_cat_resid_controls",
    "score_s_cat_no_disc_resid_controls",
    "score_s_cat_no_gran_resid_controls",
    "score_s_cat_component_resid_mean",
]
GROUP_LABELS = ("weak", "mid", "strong")


@dataclass(frozen=True)
class SensitivityOutputs:
    scores_csv: Path
    correlations_csv: Path
    group_summary_csv: Path
    candidate_summary_csv: Path
    decision_json: Path
    result_md: Path
    run_manifest_json: Path


def now_info() -> tuple[str, str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def required_columns() -> set[str]:
    return {"raw_asin", "split", "s_cat", *CONTROL_COLS, *COMPONENT_COLS}


def assert_required_columns(frame: pd.DataFrame) -> None:
    missing = required_columns() - set(frame.columns)
    if missing:
        raise ValueError(f"availability 缺少字段: {sorted(missing)}")


def numeric_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    result = frame[columns].apply(pd.to_numeric, errors="coerce")
    for column in columns:
        if result[column].isna().all():
            result[column] = 0.0
        else:
            result[column] = result[column].fillna(float(result[column].median()))
    return result


def scale_unit_interval(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    min_value = float(numeric.min())
    max_value = float(numeric.max())
    if not np.isfinite(min_value) or not np.isfinite(max_value) or np.isclose(min_value, max_value):
        return pd.Series([0.5] * len(series), index=series.index, dtype=float)
    return ((numeric - min_value) / (max_value - min_value)).clip(0.0, 1.0)


def residualize_score(frame: pd.DataFrame, score_col: str, control_cols: list[str] | None = None) -> pd.Series:
    controls = control_cols or CONTROL_COLS
    work = numeric_frame(frame, [score_col, *controls])
    y = work[score_col].to_numpy(dtype=float)
    x = work[controls].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(work)), x])
    coefficients, *_ = np.linalg.lstsq(x, y, rcond=None)
    residuals = y - x @ coefficients
    return scale_unit_interval(pd.Series(residuals, index=frame.index))


def build_sensitivity_scores(availability: pd.DataFrame) -> pd.DataFrame:
    assert_required_columns(availability)
    result = availability.copy()
    result["score_s_cat_current"] = pd.to_numeric(result["s_cat"], errors="coerce")
    result["score_s_cat_no_disc"] = result[["s_cat_gran", "s_cat_collab"]].mean(axis=1)
    result["score_s_cat_no_gran"] = result[["s_cat_disc", "s_cat_collab"]].mean(axis=1)
    result["score_s_cat_no_collab"] = result[["s_cat_gran", "s_cat_disc"]].mean(axis=1)
    result["score_s_cat_resid_controls"] = residualize_score(result, "score_s_cat_current")
    result["score_s_cat_no_disc_resid_controls"] = residualize_score(result, "score_s_cat_no_disc")
    result["score_s_cat_no_gran_resid_controls"] = residualize_score(result, "score_s_cat_no_gran")

    component_residuals = []
    for component in COMPONENT_COLS:
        component_residuals.append(residualize_score(result, component))
    result["score_s_cat_component_resid_mean"] = pd.concat(component_residuals, axis=1).mean(axis=1)
    for candidate in [*RAW_CANDIDATES, *RESIDUAL_CANDIDATES]:
        result[candidate] = scale_unit_interval(result[candidate])
    return result


def candidate_group(series: pd.Series, split: pd.Series) -> tuple[pd.Series, dict[str, float | bool]]:
    train_scores = pd.to_numeric(series[split.eq("train")], errors="coerce").dropna()
    if train_scores.empty:
        ranked = series.rank(method="average", pct=True)
        return ranked.map(rank_to_group), {"weak_max": np.nan, "mid_max": np.nan, "fallback": True}
    weak_max = float(train_scores.quantile(1 / 3))
    mid_max = float(train_scores.quantile(2 / 3))
    if np.isclose(weak_max, mid_max):
        ranked = series.rank(method="average", pct=True)
        return ranked.map(rank_to_group), {"weak_max": weak_max, "mid_max": mid_max, "fallback": True}
    return (
        series.map(lambda value: score_to_group(float(value), weak_max, mid_max)),
        {"weak_max": weak_max, "mid_max": mid_max, "fallback": False},
    )


def rank_to_group(rank_pct: float) -> str:
    if rank_pct <= 1 / 3:
        return GROUP_LABELS[0]
    if rank_pct <= 2 / 3:
        return GROUP_LABELS[1]
    return GROUP_LABELS[2]


def score_to_group(value: float, weak_max: float, mid_max: float) -> str:
    if value <= weak_max:
        return GROUP_LABELS[0]
    if value <= mid_max:
        return GROUP_LABELS[1]
    return GROUP_LABELS[2]


def candidate_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in [*RAW_CANDIDATES, *RESIDUAL_CANDIDATES] if column in frame.columns]


def build_candidate_correlations(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in candidate_columns(scores):
        for relation_type, variables in [("control", CONTROL_COLS), ("component", COMPONENT_COLS)]:
            for variable in variables:
                left = pd.to_numeric(scores[candidate], errors="coerce")
                right = pd.to_numeric(scores[variable], errors="coerce")
                pearson = safe_corr(left, right, "pearson")
                spearman = safe_corr(left, right, "spearman")
                rows.append(
                    {
                        "candidate": candidate,
                        "candidate_type": "residual" if "resid" in candidate else "raw",
                        "relation_type": relation_type,
                        "variable": variable,
                        "n": int(pd.concat([left, right], axis=1).dropna().shape[0]),
                        "pearson": pearson,
                        "spearman": spearman,
                        "abs_pearson": abs(pearson) if pd.notna(pearson) else np.nan,
                        "abs_spearman": abs(spearman) if pd.notna(spearman) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_candidate_group_summary(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = max(len(scores), 1)
    for candidate in candidate_columns(scores):
        group_col = f"{candidate}_group"
        groups, thresholds = candidate_group(scores[candidate], scores["split"])
        scores[group_col] = groups
        for group, group_df in scores.groupby(group_col, dropna=False):
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": "residual" if "resid" in candidate else "raw",
                    "group": group,
                    "item_count": int(len(group_df)),
                    "item_share": len(group_df) / total,
                    "train_item_count": int(group_df["split"].eq("train").sum()),
                    "validate_item_count": int(group_df["split"].eq("validate").sum()),
                    "test_item_count": int(group_df["split"].eq("test").sum()),
                    "score_mean": float(pd.to_numeric(group_df[candidate], errors="coerce").mean()),
                    "score_min": float(pd.to_numeric(group_df[candidate], errors="coerce").min()),
                    "score_max": float(pd.to_numeric(group_df[candidate], errors="coerce").max()),
                    "threshold_weak_max": thresholds["weak_max"],
                    "threshold_mid_max": thresholds["mid_max"],
                    "threshold_fallback": thresholds["fallback"],
                }
            )
    return pd.DataFrame(rows)


def build_candidate_summary(
    correlations: pd.DataFrame,
    group_summary: pd.DataFrame,
    control_threshold: float,
    component_threshold: float,
    min_group_share: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate in sorted(correlations["candidate"].unique()):
        corr_subset = correlations[correlations["candidate"].eq(candidate)]
        group_subset = group_summary[group_summary["candidate"].eq(candidate)]
        control_max = float(corr_subset[corr_subset["relation_type"].eq("control")]["abs_spearman"].max())
        component_max = float(corr_subset[corr_subset["relation_type"].eq("component")]["abs_spearman"].max())
        group_min_share = float(group_subset["item_share"].min()) if not group_subset.empty else 0.0
        control_pass = control_max < control_threshold
        component_pass = component_max < component_threshold
        group_pass = group_min_share >= min_group_share
        rows.append(
            {
                "candidate": candidate,
                "candidate_type": "residual" if "resid" in candidate else "raw",
                "control_max_abs_spearman": control_max,
                "component_max_abs_spearman": component_max,
                "group_min_share": group_min_share,
                "control_pass": control_pass,
                "component_pass": component_pass,
                "group_pass": group_pass,
                "all_pass": control_pass and component_pass and group_pass,
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(
        ["all_pass", "candidate_type", "control_max_abs_spearman", "component_max_abs_spearman"],
        ascending=[False, True, True, True],
    ).reset_index(drop=True)


def choose_decision(candidate_summary: pd.DataFrame) -> dict[str, object]:
    passing = candidate_summary[candidate_summary["all_pass"]]
    raw_passing = passing[passing["candidate_type"].eq("raw")]
    residual_passing = passing[passing["candidate_type"].eq("residual")]
    if not raw_passing.empty:
        selected = raw_passing.iloc[0]
        route = "ready_for_task3_raw"
        rationale = "至少一个非残差候选同时通过 control、component 和 group 阈值。"
    elif not residual_passing.empty:
        selected = residual_passing.iloc[0]
        route = "ready_for_task3_diagnostic_only"
        rationale = "只有 residual 候选通过阈值；可以作为诊断分组线索，但不宜直接作为模型输入变量。"
    else:
        selected = candidate_summary.iloc[0]
        route = "needs_variable_v2"
        rationale = "没有候选同时通过 control、component 和 group 阈值，需要继续变量定义迭代。"
    return {
        "route": route,
        "selected_candidate": selected["candidate"],
        "selected_candidate_type": selected["candidate_type"],
        "control_max_abs_spearman": float(selected["control_max_abs_spearman"]),
        "component_max_abs_spearman": float(selected["component_max_abs_spearman"]),
        "group_min_share": float(selected["group_min_share"]),
        "rationale": rationale,
    }


def evaluate_candidates(
    scores: pd.DataFrame,
    control_threshold: float = 0.70,
    component_threshold: float = 0.90,
    min_group_share: float = 0.10,
) -> dict[str, object]:
    correlations = build_candidate_correlations(scores)
    group_summary = build_candidate_group_summary(scores)
    candidate_summary = build_candidate_summary(
        correlations,
        group_summary,
        control_threshold=control_threshold,
        component_threshold=component_threshold,
        min_group_share=min_group_share,
    )
    decision = choose_decision(candidate_summary)
    decision.update(
        {
            "control_threshold": control_threshold,
            "component_threshold": component_threshold,
            "min_group_share": min_group_share,
        }
    )
    return {
        "correlations": correlations,
        "group_summary": group_summary,
        "candidate_summary": candidate_summary,
        "decision": decision,
    }


def source_block(upstream_design_links: list[str], result_dir: Path) -> str:
    lines = ["> [!info] 来源说明"]
    for index, link in enumerate(upstream_design_links, start=1):
        label = "上游设计" if index == 1 else f"上游设计 {index}"
        lines.append(f"> {label}：[[{link}]]")
    lines.append(f"> 本结果目录：`{result_dir}`")
    return "\n".join(lines)


def warning_lines(candidate_summary: pd.DataFrame) -> str:
    failing = candidate_summary[~candidate_summary["all_pass"]]
    if failing.empty:
        return "- no warning triggered"
    lines = []
    for row in failing.itertuples():
        failed_parts = []
        if not row.control_pass:
            failed_parts.append(f"control={row.control_max_abs_spearman:.4f}")
        if not row.component_pass:
            failed_parts.append(f"component={row.component_max_abs_spearman:.4f}")
        if not row.group_pass:
            failed_parts.append(f"group_min_share={row.group_min_share:.4f}")
        lines.append(f"- {row.candidate}: " + ", ".join(failed_parts))
    return "\n".join(lines)


def write_result_markdown(
    output_path: Path,
    source_path: Path,
    output_dir: Path,
    upstream_design_links: list[str],
    scores: pd.DataFrame,
    candidate_summary: pd.DataFrame,
    decision: dict[str, object],
    date: str,
    stamp: str,
    created_at: str,
) -> None:
    top_candidates = candidate_summary.sort_values(
        ["all_pass", "control_max_abs_spearman", "component_max_abs_spearman"],
        ascending=[False, True, True],
    )
    text = f"""---
title: {stamp} CCFCRec Amazon-VG category availability 敏感性审查结果
date: {date}
created_at: "{created_at}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - 敏感性审查
---

# {stamp} CCFCRec Amazon-VG category availability 敏感性审查结果

## 来源说明

{source_block(upstream_design_links, output_dir)}

## 结论

本轮不执行 Task 3，不改模型，不跑训练。自动路线判断为：

```text
route = {decision["route"]}
selected_candidate = {decision["selected_candidate"]}
rationale = {decision["rationale"]}
```

## 数据验收

> [!info] 字段说明
> `field`：验收项。
> `value`：验收项取值。

| field | value |
|---|---:|
| source | `{source_path}` |
| input_rows | {len(scores)} |
| candidate_count | {len(candidate_columns(scores))} |
| control_threshold | {decision["control_threshold"]:.2f} |
| component_threshold | {decision["component_threshold"]:.2f} |
| min_group_share | {decision["min_group_share"]:.2f} |

## Candidate Summary

> [!info] 字段说明
> `candidate`：候选分数。
> `candidate_type`：raw 表示非残差候选，residual 表示残差诊断候选。
> `control_max_abs_spearman`：候选分数与控制变量的最大 Spearman 绝对值。
> `component_max_abs_spearman`：候选分数与组成项的最大 Spearman 绝对值。
> `group_min_share`：该候选 weak/mid/strong 三组中最小 item 占比。
> `all_pass`：是否同时通过 control、component、group 阈值。

{markdown_table(top_candidates)}

## Warning

> [!info] 字段说明
> `warning`：未通过阈值的候选及失败维度；不是模型结论。

{warning_lines(candidate_summary)}

## 判断

```text
ready_for_task3_raw：可停止变量迭代，进入 Task 3 之前仍需人工确认。
ready_for_task3_diagnostic_only：继续 Task 1/2，把 residual 线索转成可解释变量。
needs_variable_v2：继续 Task 1/2，重构 A_gran/A_disc/A_collab 或聚合方式。
```

## 产物

```text
category_availability_sensitivity_scores.csv
category_availability_sensitivity_correlations.csv
category_availability_sensitivity_group_summary.csv
category_availability_sensitivity_candidate_summary.csv
category_availability_sensitivity_decision.json
run_manifest.json
```
"""
    output_path.write_text(text, encoding="utf-8")


def write_sensitivity_outputs(
    availability: pd.DataFrame,
    source_path: Path,
    output_dir: Path,
    upstream_design_links: list[str],
    control_threshold: float = 0.70,
    component_threshold: float = 0.90,
    min_group_share: float = 0.10,
) -> SensitivityOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    scores = build_sensitivity_scores(availability)
    evaluation = evaluate_candidates(
        scores,
        control_threshold=control_threshold,
        component_threshold=component_threshold,
        min_group_share=min_group_share,
    )
    correlations = evaluation["correlations"]
    group_summary = evaluation["group_summary"]
    candidate_summary = evaluation["candidate_summary"]
    decision = evaluation["decision"]
    date, stamp, created_at = now_info()

    scores_csv = output_dir / "category_availability_sensitivity_scores.csv"
    correlations_csv = output_dir / "category_availability_sensitivity_correlations.csv"
    group_summary_csv = output_dir / "category_availability_sensitivity_group_summary.csv"
    candidate_summary_csv = output_dir / "category_availability_sensitivity_candidate_summary.csv"
    decision_json = output_dir / "category_availability_sensitivity_decision.json"
    result_md = output_dir / f"{stamp} CCFCRec Amazon-VG category availability 敏感性审查结果.md"
    run_manifest_json = output_dir / "run_manifest.json"

    output_cols = [
        "raw_asin",
        "split",
        *CONTROL_COLS,
        *COMPONENT_COLS,
        *candidate_columns(scores),
        *[f"{candidate}_group" for candidate in candidate_columns(scores) if f"{candidate}_group" in scores.columns],
    ]
    scores[output_cols].to_csv(scores_csv, index=False)
    correlations.to_csv(correlations_csv, index=False)
    group_summary.to_csv(group_summary_csv, index=False)
    candidate_summary.to_csv(candidate_summary_csv, index=False)
    decision_json.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_result_markdown(
        output_path=result_md,
        source_path=source_path,
        output_dir=output_dir,
        upstream_design_links=upstream_design_links,
        scores=scores,
        candidate_summary=candidate_summary,
        decision=decision,
        date=date,
        stamp=stamp,
        created_at=created_at,
    )
    manifest = {
        "script": "analyze_amazon_vg_category_availability_sensitivity.py",
        "created_stamp": stamp,
        "created_at": created_at,
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "input_rows": int(len(availability)),
        "candidate_count": int(len(candidate_columns(scores))),
        "decision": decision,
        "outputs": {
            "scores_csv": str(scores_csv),
            "correlations_csv": str(correlations_csv),
            "group_summary_csv": str(group_summary_csv),
            "candidate_summary_csv": str(candidate_summary_csv),
            "decision_json": str(decision_json),
            "result_md": str(result_md),
            "run_manifest_json": str(run_manifest_json),
        },
    }
    run_manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return SensitivityOutputs(
        scores_csv=scores_csv,
        correlations_csv=correlations_csv,
        group_summary_csv=group_summary_csv,
        candidate_summary_csv=candidate_summary_csv,
        decision_json=decision_json,
        result_md=result_md,
        run_manifest_json=run_manifest_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--upstream-design",
        action="append",
        default=[],
        help="Obsidian note title to link in 来源说明. Can be passed multiple times.",
    )
    parser.add_argument("--control-threshold", type=float, default=0.70)
    parser.add_argument("--component-threshold", type=float, default=0.90)
    parser.add_argument("--min-group-share", type=float, default=0.10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    availability = pd.read_csv(args.availability)
    upstream_links = args.upstream_design or [
        "2026-07-04 233711 CCFCRec Amazon-VG category availability 敏感性审查诊断设计"
    ]
    outputs = write_sensitivity_outputs(
        availability=availability,
        source_path=args.availability,
        output_dir=args.output_dir,
        upstream_design_links=upstream_links,
        control_threshold=args.control_threshold,
        component_threshold=args.component_threshold,
        min_group_share=args.min_group_share,
    )
    print(f"wrote {outputs.scores_csv}")
    print(f"wrote {outputs.correlations_csv}")
    print(f"wrote {outputs.group_summary_csv}")
    print(f"wrote {outputs.candidate_summary_csv}")
    print(f"wrote {outputs.decision_json}")
    print(f"wrote {outputs.result_md}")
    print(f"wrote {outputs.run_manifest_json}")


if __name__ == "__main__":
    main()
