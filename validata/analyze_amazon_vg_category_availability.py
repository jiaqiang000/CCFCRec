#!/usr/bin/env python3
"""
Audit Amazon-VG category availability variables for CCFCRec.

The audit is descriptive. Warnings indicate variables that need human review;
they do not automatically invalidate the variable design.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


CORE_COLUMNS = [
    "category_count",
    "R_metadata_richness_score",
    "S_train_support_score",
    "P_popularity_score",
    "s_cat_gran",
    "s_cat_disc",
    "s_cat_collab",
]


@dataclass(frozen=True)
class AnalysisOutputs:
    audit_md: Path
    correlations_csv: Path
    group_summary_csv: Path
    run_manifest_json: Path


def now_info() -> tuple[str, str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d"), now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    pair = pd.concat([left, right], axis=1).dropna()
    if len(pair) < 2:
        return np.nan
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method))


def compute_correlations(availability: pd.DataFrame) -> pd.DataFrame:
    required = {"s_cat", *CORE_COLUMNS}
    missing = required - set(availability.columns)
    if missing:
        raise ValueError(f"availability 缺少字段: {sorted(missing)}")

    pairs: list[tuple[str, str]] = []
    for right in CORE_COLUMNS:
        if right != "s_cat":
            pairs.append(("s_cat", right))
    component_cols = ["s_cat_gran", "s_cat_disc", "s_cat_collab"]
    for index, left in enumerate(component_cols):
        for right in component_cols[index + 1 :]:
            pairs.append((left, right))
    for component in component_cols:
        for control in ["category_count", "R_metadata_richness_score", "S_train_support_score", "P_popularity_score"]:
            pairs.append((component, control))

    rows = []
    for left, right in pairs:
        left_series = pd.to_numeric(availability[left], errors="coerce")
        right_series = pd.to_numeric(availability[right], errors="coerce")
        pair_count = int(pd.concat([left_series, right_series], axis=1).dropna().shape[0])
        pearson = safe_corr(left_series, right_series, "pearson")
        spearman = safe_corr(left_series, right_series, "spearman")
        rows.append(
            {
                "left": left,
                "right": right,
                "n": pair_count,
                "pearson": pearson,
                "spearman": spearman,
                "abs_pearson": abs(pearson) if pd.notna(pearson) else np.nan,
                "abs_spearman": abs(spearman) if pd.notna(spearman) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_group_summary(availability: pd.DataFrame) -> pd.DataFrame:
    required = {"raw_asin", "split", "s_cat_group", "s_cat", "category_count"}
    missing = required - set(availability.columns)
    if missing:
        raise ValueError(f"availability 缺少字段: {sorted(missing)}")

    total = max(len(availability), 1)
    numeric_cols = [
        column
        for column in [
            "category_count",
            "s_cat",
            "s_cat_gran",
            "s_cat_disc",
            "s_cat_collab",
            "R_metadata_richness_score",
            "S_train_support_score",
            "P_popularity_score",
        ]
        if column in availability.columns
    ]
    rows = []
    for group, group_df in availability.groupby("s_cat_group", dropna=False):
        row = {
            "s_cat_group": group,
            "item_count": int(len(group_df)),
            "item_share": len(group_df) / total,
            "train_item_count": int(group_df["split"].eq("train").sum()),
            "validate_item_count": int(group_df["split"].eq("validate").sum()),
            "test_item_count": int(group_df["split"].eq("test").sum()),
        }
        for column in numeric_cols:
            row[f"{column}_mean"] = float(pd.to_numeric(group_df[column], errors="coerce").mean())
            row[f"{column}_min"] = float(pd.to_numeric(group_df[column], errors="coerce").min())
            row[f"{column}_max"] = float(pd.to_numeric(group_df[column], errors="coerce").max())
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values("s_cat_group").reset_index(drop=True)
    return result


def mark_correlation_warnings(correlations: pd.DataFrame, threshold: float) -> pd.DataFrame:
    result = correlations.copy()
    result["warning"] = result[["abs_pearson", "abs_spearman"]].max(axis=1) >= threshold
    return result


def build_warnings(
    correlations: pd.DataFrame,
    group_summary: pd.DataFrame,
    corr_warning_threshold: float,
    min_group_share: float,
) -> list[str]:
    warnings: list[str] = []
    if not correlations.empty:
        high_corr = correlations[correlations["warning"]]
        for row in high_corr.itertuples():
            warnings.append(
                f"high correlation: {row.left} vs {row.right}, "
                f"pearson={row.pearson:.4f}, spearman={row.spearman:.4f}, "
                f"threshold={corr_warning_threshold:.2f}"
            )
    if not group_summary.empty:
        small_groups = group_summary[group_summary["item_share"] < min_group_share]
        for row in small_groups.itertuples():
            warnings.append(
                f"small group: {row.s_cat_group}, item_count={row.item_count}, "
                f"item_share={row.item_share:.4f}, threshold={min_group_share:.2f}"
            )
    return warnings


def markdown_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "\n无数据。\n"
    display = frame.head(max_rows).copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    headers = [str(column) for column in display.columns]
    rows = []
    for _, row in display.iterrows():
        rows.append(["" if pd.isna(value) else str(value) for value in row.tolist()])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(value.replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_audit_markdown(
    availability: pd.DataFrame,
    correlations: pd.DataFrame,
    group_summary: pd.DataFrame,
    warnings: list[str],
    output_path: Path,
    source_path: Path,
    corr_warning_threshold: float,
    min_group_share: float,
    date: str,
    stamp: str,
    created_at: str,
) -> None:
    warning_text = "\n".join(f"- {item}" for item in warnings) if warnings else "- no warning triggered"
    top_corr = correlations.sort_values(["warning", "abs_spearman", "abs_pearson"], ascending=[False, False, False])
    split_counts = availability["split"].value_counts().rename_axis("split").reset_index(name="item_count")

    text = f"""---
title: {stamp} CCFCRec Amazon-VG category availability 变量审查结果
date: {date}
created_at: "{created_at}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - 变量审查
---

# {stamp} CCFCRec Amazon-VG category availability 变量审查结果

## 结论

本报告只做变量审查，不训练模型，也不判断论文结论是否成立。warning 表示需要人工复核的变量风险。

## 数据验收

> [!info] 字段说明
> `field`：验收项。
> `value`：验收项取值。

| field | value |
|---|---:|
| source | `{source_path}` |
| input_rows | {len(availability)} |
| unique_raw_asin | {availability["raw_asin"].nunique()} |
| corr_warning_threshold | {corr_warning_threshold:.2f} |
| min_group_share | {min_group_share:.2f} |

## Split 分布

> [!info] 字段说明
> `split`：item 所属数据划分。
> `item_count`：该 split 的 item 数量。

{markdown_table(split_counts)}

## Warning

> [!info] 字段说明
> `warning`：需要人工复核的风险提示；不是自动失败结论。

{warning_text}

## 相关性审查

> [!info] 字段说明
> `left`：相关性左侧变量。
> `right`：相关性右侧变量。
> `n`：参与相关性计算的 item 数。
> `pearson`：Pearson 相关系数。
> `spearman`：Spearman 相关系数。
> `warning`：是否触发相关性过高提示。

{markdown_table(top_corr)}

## s_cat_group 分布

> [!info] 字段说明
> `s_cat_group`：按 train split 阈值生成的类别可用性分组。
> `item_count`：该组 item 数量。
> `item_share`：该组占全部输出 item 比例。
> `train_item_count`：该组 train item 数量。
> `validate_item_count`：该组 validate item 数量。
> `test_item_count`：该组 test item 数量。

{markdown_table(group_summary)}

## 判断

后续是否进入 checkpoint 复评，应重点看：

```text
s_cat 是否只是 category_count 的翻版；
s_cat 是否只是 S/P support 或 popularity 的翻版；
s_cat_group 是否在 train/validate/test 中极端失衡；
A_gran/A_disc/A_collab 是否存在一项完全吞掉其他项。
```

## 产物

```text
category_availability_correlations.csv
category_availability_group_summary.csv
category_availability_audit.md
run_manifest.json
```
"""
    output_path.write_text(text, encoding="utf-8")


def write_analysis_outputs(
    availability: pd.DataFrame,
    output_dir: Path,
    source_path: Path,
    corr_warning_threshold: float = 0.85,
    min_group_share: float = 0.05,
) -> AnalysisOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    correlations = mark_correlation_warnings(compute_correlations(availability), corr_warning_threshold)
    group_summary = build_group_summary(availability)
    warnings = build_warnings(correlations, group_summary, corr_warning_threshold, min_group_share)
    date, stamp, created_at = now_info()

    correlations_csv = output_dir / "category_availability_correlations.csv"
    group_summary_csv = output_dir / "category_availability_group_summary.csv"
    audit_md = output_dir / f"{stamp} CCFCRec Amazon-VG category availability 变量审查结果.md"
    run_manifest_json = output_dir / "run_manifest.json"

    correlations.to_csv(correlations_csv, index=False)
    group_summary.to_csv(group_summary_csv, index=False)
    write_audit_markdown(
        availability=availability,
        correlations=correlations,
        group_summary=group_summary,
        warnings=warnings,
        output_path=audit_md,
        source_path=source_path,
        corr_warning_threshold=corr_warning_threshold,
        min_group_share=min_group_share,
        date=date,
        stamp=stamp,
        created_at=created_at,
    )

    manifest = {
        "script": "analyze_amazon_vg_category_availability.py",
        "created_stamp": stamp,
        "created_at": created_at,
        "source_path": str(source_path),
        "output_dir": str(output_dir),
        "input_rows": int(len(availability)),
        "unique_raw_asin": int(availability["raw_asin"].nunique()) if "raw_asin" in availability.columns else None,
        "corr_warning_threshold": corr_warning_threshold,
        "min_group_share": min_group_share,
        "warning_count": int(len(warnings)),
        "outputs": {
            "audit_md": str(audit_md),
            "correlations_csv": str(correlations_csv),
            "group_summary_csv": str(group_summary_csv),
            "run_manifest_json": str(run_manifest_json),
        },
    }
    run_manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return AnalysisOutputs(
        audit_md=audit_md,
        correlations_csv=correlations_csv,
        group_summary_csv=group_summary_csv,
        run_manifest_json=run_manifest_json,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--availability", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--corr-warning-threshold", type=float, default=0.85)
    parser.add_argument("--min-group-share", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    availability = pd.read_csv(args.availability)
    outputs = write_analysis_outputs(
        availability=availability,
        output_dir=args.output_dir,
        source_path=args.availability,
        corr_warning_threshold=args.corr_warning_threshold,
        min_group_share=args.min_group_share,
    )
    print(f"wrote {outputs.correlations_csv}")
    print(f"wrote {outputs.group_summary_csv}")
    print(f"wrote {outputs.audit_md}")
    print(f"wrote {outputs.run_manifest_json}")


if __name__ == "__main__":
    main()
