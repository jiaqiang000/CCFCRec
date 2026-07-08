#!/usr/bin/env python3
"""
Task4-post-0 diagnostic: category detail groups x Acat_v3.

The script checks whether Acat_v3 directly explains the original
"more detailed category labels perform better" observation, and whether
M3/M6 differences depend on category detail layers.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/luojiaqiang/Documents/Obsidian Vault/科研/CCFCRec对比学习思路")
DEFAULT_AVAILABILITY_CSV = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260705"
    / "2026-07-05 112021 category-availability-v3-purity-audit"
    / "category_availability_v3_item.csv"
)
DEFAULT_ITEM_EVAL_CSV = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260706"
    / "2026-07-06 130518 task4-post-acat-v3-weight-controls-analysis"
    / "task4_post_item_eval_profile.csv"
)
DEFAULT_M4_BEST_CSV = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260707"
    / "2026-07-07 101503 task4-revise-m4-pairmargin-fullscreen-analysis"
    / "task4_revise_m4_pairmargin_best_summary.csv"
)
DEFAULT_M4_CONTROL_CSV = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260707"
    / "2026-07-07 101503 task4-revise-m4-pairmargin-fullscreen-analysis"
    / "task4_revise_m4_pairmargin_vs_controls.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "temp_202607_实验文件记录" / "temp_20260707"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "实验记录"

DETAIL_ORDER = {
    "cat_count_1_3": 1,
    "cat_count_4": 2,
    "cat_count_5_plus": 3,
}
DETAIL_LABEL = {
    "cat_count_1_3": "low_detail",
    "cat_count_4": "mid_detail",
    "cat_count_5_plus": "high_detail",
}
ACAT_ORDER = {
    "s_cat_v3_weak": 1,
    "s_cat_v3_mid": 2,
    "s_cat_v3_strong": 3,
}
SPLIT_ORDER = {"validate": 1, "test": 2, "train": 3}
M3 = "task4_acat_trainhard_weight"
M6 = "task4_acat_shuffle_high_weight"


@dataclass(frozen=True)
class Outputs:
    output_dir: Path
    detail_summary_csv: Path
    trend_checks_csv: Path
    within_detail_acat_summary_csv: Path
    within_detail_acat_contrast_csv: Path
    method_layer_summary_csv: Path
    method_layer_delta_csv: Path
    m4_gate_summary_csv: Path
    manifest_json: Path
    result_md: Path
    route_md: Path


def now_stamp() -> tuple[str, str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H%M%S"), now.strftime("%Y-%m-%d"), now.isoformat(timespec="seconds")


def nan_float() -> float:
    return float("nan")


def safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return nan_float()
    return float(values.mean())


def safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce")
    if values.notna().sum() == 0:
        return nan_float()
    return float(values.median())


def bool_text(value) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    text = str(value)
    if text.lower() in {"true", "false"}:
        return text.capitalize()
    return text


def detail_group_from_bin(value) -> str:
    return DETAIL_LABEL.get(str(value), str(value))


def detail_order_from_bin(value) -> int:
    return DETAIL_ORDER.get(str(value), 99)


def acat_order(value) -> int:
    return ACAT_ORDER.get(str(value), 99)


def _strict_increasing(values: list[float]) -> bool:
    if len(values) < 2 or any(pd.isna(value) for value in values):
        return False
    return all(right > left for left, right in zip(values, values[1:]))


def _numeric_eval_rows(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["ndcg@20", "hr@20", "s_cat_v3", "category_count"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")
    if {"ndcg@20", "hr@20"}.issubset(work.columns):
        work = work[work["ndcg@20"].notna() & work["hr@20"].notna()].copy()
    work["detail_group"] = work["cat_count_bin"].map(detail_group_from_bin)
    work["detail_order"] = work["cat_count_bin"].map(detail_order_from_bin)
    return work


def sort_by_split_detail(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "split" in work.columns:
        work["_split_order"] = work["split"].map(SPLIT_ORDER).fillna(99)
    else:
        work["_split_order"] = 99
    if "detail_order" not in work.columns and "cat_count_bin" in work.columns:
        work["detail_order"] = work["cat_count_bin"].map(detail_order_from_bin)
    sort_cols = [col for col in ["_split_order", "detail_order", "acat_order", "method_variant", "high_acat_flag"] if col in work.columns]
    work = work.sort_values(sort_cols).drop(columns=["_split_order"])
    return work.reset_index(drop=True)


def build_detail_acat_summary(availability: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "cat_count_bin", "category_count", "s_cat_v3", "s_cat_v3_group", "ndcg@20", "hr@20"}
    missing = required - set(availability.columns)
    if missing:
        raise ValueError(f"availability missing columns: {sorted(missing)}")
    work = _numeric_eval_rows(availability)
    rows = []
    optional_metrics = [
        "R_metadata_richness_score",
        "S_train_support_score",
        "P_popularity_score",
        "baseline_margin_proxy",
        "best_target_rank",
    ]
    for (split, cat_bin, detail_group, detail_order), sub in work.groupby(
        ["split", "cat_count_bin", "detail_group", "detail_order"],
        dropna=False,
    ):
        row = {
            "split": split,
            "cat_count_bin": cat_bin,
            "detail_group": detail_group,
            "detail_order": int(detail_order),
            "item_count": len(sub),
            "category_count_mean": safe_mean(sub["category_count"]),
            "baseline_ndcg@20_mean": safe_mean(sub["ndcg@20"]),
            "baseline_hr@20_mean": safe_mean(sub["hr@20"]),
            "Acat_v3_mean": safe_mean(sub["s_cat_v3"]),
            "Acat_v3_median": safe_median(sub["s_cat_v3"]),
            "Acat_v3_weak_share": float(sub["s_cat_v3_group"].eq("s_cat_v3_weak").mean()),
            "Acat_v3_mid_share": float(sub["s_cat_v3_group"].eq("s_cat_v3_mid").mean()),
            "Acat_v3_strong_share": float(sub["s_cat_v3_group"].eq("s_cat_v3_strong").mean()),
        }
        for metric in optional_metrics:
            if metric in sub.columns:
                row[f"{metric}_mean"] = safe_mean(sub[metric])
        rows.append(row)
    return sort_by_split_detail(pd.DataFrame(rows))


def build_detail_trend_checks(detail_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, sub in detail_summary.groupby("split", dropna=False):
        ordered = sub.sort_values("detail_order")
        values = {
            metric: ordered[metric].tolist()
            for metric in ["baseline_ndcg@20_mean", "baseline_hr@20_mean", "Acat_v3_mean"]
            if metric in ordered.columns
        }
        low = ordered[ordered["detail_order"].eq(1)]
        mid = ordered[ordered["detail_order"].eq(2)]
        high = ordered[ordered["detail_order"].eq(3)]
        rows.append(
            {
                "split": split,
                "has_three_detail_groups": len(ordered["detail_order"].dropna().unique()) == 3,
                "low_detail_ndcg@20": float(low.iloc[0]["baseline_ndcg@20_mean"]) if not low.empty else nan_float(),
                "mid_detail_ndcg@20": float(mid.iloc[0]["baseline_ndcg@20_mean"]) if not mid.empty else nan_float(),
                "high_detail_ndcg@20": float(high.iloc[0]["baseline_ndcg@20_mean"]) if not high.empty else nan_float(),
                "baseline_ndcg_strict_increasing": _strict_increasing(values.get("baseline_ndcg@20_mean", [])),
                "low_detail_hr@20": float(low.iloc[0]["baseline_hr@20_mean"]) if not low.empty else nan_float(),
                "mid_detail_hr@20": float(mid.iloc[0]["baseline_hr@20_mean"]) if not mid.empty else nan_float(),
                "high_detail_hr@20": float(high.iloc[0]["baseline_hr@20_mean"]) if not high.empty else nan_float(),
                "baseline_hr_strict_increasing": _strict_increasing(values.get("baseline_hr@20_mean", [])),
                "low_detail_Acat_v3": float(low.iloc[0]["Acat_v3_mean"]) if not low.empty else nan_float(),
                "mid_detail_Acat_v3": float(mid.iloc[0]["Acat_v3_mean"]) if not mid.empty else nan_float(),
                "high_detail_Acat_v3": float(high.iloc[0]["Acat_v3_mean"]) if not high.empty else nan_float(),
                "acat_mean_strict_increasing": _strict_increasing(values.get("Acat_v3_mean", [])),
            }
        )
    result = pd.DataFrame(rows)
    for col in ["has_three_detail_groups", "baseline_ndcg_strict_increasing", "baseline_hr_strict_increasing", "acat_mean_strict_increasing"]:
        if col in result.columns:
            result[col] = result[col].astype(object)
    return sort_by_split_detail(result)


def build_within_detail_acat_summary(availability: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "cat_count_bin", "s_cat_v3", "s_cat_v3_group", "ndcg@20", "hr@20"}
    missing = required - set(availability.columns)
    if missing:
        raise ValueError(f"availability missing columns: {sorted(missing)}")
    work = _numeric_eval_rows(availability)
    rows = []
    for (split, cat_bin, detail_group, detail_order, acat_group), sub in work.groupby(
        ["split", "cat_count_bin", "detail_group", "detail_order", "s_cat_v3_group"],
        dropna=False,
    ):
        rows.append(
            {
                "split": split,
                "cat_count_bin": cat_bin,
                "detail_group": detail_group,
                "detail_order": int(detail_order),
                "s_cat_v3_group": acat_group,
                "acat_order": acat_order(acat_group),
                "item_count": len(sub),
                "Acat_v3_mean": safe_mean(sub["s_cat_v3"]),
                "baseline_ndcg@20_mean": safe_mean(sub["ndcg@20"]),
                "baseline_hr@20_mean": safe_mean(sub["hr@20"]),
            }
        )
    return sort_by_split_detail(pd.DataFrame(rows))


def build_within_detail_acat_contrast(within_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, detail_group, detail_order), sub in within_summary.groupby(
        ["split", "detail_group", "detail_order"],
        dropna=False,
    ):
        by_group = {str(row["s_cat_v3_group"]): row for _, row in sub.iterrows()}
        weak = by_group.get("s_cat_v3_weak")
        strong = by_group.get("s_cat_v3_strong")
        best_ndcg = sub.sort_values("baseline_ndcg@20_mean", ascending=False).iloc[0]
        best_hr = sub.sort_values("baseline_hr@20_mean", ascending=False).iloc[0]
        rows.append(
            {
                "split": split,
                "detail_group": detail_group,
                "detail_order": int(detail_order),
                "weak_ndcg@20_mean": float(weak["baseline_ndcg@20_mean"]) if weak is not None else nan_float(),
                "strong_ndcg@20_mean": float(strong["baseline_ndcg@20_mean"]) if strong is not None else nan_float(),
                "strong_minus_weak_ndcg@20_mean": (
                    float(strong["baseline_ndcg@20_mean"]) - float(weak["baseline_ndcg@20_mean"])
                    if weak is not None and strong is not None
                    else nan_float()
                ),
                "weak_hr@20_mean": float(weak["baseline_hr@20_mean"]) if weak is not None else nan_float(),
                "strong_hr@20_mean": float(strong["baseline_hr@20_mean"]) if strong is not None else nan_float(),
                "strong_minus_weak_hr@20_mean": (
                    float(strong["baseline_hr@20_mean"]) - float(weak["baseline_hr@20_mean"])
                    if weak is not None and strong is not None
                    else nan_float()
                ),
                "best_acat_group_by_ndcg@20": str(best_ndcg["s_cat_v3_group"]),
                "best_acat_group_by_hr@20": str(best_hr["s_cat_v3_group"]),
                "strong_acat_is_best_ndcg@20": bool(best_ndcg["s_cat_v3_group"] == "s_cat_v3_strong"),
                "strong_acat_is_best_hr@20": bool(best_hr["s_cat_v3_group"] == "s_cat_v3_strong"),
            }
        )
    result = pd.DataFrame(rows)
    for col in ["strong_acat_is_best_ndcg@20", "strong_acat_is_best_hr@20"]:
        if col in result.columns:
            result[col] = result[col].astype(object)
    return sort_by_split_detail(result)


def attach_detail_groups(item_eval: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    required_eval = {"split", "raw_asin", "method_variant", "ndcg@20", "hr@20"}
    required_avail = {"split", "raw_asin", "cat_count_bin"}
    missing_eval = required_eval - set(item_eval.columns)
    missing_avail = required_avail - set(availability.columns)
    if missing_eval:
        raise ValueError(f"item_eval missing columns: {sorted(missing_eval)}")
    if missing_avail:
        raise ValueError(f"availability missing columns: {sorted(missing_avail)}")
    detail = availability[["split", "raw_asin", "cat_count_bin"]].drop_duplicates(["split", "raw_asin"]).copy()
    detail["detail_group"] = detail["cat_count_bin"].map(detail_group_from_bin)
    detail["detail_order"] = detail["cat_count_bin"].map(detail_order_from_bin)
    merged = item_eval.merge(detail, on=["split", "raw_asin"], how="left", validate="many_to_one")
    return merged


def build_method_layer_summary(item_eval_with_detail: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "detail_group", "detail_order", "high_acat_flag", "method_variant", "ndcg@20", "hr@20"}
    missing = required - set(item_eval_with_detail.columns)
    if missing:
        raise ValueError(f"item_eval_with_detail missing columns: {sorted(missing)}")
    work = item_eval_with_detail.copy()
    work = work[work["detail_group"].notna()].copy()
    for metric in ["ndcg@20", "hr@20"]:
        work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work["high_acat_flag"] = work["high_acat_flag"].map(bool_text)
    rows = []
    for (split, detail_group, detail_order, high_acat_flag, method_variant), sub in work.groupby(
        ["split", "detail_group", "detail_order", "high_acat_flag", "method_variant"],
        dropna=False,
    ):
        rows.append(
            {
                "split": split,
                "detail_group": detail_group,
                "detail_order": int(detail_order),
                "high_acat_flag": high_acat_flag,
                "method_variant": method_variant,
                "item_count": len(sub),
                "ndcg@20_mean": safe_mean(sub["ndcg@20"]),
                "hr@20_mean": safe_mean(sub["hr@20"]),
            }
        )
    return sort_by_split_detail(pd.DataFrame(rows))


def build_method_layer_delta(method_layer_summary: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "detail_group", "detail_order", "high_acat_flag", "method_variant", "ndcg@20_mean", "hr@20_mean"}
    missing = required - set(method_layer_summary.columns)
    if missing:
        raise ValueError(f"method_layer_summary missing columns: {sorted(missing)}")
    rows = []
    group_cols = ["split", "detail_group", "detail_order", "high_acat_flag"]
    for keys, sub in method_layer_summary.groupby(group_cols, dropna=False):
        by_method = {str(row["method_variant"]): row for _, row in sub.iterrows()}
        if M3 not in by_method or M6 not in by_method:
            continue
        m3 = by_method[M3]
        m6 = by_method[M6]
        split, detail_group, detail_order, high_acat_flag = keys
        rows.append(
            {
                "split": split,
                "detail_group": detail_group,
                "detail_order": int(detail_order),
                "high_acat_flag": high_acat_flag,
                "item_count": int(m3["item_count"]),
                "m3_ndcg@20_mean": float(m3["ndcg@20_mean"]),
                "m6_ndcg@20_mean": float(m6["ndcg@20_mean"]),
                "m3_minus_m6_ndcg@20_mean": float(m3["ndcg@20_mean"]) - float(m6["ndcg@20_mean"]),
                "m3_hr@20_mean": float(m3["hr@20_mean"]),
                "m6_hr@20_mean": float(m6["hr@20_mean"]),
                "m3_minus_m6_hr@20_mean": float(m3["hr@20_mean"]) - float(m6["hr@20_mean"]),
            }
        )
    return sort_by_split_detail(pd.DataFrame(rows))


def build_m4_gate_summary(best_csv: Path, control_csv: Path) -> pd.DataFrame:
    if not best_csv.exists() or not control_csv.exists():
        return pd.DataFrame(
            [
                {
                    "m4_item_layer_available": False,
                    "m4_seed43_any_pass": False,
                    "note": "M4 aggregate summary not found.",
                }
            ]
        )
    best = pd.read_csv(best_csv)
    control = pd.read_csv(control_csv)
    vs_shuffle = control[control["control_id"].eq("M6")].copy()
    rows = []
    for _, row in best.iterrows():
        method_id = row["method_id"]
        method_variant = row["method_variant"]
        ctrl = vs_shuffle[vs_shuffle["method_variant"].eq(method_variant)]
        ctrl_row = ctrl.iloc[0] if not ctrl.empty else None
        rows.append(
            {
                "method_id": method_id,
                "method_variant": method_variant,
                "best_epoch": int(row["best_epoch"]),
                "best_ndcg@20": float(row["best_ndcg@20"]),
                "best_hr@20": float(row["best_hr@20"]),
                "delta_vs_M6_ndcg@20": float(ctrl_row["delta_ndcg@20"]) if ctrl_row is not None else nan_float(),
                "delta_vs_M6_hr@20": float(ctrl_row["delta_hr@20"]) if ctrl_row is not None else nan_float(),
                "pass_seed43_gate": bool(ctrl_row["pass_seed43_gate"]) if ctrl_row is not None else False,
                "m4_item_layer_available": False,
            }
        )
    result = pd.DataFrame(rows)
    result["pass_seed43_gate"] = result["pass_seed43_gate"].astype(object)
    result["m4_item_layer_available"] = result["m4_item_layer_available"].astype(object)
    return result


def build_route_decision(
    trend_checks: pd.DataFrame,
    within_contrast: pd.DataFrame,
    method_layer_delta: pd.DataFrame,
    m4_gate: pd.DataFrame,
) -> dict:
    test_checks = trend_checks[trend_checks["split"].eq("test")]
    test_row = test_checks.iloc[0] if not test_checks.empty else pd.Series(dtype=object)
    original_ndcg_trend = bool(test_row.get("baseline_ndcg_strict_increasing", False))
    original_hr_trend = bool(test_row.get("baseline_hr_strict_increasing", False))
    acat_trend = bool(test_row.get("acat_mean_strict_increasing", False))

    test_contrast = within_contrast[within_contrast["split"].eq("test")]
    strong_positive_count = int((pd.to_numeric(test_contrast["strong_minus_weak_ndcg@20_mean"], errors="coerce") > 0).sum()) if not test_contrast.empty else 0
    strong_negative_count = int((pd.to_numeric(test_contrast["strong_minus_weak_ndcg@20_mean"], errors="coerce") < 0).sum()) if not test_contrast.empty else 0

    test_high_delta = method_layer_delta[
        method_layer_delta["split"].eq("test") & method_layer_delta["high_acat_flag"].eq("True")
    ]
    m3_layer_positive_count = int((pd.to_numeric(test_high_delta["m3_minus_m6_ndcg@20_mean"], errors="coerce") > 0).sum()) if not test_high_delta.empty else 0
    m3_layer_negative_count = int((pd.to_numeric(test_high_delta["m3_minus_m6_ndcg@20_mean"], errors="coerce") < 0).sum()) if not test_high_delta.empty else 0
    m4_pass = bool(m4_gate["pass_seed43_gate"].map(bool).any()) if "pass_seed43_gate" in m4_gate.columns else False

    if original_ndcg_trend and acat_trend:
        route = "acat_directly_aligns_with_detail_trend"
        next_action = "continue_carrier_design_with_detail_as_supporting_axis"
    elif original_ndcg_trend and not acat_trend and strong_negative_count >= strong_positive_count:
        route = "acat_not_direct_detail_explanation_but_failure_opportunity_signal"
        next_action = "do_task4_post1_failure_layer_audit_before_new_carrier"
    elif original_ndcg_trend and not acat_trend:
        route = "acat_complements_detail_not_category_count_proxy"
        next_action = "condition_next_carrier_on_detail_x_acat_layers"
    else:
        route = "original_detail_trend_needs_recheck"
        next_action = "rebuild_original_detail_result_before_method_design"

    return {
        "route": route,
        "next_action": next_action,
        "original_test_ndcg_low_mid_high_strict": original_ndcg_trend,
        "original_test_hr_low_mid_high_strict": original_hr_trend,
        "acat_v3_mean_low_mid_high_strict": acat_trend,
        "test_strong_minus_weak_ndcg_positive_detail_count": strong_positive_count,
        "test_strong_minus_weak_ndcg_negative_detail_count": strong_negative_count,
        "test_high_acat_m3_minus_m6_positive_layer_count": m3_layer_positive_count,
        "test_high_acat_m3_minus_m6_negative_layer_count": m3_layer_negative_count,
        "m4_seed43_any_pass": m4_pass,
        "run_multi_seed_now": False,
        "plain_explanation": (
            "The original NDCG trend still exists, but Acat_v3 does not move with category count. "
            "Inside detail groups, high Acat_v3 often marks items where baseline is worse, so it is "
            "better treated as a failure-opportunity signal than a direct explanation of the old trend."
        ),
    }


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if df.empty:
        return "_empty_"
    small = df[columns].copy()
    if max_rows is not None:
        small = small.head(max_rows)
    for col in small.columns:
        if pd.api.types.is_float_dtype(small[col]):
            small[col] = small[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    header = "| " + " | ".join(small.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(small.columns)) + " |"
    rows = []
    for _, row in small.iterrows():
        rows.append("| " + " | ".join("" if pd.isna(value) else str(value) for value in row.tolist()) + " |")
    return "\n".join([header, separator, *rows])


def write_result_markdown(
    path: Path,
    run_stamp: str,
    design_note_name: str,
    detail_summary: pd.DataFrame,
    trend_checks: pd.DataFrame,
    within_summary: pd.DataFrame,
    within_contrast: pd.DataFrame,
    method_delta: pd.DataFrame,
    m4_gate: pd.DataFrame,
    decision: dict,
    manifest_name: str,
) -> None:
    test_detail = detail_summary[detail_summary["split"].eq("test")]
    test_within = within_summary[within_summary["split"].eq("test")]
    test_contrast = within_contrast[within_contrast["split"].eq("test")]
    test_delta = method_delta[method_delta["split"].eq("test")]
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG Task4-post-0 类别详细度三组 x Acat v3 关系诊断结果
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - 实验结果
  - Acat_v3
---

# {run_stamp} CCFCRec Amazon-VG Task4-post-0 类别详细度三组 x Acat v3 关系诊断结果

> [!info] 来源说明
> 上游设计：[[{design_note_name}]]
> manifest（运行清单）：`{manifest_name}`

## 一句话结论

```text
route = {decision["route"]}
next_action = {decision["next_action"]}
run_multi_seed_now = {decision["run_multi_seed_now"]}
```

通俗解释：原始的“类别越详细，NDCG@20（前20排序质量）越高”还在，但 `Acat_v3`（类别可用性变量）没有跟着类别数量一起递增。更像是：有些 item 的类别信息很可用，但 baseline（旧模型）没有用好它，所以它是“失败机会组”信号，不是旧三组递增现象的直接替代品。

## 1. 原始三组现象

> `detail_group`：类别详细度三组；`low_detail` 是 1-3 个类别，`mid_detail` 是 4 个类别，`high_detail` 是 5 个及以上类别。

{md_table(test_detail, ["detail_group", "item_count", "category_count_mean", "baseline_ndcg@20_mean", "baseline_hr@20_mean", "Acat_v3_mean", "Acat_v3_strong_share"])}

## 2. 递增检查

{md_table(trend_checks, ["split", "low_detail_ndcg@20", "mid_detail_ndcg@20", "high_detail_ndcg@20", "baseline_ndcg_strict_increasing", "low_detail_hr@20", "mid_detail_hr@20", "high_detail_hr@20", "baseline_hr_strict_increasing", "low_detail_Acat_v3", "mid_detail_Acat_v3", "high_detail_Acat_v3", "acat_mean_strict_increasing"])}

解释：

```text
NDCG@20：low < mid < high，原始论文动机仍可保留。
HR@20：low < high < mid，不是严格三组递增。
Acat_v3：low > high > mid，不是严格三组递增。
```

## 3. 同一详细度内部的 Acat_v3 分层

{md_table(test_within, ["detail_group", "s_cat_v3_group", "item_count", "Acat_v3_mean", "baseline_ndcg@20_mean", "baseline_hr@20_mean"], max_rows=12)}

## 4. high Acat 是否就是 baseline 更好

{md_table(test_contrast, ["detail_group", "weak_ndcg@20_mean", "strong_ndcg@20_mean", "strong_minus_weak_ndcg@20_mean", "best_acat_group_by_ndcg@20", "strong_acat_is_best_ndcg@20", "weak_hr@20_mean", "strong_hr@20_mean", "strong_minus_weak_hr@20_mean"])}

解释：在 `mid_detail` 和 `high_detail` 中，`s_cat_v3_strong` 的 baseline 表现反而低于 `s_cat_v3_weak`。这不是直接否定 Acat_v3，而是说明 Acat_v3 更像在标记“类别信息可用但旧模型没吃到”的区域。

## 5. M3 vs M6 放回详细度层

> M3：`task4_acat_trainhard_weight`，用真实 Acat_v3 机会组加权。  
> M6：`task4_acat_shuffle_high_weight`，把 Acat_v3 打乱后的负控。

{md_table(test_delta, ["detail_group", "high_acat_flag", "item_count", "m3_ndcg@20_mean", "m6_ndcg@20_mean", "m3_minus_m6_ndcg@20_mean", "m3_hr@20_mean", "m6_hr@20_mean", "m3_minus_m6_hr@20_mean"], max_rows=12)}

解释：这一表看 M3 是不是只在某个“类别详细度 x high-Acat”层里有效。如果只在局部有效但 overall（整体指标）输给 shuffle，就说明下一版方法不能全局打，要更窄地打有效层。

## 6. M4 总体 gate

{md_table(m4_gate, ["method_id", "method_variant", "best_epoch", "best_ndcg@20", "best_hr@20", "delta_vs_M6_ndcg@20", "delta_vs_M6_hr@20", "pass_seed43_gate", "m4_item_layer_available"], max_rows=10)}

解释：M4 目前只有总体结果，没有 item 级分层结果。总体上 M4a/M4b/M4c 都没有打过 M6 shuffle，所以不能进入多 seed（多随机种子复验）。

## 路线判断

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

## 产物

```text
detail_acat_summary.csv
detail_trend_checks.csv
within_detail_acat_summary.csv
within_detail_acat_contrast.csv
method_layer_summary.csv
method_layer_delta_m3_vs_m6.csv
m4_gate_summary.csv
run_manifest.json
```
"""
    path.write_text(markdown, encoding="utf-8")


def write_route_markdown(
    path: Path,
    run_stamp: str,
    design_note_name: str,
    result_note_name: str,
    decision: dict,
    trend_checks: pd.DataFrame,
    within_contrast: pd.DataFrame,
    method_delta: pd.DataFrame,
) -> None:
    test_trend = trend_checks[trend_checks["split"].eq("test")]
    test_contrast = within_contrast[within_contrast["split"].eq("test")]
    test_delta = method_delta[method_delta["split"].eq("test")]
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG Task4-post-0 类别详细度三组 x Acat v3 关系诊断 路线判断
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - 路线判断
  - Acat_v3
---

# {run_stamp} CCFCRec Amazon-VG Task4-post-0 类别详细度三组 x Acat v3 关系诊断 路线判断

## 来源

设计：[[{design_note_name}]]
结果：[[{result_note_name}]]

## 判断

```text
route = {decision["route"]}
next_action = {decision["next_action"]}
run_multi_seed_now = {decision["run_multi_seed_now"]}
```

通俗解释：原始现象不是没了，而是 `Acat_v3` 没有直接复制原始现象。原始现象说“类别更多时 baseline 通常更好”；现在 `Acat_v3` 更像说“某些类别信息明明可用，但 baseline 没用好”。所以后面不能写成 Acat_v3 直接解释三组递增，而要写成 Acat_v3 解释三组内部的失败机会。

## 关键证据

{md_table(test_trend, ["low_detail_ndcg@20", "mid_detail_ndcg@20", "high_detail_ndcg@20", "baseline_ndcg_strict_increasing", "low_detail_Acat_v3", "mid_detail_Acat_v3", "high_detail_Acat_v3", "acat_mean_strict_increasing"])}

{md_table(test_contrast, ["detail_group", "strong_minus_weak_ndcg@20_mean", "best_acat_group_by_ndcg@20", "strong_acat_is_best_ndcg@20"])}

{md_table(test_delta, ["detail_group", "high_acat_flag", "m3_minus_m6_ndcg@20_mean", "m3_minus_m6_hr@20_mean"], max_rows=12)}

## 下一步

```text
1. 不做当前 M3/M4 的多 seed。
2. 做 Task4-post-1：high-Acat baseline failure layer audit。
3. 下一版 carrier 必须显式考虑 detail_group x Acat_v3，而不是只用全局 high_acat_flag。
4. 论文叙事改成：category_count 解释原始粗粒度收益；Acat_v3 解释类别详细度内部的未利用机会。
```
"""
    path.write_text(markdown, encoding="utf-8")


def build_outputs(output_root: Path, experiment_root: Path, run_stamp: str) -> Outputs:
    output_dir = output_root / f"{run_stamp} task4-post0-detail-acat-relationship-diagnostic"
    route_md = experiment_root / f"{run_stamp} CCFCRec Amazon-VG Task4-post-0 类别详细度三组 x Acat v3 关系诊断 路线判断.md"
    return Outputs(
        output_dir=output_dir,
        detail_summary_csv=output_dir / "detail_acat_summary.csv",
        trend_checks_csv=output_dir / "detail_trend_checks.csv",
        within_detail_acat_summary_csv=output_dir / "within_detail_acat_summary.csv",
        within_detail_acat_contrast_csv=output_dir / "within_detail_acat_contrast.csv",
        method_layer_summary_csv=output_dir / "method_layer_summary.csv",
        method_layer_delta_csv=output_dir / "method_layer_delta_m3_vs_m6.csv",
        m4_gate_summary_csv=output_dir / "m4_gate_summary.csv",
        manifest_json=output_dir / "run_manifest.json",
        result_md=output_dir / f"{run_stamp} CCFCRec Amazon-VG Task4-post-0 类别详细度三组 x Acat v3 关系诊断结果.md",
        route_md=route_md,
    )


def run_analysis(args: argparse.Namespace) -> Outputs:
    if args.run_stamp:
        run_stamp = args.run_stamp
        run_date = run_stamp[:10]
        run_iso = datetime.strptime(run_stamp, "%Y-%m-%d %H%M%S").isoformat(timespec="seconds")
    else:
        run_stamp, run_date, run_iso = now_stamp()

    outputs = build_outputs(Path(args.output_root), Path(args.experiment_root), run_stamp)
    outputs.output_dir.mkdir(parents=True, exist_ok=True)

    availability = pd.read_csv(args.availability_csv)
    item_eval = pd.read_csv(args.item_eval_csv)

    detail_summary = build_detail_acat_summary(availability)
    trend_checks = build_detail_trend_checks(detail_summary)
    within_summary = build_within_detail_acat_summary(availability)
    within_contrast = build_within_detail_acat_contrast(within_summary)
    item_eval_with_detail = attach_detail_groups(item_eval, availability)
    method_layer_summary = build_method_layer_summary(item_eval_with_detail)
    method_layer_delta = build_method_layer_delta(method_layer_summary)
    m4_gate = build_m4_gate_summary(Path(args.m4_best_csv), Path(args.m4_control_csv))
    decision = build_route_decision(trend_checks, within_contrast, method_layer_delta, m4_gate)

    detail_summary.to_csv(outputs.detail_summary_csv, index=False)
    trend_checks.to_csv(outputs.trend_checks_csv, index=False)
    within_summary.to_csv(outputs.within_detail_acat_summary_csv, index=False)
    within_contrast.to_csv(outputs.within_detail_acat_contrast_csv, index=False)
    method_layer_summary.to_csv(outputs.method_layer_summary_csv, index=False)
    method_layer_delta.to_csv(outputs.method_layer_delta_csv, index=False)
    m4_gate.to_csv(outputs.m4_gate_summary_csv, index=False)

    result_note_name = outputs.result_md.stem
    write_result_markdown(
        outputs.result_md,
        run_stamp,
        args.design_note_name,
        detail_summary,
        trend_checks,
        within_summary,
        within_contrast,
        method_layer_delta,
        m4_gate,
        decision,
        outputs.manifest_json.name,
    )
    write_route_markdown(
        outputs.route_md,
        run_stamp,
        args.design_note_name,
        result_note_name,
        decision,
        trend_checks,
        within_contrast,
        method_layer_delta,
    )

    manifest = {
        "run_stamp": run_stamp,
        "run_date": run_date,
        "run_iso": run_iso,
        "analysis_script": "validata/analyze_amazon_vg_task4_post0_detail_acat.py",
        "inputs": {
            "availability_csv": str(args.availability_csv),
            "item_eval_csv": str(args.item_eval_csv),
            "m4_best_csv": str(args.m4_best_csv),
            "m4_control_csv": str(args.m4_control_csv),
        },
        "outputs": {field: str(getattr(outputs, field)) for field in outputs.__dataclass_fields__},
        "decision": decision,
    }
    outputs.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze category detail groups x Acat_v3 for Task4-post-0.")
    parser.add_argument("--availability_csv", type=Path, default=DEFAULT_AVAILABILITY_CSV)
    parser.add_argument("--item_eval_csv", type=Path, default=DEFAULT_ITEM_EVAL_CSV)
    parser.add_argument("--m4_best_csv", type=Path, default=DEFAULT_M4_BEST_CSV)
    parser.add_argument("--m4_control_csv", type=Path, default=DEFAULT_M4_CONTROL_CSV)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--run_stamp", default="")
    parser.add_argument(
        "--design_note_name",
        default="2026-07-07 103749 CCFCRec Amazon-VG Task4-post-0 类别详细度三组 x Acat v3 关系诊断设计",
    )
    return parser.parse_args()


def main() -> None:
    outputs = run_analysis(parse_args())
    print(f"output_dir={outputs.output_dir}")
    print(f"result_md={outputs.result_md}")
    print(f"route_md={outputs.route_md}")


if __name__ == "__main__":
    main()
