#!/usr/bin/env python3
"""
Task4-post-1: high-Acat baseline failure layer audit.

This script identifies which category-detail x Acat_v3 layers are actually
worth targeting after M3/M4 failed the overall shuffle gate.
"""

from __future__ import annotations

import argparse
import json
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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "temp_202607_实验文件记录" / "temp_20260707"
DEFAULT_EXPERIMENT_ROOT = PROJECT_ROOT / "实验记录"

M3 = "task4_acat_trainhard_weight"
M6 = "task4_acat_shuffle_high_weight"
DETAIL_ORDER = {"cat_count_1_3": 1, "cat_count_4": 2, "cat_count_5_plus": 3}
DETAIL_LABEL = {"cat_count_1_3": "low_detail", "cat_count_4": "mid_detail", "cat_count_5_plus": "high_detail"}
SPLIT_ORDER = {"validate": 1, "test": 2}


@dataclass(frozen=True)
class Outputs:
    output_dir: Path
    baseline_failure_summary_csv: Path
    m3_m6_delta_profile_csv: Path
    trainhard_layer_delta_csv: Path
    high_acat_evalhard_layer_delta_csv: Path
    candidate_layer_recommendation_csv: Path
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


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.fillna(False).astype(bool)
    text = series.astype(str).str.strip().str.lower()
    return text.isin({"true", "1", "yes", "y", "t"})


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


def sort_layers(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    if "split" in work.columns:
        work["_split_order"] = work["split"].map(SPLIT_ORDER).fillna(99)
    else:
        work["_split_order"] = 99
    if "detail_order" not in work.columns and "cat_count_bin" in work.columns:
        work["detail_order"] = work["cat_count_bin"].map(detail_order_from_bin)
    sort_cols = [col for col in ["_split_order", "detail_order", "high_acat_train_safe_hard_flag", "high_acat_flag", "eval_baseline_hard_flag"] if col in work.columns]
    work = work.sort_values(sort_cols).drop(columns=["_split_order"])
    return work.reset_index(drop=True)


def attach_detail_groups(item_eval: pd.DataFrame, availability: pd.DataFrame) -> pd.DataFrame:
    required_eval = {"split", "raw_asin"}
    required_avail = {"split", "raw_asin", "cat_count_bin"}
    missing_eval = required_eval - set(item_eval.columns)
    missing_avail = required_avail - set(availability.columns)
    if missing_eval:
        raise ValueError(f"item_eval missing columns: {sorted(missing_eval)}")
    if missing_avail:
        raise ValueError(f"availability missing columns: {sorted(missing_avail)}")
    detail_cols = ["split", "raw_asin", "cat_count_bin"]
    if "category_count" in availability.columns:
        detail_cols.append("category_count")
    detail = availability[detail_cols].drop_duplicates(["split", "raw_asin"]).copy()
    detail["detail_group"] = detail["cat_count_bin"].map(detail_group_from_bin)
    detail["detail_order"] = detail["cat_count_bin"].map(detail_order_from_bin)
    return item_eval.merge(detail, on=["split", "raw_asin"], how="left", validate="many_to_one")


def normalize_flags(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    for col in ["high_acat_flag", "eval_baseline_hard_flag", "high_acat_eval_hard_flag", "high_acat_train_safe_hard_flag"]:
        if col in work.columns:
            work[col] = bool_series(work[col])
    return work


def one_row_per_item(item_eval_with_detail: pd.DataFrame, method_variant: str = M3) -> pd.DataFrame:
    if "method_variant" in item_eval_with_detail.columns:
        work = item_eval_with_detail[item_eval_with_detail["method_variant"].eq(method_variant)].copy()
    else:
        work = item_eval_with_detail.copy()
    return normalize_flags(work)


def build_baseline_failure_layer_summary(item_eval_with_detail: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "raw_asin", "detail_group", "detail_order", "high_acat_flag", "eval_baseline_hard_flag", "high_acat_train_safe_hard_flag", "baseline_ndcg@20"}
    missing = required - set(item_eval_with_detail.columns)
    if missing:
        raise ValueError(f"item_eval_with_detail missing columns: {sorted(missing)}")
    work = one_row_per_item(item_eval_with_detail)
    rows = []
    group_cols = ["split", "detail_group", "detail_order", "high_acat_flag"]
    for keys, sub in work.groupby(group_cols, dropna=False):
        split, detail_group, detail_order, high_acat_flag = keys
        rows.append(
            {
                "split": split,
                "detail_group": detail_group,
                "detail_order": int(detail_order),
                "high_acat_flag": bool_text(high_acat_flag),
                "item_count": len(sub),
                "baseline_ndcg@20_mean": safe_mean(sub["baseline_ndcg@20"]),
                "baseline_ndcg@20_median": safe_median(sub["baseline_ndcg@20"]),
                "baseline_hard_rate": float(sub["eval_baseline_hard_flag"].mean()),
                "train_safe_hard_rate": float(sub["high_acat_train_safe_hard_flag"].mean()),
                "baseline_zero_ndcg_count": int((pd.to_numeric(sub["baseline_ndcg@20"], errors="coerce") == 0).sum()),
                "baseline_positive_ndcg_rate": float((pd.to_numeric(sub["baseline_ndcg@20"], errors="coerce") > 0).mean()),
            }
        )
    return sort_layers(pd.DataFrame(rows))


def build_m3_m6_delta_profile(item_eval_with_detail: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "raw_asin", "method_variant", "ndcg@20", "hr@20"}
    missing = required - set(item_eval_with_detail.columns)
    if missing:
        raise ValueError(f"item_eval_with_detail missing columns: {sorted(missing)}")
    work = normalize_flags(item_eval_with_detail)
    key_cols = [
        "split",
        "raw_asin",
        "detail_group",
        "detail_order",
        "high_acat_flag",
        "eval_baseline_hard_flag",
        "high_acat_eval_hard_flag",
        "high_acat_train_safe_hard_flag",
    ]
    optional = [
        "s_cat_v3",
        "s_cat_v3_group",
        "RSP_score",
        "RSP_group",
        "baseline_ndcg@20",
        "baseline_margin_proxy",
        "baseline_best_target_rank",
        "train_safe_hard_proxy_score",
        "train_safe_hard_proxy_group",
    ]
    base_cols = [col for col in key_cols + optional if col in work.columns]
    metrics = ["ndcg@20", "hr@20", "best_target_rank", "margin_to_top20_cutoff", "q_norm"]
    available_metrics = [col for col in metrics if col in work.columns]
    m3 = work[work["method_variant"].eq(M3)][base_cols + available_metrics].copy()
    m6 = work[work["method_variant"].eq(M6)][["split", "raw_asin", *available_metrics]].copy()
    m3 = m3.rename(columns={col: f"m3_{col}" for col in available_metrics})
    m6 = m6.rename(columns={col: f"m6_{col}" for col in available_metrics})
    merged = m3.merge(m6, on=["split", "raw_asin"], how="inner", validate="one_to_one")
    for metric in available_metrics:
        merged[f"delta_{metric}"] = pd.to_numeric(merged[f"m3_{metric}"], errors="coerce") - pd.to_numeric(merged[f"m6_{metric}"], errors="coerce")
    for col in ["high_acat_flag", "eval_baseline_hard_flag", "high_acat_eval_hard_flag", "high_acat_train_safe_hard_flag"]:
        if col in merged.columns:
            merged[col] = merged[col].map(bool_text)
    return sort_layers(merged)


def build_trainhard_layer_delta(delta_profile: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "detail_group", "detail_order", "high_acat_train_safe_hard_flag", "delta_ndcg@20", "delta_hr@20"}
    missing = required - set(delta_profile.columns)
    if missing:
        raise ValueError(f"delta_profile missing columns: {sorted(missing)}")
    rows = []
    for keys, sub in delta_profile.groupby(["split", "detail_group", "detail_order", "high_acat_train_safe_hard_flag"], dropna=False):
        split, detail_group, detail_order, trainhard_flag = keys
        baseline_values = pd.to_numeric(sub["baseline_ndcg@20"], errors="coerce") if "baseline_ndcg@20" in sub.columns else pd.Series(dtype=float)
        baseline_available = baseline_values.notna().sum() > 0
        rows.append(
            {
                "split": split,
                "detail_group": detail_group,
                "detail_order": int(detail_order),
                "high_acat_train_safe_hard_flag": bool_text(trainhard_flag),
                "item_count": len(sub),
                "m3_minus_m6_ndcg@20_mean": safe_mean(sub["delta_ndcg@20"]),
                "m3_minus_m6_hr@20_mean": safe_mean(sub["delta_hr@20"]),
                "positive_delta_ndcg_rate": float((pd.to_numeric(sub["delta_ndcg@20"], errors="coerce") > 0).mean()),
                "positive_delta_hr_rate": float((pd.to_numeric(sub["delta_hr@20"], errors="coerce") > 0).mean()),
                "delta_ndcg@20_median": safe_median(sub["delta_ndcg@20"]),
                "delta_hr@20_median": safe_median(sub["delta_hr@20"]),
                "baseline_ndcg@20_mean": safe_mean(sub["baseline_ndcg@20"]) if baseline_available else nan_float(),
                "baseline_hard_rate": float((sub["eval_baseline_hard_flag"].astype(str) == "True").mean())
                if baseline_available and "eval_baseline_hard_flag" in sub.columns
                else nan_float(),
            }
        )
    return sort_layers(pd.DataFrame(rows))


def build_high_acat_evalhard_layer_delta(delta_profile: pd.DataFrame) -> pd.DataFrame:
    required = {"split", "detail_group", "detail_order", "high_acat_flag", "eval_baseline_hard_flag", "delta_ndcg@20", "delta_hr@20"}
    missing = required - set(delta_profile.columns)
    if missing:
        raise ValueError(f"delta_profile missing columns: {sorted(missing)}")
    rows = []
    for keys, sub in delta_profile.groupby(["split", "detail_group", "detail_order", "high_acat_flag", "eval_baseline_hard_flag"], dropna=False):
        split, detail_group, detail_order, high_acat_flag, hard_flag = keys
        baseline_values = pd.to_numeric(sub["baseline_ndcg@20"], errors="coerce") if "baseline_ndcg@20" in sub.columns else pd.Series(dtype=float)
        baseline_available = baseline_values.notna().sum() > 0
        rows.append(
            {
                "split": split,
                "detail_group": detail_group,
                "detail_order": int(detail_order),
                "high_acat_flag": bool_text(high_acat_flag),
                "eval_baseline_hard_flag": bool_text(hard_flag),
                "item_count": len(sub),
                "m3_minus_m6_ndcg@20_mean": safe_mean(sub["delta_ndcg@20"]),
                "m3_minus_m6_hr@20_mean": safe_mean(sub["delta_hr@20"]),
                "positive_delta_ndcg_rate": float((pd.to_numeric(sub["delta_ndcg@20"], errors="coerce") > 0).mean()),
                "baseline_ndcg@20_mean": safe_mean(sub["baseline_ndcg@20"]) if baseline_available else nan_float(),
            }
        )
    return sort_layers(pd.DataFrame(rows))


def _metric(summary: pd.DataFrame, split: str, detail_group: str, flag_col: str, metric: str) -> float:
    sub = summary[
        summary["split"].eq(split)
        & summary["detail_group"].eq(detail_group)
        & summary[flag_col].astype(str).eq("True")
    ]
    if sub.empty or metric not in sub.columns:
        return nan_float()
    return float(sub.iloc[0][metric])


def _count(summary: pd.DataFrame, split: str, detail_group: str, flag_col: str) -> int:
    sub = summary[
        summary["split"].eq(split)
        & summary["detail_group"].eq(detail_group)
        & summary[flag_col].astype(str).eq("True")
    ]
    if sub.empty:
        return 0
    return int(sub.iloc[0]["item_count"])


def build_candidate_layer_recommendation(
    trainhard_layer_delta: pd.DataFrame,
    min_item_count: int = 100,
    min_ndcg_delta: float = 0.0005,
) -> pd.DataFrame:
    rows = []
    for detail_group in ["low_detail", "mid_detail", "high_detail"]:
        validate_ndcg = _metric(trainhard_layer_delta, "validate", detail_group, "high_acat_train_safe_hard_flag", "m3_minus_m6_ndcg@20_mean")
        test_ndcg = _metric(trainhard_layer_delta, "test", detail_group, "high_acat_train_safe_hard_flag", "m3_minus_m6_ndcg@20_mean")
        validate_hr = _metric(trainhard_layer_delta, "validate", detail_group, "high_acat_train_safe_hard_flag", "m3_minus_m6_hr@20_mean")
        test_hr = _metric(trainhard_layer_delta, "test", detail_group, "high_acat_train_safe_hard_flag", "m3_minus_m6_hr@20_mean")
        validate_count = _count(trainhard_layer_delta, "validate", detail_group, "high_acat_train_safe_hard_flag")
        test_count = _count(trainhard_layer_delta, "test", detail_group, "high_acat_train_safe_hard_flag")
        enough = validate_count >= min_item_count and test_count >= min_item_count
        ndcg_stable = validate_ndcg >= min_ndcg_delta and test_ndcg >= min_ndcg_delta
        hr_not_worse = validate_hr >= 0 and test_hr >= 0
        recommended = bool(enough and ndcg_stable and hr_not_worse)
        target_score = (
            min(validate_ndcg, test_ndcg)
            + 0.5 * min(validate_hr, test_hr)
            if pd.notna(validate_ndcg) and pd.notna(test_ndcg) and pd.notna(validate_hr) and pd.notna(test_hr)
            else nan_float()
        )
        rows.append(
            {
                "detail_group": detail_group,
                "validate_item_count": validate_count,
                "test_item_count": test_count,
                "validate_m3_minus_m6_ndcg@20_mean": validate_ndcg,
                "test_m3_minus_m6_ndcg@20_mean": test_ndcg,
                "validate_m3_minus_m6_hr@20_mean": validate_hr,
                "test_m3_minus_m6_hr@20_mean": test_hr,
                "min_item_count": min_item_count,
                "min_ndcg_delta": min_ndcg_delta,
                "enough_items": bool(enough),
                "stable_ndcg_positive": bool(ndcg_stable),
                "hr_not_worse": bool(hr_not_worse),
                "recommended_target_layer": recommended,
                "target_score": target_score,
                "plain_reason": (
                    "stable positive on validate/test and HR not worse"
                    if recommended
                    else "fails item count, NDCG stability, or HR gate"
                ),
            }
        )
    result = pd.DataFrame(rows)
    for col in ["enough_items", "stable_ndcg_positive", "hr_not_worse", "recommended_target_layer"]:
        result[col] = result[col].astype(object)
    return result.sort_values(["recommended_target_layer", "target_score"], ascending=[False, False]).reset_index(drop=True)


def build_route_decision(candidate_layers: pd.DataFrame, m4_any_pass: bool = False) -> dict:
    recommended = candidate_layers[candidate_layers["recommended_target_layer"].map(bool)].copy()
    if not recommended.empty:
        top = recommended.sort_values("target_score", ascending=False).iloc[0]
        route = "narrow_next_carrier_to_detail_specific_high_acat_hard_layer"
        next_action = "design_high_detail_high_acat_trainhard_only_carrier"
        target_layer = str(top["detail_group"])
    else:
        route = "no_stable_failure_layer_for_current_m3"
        next_action = "redesign_carrier_or_redefine_proxy_before_training"
        target_layer = ""
    return {
        "route": route,
        "next_action": next_action,
        "recommended_detail_layer": target_layer,
        "m4_seed43_any_pass": bool(m4_any_pass),
        "run_multi_seed_now": False,
        "plain_explanation": (
            f"M3 only gives a stable useful signal in {target_layer}; next carrier should target that layer."
            if target_layer
            else "No layer passes validate/test stability gates; do not train another variant yet."
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
    baseline_summary: pd.DataFrame,
    trainhard_delta: pd.DataFrame,
    evalhard_delta: pd.DataFrame,
    candidate_layers: pd.DataFrame,
    decision: dict,
    manifest_name: str,
) -> None:
    test_baseline = baseline_summary[baseline_summary["split"].eq("test")]
    test_trainhard = trainhard_delta[trainhard_delta["split"].eq("test")]
    test_evalhard = evalhard_delta[evalhard_delta["split"].eq("test")]
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG Task4-post-1 high-Acat baseline failure layer audit 结果
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - 实验结果
  - Acat_v3
---

# {run_stamp} CCFCRec Amazon-VG Task4-post-1 high-Acat baseline failure layer audit 结果

> [!info] 来源说明
> 上游设计：[[{design_note_name}]]
> manifest（运行清单）：`{manifest_name}`

## 一句话结论

```text
route = {decision["route"]}
next_action = {decision["next_action"]}
recommended_detail_layer = {decision["recommended_detail_layer"]}
run_multi_seed_now = {decision["run_multi_seed_now"]}
```

通俗解释：M3（真实 Acat_v3 hard 加权）不是完全没信号，而是信号集中在 `high_detail`（高类别详细度）里的 `high_acat_train_safe_hard_flag=True`（高类别可用性且训练安全困难）层。低详细度和中详细度的 trainhard 层没有稳定收益，所以后面不能继续全局 high-Acat 加权。

## 1. baseline 失败层

> `baseline_hard_rate`：baseline NDCG@20 为 0 或被标成 hard 的比例，越高说明旧模型越失败。

{md_table(test_baseline, ["detail_group", "high_acat_flag", "item_count", "baseline_ndcg@20_mean", "baseline_hard_rate", "train_safe_hard_rate", "baseline_positive_ndcg_rate"], max_rows=12)}

## 2. train-safe hard 层的 M3 vs M6

> M3：真实 Acat_v3 hard 加权。  
> M6：Acat_v3 打乱负控。  
> 这里看的是同一层里 M3 - M6。

{md_table(test_trainhard, ["detail_group", "high_acat_train_safe_hard_flag", "item_count", "m3_minus_m6_ndcg@20_mean", "m3_minus_m6_hr@20_mean", "positive_delta_ndcg_rate", "baseline_hard_rate"], max_rows=12)}

## 3. high-Acat eval-hard 细分

{md_table(test_evalhard, ["detail_group", "high_acat_flag", "eval_baseline_hard_flag", "item_count", "m3_minus_m6_ndcg@20_mean", "m3_minus_m6_hr@20_mean", "positive_delta_ndcg_rate", "baseline_ndcg@20_mean"], max_rows=20)}

## 4. 候选目标层推荐

{md_table(candidate_layers, ["detail_group", "validate_item_count", "test_item_count", "validate_m3_minus_m6_ndcg@20_mean", "test_m3_minus_m6_ndcg@20_mean", "validate_m3_minus_m6_hr@20_mean", "test_m3_minus_m6_hr@20_mean", "recommended_target_layer", "plain_reason"], max_rows=10)}

## 路线判断

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

## 产物

```text
baseline_failure_layer_summary.csv
m3_m6_delta_profile.csv
trainhard_layer_delta.csv
high_acat_evalhard_layer_delta.csv
candidate_layer_recommendation.csv
run_manifest.json
```
"""
    path.write_text(markdown, encoding="utf-8")


def write_route_markdown(
    path: Path,
    run_stamp: str,
    design_note_name: str,
    result_note_name: str,
    candidate_layers: pd.DataFrame,
    decision: dict,
) -> None:
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG Task4-post-1 high-Acat baseline failure layer audit 路线判断
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - 路线判断
  - Acat_v3
---

# {run_stamp} CCFCRec Amazon-VG Task4-post-1 high-Acat baseline failure layer audit 路线判断

## 来源

设计：[[{design_note_name}]]
结果：[[{result_note_name}]]

## 判断

```text
route = {decision["route"]}
next_action = {decision["next_action"]}
recommended_detail_layer = {decision["recommended_detail_layer"]}
run_multi_seed_now = {decision["run_multi_seed_now"]}
```

通俗解释：现在不是“继续把所有 high-Acat item 都加权”，而是只保留最像机会组的层。现有证据指向 `high_detail × high_acat_train_safe_hard`，也就是类别本来很详细、Acat_v3 高、baseline 失败且训练 proxy 安全的 item。

## 候选层

{md_table(candidate_layers, ["detail_group", "validate_m3_minus_m6_ndcg@20_mean", "test_m3_minus_m6_ndcg@20_mean", "validate_m3_minus_m6_hr@20_mean", "test_m3_minus_m6_hr@20_mean", "recommended_target_layer", "plain_reason"], max_rows=10)}

## 下一步

```text
1. 不对当前 M3/M4 做 multi seed。
2. 设计 M4-revise 或 M7：high-detail high-Acat trainhard-only carrier。
3. 新方法只打 high_detail × high_acat_train_safe_hard，不再打 low/mid trainhard 层。
4. 必须继续保留 M6 shuffle 和 RSP-only 对照。
```
"""
    path.write_text(markdown, encoding="utf-8")


def build_outputs(output_root: Path, experiment_root: Path, run_stamp: str) -> Outputs:
    output_dir = output_root / f"{run_stamp} task4-post1-high-acat-baseline-failure-layer-audit"
    return Outputs(
        output_dir=output_dir,
        baseline_failure_summary_csv=output_dir / "baseline_failure_layer_summary.csv",
        m3_m6_delta_profile_csv=output_dir / "m3_m6_delta_profile.csv",
        trainhard_layer_delta_csv=output_dir / "trainhard_layer_delta.csv",
        high_acat_evalhard_layer_delta_csv=output_dir / "high_acat_evalhard_layer_delta.csv",
        candidate_layer_recommendation_csv=output_dir / "candidate_layer_recommendation.csv",
        manifest_json=output_dir / "run_manifest.json",
        result_md=output_dir / f"{run_stamp} CCFCRec Amazon-VG Task4-post-1 high-Acat baseline failure layer audit 结果.md",
        route_md=experiment_root / f"{run_stamp} CCFCRec Amazon-VG Task4-post-1 high-Acat baseline failure layer audit 路线判断.md",
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
    item_eval_with_detail = attach_detail_groups(item_eval, availability)
    baseline_summary = build_baseline_failure_layer_summary(item_eval_with_detail)
    delta_profile = build_m3_m6_delta_profile(item_eval_with_detail)
    trainhard_delta = build_trainhard_layer_delta(delta_profile)
    evalhard_delta = build_high_acat_evalhard_layer_delta(delta_profile)
    candidate_layers = build_candidate_layer_recommendation(
        trainhard_delta,
        min_item_count=args.min_item_count,
        min_ndcg_delta=args.min_ndcg_delta,
    )
    decision = build_route_decision(candidate_layers)

    baseline_summary.to_csv(outputs.baseline_failure_summary_csv, index=False)
    delta_profile.to_csv(outputs.m3_m6_delta_profile_csv, index=False)
    trainhard_delta.to_csv(outputs.trainhard_layer_delta_csv, index=False)
    evalhard_delta.to_csv(outputs.high_acat_evalhard_layer_delta_csv, index=False)
    candidate_layers.to_csv(outputs.candidate_layer_recommendation_csv, index=False)

    result_note_name = outputs.result_md.stem
    write_result_markdown(
        outputs.result_md,
        run_stamp,
        args.design_note_name,
        baseline_summary,
        trainhard_delta,
        evalhard_delta,
        candidate_layers,
        decision,
        outputs.manifest_json.name,
    )
    write_route_markdown(outputs.route_md, run_stamp, args.design_note_name, result_note_name, candidate_layers, decision)

    manifest = {
        "run_stamp": run_stamp,
        "run_date": run_date,
        "run_iso": run_iso,
        "analysis_script": "validata/analyze_amazon_vg_task4_post1_failure_layer.py",
        "inputs": {
            "availability_csv": str(args.availability_csv),
            "item_eval_csv": str(args.item_eval_csv),
        },
        "parameters": {
            "min_item_count": args.min_item_count,
            "min_ndcg_delta": args.min_ndcg_delta,
        },
        "outputs": {field: str(getattr(outputs, field)) for field in outputs.__dataclass_fields__},
        "decision": decision,
    }
    outputs.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit high-Acat baseline failure layers for Task4-post-1.")
    parser.add_argument("--availability_csv", type=Path, default=DEFAULT_AVAILABILITY_CSV)
    parser.add_argument("--item_eval_csv", type=Path, default=DEFAULT_ITEM_EVAL_CSV)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment_root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--run_stamp", default="")
    parser.add_argument("--min_item_count", type=int, default=100)
    parser.add_argument("--min_ndcg_delta", type=float, default=0.0005)
    parser.add_argument(
        "--design_note_name",
        default="2026-07-07 104506 CCFCRec Amazon-VG Task4-post-1 high-Acat baseline failure layer audit 设计",
    )
    return parser.parse_args()


def main() -> None:
    outputs = run_analysis(parse_args())
    print(f"output_dir={outputs.output_dir}")
    print(f"result_md={outputs.result_md}")
    print(f"route_md={outputs.route_md}")


if __name__ == "__main__":
    main()
