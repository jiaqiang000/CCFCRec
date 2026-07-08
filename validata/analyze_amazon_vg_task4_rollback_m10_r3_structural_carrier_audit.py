#!/usr/bin/env python3
"""
CCFCRec Amazon-VG M10-R3 structural carrier audit.

Diagnostic-only offline audit. It checks whether the historical
category_conf_input structure-side carrier has Acat_v3-clean leverage, RSP
control dominance, or only old category-count / hard-opportunity leverage.
It does not train models.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/luojiaqiang/Documents/Obsidian Vault/科研/CCFCRec对比学习思路")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "temp_202607_实验文件记录" / "temp_20260709"
DEFAULT_BASELINE_ITEM_PROFILE = (
    PROJECT_ROOT
    / "temp_202606_实验文件记录"
    / "temp_20260626"
    / "2026-06-26 110821 优化后测试是否训练异常"
    / "baseline-seed43-workers8-full-diagnostic"
    / "checkpoint-selection"
    / "baseline_workers8_best74_item_profile.csv"
)
DEFAULT_CATEGORY_CONF_ITEM_PROFILE = (
    PROJECT_ROOT
    / "temp_202606_实验文件记录"
    / "temp_20260626"
    / "2026-06-26 152910 category-conf-input-seed43-workers8-commit1689414-full-diagnostic"
    / "checkpoint-selection"
    / "category_conf_input_workers8_best36_item_profile.csv"
)
DEFAULT_VALIDATION_SUMMARY = (
    PROJECT_ROOT
    / "temp_202606_实验文件记录"
    / "temp_20260626"
    / "2026-06-26 152910 category-conf-input-seed43-workers8-commit1689414-full-diagnostic"
    / "compare-to-baseline-workers8"
    / "validation_summary.csv"
)
DEFAULT_TASK4_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260706"
    / "2026-07-06 004222 task4-pre3-train-safe-hard-proxy"
    / "task4_train_safe_hard_proxy_profile.csv"
)
DEFAULT_R1_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260709"
    / "2026-07-09 013333 task4-rollback-m10-r1-recoverability-upper-bound-audit"
    / "m10_r1_upper_bound_profile.csv"
)
DESIGN_NOTE_NAME = "2026-07-09 015234 CCFCRec Amazon-VG M10-R3 structural carrier audit 诊断设计"
TOTAL_DESIGN_NOTE_NAME = "2026-07-09 010147 CCFCRec Amazon-VG Task4-rollback M10-R recoverability and carrier audit 总设计"
R2_ROUTE_NOTE_NAME = "2026-07-09 014850 CCFCRec Amazon-VG M10-R2 train-safe hard proxy rebuild 路线判断"
ANALYSIS_SCRIPT = "validata/analyze_amazon_vg_task4_rollback_m10_r3_structural_carrier_audit.py"
R1_MASK_COLUMNS = [
    "recoverability_ensemble_high_hard",
    "rank_recoverability_high_hard",
    "near_cutoff_hard",
    "failure_consensus_high",
    "acat_v3_high_hard",
    "rsp_high_hard",
]
GROUP_SCOPES = [
    "category_group",
    "s_cat_v3_group",
    "RSP_group",
    "high_acat_train_safe_hard_flag",
    "eval_baseline_hard_flag",
    "high_acat_eval_hard_flag",
    *R1_MASK_COLUMNS,
]


@dataclass(frozen=True)
class Outputs:
    output_dir: Path
    item_delta_profile_csv: Path
    group_summary_csv: Path
    control_correlation_csv: Path
    shuffle_summary_json: Path
    route_decision_json: Path
    manifest_json: Path
    result_md: Path


def now_stamp() -> tuple[str, str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H%M%S"), now.strftime("%Y-%m-%d"), now.isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _bool_series(series: pd.Series | None, index: pd.Index, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=bool)
    if series.dtype == bool:
        return series.fillna(default).astype(bool).reindex(index, fill_value=default)
    text = series.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", "", "nan", "<na>", "none"}
    result = pd.Series(default, index=series.index, dtype=bool)
    result.loc[text.isin(true_values)] = True
    result.loc[text.isin(false_values)] = False
    return result.reindex(index, fill_value=default)


def _safe_corr(left: pd.Series, right: pd.Series, method: str = "spearman") -> float:
    valid = pd.concat([_numeric(left), _numeric(right)], axis=1).dropna()
    if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))


def _load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"raw_asin": str, "asin": str}, low_memory=False)


def _normalize_item_profile(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    if "raw_asin" not in out.columns and "asin" in out.columns:
        out = out.rename(columns={"asin": "raw_asin"})
    required = {"raw_asin", "ndcg@20", "hr@20"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"{prefix} item profile missing columns: {sorted(missing)}")
    keep = ["raw_asin", "ndcg@20", "hr@20"]
    for col in ["category_group", "category_count", "gt_user_count"]:
        if col in out.columns:
            keep.append(col)
    out = out[keep].drop_duplicates("raw_asin").copy()
    out["raw_asin"] = out["raw_asin"].astype(str)
    rename = {"ndcg@20": f"{prefix}_ndcg@20", "hr@20": f"{prefix}_hr@20"}
    return out.rename(columns=rename)


def build_structural_delta_profile(
    baseline_item_profile: pd.DataFrame,
    category_conf_item_profile: pd.DataFrame,
    task4_profile: pd.DataFrame,
    r1_profile: pd.DataFrame | None = None,
) -> pd.DataFrame:
    baseline = _normalize_item_profile(baseline_item_profile, "baseline")
    category_conf = _normalize_item_profile(category_conf_item_profile, "category_conf")
    profile = baseline.merge(category_conf, on="raw_asin", how="inner", validate="one_to_one", suffixes=("", "_category_conf"))

    profile["delta_ndcg@20"] = _numeric(profile["category_conf_ndcg@20"]) - _numeric(profile["baseline_ndcg@20"])
    profile["delta_hr@20"] = _numeric(profile["category_conf_hr@20"]) - _numeric(profile["baseline_hr@20"])
    profile["ndcg@20_improved_flag"] = profile["delta_ndcg@20"].gt(0)
    profile["ndcg@20_declined_flag"] = profile["delta_ndcg@20"].lt(0)
    profile["hr@20_improved_flag"] = profile["delta_hr@20"].gt(0)
    profile["hr@20_declined_flag"] = profile["delta_hr@20"].lt(0)

    task4 = task4_profile.copy()
    if "raw_asin" not in task4.columns and "asin" in task4.columns:
        task4 = task4.rename(columns={"asin": "raw_asin"})
    task4["raw_asin"] = task4["raw_asin"].astype(str)
    task4_keep = [
        "raw_asin",
        "s_cat_v3",
        "s_cat_v3_group",
        "RSP_score",
        "RSP_group",
        "high_acat_train_safe_hard_flag",
        "eval_baseline_hard_flag",
        "high_acat_eval_hard_flag",
        "baseline_margin_proxy",
        "baseline_best_target_rank",
        "target_activity_bucket",
    ]
    if "category_count" not in profile.columns:
        task4_keep.append("category_count")
    profile = profile.merge(task4[[col for col in task4_keep if col in task4.columns]].drop_duplicates("raw_asin"), on="raw_asin", how="left")

    if r1_profile is not None and not r1_profile.empty:
        r1 = r1_profile.copy()
        if "raw_asin" not in r1.columns and "asin" in r1.columns:
            r1 = r1.rename(columns={"asin": "raw_asin"})
        r1["raw_asin"] = r1["raw_asin"].astype(str)
        keep = ["raw_asin", *[col for col in R1_MASK_COLUMNS if col in r1.columns]]
        profile = profile.merge(r1[keep].drop_duplicates("raw_asin"), on="raw_asin", how="left")

    for col in [*R1_MASK_COLUMNS, "high_acat_train_safe_hard_flag", "eval_baseline_hard_flag", "high_acat_eval_hard_flag"]:
        if col in profile.columns:
            profile[col] = _bool_series(profile[col], profile.index).map(bool).astype(object)
    return profile


def _summarize_subset(frame: pd.DataFrame, scope: str, group: Any) -> dict[str, Any]:
    return {
        "scope": scope,
        "group": str(group),
        "item_count": int(len(frame)),
        "baseline_ndcg@20_mean": float(_numeric(frame["baseline_ndcg@20"]).mean()),
        "category_conf_ndcg@20_mean": float(_numeric(frame["category_conf_ndcg@20"]).mean()),
        "delta_ndcg@20_mean": float(_numeric(frame["delta_ndcg@20"]).mean()),
        "ndcg@20_improved_rate": float(frame["ndcg@20_improved_flag"].mean()),
        "ndcg@20_declined_rate": float(frame["ndcg@20_declined_flag"].mean()),
        "baseline_hr@20_mean": float(_numeric(frame["baseline_hr@20"]).mean()),
        "category_conf_hr@20_mean": float(_numeric(frame["category_conf_hr@20"]).mean()),
        "delta_hr@20_mean": float(_numeric(frame["delta_hr@20"]).mean()),
        "hr@20_improved_rate": float(frame["hr@20_improved_flag"].mean()),
        "hr@20_declined_rate": float(frame["hr@20_declined_flag"].mean()),
    }


def build_group_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows = [_summarize_subset(profile, "overall", "all")]
    for scope in GROUP_SCOPES:
        if scope not in profile.columns:
            continue
        for group, sub in profile.groupby(scope, dropna=False):
            rows.append(_summarize_subset(sub, scope, group))
    return pd.DataFrame(rows)


def build_control_correlation(profile: pd.DataFrame) -> pd.DataFrame:
    rights = [
        ("s_cat_v3", "availability"),
        ("RSP_score", "control"),
        ("category_count", "legacy_category_count"),
        ("baseline_ndcg@20", "eval_audit"),
        ("baseline_hr@20", "eval_audit"),
        ("baseline_margin_proxy", "eval_audit"),
        ("baseline_best_target_rank", "eval_audit"),
    ]
    rows = []
    for right, family in rights:
        if right not in profile.columns:
            continue
        rows.append(
            {
                "left": "delta_ndcg@20",
                "right": right,
                "right_family": family,
                "n": int(pd.concat([_numeric(profile["delta_ndcg@20"]), _numeric(profile[right])], axis=1).dropna().shape[0]),
                "spearman": _safe_corr(profile["delta_ndcg@20"], profile[right]),
                "pearson": _safe_corr(profile["delta_ndcg@20"], profile[right], method="pearson"),
            }
        )
    return pd.DataFrame(rows)


def _group_delta(profile: pd.DataFrame, scope: str, group: str) -> float:
    if scope not in profile.columns:
        return float("nan")
    sub = profile[profile[scope].astype(str).eq(group)]
    if sub.empty:
        return float("nan")
    return float(_numeric(sub["delta_ndcg@20"]).mean())


def build_shuffle_summary(profile: pd.DataFrame, n_shuffles: int = 256, seed: int = 43) -> dict[str, Any]:
    if "s_cat_v3_group" not in profile.columns:
        return {
            "shuffle_count": 0,
            "reason": "missing s_cat_v3_group",
            "real_acat_high_minus_low_delta_ndcg@20": None,
            "shuffle_p95_acat_high_minus_low_delta_ndcg@20": None,
            "real_beats_shuffle_p95": False,
        }
    labels = profile["s_cat_v3_group"].astype(str).to_numpy()
    values = _numeric(profile["delta_ndcg@20"]).to_numpy(dtype=float)
    real_high = values[labels == "s_cat_v3_strong"]
    real_low = values[labels == "s_cat_v3_weak"]
    real_spread = float(np.nanmean(real_high) - np.nanmean(real_low)) if len(real_high) and len(real_low) else float("nan")

    rng = np.random.default_rng(seed)
    spreads: list[float] = []
    for _ in range(n_shuffles):
        shuffled = rng.permutation(labels)
        high = values[shuffled == "s_cat_v3_strong"]
        low = values[shuffled == "s_cat_v3_weak"]
        if len(high) and len(low):
            spreads.append(float(np.nanmean(high) - np.nanmean(low)))
    arr = np.array(spreads, dtype=float)
    p95 = float(np.nanpercentile(arr, 95)) if len(arr) else float("nan")
    p05 = float(np.nanpercentile(arr, 5)) if len(arr) else float("nan")
    return {
        "shuffle_count": int(len(spreads)),
        "seed": seed,
        "real_acat_high_minus_low_delta_ndcg@20": _jsonable(real_spread),
        "shuffle_mean_acat_high_minus_low_delta_ndcg@20": _jsonable(float(np.nanmean(arr)) if len(arr) else float("nan")),
        "shuffle_p05_acat_high_minus_low_delta_ndcg@20": _jsonable(p05),
        "shuffle_p95_acat_high_minus_low_delta_ndcg@20": _jsonable(p95),
        "real_beats_shuffle_p95": bool(np.isfinite(real_spread) and np.isfinite(p95) and real_spread > p95),
    }


def _summary_value(summary: pd.DataFrame, scope: str, group: str, column: str = "delta_ndcg@20_mean") -> float:
    row = summary[summary["scope"].eq(scope) & summary["group"].astype(str).eq(group)]
    if row.empty or column not in row.columns:
        return float("nan")
    return float(row.iloc[0][column])


def _validation_delta(validation_summary: pd.DataFrame | None) -> float:
    if validation_summary is None or validation_summary.empty or "best_ndcg@20" not in validation_summary.columns:
        return float("nan")
    run = validation_summary.get("run", pd.Series("", index=validation_summary.index)).astype(str)
    baseline = validation_summary[run.str.contains("baseline", case=False, na=False)]
    category = validation_summary[run.str.contains("category_conf", case=False, na=False)]
    if baseline.empty or category.empty:
        return float("nan")
    return float(_numeric(category["best_ndcg@20"]).max() - _numeric(baseline["best_ndcg@20"]).max())


def decide_route(
    validation_summary: pd.DataFrame | None,
    group_summary: pd.DataFrame,
    control_correlation: pd.DataFrame,
    shuffle_summary: dict[str, Any],
) -> dict[str, Any]:
    validation_delta = _validation_delta(validation_summary)
    overall_delta = _summary_value(group_summary, "overall", "all")
    old_weak_delta = _summary_value(group_summary, "category_group", "cat_weak_1_3")
    acat_strong_delta = _summary_value(group_summary, "s_cat_v3_group", "s_cat_v3_strong")
    acat_weak_delta = _summary_value(group_summary, "s_cat_v3_group", "s_cat_v3_weak")
    rsp_high_delta = _summary_value(group_summary, "RSP_group", "RSP_high")
    rsp_low_delta = _summary_value(group_summary, "RSP_group", "RSP_low")
    near_cutoff_delta = _summary_value(group_summary, "near_cutoff_hard", "True")
    failure_consensus_delta = _summary_value(group_summary, "failure_consensus_high", "True")
    acat_spread = shuffle_summary.get("real_acat_high_minus_low_delta_ndcg@20")
    shuffle_p95 = shuffle_summary.get("shuffle_p95_acat_high_minus_low_delta_ndcg@20")
    real_beats_shuffle = bool(shuffle_summary.get("real_beats_shuffle_p95", False))

    corr_rsp = control_correlation[control_correlation["right"].eq("RSP_score")]
    corr_acat = control_correlation[control_correlation["right"].eq("s_cat_v3")]
    rsp_spearman = float(corr_rsp.iloc[0]["spearman"]) if not corr_rsp.empty else float("nan")
    acat_spearman = float(corr_acat.iloc[0]["spearman"]) if not corr_acat.empty else float("nan")

    has_structural_leverage = any(
        np.isfinite(value) and value > 0
        for value in [validation_delta, old_weak_delta, near_cutoff_delta, failure_consensus_delta]
    )
    acat_clean = (
        np.isfinite(acat_strong_delta)
        and acat_strong_delta > 0
        and real_beats_shuffle
        and (not np.isfinite(rsp_spearman) or abs(rsp_spearman) < 0.20)
    )
    rsp_dominated = (
        np.isfinite(rsp_high_delta)
        and np.isfinite(acat_strong_delta)
        and rsp_high_delta > max(acat_strong_delta, 0.0)
        and np.isfinite(rsp_spearman)
        and (not np.isfinite(acat_spearman) or abs(rsp_spearman) > abs(acat_spearman) + 0.05)
    )

    if acat_clean:
        route = "structural_carrier_acat_clean_ready"
        next_action = "design_conservative_residual_gate_with_rsp_and_shuffle_controls"
        reason = "Acat_v3 strong group has positive structural gain and beats Acat shuffle control"
    elif rsp_dominated:
        route = "structural_signal_rsp_dominated"
        next_action = "do_not_use_as_availability_method_keep_as_control"
        reason = "structural gain is more aligned with RSP control than clean Acat_v3"
    elif has_structural_leverage:
        route = "structural_carrier_leverage_not_acat_clean"
        next_action = "do_not_rollback_old_category_conf_use_as_carrier_hint_for_r4_or_residual_design"
        reason = "structural carrier has leverage, but not clean Acat_v3-aligned leverage"
    else:
        route = "structural_carrier_not_supported"
        next_action = "skip_structural_carrier_prioritize_r4_competitor_calibration"
        reason = "no stable positive structural leverage in validation or key groups"

    return {
        "route": route,
        "next_action": next_action,
        "enter_training_now": False,
        "validation_delta_ndcg@20": _jsonable(validation_delta),
        "overall_delta_ndcg@20": _jsonable(overall_delta),
        "old_weak_category_delta_ndcg@20": _jsonable(old_weak_delta),
        "acat_strong_delta_ndcg@20": _jsonable(acat_strong_delta),
        "acat_weak_delta_ndcg@20": _jsonable(acat_weak_delta),
        "rsp_high_delta_ndcg@20": _jsonable(rsp_high_delta),
        "rsp_low_delta_ndcg@20": _jsonable(rsp_low_delta),
        "near_cutoff_hard_delta_ndcg@20": _jsonable(near_cutoff_delta),
        "failure_consensus_high_delta_ndcg@20": _jsonable(failure_consensus_delta),
        "acat_high_minus_low_delta_ndcg@20": _jsonable(acat_spread),
        "acat_shuffle_p95_delta_ndcg@20": _jsonable(shuffle_p95),
        "delta_spearman_vs_acat_v3": _jsonable(acat_spearman),
        "delta_spearman_vs_rsp": _jsonable(rsp_spearman),
        "reason": reason,
    }


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if df.empty:
        return "_empty_"
    small = df[[col for col in columns if col in df.columns]].copy()
    if max_rows is not None:
        small = small.head(max_rows)
    for col in small.columns:
        if pd.api.types.is_float_dtype(small[col]):
            small[col] = small[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    header = "| " + " | ".join(small.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(small.columns)) + " |"
    body = ["| " + " | ".join("" if pd.isna(value) else str(value) for value in row.tolist()) + " |" for _, row in small.iterrows()]
    return "\n".join([header, sep, *body])


def write_result_markdown(
    path: Path,
    run_stamp: str,
    group_summary: pd.DataFrame,
    control_correlation: pd.DataFrame,
    shuffle_summary: dict[str, Any],
    decision: dict[str, Any],
    manifest_name: str,
) -> None:
    selected_groups = group_summary[
        group_summary["scope"].isin(["overall", "category_group", "s_cat_v3_group", "RSP_group", "near_cutoff_hard", "failure_consensus_high", "high_acat_train_safe_hard_flag"])
    ]
    content = f"""---
title: {run_stamp} CCFCRec Amazon-VG M10-R3 structural carrier audit 结果
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - M10-R3
  - structural_carrier
---

# {run_stamp} CCFCRec Amazon-VG M10-R3 structural carrier audit 结果

## Material Passport

- artifact_type: experiment_diagnostic_result
- project: CCFCRec Amazon-VG category availability
- stage: M10-R3 structural carrier audit
- status: analyzed
- execution_policy: diagnostic-only（仅诊断），no training（不训练）

> [!info] 来源说明
> 上游总设计：[[{TOTAL_DESIGN_NOTE_NAME}]]
> 上游 R3 设计：[[{DESIGN_NOTE_NAME}]]
> 上游 R2 路线判断：[[{R2_ROUTE_NOTE_NAME}]]
> 分析脚本：`{ANALYSIS_SCRIPT}`
> manifest（运行清单）：`{manifest_name}`

## 结论

```text
route = {decision["route"]}
next_action = {decision["next_action"]}
enter_training_now = {decision["enter_training_now"]}
```

解释：R3 只审计 historical category_conf_input（历史类别置信输入）是否有 structural carrier（结构侧方法载体）杠杆，不训练模型，也不把旧方法直接写成 Acat_v3（第三版类别可用性）方法。

## Route Decision

```json
{json.dumps(_jsonable(decision), ensure_ascii=False, indent=2)}
```

## Group Summary

> [!info] 字段说明
> `scope`：分组字段。
> `group`：分组取值。
> `delta_ndcg@20_mean`：category_conf_input（类别置信输入）减 baseline（基线）的 NDCG@20 均值差。
> `ndcg@20_improved_rate`：NDCG@20 提升的 item（物品）比例。
> `ndcg@20_declined_rate`：NDCG@20 下降的 item（物品）比例。

{md_table(selected_groups, ["scope", "group", "item_count", "baseline_ndcg@20_mean", "category_conf_ndcg@20_mean", "delta_ndcg@20_mean", "ndcg@20_improved_rate", "ndcg@20_declined_rate", "delta_hr@20_mean"], max_rows=40)}

## Control Correlation

{md_table(control_correlation, ["left", "right", "right_family", "n", "spearman", "pearson"], max_rows=20)}

## Acat Shuffle Control

```json
{json.dumps(_jsonable(shuffle_summary), ensure_ascii=False, indent=2)}
```
"""
    path.write_text(content, encoding="utf-8")


def build_outputs(output_root: Path, run_stamp: str) -> Outputs:
    output_dir = output_root / f"{run_stamp} task4-rollback-m10-r3-structural-carrier-audit"
    return Outputs(
        output_dir=output_dir,
        item_delta_profile_csv=output_dir / "m10_r3_structural_item_delta_profile.csv",
        group_summary_csv=output_dir / "m10_r3_structural_group_summary.csv",
        control_correlation_csv=output_dir / "m10_r3_structural_control_correlation.csv",
        shuffle_summary_json=output_dir / "m10_r3_structural_shuffle_summary.json",
        route_decision_json=output_dir / "m10_r3_route_decision.json",
        manifest_json=output_dir / "run_manifest.json",
        result_md=output_dir / f"{run_stamp} CCFCRec Amazon-VG M10-R3 structural carrier audit 结果.md",
    )


def run(args: argparse.Namespace) -> Outputs:
    run_stamp, run_date, run_iso = (args.run_stamp, args.run_stamp[:10], "") if args.run_stamp else now_stamp()
    if args.run_stamp:
        run_iso = datetime.strptime(args.run_stamp, "%Y-%m-%d %H%M%S").isoformat(timespec="seconds")
    outputs = build_outputs(Path(args.output_root).expanduser().resolve(), run_stamp)
    outputs.output_dir.mkdir(parents=True, exist_ok=True)

    baseline = _load_csv(args.baseline_item_profile_path)
    category_conf = _load_csv(args.category_conf_item_profile_path)
    task4 = _load_csv(args.task4_profile_path)
    r1 = _load_csv(args.r1_profile_path)
    validation = _load_csv(args.validation_summary_path)

    profile = build_structural_delta_profile(baseline, category_conf, task4, r1)
    group_summary = build_group_summary(profile)
    control_correlation = build_control_correlation(profile)
    shuffle_summary = build_shuffle_summary(profile, n_shuffles=args.shuffle_count, seed=args.shuffle_seed)
    decision = decide_route(validation, group_summary, control_correlation, shuffle_summary)

    profile.to_csv(outputs.item_delta_profile_csv, index=False)
    group_summary.to_csv(outputs.group_summary_csv, index=False)
    control_correlation.to_csv(outputs.control_correlation_csv, index=False)
    outputs.shuffle_summary_json.write_text(json.dumps(_jsonable(shuffle_summary), ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.route_decision_json.write_text(json.dumps(_jsonable(decision), ensure_ascii=False, indent=2), encoding="utf-8")
    write_result_markdown(outputs.result_md, run_stamp, group_summary, control_correlation, shuffle_summary, decision, outputs.manifest_json.name)

    manifest = {
        "run_stamp": run_stamp,
        "run_date": run_date,
        "run_iso": run_iso,
        "experiment_stage": "M10-R3",
        "analysis_script": ANALYSIS_SCRIPT,
        "design_note": DESIGN_NOTE_NAME,
        "total_design_note": TOTAL_DESIGN_NOTE_NAME,
        "r2_route_note": R2_ROUTE_NOTE_NAME,
        "diagnostic_only_no_training": True,
        "inputs": {
            "baseline_item_profile_path": str(args.baseline_item_profile_path),
            "category_conf_item_profile_path": str(args.category_conf_item_profile_path),
            "validation_summary_path": str(args.validation_summary_path),
            "task4_profile_path": str(args.task4_profile_path),
            "r1_profile_path": str(args.r1_profile_path),
        },
        "shuffle": {"shuffle_count": args.shuffle_count, "shuffle_seed": args.shuffle_seed},
        "outputs": {field: str(getattr(outputs, field)) for field in outputs.__dataclass_fields__},
        "decision": decision,
    }
    outputs.manifest_json.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M10-R3 structural carrier audit.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--baseline-item-profile-path", default=str(DEFAULT_BASELINE_ITEM_PROFILE))
    parser.add_argument("--category-conf-item-profile-path", default=str(DEFAULT_CATEGORY_CONF_ITEM_PROFILE))
    parser.add_argument("--validation-summary-path", default=str(DEFAULT_VALIDATION_SUMMARY))
    parser.add_argument("--task4-profile-path", default=str(DEFAULT_TASK4_PROFILE))
    parser.add_argument("--r1-profile-path", default=str(DEFAULT_R1_PROFILE))
    parser.add_argument("--shuffle-count", type=int, default=256)
    parser.add_argument("--shuffle-seed", type=int, default=43)
    parser.add_argument("--run-stamp", default="")
    return parser


def main() -> None:
    outputs = run(build_arg_parser().parse_args())
    print(f"output_dir={outputs.output_dir}")
    print(f"result_md={outputs.result_md}")
    print(f"route_decision={outputs.route_decision_json}")


if __name__ == "__main__":
    main()
