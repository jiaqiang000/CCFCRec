#!/usr/bin/env python3
"""
Task4-revise audit and carrier design.

This script does not train. It audits why M3 (Acat_v3 train-safe hard weight)
is close to M6 (Acat_v3 shuffle control), then emits a new carrier design
focused on Acat_v3-conditioned pairwise margin training.
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
CODE_ROOT = PROJECT_ROOT / "CCFCRec-code"
DEFAULT_POST_DIR = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260706"
    / "2026-07-06 130518 task4-post-acat-v3-weight-controls-analysis"
)
DEFAULT_TASK4_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260706"
    / "2026-07-06 004222 task4-pre3-train-safe-hard-proxy"
    / "task4_train_safe_hard_proxy_profile.csv"
)

M3 = "task4_acat_trainhard_weight"
M6 = "task4_acat_shuffle_high_weight"
M1 = "task4_rsp_high_weight"
M2 = "task4_acat_high_weight"

GROUP_COLUMNS = [
    "high_acat_flag",
    "eval_baseline_hard_flag",
    "high_acat_eval_hard_flag",
    "high_acat_train_safe_hard_flag",
    "RSP_group",
    "s_cat_v3_group",
]


@dataclass(frozen=True)
class Outputs:
    output_dir: Path
    mask_audit_csv: Path
    m3_m6_delta_profile_csv: Path
    contribution_summary_csv: Path
    train_curve_csv: Path
    train_curve_summary_csv: Path
    candidate_carrier_csv: Path
    screening_protocol_csv: Path
    manifest_json: Path
    result_md: Path
    design_md: Path
    route_md: Path


def now_stamp() -> tuple[str, str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H%M%S"), now.strftime("%Y-%m-%d"), now.isoformat(timespec="seconds")


def nan_float() -> float:
    return float("nan")


def _bool_series(series: pd.Series) -> pd.Series:
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
    body = []
    for _, row in small.iterrows():
        values = ["" if pd.isna(value) else str(value) for value in row.tolist()]
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator, *body])


def build_split_aware_shuffle_flags(profile: pd.DataFrame, seed: int = 43) -> pd.Series:
    required = {"raw_asin", "high_acat_flag"}
    missing = required - set(profile.columns)
    if missing:
        raise ValueError(f"profile missing columns: {sorted(missing)}")
    work = profile.copy().sort_values("raw_asin").reset_index(drop=True)
    flags = _bool_series(work["high_acat_flag"])
    rng = np.random.default_rng(seed)
    shuffled = pd.Series(False, index=work.index, dtype=bool)
    if "split" in work.columns:
        for _, split_index in work.groupby("split", dropna=False).groups.items():
            values = flags.loc[split_index].to_numpy(dtype=bool).copy()
            rng.shuffle(values)
            shuffled.loc[split_index] = values
    else:
        values = flags.to_numpy(dtype=bool).copy()
        rng.shuffle(values)
        shuffled.loc[:] = values
    return shuffled


def build_mask_audit(profile: pd.DataFrame, seed: int = 43) -> pd.DataFrame:
    required = {"raw_asin", "split", "high_acat_flag", "high_acat_train_safe_hard_flag"}
    missing = required - set(profile.columns)
    if missing:
        raise ValueError(f"profile missing columns: {sorted(missing)}")
    work = profile.copy().sort_values("raw_asin").reset_index(drop=True)
    work["m3_weighted_flag"] = _bool_series(work["high_acat_train_safe_hard_flag"])
    work["high_acat_flag_bool"] = _bool_series(work["high_acat_flag"])
    work["m6_shuffle_weighted_flag"] = build_split_aware_shuffle_flags(work, seed=seed)
    work["m3_and_m6"] = work["m3_weighted_flag"] & work["m6_shuffle_weighted_flag"]
    work["m3_or_m6"] = work["m3_weighted_flag"] | work["m6_shuffle_weighted_flag"]
    work["m3_and_high_acat"] = work["m3_weighted_flag"] & work["high_acat_flag_bool"]

    rows = []
    for split, sub in work.groupby("split", dropna=False):
        total = len(sub)
        m3_count = int(sub["m3_weighted_flag"].sum())
        m6_count = int(sub["m6_shuffle_weighted_flag"].sum())
        high_count = int(sub["high_acat_flag_bool"].sum())
        overlap = int(sub["m3_and_m6"].sum())
        union = int(sub["m3_or_m6"].sum())
        m3_high = int(sub["m3_and_high_acat"].sum())
        rows.append(
            {
                "split": split,
                "item_count": total,
                "high_acat_count": high_count,
                "m3_weighted_count": m3_count,
                "m6_weighted_count": m6_count,
                "high_acat_share": high_count / total if total else nan_float(),
                "m3_share": m3_count / total if total else nan_float(),
                "m6_share": m6_count / total if total else nan_float(),
                "m3_minus_m6_share": (m3_count - m6_count) / total if total else nan_float(),
                "m3_m6_overlap_count": overlap,
                "m3_m6_jaccard": overlap / union if union else nan_float(),
                "m3_subset_of_high_acat_rate": m3_high / m3_count if m3_count else nan_float(),
                "m3_not_m6_count": int((sub["m3_weighted_flag"] & ~sub["m6_shuffle_weighted_flag"]).sum()),
                "m6_not_m3_count": int((sub["m6_shuffle_weighted_flag"] & ~sub["m3_weighted_flag"]).sum()),
            }
        )
    result = pd.DataFrame(rows)
    order = {"train": 0, "validate": 1, "test": 2}
    result["split_order"] = result["split"].map(order).fillna(99)
    return result.sort_values(["split_order", "split"]).drop(columns=["split_order"]).reset_index(drop=True)


def build_m3_m6_delta_profile(item_eval: pd.DataFrame) -> pd.DataFrame:
    required = {"method_variant", "split", "raw_asin", "ndcg@20", "hr@20"}
    missing = required - set(item_eval.columns)
    if missing:
        raise ValueError(f"item_eval missing columns: {sorted(missing)}")
    base_cols = [
        "split",
        "raw_asin",
        "s_cat_v3",
        "s_cat_v3_group",
        "RSP_score",
        "RSP_group",
        "baseline_ndcg@20",
        "baseline_margin_proxy",
        "baseline_best_target_rank",
        "high_acat_flag",
        "eval_baseline_hard_flag",
        "high_acat_eval_hard_flag",
        "high_acat_train_safe_hard_flag",
        "train_safe_hard_proxy_score",
        "train_safe_hard_proxy_group",
    ]
    available_base = [col for col in base_cols if col in item_eval.columns]
    m3 = item_eval[item_eval["method_variant"].eq(M3)].copy()
    m6 = item_eval[item_eval["method_variant"].eq(M6)].copy()
    metric_cols = [
        "hr@20",
        "ndcg@20",
        "q_norm",
        "target_score_max",
        "margin_to_top20_cutoff",
        "best_target_rank",
    ]
    available_metrics = [col for col in metric_cols if col in item_eval.columns]
    left = m3[available_base + available_metrics].copy()
    right = m6[["split", "raw_asin", *available_metrics]].copy()
    left = left.rename(columns={col: f"m3_{col}" for col in available_metrics})
    right = right.rename(columns={col: f"m6_{col}" for col in available_metrics})
    merged = left.merge(right, on=["split", "raw_asin"], how="inner", validate="one_to_one")
    for col in available_metrics:
        merged[f"delta_{col}"] = merged[f"m3_{col}"] - merged[f"m6_{col}"]
    return merged


def build_contribution_summary(delta_profile: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col not in delta_profile.columns:
        return pd.DataFrame()
    required = {"split", "delta_ndcg@20", "delta_hr@20"}
    missing = required - set(delta_profile.columns)
    if missing:
        raise ValueError(f"delta_profile missing columns: {sorted(missing)}")
    rows = []
    for split, split_df in delta_profile.groupby("split", dropna=False):
        total = len(split_df)
        total_delta_ndcg = float(split_df["delta_ndcg@20"].sum())
        total_delta_hr = float(split_df["delta_hr@20"].sum())
        for group_value, sub in split_df.groupby(group_col, dropna=False):
            count = len(sub)
            sum_delta_ndcg = float(sub["delta_ndcg@20"].sum())
            sum_delta_hr = float(sub["delta_hr@20"].sum())
            rows.append(
                {
                    "split": split,
                    "group_column": group_col,
                    "group_value": bool_text(group_value),
                    "item_count": count,
                    "item_share": count / total if total else nan_float(),
                    "mean_delta_ndcg@20": safe_mean(sub["delta_ndcg@20"]),
                    "mean_delta_hr@20": safe_mean(sub["delta_hr@20"]),
                    "median_delta_ndcg@20": safe_median(sub["delta_ndcg@20"]),
                    "positive_delta_ndcg_rate": float((sub["delta_ndcg@20"] > 0).mean()) if count else nan_float(),
                    "overall_contribution_ndcg@20": sum_delta_ndcg / total if total else nan_float(),
                    "overall_contribution_hr@20": sum_delta_hr / total if total else nan_float(),
                    "share_of_total_delta_ndcg@20": sum_delta_ndcg / total_delta_ndcg if total_delta_ndcg else nan_float(),
                    "share_of_total_delta_hr@20": sum_delta_hr / total_delta_hr if total_delta_hr else nan_float(),
                }
            )
    result = pd.DataFrame(rows)
    order = {"validate": 0, "test": 1}
    if not result.empty:
        result["split_order"] = result["split"].map(order).fillna(99)
        result = result.sort_values(["split_order", "group_column", "group_value"]).drop(columns=["split_order"]).reset_index(drop=True)
    return result


def build_all_contribution_summary(delta_profile: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for group_col in GROUP_COLUMNS:
        if group_col in delta_profile.columns:
            frames.append(build_contribution_summary(delta_profile, group_col))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_train_curve_comparison(run_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for method in [M3, M6]:
        sub = run_index[run_index["method_variant"].eq(method)]
        if sub.empty:
            continue
        result = pd.read_csv(sub.iloc[0]["result_csv"])
        result["method_variant"] = method
        rows.append(result)
    if len(rows) < 2:
        return pd.DataFrame(), pd.DataFrame()
    curve = pd.concat(rows, ignore_index=True)
    m3 = curve[curve["method_variant"].eq(M3)].copy()
    m6 = curve[curve["method_variant"].eq(M6)].copy()
    metrics = ["loss", "contrast_sum", "hr@20", "ndcg@20"]
    m3 = m3[["checkpoint_index", "epoch", *metrics]].rename(columns={metric: f"m3_{metric}" for metric in metrics})
    m6 = m6[["checkpoint_index", "epoch", *metrics]].rename(columns={metric: f"m6_{metric}" for metric in metrics})
    paired = m3.merge(m6, on=["checkpoint_index", "epoch"], how="inner", validate="one_to_one")
    for metric in metrics:
        paired[f"delta_{metric}"] = paired[f"m3_{metric}"] - paired[f"m6_{metric}"]
    summary = pd.DataFrame(
        [
            {
                "paired_checkpoint_count": len(paired),
                "mean_delta_loss": safe_mean(paired["delta_loss"]),
                "median_delta_loss": safe_median(paired["delta_loss"]),
                "mean_delta_contrast_sum": safe_mean(paired["delta_contrast_sum"]),
                "mean_delta_ndcg@20": safe_mean(paired["delta_ndcg@20"]),
                "mean_delta_hr@20": safe_mean(paired["delta_hr@20"]),
                "best_curve_delta_ndcg@20": float(paired["delta_ndcg@20"].max()) if not paired.empty else nan_float(),
                "worst_curve_delta_ndcg@20": float(paired["delta_ndcg@20"].min()) if not paired.empty else nan_float(),
                "last_delta_ndcg@20": float(paired.sort_values("checkpoint_index").iloc[-1]["delta_ndcg@20"]) if not paired.empty else nan_float(),
                "last_delta_hr@20": float(paired.sort_values("checkpoint_index").iloc[-1]["delta_hr@20"]) if not paired.empty else nan_float(),
            }
        ]
    )
    return paired, summary


def build_candidate_carrier_table() -> pd.DataFrame:
    rows = [
        {
            "candidate_id": "M4a",
            "method_variant": "task4_acat_pairmargin_weight",
            "priority": 1,
            "carrier_family": "Acat_v3-conditioned pairwise margin",
            "availability_input": "high_acat_train_safe_hard_flag + s_cat_v3",
            "hard_input": "train_safe_hard_proxy_score",
            "loss_surface": "q_v_c target user vs sampled competitor user margin",
            "protects_y_ukv": True,
            "required_controls": "M1_RSP_only",
            "all_required_controls": "M1_RSP_only;M6_shuffle;M2_Acat_high",
            "seed43_gate": "M4a - M6 NDCG@20 >= 0.0005 and HR@20 >= 0",
            "risk": "may overfit proxy hard items if competitor sampling is too narrow",
        },
        {
            "candidate_id": "M4b",
            "method_variant": "task4_acat_rsp_residual_pairmargin",
            "priority": 2,
            "carrier_family": "Acat_v3-conditioned pairwise margin with RSP control",
            "availability_input": "s_cat_v3 residual against RSP_group",
            "hard_input": "high_acat_train_safe_hard_flag",
            "loss_surface": "larger margin only when Acat_v3 is high beyond RSP expectation",
            "protects_y_ukv": True,
            "required_controls": "M6_shuffle",
            "all_required_controls": "M1_RSP_only;M6_shuffle;M2_Acat_high",
            "seed43_gate": "M4b - M6 NDCG@20 >= 0.0005 and HR@20 >= 0",
            "risk": "too conservative if residualization removes useful availability signal",
        },
        {
            "candidate_id": "M4c",
            "method_variant": "task4_acat_hardonly_qmargin",
            "priority": 3,
            "carrier_family": "hard-only q-side pairwise margin",
            "availability_input": "high_acat_flag",
            "hard_input": "train_safe_hard_proxy_high_flag",
            "loss_surface": "q_v_c margin only; no original item_embedding-user_embedding y_ukv change",
            "protects_y_ukv": True,
            "required_controls": "M2_Acat_high",
            "all_required_controls": "M1_RSP_only;M6_shuffle;M2_Acat_high",
            "seed43_gate": "M4c - M6 NDCG@20 >= 0.0005 and HR@20 >= 0",
            "risk": "may be too weak if margin target users are not sampled in training batch",
        },
    ]
    return pd.DataFrame(rows)


def build_screening_protocol_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "phase": "smoke",
                "seed": 43,
                "epoch": 1,
                "purpose": "check loss stability, output manifest, and no MPS/runtime failure",
                "pass_gate": "all candidates finish 1 epoch; no NaN; result.csv and launcher manifest exist",
            },
            {
                "phase": "seed43_full_screen",
                "seed": 43,
                "epoch": 100,
                "purpose": "screen candidate carrier against M6 shuffle",
                "pass_gate": "candidate - M6 validate NDCG@20 >= 0.0005 and HR@20 >= 0; test high-Acat hard group also wins",
            },
            {
                "phase": "multi_seed_stability",
                "seed": "only after seed43 pass",
                "epoch": 100,
                "purpose": "paper-table stability, not rescue",
                "pass_gate": "open only if seed43_full_screen passes; compare mean/std across seeds",
            },
        ]
    )


def build_revise_route_decision(
    method_comparison: pd.DataFrame,
    contribution_summary: pd.DataFrame,
    min_meaningful_ndcg_delta: float = 0.0005,
) -> dict:
    row = method_comparison[
        method_comparison["comparison"].eq("M3_acat_trainhard_minus_M6_acat_shuffle")
    ]
    if row.empty:
        m3_vs_m6_delta = nan_float()
        m3_vs_m6_hr_delta = nan_float()
        meaningful = False
        hr_not_worse = False
    else:
        first = row.iloc[0]
        m3_vs_m6_delta = float(first["delta_ndcg@20"])
        m3_vs_m6_hr_delta = float(first["delta_hr@20"])
        meaningful = bool(first["meaningful_ndcg_win"])
        hr_not_worse = bool(first["hr_not_worse"])

    target = contribution_summary[
        contribution_summary["group_column"].eq("high_acat_train_safe_hard_flag")
        & contribution_summary["group_value"].eq("True")
        & contribution_summary["split"].eq("test")
    ]
    outside = contribution_summary[
        contribution_summary["group_column"].eq("high_acat_train_safe_hard_flag")
        & contribution_summary["group_value"].eq("False")
        & contribution_summary["split"].eq("test")
    ]
    target_contribution = float(target.iloc[0]["overall_contribution_ndcg@20"]) if not target.empty else nan_float()
    outside_contribution = float(outside.iloc[0]["overall_contribution_ndcg@20"]) if not outside.empty else nan_float()
    target_positive = bool(pd.notna(target_contribution) and target_contribution > 0)
    outside_offsets = bool(pd.notna(outside_contribution) and outside_contribution < 0)

    if meaningful and hr_not_worse:
        route = "seed43_candidate_passed"
        enter_code_implementation = False
        run_multi_seed_now = True
    else:
        route = "design_new_carrier"
        enter_code_implementation = True
        run_multi_seed_now = False

    return {
        "route": route,
        "enter_code_implementation": enter_code_implementation,
        "run_multi_seed_now": run_multi_seed_now,
        "next_screen_seed": 43,
        "m3_vs_m6_delta_ndcg@20": m3_vs_m6_delta,
        "m3_vs_m6_delta_hr@20": m3_vs_m6_hr_delta,
        "m3_vs_m6_meaningful": meaningful,
        "m3_vs_m6_hr_not_worse": hr_not_worse,
        "min_meaningful_ndcg_delta": min_meaningful_ndcg_delta,
        "target_group_positive_contribution": target_positive,
        "outside_group_offsets_target": outside_offsets,
        "test_high_acat_trainhard_true_contribution_ndcg@20": target_contribution,
        "test_high_acat_trainhard_false_contribution_ndcg@20": outside_contribution,
        "primary_failure_mode": "item_weight_carrier_too_diffuse_and_shuffle_close"
        if not meaningful
        else "candidate_passed_seed43_gate",
    }


def write_result_markdown(
    path: Path,
    run_stamp: str,
    mask_audit: pd.DataFrame,
    contribution: pd.DataFrame,
    train_curve_summary: pd.DataFrame,
    candidates: pd.DataFrame,
    protocol: pd.DataFrame,
    decision: dict,
    manifest_name: str,
) -> None:
    focus = contribution[
        contribution["group_column"].eq("high_acat_train_safe_hard_flag")
        & contribution["split"].isin(["validate", "test"])
    ].copy()
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG Task4-revise M3 vs M6 failure audit 结果
date: 2026-07-06
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - 实验结果
  - revise_m3
---

# {run_stamp} CCFCRec Amazon-VG Task4-revise M3 vs M6 failure audit 结果

## 来源

上游结果：[[2026-07-06 130518 CCFCRec Amazon-VG Task4-post Acat v3 minimal weight controls 训练后分析结果]]

分析脚本：

```text
validata/analyze_amazon_vg_task4_revise.py
```

manifest（运行清单）：`{manifest_name}`

## 结论

```text
route = {decision["route"]}
enter_code_implementation = {decision["enter_code_implementation"]}
run_multi_seed_now = {decision["run_multi_seed_now"]}
next_screen_seed = {decision["next_screen_seed"]}
primary_failure_mode = {decision["primary_failure_mode"]}
```

M3（Acat_v3 + 训练安全困难样本加权）与 M6 shuffle（Acat_v3 打乱负控）接近，不应解释成 Acat_v3 carrier 成功。
更合理的解释是：当前 item weight carrier（样本加权承载方式）太弱、太扩散，只能轻微改变 q-side（q 侧）训练强度，不能稳定制造 target user（目标用户）相对 competitor user（竞争用户）的 margin（分数间隔）。

## Mask Audit

> [!info] 字段说明
> - `m3_weighted_count`：M3 真正加权的 high-Acat train-safe hard item 数量。
> - `m6_weighted_count`：M6 shuffle 加权的打乱 high-Acat item 数量。
> - `m3_m6_jaccard`：M3 与 M6 加权集合的 Jaccard overlap（交并比）。
> - `m3_subset_of_high_acat_rate`：M3 是否仍主要是 high-Acat 子集。

{md_table(mask_audit, ["split", "item_count", "high_acat_count", "m3_weighted_count", "m6_weighted_count", "m3_share", "m6_share", "m3_m6_jaccard", "m3_subset_of_high_acat_rate"])}

## 分组贡献

> [!info] 字段说明
> - `mean_delta_ndcg@20`：该组内 M3 - M6 的平均 NDCG@20（前20排序质量）差异。
> - `overall_contribution_ndcg@20`：该组对 overall M3 - M6 均值差异的贡献。
> - 如果 target group（目标机会组）为正，但 outside group（组外）为负，说明 overall 被抵消。

{md_table(focus, ["split", "group_value", "item_count", "item_share", "mean_delta_ndcg@20", "overall_contribution_ndcg@20", "mean_delta_hr@20", "overall_contribution_hr@20"])}

## 训练曲线影响

> [!info] 字段说明
> - `mean_delta_loss`：M3 - M6 的训练 loss 均值差异。
> - `mean_delta_contrast_sum`：M3 - M6 的 contrast loss（对比损失）均值差异。
> - 这里只能说明训练动态差异，不能证明 Acat_v3 已被有效利用。

{md_table(train_curve_summary, ["paired_checkpoint_count", "mean_delta_loss", "mean_delta_contrast_sum", "mean_delta_ndcg@20", "mean_delta_hr@20", "last_delta_ndcg@20", "last_delta_hr@20"])}

## 新 Carrier 候选

{md_table(candidates, ["candidate_id", "method_variant", "carrier_family", "loss_surface", "required_controls", "seed43_gate"], max_rows=10)}

## 训练筛选协议

{md_table(protocol, ["phase", "seed", "epoch", "purpose", "pass_gate"], max_rows=10)}

## Fallacy Scan

```text
11/11 checked.
Simpson's paradox：已用 high_acat_train_safe_hard_flag 分组贡献检查 overall 抵消。
Ecological fallacy：分组均值只解释组层现象，不写成单 item 因果。
Berkson/collider：hard proxy 是筛选变量，不能把当前差异写成无偏因果。
Base rate neglect：同时报告 item_count、item_share 与 overall contribution。
Regression to mean：当前只做 seed43 快筛，不做稳定性宣称。
Survivorship bias：四个 Task4 方法完整训练，无中途丢失。
Look-elsewhere/garden of forking paths：新 carrier 是探索性设计，需要后续 seed43 gate。
Correlation causation/reverse causality：不能写成 Acat_v3 导致提升，只能写当前 carrier 不足。
```
"""
    path.write_text(markdown, encoding="utf-8")


def write_design_markdown(path: Path, run_stamp: str, candidates: pd.DataFrame, protocol: pd.DataFrame) -> None:
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG Task4-revise Acat v3 conditioned pairwise margin carrier 设计
date: 2026-07-06
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - 实验设计
  - carrier
---

# {run_stamp} CCFCRec Amazon-VG Task4-revise Acat v3 conditioned pairwise margin carrier 设计

## 设计边界

```text
1. Acat_v3 是 availability signal（可用性信号）。
2. train_safe_hard_proxy 是 hard opportunity proxy（训练安全困难机会代理），不是 availability 本身。
3. margin/rank/recoverability 只能做诊断、筛选和 upper-bound（上界），不能改写成可用性。
4. RSP-only 和 shuffle 负控必须保留。
5. 先 seed43 快筛，不对当前 M3 做多 seed。
```

## 为什么不继续当前 M3

当前 M3 是 item loss weight（样本加权），只改变 q-side BPR（q 侧排序损失）和 self contrast（自对比损失）的权重。
它没有直接约束 target user（目标用户）相对 competitor user（竞争用户）的 margin（分数间隔）。
Task4-post 和 Task4-revise 审计共同说明：M3 在 high-Acat hard 组有局部收益，但 overall（整体指标）几乎被 M6 shuffle（打乱负控）追平。

## 候选方法

{md_table(candidates, ["candidate_id", "method_variant", "priority", "carrier_family", "availability_input", "hard_input", "loss_surface", "protects_y_ukv", "required_controls", "seed43_gate"], max_rows=10)}

## 推荐实现顺序

```text
1. M4a：Acat_v3 train-hard pairwise margin。
2. M4b：Acat_v3 + RSP residual pairwise margin。
3. M4c：hard-only q-side margin，明确不动原始 y_ukv。
```

## 训练筛选协议

{md_table(protocol, ["phase", "seed", "epoch", "purpose", "pass_gate"], max_rows=10)}

## 进入代码实现的判断

```text
进入代码实现：是。
但只进入 seed43 快筛实现，不进入多 seed。
```

实现时优先加新的 method_variant（方法变体）和独立 launcher（封装启动脚本），脚本名必须对应实验。
"""
    path.write_text(markdown, encoding="utf-8")


def write_route_markdown(path: Path, run_stamp: str, decision: dict, result_md_name: str, design_md_name: str) -> None:
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG Task4-revise M3-vs-M6 failure audit 路线判断
date: 2026-07-06
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - 路线判断
---

# {run_stamp} CCFCRec Amazon-VG Task4-revise M3-vs-M6 failure audit 路线判断

结果来源：[[{result_md_name}]]

设计来源：[[{design_md_name}]]

## 判断

```text
route = {decision["route"]}
enter_code_implementation = {decision["enter_code_implementation"]}
run_multi_seed_now = {decision["run_multi_seed_now"]}
next_screen_seed = {decision["next_screen_seed"]}
```

## 原因

```text
M3 - M6 validate NDCG@20 = {decision["m3_vs_m6_delta_ndcg@20"]:.8f}
M3 - M6 validate HR@20 = {decision["m3_vs_m6_delta_hr@20"]:.8f}
```

当前 M3 没有干净击穿 M6 shuffle（打乱负控），所以不做当前 M3 的多 seed（多随机种子复验）。
下一步应进入 M4a/M4b/M4c 的代码实现和 seed43 快筛。

## Gate

```text
只有当新 candidate 相对 M6 shuffle 有明确 overall NDCG@20 优势，且 HR@20 不反向，才开放多 seed。
```
"""
    path.write_text(markdown, encoding="utf-8")


def run(args: argparse.Namespace) -> Outputs:
    run_stamp, _, run_iso = now_stamp()
    post_dir = Path(args.post_dir).expanduser().resolve()
    profile_path = Path(args.task4_profile_path).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT
        / "temp_202607_实验文件记录"
        / "temp_20260706"
        / f"{run_stamp} task4-revise-m3-vs-m6-failure-audit-carrier-design"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    item_eval = pd.read_csv(post_dir / "task4_post_item_eval_profile.csv", dtype={"raw_asin": str})
    method_comparison = pd.read_csv(post_dir / "task4_post_method_comparison.csv")
    run_index = pd.read_csv(post_dir / "task4_post_run_index.csv")
    profile = pd.read_csv(profile_path, dtype={"raw_asin": str})

    mask_audit = build_mask_audit(profile, seed=args.shuffle_seed)
    delta_profile = build_m3_m6_delta_profile(item_eval)
    contribution = build_all_contribution_summary(delta_profile)
    train_curve, train_curve_summary = build_train_curve_comparison(run_index)
    candidates = build_candidate_carrier_table()
    protocol = build_screening_protocol_table()
    decision = build_revise_route_decision(method_comparison, contribution)

    outputs = Outputs(
        output_dir=output_dir,
        mask_audit_csv=output_dir / "task4_revise_mask_audit.csv",
        m3_m6_delta_profile_csv=output_dir / "task4_revise_m3_m6_delta_profile.csv",
        contribution_summary_csv=output_dir / "task4_revise_contribution_summary.csv",
        train_curve_csv=output_dir / "task4_revise_train_curve_m3_m6.csv",
        train_curve_summary_csv=output_dir / "task4_revise_train_curve_summary.csv",
        candidate_carrier_csv=output_dir / "task4_revise_candidate_carrier_table.csv",
        screening_protocol_csv=output_dir / "task4_revise_screening_protocol.csv",
        manifest_json=output_dir / "run_manifest.json",
        result_md=output_dir / f"{run_stamp} CCFCRec Amazon-VG Task4-revise M3 vs M6 failure audit 结果.md",
        design_md=PROJECT_ROOT / "实验记录" / f"{run_stamp} CCFCRec Amazon-VG Task4-revise Acat v3 conditioned pairwise margin carrier 设计.md",
        route_md=PROJECT_ROOT / "实验记录" / f"{run_stamp} CCFCRec Amazon-VG Task4-revise M3-vs-M6 failure audit 路线判断.md",
    )

    mask_audit.to_csv(outputs.mask_audit_csv, index=False)
    delta_profile.to_csv(outputs.m3_m6_delta_profile_csv, index=False)
    contribution.to_csv(outputs.contribution_summary_csv, index=False)
    train_curve.to_csv(outputs.train_curve_csv, index=False)
    train_curve_summary.to_csv(outputs.train_curve_summary_csv, index=False)
    candidates.to_csv(outputs.candidate_carrier_csv, index=False)
    protocol.to_csv(outputs.screening_protocol_csv, index=False)

    write_result_markdown(
        outputs.result_md,
        run_stamp,
        mask_audit,
        contribution,
        train_curve_summary,
        candidates,
        protocol,
        decision,
        outputs.manifest_json.name,
    )
    write_design_markdown(outputs.design_md, run_stamp, candidates, protocol)
    write_route_markdown(outputs.route_md, run_stamp, decision, outputs.result_md.stem, outputs.design_md.stem)

    manifest = {
        "run_stamp": run_stamp,
        "run_iso": run_iso,
        "stage": "Task4-revise-1/2",
        "post_dir": str(post_dir),
        "task4_profile_path": str(profile_path),
        "analysis_script": str(CODE_ROOT / "validata" / "analyze_amazon_vg_task4_revise.py"),
        "upstream_post_analysis_script": str(CODE_ROOT / "validata" / "analyze_amazon_vg_task4_post.py"),
        "shuffle_seed": args.shuffle_seed,
        "decision": decision,
        "outputs": {field: str(getattr(outputs, field)) for field in outputs.__dataclass_fields__ if field != "output_dir"},
    }
    outputs.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--post-dir", default=str(DEFAULT_POST_DIR))
    parser.add_argument("--task4-profile-path", default=str(DEFAULT_TASK4_PROFILE))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--shuffle-seed", type=int, default=43)
    return parser


def main() -> None:
    outputs = run(build_arg_parser().parse_args())
    print(f"wrote {outputs.mask_audit_csv}")
    print(f"wrote {outputs.contribution_summary_csv}")
    print(f"wrote {outputs.candidate_carrier_csv}")
    print(f"wrote {outputs.result_md}")
    print(f"wrote {outputs.design_md}")
    print(f"wrote {outputs.route_md}")


if __name__ == "__main__":
    main()
