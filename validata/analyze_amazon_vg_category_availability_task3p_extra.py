#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.24-3.43 预防性发散诊断。

脚本只复用已有 item-level CSV，不训练模型。
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
from analyze_amazon_vg_category_availability_task3p10 import _control_bucket, category_count_bucket


DESIGN_NOTE = "上游设计：[[2026-07-05 021208 CCFCRec Amazon-VG category availability v2 Task3.24-3.43 预防性发散诊断设计]]"


def now_stamp() -> tuple[str, str]:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def _corr(x: pd.Series, y: pd.Series, method: str = "spearman") -> float:
    valid = pd.concat([pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")], axis=1).dropna()
    if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))


def _prefix_bucket(series: pd.Series, prefix: str) -> pd.Series:
    base = _quantile_labels(pd.to_numeric(series, errors="coerce"))
    return base.map(
        {
            "low": f"{prefix}_low",
            "mid": f"{prefix}_mid",
            "high": f"{prefix}_high",
            "all": f"{prefix}_all",
            "unknown": f"{prefix}_unknown",
        }
    ).fillna(base)


def _norm01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.min()
    hi = values.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=series.index)
    return (values - lo) / (hi - lo)


def _mean_or_nan(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return float("nan")
    return _safe_mean(frame[column])


def _merge_selected(base: pd.DataFrame, other: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    keep = ["asin", *[column for column in columns if column in other.columns and column not in base.columns]]
    if len(keep) <= 1:
        return base
    return base.merge(other[keep].drop_duplicates("asin"), on="asin", how="left")


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(Path(path).expanduser().resolve())


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_simple_md(
    output_path: Path,
    title: str,
    task_tag: str,
    result_dir_display: str,
    decision: dict[str, Any],
    sections: list[tuple[str, pd.DataFrame, list[str]]],
) -> None:
    run_stamp = title[:17]
    body = []
    for section_title, frame, columns in sections:
        body.append(
            f"""## {section_title}

{md_table(frame, [column for column in columns if column in frame.columns], max_rows=20)}
"""
        )
    markdown = f"""---
title: {title}
date: {run_stamp[:10]}
time: "{run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
created_at: "{run_stamp[:10]} {run_stamp[11:13]}:{run_stamp[13:15]}:{run_stamp[15:17]}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - {task_tag}
---

# {title}

## 来源说明

> [!info] 来源说明
> {DESIGN_NOTE}
> 本结果目录：`{result_dir_display}`

## 结论

route decision 为 `{decision.get("route")}`。

{decision.get("reason", "")}

{chr(10).join(body)}
## 判断

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```
"""
    output_path.write_text(markdown, encoding="utf-8")


def _manifest(script: Path, output_dir: Path, outputs: dict[str, Path], decision: dict[str, Any], row_counts: dict[str, int]) -> dict[str, Any]:
    _, run_iso = now_stamp()
    return {
        "run_time": run_iso,
        "script": str(script),
        "output_dir": str(output_dir),
        "outputs": {key: str(value) for key, value in outputs.items()},
        "row_counts": row_counts,
        "route_decision": decision,
        "design_note": DESIGN_NOTE,
    }


def write_task_output(
    output_root: Path,
    result_root_display: str,
    task_no: str,
    slug: str,
    title_suffix: str,
    decision: dict[str, Any],
    outputs: dict[str, pd.DataFrame],
) -> Path:
    dir_stamp, _ = now_stamp()
    output_dir = output_root / f"{dir_stamp} category-availability-v2-task3p{task_no}-{slug}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, Path] = {}
    file_prefix = f"task3p{task_no}"
    for name, frame in outputs.items():
        path = output_dir / f"{file_prefix}_{name}.csv"
        frame.to_csv(path, index=False)
        output_paths[name] = path
    route_path = output_dir / f"{file_prefix}_route_decision.json"
    _write_json(route_path, decision)
    output_paths["route_decision"] = route_path
    result_md_path = output_dir / f"{dir_stamp} CCFCRec Amazon-VG category availability v2 Task3.{task_no} {title_suffix} 诊断结果.md"
    first_name, first_frame = next(iter(outputs.items()))
    _write_simple_md(
        result_md_path,
        f"{dir_stamp} CCFCRec Amazon-VG category availability v2 Task3.{task_no} {title_suffix} 诊断结果",
        f"Task3.{task_no}",
        f"{result_root_display.rstrip('/')}/{output_dir.name}/",
        decision,
        [(first_name, first_frame, list(first_frame.columns)[:12])],
    )
    output_paths["result_md"] = result_md_path
    manifest_path = output_dir / "run_manifest.json"
    manifest = _manifest(
        Path(__file__).resolve(),
        output_dir,
        output_paths,
        decision,
        {key: int(len(frame)) for key, frame in outputs.items()},
    )
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_dir


def build_threshold_sensitivity_summary(profile: pd.DataFrame, metric: str = "margin_proxy", cuts: tuple[float, ...] = (0.2, 0.25, 0.33, 0.4)) -> pd.DataFrame:
    values = pd.to_numeric(profile[metric], errors="coerce")
    rows = []
    for cut in cuts:
        low_threshold = values.quantile(cut)
        high_threshold = values.quantile(1.0 - cut)
        low = profile[values <= low_threshold]
        high = profile[values >= high_threshold]
        rows.append(
            {
                "metric": metric,
                "cut": cut,
                "low_threshold": low_threshold,
                "high_threshold": high_threshold,
                "low_item_count": int(len(low)),
                "high_item_count": int(len(high)),
                "low_ndcg@20_mean": _safe_mean(low["ndcg@20"]),
                "high_ndcg@20_mean": _safe_mean(high["ndcg@20"]),
                "high_minus_low_ndcg@20": _safe_mean(high["ndcg@20"]) - _safe_mean(low["ndcg@20"]),
                "low_margin_mean": _mean_or_nan(low, metric),
                "high_margin_mean": _mean_or_nan(high, metric),
            }
        )
    return pd.DataFrame(rows)


def decide_threshold_sensitivity_route(summary: pd.DataFrame) -> dict[str, Any]:
    gaps = pd.to_numeric(summary["high_minus_low_ndcg@20"], errors="coerce").dropna()
    min_gap = float(gaps.min()) if not gaps.empty else float("nan")
    positive_rate = float((gaps > 0).mean()) if not gaps.empty else 0.0
    if positive_rate == 1.0 and min_gap > 0.10:
        route = "margin_threshold_robust"
        reason = "多个 quantile cut 下 margin high-low NDCG gap 都保持强正向。"
    elif positive_rate >= 0.75 and min_gap > 0.02:
        route = "margin_threshold_fragile"
        reason = "margin gap 方向大体稳定，但对阈值敏感。"
    else:
        route = "margin_threshold_not_predictive"
        reason = "margin threshold 变化后 gap 不稳定。"
    return {"route": route, "reason": reason, "evidence": {"min_gap": _jsonable(min_gap), "positive_rate": _jsonable(positive_rate)}}


def build_cross_stability_profile(rank_profile: pd.DataFrame, residual_profile: pd.DataFrame) -> pd.DataFrame:
    frame = rank_profile.copy()
    frame = _merge_selected(frame, residual_profile, ["residual_bucket", "ndcg@20_control_residual", "matched_cell_size"])
    frame["rank_residual_group"] = frame["rank_aware_group"].astype(str) + "__" + frame["residual_bucket"].astype(str)
    return frame


def build_cross_stability_summary(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["rank_aware_group", "residual_bucket", "rank_residual_group"], dropna=False)
        .agg(
            item_count=("asin", "size"),
            ndcg20_mean=("ndcg@20", "mean"),
            margin_mean=("margin_to_top20_cutoff", "mean"),
            residual_mean=("ndcg@20_control_residual", "mean"),
        )
        .reset_index()
        .rename(columns={"ndcg20_mean": "ndcg@20_mean"})
    )


def decide_cross_stability_route(summary: pd.DataFrame) -> dict[str, Any]:
    hard = summary[summary["rank_residual_group"].astype(str).str.contains("rank_low__hard", regex=False)]
    if hard.empty:
        return {"route": "rank_residual_complementary", "reason": "rank_low 与 residual_hard 没有稳定重叠。", "evidence": {}}
    hard_row = hard.iloc[0]
    all_ndcg = pd.to_numeric(summary["ndcg@20_mean"], errors="coerce")
    if int(hard_row["item_count"]) >= 50 and float(hard_row["ndcg@20_mean"]) <= float(all_ndcg.quantile(0.25)):
        route = "rank_residual_overlap_ready"
        reason = "rank_low + residual_hard 形成足量且低 NDCG 的交集。"
    elif int(hard_row["item_count"]) >= 50:
        route = "rank_residual_complementary"
        reason = "rank 与 residual 有重叠但 hard 程度不极端。"
    else:
        route = "rank_residual_conflicting"
        reason = "rank 与 residual hard 交集过小。"
    return {"route": route, "reason": reason, "evidence": hard_row.to_dict()}


def build_consensus_component_ablation(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    components = {
        "residual_hard": frame.get("residual_bucket", "").eq("hard").astype(int),
        "margin_low_mid": frame.get("margin_bucket", "").isin(["low", "mid"]).astype(int),
        "target_gap_negative": (pd.to_numeric(frame.get("target_competitor_gap_proxy", 0), errors="coerce") < 0).astype(int),
        "modality_negative": (pd.to_numeric(frame.get("modality_alignment_proxy", 0), errors="coerce") < 0).astype(int),
    }
    rows = []
    full_score = sum(components.values())
    for removed in ["none", *components.keys()]:
        score = full_score if removed == "none" else full_score - components[removed]
        bucket = _prefix_bucket(score, "score")
        low = frame[bucket.eq("score_low")]
        high = frame[bucket.eq("score_high")]
        rows.append(
            {
                "removed_component": removed,
                "hard_item_count": int(len(high)),
                "easy_item_count": int(len(low)),
                "easy_minus_hard_ndcg@20": _safe_mean(low["ndcg@20"]) - _safe_mean(high["ndcg@20"]),
                "hard_ndcg@20_mean": _safe_mean(high["ndcg@20"]),
                "easy_ndcg@20_mean": _safe_mean(low["ndcg@20"]),
            }
        )
    return pd.DataFrame(rows)


def decide_consensus_component_route(summary: pd.DataFrame) -> dict[str, Any]:
    full = summary[summary["removed_component"].eq("none")].iloc[0]
    ablated = summary[~summary["removed_component"].eq("none")]
    full_gap = float(full["easy_minus_hard_ndcg@20"])
    min_ablation_gap = float(pd.to_numeric(ablated["easy_minus_hard_ndcg@20"], errors="coerce").min()) if not ablated.empty else float("nan")
    if full_gap > 0.05 and min_ablation_gap > 0.02:
        route = "consensus_multi_component_needed"
        reason = "删除任一组件后 consensus 仍有信号，说明多组件组合稳健。"
    elif full_gap > 0.05:
        route = "single_signal_dominates_consensus"
        reason = "某个组件删除后 consensus gap 明显塌缩。"
    else:
        route = "consensus_components_unstable"
        reason = "full consensus gap 本身不足。"
    return {"route": route, "reason": reason, "evidence": {"full_gap": _jsonable(full_gap), "min_ablation_gap": _jsonable(min_ablation_gap)}}


def build_tradeoff_summary(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    hard_mask = (
        frame.get("consensus_group", "").astype(str).str.contains("high")
        | frame.get("rank_aware_group", "").astype(str).str.contains("low")
        | frame.get("residual_bucket", "").astype(str).eq("hard")
    )
    easy_mask = (
        frame.get("consensus_group", "").astype(str).str.contains("low")
        | frame.get("rank_aware_group", "").astype(str).str.contains("high")
        | frame.get("residual_bucket", "").astype(str).eq("easy")
    )
    rows = [
        {"group": "hard_like", "item_count": int(hard_mask.sum()), "delta_ndcg@20_mean": _safe_mean(frame.loc[hard_mask, "delta_ndcg@20"])},
        {"group": "easy_like", "item_count": int(easy_mask.sum()), "delta_ndcg@20_mean": _safe_mean(frame.loc[easy_mask, "delta_ndcg@20"])},
        {"group": "all", "item_count": int(len(frame)), "delta_ndcg@20_mean": _safe_mean(frame["delta_ndcg@20"])},
    ]
    for group_col in ["consensus_group", "rank_aware_group", "residual_bucket"]:
        if group_col not in frame.columns:
            continue
        for value, sub in frame.groupby(group_col, dropna=False):
            rows.append({"group": f"{group_col}={value}", "item_count": int(len(sub)), "delta_ndcg@20_mean": _safe_mean(sub["delta_ndcg@20"])})
    return pd.DataFrame(rows)


def decide_tradeoff_route(summary: pd.DataFrame) -> dict[str, Any]:
    hard = summary[summary["group"].eq("hard_like")].iloc[0]
    easy = summary[summary["group"].eq("easy_like")].iloc[0]
    hard_delta = float(hard["delta_ndcg@20_mean"])
    easy_delta = float(easy["delta_ndcg@20_mean"])
    if hard_delta > 0 and easy_delta < 0:
        route = "hard_gain_easy_loss_tradeoff"
        reason = "method 对 hard-like 组有收益，但对 easy-like 组有损失。"
    elif hard_delta > 0 and easy_delta >= 0:
        route = "method_gain_broadly_positive"
        reason = "hard-like 与 easy-like 组均非负。"
    else:
        route = "method_delta_noise"
        reason = "method delta 没有形成可利用 tradeoff。"
    return {"route": route, "reason": reason, "evidence": {"hard_like_delta_ndcg@20": _jsonable(hard_delta), "easy_like_delta_ndcg@20": _jsonable(easy_delta)}}


def build_norm_shift_summary(delta: pd.DataFrame) -> pd.DataFrame:
    frame = delta.copy()
    frame["abs_delta_q_norm"] = pd.to_numeric(frame["delta_q_norm"], errors="coerce").abs()
    frame["norm_shift_bucket"] = _prefix_bucket(frame["abs_delta_q_norm"], "norm_shift")
    rows = []
    for bucket, sub in frame.groupby("norm_shift_bucket", dropna=False):
        rows.append(
            {
                "norm_shift_bucket": bucket,
                "item_count": int(len(sub)),
                "delta_ndcg@20_mean": _safe_mean(sub["delta_ndcg@20"]),
                "abs_delta_q_norm_mean": _safe_mean(sub["abs_delta_q_norm"]),
                "delta_margin_to_top20_cutoff_mean": _safe_mean(sub["delta_margin_to_top20_cutoff"]),
            }
        )
    rows.append({"norm_shift_bucket": "corr", "item_count": int(len(frame)), "delta_ndcg@20_mean": _corr(frame["abs_delta_q_norm"], frame["delta_ndcg@20"]), "abs_delta_q_norm_mean": _safe_mean(frame["abs_delta_q_norm"]), "delta_margin_to_top20_cutoff_mean": _corr(frame["abs_delta_q_norm"], frame["delta_margin_to_top20_cutoff"])})
    return pd.DataFrame(rows)


def decide_norm_shift_route(summary: pd.DataFrame) -> dict[str, Any]:
    buckets = summary[summary["norm_shift_bucket"].astype(str).str.startswith("norm_shift")]
    high = buckets[buckets["norm_shift_bucket"].astype(str).str.contains("high")]
    low = buckets[buckets["norm_shift_bucket"].astype(str).str.contains("low")]
    high_delta = float(high.iloc[0]["delta_ndcg@20_mean"]) if not high.empty else float("nan")
    low_delta = float(low.iloc[0]["delta_ndcg@20_mean"]) if not low.empty else float("nan")
    if high_delta < low_delta - 0.001:
        route = "norm_shift_harm_risk_high"
        reason = "大 norm shift 桶的 delta_ndcg 更差。"
    elif high_delta >= low_delta:
        route = "norm_shift_mostly_benign"
        reason = "大 norm shift 桶未显示更高损害。"
    else:
        route = "norm_shift_not_diagnostic"
        reason = "norm shift 与 delta 关系弱。"
    return {"route": route, "reason": reason, "evidence": {"high_delta": _jsonable(high_delta), "low_delta": _jsonable(low_delta)}}


def build_lift_decomposition_summary(delta: pd.DataFrame) -> pd.DataFrame:
    frame = delta.copy()
    frame["target_lift_bucket"] = _prefix_bucket(frame["delta_target_score_max"], "target_lift")
    frame["competitor_pressure_bucket"] = _prefix_bucket(frame["delta_top20_user_norm_mean"].fillna(0) + frame["delta_top20_history_interaction_count_mean"].fillna(0), "competitor_lift")
    rows = []
    for group_col in ["target_lift_bucket", "competitor_pressure_bucket"]:
        for value, sub in frame.groupby(group_col, dropna=False):
            rows.append(
                {
                    "group_col": group_col,
                    "group_value": value,
                    "item_count": int(len(sub)),
                    "delta_ndcg@20_mean": _safe_mean(sub["delta_ndcg@20"]),
                    "delta_margin_to_top20_cutoff_mean": _safe_mean(sub["delta_margin_to_top20_cutoff"]),
                    "delta_target_score_max_mean": _safe_mean(sub["delta_target_score_max"]),
                }
            )
    return pd.DataFrame(rows)


def decide_lift_decomposition_route(summary: pd.DataFrame) -> dict[str, Any]:
    target_high = summary[summary["group_col"].eq("target_lift_bucket") & summary["group_value"].astype(str).str.contains("high")]
    comp_high = summary[summary["group_col"].eq("competitor_pressure_bucket") & summary["group_value"].astype(str).str.contains("high")]
    target_delta = float(target_high.iloc[0]["delta_ndcg@20_mean"]) if not target_high.empty else float("nan")
    comp_delta = float(comp_high.iloc[0]["delta_ndcg@20_mean"]) if not comp_high.empty else float("nan")
    if target_delta > 0 and comp_delta < 0:
        route = "target_lift_effective"
        reason = "target lift 有收益，而 competitor pressure lift 有损害。"
    elif target_delta > 0 and comp_delta >= 0:
        route = "target_lift_without_margin"
        reason = "target lift 有收益，但 competitor pressure 未形成明显损害。"
    elif comp_delta < 0:
        route = "competitor_lift_harm"
        reason = "competitor pressure high 对 delta 不利。"
    else:
        route = "pairwise_margin_improvement_needed"
        reason = "target/competitor 单侧 lift 无法解释，需要 pairwise margin。"
    return {"route": route, "reason": reason, "evidence": {"target_high_delta": _jsonable(target_delta), "competitor_high_delta": _jsonable(comp_delta)}}


def build_rank_jump_summary(delta: pd.DataFrame) -> pd.DataFrame:
    frame = delta.copy()
    frame["baseline_rank_band"] = pd.cut(
        pd.to_numeric(frame["baseline_best_target_rank"], errors="coerce"),
        bins=[-1, 20, 100, 1000, float("inf")],
        labels=["top20", "near_100", "mid_1000", "deep_1000_plus"],
    ).astype(str)
    frame["rank_improvement"] = -pd.to_numeric(frame["delta_best_target_rank"], errors="coerce")
    return (
        frame.groupby("baseline_rank_band", dropna=False)
        .agg(item_count=("asin", "size"), delta_ndcg20_mean=("delta_ndcg@20", "mean"), rank_improvement_mean=("rank_improvement", "mean"))
        .reset_index()
        .rename(columns={"delta_ndcg20_mean": "delta_ndcg@20_mean"})
    )


def decide_rank_jump_route(summary: pd.DataFrame) -> dict[str, Any]:
    near = summary[summary["baseline_rank_band"].isin(["top20", "near_100"])]
    deep = summary[summary["baseline_rank_band"].isin(["mid_1000", "deep_1000_plus"])]
    near_delta = _safe_mean(near["delta_ndcg@20_mean"]) if not near.empty else float("nan")
    deep_delta = _safe_mean(deep["delta_ndcg@20_mean"]) if not deep.empty else float("nan")
    if near_delta > deep_delta and near_delta > 0:
        route = "near_cutoff_recovery_only"
        reason = "method 主要帮助 near-cutoff / shallow rank item。"
    elif deep_delta > 0:
        route = "deep_rank_recovery_signal"
        reason = "deep rank item 也有恢复信号。"
    else:
        route = "rank_jump_not_useful"
        reason = "rank band response 无明显可用信号。"
    return {"route": route, "reason": reason, "evidence": {"near_delta": _jsonable(near_delta), "deep_delta": _jsonable(deep_delta)}}


def build_cutoff_fragility_summary(baseline_score: pd.DataFrame, delta: pd.DataFrame) -> pd.DataFrame:
    frame = baseline_score[["asin", "local_gap_20_21", "margin_to_top20_cutoff"]].merge(delta[["asin", "delta_ndcg@20", "delta_hr@20"]], on="asin", how="left")
    frame["abs_delta_ndcg@20"] = pd.to_numeric(frame["delta_ndcg@20"], errors="coerce").abs()
    frame["local_gap_bucket"] = _prefix_bucket(-pd.to_numeric(frame["local_gap_20_21"], errors="coerce"), "fragility")
    return (
        frame.groupby("local_gap_bucket", dropna=False)
        .agg(item_count=("asin", "size"), abs_delta_ndcg20_mean=("abs_delta_ndcg@20", "mean"), delta_ndcg20_mean=("delta_ndcg@20", "mean"), local_gap_mean=("local_gap_20_21", "mean"))
        .reset_index()
        .rename(columns={"abs_delta_ndcg20_mean": "abs_delta_ndcg@20_mean", "delta_ndcg20_mean": "delta_ndcg@20_mean"})
    )


def decide_cutoff_fragility_route(summary: pd.DataFrame) -> dict[str, Any]:
    high = summary[summary["local_gap_bucket"].astype(str).str.contains("high")]
    low = summary[summary["local_gap_bucket"].astype(str).str.contains("low")]
    high_abs = float(high.iloc[0]["abs_delta_ndcg@20_mean"]) if not high.empty else float("nan")
    low_abs = float(low.iloc[0]["abs_delta_ndcg@20_mean"]) if not low.empty else float("nan")
    if high_abs > low_abs + 0.001:
        route = "cutoff_boundary_fragile"
        reason = "cutoff fragile 组的 method 扰动更大。"
    elif high_abs <= low_abs:
        route = "cutoff_gap_not_driver"
        reason = "局部 gap 小并未带来更高扰动。"
    else:
        route = "cutoff_signal_confounded"
        reason = "cutoff fragility 信号弱。"
    return {"route": route, "reason": reason, "evidence": {"fragile_abs_delta": _jsonable(high_abs), "stable_abs_delta": _jsonable(low_abs)}}


def build_rank_band_response_summary(delta: pd.DataFrame) -> pd.DataFrame:
    return build_rank_jump_summary(delta)


def decide_rank_band_response_route(summary: pd.DataFrame) -> dict[str, Any]:
    best = summary.sort_values("delta_ndcg@20_mean", ascending=False).iloc[0]
    if float(best["delta_ndcg@20_mean"]) > 0.001 and int(best["item_count"]) >= 50:
        route = "rank_band_response_actionable"
        reason = "至少一个 baseline rank band 有可用正响应。"
    elif str(best["baseline_rank_band"]) == "top20":
        route = "only_easy_band_helped"
        reason = "收益主要在 easy/top20 band。"
    else:
        route = "rank_band_response_flat"
        reason = "rank band response 不明显。"
    return {"route": route, "reason": reason, "evidence": best.to_dict()}


def build_cold_start_summary(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    frame["target_history_bucket"] = _prefix_bucket(frame["target_history_interaction_count_mean"], "target_history")
    rows = []
    for group_cols in [["target_history_bucket"], ["target_history_bucket", "rank_aware_group"], ["target_history_bucket", "residual_bucket"]]:
        for keys, sub in frame.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: value for col, value in zip(group_cols, keys)}
            row.update({"scope": "+".join(group_cols), "item_count": int(len(sub)), "ndcg@20_mean": _safe_mean(sub["ndcg@20"]), "margin_to_top20_cutoff_mean": _safe_mean(sub["margin_to_top20_cutoff"])})
            rows.append(row)
    return pd.DataFrame(rows)


def decide_cold_start_route(summary: pd.DataFrame) -> dict[str, Any]:
    single = summary[summary["scope"].eq("target_history_bucket")]
    low = single[single["target_history_bucket"].astype(str).str.contains("low")]
    high = single[single["target_history_bucket"].astype(str).str.contains("high")]
    gap = float(high.iloc[0]["ndcg@20_mean"] - low.iloc[0]["ndcg@20_mean"]) if not low.empty and not high.empty else float("nan")
    if gap > 0.10:
        route = "target_cold_start_key_risk"
        reason = "target history low 与 failure 强相关。"
    elif gap > 0.03:
        route = "cold_start_interacts_with_margin"
        reason = "cold-start 有中等信号，应和 rank/margin 联合使用。"
    else:
        route = "cold_start_not_primary"
        reason = "cold-start 不是主要 failure driver。"
    return {"route": route, "reason": reason, "evidence": {"history_high_minus_low_ndcg@20": _jsonable(gap)}}


def build_competitor_activity_conditional_summary(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    frame["top20_activity_bucket"] = _prefix_bucket(frame["top20_history_interaction_count_mean"], "top20_activity")
    rows = []
    for condition_col, condition_value in [("rank_aware_group", "rank_low"), ("residual_bucket", "hard"), ("all", "all")]:
        sub_frame = frame if condition_col == "all" else frame[frame[condition_col].eq(condition_value)]
        for bucket, sub in sub_frame.groupby("top20_activity_bucket", dropna=False):
            rows.append({"condition": f"{condition_col}={condition_value}", "top20_activity_bucket": bucket, "item_count": int(len(sub)), "ndcg@20_mean": _safe_mean(sub["ndcg@20"]), "margin_to_top20_cutoff_mean": _safe_mean(sub["margin_to_top20_cutoff"])})
    return pd.DataFrame(rows)


def decide_competitor_activity_route(summary: pd.DataFrame) -> dict[str, Any]:
    conditional = summary[summary["condition"].isin(["rank_aware_group=rank_low", "residual_bucket=hard"])]
    high = conditional[conditional["top20_activity_bucket"].astype(str).str.contains("high")]
    low = conditional[conditional["top20_activity_bucket"].astype(str).str.contains("low")]
    high_mean = _safe_mean(high["ndcg@20_mean"]) if not high.empty else float("nan")
    low_mean = _safe_mean(low["ndcg@20_mean"]) if not low.empty else float("nan")
    if high_mean < low_mean - 0.01:
        route = "conditional_competitor_suppression_opportunity"
        reason = "hard 条件内高活跃 competitor 对应更低 NDCG。"
    elif high_mean > low_mean:
        route = "competitor_activity_easy_bias"
        reason = "高活跃 competitor 更多出现在 easy item。"
    else:
        route = "competitor_activity_not_actionable"
        reason = "competitor activity 在 hard 条件内不可操作。"
    return {"route": route, "reason": reason, "evidence": {"conditional_high_ndcg": _jsonable(high_mean), "conditional_low_ndcg": _jsonable(low_mean)}}


def build_modality_conflict_profile(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    cols = [
        "target_minus_top20_history_q_cosine_mean",
        "target_minus_top20_history_attr_cosine_mean",
        "target_minus_top20_history_img_cosine_mean",
    ]
    values = pd.concat([pd.to_numeric(frame[col], errors="coerce") for col in cols if col in frame.columns], axis=1)
    frame["modality_gap_min"] = values.min(axis=1)
    frame["modality_gap_max"] = values.max(axis=1)
    frame["modality_gap_spread"] = frame["modality_gap_max"] - frame["modality_gap_min"]
    frame["modality_sign_disagreement"] = (frame["modality_gap_min"] < 0) & (frame["modality_gap_max"] > 0)
    frame["modality_conflict_score"] = frame["modality_gap_spread"].fillna(0) * frame["modality_sign_disagreement"].astype(int)
    frame["modality_conflict_group"] = _prefix_bucket(frame["modality_conflict_score"], "conflict")
    return frame


def build_modality_conflict_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, sub in frame.groupby("modality_conflict_group", dropna=False):
        rows.append(
            {
                "modality_conflict_group": group,
                "item_count": int(len(sub)),
                "ndcg@20_mean": _safe_mean(sub["ndcg@20"]),
                "conflict_score_mean": _safe_mean(sub["modality_conflict_score"]),
                "consensus_high_rate": float(sub.get("consensus_group", pd.Series("", index=sub.index)).astype(str).str.contains("high").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("modality_conflict_group").reset_index(drop=True)


def decide_modality_conflict_route(summary: pd.DataFrame) -> dict[str, Any]:
    high = summary[summary["modality_conflict_group"].astype(str).str.contains("high")]
    low = summary[summary["modality_conflict_group"].astype(str).str.contains("low")]
    high_ndcg = float(high.iloc[0]["ndcg@20_mean"]) if not high.empty else float("nan")
    low_ndcg = float(low.iloc[0]["ndcg@20_mean"]) if not low.empty else float("nan")
    hard_rate = float(high.iloc[0]["consensus_high_rate"]) if not high.empty else 0.0
    if low_ndcg - high_ndcg > 0.10 and hard_rate >= 0.4:
        route = "modality_conflict_actionable"
        reason = "modality conflict high 组低 NDCG 且与 consensus hard 重叠。"
    elif low_ndcg - high_ndcg <= 0.02:
        route = "single_modality_sufficient"
        reason = "modality conflict 未明显解释 failure。"
    else:
        route = "modality_conflict_not_predictive"
        reason = "modality conflict 有弱信号但不足以单独使用。"
    return {"route": route, "reason": reason, "evidence": {"low_minus_high_ndcg@20": _jsonable(low_ndcg - high_ndcg), "conflict_high_consensus_high_rate": _jsonable(hard_rate)}}


def build_coverage_summary(content: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = content.copy()
    count_cols = [col for col in frame.columns if col.endswith("_known_user_count") or col in ["target_known_history_user_count", "top20_known_history_user_count"]]
    if count_cols:
        frame["coverage_score"] = pd.concat([_norm01(frame[col]) for col in count_cols], axis=1).mean(axis=1)
    else:
        frame["coverage_score"] = 1.0
    frame["coverage_group"] = _prefix_bucket(frame["coverage_score"], "coverage")
    summary = (
        frame.groupby("coverage_group", dropna=False)
        .agg(item_count=("asin", "size"), ndcg20_mean=("ndcg@20", "mean"), margin_mean=("margin_to_top20_cutoff", "mean"), coverage_score_mean=("coverage_score", "mean"))
        .reset_index()
        .rename(columns={"ndcg20_mean": "ndcg@20_mean"})
    )
    return frame, summary


def decide_coverage_route(summary: pd.DataFrame) -> dict[str, Any]:
    high = summary[summary["coverage_group"].astype(str).str.contains("high")]
    low = summary[summary["coverage_group"].astype(str).str.contains("low")]
    gap = float(high.iloc[0]["ndcg@20_mean"] - low.iloc[0]["ndcg@20_mean"]) if not high.empty and not low.empty else float("nan")
    if gap > 0.05:
        route = "coverage_risk_high"
        reason = "low coverage 组显著更差。"
    elif gap > 0.02:
        route = "coverage_interacts_with_hard_group"
        reason = "coverage 有弱到中等风险，应作为 gate。"
    else:
        route = "coverage_risk_low"
        reason = "coverage 对 failure 解释有限。"
    return {"route": route, "reason": reason, "evidence": {"coverage_high_minus_low_ndcg@20": _jsonable(gap)}}


def build_tail_risk_summary(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    frame["category_count_bucket"] = frame["category_count"].map(category_count_bucket)
    frame["gt_tail_bucket"] = frame["gt_user_count"].map(lambda value: "gt_tail_low" if value <= 1 else ("gt_tail_mid" if value <= 3 else "gt_tail_high"))
    return (
        frame.groupby(["gt_tail_bucket", "category_count_bucket"], dropna=False)
        .agg(item_count=("asin", "size"), ndcg20_mean=("ndcg@20", "mean"), rank_low_rate=("rank_aware_group", lambda s: s.astype(str).eq("rank_low").mean()))
        .reset_index()
        .rename(columns={"ndcg20_mean": "ndcg@20_mean"})
    )


def decide_tail_risk_route(summary: pd.DataFrame) -> dict[str, Any]:
    tail = summary[summary["gt_tail_bucket"].eq("gt_tail_low")]
    other = summary[~summary["gt_tail_bucket"].eq("gt_tail_low")]
    gap = _safe_mean(other["ndcg@20_mean"]) - _safe_mean(tail["ndcg@20_mean"])
    if gap > 0.05:
        route = "joint_tail_risk_supported"
        reason = "gt/category tail 组明显更差。"
    elif gap > 0.02:
        route = "tail_risk_control_needed"
        reason = "tail 风险中等，需要作为控制变量。"
    else:
        route = "tail_risk_not_primary"
        reason = "tail 不是主要 failure driver。"
    return {"route": route, "reason": reason, "evidence": {"non_tail_minus_tail_ndcg@20": _jsonable(gap)}}


def build_v2_disagreement_summary(profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = profile.copy()
    frame["disc_collab_gap_abs"] = (pd.to_numeric(frame["s_cat_v2_disc_within_control"], errors="coerce") - pd.to_numeric(frame["s_cat_v2_collab_within_control"], errors="coerce")).abs()
    frame["v1_v2_gap_abs"] = (pd.to_numeric(frame["s_cat"], errors="coerce") - pd.to_numeric(frame["s_cat_v1"], errors="coerce")).abs()
    frame["v2_disagreement_score"] = frame[["disc_collab_gap_abs", "v1_v2_gap_abs"]].mean(axis=1)
    frame["v2_disagreement_group"] = _prefix_bucket(frame["v2_disagreement_score"], "v2_disagree")
    summary = (
        frame.groupby("v2_disagreement_group", dropna=False)
        .agg(item_count=("asin", "size"), ndcg20_mean=("ndcg@20", "mean"), rank_low_rate=("rank_aware_group", lambda s: s.astype(str).eq("rank_low").mean()), disagreement_mean=("v2_disagreement_score", "mean"))
        .reset_index()
        .rename(columns={"ndcg20_mean": "ndcg@20_mean"})
    )
    return frame, summary


def decide_v2_disagreement_route(summary: pd.DataFrame) -> dict[str, Any]:
    high = summary[summary["v2_disagreement_group"].astype(str).str.contains("high")]
    low = summary[summary["v2_disagreement_group"].astype(str).str.contains("low")]
    gap = float(low.iloc[0]["ndcg@20_mean"] - high.iloc[0]["ndcg@20_mean"]) if not high.empty and not low.empty else float("nan")
    if gap > 0.05:
        route = "v2_disagreement_explains_weak_failure"
        reason = "v2 component disagreement high 组更 hard。"
    elif gap > 0.02:
        route = "v2_components_need_rebuild"
        reason = "v2 disagreement 有弱信号，应重建变量。"
    else:
        route = "v2_disagreement_not_failure_related"
        reason = "v2 disagreement 与 failure 关系弱。"
    return {"route": route, "reason": reason, "evidence": {"low_minus_high_disagreement_ndcg@20": _jsonable(gap)}}


def build_pareto_summary(profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_col in ["rank_aware_group", "consensus_group", "residual_bucket", "target_history_bucket", "target_norm_bucket", "archetype"]:
        if group_col not in profile.columns:
            continue
        for value, sub in profile.groupby(group_col, dropna=False):
            mean_delta = _safe_mean(sub["delta_ndcg@20"])
            rows.append({"group_col": group_col, "group_value": value, "item_count": int(len(sub)), "delta_ndcg@20_mean": mean_delta, "pareto_label": "helped" if mean_delta > 0 else ("harmed" if mean_delta < 0 else "flat")})
    return pd.DataFrame(rows)


def decide_pareto_route(summary: pd.DataFrame) -> dict[str, Any]:
    helped = summary[(summary["pareto_label"].eq("helped")) & (summary["item_count"] >= 50)]
    harmed = summary[(summary["pareto_label"].eq("harmed")) & (summary["item_count"] >= 50)]
    if not helped.empty and not harmed.empty:
        route = "pareto_tradeoff_requires_gate"
        reason = "存在足量 helped 与 harmed subgroup，后续需要 gate。"
    elif not helped.empty:
        route = "pareto_safe_groups_found"
        reason = "主要找到 helped subgroup，harmed 不明显。"
    else:
        route = "pareto_signal_too_weak"
        reason = "Pareto subgroup 信号弱。"
    return {"route": route, "reason": reason, "evidence": {"helped_group_count": int(len(helped)), "harmed_group_count": int(len(harmed))}}


def build_checkpoint_group_stability(profile: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for group_value, sub in profile.groupby(group_col, dropna=False):
        by = sub.groupby("checkpoint_label", dropna=False)["ndcg@20"].mean()
        def get(label: str) -> float:
            return float(by[label]) if label in by.index else float("nan")
        rows.append(
            {
                "group_col": group_col,
                "group_value": group_value,
                "item_count": int(sub["asin"].nunique()),
                "baseline_best_ndcg@20": get("baseline_best"),
                "baseline_last_ndcg@20": get("baseline_last"),
                "category_conf_best_ndcg@20": get("category_conf_best"),
                "category_conf_last_ndcg@20": get("category_conf_last"),
                "method_best_minus_baseline_best_ndcg@20": get("category_conf_best") - get("baseline_best"),
                "method_last_minus_baseline_last_ndcg@20": get("category_conf_last") - get("baseline_last"),
            }
        )
    return pd.DataFrame(rows)


def decide_checkpoint_group_stability_route(summary: pd.DataFrame) -> dict[str, Any]:
    best = pd.to_numeric(summary["method_best_minus_baseline_best_ndcg@20"], errors="coerce")
    last = pd.to_numeric(summary["method_last_minus_baseline_last_ndcg@20"], errors="coerce")
    stable_positive = summary[(best > 0) & (last > 0)]
    unstable = summary[(best * last) < 0]
    if len(stable_positive) >= 1 and len(unstable) <= max(1, len(summary) // 5):
        route = "new_groups_checkpoint_stable"
        reason = "至少一个新 hard group 在 best/last 下 method delta 同向为正。"
    elif len(unstable) > 0:
        route = "new_groups_checkpoint_unstable"
        reason = "部分 group 的 method delta 在 best/last 下反向。"
    else:
        route = "checkpoint_signal_needs_retrain"
        reason = "checkpoint group signal 未形成稳定正向。"
    return {"route": route, "reason": reason, "evidence": {"stable_positive_group_count": int(len(stable_positive)), "unstable_group_count": int(len(unstable))}}


def build_archetype_response_summary(profile: pd.DataFrame) -> pd.DataFrame:
    expanded = []
    for _, row in profile.iterrows():
        for archetype in str(row.get("archetype", "uncategorized")).split(";"):
            data = row.to_dict()
            data["archetype_single"] = archetype
            expanded.append(data)
    frame = pd.DataFrame(expanded)
    return (
        frame.groupby("archetype_single", dropna=False)
        .agg(item_count=("asin", "size"), delta_ndcg20_mean=("delta_ndcg@20", "mean"), baseline_ndcg20_mean=("baseline_ndcg@20", "mean"))
        .reset_index()
        .rename(columns={"delta_ndcg20_mean": "delta_ndcg@20_mean", "baseline_ndcg20_mean": "baseline_ndcg@20_mean"})
    )


def decide_archetype_response_route(summary: pd.DataFrame) -> dict[str, Any]:
    actionable = summary[summary["item_count"] >= 50]
    if actionable.empty:
        return {"route": "archetype_response_too_sparse", "reason": "archetype 样本不足。", "evidence": {}}
    spread = float(actionable["delta_ndcg@20_mean"].max() - actionable["delta_ndcg@20_mean"].min())
    if spread > 0.01:
        route = "archetype_response_actionable"
        reason = "不同 archetype 的 method response 明显不同。"
    else:
        route = "archetype_response_uniform"
        reason = "archetype response 基本一致。"
    return {"route": route, "reason": reason, "evidence": {"response_spread": _jsonable(spread), "archetype_count": int(len(actionable))}}


def build_near_cutoff_summary(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    frame["near_cutoff_group"] = frame.get("target_rank_near_cutoff", False).astype(bool).map({True: "near_cutoff", False: "not_near_cutoff"})
    return (
        frame.groupby(["near_cutoff_group", "residual_bucket"], dropna=False)
        .agg(item_count=("asin", "size"), delta_ndcg20_mean=("delta_ndcg@20", "mean"), baseline_ndcg20_mean=("baseline_ndcg@20", "mean"))
        .reset_index()
        .rename(columns={"delta_ndcg20_mean": "delta_ndcg@20_mean", "baseline_ndcg20_mean": "baseline_ndcg@20_mean"})
    )


def decide_near_cutoff_route(summary: pd.DataFrame) -> dict[str, Any]:
    near = summary[summary["near_cutoff_group"].eq("near_cutoff")]
    other = summary[summary["near_cutoff_group"].eq("not_near_cutoff")]
    near_delta = _safe_mean(near["delta_ndcg@20_mean"]) if not near.empty else float("nan")
    other_delta = _safe_mean(other["delta_ndcg@20_mean"]) if not other.empty else float("nan")
    if near_delta > other_delta + 0.001:
        route = "near_cutoff_recovery_supported"
        reason = "near-cutoff hard item 更容易被恢复。"
    elif other_delta > near_delta:
        route = "deep_hard_recovery_needed"
        reason = "non-near-cutoff item response 更强或 near-cutoff 不占优。"
    else:
        route = "near_cutoff_not_recoverable"
        reason = "near-cutoff recovery 信号弱。"
    return {"route": route, "reason": reason, "evidence": {"near_delta": _jsonable(near_delta), "not_near_delta": _jsonable(other_delta)}}


def build_proxy_ensemble_profile(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile.copy()
    margin = _norm01(frame.get("margin_proxy", frame.get("margin_to_top20_cutoff", pd.Series(0, index=frame.index))))
    target_gap = _norm01(frame.get("target_competitor_gap_proxy", pd.Series(0, index=frame.index)))
    residual_easy = frame.get("residual_bucket", pd.Series("", index=frame.index)).astype(str).eq("easy").astype(float)
    conflict_penalty = _norm01(frame.get("modality_conflict_score", pd.Series(0, index=frame.index)))
    coverage = _norm01(frame.get("coverage_score", pd.Series(1, index=frame.index)))
    frame["proxy_ensemble_score"] = margin + target_gap + residual_easy + coverage - conflict_penalty
    frame["proxy_ensemble_group"] = _prefix_bucket(frame["proxy_ensemble_score"], "ensemble")
    return frame


def build_proxy_ensemble_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, sub in frame.groupby("proxy_ensemble_group", dropna=False):
        rows.append({"proxy_ensemble_group": group, "item_count": int(len(sub)), "ndcg@20_mean": _safe_mean(sub["ndcg@20"]), "delta_ndcg@20_mean": _mean_or_nan(sub, "delta_ndcg@20"), "proxy_ensemble_score_mean": _safe_mean(sub["proxy_ensemble_score"])})
    out = pd.DataFrame(rows).sort_values("proxy_ensemble_group").reset_index(drop=True)
    high = out[out["proxy_ensemble_group"].astype(str).str.contains("high")]
    low = out[out["proxy_ensemble_group"].astype(str).str.contains("low")]
    out["ensemble_high_minus_low_ndcg@20"] = (float(high.iloc[0]["ndcg@20_mean"]) - float(low.iloc[0]["ndcg@20_mean"])) if not high.empty and not low.empty else float("nan")
    return out


def decide_proxy_ensemble_route(summary: pd.DataFrame) -> dict[str, Any]:
    gap = float(pd.to_numeric(summary["ensemble_high_minus_low_ndcg@20"], errors="coerce").dropna().iloc[0]) if summary["ensemble_high_minus_low_ndcg@20"].notna().any() else float("nan")
    if gap > 0.10:
        route = "proxy_ensemble_ready"
        reason = "proxy ensemble high-low baseline gap 足够大。"
    elif gap > 0.05:
        route = "single_margin_proxy_preferred"
        reason = "ensemble 有信号但不够强，保留 margin 简化方案。"
    else:
        route = "ensemble_needs_feature_rebuild"
        reason = "ensemble 当前不足以替代单一 margin。"
    return {"route": route, "reason": reason, "evidence": {"ensemble_high_minus_low_ndcg@20": _jsonable(gap)}}


def load_inputs(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    task3_dir = Path(args.task3_dir).expanduser().resolve()
    return {
        "item": _read_csv(task3_dir / "task3_item_profile.csv"),
        "baseline_score": _read_csv(task3_dir / "baseline_score_margin" / "item_score_margin_profile.csv"),
        "category_score": _read_csv(task3_dir / "category_conf_input_score_margin" / "item_score_margin_profile.csv"),
        "baseline_content": _read_csv(task3_dir / "baseline_content_cf_alignment" / "content_cf_alignment_profile.csv"),
        "delta": _read_csv(Path(args.task3p5_dir).expanduser().resolve() / "task3p5_item_delta_profile.csv"),
        "p10": _read_csv(args.task3p10_profile),
        "p17": _read_csv(args.task3p17_profile),
        "p23": _read_csv(args.task3p23_profile),
        "p21": _read_csv(args.task3p21_full_profile),
    }


def build_delta_group_profile(delta: pd.DataFrame, p17: pd.DataFrame, p23: pd.DataFrame, p10: pd.DataFrame, p21: pd.DataFrame | None = None) -> pd.DataFrame:
    columns = ["delta_ndcg@20", "delta_hr@20", "delta_q_norm", "delta_target_score_max", "delta_margin_to_top20_cutoff", "delta_best_target_rank", "baseline_best_target_rank", "baseline_ndcg@20", "category_conf_ndcg@20", "delta_top20_user_norm_mean", "delta_top20_history_interaction_count_mean"]
    frame = delta[["asin", *[col for col in columns if col in delta.columns]]].copy()
    frame = _merge_selected(frame, p17, ["rank_aware_group", "competition_aware_group", "margin_proxy", "target_competitor_gap_proxy", "ndcg@20", "margin_to_top20_cutoff"])
    frame = _merge_selected(frame, p23, ["consensus_group", "consensus_score", "archetype", "modality_alignment_proxy"])
    frame = _merge_selected(frame, p10, ["residual_bucket", "target_rank_near_cutoff", "target_history_interaction_count_mean", "top20_history_interaction_count_mean", "target_activity_bucket", "target_norm_bucket", "margin_bucket"])
    if p21 is not None and not p21.empty:
        dedup = p21[p21["checkpoint_label"].eq("baseline_best")].copy()
        frame = _merge_selected(frame, dedup, ["target_history_bucket", "target_norm_bucket"])
    return frame


def run(args: argparse.Namespace) -> None:
    inputs = load_inputs(args)
    output_root = Path(args.output_root).expanduser().resolve()
    result_root_display = args.result_root_display.rstrip("/")
    route_decisions: dict[str, dict[str, Any]] = {}
    output_dirs: dict[str, str] = {}

    p17 = inputs["p17"].copy()
    p23 = inputs["p23"].copy()
    p10 = inputs["p10"].copy()
    delta = inputs["delta"].copy()
    p21 = inputs["p21"].copy()
    delta_group = build_delta_group_profile(delta, p17, p23, p10, p21)

    t24_summary = build_threshold_sensitivity_summary(p17, metric="margin_proxy")
    t24_decision = decide_threshold_sensitivity_route(t24_summary)
    route_decisions["Task3.24"] = t24_decision
    output_dirs["Task3.24"] = str(write_task_output(output_root, result_root_display, "24", "threshold-sensitivity", "threshold sensitivity", t24_decision, {"threshold_sensitivity_profile": p17, "threshold_sensitivity_summary": t24_summary}))

    t25_profile = build_cross_stability_profile(p17, p10)
    t25_summary = build_cross_stability_summary(t25_profile)
    t25_decision = decide_cross_stability_route(t25_summary)
    route_decisions["Task3.25"] = t25_decision
    output_dirs["Task3.25"] = str(write_task_output(output_root, result_root_display, "25", "rank-residual-cross", "rank residual cross", t25_decision, {"cross_stability_profile": t25_profile, "cross_stability_summary": t25_summary}))

    t26_summary = build_consensus_component_ablation(p23)
    t26_decision = decide_consensus_component_route(t26_summary)
    route_decisions["Task3.26"] = t26_decision
    output_dirs["Task3.26"] = str(write_task_output(output_root, result_root_display, "26", "consensus-component-ablation", "consensus component ablation", t26_decision, {"component_ablation_summary": t26_summary}))

    t27_summary = build_tradeoff_summary(delta_group)
    t27_decision = decide_tradeoff_route(t27_summary)
    route_decisions["Task3.27"] = t27_decision
    output_dirs["Task3.27"] = str(write_task_output(output_root, result_root_display, "27", "gain-loss-tradeoff", "gain loss tradeoff", t27_decision, {"tradeoff_profile": delta_group, "tradeoff_summary": t27_summary}))

    t28_summary = build_norm_shift_summary(delta)
    t28_decision = decide_norm_shift_route(t28_summary)
    route_decisions["Task3.28"] = t28_decision
    output_dirs["Task3.28"] = str(write_task_output(output_root, result_root_display, "28", "norm-shift-risk", "norm shift risk", t28_decision, {"norm_shift_profile": delta, "norm_shift_summary": t28_summary}))

    t29_summary = build_lift_decomposition_summary(delta)
    t29_decision = decide_lift_decomposition_route(t29_summary)
    route_decisions["Task3.29"] = t29_decision
    output_dirs["Task3.29"] = str(write_task_output(output_root, result_root_display, "29", "lift-decomposition", "lift decomposition", t29_decision, {"lift_decomposition_profile": delta, "lift_decomposition_summary": t29_summary}))

    t30_summary = build_rank_jump_summary(delta)
    t30_decision = decide_rank_jump_route(t30_summary)
    route_decisions["Task3.30"] = t30_decision
    output_dirs["Task3.30"] = str(write_task_output(output_root, result_root_display, "30", "rank-jump-recovery", "rank jump recovery", t30_decision, {"rank_jump_profile": delta, "rank_jump_summary": t30_summary}))

    t31_summary = build_cutoff_fragility_summary(inputs["baseline_score"], delta)
    t31_decision = decide_cutoff_fragility_route(t31_summary)
    route_decisions["Task3.31"] = t31_decision
    output_dirs["Task3.31"] = str(write_task_output(output_root, result_root_display, "31", "cutoff-fragility", "cutoff fragility", t31_decision, {"cutoff_fragility_summary": t31_summary}))

    t32_summary = build_rank_band_response_summary(delta)
    t32_decision = decide_rank_band_response_route(t32_summary)
    route_decisions["Task3.32"] = t32_decision
    output_dirs["Task3.32"] = str(write_task_output(output_root, result_root_display, "32", "rank-band-response", "rank band response", t32_decision, {"rank_band_response_profile": delta, "rank_band_response_summary": t32_summary}))

    t33_profile = p10.merge(p17[["asin", "rank_aware_group"]].drop_duplicates("asin"), on="asin", how="left")
    t33_summary = build_cold_start_summary(t33_profile)
    t33_decision = decide_cold_start_route(t33_summary)
    route_decisions["Task3.33"] = t33_decision
    output_dirs["Task3.33"] = str(write_task_output(output_root, result_root_display, "33", "cold-start-stratification", "cold start stratification", t33_decision, {"cold_start_profile": t33_profile, "cold_start_summary": t33_summary}))

    t34_profile = p10.merge(p17[["asin", "rank_aware_group"]].drop_duplicates("asin"), on="asin", how="left")
    t34_summary = build_competitor_activity_conditional_summary(t34_profile)
    t34_decision = decide_competitor_activity_route(t34_summary)
    route_decisions["Task3.34"] = t34_decision
    output_dirs["Task3.34"] = str(write_task_output(output_root, result_root_display, "34", "competitor-activity-conditional", "competitor activity conditional", t34_decision, {"competitor_activity_profile": t34_profile, "competitor_activity_summary": t34_summary}))

    t35_profile = build_modality_conflict_profile(p23)
    t35_summary = build_modality_conflict_summary(t35_profile)
    t35_decision = decide_modality_conflict_route(t35_summary)
    route_decisions["Task3.35"] = t35_decision
    output_dirs["Task3.35"] = str(write_task_output(output_root, result_root_display, "35", "modality-conflict", "modality conflict", t35_decision, {"modality_conflict_profile": t35_profile, "modality_conflict_summary": t35_summary}))

    t36_profile, t36_summary = build_coverage_summary(inputs["baseline_content"])
    t36_decision = decide_coverage_route(t36_summary)
    route_decisions["Task3.36"] = t36_decision
    output_dirs["Task3.36"] = str(write_task_output(output_root, result_root_display, "36", "coverage-risk", "coverage risk", t36_decision, {"coverage_profile": t36_profile, "coverage_summary": t36_summary}))

    t37_profile = p17.copy()
    t37_summary = build_tail_risk_summary(t37_profile)
    t37_decision = decide_tail_risk_route(t37_summary)
    route_decisions["Task3.37"] = t37_decision
    output_dirs["Task3.37"] = str(write_task_output(output_root, result_root_display, "37", "tail-risk", "tail risk", t37_decision, {"tail_risk_profile": t37_profile, "tail_risk_summary": t37_summary}))

    t38_profile = inputs["item"].merge(p17[["asin", "ndcg@20", "rank_aware_group"]].drop_duplicates("asin"), on="asin", how="left")
    t38_profile, t38_summary = build_v2_disagreement_summary(t38_profile)
    t38_decision = decide_v2_disagreement_route(t38_summary)
    route_decisions["Task3.38"] = t38_decision
    output_dirs["Task3.38"] = str(write_task_output(output_root, result_root_display, "38", "v2-disagreement", "v2 disagreement", t38_decision, {"v2_disagreement_profile": t38_profile, "v2_disagreement_summary": t38_summary}))

    t39_profile = delta_group.copy()
    t39_summary = build_pareto_summary(t39_profile)
    t39_decision = decide_pareto_route(t39_summary)
    route_decisions["Task3.39"] = t39_decision
    output_dirs["Task3.39"] = str(write_task_output(output_root, result_root_display, "39", "pareto-group-map", "pareto group map", t39_decision, {"pareto_profile": t39_profile, "pareto_summary": t39_summary}))

    stability_frames = []
    for group_col in ["rank_aware_group", "consensus_group", "residual_bucket", "target_history_bucket", "target_norm_bucket", "archetype"]:
        if group_col in p21.columns:
            stability_frames.append(build_checkpoint_group_stability(p21, group_col))
    t40_summary = pd.concat(stability_frames, ignore_index=True)
    t40_decision = decide_checkpoint_group_stability_route(t40_summary)
    route_decisions["Task3.40"] = t40_decision
    output_dirs["Task3.40"] = str(write_task_output(output_root, result_root_display, "40", "checkpoint-new-groups", "checkpoint new groups", t40_decision, {"checkpoint_new_group_profile": p21, "checkpoint_new_group_summary": t40_summary}))

    t41_profile = delta_group.copy()
    t41_summary = build_archetype_response_summary(t41_profile)
    t41_decision = decide_archetype_response_route(t41_summary)
    route_decisions["Task3.41"] = t41_decision
    output_dirs["Task3.41"] = str(write_task_output(output_root, result_root_display, "41", "archetype-response", "archetype response", t41_decision, {"archetype_response_profile": t41_profile, "archetype_response_summary": t41_summary}))

    t42_profile = delta_group.copy()
    t42_summary = build_near_cutoff_summary(t42_profile)
    t42_decision = decide_near_cutoff_route(t42_summary)
    route_decisions["Task3.42"] = t42_decision
    output_dirs["Task3.42"] = str(write_task_output(output_root, result_root_display, "42", "near-cutoff-recovery", "near cutoff recovery", t42_decision, {"near_cutoff_profile": t42_profile, "near_cutoff_summary": t42_summary}))

    t43_seed = p23.copy()
    t43_seed = _merge_selected(t43_seed, delta, ["delta_ndcg@20"])
    t43_seed = _merge_selected(t43_seed, t35_profile, ["modality_conflict_score"])
    t43_seed = _merge_selected(t43_seed, t36_profile, ["coverage_score"])
    t43_profile = build_proxy_ensemble_profile(t43_seed)
    t43_summary = build_proxy_ensemble_summary(t43_profile)
    t43_decision = decide_proxy_ensemble_route(t43_summary)
    route_decisions["Task3.43"] = t43_decision
    output_dirs["Task3.43"] = str(write_task_output(output_root, result_root_display, "43", "proxy-ensemble", "proxy ensemble", t43_decision, {"proxy_ensemble_profile": t43_profile, "proxy_ensemble_summary": t43_summary}))

    print(json.dumps({"route_decisions": route_decisions, "output_dirs": output_dirs}, ensure_ascii=False, indent=2))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run CCFCRec Amazon-VG category availability v2 Task3.24-3.43 diagnostics")
    parser.add_argument("--task3-dir", required=True)
    parser.add_argument("--task3p5-dir", required=True)
    parser.add_argument("--task3p10-profile", required=True)
    parser.add_argument("--task3p17-profile", required=True)
    parser.add_argument("--task3p23-profile", required=True)
    parser.add_argument("--task3p21-full-profile", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--result-root-display", required=True)
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
