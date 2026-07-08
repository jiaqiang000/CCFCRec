#!/usr/bin/env python3
"""
CCFCRec Amazon-VG M10-R1 recoverability upper-bound audit.

Diagnostic-only offline audit. It estimates whether recoverability-style hard
item masks have enough item-level NDCG@20 headroom to justify continuing the
overall 3% search.
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
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "temp_202607_实验文件记录" / "temp_20260709"
DEFAULT_TASK4_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260706"
    / "2026-07-06 004222 task4-pre3-train-safe-hard-proxy"
    / "task4_train_safe_hard_proxy_profile.csv"
)
DEFAULT_R0_DELTA_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260709"
    / "2026-07-09 012326 task4-rollback-m10-r0-m9-post-audit"
    / "m10_r0_m9_item_delta_profile.csv"
)
DEFAULT_NEAR_CUTOFF_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260705"
    / "2026-07-05 021842 category-availability-v2-task3p42-near-cutoff-recovery"
    / "task3p42_near_cutoff_profile.csv"
)
DEFAULT_RECOVERABILITY_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260705"
    / "2026-07-05 021842 category-availability-v2-task3p43-proxy-ensemble"
    / "task3p43_proxy_ensemble_profile.csv"
)
DEFAULT_FAILURE_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260705"
    / "2026-07-05 021000 category-availability-v2-task3p23-consensus-signal"
    / "task3p23_consensus_signal_profile.csv"
)
DEFAULT_RANK_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260705"
    / "2026-07-05 020500 category-availability-v2-task3p17-alt-availability"
    / "task3p17_alt_availability_profile.csv"
)
DESIGN_NOTE_NAME = "2026-07-09 012824 CCFCRec Amazon-VG M10-R1 recoverability upper-bound audit 诊断设计"
TOTAL_DESIGN_NOTE_NAME = "2026-07-09 010147 CCFCRec Amazon-VG Task4-rollback M10-R recoverability and carrier audit 总设计"
R0_RESULT_NOTE_NAME = "2026-07-09 012326 CCFCRec Amazon-VG M10-R0 M9 post audit 结果"
ANALYSIS_SCRIPT = "validata/analyze_amazon_vg_task4_rollback_m10_r1_recoverability_upper_bound.py"

DEFAULT_CANDIDATE_COLUMNS = [
    "recoverability_ensemble_high_hard",
    "rank_recoverability_high_hard",
    "near_cutoff_hard",
    "failure_consensus_high",
    "acat_v3_high_hard",
    "rsp_high_hard",
    "high_acat_train_safe_hard_flag",
    "recoverability_ensemble_high_all",
    "rank_recoverability_high_all",
]

RECOVERABILITY_CANDIDATES = {
    "recoverability_ensemble_high_hard",
    "rank_recoverability_high_hard",
    "near_cutoff_hard",
    "failure_consensus_high",
}
CONTROL_CANDIDATES = {
    "acat_v3_high_hard",
    "rsp_high_hard",
}
OPTIONAL_CANDIDATES = {
    "high_acat_train_safe_hard_flag",
    "recoverability_ensemble_high_all",
    "rank_recoverability_high_all",
}


@dataclass(frozen=True)
class Outputs:
    output_dir: Path
    upper_bound_profile_csv: Path
    upper_bound_summary_csv: Path
    placebo_summary_csv: Path
    rsp_acat_control_summary_csv: Path
    route_decision_json: Path
    manifest_json: Path
    result_md: Path


def now_stamp() -> tuple[str, str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H%M%S"), now.strftime("%Y-%m-%d"), now.isoformat(timespec="seconds")


def _load_profile(profile: pd.DataFrame | Path | str | None) -> pd.DataFrame:
    if profile is None:
        return pd.DataFrame()
    if isinstance(profile, pd.DataFrame):
        return profile.copy()
    path = Path(profile)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"raw_asin": str, "asin": str}, low_memory=False)


def _normalize_asin_key(profile: pd.DataFrame) -> pd.DataFrame:
    out = profile.copy()
    if "raw_asin" not in out.columns and "asin" in out.columns:
        out = out.rename(columns={"asin": "raw_asin"})
    if "raw_asin" in out.columns:
        out["raw_asin"] = out["raw_asin"].astype(str)
    return out


def _merge_profile(
    base: pd.DataFrame,
    profile: pd.DataFrame | Path | str | None,
    keep_cols: list[str],
    rename: dict[str, str] | None = None,
) -> pd.DataFrame:
    prof = _normalize_asin_key(_load_profile(profile))
    if prof.empty or "raw_asin" not in prof.columns:
        return base
    rename = rename or {}
    selected = ["raw_asin", *[col for col in keep_cols if col in prof.columns]]
    prof = prof[selected].rename(columns=rename).drop_duplicates("raw_asin")
    existing = set(base.columns) - {"raw_asin"}
    drop_cols = [col for col in prof.columns if col in existing]
    if drop_cols:
        prof = prof.drop(columns=drop_cols)
    return base.merge(prof, on="raw_asin", how="left", validate="many_to_one")


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.map(bool)
    return series.map(lambda value: str(value).strip().lower() in {"true", "1", "yes", "y"})


def _mask(series: pd.Series) -> pd.Series:
    return _to_bool(series).map(bool).astype(object)


def _safe_float(value) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def _safe_mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


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


def single_hit_ndcg_floor(k: int = 20) -> float:
    return 1.0 / math.log2(k + 1)


def candidate_type(candidate: str) -> str:
    if candidate in RECOVERABILITY_CANDIDATES:
        return "recoverability"
    if candidate == "acat_v3_high_hard":
        return "acat_control"
    if candidate == "rsp_high_hard":
        return "rsp_control"
    if candidate in OPTIONAL_CANDIDATES:
        return "optional_report"
    return "other"


def build_upper_bound_profile(
    task4_profile: pd.DataFrame | Path | str,
    near_cutoff_profile: pd.DataFrame | Path | str | None,
    recoverability_profile: pd.DataFrame | Path | str | None,
    failure_consensus_profile: pd.DataFrame | Path | str | None,
    rank_recoverability_profile: pd.DataFrame | Path | str | None,
    r0_delta_profile: pd.DataFrame | Path | str | None = None,
) -> pd.DataFrame:
    task4 = _normalize_asin_key(_load_profile(task4_profile))
    required = {"raw_asin", "split", "baseline_ndcg@20", "eval_baseline_hard_flag"}
    missing = required - set(task4.columns)
    if missing:
        raise ValueError(f"task4 profile missing columns: {sorted(missing)}")
    profile = task4[task4["split"].astype(str).eq("test")].copy()
    keep_cols = [
        "raw_asin",
        "split",
        "category_count",
        "cat_count_bin",
        "s_cat_v3",
        "s_cat_v3_group",
        "high_acat_flag",
        "RSP_group",
        "baseline_ndcg@20",
        "baseline_margin_proxy",
        "baseline_best_target_rank",
        "eval_baseline_hard_flag",
        "high_acat_train_safe_hard_flag",
        "train_safe_hard_proxy_group",
    ]
    profile = profile[[col for col in keep_cols if col in profile.columns]].copy()
    profile["raw_asin"] = profile["raw_asin"].astype(str)
    profile["baseline_ndcg@20"] = pd.to_numeric(profile["baseline_ndcg@20"], errors="coerce")
    profile["eval_baseline_hard_flag"] = _mask(profile["eval_baseline_hard_flag"])
    if "high_acat_flag" in profile.columns:
        profile["high_acat_flag"] = _mask(profile["high_acat_flag"])
    else:
        profile["high_acat_flag"] = profile.get("s_cat_v3_group", "").astype(str).eq("s_cat_v3_strong").map(bool).astype(object)
    if "high_acat_train_safe_hard_flag" in profile.columns:
        profile["high_acat_train_safe_hard_flag"] = _mask(profile["high_acat_train_safe_hard_flag"])

    profile = _merge_profile(profile, near_cutoff_profile, ["target_rank_near_cutoff", "target_activity_bucket"])
    profile = _merge_profile(
        profile,
        recoverability_profile,
        ["proxy_ensemble_group", "proxy_ensemble_score"],
        rename={
            "proxy_ensemble_group": "recoverability_proxy_ensemble_group",
            "proxy_ensemble_score": "recoverability_proxy_ensemble_score",
        },
    )
    profile = _merge_profile(
        profile,
        failure_consensus_profile,
        ["consensus_group", "consensus_score"],
        rename={"consensus_group": "failure_consensus_group", "consensus_score": "failure_consensus_score"},
    )
    profile = _merge_profile(
        profile,
        rank_recoverability_profile,
        ["rank_aware_group", "competition_aware_group"],
        rename={
            "rank_aware_group": "rank_recoverability_group",
            "competition_aware_group": "competition_recoverability_group",
        },
    )
    if r0_delta_profile is not None:
        r0 = _normalize_asin_key(_load_profile(r0_delta_profile))
        if not r0.empty:
            r0 = r0[r0.get("split", "test").astype(str).eq("test")].copy() if "split" in r0.columns else r0
            r0_keep = ["raw_asin", "alpha", "delta_ndcg@20", "m9_helped_flag", "m9_harmed_flag"]
            r0 = r0[[col for col in r0_keep if col in r0.columns]].copy()
            if "alpha" in r0.columns:
                r0 = r0.sort_values("alpha").drop_duplicates("raw_asin")
            profile = profile.merge(r0, on="raw_asin", how="left", validate="one_to_one", suffixes=("", "_m9"))

    hard = _to_bool(profile["eval_baseline_hard_flag"])
    profile["recoverability_ensemble_high_all"] = (
        profile.get("recoverability_proxy_ensemble_group", pd.Series("", index=profile.index)).astype(str).eq("ensemble_high")
    ).map(bool).astype(object)
    profile["recoverability_ensemble_high_hard"] = (profile["recoverability_ensemble_high_all"].map(bool) & hard).map(bool).astype(object)
    profile["rank_recoverability_high_all"] = (
        profile.get("rank_recoverability_group", pd.Series("", index=profile.index)).astype(str).eq("rank_high")
    ).map(bool).astype(object)
    profile["rank_recoverability_high_hard"] = (profile["rank_recoverability_high_all"].map(bool) & hard).map(bool).astype(object)
    profile["near_cutoff_hard"] = (
        _to_bool(profile.get("target_rank_near_cutoff", pd.Series(False, index=profile.index))) & hard
    ).map(bool).astype(object)
    profile["failure_consensus_high"] = (
        profile.get("failure_consensus_group", pd.Series("", index=profile.index)).astype(str).eq("consensus_high")
    ).map(bool).astype(object)
    profile["acat_v3_high_hard"] = (profile["high_acat_flag"].map(bool) & hard).map(bool).astype(object)
    profile["rsp_high_hard"] = (
        profile.get("RSP_group", pd.Series("", index=profile.index)).astype(str).eq("RSP_high") & hard
    ).map(bool).astype(object)
    return profile.reset_index(drop=True)


def _candidate_upper_bound_row(profile: pd.DataFrame, candidate: str, floor: float) -> dict:
    mask = _to_bool(profile[candidate]) if candidate in profile.columns else pd.Series(False, index=profile.index)
    baseline = pd.to_numeric(profile["baseline_ndcg@20"], errors="coerce").fillna(0.0)
    single_hit_gain = (baseline.map(lambda value: max(value, floor)) - baseline).clip(lower=0.0)
    perfect_gain = (1.0 - baseline).clip(lower=0.0)
    selected = mask.astype(float)
    baseline_mean = float(baseline.mean())
    three_pct_abs_gate = baseline_mean * 0.03
    single_gain = float((single_hit_gain * selected).mean())
    perfect_gain_abs = float((perfect_gain * selected).mean())
    row = {
        "candidate": candidate,
        "candidate_type": candidate_type(candidate),
        "item_count": int(len(profile)),
        "selected_count": int(mask.sum()),
        "selected_share": float(mask.mean()) if len(mask) else 0.0,
        "baseline_item_ndcg_mean": baseline_mean,
        "selected_baseline_ndcg@20_mean": _safe_mean(baseline[mask]),
        "three_pct_abs_gate": three_pct_abs_gate,
        "single_hit_ndcg_floor": floor,
        "single_hit_upper_gain_abs": single_gain,
        "single_hit_upper_gain_pct_vs_item_baseline": single_gain / baseline_mean * 100 if baseline_mean else float("nan"),
        "perfect_recovery_upper_gain_abs": perfect_gain_abs,
        "perfect_recovery_upper_gain_pct_vs_item_baseline": perfect_gain_abs / baseline_mean * 100 if baseline_mean else float("nan"),
        "passes_single_hit_3pct_gate": bool(single_gain >= three_pct_abs_gate),
        "passes_perfect_recovery_3pct_gate": bool(perfect_gain_abs >= three_pct_abs_gate),
    }
    for col in ["passes_single_hit_3pct_gate", "passes_perfect_recovery_3pct_gate"]:
        row[col] = bool(row[col])
    return row


def build_upper_bound_summary(
    profile: pd.DataFrame,
    candidate_columns: list[str] | None = None,
    k: int = 20,
) -> pd.DataFrame:
    candidate_columns = DEFAULT_CANDIDATE_COLUMNS if candidate_columns is None else candidate_columns
    floor = single_hit_ndcg_floor(k)
    rows = [
        _candidate_upper_bound_row(profile, candidate, floor)
        for candidate in candidate_columns
        if candidate in profile.columns
    ]
    summary = pd.DataFrame(rows)
    for col in ["passes_single_hit_3pct_gate", "passes_perfect_recovery_3pct_gate"]:
        if col in summary:
            summary[col] = summary[col].map(bool).astype(object)
    return summary


def build_placebo_summary(
    profile: pd.DataFrame,
    candidate_columns: list[str] | None = None,
    shuffle_count: int = 100,
    random_seed: int = 43,
    k: int = 20,
) -> pd.DataFrame:
    candidate_columns = DEFAULT_CANDIDATE_COLUMNS if candidate_columns is None else candidate_columns
    rng = np.random.default_rng(random_seed)
    floor = single_hit_ndcg_floor(k)
    rows = []
    for candidate in candidate_columns:
        if candidate not in profile.columns:
            continue
        selected_count = int(_to_bool(profile[candidate]).sum())
        gains_single = []
        gains_perfect = []
        for _ in range(shuffle_count):
            shuffled_mask = np.zeros(len(profile), dtype=bool)
            if selected_count > 0:
                chosen = rng.choice(len(profile), size=selected_count, replace=False)
                shuffled_mask[chosen] = True
            temp = profile[["raw_asin", "baseline_ndcg@20"]].copy() if "raw_asin" in profile.columns else profile[["baseline_ndcg@20"]].copy()
            temp[candidate] = shuffled_mask
            row = _candidate_upper_bound_row(temp, candidate, floor)
            gains_single.append(row["single_hit_upper_gain_abs"])
            gains_perfect.append(row["perfect_recovery_upper_gain_abs"])
        single_arr = np.asarray(gains_single, dtype=float)
        perfect_arr = np.asarray(gains_perfect, dtype=float)
        rows.append(
            {
                "candidate": candidate,
                "selected_count": selected_count,
                "shuffle_count": int(shuffle_count),
                "single_hit_upper_gain_abs_mean": float(np.nanmean(single_arr)) if single_arr.size else float("nan"),
                "single_hit_upper_gain_abs_p05": float(np.nanpercentile(single_arr, 5)) if single_arr.size else float("nan"),
                "single_hit_upper_gain_abs_p95": float(np.nanpercentile(single_arr, 95)) if single_arr.size else float("nan"),
                "single_hit_upper_gain_abs_max": float(np.nanmax(single_arr)) if single_arr.size else float("nan"),
                "perfect_recovery_upper_gain_abs_mean": float(np.nanmean(perfect_arr)) if perfect_arr.size else float("nan"),
                "perfect_recovery_upper_gain_abs_p05": float(np.nanpercentile(perfect_arr, 5)) if perfect_arr.size else float("nan"),
                "perfect_recovery_upper_gain_abs_p95": float(np.nanpercentile(perfect_arr, 95)) if perfect_arr.size else float("nan"),
                "perfect_recovery_upper_gain_abs_max": float(np.nanmax(perfect_arr)) if perfect_arr.size else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_rsp_acat_control_summary(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    return summary[summary["candidate_type"].isin(["acat_control", "rsp_control"])].reset_index(drop=True)


def _placebo_value(placebo: pd.DataFrame, candidate: str, column: str) -> float:
    row = placebo[placebo["candidate"].eq(candidate)]
    if row.empty or column not in row.columns:
        return float("nan")
    return _safe_float(row.iloc[0][column])


def _best_row(frame: pd.DataFrame, metric: str) -> pd.Series | None:
    if frame.empty or metric not in frame.columns:
        return None
    return frame.sort_values(metric, ascending=False).iloc[0]


def build_route_decision(summary: pd.DataFrame, placebo: pd.DataFrame) -> dict:
    if summary.empty:
        return {
            "route": "r1_inconclusive_missing_inputs",
            "next_action": "rebuild_r1_inputs",
            "continue_overall_3pct_search": False,
            "reason": "upper-bound summary is empty",
        }
    real = summary[summary["candidate_type"].eq("recoverability")].copy()
    controls = summary[summary["candidate_type"].isin(["acat_control", "rsp_control"])].copy()
    if real.empty:
        return {
            "route": "r1_inconclusive_missing_inputs",
            "next_action": "rebuild_recoverability_candidate_masks",
            "continue_overall_3pct_search": False,
            "reason": "recoverability candidate rows are missing",
        }
    real["single_hit_shuffle_p95"] = real["candidate"].map(
        lambda candidate: _placebo_value(placebo, candidate, "single_hit_upper_gain_abs_p95")
    )
    real["perfect_shuffle_p95"] = real["candidate"].map(
        lambda candidate: _placebo_value(placebo, candidate, "perfect_recovery_upper_gain_abs_p95")
    )
    real["beats_single_hit_shuffle_p95"] = real.apply(
        lambda row: pd.notna(row["single_hit_shuffle_p95"])
        and float(row["single_hit_upper_gain_abs"]) > float(row["single_hit_shuffle_p95"]),
        axis=1,
    )
    best_real_single = _best_row(real, "single_hit_upper_gain_abs")
    best_real_perfect = _best_row(real, "perfect_recovery_upper_gain_abs")
    best_control_single = _safe_float(controls["single_hit_upper_gain_abs"].max()) if not controls.empty else float("nan")
    best_control_candidate = ""
    if not controls.empty:
        best_control_candidate = str(controls.sort_values("single_hit_upper_gain_abs", ascending=False).iloc[0]["candidate"])

    recoverability_beats_controls = bool(
        best_real_single is not None
        and (pd.isna(best_control_single) or best_real_single["single_hit_upper_gain_abs"] > best_control_single)
    )
    control_dominance_warning = bool(best_real_single is not None and pd.notna(best_control_single) and not recoverability_beats_controls)
    single_pass = False
    if best_real_single is not None:
        single_pass = bool(
            best_real_single["single_hit_upper_gain_abs"] >= best_real_single["three_pct_abs_gate"]
            and bool(best_real_single["beats_single_hit_shuffle_p95"])
        )
    if single_pass:
        route = "recoverability_single_hit_space_exists"
        next_action = (
            "run_r2_train_safe_proxy_rebuild_with_rsp_acat_control_warning"
            if control_dominance_warning
            else "run_r2_train_safe_proxy_rebuild_before_new_training"
        )
        continue_search = True
        reason = (
            "recoverability candidate single-hit upper bound exceeds 3% gate and shuffle p95; controls are larger"
            if control_dominance_warning
            else "recoverability candidate single-hit upper bound exceeds 3% gate, shuffle p95, and controls"
        )
    elif best_real_single is not None and pd.notna(best_control_single) and best_real_single["single_hit_upper_gain_abs"] <= best_control_single:
        route = "recoverability_not_better_than_controls"
        next_action = "do_not_use_recoverability_as_main_method_entry_without_new_evidence"
        continue_search = False
        reason = "best recoverability single-hit upper bound does not beat RSP/Acat controls"
    elif best_real_perfect is not None and best_real_perfect["perfect_recovery_upper_gain_abs"] >= best_real_perfect["three_pct_abs_gate"]:
        route = "only_perfect_recovery_space_exists"
        next_action = "do_not_train_yet_reframe_to_mechanism_or_subgroup"
        continue_search = False
        reason = "only perfect-recovery upper bound clears the 3% gate"
    else:
        route = "no_recoverability_overall_3pct_space"
        next_action = "stop_overall_3pct_search_and_reframe"
        continue_search = False
        reason = "recoverability upper bounds do not clear the 3% gate"

    return {
        "route": route,
        "next_action": next_action,
        "continue_overall_3pct_search": continue_search,
        "best_real_single_candidate": "" if best_real_single is None else str(best_real_single["candidate"]),
        "best_real_single_hit_upper_gain_abs": None if best_real_single is None else _safe_float(best_real_single["single_hit_upper_gain_abs"]),
        "best_real_single_hit_shuffle_p95": None if best_real_single is None else _safe_float(best_real_single["single_hit_shuffle_p95"]),
        "best_control_single_candidate": best_control_candidate,
        "best_control_single_hit_upper_gain_abs": None if pd.isna(best_control_single) else _safe_float(best_control_single),
        "recoverability_beats_controls": recoverability_beats_controls,
        "control_dominance_warning": control_dominance_warning,
        "best_real_perfect_candidate": "" if best_real_perfect is None else str(best_real_perfect["candidate"]),
        "best_real_perfect_recovery_upper_gain_abs": None if best_real_perfect is None else _safe_float(best_real_perfect["perfect_recovery_upper_gain_abs"]),
        "three_pct_abs_gate": None if best_real_single is None else _safe_float(best_real_single["three_pct_abs_gate"]),
        "reason": reason,
    }


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    profile: pd.DataFrame,
    summary: pd.DataFrame,
    placebo: pd.DataFrame,
    controls: pd.DataFrame,
    decision: dict,
    manifest_name: str,
) -> None:
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG M10-R1 recoverability upper-bound audit 结果
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - M10-R1
  - recoverability
  - upper_bound
---

# {run_stamp} CCFCRec Amazon-VG M10-R1 recoverability upper-bound audit 结果

## Material Passport

- artifact_type: experiment_diagnostic_result
- project: CCFCRec Amazon-VG category availability
- stage: M10-R1 recoverability upper-bound audit
- status: analyzed
- execution_policy: diagnostic-only（仅诊断），no training（不训练）

> [!info] 来源说明
> 上游总设计：[[{TOTAL_DESIGN_NOTE_NAME}]]
> 上游 R1 设计：[[{DESIGN_NOTE_NAME}]]
> R0 结果：[[{R0_RESULT_NOTE_NAME}]]
> 分析脚本：`{ANALYSIS_SCRIPT}`
> manifest（运行清单）：`{manifest_name}`

## 结论

```text
route = {decision["route"]}
next_action = {decision["next_action"]}
continue_overall_3pct_search = {decision["continue_overall_3pct_search"]}
```

解释：本轮只做 recoverability upper-bound（可恢复性上界）诊断，不训练模型，不把 recoverability（可恢复性代理）写成 Acat_v3（第三版类别可用性）。

## Upper Bound Summary

> [!info] 字段说明
> `single_hit_upper_gain_abs`：选中 item（物品）至少有一个目标用户进入 Top20（前20）时的整体 NDCG@20 绝对增益上界。
> `perfect_recovery_upper_gain_abs`：选中 item 完全恢复到 NDCG@20 = 1 时的整体绝对增益上界。
> `three_pct_abs_gate`：test item-level baseline mean（测试逐物品基线均值）的 3% 绝对门槛。

{md_table(summary, ["candidate", "candidate_type", "selected_count", "selected_share", "baseline_item_ndcg_mean", "single_hit_upper_gain_abs", "single_hit_upper_gain_pct_vs_item_baseline", "perfect_recovery_upper_gain_abs", "perfect_recovery_upper_gain_pct_vs_item_baseline", "three_pct_abs_gate"], max_rows=20)}

## Placebo Summary

> [!info] 字段说明
> `single_hit_upper_gain_abs_p95`：等规模 shuffle（打乱负控）的 single-hit upper bound（单次命中上界）95 分位。
> `perfect_recovery_upper_gain_abs_p95`：等规模 shuffle（打乱负控）的 perfect-recovery upper bound（完全恢复上界）95 分位。

{md_table(placebo, ["candidate", "selected_count", "shuffle_count", "single_hit_upper_gain_abs_mean", "single_hit_upper_gain_abs_p95", "perfect_recovery_upper_gain_abs_mean", "perfect_recovery_upper_gain_abs_p95"], max_rows=20)}

## RSP / Acat Controls

> [!info] 字段说明
> `acat_v3_high_hard`：高 Acat_v3（第三版类别可用性）且 baseline hard（基线困难）。
> `rsp_high_hard`：RSP（丰富度/支持度/流行度）高且 baseline hard（基线困难）。

{md_table(controls, ["candidate", "candidate_type", "selected_count", "single_hit_upper_gain_abs", "perfect_recovery_upper_gain_abs"], max_rows=10)}

## Route Decision

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```

## Profile Size

```text
profile_rows = {len(profile)}
```
"""
    output_path.write_text(markdown, encoding="utf-8")


def build_outputs(output_root: Path, run_stamp: str) -> Outputs:
    output_dir = output_root / f"{run_stamp} task4-rollback-m10-r1-recoverability-upper-bound-audit"
    return Outputs(
        output_dir=output_dir,
        upper_bound_profile_csv=output_dir / "m10_r1_upper_bound_profile.csv",
        upper_bound_summary_csv=output_dir / "m10_r1_upper_bound_summary.csv",
        placebo_summary_csv=output_dir / "m10_r1_placebo_summary.csv",
        rsp_acat_control_summary_csv=output_dir / "m10_r1_rsp_acat_control_summary.csv",
        route_decision_json=output_dir / "m10_r1_route_decision.json",
        manifest_json=output_dir / "run_manifest.json",
        result_md=output_dir / f"{run_stamp} CCFCRec Amazon-VG M10-R1 recoverability upper-bound audit 结果.md",
    )


def run(args: argparse.Namespace) -> Outputs:
    run_stamp, run_date, run_iso = (args.run_stamp, args.run_stamp[:10], "") if args.run_stamp else now_stamp()
    if args.run_stamp:
        run_iso = datetime.strptime(args.run_stamp, "%Y-%m-%d %H%M%S").isoformat(timespec="seconds")
    outputs = build_outputs(Path(args.output_root).expanduser().resolve(), run_stamp)
    outputs.output_dir.mkdir(parents=True, exist_ok=True)

    profile = build_upper_bound_profile(
        task4_profile=args.task4_profile_path,
        near_cutoff_profile=args.near_cutoff_profile_path,
        recoverability_profile=args.recoverability_profile_path,
        failure_consensus_profile=args.failure_consensus_profile_path,
        rank_recoverability_profile=args.rank_recoverability_profile_path,
        r0_delta_profile=args.r0_delta_profile_path,
    )
    summary = build_upper_bound_summary(profile)
    placebo = build_placebo_summary(profile, shuffle_count=args.shuffle_count, random_seed=args.random_seed)
    controls = build_rsp_acat_control_summary(summary)
    decision = build_route_decision(summary, placebo)

    profile.to_csv(outputs.upper_bound_profile_csv, index=False)
    summary.to_csv(outputs.upper_bound_summary_csv, index=False)
    placebo.to_csv(outputs.placebo_summary_csv, index=False)
    controls.to_csv(outputs.rsp_acat_control_summary_csv, index=False)
    outputs.route_decision_json.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    write_result_markdown(
        output_path=outputs.result_md,
        run_stamp=run_stamp,
        profile=profile,
        summary=summary,
        placebo=placebo,
        controls=controls,
        decision=decision,
        manifest_name=outputs.manifest_json.name,
    )

    manifest = {
        "run_stamp": run_stamp,
        "run_date": run_date,
        "run_iso": run_iso,
        "experiment_stage": "M10-R1",
        "analysis_script": ANALYSIS_SCRIPT,
        "design_note": DESIGN_NOTE_NAME,
        "total_design_note": TOTAL_DESIGN_NOTE_NAME,
        "diagnostic_only_no_training": True,
        "inputs": {
            "task4_profile_path": str(args.task4_profile_path),
            "r0_delta_profile_path": str(args.r0_delta_profile_path),
            "near_cutoff_profile_path": str(args.near_cutoff_profile_path),
            "recoverability_profile_path": str(args.recoverability_profile_path),
            "failure_consensus_profile_path": str(args.failure_consensus_profile_path),
            "rank_recoverability_profile_path": str(args.rank_recoverability_profile_path),
        },
        "parameters": {
            "shuffle_count": args.shuffle_count,
            "random_seed": args.random_seed,
        },
        "outputs": {field: str(getattr(outputs, field)) for field in outputs.__dataclass_fields__},
        "decision": decision,
    }
    outputs.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M10-R1 recoverability upper-bound audit.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--task4-profile-path", default=str(DEFAULT_TASK4_PROFILE))
    parser.add_argument("--r0-delta-profile-path", default=str(DEFAULT_R0_DELTA_PROFILE))
    parser.add_argument("--near-cutoff-profile-path", default=str(DEFAULT_NEAR_CUTOFF_PROFILE))
    parser.add_argument("--recoverability-profile-path", default=str(DEFAULT_RECOVERABILITY_PROFILE))
    parser.add_argument("--failure-consensus-profile-path", default=str(DEFAULT_FAILURE_PROFILE))
    parser.add_argument("--rank-recoverability-profile-path", default=str(DEFAULT_RANK_PROFILE))
    parser.add_argument("--shuffle-count", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=43)
    parser.add_argument("--run-stamp", default="")
    return parser


def main() -> None:
    outputs = run(build_arg_parser().parse_args())
    print(f"output_dir={outputs.output_dir}")
    print(f"result_md={outputs.result_md}")
    print(f"route_decision={outputs.route_decision_json}")


if __name__ == "__main__":
    main()
