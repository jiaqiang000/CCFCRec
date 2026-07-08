#!/usr/bin/env python3
"""
Build Task4-pre-2 Acat_v3 target profile and trainability audit for Amazon-VG.

The profile separates train-usable availability inputs from eval-only baseline
failure diagnostics to avoid leaking test-derived hard labels into training.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DESIGN_NOTE = (
    "上游设计：[[2026-07-05 122744 CCFCRec Amazon-VG Task4 Acat_v3 high-failure pairwise margin carrier 设计]]"
)

CONTROL_COLUMNS = [
    "category_count",
    "R_metadata_richness_score",
    "S_train_support_score",
    "P_popularity_score",
]


@dataclass(frozen=True)
class TargetProfileOutputs:
    profile_csv: Path
    group_summary_csv: Path
    split_summary_csv: Path
    category_concentration_csv: Path
    trainability_summary_csv: Path
    route_json: Path
    result_md: Path
    manifest_json: Path


def now_stamp() -> tuple[str, str]:
    now = datetime.now().astimezone()
    return now.strftime("%Y-%m-%d %H%M%S"), now.isoformat(timespec="seconds")


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_mean(series: pd.Series) -> float:
    values = _numeric(series).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _safe_bool_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(series.fillna(False).astype(bool).mean())


def _require_columns(frame: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} 缺少字段: {missing}")


def percentile_score(series: pd.Series) -> pd.Series:
    values = _numeric(series)
    if values.dropna().nunique() <= 1:
        return pd.Series(0.5, index=series.index, dtype=float)
    return values.rank(method="average", pct=True).fillna(0.5).astype(float)


def tertile_group(score: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
    values = _numeric(score)
    if values.dropna().nunique() <= 1:
        return pd.Series(labels[1], index=score.index)
    low = values.quantile(1 / 3)
    high = values.quantile(2 / 3)
    return pd.Series(
        np.select([values <= low, values >= high], [labels[0], labels[2]], default=labels[1]),
        index=score.index,
    )


def build_rsp_score(acat: pd.DataFrame) -> pd.DataFrame:
    _require_columns(acat, CONTROL_COLUMNS, "Acat_v3")
    result = acat.copy()
    percentiles = [percentile_score(result[column]) for column in CONTROL_COLUMNS]
    result["RSP_score"] = pd.concat(percentiles, axis=1).mean(axis=1)
    result["RSP_group"] = tertile_group(result["RSP_score"], ("RSP_low", "RSP_mid", "RSP_high"))
    return result


def _normalise_recoverability(recoverability: pd.DataFrame) -> pd.DataFrame:
    frame = recoverability.copy()
    if "raw_asin" not in frame.columns and "asin" in frame.columns:
        frame = frame.rename(columns={"asin": "raw_asin"})
    _require_columns(frame, ["raw_asin"], "recoverability")
    rename = {
        "ndcg@20": "baseline_ndcg@20",
        "margin_proxy": "baseline_margin_proxy",
        "best_target_rank": "baseline_best_target_rank",
    }
    return frame.rename(columns={key: value for key, value in rename.items() if key in frame.columns})


def build_eval_baseline_hard_flag(profile: pd.DataFrame) -> pd.Series:
    eval_rows = profile["baseline_ndcg@20"].notna() | profile["baseline_margin_proxy"].notna() | profile["baseline_best_target_rank"].notna()
    hard = pd.Series(False, index=profile.index)
    eval_frame = profile[eval_rows]
    if eval_frame.empty:
        return hard

    parts = []
    if "baseline_ndcg@20" in eval_frame.columns and eval_frame["baseline_ndcg@20"].notna().any():
        ndcg_cut = _numeric(eval_frame["baseline_ndcg@20"]).quantile(1 / 3)
        parts.append(_numeric(profile["baseline_ndcg@20"]) <= ndcg_cut)
    if "baseline_margin_proxy" in eval_frame.columns and eval_frame["baseline_margin_proxy"].notna().any():
        margin_cut = _numeric(eval_frame["baseline_margin_proxy"]).quantile(1 / 3)
        parts.append(_numeric(profile["baseline_margin_proxy"]) <= margin_cut)
    if "baseline_best_target_rank" in eval_frame.columns and eval_frame["baseline_best_target_rank"].notna().any():
        rank_cut = _numeric(eval_frame["baseline_best_target_rank"]).quantile(2 / 3)
        parts.append(_numeric(profile["baseline_best_target_rank"]) >= rank_cut)

    if not parts:
        return hard
    combined = pd.concat(parts, axis=1).fillna(False).any(axis=1)
    return combined & eval_rows


def build_target_profile(acat: pd.DataFrame, recoverability: pd.DataFrame) -> pd.DataFrame:
    required = [
        "raw_asin",
        "split",
        "s_cat_v3",
        "s_cat_v3_group",
        "category_raw",
        "category_tokens",
        *CONTROL_COLUMNS,
    ]
    _require_columns(acat, required, "Acat_v3")
    scored = build_rsp_score(acat)
    recoverability_norm = _normalise_recoverability(recoverability)
    keep_recoverability = [
        "raw_asin",
        "baseline_ndcg@20",
        "baseline_margin_proxy",
        "baseline_best_target_rank",
        "gt_user_count",
        "target_activity_bucket",
        "target_history_interaction_count_mean",
        "raw_test_user_count",
        "proxy_ensemble_score",
        "consensus_score",
    ]
    profile = scored.merge(
        recoverability_norm[[column for column in keep_recoverability if column in recoverability_norm.columns]].drop_duplicates("raw_asin"),
        on="raw_asin",
        how="left",
    )
    profile["high_acat_flag"] = profile["s_cat_v3_group"].eq("s_cat_v3_strong")
    profile["eval_baseline_hard_flag"] = build_eval_baseline_hard_flag(profile)
    profile["high_acat_eval_hard_flag"] = profile["high_acat_flag"] & profile["eval_baseline_hard_flag"]
    profile["eval_metric_available_flag"] = (
        profile["baseline_ndcg@20"].notna()
        | profile["baseline_margin_proxy"].notna()
        | profile["baseline_best_target_rank"].notna()
    )
    profile["train_safe_hard_proxy_available"] = False
    profile["hard_flag_policy"] = "eval_only_not_train_input"
    return profile


def build_group_summary(profile: pd.DataFrame) -> pd.DataFrame:
    groups = [
        ("all", pd.Series(True, index=profile.index)),
        ("high_acat", profile["high_acat_flag"].astype(bool)),
        ("eval_baseline_hard", profile["eval_baseline_hard_flag"].astype(bool)),
        ("high_acat_eval_hard", profile["high_acat_eval_hard_flag"].astype(bool)),
        ("RSP_high", profile["RSP_group"].eq("RSP_high")),
    ]
    rows = []
    for group_name, mask in groups:
        frame = profile[mask]
        rows.append(
            {
                "group": group_name,
                "item_count": int(len(frame)),
                "eval_item_count": int(frame["eval_metric_available_flag"].sum()) if "eval_metric_available_flag" in frame.columns else 0,
                "s_cat_v3_mean": _safe_mean(frame["s_cat_v3"]) if "s_cat_v3" in frame.columns else float("nan"),
                "RSP_score_mean": _safe_mean(frame["RSP_score"]) if "RSP_score" in frame.columns else float("nan"),
                "baseline_ndcg@20_mean": _safe_mean(frame["baseline_ndcg@20"]) if "baseline_ndcg@20" in frame.columns else float("nan"),
                "baseline_margin_proxy_mean": _safe_mean(frame["baseline_margin_proxy"]) if "baseline_margin_proxy" in frame.columns else float("nan"),
                "baseline_best_target_rank_mean": _safe_mean(frame["baseline_best_target_rank"]) if "baseline_best_target_rank" in frame.columns else float("nan"),
                "gt_user_count_mean": _safe_mean(frame["gt_user_count"]) if "gt_user_count" in frame.columns else float("nan"),
                "target_history_interaction_count_mean": _safe_mean(frame["target_history_interaction_count_mean"]) if "target_history_interaction_count_mean" in frame.columns else float("nan"),
                "RSP_high_share": _safe_bool_mean(frame["RSP_group"].eq("RSP_high")) if "RSP_group" in frame.columns else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_split_summary(profile: pd.DataFrame) -> pd.DataFrame:
    return (
        profile.groupby("split", dropna=False)
        .agg(
            item_count=("raw_asin", "size"),
            high_acat_count=("high_acat_flag", "sum"),
            eval_item_count=("eval_metric_available_flag", "sum"),
            eval_baseline_hard_count=("eval_baseline_hard_flag", "sum"),
            high_acat_eval_hard_count=("high_acat_eval_hard_flag", "sum"),
            s_cat_v3_mean=("s_cat_v3", "mean"),
            RSP_score_mean=("RSP_score", "mean"),
        )
        .reset_index()
    )


def build_category_concentration(profile: pd.DataFrame, group_col: str = "high_acat_eval_hard_flag") -> pd.DataFrame:
    if group_col not in profile.columns:
        raise ValueError(f"缺少字段: {group_col}")
    frame = profile[profile[group_col].astype(bool)].copy()
    if frame.empty:
        return pd.DataFrame(columns=["category_raw", "item_count", "share"])
    total = len(frame)
    result = (
        frame.groupby("category_raw", dropna=False)
        .agg(item_count=("raw_asin", "size"))
        .reset_index()
        .sort_values("item_count", ascending=False)
    )
    result["share"] = result["item_count"] / total
    return result


def build_trainability_summary(profile: pd.DataFrame) -> pd.DataFrame:
    eval_mask = profile["eval_metric_available_flag"].astype(bool)
    opportunity = profile["high_acat_eval_hard_flag"].astype(bool)
    opportunity_frame = profile[opportunity]
    concentration = build_category_concentration(profile)
    top_category_share = float(concentration["share"].iloc[0]) if not concentration.empty else 0.0
    eval_count = int(eval_mask.sum())
    opportunity_count = int(opportunity.sum())
    rsp_high_share = _safe_bool_mean(opportunity_frame["RSP_group"].eq("RSP_high")) if not opportunity_frame.empty else 0.0
    gt1_share = _safe_bool_mean(opportunity_frame["gt_user_count"].eq(1)) if "gt_user_count" in opportunity_frame.columns and not opportunity_frame.empty else 0.0
    target_low_share = _safe_bool_mean(opportunity_frame["target_activity_bucket"].eq("low")) if "target_activity_bucket" in opportunity_frame.columns and not opportunity_frame.empty else 0.0
    row = {
        "total_item_count": int(len(profile)),
        "eval_item_count": eval_count,
        "high_acat_count": int(profile["high_acat_flag"].sum()),
        "eval_baseline_hard_count": int(profile["eval_baseline_hard_flag"].sum()),
        "high_acat_eval_hard_count": opportunity_count,
        "high_acat_eval_hard_share_total": opportunity_count / len(profile) if len(profile) else 0.0,
        "high_acat_eval_hard_share_eval": opportunity_count / eval_count if eval_count else 0.0,
        "high_acat_eval_hard_top_category_share": top_category_share,
        "high_acat_eval_hard_rsp_high_share": rsp_high_share,
        "high_acat_eval_hard_gt1_share": gt1_share,
        "high_acat_eval_hard_target_activity_low_share": target_low_share,
        "train_safe_hard_proxy_available": False,
        "hard_flag_policy": "eval_baseline_hard_flag is eval-only and must not be used as training input",
    }
    return pd.DataFrame([row])


def decide_trainability_route(summary: pd.DataFrame) -> dict[str, Any]:
    if summary.empty:
        return {"route": "target_profile_missing", "reason": "没有 trainability summary。", "evidence": {}}
    row = summary.iloc[0]
    count = int(row["high_acat_eval_hard_count"])
    share_eval = float(row["high_acat_eval_hard_share_eval"])
    top_category_share = float(row["high_acat_eval_hard_top_category_share"])
    rsp_high_share = float(row["high_acat_eval_hard_rsp_high_share"])
    train_proxy_available = bool(row["train_safe_hard_proxy_available"])

    if count < 100 or share_eval < 0.03 or top_category_share >= 0.50:
        route = "target_profile_too_small_or_concentrated"
        reason = "high-Acat eval-hard 机会组过小、占比过低或类别过度集中，不能直接训练。"
    elif rsp_high_share >= 0.70:
        route = "target_profile_rsp_overlap_too_high"
        reason = "high-Acat eval-hard 机会组与 RSP_high 高度重叠，availability 独立性不足。"
    elif not train_proxy_available:
        route = "target_profile_ready_train_proxy_needed"
        reason = "机会组规模和分散性足够，但 hard flag 仍是 eval-only；下一步必须构造 train-safe hard proxy。"
    else:
        route = "target_profile_ready_for_m3"
        reason = "机会组可训练性通过，且已有 train-safe hard proxy，可进入 M3 训练设计。"

    return {
        "route": route,
        "reason": reason,
        "evidence": {
            "high_acat_eval_hard_count": _jsonable(count),
            "high_acat_eval_hard_share_eval": _jsonable(share_eval),
            "high_acat_eval_hard_top_category_share": _jsonable(top_category_share),
            "high_acat_eval_hard_rsp_high_share": _jsonable(rsp_high_share),
            "train_safe_hard_proxy_available": train_proxy_available,
        },
    }


def _md_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_无数据_"
    selected = frame.head(max_rows)
    headers = list(selected.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in selected.iterrows():
        cells = []
        for header in headers:
            value = row[header]
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, float):
                cells.append(f"{value:.6g}")
            else:
                cells.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_outputs(
    profile: pd.DataFrame,
    group_summary: pd.DataFrame,
    split_summary: pd.DataFrame,
    category_concentration: pd.DataFrame,
    trainability_summary: pd.DataFrame,
    decision: dict[str, Any],
    output_dir: Path,
    input_files: dict[str, str],
) -> TargetProfileOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp, run_iso = now_stamp()
    profile_csv = output_dir / "task4_training_target_profile.csv"
    group_summary_csv = output_dir / "task4_training_target_group_summary.csv"
    split_summary_csv = output_dir / "task4_training_target_split_summary.csv"
    category_concentration_csv = output_dir / "task4_training_target_category_concentration.csv"
    trainability_summary_csv = output_dir / "task4_training_target_trainability_summary.csv"
    route_json = output_dir / "task4_training_target_route_decision.json"
    result_md = output_dir / f"{stamp} CCFCRec Amazon-VG Task4-pre-2 Acat v3 target profile 可训练性审计结果.md"
    manifest_json = output_dir / "run_manifest.json"

    profile.to_csv(profile_csv, index=False)
    group_summary.to_csv(group_summary_csv, index=False)
    split_summary.to_csv(split_summary_csv, index=False)
    category_concentration.to_csv(category_concentration_csv, index=False)
    trainability_summary.to_csv(trainability_summary_csv, index=False)
    route_json.write_text(json.dumps(_jsonable(decision), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    markdown = f"""---
title: {stamp} CCFCRec Amazon-VG Task4-pre-2 Acat v3 target profile 可训练性审计结果
date: {stamp[:10]}
time: "{stamp[11:13]}:{stamp[13:15]}:{stamp[15:17]}"
created_at: "{stamp[:10]} {stamp[11:13]}:{stamp[13:15]}:{stamp[15:17]}"
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - Acat_v3
  - Task4-pre-2
---

# {stamp} CCFCRec Amazon-VG Task4-pre-2 Acat v3 target profile 可训练性审计结果

## 来源说明

> [!info] 来源说明
> {DESIGN_NOTE}
> 本结果目录：`{output_dir}/`

## 结论

route decision 为 `{decision.get("route")}`。

{decision.get("reason", "")}

## 术语说明

- `high_acat_flag`：高 Acat_v3，属于可训练 availability 输入。
- `eval_baseline_hard_flag`：评估侧 baseline failure 诊断，不能作为训练输入。
- `high_acat_eval_hard_flag`：高 Acat_v3 且评估侧 baseline hard 的方法机会组诊断。
- `RSP_score`：由 category_count/R/S/P 组合出的控制变量对照分数。

## Route Evidence

```json
{json.dumps(_jsonable(decision.get("evidence", {})), ensure_ascii=False, indent=2)}
```

## Trainability Summary

{_md_table(trainability_summary)}

## Group Summary

{_md_table(group_summary)}

## Split Summary

{_md_table(split_summary)}

## Category Concentration

{_md_table(category_concentration)}
"""
    result_md.write_text(markdown, encoding="utf-8")

    manifest = {
        "run_time": run_iso,
        "script": str(Path(__file__).resolve()),
        "output_dir": str(output_dir),
        "input_files": input_files,
        "outputs": {
            "profile_csv": str(profile_csv),
            "group_summary_csv": str(group_summary_csv),
            "split_summary_csv": str(split_summary_csv),
            "category_concentration_csv": str(category_concentration_csv),
            "trainability_summary_csv": str(trainability_summary_csv),
            "route_json": str(route_json),
            "result_md": str(result_md),
        },
        "row_counts": {
            "profile": int(len(profile)),
            "group_summary": int(len(group_summary)),
            "split_summary": int(len(split_summary)),
            "category_concentration": int(len(category_concentration)),
            "trainability_summary": int(len(trainability_summary)),
        },
        "route_decision": decision,
        "design_note": DESIGN_NOTE,
    }
    manifest_json.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    return TargetProfileOutputs(
        profile_csv=profile_csv,
        group_summary_csv=group_summary_csv,
        split_summary_csv=split_summary_csv,
        category_concentration_csv=category_concentration_csv,
        trainability_summary_csv=trainability_summary_csv,
        route_json=route_json,
        result_md=result_md,
        manifest_json=manifest_json,
    )


def run_target_profile_audit(acat_v3_path: Path, recoverability_path: Path, output_dir: Path) -> TargetProfileOutputs:
    acat = pd.read_csv(acat_v3_path)
    recoverability = pd.read_csv(recoverability_path)
    profile = build_target_profile(acat, recoverability)
    group_summary = build_group_summary(profile)
    split_summary = build_split_summary(profile)
    category_concentration = build_category_concentration(profile)
    trainability_summary = build_trainability_summary(profile)
    decision = decide_trainability_route(trainability_summary)
    return write_outputs(
        profile=profile,
        group_summary=group_summary,
        split_summary=split_summary,
        category_concentration=category_concentration,
        trainability_summary=trainability_summary,
        decision=decision,
        output_dir=output_dir,
        input_files={
            "acat_v3": str(acat_v3_path),
            "recoverability": str(recoverability_path),
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acat-v3", type=Path, required=True)
    parser.add_argument("--recoverability-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_target_profile_audit(
        acat_v3_path=args.acat_v3,
        recoverability_path=args.recoverability_profile,
        output_dir=args.output_dir,
    )
    print(f"wrote {outputs.profile_csv}")
    print(f"wrote {outputs.group_summary_csv}")
    print(f"wrote {outputs.split_summary_csv}")
    print(f"wrote {outputs.category_concentration_csv}")
    print(f"wrote {outputs.trainability_summary_csv}")
    print(f"wrote {outputs.route_json}")
    print(f"wrote {outputs.result_md}")
    print(f"wrote {outputs.manifest_json}")


if __name__ == "__main__":
    main()
