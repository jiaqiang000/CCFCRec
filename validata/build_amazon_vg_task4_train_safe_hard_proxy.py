#!/usr/bin/env python3
"""
Build Task4-pre-3 train-safe hard proxy audit for Amazon-VG.

The proxy scores are constructed only from Acat_v3, category metadata, and
train-graph statistics. Eval/test ranking diagnostics are used only for audit.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DESIGN_NOTE = (
    "上游设计：[[2026-07-06 003849 CCFCRec Amazon-VG Task4-pre-3 Acat v3 train-safe hard proxy 构造审计设计]]"
)

SELECTED_CANDIDATE = "category_neighbor_mismatch_proxy"

CANDIDATE_SCORE_COLUMNS = [
    "support_tail_proxy_score",
    "collab_noise_proxy_score",
    "acat_rsp_gap_proxy_score",
    "category_neighbor_mismatch_proxy_score",
]

FORBIDDEN_SCORE_COLUMNS = [
    "hr@20",
    "ndcg@20",
    "baseline_ndcg@20",
    "margin_proxy",
    "baseline_margin_proxy",
    "best_target_rank",
    "baseline_best_target_rank",
    "eval_baseline_hard_flag",
    "high_acat_eval_hard_flag",
    "proxy_ensemble_score",
    "proxy_ensemble_score_x",
    "proxy_ensemble_score_y",
    "consensus_score",
    "consensus_score_x",
    "consensus_score_y",
    "delta_ndcg@20",
]

REQUIRED_COLUMNS = [
    "raw_asin",
    "split",
    "category_raw",
    "category_tokens",
    "category_count",
    "s_cat_v3",
    "s_cat_v3_group",
    "Acat_v3_disc_residual_pct",
    "Acat_v3_collab_residual_pct",
    "RSP_score",
    "RSP_group",
    "S_train_support_score",
    "P_popularity_score",
    "S_train_token_user_support_mean",
    "S_train_token_interaction_support_mean",
    "A_collab_user_set_jaccard_mean",
    "A_collab_support_entropy_mean",
    "A_collab_train_token_user_support_mean",
    "high_acat_flag",
]

AUDIT_COLUMNS = [
    "eval_metric_available_flag",
    "eval_baseline_hard_flag",
    "high_acat_eval_hard_flag",
    "baseline_ndcg@20",
    "baseline_margin_proxy",
    "baseline_best_target_rank",
]

ROUTE_THRESHOLDS = {
    "min_high_acat_train_proxy_hard_count": 500,
    "min_high_acat_eval_proxy_high_count": 300,
    "min_high_acat_eval_hard_capture_rate": 0.50,
    "max_abs_spearman_vs_RSP_score": 0.70,
    "max_high_acat_proxy_high_rsp_high_share": 0.50,
}


@dataclass(frozen=True)
class ProxyOutputs:
    profile_csv: Path
    candidate_summary_csv: Path
    overlap_summary_csv: Path
    correlation_csv: Path
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
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} 缺少字段: {missing}")


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _bool_series(series: pd.Series | None, index: pd.Index, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=bool)
    if series.dtype == bool:
        return series.fillna(default).astype(bool)
    lowered = series.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", "", "nan", "<na>", "none"}
    result = pd.Series(default, index=series.index, dtype=bool)
    result.loc[lowered.isin(true_values)] = True
    result.loc[lowered.isin(false_values)] = False
    return result.reindex(index, fill_value=default)


def _safe_mean(series: pd.Series) -> float:
    values = _numeric(series).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _safe_bool_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    return float(_bool_series(series, series.index).mean())


def _safe_corr(left: pd.Series, right: pd.Series, method: str) -> float:
    valid = pd.concat([_numeric(left), _numeric(right)], axis=1).dropna()
    if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))


def train_percentile(frame: pd.DataFrame, column: str, inverse: bool = False) -> pd.Series:
    _require_columns(frame, ["split", column], f"train_percentile({column})")
    values = _numeric(frame[column])
    if inverse:
        values = -values
    train_values = values[frame["split"].eq("train")].replace([np.inf, -np.inf], np.nan).dropna()
    if train_values.empty or train_values.nunique() <= 1:
        return pd.Series(0.5, index=frame.index, dtype=float)

    sorted_train = np.sort(train_values.to_numpy(dtype=float))
    percentiles: list[float] = []
    for value in values.to_numpy(dtype=float):
        if not np.isfinite(value):
            percentiles.append(0.5)
        else:
            percentiles.append(float(np.searchsorted(sorted_train, value, side="right") / len(sorted_train)))
    return pd.Series(percentiles, index=frame.index, dtype=float).clip(0.0, 1.0)


def _mean_series(series_list: list[pd.Series]) -> pd.Series:
    return pd.concat(series_list, axis=1).apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0.5).clip(0.0, 1.0)


def _score_to_group(value: float, low_cut: float, high_cut: float) -> str:
    if value <= low_cut:
        return "low"
    if value >= high_cut:
        return "high"
    return "mid"


def _train_tertile_group(score: pd.Series, split: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    train_scores = _numeric(score[split.eq("train")]).dropna()
    if train_scores.empty or train_scores.nunique() <= 1:
        ranked = _numeric(score).rank(method="average", pct=True).fillna(0.5)
        group = pd.Series(
            np.select([ranked <= 1 / 3, ranked >= 2 / 3], ["low", "high"], default="mid"),
            index=score.index,
        )
        return group, {"low_cut": None, "high_cut": None, "fallback": True}
    low_cut = float(train_scores.quantile(1 / 3))
    high_cut = float(train_scores.quantile(2 / 3))
    if np.isclose(low_cut, high_cut):
        ranked = _numeric(score).rank(method="average", pct=True).fillna(0.5)
        group = pd.Series(
            np.select([ranked <= 1 / 3, ranked >= 2 / 3], ["low", "high"], default="mid"),
            index=score.index,
        )
        return group, {"low_cut": low_cut, "high_cut": high_cut, "fallback": True}
    group = _numeric(score).map(lambda value: _score_to_group(float(value), low_cut, high_cut))
    return group, {"low_cut": low_cut, "high_cut": high_cut, "fallback": False}


def _candidate_name(score_col: str) -> str:
    return score_col.removesuffix("_score")


def build_proxy_profile(target_profile: pd.DataFrame) -> pd.DataFrame:
    _require_columns(target_profile, REQUIRED_COLUMNS, "target_profile")
    frame = target_profile.copy()

    frame["support_tail_proxy_score"] = _mean_series(
        [
            train_percentile(frame, "S_train_support_score", inverse=True),
            train_percentile(frame, "P_popularity_score", inverse=True),
            train_percentile(frame, "S_train_token_user_support_mean", inverse=True),
            train_percentile(frame, "S_train_token_interaction_support_mean", inverse=True),
        ]
    )
    frame["collab_noise_proxy_score"] = _mean_series(
        [
            train_percentile(frame, "A_collab_user_set_jaccard_mean", inverse=True),
            train_percentile(frame, "A_collab_support_entropy_mean"),
            train_percentile(frame, "A_collab_train_token_user_support_mean", inverse=True),
        ]
    )
    frame["acat_rsp_gap_proxy_score"] = _mean_series(
        [
            _numeric(frame["s_cat_v3"]).fillna(0.5),
            train_percentile(frame, "RSP_score", inverse=True),
        ]
    )
    frame["category_neighbor_mismatch_proxy_score"] = _mean_series(
        [
            _numeric(frame["s_cat_v3"]).fillna(0.5),
            _numeric(frame["Acat_v3_disc_residual_pct"]).fillna(0.5),
            _numeric(frame["Acat_v3_collab_residual_pct"]).fillna(0.5),
            train_percentile(frame, "A_collab_user_set_jaccard_mean", inverse=True),
            train_percentile(frame, "A_collab_support_entropy_mean"),
            train_percentile(frame, "S_train_token_user_support_mean", inverse=True),
        ]
    )

    for score_col in CANDIDATE_SCORE_COLUMNS:
        candidate = _candidate_name(score_col)
        group, meta = _train_tertile_group(frame[score_col], frame["split"])
        frame[f"{candidate}_group"] = group.map(lambda value: f"{candidate}_{value}")
        frame[f"{candidate}_high_flag"] = group.eq("high")
        frame[f"{candidate}_train_high_cut"] = meta["high_cut"]

    selected_score = f"{SELECTED_CANDIDATE}_score"
    selected_group = f"{SELECTED_CANDIDATE}_group"
    selected_high = f"{SELECTED_CANDIDATE}_high_flag"
    frame["train_safe_hard_proxy_score"] = frame[selected_score]
    frame["train_safe_hard_proxy_group"] = frame[selected_group]
    frame["train_safe_hard_proxy_high_flag"] = frame[selected_high]
    frame["high_acat_flag"] = _bool_series(frame["high_acat_flag"], frame.index)
    frame["high_acat_train_safe_hard_flag"] = frame["high_acat_flag"] & frame["train_safe_hard_proxy_high_flag"]
    frame["train_safe_hard_proxy_source"] = SELECTED_CANDIDATE
    frame["train_safe_hard_proxy_policy"] = "metadata_train_graph_only_no_eval_metrics"
    return frame.sort_values("raw_asin").reset_index(drop=True)


def _eval_mask(frame: pd.DataFrame) -> pd.Series:
    if "eval_metric_available_flag" in frame.columns:
        return _bool_series(frame["eval_metric_available_flag"], frame.index)
    audit_available = [column for column in ["baseline_ndcg@20", "baseline_margin_proxy", "baseline_best_target_rank"] if column in frame.columns]
    if not audit_available:
        return pd.Series(False, index=frame.index, dtype=bool)
    mask = pd.Series(False, index=frame.index, dtype=bool)
    for column in audit_available:
        mask = mask | _numeric(frame[column]).notna()
    return mask


def _hard_mask(frame: pd.DataFrame) -> pd.Series:
    if "eval_baseline_hard_flag" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return _bool_series(frame["eval_baseline_hard_flag"], frame.index)


def _high_acat_mask(frame: pd.DataFrame) -> pd.Series:
    if "high_acat_flag" in frame.columns:
        return _bool_series(frame["high_acat_flag"], frame.index)
    return frame["s_cat_v3_group"].eq("s_cat_v3_strong")


def _rsp_high_mask(frame: pd.DataFrame) -> pd.Series:
    if "RSP_group" not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame["RSP_group"].astype(str).eq("RSP_high")


def _rate(mask: pd.Series, denominator: pd.Series) -> float:
    denom_count = int(denominator.sum())
    if denom_count <= 0:
        return float("nan")
    return float((mask & denominator).sum() / denom_count)


def build_correlation_summary(proxy_profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eval_rows = _eval_mask(proxy_profile)
    hard = _hard_mask(proxy_profile).astype(float)
    right_columns = [
        ("RSP_score", "control", proxy_profile.index),
        ("s_cat_v3", "availability", proxy_profile.index),
        ("S_train_support_score", "control", proxy_profile.index),
        ("P_popularity_score", "control", proxy_profile.index),
        ("category_count", "control", proxy_profile.index),
        ("eval_baseline_hard_flag", "eval_audit", proxy_profile[eval_rows].index),
        ("baseline_ndcg@20", "eval_audit", proxy_profile[eval_rows].index),
    ]
    for score_col in CANDIDATE_SCORE_COLUMNS:
        candidate = _candidate_name(score_col)
        for right, family, index in right_columns:
            if right == "eval_baseline_hard_flag":
                right_series = hard
            elif right in proxy_profile.columns:
                right_series = proxy_profile[right]
            else:
                continue
            frame = proxy_profile.loc[index]
            rows.append(
                {
                    "candidate": candidate,
                    "score_col": score_col,
                    "right": right,
                    "right_family": family,
                    "n": int(len(frame)),
                    "pearson": _safe_corr(proxy_profile.loc[index, score_col], right_series.loc[index], "pearson"),
                    "spearman": _safe_corr(proxy_profile.loc[index, score_col], right_series.loc[index], "spearman"),
                }
            )
    return pd.DataFrame(rows)


def build_overlap_summary(proxy_profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    eval_rows = _eval_mask(proxy_profile)
    hard = _hard_mask(proxy_profile)
    high_acat = _high_acat_mask(proxy_profile)
    rsp_high = _rsp_high_mask(proxy_profile)
    high_acat_eval = eval_rows & high_acat
    high_acat_eval_hard = eval_rows & high_acat & hard

    for score_col in CANDIDATE_SCORE_COLUMNS:
        candidate = _candidate_name(score_col)
        high = _bool_series(proxy_profile[f"{candidate}_high_flag"], proxy_profile.index)
        eval_high = eval_rows & high
        high_acat_proxy = high_acat & high
        high_acat_eval_proxy = eval_rows & high_acat_proxy
        rows.append(
            {
                "candidate": candidate,
                "eval_proxy_high_count": int(eval_high.sum()),
                "eval_proxy_high_hard_rate": _safe_bool_mean(hard[eval_high]) if eval_high.any() else float("nan"),
                "high_acat_eval_count": int(high_acat_eval.sum()),
                "high_acat_eval_hard_count": int(high_acat_eval_hard.sum()),
                "high_acat_eval_proxy_high_count": int(high_acat_eval_proxy.sum()),
                "high_acat_proxy_high_eval_hard_rate": _safe_bool_mean(hard[high_acat_eval_proxy])
                if high_acat_eval_proxy.any()
                else float("nan"),
                "high_acat_eval_hard_base_rate": _safe_bool_mean(hard[high_acat_eval]) if high_acat_eval.any() else float("nan"),
                "high_acat_eval_hard_capture_rate": _rate(high, high_acat_eval_hard),
                "eval_hard_capture_rate": _rate(high, eval_rows & hard),
                "proxy_high_rsp_high_share": _safe_bool_mean(rsp_high[eval_high]) if eval_high.any() else float("nan"),
                "high_acat_proxy_high_rsp_high_share": _safe_bool_mean(rsp_high[high_acat_eval_proxy])
                if high_acat_eval_proxy.any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_candidate_summary(proxy_profile: pd.DataFrame) -> pd.DataFrame:
    overlap = build_overlap_summary(proxy_profile)
    corr = build_correlation_summary(proxy_profile)
    rows: list[dict[str, Any]] = []
    high_acat = _high_acat_mask(proxy_profile)
    for score_col in CANDIDATE_SCORE_COLUMNS:
        candidate = _candidate_name(score_col)
        high_col = f"{candidate}_high_flag"
        high = _bool_series(proxy_profile[high_col], proxy_profile.index)
        row: dict[str, Any] = {
            "candidate": candidate,
            "score_col": score_col,
            "train_high_count": int((proxy_profile["split"].eq("train") & high).sum()),
            "validate_high_count": int((proxy_profile["split"].eq("validate") & high).sum()),
            "test_high_count": int((proxy_profile["split"].eq("test") & high).sum()),
            "high_acat_train_proxy_hard_count": int((proxy_profile["split"].eq("train") & high_acat & high).sum()),
            "score_mean": _safe_mean(proxy_profile[score_col]),
            "score_train_mean": _safe_mean(proxy_profile.loc[proxy_profile["split"].eq("train"), score_col]),
        }
        overlap_row = overlap[overlap["candidate"].eq(candidate)]
        if not overlap_row.empty:
            row.update(overlap_row.iloc[0].to_dict())
        for right in ["RSP_score", "s_cat_v3", "S_train_support_score", "P_popularity_score", "category_count", "eval_baseline_hard_flag", "baseline_ndcg@20"]:
            corr_row = corr[(corr["candidate"].eq(candidate)) & (corr["right"].eq(right))]
            if not corr_row.empty:
                row[f"spearman_vs_{right}"] = corr_row.iloc[0]["spearman"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("candidate").reset_index(drop=True)


def decide_proxy_route(candidate_summary: pd.DataFrame, selected_candidate: str = SELECTED_CANDIDATE) -> dict[str, Any]:
    selected = candidate_summary[candidate_summary["candidate"].eq(selected_candidate)]
    if selected.empty:
        return {
            "route": "train_safe_proxy_needs_validation_margin",
            "reason": f"selected candidate not found: {selected_candidate}",
            "selected_candidate": selected_candidate,
        }
    row = selected.iloc[0].to_dict()
    count_train = float(row.get("high_acat_train_proxy_hard_count", 0) or 0)
    count_eval = float(row.get("high_acat_eval_proxy_high_count", 0) or 0)
    hard_rate = float(row.get("high_acat_proxy_high_eval_hard_rate", float("nan")))
    base_rate = float(row.get("high_acat_eval_hard_base_rate", float("nan")))
    capture_rate = float(row.get("high_acat_eval_hard_capture_rate", float("nan")))
    rsp_corr = float(row.get("spearman_vs_RSP_score", float("nan")))
    rsp_share = float(row.get("high_acat_proxy_high_rsp_high_share", float("nan")))

    if abs(rsp_corr) >= ROUTE_THRESHOLDS["max_abs_spearman_vs_RSP_score"] or rsp_share >= ROUTE_THRESHOLDS["max_high_acat_proxy_high_rsp_high_share"]:
        route = "train_safe_proxy_rsp_overlap_too_high"
        reason = "selected proxy 与 RSP_score 或 RSP_high 过度重合。"
    elif count_train < ROUTE_THRESHOLDS["min_high_acat_train_proxy_hard_count"] or count_eval < ROUTE_THRESHOLDS["min_high_acat_eval_proxy_high_count"]:
        route = "train_safe_proxy_too_small"
        reason = "high Acat_v3 与 train-safe proxy 交叉组规模不足。"
    elif (np.isfinite(base_rate) and np.isfinite(hard_rate) and hard_rate < base_rate) or (
        np.isfinite(capture_rate) and capture_rate < ROUTE_THRESHOLDS["min_high_acat_eval_hard_capture_rate"]
    ):
        route = "train_safe_proxy_weak_eval_overlap"
        reason = "proxy 训练安全，但 eval-hard 机会组命中不足。"
    else:
        route = "train_safe_proxy_ready_for_m3"
        reason = "selected proxy 训练安全，规模足够，且能命中 high Acat_v3 eval-hard 机会组。"

    return {
        "route": route,
        "reason": reason,
        "selected_candidate": selected_candidate,
        "thresholds": ROUTE_THRESHOLDS,
        "evidence": _jsonable(row),
    }


def md_table(frame: pd.DataFrame, columns: list[str] | None = None, max_rows: int | None = None) -> str:
    if columns is None:
        columns = list(frame.columns)
    table = frame[columns].copy()
    if max_rows is not None:
        table = table.head(max_rows)
    if table.empty:
        return "_empty_"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in table.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                if math.isnan(value):
                    values.append("")
                else:
                    values.append(f"{value:.6g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_result_md(
    path: Path,
    result_dir: Path,
    route_decision: dict[str, Any],
    candidate_summary: pd.DataFrame,
    overlap_summary: pd.DataFrame,
    correlation: pd.DataFrame,
) -> None:
    selected = route_decision.get("selected_candidate", SELECTED_CANDIDATE)
    selected_corr = correlation[correlation["candidate"].eq(selected)].copy()
    content = f"""---
title: {path.stem}
date: {path.name[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - category_availability
  - Acat_v3
  - Task4-pre-3
---

# {path.stem}

## 来源说明

> [!info] 来源说明
> {DESIGN_NOTE}
> 本结果目录：`{result_dir}`

## 结论

route decision 为 `{route_decision.get("route")}`。

{route_decision.get("reason")}

## 字段解释

- `train_safe_hard_proxy_score`：最终训练可用 hard proxy 分数，不使用 test/eval 排序结果构造。
- `high_acat_train_safe_hard_flag`：高 Acat_v3 且 train-safe hard proxy 为 high 的交叉训练机会组。
- `capture_rate`：eval 机会组中被 train-safe proxy high 命中的比例，只用于审计。
- `RSP_score`：R/S/P/category_count 控制对照分数。

## Route Evidence

```json
{json.dumps(_jsonable(route_decision.get("evidence", {})), ensure_ascii=False, indent=2)}
```

## Candidate Summary

{md_table(candidate_summary)}

## Overlap Summary

{md_table(overlap_summary)}

## Selected Candidate Correlation

{md_table(selected_corr)}

## 产物

```text
task4_train_safe_hard_proxy_profile.csv
task4_train_safe_hard_proxy_candidate_summary.csv
task4_train_safe_hard_proxy_overlap_summary.csv
task4_train_safe_hard_proxy_correlation.csv
task4_train_safe_hard_proxy_route_decision.json
run_manifest.json
```
"""
    path.write_text(content, encoding="utf-8")


def run_audit(target_profile_path: Path, output_dir: Path) -> ProxyOutputs:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp, iso_time = now_stamp()
    target_profile = pd.read_csv(target_profile_path, low_memory=False)
    proxy_profile = build_proxy_profile(target_profile)
    candidate_summary = build_candidate_summary(proxy_profile)
    overlap_summary = build_overlap_summary(proxy_profile)
    correlation = build_correlation_summary(proxy_profile)
    route_decision = decide_proxy_route(candidate_summary)

    outputs = ProxyOutputs(
        profile_csv=output_dir / "task4_train_safe_hard_proxy_profile.csv",
        candidate_summary_csv=output_dir / "task4_train_safe_hard_proxy_candidate_summary.csv",
        overlap_summary_csv=output_dir / "task4_train_safe_hard_proxy_overlap_summary.csv",
        correlation_csv=output_dir / "task4_train_safe_hard_proxy_correlation.csv",
        route_json=output_dir / "task4_train_safe_hard_proxy_route_decision.json",
        result_md=output_dir / f"{stamp} CCFCRec Amazon-VG Task4-pre-3 Acat v3 train-safe hard proxy 构造审计结果.md",
        manifest_json=output_dir / "run_manifest.json",
    )
    proxy_profile.to_csv(outputs.profile_csv, index=False)
    candidate_summary.to_csv(outputs.candidate_summary_csv, index=False)
    overlap_summary.to_csv(outputs.overlap_summary_csv, index=False)
    correlation.to_csv(outputs.correlation_csv, index=False)
    outputs.route_json.write_text(json.dumps(_jsonable(route_decision), ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_at": iso_time,
        "script": Path(__file__).name,
        "target_profile_path": str(target_profile_path),
        "output_dir": str(output_dir),
        "selected_candidate": SELECTED_CANDIDATE,
        "forbidden_score_columns": FORBIDDEN_SCORE_COLUMNS,
        "route": route_decision.get("route"),
        "outputs": {field: str(getattr(outputs, field)) for field in outputs.__dataclass_fields__},
    }
    outputs.manifest_json.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    write_result_md(outputs.result_md, output_dir, route_decision, candidate_summary, overlap_summary, correlation)
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-profile", required=True, type=Path, help="Task4-pre-2 task4_training_target_profile.csv")
    parser.add_argument("--output-dir", required=True, type=Path, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_audit(args.target_profile.expanduser().resolve(), args.output_dir.expanduser().resolve())
    print(json.dumps({"result_md": str(outputs.result_md), "route_json": str(outputs.route_json)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
