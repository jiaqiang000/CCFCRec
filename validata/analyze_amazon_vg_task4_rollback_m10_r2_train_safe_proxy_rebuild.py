#!/usr/bin/env python3
"""
CCFCRec Amazon-VG M10-R2 train-safe hard proxy rebuild audit.

Diagnostic-only offline audit. It rebuilds candidate hard proxies and checks
whether any train-deployable proxy has enough discrimination to justify later
carrier design. It does not train models.
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
DEFAULT_R0_DELTA_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260709"
    / "2026-07-09 012326 task4-rollback-m10-r0-m9-post-audit"
    / "m10_r0_m9_item_delta_profile.csv"
)
DESIGN_NOTE_NAME = "2026-07-09 013853 CCFCRec Amazon-VG M10-R2 train-safe hard proxy rebuild 诊断设计"
TOTAL_DESIGN_NOTE_NAME = "2026-07-09 010147 CCFCRec Amazon-VG Task4-rollback M10-R recoverability and carrier audit 总设计"
R1_RESULT_NOTE_NAME = "2026-07-09 013333 CCFCRec Amazon-VG M10-R1 recoverability upper-bound audit 结果"
ANALYSIS_SCRIPT = "validata/analyze_amazon_vg_task4_rollback_m10_r2_train_safe_proxy_rebuild.py"

GATE = {
    "min_hard_rate_lift": 0.08,
    "min_spearman_vs_eval_hard_flag": 0.20,
    "max_abs_spearman_vs_rsp": 0.70,
    "max_proxy_high_rsp_high_share": 0.50,
    "min_selected_count": 500,
    "max_selected_share": 0.50,
}

TRAIN_DEPLOYABLE_SCORES = [
    "support_tail_proxy_score",
    "collab_noise_proxy_score",
    "acat_rsp_gap_proxy_score",
    "category_neighbor_mismatch_proxy_score",
    "competitor_pressure_proxy_score",
    "train_graph_near_cutoff_proxy_score",
    "residual_acat_pressure_proxy_score",
    "sampled_negative_pressure_proxy_score",
]
DIAGNOSTIC_SCORES = [
    "eval_margin_proxy_score",
    "eval_near_cutoff_proxy_score",
]
ALL_SCORE_COLUMNS = [*TRAIN_DEPLOYABLE_SCORES, *DIAGNOSTIC_SCORES]
R1_MASK_COLUMNS = [
    "recoverability_ensemble_high_hard",
    "rank_recoverability_high_hard",
    "near_cutoff_hard",
    "failure_consensus_high",
    "acat_v3_high_hard",
    "rsp_high_hard",
]


@dataclass(frozen=True)
class Outputs:
    output_dir: Path
    proxy_profile_csv: Path
    candidate_summary_csv: Path
    control_correlation_csv: Path
    r1_overlap_summary_csv: Path
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


def _safe_mean(series: pd.Series) -> float:
    values = _numeric(series).dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _safe_corr(left: pd.Series, right: pd.Series, method: str = "spearman") -> float:
    valid = pd.concat([_numeric(left), _numeric(right)], axis=1).dropna()
    if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return float("nan")
    return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method=method))


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


def _score_name(score_col: str) -> str:
    return score_col.removesuffix("_score")


def _candidate_type(score_col: str) -> str:
    return "diagnostic" if score_col in DIAGNOSTIC_SCORES else "train_deployable"


def _percentile(values: pd.Series, reference_mask: pd.Series | None = None, inverse: bool = False) -> pd.Series:
    numeric = _numeric(values)
    if inverse:
        numeric = -numeric
    ref = numeric[reference_mask] if reference_mask is not None else numeric
    ref = ref.replace([np.inf, -np.inf], np.nan).dropna()
    if ref.empty or ref.nunique() <= 1:
        ranked = numeric.rank(method="average", pct=True).fillna(0.5)
        return ranked.astype(float).clip(0.0, 1.0)
    sorted_ref = np.sort(ref.to_numpy(dtype=float))
    out = []
    for value in numeric.to_numpy(dtype=float):
        if not np.isfinite(value):
            out.append(0.5)
        else:
            out.append(float(np.searchsorted(sorted_ref, value, side="right") / len(sorted_ref)))
    return pd.Series(out, index=values.index, dtype=float).clip(0.0, 1.0)


def _mean_scores(scores: list[pd.Series], index: pd.Index) -> pd.Series:
    available = [score for score in scores if score is not None]
    if not available:
        return pd.Series(0.5, index=index, dtype=float)
    return pd.concat(available, axis=1).apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0.5).clip(0.0, 1.0)


def _train_mask(frame: pd.DataFrame) -> pd.Series:
    return frame.get("split", pd.Series("", index=frame.index)).astype(str).eq("train")


def _eval_mask(frame: pd.DataFrame) -> pd.Series:
    if "eval_metric_available_flag" in frame.columns:
        flag = _bool_series(frame["eval_metric_available_flag"], frame.index)
        if flag.any():
            return flag
    return frame.get("split", pd.Series("", index=frame.index)).astype(str).isin(["validate", "test"]) & _numeric(
        frame.get("baseline_ndcg@20", pd.Series(np.nan, index=frame.index))
    ).notna()


def _tertile_high(score: pd.Series, reference_mask: pd.Series | None = None) -> pd.Series:
    ref = _numeric(score[reference_mask]) if reference_mask is not None else _numeric(score)
    ref = ref.dropna()
    if ref.empty or ref.nunique() <= 1:
        return _numeric(score).rank(method="average", pct=True).fillna(0.0).ge(2 / 3)
    cut = float(ref.quantile(2 / 3))
    return _numeric(score).ge(cut)


def _load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"raw_asin": str, "asin": str}, low_memory=False)


def _normalize_asin(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "raw_asin" not in out.columns and "asin" in out.columns:
        out = out.rename(columns={"asin": "raw_asin"})
    if "raw_asin" in out.columns:
        out["raw_asin"] = out["raw_asin"].astype(str)
    return out


def build_proxy_profile(task4_profile: pd.DataFrame, r1_profile: pd.DataFrame | None = None, r0_delta_profile: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = _normalize_asin(task4_profile).copy()
    required = {"raw_asin", "split", "eval_baseline_hard_flag", "baseline_ndcg@20", "RSP_score", "RSP_group"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"task4 profile missing columns: {sorted(missing)}")
    train = _train_mask(frame)
    eval_rows = _eval_mask(frame)

    if "support_tail_proxy_score" not in frame.columns:
        frame["support_tail_proxy_score"] = _mean_scores(
            [
                _percentile(frame.get("S_train_support_score", pd.Series(np.nan, index=frame.index)), train, inverse=True),
                _percentile(frame.get("P_popularity_score", pd.Series(np.nan, index=frame.index)), train, inverse=True),
                _percentile(frame.get("S_train_token_user_support_mean", pd.Series(np.nan, index=frame.index)), train, inverse=True),
            ],
            frame.index,
        )
    if "collab_noise_proxy_score" not in frame.columns:
        frame["collab_noise_proxy_score"] = _mean_scores(
            [
                _percentile(frame.get("A_collab_user_set_jaccard_mean", pd.Series(np.nan, index=frame.index)), train, inverse=True),
                _percentile(frame.get("A_collab_support_entropy_mean", pd.Series(np.nan, index=frame.index)), train),
                _percentile(frame.get("A_collab_train_token_user_support_mean", pd.Series(np.nan, index=frame.index)), train, inverse=True),
            ],
            frame.index,
        )
    if "acat_rsp_gap_proxy_score" not in frame.columns:
        frame["acat_rsp_gap_proxy_score"] = _mean_scores(
            [
                _numeric(frame.get("s_cat_v3", pd.Series(0.5, index=frame.index))).fillna(0.5),
                _percentile(frame.get("RSP_score", pd.Series(np.nan, index=frame.index)), train, inverse=True),
            ],
            frame.index,
        )
    if "category_neighbor_mismatch_proxy_score" not in frame.columns:
        frame["category_neighbor_mismatch_proxy_score"] = _mean_scores(
            [
                _numeric(frame.get("s_cat_v3", pd.Series(0.5, index=frame.index))).fillna(0.5),
                _numeric(frame.get("Acat_v3_disc_residual_pct", pd.Series(0.5, index=frame.index))).fillna(0.5),
                _numeric(frame.get("Acat_v3_collab_residual_pct", pd.Series(0.5, index=frame.index))).fillna(0.5),
                _percentile(frame.get("A_collab_user_set_jaccard_mean", pd.Series(np.nan, index=frame.index)), train, inverse=True),
                _percentile(frame.get("A_collab_support_entropy_mean", pd.Series(np.nan, index=frame.index)), train),
            ],
            frame.index,
        )

    frame["competitor_pressure_proxy_score"] = _mean_scores(
        [
            _percentile(frame.get("P_popularity_score", pd.Series(np.nan, index=frame.index)), train),
            _percentile(frame.get("A_collab_train_token_user_support_mean", pd.Series(np.nan, index=frame.index)), train),
            _percentile(frame.get("S_train_token_user_support_mean", pd.Series(np.nan, index=frame.index)), train),
            _percentile(frame.get("S_train_token_interaction_support_mean", pd.Series(np.nan, index=frame.index)), train),
        ],
        frame.index,
    )
    frame["train_graph_near_cutoff_proxy_score"] = _mean_scores(
        [
            _percentile(frame.get("category_count", pd.Series(np.nan, index=frame.index)), train),
            _percentile(frame.get("A_collab_support_entropy_mean", pd.Series(np.nan, index=frame.index)), train),
            _percentile(frame.get("A_collab_user_set_jaccard_mean", pd.Series(np.nan, index=frame.index)), train, inverse=True),
            _percentile(frame.get("S_train_support_score", pd.Series(np.nan, index=frame.index)), train, inverse=True),
        ],
        frame.index,
    )
    frame["sampled_negative_pressure_proxy_score"] = _mean_scores(
        [
            _percentile(frame.get("P_popularity_score", pd.Series(np.nan, index=frame.index)), train),
            _percentile(frame.get("S_train_token_interaction_support_mean", pd.Series(np.nan, index=frame.index)), train),
            _percentile(frame.get("A_collab_train_token_user_support_mean", pd.Series(np.nan, index=frame.index)), train),
        ],
        frame.index,
    )
    frame["residual_acat_pressure_proxy_score"] = _mean_scores(
        [
            _numeric(frame.get("s_cat_v3", pd.Series(0.5, index=frame.index))).fillna(0.5),
            _percentile(frame.get("RSP_score", pd.Series(np.nan, index=frame.index)), train, inverse=True),
            frame["competitor_pressure_proxy_score"],
        ],
        frame.index,
    )

    baseline_margin = frame.get("baseline_margin_proxy", frame.get("margin_to_top20_cutoff", pd.Series(np.nan, index=frame.index)))
    baseline_rank = frame.get("baseline_best_target_rank", frame.get("best_target_rank", pd.Series(np.nan, index=frame.index)))
    frame["eval_margin_proxy_score"] = _mean_scores(
        [
            _percentile(baseline_margin, eval_rows, inverse=True),
            _percentile(frame.get("baseline_ndcg@20", pd.Series(np.nan, index=frame.index)), eval_rows, inverse=True),
        ],
        frame.index,
    )
    rank_numeric = _numeric(baseline_rank)
    near_score = 1.0 / (1.0 + (rank_numeric - 20.0).abs())
    near_score = near_score.where(rank_numeric.ge(20), other=0.0).fillna(0.0)
    frame["eval_near_cutoff_proxy_score"] = _mean_scores([_percentile(near_score, eval_rows)], frame.index)

    for score_col in ALL_SCORE_COLUMNS:
        candidate = _score_name(score_col)
        reference = eval_rows if score_col in DIAGNOSTIC_SCORES else train
        high = _tertile_high(frame[score_col], reference)
        if score_col == "eval_near_cutoff_proxy_score" and "target_rank_near_cutoff" in frame.columns:
            high = _bool_series(frame["target_rank_near_cutoff"], frame.index) | high
        frame[f"{candidate}_high_flag"] = high.map(bool).astype(object)

    if r1_profile is not None and not r1_profile.empty:
        r1 = _normalize_asin(r1_profile)
        keep = ["raw_asin", *[col for col in R1_MASK_COLUMNS if col in r1.columns]]
        frame = frame.merge(r1[keep].drop_duplicates("raw_asin"), on="raw_asin", how="left", validate="many_to_one")
    if r0_delta_profile is not None and not r0_delta_profile.empty:
        r0 = _normalize_asin(r0_delta_profile)
        if "split" in r0.columns:
            r0 = r0[r0["split"].astype(str).eq("test")].copy()
        if "alpha" in r0.columns:
            r0 = r0.sort_values("alpha").drop_duplicates("raw_asin")
        keep = ["raw_asin", *[col for col in ["delta_ndcg@20", "m9_helped_flag", "m9_harmed_flag"] if col in r0.columns]]
        frame = frame.merge(r0[keep], on="raw_asin", how="left", validate="many_to_one", suffixes=("", "_m9"))
    return frame


def build_candidate_summary(proxy_profile: pd.DataFrame) -> pd.DataFrame:
    eval_rows = _eval_mask(proxy_profile)
    hard = _bool_series(proxy_profile["eval_baseline_hard_flag"], proxy_profile.index)
    base_hard_rate = float(hard[eval_rows].mean()) if eval_rows.any() else float("nan")
    rsp_high = proxy_profile.get("RSP_group", pd.Series("", index=proxy_profile.index)).astype(str).eq("RSP_high")
    rows = []
    for score_col in ALL_SCORE_COLUMNS:
        if score_col not in proxy_profile.columns:
            continue
        candidate = _score_name(score_col)
        high = _bool_series(proxy_profile.get(f"{candidate}_high_flag"), proxy_profile.index)
        selected = eval_rows & high
        selected_count = int(selected.sum())
        selected_share = float(selected_count / int(eval_rows.sum())) if eval_rows.any() else float("nan")
        proxy_high_hard_rate = float(hard[selected].mean()) if selected.any() else float("nan")
        hard_rate_lift = proxy_high_hard_rate - base_hard_rate if np.isfinite(proxy_high_hard_rate) and np.isfinite(base_hard_rate) else float("nan")
        spearman_hard = _safe_corr(proxy_profile.loc[eval_rows, score_col], hard.loc[eval_rows].astype(float))
        spearman_ndcg = _safe_corr(proxy_profile.loc[eval_rows, score_col], proxy_profile.loc[eval_rows, "baseline_ndcg@20"])
        spearman_rsp = _safe_corr(proxy_profile[score_col], proxy_profile["RSP_score"]) if "RSP_score" in proxy_profile.columns else float("nan")
        rsp_share = float(rsp_high[selected].mean()) if selected.any() else float("nan")
        passes = bool(
            np.isfinite(hard_rate_lift)
            and hard_rate_lift >= GATE["min_hard_rate_lift"]
            and np.isfinite(spearman_hard)
            and spearman_hard >= GATE["min_spearman_vs_eval_hard_flag"]
            and (not np.isfinite(spearman_rsp) or abs(spearman_rsp) < GATE["max_abs_spearman_vs_rsp"])
            and (not np.isfinite(rsp_share) or rsp_share < GATE["max_proxy_high_rsp_high_share"])
            and selected_count >= GATE["min_selected_count"]
            and (not np.isfinite(selected_share) or selected_share <= GATE["max_selected_share"])
        )
        rows.append(
            {
                "candidate": candidate,
                "score_col": score_col,
                "candidate_type": _candidate_type(score_col),
                "eval_item_count": int(eval_rows.sum()),
                "selected_count": selected_count,
                "selected_share": selected_share,
                "base_hard_rate": base_hard_rate,
                "proxy_high_hard_rate": proxy_high_hard_rate,
                "hard_rate_lift": hard_rate_lift,
                "spearman_vs_eval_baseline_hard_flag": spearman_hard,
                "spearman_vs_baseline_ndcg@20": spearman_ndcg,
                "spearman_vs_RSP_score": spearman_rsp,
                "proxy_high_rsp_high_share": rsp_share,
                "passes_main_gate": passes,
                "gate_min_hard_rate_lift": GATE["min_hard_rate_lift"],
                "gate_min_spearman_vs_eval_hard_flag": GATE["min_spearman_vs_eval_hard_flag"],
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["passes_main_gate"] = summary["passes_main_gate"].map(bool).astype(object)
        summary = summary.sort_values(["passes_main_gate", "candidate_type", "hard_rate_lift"], ascending=[False, True, False]).reset_index(drop=True)
    return summary


def build_control_correlation(proxy_profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rights = [
        ("RSP_score", "control"),
        ("s_cat_v3", "availability"),
        ("S_train_support_score", "control"),
        ("P_popularity_score", "control"),
        ("category_count", "control"),
        ("baseline_ndcg@20", "eval_audit"),
        ("eval_baseline_hard_flag", "eval_audit"),
    ]
    eval_rows = _eval_mask(proxy_profile)
    hard = _bool_series(proxy_profile.get("eval_baseline_hard_flag"), proxy_profile.index).astype(float)
    for score_col in ALL_SCORE_COLUMNS:
        if score_col not in proxy_profile.columns:
            continue
        for right, family in rights:
            if right == "eval_baseline_hard_flag":
                mask = eval_rows
                right_series = hard
            elif right == "baseline_ndcg@20":
                mask = eval_rows
                right_series = proxy_profile[right]
            elif right in proxy_profile.columns:
                mask = pd.Series(True, index=proxy_profile.index)
                right_series = proxy_profile[right]
            else:
                continue
            rows.append(
                {
                    "candidate": _score_name(score_col),
                    "score_col": score_col,
                    "candidate_type": _candidate_type(score_col),
                    "right": right,
                    "right_family": family,
                    "n": int(mask.sum()),
                    "spearman": _safe_corr(proxy_profile.loc[mask, score_col], right_series.loc[mask]),
                    "pearson": _safe_corr(proxy_profile.loc[mask, score_col], right_series.loc[mask], method="pearson"),
                }
            )
    return pd.DataFrame(rows)


def build_r1_overlap_summary(proxy_profile: pd.DataFrame, r1_profile: pd.DataFrame | None = None) -> pd.DataFrame:
    frame = proxy_profile.copy()
    if r1_profile is not None and not r1_profile.empty and not set(R1_MASK_COLUMNS).intersection(frame.columns):
        r1 = _normalize_asin(r1_profile)
        keep = ["raw_asin", *[col for col in R1_MASK_COLUMNS if col in r1.columns]]
        frame = frame.merge(r1[keep].drop_duplicates("raw_asin"), on="raw_asin", how="left", validate="many_to_one")
    eval_rows = _eval_mask(frame)
    rows = []
    for score_col in ALL_SCORE_COLUMNS:
        if score_col not in frame.columns:
            continue
        candidate = _score_name(score_col)
        selected = eval_rows & _bool_series(frame.get(f"{candidate}_high_flag"), frame.index)
        for mask_col in R1_MASK_COLUMNS:
            if mask_col not in frame.columns:
                continue
            r1_mask = eval_rows & _bool_series(frame[mask_col], frame.index)
            overlap = selected & r1_mask
            rows.append(
                {
                    "candidate": candidate,
                    "candidate_type": _candidate_type(score_col),
                    "r1_mask": mask_col,
                    "selected_count": int(selected.sum()),
                    "r1_mask_count": int(r1_mask.sum()),
                    "selected_r1_overlap_count": int(overlap.sum()),
                    "selected_r1_overlap_rate": float(overlap.sum() / selected.sum()) if selected.any() else float("nan"),
                    "r1_capture_rate": float(overlap.sum() / r1_mask.sum()) if r1_mask.any() else float("nan"),
                }
            )
    return pd.DataFrame(rows)


def decide_route(candidate_summary: pd.DataFrame) -> dict[str, Any]:
    if candidate_summary.empty:
        return {
            "route": "r2_inconclusive_missing_inputs",
            "next_action": "rebuild_proxy_inputs",
            "enter_training_now": False,
            "reason": "candidate summary is empty",
        }
    passed = candidate_summary[candidate_summary["passes_main_gate"].map(bool)].copy()
    train_passed = passed[passed["candidate_type"].eq("train_deployable")].copy()
    diagnostic_passed = passed[passed["candidate_type"].eq("diagnostic")].copy()
    if not train_passed.empty:
        selected = train_passed.sort_values(["hard_rate_lift", "spearman_vs_eval_baseline_hard_flag"], ascending=False).iloc[0]
        route = "train_deployable_proxy_ready"
        next_action = "design_r3_r4_carrier_or_seed43_screen_with_controls"
        reason = "at least one train-deployable proxy passes hard-rate lift and correlation gates"
    elif not diagnostic_passed.empty:
        selected = diagnostic_passed.sort_values(["hard_rate_lift", "spearman_vs_eval_baseline_hard_flag"], ascending=False).iloc[0]
        route = "diagnostic_proxy_only_space_exists"
        next_action = "do_not_train_add_train_time_pressure_features_or_go_r4"
        reason = "diagnostic proxy passes but train-deployable proxies do not"
    else:
        selected = candidate_summary.sort_values(["hard_rate_lift", "spearman_vs_eval_baseline_hard_flag"], ascending=False).iloc[0]
        route = "proxy_rebuild_not_enough"
        next_action = "skip_proxy_weighting_prioritize_r3_or_r4_carrier"
        reason = "no proxy passes the discrimination gate"
    return {
        "route": route,
        "next_action": next_action,
        "enter_training_now": False,
        "selected_candidate": str(selected["candidate"]),
        "selected_candidate_type": str(selected["candidate_type"]),
        "selected_hard_rate_lift": _jsonable(selected.get("hard_rate_lift")),
        "selected_spearman_vs_eval_baseline_hard_flag": _jsonable(selected.get("spearman_vs_eval_baseline_hard_flag")),
        "gate": GATE,
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
    candidate_summary: pd.DataFrame,
    control_correlation: pd.DataFrame,
    r1_overlap: pd.DataFrame,
    decision: dict[str, Any],
    manifest_name: str,
) -> None:
    top_candidates = candidate_summary.sort_values(["passes_main_gate", "hard_rate_lift"], ascending=[False, False])
    selected_corr = control_correlation[control_correlation["candidate"].eq(decision.get("selected_candidate", ""))]
    selected_overlap = r1_overlap[r1_overlap["candidate"].eq(decision.get("selected_candidate", ""))]
    content = f"""---
title: {run_stamp} CCFCRec Amazon-VG M10-R2 train-safe hard proxy rebuild 结果
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - M10-R2
  - train_safe_proxy
---

# {run_stamp} CCFCRec Amazon-VG M10-R2 train-safe hard proxy rebuild 结果

## Material Passport

- artifact_type: experiment_diagnostic_result
- project: CCFCRec Amazon-VG category availability
- stage: M10-R2 train-safe hard proxy rebuild
- status: analyzed
- execution_policy: diagnostic-only（仅诊断），no training（不训练）

> [!info] 来源说明
> 上游总设计：[[{TOTAL_DESIGN_NOTE_NAME}]]
> 上游 R2 设计：[[{DESIGN_NOTE_NAME}]]
> R1 结果：[[{R1_RESULT_NOTE_NAME}]]
> 分析脚本：`{ANALYSIS_SCRIPT}`
> manifest（运行清单）：`{manifest_name}`

## 结论

```text
route = {decision["route"]}
next_action = {decision["next_action"]}
selected_candidate = {decision["selected_candidate"]}
enter_training_now = {decision["enter_training_now"]}
```

解释：R2 只审计 proxy（代理）区分度，不训练模型。diagnostic proxy（诊断代理）如果通过，不代表可以作为训练输入；train-deployable proxy（训练可用代理）通过后也仍需 R3/R4 carrier（方法载体）设计和 shuffle/RSP/Acat controls（打乱/控制变量/类别可用性对照）。

## Candidate Summary

> [!info] 字段说明
> `hard_rate_lift`：proxy high（代理高组）困难率减整体基础困难率。
> `spearman_vs_eval_baseline_hard_flag`：proxy 分数与评估困难标记的 Spearman（秩相关）。
> `passes_main_gate`：是否通过 R2 设定的主准入门槛。

{md_table(top_candidates, ["candidate", "candidate_type", "selected_count", "selected_share", "base_hard_rate", "proxy_high_hard_rate", "hard_rate_lift", "spearman_vs_eval_baseline_hard_flag", "spearman_vs_RSP_score", "proxy_high_rsp_high_share", "passes_main_gate"], max_rows=20)}

## Selected Candidate Correlation

{md_table(selected_corr, ["candidate", "candidate_type", "right", "right_family", "n", "spearman", "pearson"], max_rows=20)}

## Selected Candidate R1 Overlap

{md_table(selected_overlap, ["candidate", "r1_mask", "selected_count", "r1_mask_count", "selected_r1_overlap_count", "selected_r1_overlap_rate", "r1_capture_rate"], max_rows=20)}

## Route Decision

```json
{json.dumps(_jsonable(decision), ensure_ascii=False, indent=2)}
```
"""
    path.write_text(content, encoding="utf-8")


def build_outputs(output_root: Path, run_stamp: str) -> Outputs:
    output_dir = output_root / f"{run_stamp} task4-rollback-m10-r2-train-safe-hard-proxy-rebuild"
    return Outputs(
        output_dir=output_dir,
        proxy_profile_csv=output_dir / "m10_r2_proxy_profile.csv",
        candidate_summary_csv=output_dir / "m10_r2_candidate_summary.csv",
        control_correlation_csv=output_dir / "m10_r2_control_correlation.csv",
        r1_overlap_summary_csv=output_dir / "m10_r2_r1_overlap_summary.csv",
        route_decision_json=output_dir / "m10_r2_route_decision.json",
        manifest_json=output_dir / "run_manifest.json",
        result_md=output_dir / f"{run_stamp} CCFCRec Amazon-VG M10-R2 train-safe hard proxy rebuild 结果.md",
    )


def run(args: argparse.Namespace) -> Outputs:
    run_stamp, run_date, run_iso = (args.run_stamp, args.run_stamp[:10], "") if args.run_stamp else now_stamp()
    if args.run_stamp:
        run_iso = datetime.strptime(args.run_stamp, "%Y-%m-%d %H%M%S").isoformat(timespec="seconds")
    outputs = build_outputs(Path(args.output_root).expanduser().resolve(), run_stamp)
    outputs.output_dir.mkdir(parents=True, exist_ok=True)

    task4 = _load_csv(args.task4_profile_path)
    r1 = _load_csv(args.r1_profile_path)
    r0 = _load_csv(args.r0_delta_profile_path)
    profile = build_proxy_profile(task4, r1, r0)
    candidate_summary = build_candidate_summary(profile)
    control_correlation = build_control_correlation(profile)
    r1_overlap = build_r1_overlap_summary(profile)
    decision = decide_route(candidate_summary)

    profile.to_csv(outputs.proxy_profile_csv, index=False)
    candidate_summary.to_csv(outputs.candidate_summary_csv, index=False)
    control_correlation.to_csv(outputs.control_correlation_csv, index=False)
    r1_overlap.to_csv(outputs.r1_overlap_summary_csv, index=False)
    outputs.route_decision_json.write_text(json.dumps(_jsonable(decision), ensure_ascii=False, indent=2), encoding="utf-8")
    write_result_markdown(
        outputs.result_md,
        run_stamp,
        candidate_summary,
        control_correlation,
        r1_overlap,
        decision,
        outputs.manifest_json.name,
    )

    manifest = {
        "run_stamp": run_stamp,
        "run_date": run_date,
        "run_iso": run_iso,
        "experiment_stage": "M10-R2",
        "analysis_script": ANALYSIS_SCRIPT,
        "design_note": DESIGN_NOTE_NAME,
        "total_design_note": TOTAL_DESIGN_NOTE_NAME,
        "diagnostic_only_no_training": True,
        "inputs": {
            "task4_profile_path": str(args.task4_profile_path),
            "r1_profile_path": str(args.r1_profile_path),
            "r0_delta_profile_path": str(args.r0_delta_profile_path),
        },
        "gate": GATE,
        "outputs": {field: str(getattr(outputs, field)) for field in outputs.__dataclass_fields__},
        "decision": decision,
    }
    outputs.manifest_json.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M10-R2 train-safe hard proxy rebuild audit.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--task4-profile-path", default=str(DEFAULT_TASK4_PROFILE))
    parser.add_argument("--r1-profile-path", default=str(DEFAULT_R1_PROFILE))
    parser.add_argument("--r0-delta-profile-path", default=str(DEFAULT_R0_DELTA_PROFILE))
    parser.add_argument("--run-stamp", default="")
    return parser


def main() -> None:
    outputs = run(build_arg_parser().parse_args())
    print(f"output_dir={outputs.output_dir}")
    print(f"result_md={outputs.result_md}")
    print(f"route_decision={outputs.route_decision_json}")


if __name__ == "__main__":
    main()
