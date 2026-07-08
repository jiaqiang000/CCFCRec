#!/usr/bin/env python3
"""
CCFCRec Amazon-VG M10-R0 M9 post audit.

This is a diagnostic-only rollback audit. It does not train models; it reuses
M9 checkpoints to inspect item-level real-minus-shuffle effects.
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
DEFAULT_RESULT_ROOT = Path(
    "/Volumes/MyPassport/CCFCRec对比学习思路硬盘/实验记录硬盘/ccfcrec_result/"
    "2026-07-08_154852_task4_highdetail_qonly_alpha_sweep_m9_seed43_workers8_fast_uniform_mps_100epoch"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "temp_202607_实验文件记录" / "temp_20260709"
DEFAULT_TASK4_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260706"
    / "2026-07-06 004222 task4-pre3-train-safe-hard-proxy"
    / "task4_train_safe_hard_proxy_profile.csv"
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
DEFAULT_BASELINE_VAL_NDCG = 0.12381452117095852
DEFAULT_BASELINE_VAL_HR = 0.02062098905247263
DESIGN_NOTE_NAME = "2026-07-09 010428 CCFCRec Amazon-VG M10-R0 M9 post audit 诊断设计"
TOTAL_DESIGN_NOTE_NAME = "2026-07-09 010147 CCFCRec Amazon-VG Task4-rollback M10-R recoverability and carrier audit 总设计"
ANALYSIS_SCRIPT = "validata/analyze_amazon_vg_task4_rollback_m10_r0_m9_post_audit.py"

DEFAULT_GROUP_COLUMNS = [
    "high_acat_train_safe_hard_flag",
    "high_detail_flag",
    "near_cutoff_group",
    "recoverability_proxy_ensemble_group",
    "failure_consensus_group",
    "rank_recoverability_group",
    "s_cat_v3_group",
    "RSP_group",
    "train_safe_hard_proxy_group",
    "target_history_bucket",
]


@dataclass(frozen=True)
class Outputs:
    output_dir: Path
    validation_curve_summary_csv: Path
    item_eval_profile_csv: Path
    item_delta_profile_csv: Path
    layer_summary_csv: Path
    harmed_group_summary_csv: Path
    acat_correlation_csv: Path
    recoverability_overlap_summary_csv: Path
    route_decision_json: Path
    manifest_json: Path
    result_md: Path


def now_stamp() -> tuple[str, str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H%M%S"), now.strftime("%Y-%m-%d"), now.isoformat(timespec="seconds")


def safe_float(value) -> float:
    if pd.isna(value):
        return float("nan")
    return float(value)


def safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return float("nan")
    return float(numeric.mean())


def safe_sum_positive(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return float(numeric[numeric > 0].sum())


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


def select_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested != "auto":
        return requested
    try:
        import torch
    except ModuleNotFoundError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def bool_text(value) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    return str(value)


def alpha_label(alpha: float) -> str:
    if abs(alpha - 0.75) < 1e-9:
        return "075"
    if abs(alpha - 1.0) < 1e-9:
        return "100"
    return str(alpha).replace(".", "")


def classify_m9_run(method_variant: str, alpha: float, disable_q: bool, disable_self: bool) -> str:
    label = alpha_label(float(alpha))
    if method_variant == "task4_highdetail_trainhard_weight" and disable_self and not disable_q:
        return f"M9a{label}_q_only_real"
    if method_variant == "task4_highdetail_trainhard_shuffle_weight" and disable_self and not disable_q:
        return f"M9a{label}_q_only_shuffle"
    return f"{method_variant}_alpha{alpha}"


def infer_alpha_from_label(label: str) -> float:
    if "a075" in label:
        return 0.75
    if "a100" in label:
        return 1.0
    return float("nan")


def discover_run_dirs(result_root: Path) -> list[Path]:
    return sorted(
        [
            path
            for path in result_root.iterdir()
            if path.is_dir() and (path / "result.csv").exists() and (path / "run_config.json").exists()
        ]
    )


def select_best_checkpoint(result: pd.DataFrame) -> pd.Series:
    return result.sort_values(
        ["ndcg@20", "hr@20", "epoch", "checkpoint_index"],
        ascending=[False, False, True, True],
    ).iloc[0]


def build_best_summary(result_root: Path) -> pd.DataFrame:
    rows = []
    for run_dir in discover_run_dirs(result_root):
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        method = str(config.get("method_variant", ""))
        alpha = float(config.get("task4_loss_alpha", 0.0))
        disable_q = bool(config.get("task4_disable_q_bpr_weight", False))
        disable_self = bool(config.get("task4_disable_self_contrast_weight", False))
        label = classify_m9_run(method, alpha, disable_q, disable_self)
        result = pd.read_csv(run_dir / "result.csv")
        best = select_best_checkpoint(result)
        last = result.sort_values(["epoch", "checkpoint_index"]).iloc[-1]
        checkpoint_path = run_dir / f"{int(best['checkpoint_index'])}.pt"
        rows.append(
            {
                "run_label": label,
                "method_label": label,
                "method_variant": method,
                "run_dir_name": run_dir.name,
                "run_dir": str(run_dir),
                "alpha": alpha,
                "is_shuffle": method == "task4_highdetail_trainhard_shuffle_weight",
                "disable_q_bpr_weight": disable_q,
                "disable_self_contrast_weight": disable_self,
                "best_epoch": int(best["epoch"]),
                "best_checkpoint_index": int(best["checkpoint_index"]),
                "best_checkpoint_path": str(checkpoint_path),
                "best_checkpoint_exists": checkpoint_path.exists(),
                "best_ndcg@20": safe_float(best["ndcg@20"]),
                "best_hr@20": safe_float(best["hr@20"]),
                "last_epoch": int(last["epoch"]),
                "last_checkpoint_index": int(last["checkpoint_index"]),
                "last_ndcg@20": safe_float(last["ndcg@20"]),
                "last_hr@20": safe_float(last["hr@20"]),
                "peak_minus_last_ndcg@20": safe_float(best["ndcg@20"]) - safe_float(last["ndcg@20"]),
                "peak_minus_last_hr@20": safe_float(best["hr@20"]) - safe_float(last["hr@20"]),
                "seed": config.get("seed", ""),
                "num_workers": config.get("num_workers", ""),
                "negative_sampling_mode": config.get("negative_sampling_mode", ""),
            }
        )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["alpha", "is_shuffle", "run_dir_name"]).reset_index(drop=True)
    return summary


def build_validation_curve_summary(
    best_summary: pd.DataFrame,
    baseline_val_ndcg: float = DEFAULT_BASELINE_VAL_NDCG,
    baseline_val_hr: float = DEFAULT_BASELINE_VAL_HR,
) -> pd.DataFrame:
    if best_summary.empty:
        return pd.DataFrame()
    rows = []
    for _, row in best_summary.iterrows():
        rows.append(
            {
                "row_type": "run",
                "run_label": row["run_label"],
                "comparison": "",
                "alpha": float(row["alpha"]),
                "is_shuffle": bool(row["is_shuffle"]),
                "best_epoch": int(row["best_epoch"]),
                "best_checkpoint_index": int(row["best_checkpoint_index"]),
                "best_ndcg@20": float(row["best_ndcg@20"]),
                "best_hr@20": float(row["best_hr@20"]),
                "last_epoch": int(row["last_epoch"]),
                "last_ndcg@20": float(row["last_ndcg@20"]),
                "last_hr@20": float(row["last_hr@20"]),
                "peak_minus_last_ndcg@20": float(row["peak_minus_last_ndcg@20"]),
                "peak_minus_last_hr@20": float(row["peak_minus_last_hr@20"]),
                "delta_ndcg@20": float(row["best_ndcg@20"]) - baseline_val_ndcg,
                "delta_hr@20": float(row["best_hr@20"]) - baseline_val_hr,
                "pct_ndcg@20_vs_validation_baseline": (float(row["best_ndcg@20"]) - baseline_val_ndcg)
                / baseline_val_ndcg
                * 100,
                "pct_hr@20_vs_validation_baseline": (float(row["best_hr@20"]) - baseline_val_hr)
                / baseline_val_hr
                * 100,
            }
        )
    for alpha in sorted(best_summary["alpha"].dropna().unique()):
        real = best_summary[best_summary["alpha"].eq(alpha) & ~best_summary["is_shuffle"]]
        shuffle = best_summary[best_summary["alpha"].eq(alpha) & best_summary["is_shuffle"]]
        if real.empty or shuffle.empty:
            continue
        real_row = real.iloc[0]
        shuffle_row = shuffle.iloc[0]
        delta_ndcg = float(real_row["best_ndcg@20"]) - float(shuffle_row["best_ndcg@20"])
        delta_hr = float(real_row["best_hr@20"]) - float(shuffle_row["best_hr@20"])
        rows.append(
            {
                "row_type": "real_minus_shuffle",
                "run_label": "",
                "comparison": f"M9a{alpha_label(float(alpha))}_real_minus_shuffle",
                "alpha": float(alpha),
                "is_shuffle": "",
                "best_epoch": "",
                "best_checkpoint_index": "",
                "best_ndcg@20": "",
                "best_hr@20": "",
                "last_epoch": "",
                "last_ndcg@20": "",
                "last_hr@20": "",
                "peak_minus_last_ndcg@20": "",
                "peak_minus_last_hr@20": "",
                "delta_ndcg@20": delta_ndcg,
                "delta_hr@20": delta_hr,
                "pct_ndcg@20_vs_validation_baseline": delta_ndcg / baseline_val_ndcg * 100,
                "pct_hr@20_vs_validation_baseline": delta_hr / baseline_val_hr * 100,
            }
        )
    return pd.DataFrame(rows)


def build_item_eval_best_table(best_summary: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "method_variant",
        "method_label",
        "run_label",
        "alpha",
        "is_shuffle",
        "run_dir",
        "best_checkpoint_index",
        "best_checkpoint_path",
    ]
    return best_summary[keep].copy()


def attach_m9_run_metadata(item_eval: pd.DataFrame, best_summary: pd.DataFrame) -> pd.DataFrame:
    if item_eval.empty:
        return item_eval.copy()
    out = item_eval.copy()
    if "run_label" not in out.columns:
        if "method_label" in out.columns:
            out["run_label"] = out["method_label"].astype(str)
        else:
            out["run_label"] = out["method_variant"].astype(str)
    meta_cols = ["run_label", "alpha", "is_shuffle", "run_dir"]
    meta = best_summary[[col for col in meta_cols if col in best_summary.columns]].drop_duplicates("run_label")
    add_cols = [col for col in ["alpha", "is_shuffle", "run_dir"] if col not in out.columns]
    if add_cols:
        out = out.merge(meta[["run_label", *add_cols]], on="run_label", how="left", validate="many_to_one")
    if "alpha" not in out.columns:
        out["alpha"] = out["run_label"].map(infer_alpha_from_label)
    if "is_shuffle" not in out.columns:
        out["is_shuffle"] = out["run_label"].astype(str).str.contains("shuffle")
    out["raw_asin"] = out["raw_asin"].astype(str)
    return out


def build_item_delta_profile(item_eval: pd.DataFrame, min_abs_ndcg_delta: float = 1e-12) -> pd.DataFrame:
    if item_eval.empty:
        return pd.DataFrame()
    required = {"alpha", "is_shuffle", "split", "raw_asin", "ndcg@20", "hr@20"}
    missing = required - set(item_eval.columns)
    if missing:
        raise ValueError(f"item_eval missing columns: {sorted(missing)}")
    keyed = item_eval.copy()
    keyed["raw_asin"] = keyed["raw_asin"].astype(str)
    real = keyed[~keyed["is_shuffle"].map(bool)].copy()
    shuffle = keyed[keyed["is_shuffle"].map(bool)].copy()
    key_cols = ["alpha", "split", "raw_asin"]
    merged = real.merge(shuffle, on=key_cols, how="inner", suffixes=("_real", "_shuffle"), validate="one_to_one")
    rows = []
    metric_cols = [
        "ndcg@20",
        "hr@20",
        "margin_to_top20_cutoff",
        "best_target_rank",
        "q_norm",
        "target_score_max",
        "target_score_mean",
        "target_user_norm_mean",
    ]
    for _, row in merged.iterrows():
        out = {
            "alpha": float(row["alpha"]),
            "split": row["split"],
            "raw_asin": row["raw_asin"],
            "real_run_label": row.get("run_label_real", row.get("method_label_real", "")),
            "shuffle_run_label": row.get("run_label_shuffle", row.get("method_label_shuffle", "")),
        }
        for metric in metric_cols:
            real_col = f"{metric}_real"
            shuffle_col = f"{metric}_shuffle"
            if real_col in merged.columns and shuffle_col in merged.columns:
                real_value = safe_float(row[real_col])
                shuffle_value = safe_float(row[shuffle_col])
                out[f"real_{metric}"] = real_value
                out[f"shuffle_{metric}"] = shuffle_value
                out[f"delta_{metric}"] = real_value - shuffle_value
        delta_ndcg = out.get("delta_ndcg@20", float("nan"))
        out["m9_helped_flag"] = bool(pd.notna(delta_ndcg) and delta_ndcg > min_abs_ndcg_delta)
        out["m9_harmed_flag"] = bool(pd.notna(delta_ndcg) and delta_ndcg < -min_abs_ndcg_delta)
        rows.append(out)
    result = pd.DataFrame(rows)
    for col in ["m9_helped_flag", "m9_harmed_flag"]:
        if col in result:
            result[col] = result[col].map(bool).astype(object)
    if not result.empty:
        result = result.sort_values(["alpha", "split", "raw_asin"]).reset_index(drop=True)
    return result


def _load_profile(profile: pd.DataFrame | Path | str | None) -> pd.DataFrame:
    if profile is None:
        return pd.DataFrame()
    if isinstance(profile, pd.DataFrame):
        return profile.copy()
    path = Path(profile)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype={"raw_asin": str, "asin": str})


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
    derive_near_cutoff: bool = False,
) -> pd.DataFrame:
    prof = _normalize_asin_key(_load_profile(profile))
    if prof.empty or "raw_asin" not in prof.columns:
        return base
    if derive_near_cutoff and "near_cutoff_group" not in prof.columns and "target_rank_near_cutoff" in prof.columns:
        prof["near_cutoff_group"] = prof["target_rank_near_cutoff"].map(
            lambda value: "near_cutoff" if bool(value) else "not_near_cutoff"
        )
    rename = rename or {}
    selected = ["raw_asin", *[col for col in keep_cols if col in prof.columns]]
    if "split" in prof.columns and "split" in base.columns:
        selected.insert(1, "split")
        on_cols = ["raw_asin", "split"]
    else:
        on_cols = ["raw_asin"]
    prof = prof[selected].rename(columns=rename).drop_duplicates(on_cols)
    existing = set(base.columns) - set(on_cols)
    drop_cols = [col for col in prof.columns if col in existing]
    if drop_cols:
        prof = prof.drop(columns=drop_cols)
    return base.merge(prof, on=on_cols, how="left", validate="many_to_one")


def enrich_item_delta_profile(
    delta_profile: pd.DataFrame,
    task4_profile: pd.DataFrame | Path | str | None,
    near_cutoff_profile: pd.DataFrame | Path | str | None,
    recoverability_profile: pd.DataFrame | Path | str | None,
    failure_consensus_profile: pd.DataFrame | Path | str | None,
    rank_recoverability_profile: pd.DataFrame | Path | str | None,
) -> pd.DataFrame:
    enriched = delta_profile.copy()
    if enriched.empty:
        return enriched
    enriched["raw_asin"] = enriched["raw_asin"].astype(str)
    task4_cols = [
        "category_count",
        "cat_count_bin",
        "high_detail_flag",
        "s_cat_v3",
        "s_cat_v3_group",
        "high_acat_flag",
        "train_safe_hard_proxy_score",
        "train_safe_hard_proxy_group",
        "high_acat_train_safe_hard_flag",
        "RSP_score",
        "RSP_group",
        "baseline_ndcg@20",
        "baseline_margin_proxy",
        "baseline_best_target_rank",
        "target_history_bucket",
    ]
    enriched = _merge_profile(enriched, task4_profile, task4_cols)
    if "high_detail_flag" not in enriched.columns and "category_count" in enriched.columns:
        enriched["high_detail_flag"] = pd.to_numeric(enriched["category_count"], errors="coerce").ge(5)
    near_cols = ["target_rank_near_cutoff", "near_cutoff_group", "target_history_bucket", "target_activity_bucket"]
    enriched = _merge_profile(enriched, near_cutoff_profile, near_cols, derive_near_cutoff=True)
    recoverability_cols = ["proxy_ensemble_group", "proxy_ensemble_score", "coverage_score"]
    enriched = _merge_profile(
        enriched,
        recoverability_profile,
        recoverability_cols,
        rename={
            "proxy_ensemble_group": "recoverability_proxy_ensemble_group",
            "proxy_ensemble_score": "recoverability_proxy_ensemble_score",
        },
    )
    failure_cols = ["consensus_group", "consensus_score"]
    enriched = _merge_profile(
        enriched,
        failure_consensus_profile,
        failure_cols,
        rename={"consensus_group": "failure_consensus_group", "consensus_score": "failure_consensus_score"},
    )
    rank_cols = ["rank_aware_group", "competition_aware_group"]
    enriched = _merge_profile(
        enriched,
        rank_recoverability_profile,
        rank_cols,
        rename={
            "rank_aware_group": "rank_recoverability_group",
            "competition_aware_group": "competition_recoverability_group",
        },
    )
    return enriched


def build_layer_summary(item_delta_profile: pd.DataFrame, group_columns: list[str] | None = None) -> pd.DataFrame:
    if item_delta_profile.empty:
        return pd.DataFrame()
    group_columns = DEFAULT_GROUP_COLUMNS if group_columns is None else group_columns
    rows = []

    def append_summary(alpha, split, group_col, group_value, sub: pd.DataFrame) -> None:
        rows.append(
            {
                "alpha": float(alpha),
                "split": split,
                "group_column": group_col,
                "group_value": bool_text(group_value),
                "item_count": int(len(sub)),
                "helped_item_count": int(sub["m9_helped_flag"].map(bool).sum()) if "m9_helped_flag" in sub else 0,
                "harmed_item_count": int(sub["m9_harmed_flag"].map(bool).sum()) if "m9_harmed_flag" in sub else 0,
                "helped_item_share": float(sub["m9_helped_flag"].map(bool).mean()) if "m9_helped_flag" in sub else 0.0,
                "harmed_item_share": float(sub["m9_harmed_flag"].map(bool).mean()) if "m9_harmed_flag" in sub else 0.0,
                "delta_ndcg@20_mean": safe_mean(sub["delta_ndcg@20"]),
                "delta_ndcg@20_sum": safe_float(pd.to_numeric(sub["delta_ndcg@20"], errors="coerce").sum()),
                "positive_delta_ndcg@20_sum": safe_sum_positive(sub["delta_ndcg@20"]),
                "delta_hr@20_mean": safe_mean(sub["delta_hr@20"]) if "delta_hr@20" in sub else float("nan"),
                "delta_margin_to_top20_cutoff_mean": safe_mean(sub["delta_margin_to_top20_cutoff"])
                if "delta_margin_to_top20_cutoff" in sub
                else float("nan"),
                "delta_best_target_rank_mean": safe_mean(sub["delta_best_target_rank"])
                if "delta_best_target_rank" in sub
                else float("nan"),
            }
        )

    for (alpha, split), alpha_split in item_delta_profile.groupby(["alpha", "split"], dropna=False):
        append_summary(alpha, split, "overall", "all", alpha_split)
        for group_col in group_columns:
            if group_col not in alpha_split.columns:
                continue
            for group_value, sub in alpha_split.groupby(group_col, dropna=False):
                append_summary(alpha, split, group_col, group_value, sub)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["alpha", "split", "group_column", "group_value"]).reset_index(drop=True)
    return summary


def build_harmed_group_summary(layer_summary: pd.DataFrame) -> pd.DataFrame:
    if layer_summary.empty:
        return pd.DataFrame()
    harmed = layer_summary[layer_summary["harmed_item_count"].gt(0)].copy()
    if harmed.empty:
        return harmed
    harmed["harmed_pressure_score"] = harmed["harmed_item_share"] * harmed["item_count"]
    return harmed.sort_values(
        ["alpha", "split", "harmed_item_share", "harmed_item_count"],
        ascending=[True, True, False, False],
    ).reset_index(drop=True)


def build_acat_correlation(item_delta_profile: pd.DataFrame) -> pd.DataFrame:
    if item_delta_profile.empty or "s_cat_v3" not in item_delta_profile.columns:
        return pd.DataFrame()
    rows = []
    for (alpha, split), sub in item_delta_profile.groupby(["alpha", "split"], dropna=False):
        for metric in ["delta_ndcg@20", "delta_hr@20", "delta_margin_to_top20_cutoff", "delta_best_target_rank"]:
            if metric not in sub.columns:
                continue
            pair = sub[["s_cat_v3", metric]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 3 or pair["s_cat_v3"].nunique() < 2 or pair[metric].nunique() < 2:
                spearman = float("nan")
                pearson = float("nan")
            else:
                spearman = safe_float(pair["s_cat_v3"].corr(pair[metric], method="spearman"))
                pearson = safe_float(pair["s_cat_v3"].corr(pair[metric], method="pearson"))
            rows.append(
                {
                    "alpha": float(alpha),
                    "split": split,
                    "metric": metric,
                    "item_count": int(len(pair)),
                    "spearman_with_s_cat_v3": spearman,
                    "pearson_with_s_cat_v3": pearson,
                }
            )
    return pd.DataFrame(rows)


def build_recoverability_overlap_summary(item_delta_profile: pd.DataFrame) -> pd.DataFrame:
    if item_delta_profile.empty:
        return pd.DataFrame()
    group_cols = [
        col
        for col in ["near_cutoff_group", "recoverability_proxy_ensemble_group", "failure_consensus_group", "rank_recoverability_group"]
        if col in item_delta_profile.columns
    ]
    if not group_cols:
        return pd.DataFrame()
    rows = []
    for alpha, alpha_sub in item_delta_profile.groupby("alpha", dropna=False):
        for group_col in group_cols:
            for group_value, sub in alpha_sub.groupby(group_col, dropna=False):
                rows.append(
                    {
                        "alpha": float(alpha),
                        "group_column": group_col,
                        "group_value": bool_text(group_value),
                        "item_count": int(len(sub)),
                        "helped_item_count": int(sub["m9_helped_flag"].map(bool).sum()) if "m9_helped_flag" in sub else 0,
                        "harmed_item_count": int(sub["m9_harmed_flag"].map(bool).sum()) if "m9_harmed_flag" in sub else 0,
                        "delta_ndcg@20_mean": safe_mean(sub["delta_ndcg@20"]),
                        "positive_delta_ndcg@20_sum": safe_sum_positive(sub["delta_ndcg@20"]),
                    }
                )
    return pd.DataFrame(rows)


def _dominant_positive_group(
    layer_summary: pd.DataFrame,
    group_column: str,
    positive_values: set[str],
    min_positive_share: float,
) -> bool:
    if layer_summary.empty:
        return False
    sub = layer_summary[layer_summary["group_column"].eq(group_column)].copy()
    overall = layer_summary[layer_summary["group_column"].eq("overall")].copy()
    if sub.empty or overall.empty:
        return False
    sub["is_positive_value"] = sub["group_value"].astype(str).isin(positive_values)
    key_cols = ["alpha", "split"]
    for key, overall_slice in overall.groupby(key_cols, dropna=False):
        total_positive = safe_float(overall_slice.iloc[0]["positive_delta_ndcg@20_sum"])
        if total_positive <= 0:
            continue
        focused = sub[sub["is_positive_value"]]
        for key_col, key_value in zip(key_cols, key):
            focused = focused[focused[key_col].eq(key_value)]
        if focused.empty:
            continue
        focused_positive = safe_float(pd.to_numeric(focused["positive_delta_ndcg@20_sum"], errors="coerce").sum())
        focused_mean = safe_mean(focused["delta_ndcg@20_mean"])
        if focused_positive / total_positive >= min_positive_share and pd.notna(focused_mean) and focused_mean > 0:
            return True
    return False


def _max_abs_acat_spearman(acat_correlation: pd.DataFrame) -> float:
    if acat_correlation.empty or "spearman_with_s_cat_v3" not in acat_correlation.columns:
        return float("nan")
    ndcg = acat_correlation[acat_correlation["metric"].eq("delta_ndcg@20")]
    values = pd.to_numeric(ndcg["spearman_with_s_cat_v3"], errors="coerce").abs().dropna()
    if values.empty:
        return float("nan")
    return float(values.max())


def build_route_decision(
    item_delta_profile: pd.DataFrame,
    layer_summary: pd.DataFrame,
    acat_correlation: pd.DataFrame,
    min_helped_share: float = 0.20,
    min_harmed_share: float = 0.20,
    min_dominant_positive_share: float = 0.60,
    min_abs_acat_spearman: float = 0.10,
) -> dict:
    if item_delta_profile.empty or "delta_ndcg@20" not in item_delta_profile.columns:
        return {
            "route": "m9_item_level_inconclusive",
            "next_action": "rebuild_item_level_audit_before_r1",
            "run_m9_style_m10": False,
            "run_multi_seed_now": False,
            "reason": "missing item-level real-minus-shuffle delta",
        }
    helped_share = float(item_delta_profile["m9_helped_flag"].map(bool).mean()) if "m9_helped_flag" in item_delta_profile else 0.0
    harmed_share = float(item_delta_profile["m9_harmed_flag"].map(bool).mean()) if "m9_harmed_flag" in item_delta_profile else 0.0
    max_abs_acat = _max_abs_acat_spearman(acat_correlation)
    if helped_share >= min_helped_share and harmed_share >= min_harmed_share:
        route = "m9_tradeoff_requires_rollback"
        next_action = "design_gated_or_competitor_calibrated_carrier"
        reason = "helped and harmed item groups are both non-trivial"
    elif _dominant_positive_group(
        layer_summary,
        "near_cutoff_group",
        {"near_cutoff", "True", "true", "1"},
        min_dominant_positive_share,
    ):
        route = "m9_signal_near_cutoff_only"
        next_action = "keep_near_cutoff_as_r1_recoverability_upper_bound_layer"
        reason = "positive real-minus-shuffle effect is concentrated near the top20 boundary"
    elif _dominant_positive_group(
        layer_summary,
        "high_acat_train_safe_hard_flag",
        {"True", "true", "1"},
        min_dominant_positive_share,
    ):
        route = "m9_signal_highdetail_acat_layer_specific"
        next_action = "audit_this_layer_in_r1_and_preserve_for_r2_proxy_rebuild"
        reason = "positive real-minus-shuffle effect is concentrated in high Acat train-safe hard layer"
    elif math.isnan(max_abs_acat) or max_abs_acat < min_abs_acat_spearman:
        route = "m9_signal_not_acat_aligned"
        next_action = "do_not_continue_acat_only_loss_weight"
        reason = "delta_ndcg@20 has weak or unavailable continuous alignment with s_cat_v3"
    else:
        route = "m9_signal_not_acat_aligned"
        next_action = "use_recoverability_r1_before_new_training"
        reason = "no dominant Acat layer or near-cutoff layer explains enough positive effect"
    return {
        "route": route,
        "next_action": next_action,
        "run_m9_style_m10": False,
        "run_multi_seed_now": False,
        "helped_item_share": helped_share,
        "harmed_item_share": harmed_share,
        "max_abs_spearman_delta_ndcg_with_s_cat_v3": max_abs_acat,
        "min_helped_share": min_helped_share,
        "min_harmed_share": min_harmed_share,
        "min_dominant_positive_share": min_dominant_positive_share,
        "min_abs_acat_spearman": min_abs_acat_spearman,
        "reason": reason,
    }


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    result_root: Path,
    validation_curve: pd.DataFrame,
    layer_summary: pd.DataFrame,
    harmed_summary: pd.DataFrame,
    acat_correlation: pd.DataFrame,
    decision: dict,
    manifest_name: str,
) -> None:
    run_rows = validation_curve[validation_curve["row_type"].eq("run")] if "row_type" in validation_curve else validation_curve
    pair_rows = validation_curve[validation_curve["row_type"].eq("real_minus_shuffle")] if "row_type" in validation_curve else pd.DataFrame()
    top_layers = (
        layer_summary[layer_summary["group_column"].ne("overall") & layer_summary["group_value"].astype(str).ne("NA")]
        .sort_values("positive_delta_ndcg@20_sum", ascending=False)
        .head(12)
        if not layer_summary.empty
        else pd.DataFrame()
    )
    harmed_top = harmed_summary.head(12) if not harmed_summary.empty else pd.DataFrame()
    acat_ndcg = (
        acat_correlation[acat_correlation["metric"].eq("delta_ndcg@20")]
        if not acat_correlation.empty and "metric" in acat_correlation
        else pd.DataFrame()
    )
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG M10-R0 M9 post audit 结果
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - M10-R0
  - post_audit
---

# {run_stamp} CCFCRec Amazon-VG M10-R0 M9 post audit 结果

## Material Passport

- artifact_type: experiment_diagnostic_result
- project: CCFCRec Amazon-VG category availability
- stage: M10-R0 M9 post audit
- status: analyzed
- execution_policy: diagnostic-only（仅诊断），no training（不训练）

> [!info] 来源说明
> 上游总设计：[[{TOTAL_DESIGN_NOTE_NAME}]]
> 上游诊断设计：[[{DESIGN_NOTE_NAME}]]
> 训练产物来源：`{result_root}`
> 分析脚本：`{ANALYSIS_SCRIPT}`
> manifest（运行清单）：`{manifest_name}`

## 结论

```text
route = {decision["route"]}
next_action = {decision["next_action"]}
run_m9_style_m10 = {decision["run_m9_style_m10"]}
run_multi_seed_now = {decision["run_multi_seed_now"]}
```

解释：本轮只审计 M9（第九组候选）的 real-shuffle（真实目标减打乱负控）逐物品差异，不训练新模型。route（路线）判断用于决定 R1 recoverability upper-bound（可恢复性上界审计）如何分层，不支持直接进入 M9-style M10（沿 M9 小改的第十组候选）。

## Validation 曲线

> [!info] 字段说明
> `run_label`：M9 运行标签。
> `best_epoch`：按 validation NDCG@20（验证集前20排序质量）选择的最佳轮次。
> `peak_minus_last_ndcg@20`：最佳轮次减末轮的 NDCG@20，表示后段衰减。
> `pct_ndcg@20_vs_validation_baseline`：相对同协议 validation baseline（验证集基线）的百分比。

{md_table(run_rows, ["run_label", "alpha", "best_epoch", "best_ndcg@20", "best_hr@20", "last_ndcg@20", "peak_minus_last_ndcg@20", "pct_ndcg@20_vs_validation_baseline"], max_rows=12) if not run_rows.empty else "_empty_"}

## Real Minus Shuffle

> [!info] 字段说明
> `delta_ndcg@20`：real（真实目标）减 shuffle（打乱负控）的 NDCG@20。
> `pct_ndcg@20_vs_validation_baseline`：净差值相对 validation baseline（验证集基线）的百分比。

{md_table(pair_rows, ["comparison", "alpha", "delta_ndcg@20", "delta_hr@20", "pct_ndcg@20_vs_validation_baseline", "pct_hr@20_vs_validation_baseline"], max_rows=12) if not pair_rows.empty else "_empty_"}

## 分层正向来源

> [!info] 字段说明
> `group_column`：分层字段。
> `group_value`：分层取值。
> `positive_delta_ndcg@20_sum`：该层内正向 real-shuffle NDCG@20 差值之和。
> `harmed_item_count`：该层内 real-shuffle NDCG@20 为负的 item（物品）数量。

{md_table(top_layers, ["alpha", "split", "group_column", "group_value", "item_count", "helped_item_count", "harmed_item_count", "delta_ndcg@20_mean", "positive_delta_ndcg@20_sum"], max_rows=12) if not top_layers.empty else "_item-level（逐物品）分层尚未生成。_"}

## Harmed Group

> [!info] 字段说明
> `harmed_item_share`：该层内受损 item（物品）占比。
> `harmed_pressure_score`：受损占比乘样本量，用于排序查看明显受损层。

{md_table(harmed_top, ["alpha", "split", "group_column", "group_value", "item_count", "harmed_item_count", "harmed_item_share", "delta_ndcg@20_mean"], max_rows=12) if not harmed_top.empty else "_未发现 harmed group（受损组），或 item-level（逐物品）分层尚未生成。_"}

## Acat 相关性

> [!info] 字段说明
> `spearman_with_s_cat_v3`：delta（差值）与 Acat_v3 连续值的 Spearman（秩相关）。
> `pearson_with_s_cat_v3`：delta（差值）与 Acat_v3 连续值的 Pearson（线性相关）。

{md_table(acat_ndcg, ["alpha", "split", "metric", "item_count", "spearman_with_s_cat_v3", "pearson_with_s_cat_v3"], max_rows=8) if not acat_ndcg.empty else "_Acat_v3（第三版类别可用性）相关性尚不可判定。_"}

## Route Decision

```json
{json.dumps(decision, ensure_ascii=False, indent=2)}
```
"""
    output_path.write_text(markdown, encoding="utf-8")


def build_outputs(output_root: Path, run_stamp: str) -> Outputs:
    output_dir = output_root / f"{run_stamp} task4-rollback-m10-r0-m9-post-audit"
    return Outputs(
        output_dir=output_dir,
        validation_curve_summary_csv=output_dir / "m10_r0_m9_validation_curve_summary.csv",
        item_eval_profile_csv=output_dir / "m10_r0_m9_item_eval_profile.csv",
        item_delta_profile_csv=output_dir / "m10_r0_m9_item_delta_profile.csv",
        layer_summary_csv=output_dir / "m10_r0_m9_layer_summary.csv",
        harmed_group_summary_csv=output_dir / "m10_r0_m9_harmed_group_summary.csv",
        acat_correlation_csv=output_dir / "m10_r0_m9_acat_correlation.csv",
        recoverability_overlap_summary_csv=output_dir / "m10_r0_m9_recoverability_overlap_summary.csv",
        route_decision_json=output_dir / "m10_r0_route_decision.json",
        manifest_json=output_dir / "run_manifest.json",
        result_md=output_dir / f"{run_stamp} CCFCRec Amazon-VG M10-R0 M9 post audit 结果.md",
    )


def run(args: argparse.Namespace) -> Outputs:
    run_stamp, run_date, run_iso = (args.run_stamp, args.run_stamp[:10], "") if args.run_stamp else now_stamp()
    if args.run_stamp:
        run_iso = datetime.strptime(args.run_stamp, "%Y-%m-%d %H%M%S").isoformat(timespec="seconds")
    outputs = build_outputs(Path(args.output_root).expanduser().resolve(), run_stamp)
    outputs.output_dir.mkdir(parents=True, exist_ok=True)

    result_root = Path(args.result_root).expanduser().resolve()
    best_summary = build_best_summary(result_root)
    validation_curve = build_validation_curve_summary(best_summary, args.baseline_val_ndcg, args.baseline_val_hr)

    item_eval_profile = pd.DataFrame()
    item_delta = pd.DataFrame()
    layer_summary = pd.DataFrame()
    harmed_summary = pd.DataFrame()
    acat_correlation = pd.DataFrame()
    recoverability_overlap = pd.DataFrame()

    device_name = select_device(args.device)
    if args.reuse_item_eval_csv:
        item_eval_profile = pd.read_csv(args.reuse_item_eval_csv, dtype={"raw_asin": str})
    elif not args.skip_item_eval:
        from analyze_amazon_vg_task4_post import evaluate_best_checkpoints_item_level

        code_root = Path(args.code_root).expanduser().resolve() if args.code_root else Path(__file__).resolve().parents[1]
        data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else code_root / "Amazon VG" / "data"
        best_table = build_item_eval_best_table(best_summary)
        raw_item_eval = evaluate_best_checkpoints_item_level(
            best_table=best_table,
            code_root=code_root,
            data_dir=data_dir,
            device_name=device_name,
            batch_size=args.item_eval_batch_size,
        )
        item_eval_profile = attach_m9_run_metadata(raw_item_eval, best_summary)

    if not item_eval_profile.empty:
        if "alpha" not in item_eval_profile.columns or "is_shuffle" not in item_eval_profile.columns:
            item_eval_profile = attach_m9_run_metadata(item_eval_profile, best_summary)
        item_delta = build_item_delta_profile(item_eval_profile)
        item_delta = enrich_item_delta_profile(
            item_delta,
            args.task4_profile_path,
            args.near_cutoff_profile_path,
            args.recoverability_profile_path,
            args.failure_consensus_profile_path,
            args.rank_recoverability_profile_path,
        )
        layer_summary = build_layer_summary(item_delta)
        harmed_summary = build_harmed_group_summary(layer_summary)
        acat_correlation = build_acat_correlation(item_delta)
        recoverability_overlap = build_recoverability_overlap_summary(item_delta)

    decision = build_route_decision(item_delta, layer_summary, acat_correlation)

    validation_curve.to_csv(outputs.validation_curve_summary_csv, index=False)
    item_eval_profile.to_csv(outputs.item_eval_profile_csv, index=False)
    item_delta.to_csv(outputs.item_delta_profile_csv, index=False)
    layer_summary.to_csv(outputs.layer_summary_csv, index=False)
    harmed_summary.to_csv(outputs.harmed_group_summary_csv, index=False)
    acat_correlation.to_csv(outputs.acat_correlation_csv, index=False)
    recoverability_overlap.to_csv(outputs.recoverability_overlap_summary_csv, index=False)
    outputs.route_decision_json.write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    write_result_markdown(
        output_path=outputs.result_md,
        run_stamp=run_stamp,
        result_root=result_root,
        validation_curve=validation_curve,
        layer_summary=layer_summary,
        harmed_summary=harmed_summary,
        acat_correlation=acat_correlation,
        decision=decision,
        manifest_name=outputs.manifest_json.name,
    )

    manifest = {
        "run_stamp": run_stamp,
        "run_date": run_date,
        "run_iso": run_iso,
        "experiment_stage": "M10-R0",
        "analysis_script": ANALYSIS_SCRIPT,
        "design_note": DESIGN_NOTE_NAME,
        "total_design_note": TOTAL_DESIGN_NOTE_NAME,
        "diagnostic_only_no_training": True,
        "inputs": {
            "result_root": str(result_root),
            "task4_profile_path": str(args.task4_profile_path),
            "near_cutoff_profile_path": str(args.near_cutoff_profile_path),
            "recoverability_profile_path": str(args.recoverability_profile_path),
            "failure_consensus_profile_path": str(args.failure_consensus_profile_path),
            "rank_recoverability_profile_path": str(args.rank_recoverability_profile_path),
            "reuse_item_eval_csv": str(args.reuse_item_eval_csv) if args.reuse_item_eval_csv else "",
        },
        "parameters": {
            "baseline_val_ndcg": args.baseline_val_ndcg,
            "baseline_val_hr": args.baseline_val_hr,
            "device_requested": args.device,
            "device_used_for_item_eval": device_name if not args.skip_item_eval else "skipped",
            "skip_item_eval": bool(args.skip_item_eval),
            "item_eval_batch_size": args.item_eval_batch_size,
        },
        "outputs": {field: str(getattr(outputs, field)) for field in outputs.__dataclass_fields__},
        "decision": decision,
    }
    outputs.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run M10-R0 M9 post audit.")
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--task4-profile-path", default=str(DEFAULT_TASK4_PROFILE))
    parser.add_argument("--near-cutoff-profile-path", default=str(DEFAULT_NEAR_CUTOFF_PROFILE))
    parser.add_argument("--recoverability-profile-path", default=str(DEFAULT_RECOVERABILITY_PROFILE))
    parser.add_argument("--failure-consensus-profile-path", default=str(DEFAULT_FAILURE_PROFILE))
    parser.add_argument("--rank-recoverability-profile-path", default=str(DEFAULT_RANK_PROFILE))
    parser.add_argument("--code-root", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--item-eval-batch-size", type=int, default=256)
    parser.add_argument("--baseline-val-ndcg", type=float, default=DEFAULT_BASELINE_VAL_NDCG)
    parser.add_argument("--baseline-val-hr", type=float, default=DEFAULT_BASELINE_VAL_HR)
    parser.add_argument("--reuse-item-eval-csv", default="")
    parser.add_argument("--skip-item-eval", action="store_true")
    parser.add_argument("--run-stamp", default="")
    return parser


def main() -> None:
    outputs = run(build_arg_parser().parse_args())
    print(f"output_dir={outputs.output_dir}")
    print(f"result_md={outputs.result_md}")
    print(f"route_decision={outputs.route_decision_json}")


if __name__ == "__main__":
    main()
