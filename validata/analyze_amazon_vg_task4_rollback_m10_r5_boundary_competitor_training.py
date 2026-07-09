#!/usr/bin/env python3
"""
CCFCRec Amazon-VG M10-R5 boundary competitor training analysis.

Parses the four seed43 100epoch branches, compares real/shuffle/RSP/Acat
controls, and applies the pre-registered Task4 route gates.
"""

from __future__ import annotations

import argparse
import json
import math
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path("/Users/luojiaqiang/Documents/Obsidian Vault/科研/CCFCRec对比学习思路")
CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "temp_202607_实验文件记录" / "temp_20260709"
DEFAULT_RESULT_ROOT = Path(
    "/Volumes/MyPassport/CCFCRec对比学习思路硬盘/实验记录硬盘/ccfcrec_result/"
    "2026-07-09_124756_task4_boundary_competitor_m10r5_seed43_workers8_fast_uniform_mps_100epoch"
)
DEFAULT_VALIDATE_CSV = CODE_ROOT / "Amazon VG" / "data" / "validate_rating.csv"
DEFAULT_TRAIN_CSV = CODE_ROOT / "Amazon VG" / "data" / "train_rating.csv"
DEFAULT_BASELINE_RESULT = (
    PROJECT_ROOT
    / "实验记录"
    / "2026-06-27 162750 baseline-seed43-workers8-fast-uniform-commit873a171"
    / "baseline_seed43_workers8_fast_uniform_commit873a171_2026-06-27_15_54_28_slim_best74_last100.tar.gz"
)
ANALYSIS_SCRIPT = "validata/analyze_amazon_vg_task4_rollback_m10_r5_boundary_competitor_training.py"
TOTAL_DESIGN_NOTE_NAME = "2026-07-09 010147 CCFCRec Amazon-VG Task4-rollback M10-R recoverability and carrier audit 总设计"
R5_DESIGN_NOTE_NAME = "2026-07-09 130000 CCFCRec Amazon-VG M10-R5 boundary competitor sampling 代码阅读与离线审计设计"
R5_AUDIT_ROUTE_NOTE_NAME = "2026-07-09 123230 CCFCRec Amazon-VG M10-R5 boundary competitor offline audit 路线判断"
R4_TRAINING_ROUTE_NOTE_NAME = "2026-07-09 120712 CCFCRec Amazon-VG M10-R4 competitor pair training 路线判断"

M9_REAL_SHUFFLE_GATE_ABS = 0.000647
TARGET_NET_GATE_ABS = 0.0015
EARLY_TRANSIENT_EPOCH_RATIO = 0.20
R4_REAL_BEST_NDCG_AT_20 = 0.1234194167381099

ROLE_BY_METHOD_VARIANT = {
    "task4_boundary_competitor_pair": "real",
    "task4_boundary_competitor_pair_shuffle": "shuffle",
    "task4_boundary_competitor_pair_rsp_control": "rsp_control",
    "task4_boundary_competitor_pair_acat_control": "acat_control",
}
CONTROL_ROLES = ["shuffle", "rsp_control", "acat_control"]
REQUIRED_ROLES = ["real", *CONTROL_ROLES]


@dataclass(frozen=True)
class Outputs:
    output_dir: Path
    validation_curve_csv: Path
    validation_summary_csv: Path
    pair_comparison_csv: Path
    baseline_comparison_csv: Path
    baseline_summary_json: Path
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


def _round_float(value: Any, digits: int = 12) -> float:
    if value is None or pd.isna(value):
        return float("nan")
    return round(float(value), digits)


def _read_manifest_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    manifest: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        manifest[key.strip()] = value.strip()
    return manifest


def _run_dirs(result_root: Path) -> list[Path]:
    if not result_root.exists():
        raise FileNotFoundError(f"result root does not exist: {result_root}")
    return sorted(
        path
        for path in result_root.iterdir()
        if path.is_dir() and path.name != "logs" and (path / "result.csv").exists() and (path / "run_config.json").exists()
    )


def _role_from_variant(method_variant: str) -> str:
    role = ROLE_BY_METHOD_VARIANT.get(method_variant)
    if role is None:
        raise ValueError(f"unknown R5 method_variant: {method_variant}")
    return role


def _alpha_tag(alpha: Any) -> str:
    if alpha is None or pd.isna(alpha):
        return "aNA"
    return f"a{int(round(float(alpha) * 100)):03d}"


def _run_label(role: str, alpha: Any) -> str:
    return f"R5{_alpha_tag(alpha)}_{role}"


def build_validation_curve(result_root: Path | str) -> pd.DataFrame:
    root = Path(result_root).expanduser()
    manifest = _read_manifest_env(root / "launcher_manifest.env")
    rows: list[pd.DataFrame] = []
    for run_dir in _run_dirs(root):
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        method_variant = str(config.get("method_variant", ""))
        role = _role_from_variant(method_variant)
        alpha = config.get("task4_competitor_alpha")
        result = pd.read_csv(run_dir / "result.csv")
        result.insert(0, "result_root", str(root))
        result.insert(1, "run_dir", run_dir.name)
        result.insert(2, "run_label", _run_label(role, alpha))
        result.insert(3, "role", role)
        result.insert(4, "method_variant", method_variant)
        result.insert(5, "alpha", alpha)
        result.insert(6, "margin", config.get("task4_competitor_margin"))
        result.insert(7, "competitor_k", config.get("task4_competitor_k"))
        result.insert(8, "seed", config.get("seed", manifest.get("SEED")))
        result.insert(9, "num_workers", config.get("num_workers", manifest.get("NUM_WORKERS")))
        result.insert(10, "negative_sampling_mode", config.get("negative_sampling_mode", manifest.get("NEGATIVE_SAMPLING_MODE")))
        result.insert(11, "expected_epoch", config.get("epoch", manifest.get("EPOCH")))
        result.insert(12, "batch_size", config.get("batch_size", manifest.get("BATCH_SIZE")))
        rows.append(result)
    if not rows:
        raise ValueError(f"no R5 result.csv/run_config.json pairs found under: {root}")
    curve = pd.concat(rows, ignore_index=True)
    for col in ["alpha", "margin", "competitor_k", "seed", "num_workers", "expected_epoch", "batch_size"]:
        if col in curve.columns:
            curve[col] = pd.to_numeric(curve[col], errors="coerce")
    return curve


def _best_row(frame: pd.DataFrame, metric: str) -> pd.Series:
    values = pd.to_numeric(frame[metric], errors="coerce")
    if values.isna().all():
        raise ValueError(f"metric has no numeric values: {metric}")
    return frame.loc[values.idxmax()]


def build_validation_summary(curve: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "run_label",
        "role",
        "method_variant",
        "alpha",
        "margin",
        "competitor_k",
        "seed",
        "num_workers",
        "negative_sampling_mode",
        "expected_epoch",
        "batch_size",
        "run_dir",
    ]
    rows: list[dict[str, Any]] = []
    for keys, sub in curve.groupby(group_cols, dropna=False, sort=False):
        meta = dict(zip(group_cols, keys))
        sub = sub.sort_values(["epoch", "checkpoint_index"]).reset_index(drop=True)
        best_ndcg = _best_row(sub, "ndcg@20")
        best_hr = _best_row(sub, "hr@20")
        last = sub.iloc[-1]
        max_epoch = int(pd.to_numeric(sub["epoch"], errors="coerce").max())
        expected_epoch = meta.get("expected_epoch")
        expected_epoch = None if pd.isna(expected_epoch) else int(expected_epoch)
        best_ndcg_value = _round_float(best_ndcg["ndcg@20"])
        last_ndcg_value = _round_float(last["ndcg@20"])
        rows.append(
            {
                **meta,
                "rows": int(len(sub)),
                "min_epoch": int(pd.to_numeric(sub["epoch"], errors="coerce").min()),
                "max_epoch": max_epoch,
                "completed_expected_epoch": bool(expected_epoch is None or max_epoch >= expected_epoch),
                "best_ndcg@20": best_ndcg_value,
                "best_ndcg_epoch": int(best_ndcg["epoch"]),
                "best_ndcg_checkpoint_index": int(best_ndcg["checkpoint_index"]),
                "best_hr@20_at_best_ndcg": _round_float(best_ndcg["hr@20"]),
                "best_hr@20": _round_float(best_hr["hr@20"]),
                "best_hr_epoch": int(best_hr["epoch"]),
                "ndcg@20_at_best_hr": _round_float(best_hr["ndcg@20"]),
                "last_ndcg@20": last_ndcg_value,
                "last_hr@20": _round_float(last["hr@20"]),
                "best_ndcg_peak_minus_last": _round_float(best_ndcg_value - last_ndcg_value),
                "best_ndcg_epoch_ratio": _round_float(float(best_ndcg["epoch"]) / max(max_epoch, 1)),
                "early_transient_peak_flag": bool(float(best_ndcg["epoch"]) <= max(max_epoch, 1) * EARLY_TRANSIENT_EPOCH_RATIO),
            }
        )
    summary = pd.DataFrame(rows)
    for col in ["completed_expected_epoch", "early_transient_peak_flag"]:
        summary[col] = summary[col].map(bool).astype(object)
    return summary.sort_values(["alpha", "role"]).reset_index(drop=True)


def _summary_role_row(summary: pd.DataFrame, role: str) -> pd.Series | None:
    rows = summary[summary["role"].eq(role)]
    if rows.empty:
        return None
    if len(rows) > 1:
        rows = rows.sort_values("best_ndcg@20", ascending=False)
    return rows.iloc[0]


def build_pair_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    real = _summary_role_row(summary, "real")
    if real is None:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for role in CONTROL_ROLES:
        other = _summary_role_row(summary, role)
        if other is None:
            continue
        delta_ndcg = _round_float(real["best_ndcg@20"] - other["best_ndcg@20"])
        delta_hr = _round_float(real["best_hr@20"] - other["best_hr@20"])
        delta_last_ndcg = _round_float(real["last_ndcg@20"] - other["last_ndcg@20"])
        rows.append(
            {
                "comparison": f"real_minus_{role}",
                "left_role": "real",
                "right_role": role,
                "alpha": real.get("alpha"),
                "left_best_ndcg@20": _round_float(real["best_ndcg@20"]),
                "right_best_ndcg@20": _round_float(other["best_ndcg@20"]),
                "delta_best_ndcg@20": delta_ndcg,
                "delta_best_ndcg_pct_vs_real": _round_float(delta_ndcg / max(float(real["best_ndcg@20"]), 1e-12) * 100.0),
                "left_best_hr@20": _round_float(real["best_hr@20"]),
                "right_best_hr@20": _round_float(other["best_hr@20"]),
                "delta_best_hr@20": delta_hr,
                "left_last_ndcg@20": _round_float(real["last_ndcg@20"]),
                "right_last_ndcg@20": _round_float(other["last_ndcg@20"]),
                "delta_last_ndcg@20": delta_last_ndcg,
                "left_best_epoch": int(real["best_ndcg_epoch"]),
                "right_best_epoch": int(other["best_ndcg_epoch"]),
                "passes_m9_net_gate": bool(role == "shuffle" and delta_ndcg > M9_REAL_SHUFFLE_GATE_ABS),
                "passes_target_net_gate": bool(role == "shuffle" and delta_ndcg >= TARGET_NET_GATE_ABS),
                "not_losing_to_control": bool(delta_ndcg >= 0),
            }
        )
    comparison = pd.DataFrame(rows)
    for col in ["passes_m9_net_gate", "passes_target_net_gate", "not_losing_to_control"]:
        if col in comparison.columns:
            comparison[col] = comparison[col].map(bool).astype(object)
    return comparison


def _read_baseline_result(path: Path) -> pd.DataFrame:
    if path.is_dir():
        return pd.read_csv(path / "result.csv")
    if path.name.endswith(".tar.gz"):
        with tarfile.open(path, "r:gz") as tar:
            matches = [member for member in tar.getmembers() if member.name.endswith("/result.csv")]
            if len(matches) != 1:
                raise ValueError(f"expected exactly one result.csv in baseline package, got {len(matches)}")
            with tar.extractfile(matches[0]) as file_obj:
                if file_obj is None:
                    raise ValueError(f"cannot read baseline result member: {matches[0].name}")
                return pd.read_csv(file_obj)
    return pd.read_csv(path)


def build_metric_scope_summary(validate_csv: Path | str | None, train_csv: Path | str | None) -> dict[str, Any]:
    scope = {
        "evaluation_split": "validation",
        "metric_scope": "validation_split_all_unique_cold_start_items",
        "metric_scope_explanation": (
            "Macro-average over every unique asin in Amazon VG/data/validate_rating.csv; "
            "no test split, no training-item/full-catalog metric, and no Task4 subgroup filter."
        ),
        "aggregation_unit": "unique_validation_item_macro_average",
        "group_filter": "none",
        "cold_start_scope_note": (
            "The validation items are disjoint from train_rating.csv items, so this is the whole validation "
            "cold-start item split rather than a selected cold-start subgroup."
        ),
    }
    if validate_csv is None:
        return scope
    validate_path = Path(validate_csv).expanduser()
    scope["evaluation_csv"] = str(validate_path)
    if not validate_path.exists():
        scope["evaluation_csv_exists"] = False
        return scope

    validate = pd.read_csv(validate_path)
    scope["evaluation_csv_exists"] = True
    scope["evaluation_interaction_count"] = int(len(validate))
    scope["evaluation_item_count"] = int(validate["asin"].nunique())
    scope["evaluation_user_count"] = int(validate["reviewerID"].nunique())
    if train_csv is not None:
        train_path = Path(train_csv).expanduser()
        scope["train_csv"] = str(train_path)
        if train_path.exists():
            train = pd.read_csv(train_path, usecols=["asin"])
            overlap = set(validate["asin"].dropna()) & set(train["asin"].dropna())
            scope["train_csv_exists"] = True
            scope["train_item_count"] = int(train["asin"].nunique())
            scope["train_item_overlap_count"] = int(len(overlap))
            scope["is_cold_start_item_split"] = len(overlap) == 0
        else:
            scope["train_csv_exists"] = False
    return scope


def build_baseline_summary(
    baseline_result: Path | str,
    validate_csv: Path | str | None = None,
    train_csv: Path | str | None = None,
) -> dict[str, Any]:
    path = Path(baseline_result).expanduser()
    result = _read_baseline_result(path)
    best_ndcg = _best_row(result, "ndcg@20")
    best_hr = _best_row(result, "hr@20")
    last = result.sort_values(["epoch", "checkpoint_index"]).iloc[-1]
    summary = {
        "baseline_result": str(path),
        "baseline_label": "baseline_seed43_workers8_fast_uniform",
        "rows": int(len(result)),
        "max_epoch": int(pd.to_numeric(result["epoch"], errors="coerce").max()),
        "best_ndcg@20": _round_float(best_ndcg["ndcg@20"]),
        "best_ndcg_epoch": int(best_ndcg["epoch"]),
        "best_hr@20_at_best_ndcg": _round_float(best_ndcg["hr@20"]),
        "best_hr@20": _round_float(best_hr["hr@20"]),
        "best_hr_epoch": int(best_hr["epoch"]),
        "ndcg@20_at_best_hr": _round_float(best_hr["ndcg@20"]),
        "last_ndcg@20": _round_float(last["ndcg@20"]),
        "last_hr@20": _round_float(last["hr@20"]),
    }
    summary.update(build_metric_scope_summary(validate_csv, train_csv))
    return summary


def _pct_delta(delta: float, baseline: float) -> float:
    if not np.isfinite(delta) or not np.isfinite(baseline) or baseline == 0:
        return float("nan")
    return delta / baseline * 100.0


def build_baseline_comparison(summary: pd.DataFrame, baseline: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    baseline_best_ndcg = float(baseline["best_ndcg@20"])
    baseline_last_ndcg = float(baseline["last_ndcg@20"])
    baseline_best_hr = float(baseline["best_hr@20"])
    baseline_last_hr = float(baseline["last_hr@20"])
    for _, row in summary.sort_values(["role", "run_label"]).iterrows():
        best_ndcg_delta = float(row["best_ndcg@20"]) - baseline_best_ndcg
        last_ndcg_delta = float(row["last_ndcg@20"]) - baseline_last_ndcg
        best_hr_delta = float(row["best_hr@20"]) - baseline_best_hr
        last_hr_delta = float(row["last_hr@20"]) - baseline_last_hr
        rows.append(
            {
                "run_label": row["run_label"],
                "role": row["role"],
                "method_variant": row["method_variant"],
                "evaluation_split": baseline.get("evaluation_split"),
                "metric_scope": baseline.get("metric_scope"),
                "group_filter": baseline.get("group_filter"),
                "evaluation_item_count": baseline.get("evaluation_item_count"),
                "baseline_best_ndcg@20": _round_float(baseline_best_ndcg),
                "branch_best_ndcg@20": _round_float(row["best_ndcg@20"]),
                "delta_best_ndcg@20_vs_baseline": _round_float(best_ndcg_delta),
                "pct_best_ndcg@20_vs_baseline": _round_float(_pct_delta(best_ndcg_delta, baseline_best_ndcg)),
                "baseline_last_ndcg@20": _round_float(baseline_last_ndcg),
                "branch_last_ndcg@20": _round_float(row["last_ndcg@20"]),
                "delta_last_ndcg@20_vs_baseline": _round_float(last_ndcg_delta),
                "pct_last_ndcg@20_vs_baseline": _round_float(_pct_delta(last_ndcg_delta, baseline_last_ndcg)),
                "baseline_best_hr@20": _round_float(baseline_best_hr),
                "branch_best_hr@20": _round_float(row["best_hr@20"]),
                "delta_best_hr@20_vs_baseline": _round_float(best_hr_delta),
                "pct_best_hr@20_vs_baseline": _round_float(_pct_delta(best_hr_delta, baseline_best_hr)),
                "baseline_last_hr@20": _round_float(baseline_last_hr),
                "branch_last_hr@20": _round_float(row["last_hr@20"]),
                "delta_last_hr@20_vs_baseline": _round_float(last_hr_delta),
                "pct_last_hr@20_vs_baseline": _round_float(_pct_delta(last_hr_delta, baseline_last_hr)),
            }
        )
    return pd.DataFrame(rows)


def _comparison_delta(comparison: pd.DataFrame, name: str) -> float:
    rows = comparison[comparison["comparison"].eq(name)]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0]["delta_best_ndcg@20"])


def decide_route(
    summary: pd.DataFrame,
    pair_comparison: pd.DataFrame,
    baseline_comparison: pd.DataFrame | None = None,
) -> dict[str, Any]:
    present_roles = set(summary.get("role", pd.Series(dtype=str)).astype(str))
    missing_roles = [role for role in REQUIRED_ROLES if role not in present_roles]
    if missing_roles:
        return {
            "route": "r5_incomplete_or_failed",
            "next_action": "inspect_missing_or_failed_runs",
            "open_multi_seed": False,
            "open_alpha_sweep": False,
            "reason": "required real/shuffle/control validation branches are missing",
            "missing_roles": missing_roles,
        }

    real = _summary_role_row(summary, "real")
    if real is None:
        raise ValueError("summary unexpectedly lacks real role")
    real_minus_shuffle = _comparison_delta(pair_comparison, "real_minus_shuffle")
    real_minus_rsp = _comparison_delta(pair_comparison, "real_minus_rsp_control")
    real_minus_acat = _comparison_delta(pair_comparison, "real_minus_acat_control")
    passes_m9 = bool(real_minus_shuffle > M9_REAL_SHUFFLE_GATE_ABS)
    passes_target = bool(real_minus_shuffle >= TARGET_NET_GATE_ABS)
    not_losing_rsp = bool(np.isfinite(real_minus_rsp) and real_minus_rsp >= 0)
    not_losing_acat = bool(np.isfinite(real_minus_acat) and real_minus_acat >= 0)
    not_early_transient = not bool(real.get("early_transient_peak_flag", False))
    completed = bool(real.get("completed_expected_epoch", True))
    real_abs_drop_vs_r4 = float(real["best_ndcg@20"]) - R4_REAL_BEST_NDCG_AT_20
    real_baseline_row = None
    if baseline_comparison is not None and not baseline_comparison.empty:
        real_baseline_rows = baseline_comparison[baseline_comparison["role"].eq("real")]
        if not real_baseline_rows.empty:
            real_baseline_row = real_baseline_rows.iloc[0]

    if not completed:
        route = "r5_incomplete_or_failed"
        next_action = "rerun_failed_or_incomplete_branch"
        reason = "real branch did not complete expected epochs"
        open_multi_seed = False
        open_alpha_sweep = False
    elif passes_target and not_losing_rsp and not_losing_acat and not_early_transient:
        route = "r5_boundary_competitor_ready_for_multi_seed"
        next_action = "run_multi_seed_confirmation_same_protocol"
        reason = "real-shuffle clears target gate and real does not lose to controls"
        open_multi_seed = True
        open_alpha_sweep = False
    elif not not_losing_rsp:
        route = "r5_boundary_competitor_rsp_control_dominated"
        next_action = "do_not_open_multiseed_analyze_rsp_dominance_before_new_carrier"
        reason = "real clears shuffle but loses to RSP control, so the gain is not Acat-specific"
        open_multi_seed = False
        open_alpha_sweep = False
    elif not not_losing_acat:
        route = "r5_boundary_competitor_acat_control_dominated"
        next_action = "do_not_open_multiseed_recheck_acat_specificity"
        reason = "real does not beat Acat control"
        open_multi_seed = False
        open_alpha_sweep = False
    elif not not_early_transient:
        route = "r5_boundary_competitor_early_transient_gain"
        next_action = "do_not_open_multiseed_inspect_regularization_or_strength"
        reason = "best real gain appears in the early transient window"
        open_multi_seed = False
        open_alpha_sweep = False
    elif passes_m9:
        route = "r5_boundary_competitor_weak_positive_not_multiseed_ready"
        next_action = "do_not_open_multiseed_redesign_control_aware_boundary_carrier"
        reason = "real-shuffle clears M9 gate but misses the target gate"
        open_multi_seed = False
        open_alpha_sweep = False
    else:
        route = "r5_boundary_competitor_no_reliable_gain"
        next_action = "stop_current_boundary_competitor_variant"
        reason = "real-shuffle does not clear the M9 net gain gate"
        open_multi_seed = False
        open_alpha_sweep = False

    peak_minus_last = real.get("best_ndcg_peak_minus_last")
    if peak_minus_last is None or pd.isna(peak_minus_last):
        peak_minus_last = float(real["best_ndcg@20"]) - float(real["last_ndcg@20"])

    return {
        "route": route,
        "next_action": next_action,
        "open_multi_seed": open_multi_seed,
        "open_alpha_sweep": open_alpha_sweep,
        "reason": reason,
        "real_best_ndcg@20": _round_float(real["best_ndcg@20"]),
        "real_best_epoch": int(real["best_ndcg_epoch"]),
        "real_last_ndcg@20": _round_float(real["last_ndcg@20"]),
        "real_peak_minus_last_ndcg@20": _round_float(peak_minus_last),
        "real_early_transient_peak_flag": bool(real.get("early_transient_peak_flag", False)),
        "real_minus_shuffle_best_ndcg@20": _round_float(real_minus_shuffle),
        "real_minus_rsp_control_best_ndcg@20": _round_float(real_minus_rsp),
        "real_minus_acat_control_best_ndcg@20": _round_float(real_minus_acat),
        "real_minus_r4_real_best_ndcg@20": _round_float(real_abs_drop_vs_r4),
        "real_minus_baseline_best_ndcg@20": (
            None if real_baseline_row is None else _round_float(real_baseline_row["delta_best_ndcg@20_vs_baseline"])
        ),
        "real_pct_best_ndcg@20_vs_baseline": (
            None if real_baseline_row is None else _round_float(real_baseline_row["pct_best_ndcg@20_vs_baseline"])
        ),
        "passes_m9_net_gate": passes_m9,
        "passes_target_net_gate": passes_target,
        "not_losing_rsp_control": not_losing_rsp,
        "not_losing_acat_control": not_losing_acat,
        "m9_real_shuffle_gate_abs": M9_REAL_SHUFFLE_GATE_ABS,
        "target_net_gate_abs": TARGET_NET_GATE_ABS,
        "r4_real_best_ndcg@20": R4_REAL_BEST_NDCG_AT_20,
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
    result_root: Path,
    validation_summary: pd.DataFrame,
    pair_comparison: pd.DataFrame,
    baseline_summary: dict[str, Any],
    baseline_comparison: pd.DataFrame,
    decision: dict[str, Any],
    manifest_name: str,
) -> None:
    evaluation_csv = baseline_summary.get("evaluation_csv", "unknown")
    evaluation_item_count = baseline_summary.get("evaluation_item_count", "unknown")
    evaluation_interaction_count = baseline_summary.get("evaluation_interaction_count", "unknown")
    train_item_overlap_count = baseline_summary.get("train_item_overlap_count", "unknown")
    content = f"""---
title: {run_stamp} CCFCRec Amazon-VG M10-R5 boundary competitor training 结果
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - M10-R5
  - boundary_competitor
---

# {run_stamp} CCFCRec Amazon-VG M10-R5 boundary competitor training 结果

## Material Passport（材料护照）

- artifact_type: experiment_training_result
- project: CCFCRec Amazon-VG category availability
- stage: M10-R5 boundary competitor carrier
- status: analyzed
- result_root: `{result_root}`

> [!info] 来源说明
> 上游总设计：[[{TOTAL_DESIGN_NOTE_NAME}]]
> R5 代码阅读与审计设计：[[{R5_DESIGN_NOTE_NAME}]]
> R5 offline audit（离线审计）路线判断：[[{R5_AUDIT_ROUTE_NOTE_NAME}]]
> R4 training（训练）路线判断：[[{R4_TRAINING_ROUTE_NOTE_NAME}]]
> 分析脚本：`{ANALYSIS_SCRIPT}`
> baseline（基线）来源：`{baseline_summary["baseline_result"]}`
> manifest（运行清单）：`{manifest_name}`

## 结论

```text
route = {decision["route"]}
next_action = {decision["next_action"]}
open_multi_seed = {decision["open_multi_seed"]}
open_alpha_sweep = {decision["open_alpha_sweep"]}
```

解释：M10-R5 boundary competitor（边界竞争用户）相对 shuffle（打乱负控）有明显净收益，但所有分支相对 baseline（基线）都有显著下降；real（真实载体）还输给 RSP control（RSP 对照），因此不能证明收益来自 Acat/recoverability（类别可用性/可恢复性）本身。

## Metric Scope（指标口径）

> [!warning] 口径说明
> 本报告的 Validation Summary（验证摘要）和 Baseline Comparison（基线对比）都使用同一 evaluation split（评估划分）：`{evaluation_csv}`。
> aggregation（聚合方式）：对该 CSV（逗号分隔文件）中的全部 unique asin（唯一物品）逐 item（物品）计算后做 macro average（宏平均）。
> item scope（物品口径）：{evaluation_item_count} 个 validation cold-start item（验证集冷启动物品），{evaluation_interaction_count} 条 validation interaction（验证交互），与 train_rating.csv（训练交互文件）的 item（物品）重叠数为 {train_item_overlap_count}。
> 这不是 test split（测试集划分）指标，不是全训练物品/全目录物品指标，也不是 high-Acat（高类别可用性）、RSP（随机/流行度相关对照信号）、near-cutoff（接近截断线）等 Task4 subgroup（第四任务子组）指标。

## Route Decision（路线决策）

```json
{json.dumps(_jsonable(decision), ensure_ascii=False, indent=2)}
```

## Validation Summary（验证摘要）

{md_table(validation_summary, ["run_label", "role", "method_variant", "best_ndcg@20", "best_ndcg_epoch", "last_ndcg@20", "best_ndcg_peak_minus_last", "best_hr@20", "best_hr_epoch", "completed_expected_epoch", "early_transient_peak_flag"], max_rows=20)}

## Pair Comparison（成对对比）

{md_table(pair_comparison, ["comparison", "delta_best_ndcg@20", "delta_best_ndcg_pct_vs_real", "delta_best_hr@20", "delta_last_ndcg@20", "left_best_epoch", "right_best_epoch", "passes_m9_net_gate", "passes_target_net_gate", "not_losing_to_control"], max_rows=20)}

## Baseline Summary（基线摘要）

```json
{json.dumps(_jsonable(baseline_summary), ensure_ascii=False, indent=2)}
```

## Baseline Comparison（基线对比）

> [!info] 字段说明
> `delta_best_ndcg@20_vs_baseline`：当前分支 best NDCG@20（最佳前20归一化折损累计增益）减 baseline（基线）best NDCG@20。
> `pct_best_ndcg@20_vs_baseline`：上述差值除以 baseline（基线）best NDCG@20 后的百分比变化。
> `metric_scope`：该 baseline（基线）对比的评估口径；本报告为 validation split（验证集划分）全体 cold-start item（冷启动物品）宏平均。

{md_table(baseline_comparison, ["run_label", "role", "metric_scope", "evaluation_item_count", "branch_best_ndcg@20", "baseline_best_ndcg@20", "delta_best_ndcg@20_vs_baseline", "pct_best_ndcg@20_vs_baseline", "branch_last_ndcg@20", "baseline_last_ndcg@20", "delta_last_ndcg@20_vs_baseline", "pct_last_ndcg@20_vs_baseline", "branch_best_hr@20", "baseline_best_hr@20", "delta_best_hr@20_vs_baseline", "pct_best_hr@20_vs_baseline"], max_rows=20)}
"""
    path.write_text(content, encoding="utf-8")


def build_outputs(output_root: Path, run_stamp: str) -> Outputs:
    output_dir = output_root / f"{run_stamp} task4-rollback-m10-r5-boundary-competitor-training-analysis"
    return Outputs(
        output_dir=output_dir,
        validation_curve_csv=output_dir / "m10_r5_validation_curve.csv",
        validation_summary_csv=output_dir / "m10_r5_validation_summary.csv",
        pair_comparison_csv=output_dir / "m10_r5_pair_comparison.csv",
        baseline_comparison_csv=output_dir / "m10_r5_baseline_comparison.csv",
        baseline_summary_json=output_dir / "m10_r5_baseline_summary.json",
        route_decision_json=output_dir / "m10_r5_route_decision.json",
        manifest_json=output_dir / "run_manifest.json",
        result_md=output_dir / f"{run_stamp} CCFCRec Amazon-VG M10-R5 boundary competitor training 结果.md",
    )


def run(args: argparse.Namespace) -> Outputs:
    run_stamp, run_date, run_iso = (args.run_stamp, args.run_stamp[:10], "") if args.run_stamp else now_stamp()
    if args.run_stamp:
        run_iso = datetime.strptime(args.run_stamp, "%Y-%m-%d %H%M%S").isoformat(timespec="seconds")
    result_root = Path(args.result_root).expanduser().resolve()
    outputs = build_outputs(Path(args.output_root).expanduser().resolve(), run_stamp)
    outputs.output_dir.mkdir(parents=True, exist_ok=True)

    curve = build_validation_curve(result_root)
    summary = build_validation_summary(curve)
    comparison = build_pair_comparison(summary)
    validate_csv = Path(args.validate_csv).expanduser().resolve()
    train_csv = Path(args.train_csv).expanduser().resolve()
    baseline_summary = build_baseline_summary(args.baseline_result, validate_csv, train_csv)
    baseline_comparison = build_baseline_comparison(summary, baseline_summary)
    decision = decide_route(summary, comparison, baseline_comparison)

    curve.to_csv(outputs.validation_curve_csv, index=False)
    summary.to_csv(outputs.validation_summary_csv, index=False)
    comparison.to_csv(outputs.pair_comparison_csv, index=False)
    baseline_comparison.to_csv(outputs.baseline_comparison_csv, index=False)
    outputs.baseline_summary_json.write_text(json.dumps(_jsonable(baseline_summary), ensure_ascii=False, indent=2), encoding="utf-8")
    outputs.route_decision_json.write_text(json.dumps(_jsonable(decision), ensure_ascii=False, indent=2), encoding="utf-8")
    write_result_markdown(
        outputs.result_md,
        run_stamp,
        result_root,
        summary,
        comparison,
        baseline_summary,
        baseline_comparison,
        decision,
        outputs.manifest_json.name,
    )

    manifest = {
        "run_stamp": run_stamp,
        "run_date": run_date,
        "run_iso": run_iso,
        "experiment_stage": "M10-R5",
        "analysis_script": ANALYSIS_SCRIPT,
        "result_root": str(result_root),
        "baseline_result": str(Path(args.baseline_result).expanduser()),
        "validate_csv": str(validate_csv),
        "train_csv": str(train_csv),
        "design_note": R5_DESIGN_NOTE_NAME,
        "audit_route_note": R5_AUDIT_ROUTE_NOTE_NAME,
        "training_route_note": R4_TRAINING_ROUTE_NOTE_NAME,
        "gates": {
            "m9_real_shuffle_gate_abs": M9_REAL_SHUFFLE_GATE_ABS,
            "target_net_gate_abs": TARGET_NET_GATE_ABS,
            "early_transient_epoch_ratio": EARLY_TRANSIENT_EPOCH_RATIO,
            "r4_real_best_ndcg@20": R4_REAL_BEST_NDCG_AT_20,
        },
        "outputs": {field: str(getattr(outputs, field)) for field in outputs.__dataclass_fields__},
        "baseline_summary": baseline_summary,
        "decision": decision,
    }
    outputs.manifest_json.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze M10-R5 boundary competitor training results.")
    parser.add_argument("--result-root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--baseline-result", default=str(DEFAULT_BASELINE_RESULT))
    parser.add_argument("--validate-csv", default=str(DEFAULT_VALIDATE_CSV))
    parser.add_argument("--train-csv", default=str(DEFAULT_TRAIN_CSV))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-stamp", default="")
    return parser


def main() -> None:
    outputs = run(build_arg_parser().parse_args())
    print(f"analysis output: {outputs.output_dir}")
    print(f"result markdown: {outputs.result_md}")
    print(f"route decision: {outputs.route_decision_json}")


if __name__ == "__main__":
    main()
