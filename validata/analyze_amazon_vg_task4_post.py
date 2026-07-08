#!/usr/bin/env python3
"""
CCFCRec Amazon-VG Task4-post analysis.

This script analyzes the four Task4 Acat_v3 minimal weight control runs:
M1 RSP-only hard weight, M2 Acat_v3 high weight, M6 Acat_v3 shuffle control,
and M3 Acat_v3 train-safe hard weight.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch


METHOD_INFO = {
    "task4_rsp_high_weight": {
        "method_id": "M1",
        "method_label": "M1_rsp_only_hard_weight",
        "order": 1,
    },
    "task4_acat_high_weight": {
        "method_id": "M2",
        "method_label": "M2_acat_v3_high_weight",
        "order": 2,
    },
    "task4_acat_shuffle_high_weight": {
        "method_id": "M6",
        "method_label": "M6_acat_v3_shuffle_high_weight",
        "order": 3,
    },
    "task4_acat_trainhard_weight": {
        "method_id": "M3",
        "method_label": "M3_acat_v3_train_safe_hard_weight",
        "order": 4,
    },
}

CORE_METHODS = [
    "task4_rsp_high_weight",
    "task4_acat_high_weight",
    "task4_acat_shuffle_high_weight",
    "task4_acat_trainhard_weight",
]

GROUP_COLUMNS = [
    "high_acat_flag",
    "eval_baseline_hard_flag",
    "high_acat_eval_hard_flag",
    "high_acat_train_safe_hard_flag",
    "RSP_group",
    "s_cat_v3_group",
]

METRIC_COLUMNS = ["hr@5", "hr@10", "hr@20", "ndcg@5", "ndcg@10", "ndcg@20"]
EXPERIMENT_ID = "task4_acat_v3_weight_controls_m1_m2_m6_m3"
TRAINING_LAUNCHER_SCRIPT = (
    "scripts/run_task4_acat_v3_weight_controls_m1_m2_m6_m3_seed43_fast_uniform_mps_100epoch.sh"
)
ANALYSIS_SCRIPT = "validata/analyze_amazon_vg_task4_post.py"
DESIGN_NOTE_NAME = "2026-07-06 130109 CCFCRec Amazon-VG Task4-post Acat v3 minimal weight controls 训练后分析设计"


@dataclass(frozen=True)
class AnalysisOutputs:
    output_dir: Path
    run_index_csv: Path
    result_summary_csv: Path
    best_checkpoint_csv: Path
    method_comparison_csv: Path
    training_dynamics_csv: Path
    item_eval_csv: Path
    group_effect_csv: Path
    group_delta_csv: Path
    manifest_json: Path
    result_md: Path


def now_stamp() -> tuple[str, str, str]:
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H%M%S"), now.strftime("%Y-%m-%d"), now.isoformat(timespec="seconds")


def nan_float() -> float:
    return float("nan")


def safe_float(value) -> float:
    if pd.isna(value):
        return nan_float()
    return float(value)


def safe_mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return nan_float()
    return float(numeric.mean())


def bool_text(value) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    text = str(value)
    if text.lower() in {"true", "false"}:
        return text.capitalize()
    return text


def method_order(method_variant: str) -> int:
    return int(METHOD_INFO.get(method_variant, {"order": 99})["order"])


def infer_device_from_result_root(result_root: Path) -> str:
    name = result_root.name.lower()
    for candidate in ["mps", "cuda", "cpu"]:
        if f"_{candidate}_" in name or name.endswith(f"_{candidate}"):
            return candidate
    master_log = result_root / "logs" / "master.log"
    if master_log.exists():
        match = re.search(r"CCFCREC_DEVICE=([A-Za-z0-9_+-]+)", master_log.read_text(encoding="utf-8", errors="ignore"))
        if match:
            return match.group(1)
    return "unknown"


def select_device(requested: str) -> str:
    requested = requested.strip().lower()
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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


def discover_run_dirs(result_root: Path) -> list[Path]:
    run_dirs = []
    for child in result_root.iterdir():
        if child.is_dir() and (child / "result.csv").exists() and (child / "run_config.json").exists():
            run_dirs.append(child)
    return sorted(run_dirs)


def checkpoint_indices(run_dir: Path) -> list[int]:
    indices = []
    for path in run_dir.glob("*.pt"):
        try:
            indices.append(int(path.stem))
        except ValueError:
            continue
    return sorted(indices)


def load_run_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))


def build_run_index(result_root: Path) -> pd.DataFrame:
    rows = []
    device = infer_device_from_result_root(result_root)
    for run_dir in discover_run_dirs(result_root):
        config = load_run_config(run_dir)
        result = pd.read_csv(run_dir / "result.csv")
        checkpoints = checkpoint_indices(run_dir)
        method_variant = str(config.get("method_variant", "unknown"))
        info = METHOD_INFO.get(method_variant, {})
        rows.append(
            {
                "method_id": info.get("method_id", ""),
                "method_label": info.get("method_label", method_variant),
                "method_variant": method_variant,
                "run_dir_name": run_dir.name,
                "run_dir": str(run_dir),
                "result_csv": str(run_dir / "result.csv"),
                "save_dict_path": str(run_dir / "save_dict.pkl"),
                "seed": config.get("seed", ""),
                "num_workers": config.get("num_workers", ""),
                "batch_size": config.get("batch_size", ""),
                "negative_sampling_mode": config.get("negative_sampling_mode", ""),
                "device": device,
                "epoch": int(result["epoch"].max()) if "epoch" in result.columns and not result.empty else "",
                "checkpoint_count": len(checkpoints),
                "min_checkpoint_index": min(checkpoints) if checkpoints else "",
                "max_checkpoint_index": max(checkpoints) if checkpoints else "",
                "task4_loss_alpha": config.get("task4_loss_alpha", ""),
                "task4_shuffle_seed": config.get("task4_shuffle_seed", ""),
                "task4_profile_path": config.get("task4_profile_path", ""),
            }
        )
    index = pd.DataFrame(rows)
    if not index.empty:
        index["method_order"] = index["method_variant"].map(method_order)
        index = index.sort_values(["method_order", "run_dir_name"]).drop(columns=["method_order"]).reset_index(drop=True)
    return index


def select_best_checkpoint_from_result(
    result: pd.DataFrame,
    primary_metric: str = "ndcg@20",
    tie_metric: str = "hr@20",
) -> pd.Series:
    required = {"checkpoint_index", "epoch", primary_metric, tie_metric}
    missing = required - set(result.columns)
    if missing:
        raise ValueError(f"result.csv missing columns: {sorted(missing)}")
    ordered = result.sort_values(
        [primary_metric, tie_metric, "epoch", "checkpoint_index"],
        ascending=[False, False, True, True],
    )
    return ordered.iloc[0]


def build_best_checkpoint_table(run_index: pd.DataFrame, primary_metric: str = "ndcg@20") -> pd.DataFrame:
    rows = []
    for row in run_index.itertuples(index=False):
        result = pd.read_csv(row.result_csv)
        best = select_best_checkpoint_from_result(result, primary_metric=primary_metric)
        checkpoint_path = Path(row.run_dir) / f"{int(best['checkpoint_index'])}.pt"
        method_variant = row.method_variant
        info = METHOD_INFO.get(method_variant, {})
        rows.append(
            {
                "method_id": info.get("method_id", ""),
                "method_label": info.get("method_label", method_variant),
                "method_variant": method_variant,
                "run_dir_name": row.run_dir_name,
                "run_dir": row.run_dir,
                "best_checkpoint_index": int(best["checkpoint_index"]),
                "best_epoch": int(best["epoch"]),
                "best_checkpoint_path": str(checkpoint_path),
                "best_checkpoint_exists": checkpoint_path.exists(),
                "selection_metric": primary_metric,
                "best_hr@5": safe_float(best["hr@5"]),
                "best_hr@10": safe_float(best["hr@10"]),
                "best_hr@20": safe_float(best["hr@20"]),
                "best_ndcg@5": safe_float(best["ndcg@5"]),
                "best_ndcg@10": safe_float(best["ndcg@10"]),
                "best_ndcg@20": safe_float(best["ndcg@20"]),
                "seed": getattr(row, "seed"),
                "num_workers": getattr(row, "num_workers"),
                "negative_sampling_mode": getattr(row, "negative_sampling_mode"),
                "device": getattr(row, "device"),
            }
        )
    best_table = pd.DataFrame(rows)
    if not best_table.empty:
        best_table["method_order"] = best_table["method_variant"].map(method_order)
        best_table = best_table.sort_values("method_order").drop(columns=["method_order"]).reset_index(drop=True)
    return best_table


def build_training_dynamics(run_index: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in run_index.itertuples(index=False):
        result = pd.read_csv(row.result_csv)
        best = select_best_checkpoint_from_result(result)
        final = result.sort_values(["epoch", "checkpoint_index"]).iloc[-1]
        last10 = result.sort_values(["epoch", "checkpoint_index"]).tail(10)
        peak_minus_final = safe_float(best["ndcg@20"]) - safe_float(final["ndcg@20"])
        final_epoch = int(final["epoch"])
        best_epoch = int(best["epoch"])
        rows.append(
            {
                "method_variant": row.method_variant,
                "method_label": getattr(row, "method_label"),
                "best_epoch": best_epoch,
                "best_checkpoint_index": int(best["checkpoint_index"]),
                "final_epoch": final_epoch,
                "final_checkpoint_index": int(final["checkpoint_index"]),
                "best_epoch_fraction": best_epoch / final_epoch if final_epoch else nan_float(),
                "best_ndcg@20": safe_float(best["ndcg@20"]),
                "final_ndcg@20": safe_float(final["ndcg@20"]),
                "peak_minus_final_ndcg@20": peak_minus_final,
                "best_hr@20": safe_float(best["hr@20"]),
                "final_hr@20": safe_float(final["hr@20"]),
                "peak_minus_final_hr@20": safe_float(best["hr@20"]) - safe_float(final["hr@20"]),
                "last10_ndcg@20_mean": safe_mean(last10["ndcg@20"]),
                "last10_ndcg@20_std": float(pd.to_numeric(last10["ndcg@20"], errors="coerce").std(ddof=0)),
                "late_decay_flag": bool(peak_minus_final > 0.001),
            }
        )
    dynamics = pd.DataFrame(rows)
    if not dynamics.empty:
        dynamics["method_order"] = dynamics["method_variant"].map(method_order)
        dynamics = dynamics.sort_values("method_order").drop(columns=["method_order"]).reset_index(drop=True)
    return dynamics


def metric_value(best_table: pd.DataFrame, method: str, metric: str) -> float:
    sub = best_table[best_table["method_variant"].eq(method)]
    if sub.empty:
        return nan_float()
    return safe_float(sub.iloc[0][metric])


def build_method_comparison(best_table: pd.DataFrame, min_meaningful_ndcg_delta: float = 0.0005) -> pd.DataFrame:
    pairs = [
        ("M3_acat_trainhard_minus_M1_rsp_only", "task4_acat_trainhard_weight", "task4_rsp_high_weight"),
        ("M3_acat_trainhard_minus_M6_acat_shuffle", "task4_acat_trainhard_weight", "task4_acat_shuffle_high_weight"),
        ("M3_acat_trainhard_minus_M2_acat_high", "task4_acat_trainhard_weight", "task4_acat_high_weight"),
        ("M2_acat_high_minus_M6_acat_shuffle", "task4_acat_high_weight", "task4_acat_shuffle_high_weight"),
    ]
    rows = []
    for comparison, left, right in pairs:
        left_ndcg = metric_value(best_table, left, "best_ndcg@20")
        right_ndcg = metric_value(best_table, right, "best_ndcg@20")
        left_hr = metric_value(best_table, left, "best_hr@20")
        right_hr = metric_value(best_table, right, "best_hr@20")
        delta_ndcg = left_ndcg - right_ndcg if pd.notna(left_ndcg) and pd.notna(right_ndcg) else nan_float()
        delta_hr = left_hr - right_hr if pd.notna(left_hr) and pd.notna(right_hr) else nan_float()
        rows.append(
            {
                "comparison": comparison,
                "left_method": left,
                "right_method": right,
                "left_best_ndcg@20": left_ndcg,
                "right_best_ndcg@20": right_ndcg,
                "delta_ndcg@20": delta_ndcg,
                "left_best_hr@20": left_hr,
                "right_best_hr@20": right_hr,
                "delta_hr@20": delta_hr,
                "min_meaningful_ndcg_delta": float(min_meaningful_ndcg_delta),
                "meaningful_ndcg_win": bool(delta_ndcg >= min_meaningful_ndcg_delta),
                "raw_ndcg_win": bool(delta_ndcg > 0),
                "hr_not_worse": bool(delta_hr >= 0),
            }
        )
    result = pd.DataFrame(rows)
    for col in ["meaningful_ndcg_win", "raw_ndcg_win", "hr_not_worse"]:
        result[col] = result[col].astype(object)
    return result


def build_result_summary(best_table: pd.DataFrame, dynamics: pd.DataFrame) -> pd.DataFrame:
    if best_table.empty:
        return pd.DataFrame()
    summary = best_table.merge(
        dynamics[
            [
                "method_variant",
                "final_epoch",
                "final_ndcg@20",
                "final_hr@20",
                "peak_minus_final_ndcg@20",
                "last10_ndcg@20_mean",
                "last10_ndcg@20_std",
                "late_decay_flag",
            ]
        ],
        on="method_variant",
        how="left",
    )
    return summary


def _comparison_row(comparison: pd.DataFrame, name: str) -> pd.Series | None:
    sub = comparison[comparison["comparison"].eq(name)]
    if sub.empty:
        return None
    return sub.iloc[0]


def _best_method_by_ndcg(best_table: pd.DataFrame) -> str:
    if best_table.empty:
        return ""
    return str(best_table.sort_values("best_ndcg@20", ascending=False).iloc[0]["method_variant"])


def _group_delta_value(group_delta_summary: pd.DataFrame, split: str, group_column: str, group_value: str, metric: str) -> float:
    if group_delta_summary.empty:
        return nan_float()
    sub = group_delta_summary[
        group_delta_summary["split"].eq(split)
        & group_delta_summary["group_column"].eq(group_column)
        & group_delta_summary["group_value"].astype(str).eq(group_value)
    ]
    if sub.empty or metric not in sub.columns:
        return nan_float()
    return safe_float(sub.iloc[0][metric])


def build_route_decision(
    best_table: pd.DataFrame,
    comparison: pd.DataFrame,
    dynamics: pd.DataFrame,
    group_delta_summary: pd.DataFrame,
    min_meaningful_ndcg_delta: float = 0.0005,
) -> dict:
    m3_vs_m1 = _comparison_row(comparison, "M3_acat_trainhard_minus_M1_rsp_only")
    m3_vs_m6 = _comparison_row(comparison, "M3_acat_trainhard_minus_M6_acat_shuffle")
    m3_best = _best_method_by_ndcg(best_table) == "task4_acat_trainhard_weight"
    m3_late_decay = False
    if not dynamics.empty:
        sub = dynamics[dynamics["method_variant"].eq("task4_acat_trainhard_weight")]
        if not sub.empty:
            m3_late_decay = bool(sub.iloc[0].get("late_decay_flag", False))

    m3_m1_meaningful = bool(m3_vs_m1 is not None and m3_vs_m1["meaningful_ndcg_win"])
    m3_m6_meaningful = bool(m3_vs_m6 is not None and m3_vs_m6["meaningful_ndcg_win"])
    m3_m6_raw = bool(m3_vs_m6 is not None and m3_vs_m6["raw_ndcg_win"])
    m3_m6_hr_not_worse = bool(m3_vs_m6 is not None and m3_vs_m6["hr_not_worse"])

    test_trainhard_m3_minus_m6 = _group_delta_value(
        group_delta_summary,
        "test",
        "high_acat_train_safe_hard_flag",
        "True",
        "m3_minus_m6_ndcg@20_mean",
    )
    test_evalhard_m3_minus_m6 = _group_delta_value(
        group_delta_summary,
        "test",
        "high_acat_eval_hard_flag",
        "True",
        "m3_minus_m6_ndcg@20_mean",
    )
    group_supports_m3 = any(
        pd.notna(value) and value >= min_meaningful_ndcg_delta
        for value in [test_trainhard_m3_minus_m6, test_evalhard_m3_minus_m6]
    )

    if m3_m1_meaningful and m3_m6_meaningful and m3_m6_hr_not_worse and not m3_late_decay and group_supports_m3:
        route = "go_m4"
        go_m4 = True
        need_seed_repeat = True
        seed_repeat_gate_status = "open_after_seed43_pass"
        next_action = "run_multi_seed_stability_before_final_table"
    elif m3_best and (m3_m6_raw or m3_m1_meaningful):
        route = "revise_m3"
        go_m4 = False
        need_seed_repeat = False
        seed_repeat_gate_status = "not_open_current_m3"
        next_action = "revise_m3_or_new_carrier_seed43_screen_first"
    else:
        route = "stop_carrier"
        go_m4 = False
        need_seed_repeat = False
        seed_repeat_gate_status = "not_open"
        next_action = "stop_current_carrier_and_design_new_seed43_screen"

    return {
        "route": route,
        "go_m4": go_m4,
        "need_seed_repeat": need_seed_repeat,
        "seed_repeat_gate_status": seed_repeat_gate_status,
        "next_action": next_action,
        "m3_best_validate_ndcg": m3_best,
        "m3_vs_m1_meaningful_ndcg_win": m3_m1_meaningful,
        "m3_vs_m6_meaningful_ndcg_win": m3_m6_meaningful,
        "m3_vs_m6_raw_ndcg_win": m3_m6_raw,
        "m3_vs_m6_hr_not_worse": m3_m6_hr_not_worse,
        "m3_late_decay_flag": m3_late_decay,
        "group_supports_m3_vs_shuffle": group_supports_m3,
        "test_high_acat_train_safe_hard_m3_minus_m6_ndcg@20": test_trainhard_m3_minus_m6,
        "test_high_acat_eval_hard_m3_minus_m6_ndcg@20": test_evalhard_m3_minus_m6,
        "min_meaningful_ndcg_delta": min_meaningful_ndcg_delta,
    }


def build_model_args_from_state_dict(state_dict: dict[str, torch.Tensor]) -> SimpleNamespace:
    method_variant = "baseline"
    category_conf_dim = 16
    category_conf_max_count = 5
    category_gate_scale = 0.5
    if "category_conf_embedding.weight" in state_dict:
        method_variant = "category_conf_input"
        category_conf_dim = int(state_dict["category_conf_embedding.weight"].shape[1])
    if "category_fusion_gate.weight" in state_dict:
        method_variant = "category_conf_fusion_gate"
    return SimpleNamespace(
        attr_num=int(state_dict["attr_matrix"].shape[0]),
        attr_present_dim=int(state_dict["attr_matrix"].shape[1]),
        implicit_dim=int(state_dict["user_embedding"].shape[1]),
        cat_implicit_dim=int(state_dict["gen_layer1.weight"].shape[0]),
        user_number=int(state_dict["user_embedding"].shape[0]),
        item_number=int(state_dict["item_embedding"].shape[0]),
        pretrain=False,
        pretrain_update=False,
        method_variant=method_variant,
        category_conf_dim=category_conf_dim,
        category_conf_max_count=category_conf_max_count,
        category_gate_scale=category_gate_scale,
    )


def load_model_from_checkpoint(code_root: Path, checkpoint_path: Path, device_name: str):
    os.environ["CCFCREC_DEVICE"] = device_name
    amazon_code_dir = code_root / "Amazon VG"
    if str(amazon_code_dir) not in sys.path:
        sys.path.insert(0, str(amazon_code_dir))
    from model import CCFCRec

    state_dict = torch.load(checkpoint_path, map_location=device_name)
    model = CCFCRec(build_model_args_from_state_dict(state_dict))
    model.load_state_dict(state_dict)
    model.to(torch.device(device_name))
    model.eval()
    return model


def load_save_dict(run_dir: Path) -> dict:
    with open(run_dir / "save_dict.pkl", "rb") as file:
        return pickle.load(file)


def dcg_from_hits(hits: list[float]) -> float:
    if not hits:
        return 0.0
    arr = np.asarray(hits, dtype=float)
    return float(np.sum((np.power(2, arr) - 1) / np.log2(np.arange(2, arr.size + 2))))


def ranking_metrics_for_recommendations(recommend_users: list[int], target_indices: list[int], k: int) -> tuple[float, float, int]:
    top_users = recommend_users[:k]
    target_set = set(target_indices)
    hits = [1.0 if user in target_set else 0.0 for user in top_users]
    hit_count = int(sum(hits))
    hr = hit_count / k
    ideal = dcg_from_hits(sorted(hits, reverse=True))
    ndcg = dcg_from_hits(hits) / ideal if ideal > 0 else 0.0
    return hr, ndcg, hit_count


def split_targets(split_df: pd.DataFrame, user_ser_dict: dict[str, int]) -> tuple[list[str], dict[str, dict[str, object]]]:
    split_df = split_df.copy()
    split_df["asin"] = split_df["asin"].astype(str)
    split_df["reviewerID"] = split_df["reviewerID"].astype(str)
    items = list(dict.fromkeys(split_df["asin"].tolist()))
    targets = {}
    for asin, sub in split_df.groupby("asin", sort=False):
        raw_users = sub["reviewerID"].tolist()
        mapped_users = [user_ser_dict.get(user) for user in raw_users]
        known_users = sorted({int(user) for user in mapped_users if user is not None})
        targets[str(asin)] = {
            "raw_target_user_count": len(raw_users),
            "mapped_target_user_count": len(known_users),
            "unknown_target_user_count": len(raw_users) - sum(user is not None for user in mapped_users),
            "target_user_indices": known_users,
        }
    return items, targets


def evaluate_split_item_metrics(
    model,
    save_dict: dict,
    split_csv: Path,
    split_name: str,
    method_variant: str,
    method_label: str,
    checkpoint_index: int,
    device_name: str,
    batch_size: int = 256,
) -> pd.DataFrame:
    amazon_code_dir = Path(__file__).resolve().parents[1] / "Amazon VG"
    if str(amazon_code_dir) not in sys.path:
        sys.path.insert(0, str(amazon_code_dir))
    from support import build_item_feature_tensors

    device = torch.device(device_name)
    split_df = pd.read_csv(split_csv, dtype={"asin": str, "reviewerID": str})
    items, targets = split_targets(split_df, save_dict["user_ser_dict"])
    item_serialize_dict = {item: item_idx for item_idx, item in enumerate(items)}
    category_tensor, image_tensor = build_item_feature_tensors(
        item_serialize_dict=item_serialize_dict,
        img_features=save_dict["img_feature_dict"],
        genres=save_dict["asin_category_int_map"],
        category_num=int(save_dict["category_ser_map_len"]),
    )

    model = model.to(device)
    model.eval()
    user_embedding = model.user_embedding.to(device)
    user_norms = torch.norm(user_embedding, dim=1)
    rows = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            end = min(start + batch_size, len(items))
            batch_items = items[start:end]
            genres = category_tensor[start:end].to(device)
            image_features = image_tensor[start:end].to(device)
            q_v_c = model(genres, image_features, len(batch_items))
            scores_batch = torch.matmul(q_v_c, user_embedding.t())
            top_values, top_indices = torch.topk(scores_batch, k=20, dim=1, largest=True)
            q_norms = torch.norm(q_v_c, dim=1)
            for row_idx, asin in enumerate(batch_items):
                target_info = targets[asin]
                target_indices = list(target_info["target_user_indices"])
                recommend_users = [int(value) for value in top_indices[row_idx].detach().cpu().tolist()]
                hr5, ndcg5, hit5 = ranking_metrics_for_recommendations(recommend_users, target_indices, 5)
                hr10, ndcg10, hit10 = ranking_metrics_for_recommendations(recommend_users, target_indices, 10)
                hr20, ndcg20, hit20 = ranking_metrics_for_recommendations(recommend_users, target_indices, 20)
                scores = scores_batch[row_idx]
                if target_indices:
                    target_tensor = torch.tensor(target_indices, dtype=torch.long, device=device)
                    target_scores = scores[target_tensor]
                    best_target_score = float(target_scores.max().item())
                    target_score_mean = float(target_scores.mean().item())
                    best_target_rank = int((scores > target_scores.max()).sum().item()) + 1
                    target_user_norm_mean = float(user_norms[target_tensor].mean().item())
                    margin_to_top20_cutoff = best_target_score - float(top_values[row_idx, 19].item())
                else:
                    best_target_score = nan_float()
                    target_score_mean = nan_float()
                    best_target_rank = nan_float()
                    target_user_norm_mean = nan_float()
                    margin_to_top20_cutoff = nan_float()
                rows.append(
                    {
                        "method_variant": method_variant,
                        "method_label": method_label,
                        "checkpoint_index": checkpoint_index,
                        "split": split_name,
                        "raw_asin": asin,
                        "hr@5": hr5,
                        "hr@10": hr10,
                        "hr@20": hr20,
                        "ndcg@5": ndcg5,
                        "ndcg@10": ndcg10,
                        "ndcg@20": ndcg20,
                        "hit_count@5": hit5,
                        "hit_count@10": hit10,
                        "hit_count@20": hit20,
                        "q_norm": float(q_norms[row_idx].item()),
                        "target_score_max": best_target_score,
                        "target_score_mean": target_score_mean,
                        "margin_to_top20_cutoff": margin_to_top20_cutoff,
                        "best_target_rank": best_target_rank,
                        "target_user_norm_mean": target_user_norm_mean,
                        "raw_target_user_count": target_info["raw_target_user_count"],
                        "mapped_target_user_count": target_info["mapped_target_user_count"],
                        "unknown_target_user_count": target_info["unknown_target_user_count"],
                    }
                )
    return pd.DataFrame(rows)


def evaluate_best_checkpoints_item_level(
    best_table: pd.DataFrame,
    code_root: Path,
    data_dir: Path,
    device_name: str,
    batch_size: int,
) -> pd.DataFrame:
    frames = []
    split_paths = {
        "validate": data_dir / "validate_rating.csv",
        "test": data_dir / "test_rating.csv",
    }
    for row in best_table.itertuples(index=False):
        run_dir = Path(row.run_dir)
        print(
            f"item-eval start method={row.method_variant} "
            f"checkpoint={row.best_checkpoint_index} device={device_name}",
            flush=True,
        )
        save_dict = load_save_dict(run_dir)
        model = load_model_from_checkpoint(code_root, Path(row.best_checkpoint_path), device_name)
        for split_name, split_csv in split_paths.items():
            print(f"item-eval split={split_name} method={row.method_variant}", flush=True)
            frames.append(
                evaluate_split_item_metrics(
                    model=model,
                    save_dict=save_dict,
                    split_csv=split_csv,
                    split_name=split_name,
                    method_variant=row.method_variant,
                    method_label=row.method_label,
                    checkpoint_index=int(row.best_checkpoint_index),
                    device_name=device_name,
                    batch_size=batch_size,
                )
            )
        del model
        if device_name == "cuda":
            torch.cuda.empty_cache()
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def merge_task4_profile(item_eval: pd.DataFrame, task4_profile_path: Path) -> pd.DataFrame:
    profile = pd.read_csv(task4_profile_path, dtype={"raw_asin": str})
    keep_cols = [
        "raw_asin",
        "split",
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
    available = [col for col in keep_cols if col in profile.columns]
    merged = item_eval.merge(profile[available], on=["raw_asin", "split"], how="left", validate="many_to_one")
    return merged


def build_group_effect_summary(item_profile: pd.DataFrame) -> pd.DataFrame:
    rows = []
    available_group_cols = [col for col in GROUP_COLUMNS if col in item_profile.columns]
    for (method_variant, split), method_split in item_profile.groupby(["method_variant", "split"], dropna=False):
        method_label = METHOD_INFO.get(method_variant, {}).get("method_label", method_variant)
        for group_col in available_group_cols:
            for group_value, sub in method_split.groupby(group_col, dropna=False):
                rows.append(
                    {
                        "method_variant": method_variant,
                        "method_label": method_label,
                        "split": split,
                        "group_column": group_col,
                        "group_value": bool_text(group_value),
                        "item_count": int(len(sub)),
                        "ndcg@20_mean": safe_mean(sub["ndcg@20"]),
                        "hr@20_mean": safe_mean(sub["hr@20"]),
                        "q_norm_mean": safe_mean(sub["q_norm"]) if "q_norm" in sub.columns else nan_float(),
                        "margin_to_top20_cutoff_mean": safe_mean(sub["margin_to_top20_cutoff"])
                        if "margin_to_top20_cutoff" in sub.columns
                        else nan_float(),
                        "best_target_rank_median": safe_float(pd.to_numeric(sub["best_target_rank"], errors="coerce").median())
                        if "best_target_rank" in sub.columns
                        else nan_float(),
                    }
                )
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary["method_order"] = summary["method_variant"].map(method_order)
        summary = summary.sort_values(["split", "group_column", "group_value", "method_order"]).drop(columns=["method_order"]).reset_index(drop=True)
    return summary


def build_group_delta_summary(group_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if group_summary.empty:
        return pd.DataFrame()
    key_cols = ["split", "group_column", "group_value"]
    for key, sub in group_summary.groupby(key_cols, dropna=False):
        indexed = sub.set_index("method_variant")
        if "task4_acat_trainhard_weight" not in indexed.index:
            continue
        row = dict(zip(key_cols, key))
        if "item_count" in indexed.columns:
            row["item_count_m3"] = int(indexed.loc["task4_acat_trainhard_weight"]["item_count"])
        else:
            row["item_count_m3"] = ""
        for metric in ["ndcg@20_mean", "hr@20_mean"]:
            m3 = safe_float(indexed.loc["task4_acat_trainhard_weight"][metric])
            for method, label in [
                ("task4_rsp_high_weight", "m1"),
                ("task4_acat_high_weight", "m2"),
                ("task4_acat_shuffle_high_weight", "m6"),
            ]:
                if method in indexed.index:
                    other = safe_float(indexed.loc[method][metric])
                    row[f"m3_minus_{label}_{metric}"] = m3 - other
                else:
                    row[f"m3_minus_{label}_{metric}"] = nan_float()
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(key_cols).reset_index(drop=True)
    return result


def write_result_markdown(
    output_path: Path,
    run_stamp: str,
    design_note_name: str,
    result_root: Path,
    result_summary: pd.DataFrame,
    best_table: pd.DataFrame,
    comparison: pd.DataFrame,
    dynamics: pd.DataFrame,
    group_delta: pd.DataFrame,
    decision: dict,
    manifest_name: str,
) -> None:
    high_group = group_delta[
        group_delta["group_column"].eq("high_acat_train_safe_hard_flag")
        & group_delta["group_value"].astype(str).eq("True")
    ].copy() if not group_delta.empty else pd.DataFrame()
    high_group = high_group[high_group["split"].isin(["validate", "test"])] if not high_group.empty else high_group
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG Task4-post Acat v3 minimal weight controls 训练后分析结果
date: 2026-07-06
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - 实验结果
  - 训练后分析
---

# {run_stamp} CCFCRec Amazon-VG Task4-post Acat v3 minimal weight controls 训练后分析结果

## 来源

设计来源：[[{design_note_name}]]

训练启动脚本：

```text
{TRAINING_LAUNCHER_SCRIPT}
```

分析脚本：

```text
{ANALYSIS_SCRIPT}
```

训练结果根目录：

```text
{result_root}
```

manifest（运行清单）：`{manifest_name}`

## 结论

```text
route = {decision["route"]}
go_m4 = {decision["go_m4"]}
need_seed_repeat = {decision["need_seed_repeat"]}
seed_repeat_gate_status = {decision.get("seed_repeat_gate_status", "")}
next_action = {decision.get("next_action", "")}
```

解释：

```text
M3（Acat_v3 + 训练安全困难样本加权）是 validate NDCG@20 最高的方法，但相对 M6 shuffle（Acat_v3 打乱负控）的 NDCG@20 优势没有达到 meaningful threshold（有意义阈值），且 HR@20 不占优。
因此当前证据不能直接支持进入 M4（Acat_v3 条件化成对边际损失），也不应该优先对当前 M3 做大规模 seed repeat（多随机种子复验）。
更稳路线是 revise_m3（修改 M3）或重新设计 carrier（方法承载方式），继续用 seed43（单随机种子快速筛选）先筛掉没希望的方法。
```

## Best Checkpoint

> [!info] 字段说明
> - `best_epoch`：按 validate NDCG@20（验证集前20排序质量）选择的最佳轮次。
> - `best_checkpoint_index`：最佳模型保存点编号。
> - `best_ndcg@20`：最佳 checkpoint 的验证集 NDCG@20。
> - `best_hr@20`：最佳 checkpoint 的验证集 HR@20（前20命中率）。

{md_table(best_table, ["method_id", "method_variant", "best_epoch", "best_checkpoint_index", "best_ndcg@20", "best_hr@20"])}

## 方法主比较

> [!info] 字段说明
> - `delta_ndcg@20`：左侧方法减右侧方法的 NDCG@20 差异。
> - `meaningful_ndcg_win`：是否超过预设有意义差异阈值。
> - `hr_not_worse`：HR@20 是否不低于对照。

{md_table(comparison, ["comparison", "delta_ndcg@20", "delta_hr@20", "meaningful_ndcg_win", "hr_not_worse"])}

## 训练动态

> [!info] 字段说明
> - `peak_minus_final_ndcg@20`：最佳 checkpoint 减最后 checkpoint，越大说明后期衰减越明显。
> - `last10_ndcg@20_std`：最后 10 个 checkpoint 的波动。
> - `late_decay_flag`：是否存在超过 0.001 的峰值到末轮衰减。

{md_table(dynamics, ["method_variant", "best_epoch", "final_epoch", "best_ndcg@20", "final_ndcg@20", "peak_minus_final_ndcg@20", "last10_ndcg@20_std", "late_decay_flag"])}

## 分组 Delta

> [!info] 字段说明
> - `high_acat_train_safe_hard_flag=True`：高 Acat_v3（类别可用性）且训练安全困难机会组。
> - `m3_minus_m1_ndcg@20_mean`：M3 减 M1 的分组平均 NDCG@20。
> - `m3_minus_m6_ndcg@20_mean`：M3 减 M6 shuffle 负控的分组平均 NDCG@20。

{md_table(high_group, ["split", "group_column", "group_value", "item_count_m3", "m3_minus_m1_ndcg@20_mean", "m3_minus_m6_ndcg@20_mean", "m3_minus_m1_hr@20_mean", "m3_minus_m6_hr@20_mean"]) if not high_group.empty else "_分组复评未生成或没有 high_acat_train_safe_hard_flag=True 行。_"}

## Fallacy Scan

```text
11/11 checked.
Simpson's paradox：已用分组 delta 审计；多 seed 复验只在新 carrier 通过 seed43 快筛后开放。
Ecological fallacy：结果单位是 item-level，不把组均值直接写成单 item 因果。
Berkson/collider：当前为训练后观察性比较，不能写成因果提升。
Base rate neglect：HR/NDCG 同时报告，分组 item_count 同时报告。
Regression to mean：只用单 seed，存在波动风险。
Survivorship bias：四方法均完整 100 epoch，无训练中途丢失。
Look-elsewhere/garden of forking paths：本次比较对象来自预先 Task4 设计，但分组解释仍应标为探索性。
Correlation causation/reverse causality：不能写成 Acat_v3 直接导致推荐提升，只能写当前 carrier 响应证据不足。
```

## 路线判断

```text
最终路线：{decision["route"]}
```

不要直接进入 M4。下一步应优先：

```text
1. revise_m3：调整 Acat_v3 + hard carrier（方法承载方式），避免与 shuffle 负控几乎无差异。
2. new carrier：优先找新的 carrier（方法承载方式），尤其是更明确的 pairwise margin carrier（成对边际损失承载方式）。
3. seed43 screen：所有候选先用 seed43 快筛；只有相对 M6 shuffle 有明显 overall 优势且 HR@20 不反向，才进入多 seed 复验。
```
"""
    output_path.write_text(markdown, encoding="utf-8")


def write_route_markdown(
    output_path: Path,
    run_stamp: str,
    result_md_name: str,
    decision: dict,
    comparison: pd.DataFrame,
) -> None:
    m3_vs_m6 = _comparison_row(comparison, "M3_acat_trainhard_minus_M6_acat_shuffle")
    delta = safe_float(m3_vs_m6["delta_ndcg@20"]) if m3_vs_m6 is not None else nan_float()
    hr_delta = safe_float(m3_vs_m6["delta_hr@20"]) if m3_vs_m6 is not None else nan_float()
    markdown = f"""---
title: {run_stamp} CCFCRec Amazon-VG Task4-post Acat v3 minimal weight controls 路线判断
date: 2026-07-06
tags:
  - CCFCRec
  - Amazon-VG
  - Task4
  - 路线判断
---

# {run_stamp} CCFCRec Amazon-VG Task4-post Acat v3 minimal weight controls 路线判断

结果来源：[[{result_md_name}]]

## 判断

```text
route = {decision["route"]}
go_m4 = {decision["go_m4"]}
need_seed_repeat = {decision["need_seed_repeat"]}
seed_repeat_gate_status = {decision.get("seed_repeat_gate_status", "")}
next_action = {decision.get("next_action", "")}
```

关键原因：

```text
M3 - M6 validate NDCG@20 = {delta:.8f}
M3 - M6 validate HR@20 = {hr_delta:.8f}
```

这不是一个足够干净的 go_m4（进入 M4）信号。M3（Acat_v3 + 训练安全困难样本加权）虽然在 validate NDCG@20 上是最高，但相对 M6 shuffle（Acat_v3 打乱负控）的差距几乎为零，且 HR@20 反向。
因此当前 M3 不应进入大规模多 seed（多随机种子）复验；多 seed 只用于已经通过 seed43 快筛的新候选。

## 下一步

```text
1. 不直接进入 M4。
2. 先把当前 carrier 记为 revise_m3。
3. 优先 revise_m3 或找新 carrier，用 seed43 快速筛选。
4. 只有当新候选相对 M6 shuffle 有明显 overall 优势且 HR@20 不反向，再做多 seed 稳定性复验。
```
"""
    output_path.write_text(markdown, encoding="utf-8")


def run(args: argparse.Namespace) -> AnalysisOutputs:
    code_root = Path(args.code_root).expanduser().resolve() if args.code_root else Path(__file__).resolve().parents[1]
    project_root = code_root.parent
    result_root = Path(args.result_root).expanduser().resolve()
    task4_profile = Path(args.task4_profile_path).expanduser().resolve()
    data_dir = Path(args.data_dir).expanduser().resolve() if args.data_dir else code_root / "Amazon VG" / "data"
    run_stamp, run_date, run_iso = now_stamp()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else project_root
        / "temp_202607_实验文件记录"
        / "temp_20260706"
        / f"{run_stamp} task4-post-acat-v3-weight-controls-analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_index = build_run_index(result_root)
    best_table = build_best_checkpoint_table(run_index)
    dynamics = build_training_dynamics(run_index)
    comparison = build_method_comparison(best_table, min_meaningful_ndcg_delta=args.min_meaningful_ndcg_delta)
    result_summary = build_result_summary(best_table, dynamics)

    item_eval_profile = pd.DataFrame()
    group_effect = pd.DataFrame()
    group_delta = pd.DataFrame()
    device_name = select_device(args.device)
    if not args.skip_item_eval:
        item_eval = evaluate_best_checkpoints_item_level(
            best_table=best_table,
            code_root=code_root,
            data_dir=data_dir,
            device_name=device_name,
            batch_size=args.item_eval_batch_size,
        )
        item_eval_profile = merge_task4_profile(item_eval, task4_profile)
        group_effect = build_group_effect_summary(item_eval_profile)
        group_delta = build_group_delta_summary(group_effect)

    decision = build_route_decision(
        best_table=best_table,
        comparison=comparison,
        dynamics=dynamics,
        group_delta_summary=group_delta,
        min_meaningful_ndcg_delta=args.min_meaningful_ndcg_delta,
    )

    outputs = AnalysisOutputs(
        output_dir=output_dir,
        run_index_csv=output_dir / "task4_post_run_index.csv",
        result_summary_csv=output_dir / "task4_post_result_summary.csv",
        best_checkpoint_csv=output_dir / "task4_post_best_checkpoint_table.csv",
        method_comparison_csv=output_dir / "task4_post_method_comparison.csv",
        training_dynamics_csv=output_dir / "task4_post_training_dynamics.csv",
        item_eval_csv=output_dir / "task4_post_item_eval_profile.csv",
        group_effect_csv=output_dir / "task4_post_group_effect_summary.csv",
        group_delta_csv=output_dir / "task4_post_group_delta_summary.csv",
        manifest_json=output_dir / "run_manifest.json",
        result_md=output_dir / f"{run_stamp} CCFCRec Amazon-VG Task4-post Acat v3 minimal weight controls 训练后分析结果.md",
    )

    run_index.to_csv(outputs.run_index_csv, index=False)
    result_summary.to_csv(outputs.result_summary_csv, index=False)
    best_table.to_csv(outputs.best_checkpoint_csv, index=False)
    comparison.to_csv(outputs.method_comparison_csv, index=False)
    dynamics.to_csv(outputs.training_dynamics_csv, index=False)
    item_eval_profile.to_csv(outputs.item_eval_csv, index=False)
    group_effect.to_csv(outputs.group_effect_csv, index=False)
    group_delta.to_csv(outputs.group_delta_csv, index=False)

    write_result_markdown(
        output_path=outputs.result_md,
        run_stamp=run_stamp,
        design_note_name=DESIGN_NOTE_NAME,
        result_root=result_root,
        result_summary=result_summary,
        best_table=best_table,
        comparison=comparison,
        dynamics=dynamics,
        group_delta=group_delta,
        decision=decision,
        manifest_name=outputs.manifest_json.name,
    )

    route_md = project_root / "实验记录" / f"{run_stamp} CCFCRec Amazon-VG Task4-post Acat v3 minimal weight controls 路线判断.md"
    write_route_markdown(
        output_path=route_md,
        run_stamp=run_stamp,
        result_md_name=outputs.result_md.stem,
        decision=decision,
        comparison=comparison,
    )

    manifest = {
        "run_stamp": run_stamp,
        "run_iso": run_iso,
        "experiment_id": EXPERIMENT_ID,
        "experiment_stage": "Task4-post",
        "training_launcher_script": str(code_root / TRAINING_LAUNCHER_SCRIPT),
        "analysis_script": str(code_root / ANALYSIS_SCRIPT),
        "design_note": DESIGN_NOTE_NAME,
        "result_root": str(result_root),
        "task4_profile_path": str(task4_profile),
        "code_root": str(code_root),
        "data_dir": str(data_dir),
        "device_requested": args.device,
        "device_used_for_item_eval": device_name if not args.skip_item_eval else "skipped",
        "item_eval_batch_size": args.item_eval_batch_size,
        "skip_item_eval": bool(args.skip_item_eval),
        "min_meaningful_ndcg_delta": args.min_meaningful_ndcg_delta,
        "decision": decision,
        "outputs": {field: str(getattr(outputs, field)) for field in outputs.__dataclass_fields__ if field != "output_dir"},
        "route_markdown": str(route_md),
    }
    outputs.manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-root",
        default="/Volumes/MyPassport/CCFCRec对比学习思路硬盘/实验记录硬盘/ccfcrec_result/2026-07-06_015834_task4_acat_v3_weight_controls_m1_m2_m6_m3_seed43_workers8_fast_uniform_mps_100epoch",
    )
    parser.add_argument(
        "--task4-profile-path",
        default="/Users/luojiaqiang/Documents/Obsidian Vault/科研/CCFCRec对比学习思路/temp_202607_实验文件记录/temp_20260706/2026-07-06 004222 task4-pre3-train-safe-hard-proxy/task4_train_safe_hard_proxy_profile.csv",
    )
    parser.add_argument("--code-root", default="")
    parser.add_argument("--data-dir", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--item-eval-batch-size", type=int, default=256)
    parser.add_argument("--min-meaningful-ndcg-delta", type=float, default=0.0005)
    parser.add_argument("--skip-item-eval", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = run(args)
    print(f"wrote {outputs.result_summary_csv}")
    print(f"wrote {outputs.best_checkpoint_csv}")
    print(f"wrote {outputs.method_comparison_csv}")
    print(f"wrote {outputs.result_md}")


if __name__ == "__main__":
    main()
