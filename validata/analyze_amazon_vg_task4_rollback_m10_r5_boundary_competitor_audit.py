#!/usr/bin/env python3
"""
CCFCRec Amazon-VG M10-R5 boundary competitor offline audit.

This script mines train-safe boundary competitor users from the seed43
workers8-fast_uniform baseline checkpoint. Candidate generation uses only train
interactions and train-safe Task4 masks; eval/test columns are not used as
training inputs.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import pickle
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


PROJECT_ROOT = Path("/Users/luojiaqiang/Documents/Obsidian Vault/科研/CCFCRec对比学习思路")
CODE_ROOT = PROJECT_ROOT / "CCFCRec-code"
AMAZON_VG_DIR = CODE_ROOT / "Amazon VG"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "temp_202607_实验文件记录" / "temp_20260709"
DEFAULT_TASK4_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录"
    / "temp_20260706"
    / "2026-07-06 004222 task4-pre3-train-safe-hard-proxy"
    / "task4_train_safe_hard_proxy_profile.csv"
)
DEFAULT_BASELINE_PACKAGE = (
    PROJECT_ROOT
    / "实验记录"
    / "2026-06-27 162750 baseline-seed43-workers8-fast-uniform-commit873a171"
    / "baseline_seed43_workers8_fast_uniform_commit873a171_2026-06-27_15_54_28_slim_best74_last100.tar.gz"
)
DEFAULT_TRAIN_RATING = AMAZON_VG_DIR / "data" / "train_rating.csv"
DEFAULT_TRAIN_WITHNEG_RATING = AMAZON_VG_DIR / "data" / "train_withneg_rating.csv"
ANALYSIS_SCRIPT = "validata/analyze_amazon_vg_task4_rollback_m10_r5_boundary_competitor_audit.py"
DESIGN_NOTE_NAME = "2026-07-09 130000 CCFCRec Amazon-VG M10-R5 boundary competitor sampling 代码阅读与离线审计设计"
R4_ROUTE_NOTE_NAME = "2026-07-09 120712 CCFCRec Amazon-VG M10-R4 competitor pair training 路线判断"

TRAIN_INPUT_COLUMNS_USED = [
    "raw_asin",
    "split",
    "cat_count_bin",
    "category_count",
    "high_acat_train_safe_hard_flag",
    "train_safe_hard_proxy_score",
    "s_cat_v3",
    "s_cat_v3_group",
    "high_acat_flag",
    "RSP_group",
    "RSP_score",
]
FORBIDDEN_TRAIN_COLUMNS = {
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
}


@dataclass(frozen=True)
class Outputs:
    output_dir: Path
    boundary_cache_csv: Path
    boundary_audit_profile_csv: Path
    branch_summary_csv: Path
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


def _numeric(series: pd.Series | None, index: pd.Index, default: float = 0.0) -> pd.Series:
    if series is None:
        return pd.Series(default, index=index, dtype=float)
    return pd.to_numeric(series, errors="coerce").fillna(default).reindex(index, fill_value=default)


def _clip01(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)


def _lookup_serial_id(mapping: dict[Any, int], raw_value: Any) -> int | None:
    if pd.isna(raw_value):
        return None
    text = str(raw_value).strip()
    candidates: list[Any] = [raw_value, text]
    if text:
        try:
            int_value = int(text)
            candidates.extend([int_value, np.int64(int_value)])
        except ValueError:
            pass
    for candidate in candidates:
        if candidate in mapping:
            return int(mapping[candidate])
    return None


def _read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def load_task4_profile(path: str | Path) -> pd.DataFrame:
    profile = _read_csv(path)
    if "raw_asin" not in profile.columns:
        if "asin" in profile.columns:
            profile = profile.rename(columns={"asin": "raw_asin"})
        else:
            raise ValueError("Task4 profile missing raw_asin")
    return profile


def add_branch_flags(profile: pd.DataFrame, shuffle_seed: int = 43) -> pd.DataFrame:
    work = profile.copy().sort_values("raw_asin").reset_index(drop=True)
    index = work.index
    if "split" not in work.columns:
        raise ValueError("Task4 profile missing split")
    split_train = work["split"].astype(str).eq("train")
    if "cat_count_bin" in work.columns:
        high_detail = work["cat_count_bin"].astype(str).eq("cat_count_5_plus")
    elif "category_count" in work.columns:
        high_detail = _numeric(work["category_count"], index).ge(5)
    else:
        raise ValueError("Task4 profile missing cat_count_bin/category_count")
    if "high_acat_train_safe_hard_flag" not in work.columns:
        raise ValueError("Task4 profile missing high_acat_train_safe_hard_flag")
    trainhard = _bool_series(work["high_acat_train_safe_hard_flag"], index)
    if "high_acat_flag" in work.columns:
        high_acat = _bool_series(work["high_acat_flag"], index)
    elif "s_cat_v3_group" in work.columns:
        high_acat = work["s_cat_v3_group"].astype(str).eq("s_cat_v3_strong")
    else:
        raise ValueError("Task4 profile missing high_acat_flag/s_cat_v3_group")
    rsp_high = work["RSP_group"].astype(str).eq("RSP_high") if "RSP_group" in work.columns else pd.Series(False, index=index)

    real_flags = high_detail & trainhard
    rng = np.random.default_rng(shuffle_seed)
    shuffled_flags = pd.Series(False, index=index)
    shuffled_scores = pd.Series(0.0, index=index, dtype=float)
    acat_score = _clip01(_numeric(work.get("s_cat_v3"), index))
    hard_score = _clip01(_numeric(work.get("train_safe_hard_proxy_score"), index))
    real_scores = (acat_score + hard_score) / 2.0
    group_cols = [col for col in ["split", "cat_count_bin"] if col in work.columns]
    groups = work.groupby(group_cols, dropna=False).groups if group_cols else {None: index}
    for _, group_index in groups.items():
        group_index = pd.Index(group_index)
        permutation = rng.permutation(len(group_index))
        shuffled_flags.loc[group_index] = real_flags.loc[group_index].to_numpy(dtype=bool)[permutation]
        shuffled_scores.loc[group_index] = real_scores.loc[group_index].to_numpy(dtype=float)[permutation]

    work["r5_real_target_flag"] = (split_train & real_flags).map(bool).astype(object)
    work["r5_shuffle_target_flag"] = (split_train & high_detail & shuffled_flags).map(bool).astype(object)
    work["r5_rsp_control_target_flag"] = (split_train & high_detail & rsp_high).map(bool).astype(object)
    work["r5_acat_control_target_flag"] = (split_train & high_detail & high_acat).map(bool).astype(object)
    work["r5_union_target_flag"] = (
        work["r5_real_target_flag"].astype(bool)
        | work["r5_shuffle_target_flag"].astype(bool)
        | work["r5_rsp_control_target_flag"].astype(bool)
        | work["r5_acat_control_target_flag"].astype(bool)
    ).map(bool).astype(object)
    work["r5_real_score"] = real_scores
    work["r5_shuffle_score"] = shuffled_scores
    work["r5_rsp_score"] = _clip01(_numeric(work.get("RSP_score"), index, default=1.0))
    work["r5_acat_score"] = acat_score
    return work


def _tar_member(tar: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    matches = [member for member in tar.getmembers() if member.name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one tar member ending with {suffix}, got {len(matches)}")
    return matches[0]


def load_json_from_tar(package_path: str | Path, suffix: str) -> dict[str, Any]:
    with tarfile.open(package_path, "r:gz") as tar:
        member = _tar_member(tar, suffix)
        with tar.extractfile(member) as file_obj:
            if file_obj is None:
                raise ValueError(f"cannot read tar member: {member.name}")
            return json.loads(file_obj.read().decode("utf-8"))


def load_pickle_from_tar(package_path: str | Path, suffix: str) -> dict[str, Any]:
    with tarfile.open(package_path, "r:gz") as tar:
        member = _tar_member(tar, suffix)
        with tar.extractfile(member) as file_obj:
            if file_obj is None:
                raise ValueError(f"cannot read tar member: {member.name}")
            return pickle.load(file_obj)


def load_torch_state_from_tar(package_path: str | Path, suffix: str) -> dict[str, torch.Tensor]:
    with tarfile.open(package_path, "r:gz") as tar:
        member = _tar_member(tar, suffix)
        with tar.extractfile(member) as file_obj:
            if file_obj is None:
                raise ValueError(f"cannot read tar member: {member.name}")
            return torch.load(io.BytesIO(file_obj.read()), map_location="cpu")


def build_model_args_from_state_dict(state_dict: dict[str, torch.Tensor], run_config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        attr_num=int(state_dict["attr_matrix"].shape[0]),
        attr_present_dim=int(state_dict["attr_matrix"].shape[1]),
        implicit_dim=int(state_dict["user_embedding"].shape[1]),
        cat_implicit_dim=int(state_dict["gen_layer1.weight"].shape[0]),
        user_number=int(state_dict["user_embedding"].shape[0]),
        item_number=int(state_dict["item_embedding"].shape[0]),
        pretrain=False,
        pretrain_update=False,
        method_variant=str(run_config.get("method_variant", "baseline")),
        category_conf_dim=int(run_config.get("category_conf_dim", 16)),
        category_conf_max_count=int(run_config.get("category_conf_max_count", 5)),
        category_gate_scale=float(run_config.get("category_gate_scale", 0.5)),
    )


def resolve_device_name(requested: str) -> str:
    requested = requested.strip().lower()
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if requested == "cuda" and torch.cuda.is_available():
        return "cuda"
    if requested == "cpu":
        return "cpu"
    raise RuntimeError(f"unsupported or unavailable device: {requested}")


def load_baseline_model(package_path: str | Path, device_name: str):
    os.environ["CCFCREC_DEVICE"] = device_name
    if str(AMAZON_VG_DIR) not in sys.path:
        sys.path.insert(0, str(AMAZON_VG_DIR))
    from model import CCFCRec

    run_config = load_json_from_tar(package_path, "/run_config.json")
    state_dict = load_torch_state_from_tar(package_path, "/best_epoch_74.pt")
    model_args = build_model_args_from_state_dict(state_dict, run_config)
    model = CCFCRec(model_args)
    model.load_state_dict(state_dict)
    model = model.to(torch.device(device_name))
    model.eval()
    return model, run_config, state_dict


def build_item_positive_user_sets(
    train_rating: pd.DataFrame,
    user_serialize_dict: dict[Any, int],
    item_serialize_dict: dict[Any, int],
) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for _, row in train_rating.iterrows():
        serial_item = _lookup_serial_id(item_serialize_dict, row["asin"])
        serial_user = _lookup_serial_id(user_serialize_dict, row["reviewerID"])
        if serial_item is None or serial_user is None:
            continue
        result.setdefault(serial_item, set()).add(serial_user)
    return result


def build_existing_neg_user_sets(
    train_withneg_rating: pd.DataFrame,
    user_serialize_dict: dict[Any, int],
    item_serialize_dict: dict[Any, int],
) -> dict[int, set[int]]:
    result: dict[int, set[int]] = {}
    for _, row in train_withneg_rating.iterrows():
        serial_item = _lookup_serial_id(item_serialize_dict, row["asin"])
        serial_user = _lookup_serial_id(user_serialize_dict, row["neg_user"])
        if serial_item is None or serial_user is None:
            continue
        result.setdefault(serial_item, set()).add(serial_user)
    return result


def build_target_item_table(
    flagged_profile: pd.DataFrame,
    item_serialize_dict: dict[Any, int],
    max_target_items: int = 0,
    seed: int = 43,
) -> pd.DataFrame:
    targets = flagged_profile[flagged_profile["r5_union_target_flag"].astype(bool)].copy()
    targets["serial_item_id"] = targets["raw_asin"].map(lambda raw: _lookup_serial_id(item_serialize_dict, raw))
    targets = targets.dropna(subset=["serial_item_id"]).drop_duplicates("serial_item_id").copy()
    targets["serial_item_id"] = targets["serial_item_id"].astype(int)
    targets = targets.sort_values(["serial_item_id", "raw_asin"]).reset_index(drop=True)
    if max_target_items and len(targets) > max_target_items:
        targets = targets.sample(n=max_target_items, random_state=seed).sort_values("serial_item_id").reset_index(drop=True)
    return targets


def select_boundary_from_score_row(
    scores: np.ndarray,
    positive_users: set[int],
    existing_neg_users: set[int],
    serial_user_to_raw: dict[int, Any],
    top_user_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    if top_user_ids is None:
        top_user_ids = np.argsort(-scores)
    selected_user: int | None = None
    selected_rank: int | None = None
    selected_score: float | None = None
    for rank_idx, user_id in enumerate(top_user_ids, start=1):
        user_id = int(user_id)
        if user_id not in positive_users:
            selected_user = user_id
            selected_rank = rank_idx
            selected_score = float(scores[user_id])
            break

    positive_list = [user for user in sorted(positive_users) if 0 <= user < len(scores)]
    positive_scores = scores[positive_list] if positive_list else np.asarray([], dtype=np.float32)
    best_positive_score = float(np.max(positive_scores)) if positive_scores.size else float("nan")
    mean_positive_score = float(np.mean(positive_scores)) if positive_scores.size else float("nan")

    existing_list = [user for user in sorted(existing_neg_users) if 0 <= user < len(scores)]
    existing_scores = scores[existing_list] if existing_list else np.asarray([], dtype=np.float32)
    best_existing_neg_score = float(np.max(existing_scores)) if existing_scores.size else float("nan")
    best_existing_neg_user = int(existing_list[int(np.argmax(existing_scores))]) if existing_scores.size else None
    best_existing_neg_rank = int(np.sum(scores > best_existing_neg_score) + 1) if existing_scores.size else None

    return {
        "candidate_found_flag": selected_user is not None,
        "boundary_competitor_serial_user": selected_user,
        "boundary_competitor_user": None if selected_user is None else serial_user_to_raw.get(int(selected_user)),
        "boundary_rank": selected_rank,
        "boundary_score": selected_score,
        "positive_user_count": int(len(positive_users)),
        "best_train_positive_score": best_positive_score,
        "mean_train_positive_score": mean_positive_score,
        "boundary_minus_best_positive": float("nan") if selected_score is None else selected_score - best_positive_score,
        "boundary_minus_mean_positive": float("nan") if selected_score is None else selected_score - mean_positive_score,
        "existing_neg_user_count": int(len(existing_neg_users)),
        "best_existing_neg_serial_user": best_existing_neg_user,
        "best_existing_neg_user": None if best_existing_neg_user is None else serial_user_to_raw.get(best_existing_neg_user),
        "best_existing_neg_score": best_existing_neg_score,
        "best_existing_neg_rank": best_existing_neg_rank,
        "boundary_minus_best_existing_neg": float("nan") if selected_score is None else selected_score - best_existing_neg_score,
        "leak_safe_flag": bool(selected_user is None or selected_user not in positive_users),
    }


def mine_boundary_competitors(
    model,
    target_items: pd.DataFrame,
    item_category_tensor: torch.Tensor,
    item_image_feature_tensor: torch.Tensor,
    positive_user_sets: dict[int, set[int]],
    existing_neg_user_sets: dict[int, set[int]],
    serial_user_to_raw: dict[int, Any],
    device_name: str,
    scan_k: int,
    batch_size: int,
) -> pd.DataFrame:
    device = torch.device(device_name)
    user_number = int(model.user_embedding.shape[0])
    top_k = min(int(scan_k), user_number)
    rows: list[dict[str, Any]] = []
    model.eval()
    item_category_tensor = item_category_tensor.to(device)
    item_image_feature_tensor = item_image_feature_tensor.to(device)
    for batch_start in tqdm(range(0, len(target_items), batch_size), desc="mine boundary competitors"):
        batch = target_items.iloc[batch_start: batch_start + batch_size].copy()
        serial_items = torch.as_tensor(batch["serial_item_id"].to_numpy(dtype=np.int64), dtype=torch.long, device=device)
        genres = item_category_tensor[serial_items]
        image_features = item_image_feature_tensor[serial_items]
        with torch.no_grad():
            q_v_c = model(genres, image_features, genres.shape[0])
            score_tensor = torch.matmul(q_v_c, model.user_embedding.t())
            _, top_indices = torch.topk(score_tensor, k=top_k, dim=1)
        scores_np = score_tensor.detach().cpu().numpy()
        top_np = top_indices.detach().cpu().numpy()
        for local_idx, (_, source_row) in enumerate(batch.iterrows()):
            serial_item = int(source_row["serial_item_id"])
            selected = select_boundary_from_score_row(
                scores=scores_np[local_idx],
                positive_users=positive_user_sets.get(serial_item, set()),
                existing_neg_users=existing_neg_user_sets.get(serial_item, set()),
                serial_user_to_raw=serial_user_to_raw,
                top_user_ids=top_np[local_idx],
            )
            rows.append(
                {
                    "raw_asin": source_row["raw_asin"],
                    "serial_item_id": serial_item,
                    "r5_real_target_flag": bool(source_row["r5_real_target_flag"]),
                    "r5_shuffle_target_flag": bool(source_row["r5_shuffle_target_flag"]),
                    "r5_rsp_control_target_flag": bool(source_row["r5_rsp_control_target_flag"]),
                    "r5_acat_control_target_flag": bool(source_row["r5_acat_control_target_flag"]),
                    **selected,
                }
            )
    out = pd.DataFrame(rows)
    for col in ["candidate_found_flag", "leak_safe_flag", "r5_real_target_flag", "r5_shuffle_target_flag", "r5_rsp_control_target_flag", "r5_acat_control_target_flag"]:
        if col in out.columns:
            out[col] = out[col].map(bool).astype(object)
    return out


def summarize_boundary_audit(profile: pd.DataFrame) -> pd.DataFrame:
    scopes = [
        ("overall", "all", pd.Series(True, index=profile.index)),
        ("branch", "real", profile["r5_real_target_flag"].astype(bool)),
        ("branch", "shuffle", profile["r5_shuffle_target_flag"].astype(bool)),
        ("branch", "rsp_control", profile["r5_rsp_control_target_flag"].astype(bool)),
        ("branch", "acat_control", profile["r5_acat_control_target_flag"].astype(bool)),
    ]
    rows: list[dict[str, Any]] = []
    for scope, group, mask in scopes:
        sub = profile[mask].copy()
        if sub.empty:
            continue
        found = sub["candidate_found_flag"].astype(bool)
        leak_safe = sub["leak_safe_flag"].astype(bool)
        boundary_rank = pd.to_numeric(sub["boundary_rank"], errors="coerce")
        gain = pd.to_numeric(sub["boundary_minus_best_existing_neg"], errors="coerce")
        pos_gap = pd.to_numeric(sub["boundary_minus_mean_positive"], errors="coerce")
        rows.append(
            {
                "scope": scope,
                "group": group,
                "item_count": int(len(sub)),
                "candidate_found_rate": float(found.mean()),
                "leak_violation_count": int((~leak_safe).sum()),
                "boundary_rank_median": float(boundary_rank.median()),
                "boundary_rank_p90": float(boundary_rank.quantile(0.90)),
                "boundary_top20_rate": float(boundary_rank.le(20).mean()),
                "boundary_top100_rate": float(boundary_rank.le(100).mean()),
                "existing_neg_available_rate": float(pd.to_numeric(sub["existing_neg_user_count"], errors="coerce").gt(0).mean()),
                "boundary_minus_best_existing_neg_mean": float(gain.mean()),
                "boundary_beats_best_existing_neg_rate": float(gain.gt(0).mean()),
                "boundary_minus_mean_positive_mean": float(pos_gap.mean()),
            }
        )
    return pd.DataFrame(rows)


def decide_route(summary: pd.DataFrame, audit_profile: pd.DataFrame, max_target_items: int) -> dict[str, Any]:
    real = summary[(summary["scope"].eq("branch")) & (summary["group"].eq("real"))]
    overall = summary[(summary["scope"].eq("overall")) & (summary["group"].eq("all"))]
    real_row = real.iloc[0].to_dict() if not real.empty else {}
    overall_row = overall.iloc[0].to_dict() if not overall.empty else {}
    sampled = bool(max_target_items and len(audit_profile) >= max_target_items)
    gates = {
        "not_sampled_full_cache": not sampled,
        "real_item_count_ge_1000": int(real_row.get("item_count", 0)) >= 1000,
        "real_candidate_found_rate_ge_0_98": float(real_row.get("candidate_found_rate", 0.0)) >= 0.98,
        "overall_leak_violation_count_eq_0": int(overall_row.get("leak_violation_count", 999999)) == 0,
        "real_boundary_top100_rate_ge_0_95": float(real_row.get("boundary_top100_rate", 0.0)) >= 0.95,
        "real_beats_existing_neg_rate_ge_0_80": float(real_row.get("boundary_beats_best_existing_neg_rate", 0.0)) >= 0.80,
    }
    feasible = all(gates.values())
    return {
        "route": "r5_boundary_competitor_cache_feasible_training_node" if feasible else "r5_boundary_competitor_needs_more_audit_or_redesign",
        "training_node_reached": bool(feasible),
        "open_formal_100epoch": bool(feasible),
        "max_target_items": int(max_target_items),
        "sampled_audit": sampled,
        "gates": gates,
        "real_summary": _jsonable(real_row),
        "overall_summary": _jsonable(overall_row),
        "train_input_columns_used": TRAIN_INPUT_COLUMNS_USED,
        "forbidden_train_columns_used": sorted(set(TRAIN_INPUT_COLUMNS_USED).intersection(FORBIDDEN_TRAIN_COLUMNS)),
        "expected_training_branches": [
            "task4_boundary_competitor_pair",
            "task4_boundary_competitor_pair_shuffle",
            "task4_boundary_competitor_pair_rsp_control",
            "task4_boundary_competitor_pair_acat_control",
        ],
    }


def build_outputs(output_root: str | Path, stamp: str) -> Outputs:
    output_dir = Path(output_root) / f"{stamp} task4-rollback-m10-r5-boundary-competitor-audit"
    output_dir.mkdir(parents=True, exist_ok=False)
    return Outputs(
        output_dir=output_dir,
        boundary_cache_csv=output_dir / "m10_r5_boundary_competitor_cache.csv",
        boundary_audit_profile_csv=output_dir / "m10_r5_boundary_competitor_audit_profile.csv",
        branch_summary_csv=output_dir / "m10_r5_boundary_competitor_branch_summary.csv",
        route_decision_json=output_dir / "m10_r5_boundary_competitor_route_decision.json",
        manifest_json=output_dir / "manifest.json",
        result_md=output_dir / f"{stamp} CCFCRec Amazon-VG M10-R5 boundary competitor offline audit 结果.md",
    )


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_empty_"
    columns = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for col in frame.columns:
            value = row[col]
            if isinstance(value, float):
                value = "" if math.isnan(value) else f"{value:.12g}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_report(outputs: Outputs, decision: dict[str, Any], summary: pd.DataFrame, manifest: dict[str, Any]) -> None:
    real_summary = decision.get("real_summary", {})
    lines = [
        "# CCFCRec Amazon-VG M10-R5 boundary competitor offline audit 结果",
        "",
        "## Material Passport",
        "",
        f"- Artifact Type: Experiment Result",
        f"- Experiment: M10-R5 boundary competitor sampling（边界竞争用户采样）",
        f"- Verification Status: ANALYZED",
        f"- Created At: {manifest['created_at']}",
        f"- Script: `{ANALYSIS_SCRIPT}`",
        f"- Upstream Route: `{R4_ROUTE_NOTE_NAME}.md`",
        "",
        "## Route Decision",
        "",
        f"- route（路线）: `{decision['route']}`",
        f"- training_node_reached（到达训练节点）: `{decision['training_node_reached']}`",
        f"- open_formal_100epoch（可进入正式100轮训练）: `{decision['open_formal_100epoch']}`",
        f"- sampled_audit（抽样审计）: `{decision['sampled_audit']}`",
        f"- forbidden_train_columns_used（误用禁止训练字段）: `{decision['forbidden_train_columns_used']}`",
        "",
        "## Key Metrics",
        "",
        f"- real item_count（真实分支物品数）: {real_summary.get('item_count')}",
        f"- real candidate_found_rate（真实分支候选覆盖率）: {real_summary.get('candidate_found_rate')}",
        f"- real boundary_top100_rate（真实分支前100边界率）: {real_summary.get('boundary_top100_rate')}",
        f"- real beats_existing_neg_rate（真实分支强于原负用户比例）: {real_summary.get('boundary_beats_best_existing_neg_rate')}",
        f"- overall leak_violation_count（总体泄漏数）: {decision.get('overall_summary', {}).get('leak_violation_count')}",
        "",
        "## Outputs",
        "",
        f"- boundary cache（边界缓存）: `{outputs.boundary_cache_csv}`",
        f"- audit profile（审计明细）: `{outputs.boundary_audit_profile_csv}`",
        f"- branch summary（分支摘要）: `{outputs.branch_summary_csv}`",
        f"- route decision（路线判断）: `{outputs.route_decision_json}`",
        "",
        "## Branch Summary",
        "",
        dataframe_to_markdown(summary),
        "",
        "## Interpretation",
        "",
        "Candidate mining（候选挖掘）只排除 train positives（训练正用户），没有读取验证/测试标签作为训练输入。"
        "如果 route（路线）为 `r5_boundary_competitor_cache_feasible_training_node`，下一步是用生成的 cache（缓存）启动四分支训练。",
    ]
    outputs.result_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> Outputs:
    expected_hash_seed = str(args.expected_pythonhashseed)
    actual_hash_seed = os.environ.get("PYTHONHASHSEED")
    if actual_hash_seed != expected_hash_seed:
        raise RuntimeError(
            f"PYTHONHASHSEED must be {expected_hash_seed} before process start; got {actual_hash_seed!r}. "
            "Re-run with PYTHONHASHSEED=43."
        )
    stamp, _, created_at = now_stamp()
    outputs = build_outputs(args.output_root, stamp)
    device_name = resolve_device_name(args.device)

    flagged_profile = add_branch_flags(load_task4_profile(args.task4_profile), shuffle_seed=args.shuffle_seed)
    train_df = _read_csv(args.train_rating)
    train_withneg_df = _read_csv(args.train_withneg_rating)
    save_dict = load_pickle_from_tar(args.baseline_package, "/save_dict.pkl")
    user_ser_dict = save_dict["user_ser_dict"]
    serial_user_to_raw = {int(serial): raw for raw, serial in user_ser_dict.items()}

    if str(AMAZON_VG_DIR) not in sys.path:
        sys.path.insert(0, str(AMAZON_VG_DIR))
    from support import build_item_feature_tensors, serialize_item

    item_ser_dict = serialize_item(train_df["asin"])
    model, run_config, state_dict = load_baseline_model(args.baseline_package, device_name)
    if len(item_ser_dict) != int(state_dict["item_embedding"].shape[0]):
        raise ValueError(
            f"item mapping length mismatch: rebuilt={len(item_ser_dict)} checkpoint={state_dict['item_embedding'].shape[0]}"
        )
    target_items = build_target_item_table(
        flagged_profile,
        item_ser_dict,
        max_target_items=args.max_target_items,
        seed=args.seed,
    )
    positive_user_sets = build_item_positive_user_sets(train_df, user_ser_dict, item_ser_dict)
    existing_neg_user_sets = build_existing_neg_user_sets(train_withneg_df, user_ser_dict, item_ser_dict)
    item_category_tensor, item_image_feature_tensor = build_item_feature_tensors(
        item_serialize_dict=item_ser_dict,
        img_features=save_dict["img_feature_dict"],
        genres=save_dict["asin_category_int_map"],
        category_num=int(save_dict["category_ser_map_len"]),
    )
    audit_profile = mine_boundary_competitors(
        model=model,
        target_items=target_items,
        item_category_tensor=item_category_tensor,
        item_image_feature_tensor=item_image_feature_tensor,
        positive_user_sets=positive_user_sets,
        existing_neg_user_sets=existing_neg_user_sets,
        serial_user_to_raw=serial_user_to_raw,
        device_name=device_name,
        scan_k=args.scan_k,
        batch_size=args.score_batch_size,
    )
    summary = summarize_boundary_audit(audit_profile)
    decision = decide_route(summary, audit_profile, max_target_items=args.max_target_items)
    cache_cols = [
        "raw_asin",
        "serial_item_id",
        "boundary_competitor_user",
        "boundary_competitor_serial_user",
        "boundary_rank",
        "boundary_score",
        "r5_real_target_flag",
        "r5_shuffle_target_flag",
        "r5_rsp_control_target_flag",
        "r5_acat_control_target_flag",
        "candidate_found_flag",
        "leak_safe_flag",
    ]
    audit_profile[cache_cols].to_csv(outputs.boundary_cache_csv, index=False)
    audit_profile.to_csv(outputs.boundary_audit_profile_csv, index=False)
    summary.to_csv(outputs.branch_summary_csv, index=False)
    outputs.route_decision_json.write_text(json.dumps(_jsonable(decision), ensure_ascii=False, indent=2), encoding="utf-8")
    manifest = {
        "created_at": created_at,
        "script": ANALYSIS_SCRIPT,
        "task4_profile": str(args.task4_profile),
        "baseline_package": str(args.baseline_package),
        "baseline_run_config": _jsonable(run_config),
        "train_rating": str(args.train_rating),
        "train_withneg_rating": str(args.train_withneg_rating),
        "device": device_name,
        "score_batch_size": int(args.score_batch_size),
        "scan_k": int(args.scan_k),
        "max_target_items": int(args.max_target_items),
        "seed": int(args.seed),
        "shuffle_seed": int(args.shuffle_seed),
        "pythonhashseed": actual_hash_seed,
        "target_item_count": int(len(target_items)),
        "train_input_columns_used": TRAIN_INPUT_COLUMNS_USED,
        "forbidden_train_columns_used": decision["forbidden_train_columns_used"],
    }
    outputs.manifest_json.write_text(json.dumps(_jsonable(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(outputs, decision, summary, manifest)
    print(json.dumps(_jsonable({"output_dir": str(outputs.output_dir), **decision}), ensure_ascii=False, indent=2))
    return outputs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task4-profile", type=Path, default=DEFAULT_TASK4_PROFILE)
    parser.add_argument("--baseline-package", type=Path, default=DEFAULT_BASELINE_PACKAGE)
    parser.add_argument("--train-rating", type=Path, default=DEFAULT_TRAIN_RATING)
    parser.add_argument("--train-withneg-rating", type=Path, default=DEFAULT_TRAIN_WITHNEG_RATING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--score-batch-size", type=int, default=128)
    parser.add_argument("--scan-k", type=int, default=500)
    parser.add_argument("--max-target-items", type=int, default=0, help="0 means full target union")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--shuffle-seed", type=int, default=43)
    parser.add_argument("--expected-pythonhashseed", type=int, default=43)
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())
