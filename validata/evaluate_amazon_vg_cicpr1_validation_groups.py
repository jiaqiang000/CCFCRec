#!/usr/bin/env python3
"""Evaluate CICP-R1 checkpoints on validation cold items only."""

from __future__ import annotations

import argparse
import io
import json
import os
import pickle
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


BASELINE_LABEL = "baseline_seed43_workers8_fast_uniform"
METHOD_LABELS = {
    "cicpr1_e4_residual": "CICPR1E1_e4_residual",
    "cicpr1_modality_routing": "CICPR1E2_modality_routing",
    "cicpr1_category_expert": "CICPR1E3_category_expert",
    "cicpr1_alignment_curriculum": "CICPR1E4_alignment_curriculum",
    "cicpr1_counterfactual_margin": "CICPR1E5_counterfactual_margin",
    "cicpr1_adaptive_attention": "CICPR1E6_adaptive_attention",
}
FORBIDDEN_EVALUATION_COLUMNS = {
    "hr@5",
    "hr@10",
    "hr@20",
    "ndcg@5",
    "ndcg@10",
    "ndcg@20",
    "baseline_hr@20",
    "baseline_ndcg@20",
    "baseline_margin_proxy",
    "baseline_best_target_rank",
    "best_target_rank",
    "eval_baseline_hard_flag",
    "delta_hr@20",
    "delta_ndcg@20",
}


def select_device(requested: str) -> str:
    import torch

    requested = requested.strip().lower()
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested not in {"cpu", "mps", "cuda"}:
        raise ValueError(f"unsupported device: {requested}")
    return requested


def _tar_member(tar: tarfile.TarFile, suffix: str) -> tarfile.TarInfo:
    matches = [member for member in tar.getmembers() if member.name.endswith(suffix)]
    if len(matches) != 1:
        raise ValueError(f"expected one tar member ending with {suffix}, got {len(matches)}")
    return matches[0]


def _read_tar_bytes(tar: tarfile.TarFile, suffix: str) -> bytes:
    file_obj = tar.extractfile(_tar_member(tar, suffix))
    if file_obj is None:
        raise ValueError(f"cannot read tar member ending with {suffix}")
    return file_obj.read()


def _best_row(result: pd.DataFrame) -> pd.Series:
    return result.sort_values(
        ["ndcg@20", "hr@20", "epoch", "checkpoint_index"],
        ascending=[False, False, True, True],
    ).iloc[0]


def load_baseline_package(
    package: Path,
    device_name: str,
) -> tuple[dict[str, Any], dict, dict, pd.Series]:
    import torch

    with tarfile.open(package, "r:gz") as tar:
        config = json.loads(_read_tar_bytes(tar, "/run_config.json").decode("utf-8"))
        save_dict = pickle.loads(_read_tar_bytes(tar, "/save_dict.pkl"))
        result = pd.read_csv(io.BytesIO(_read_tar_bytes(tar, "/result.csv")))
        best = _best_row(result)
        epoch = int(best["epoch"])
        checkpoint = torch.load(
            io.BytesIO(_read_tar_bytes(tar, f"/best_epoch_{epoch}.pt")),
            map_location=device_name,
            weights_only=True,
        )
    return config, save_dict, checkpoint, best


def load_run_package(
    run_dir: Path,
    device_name: str,
) -> tuple[dict[str, Any], dict, dict, pd.Series]:
    import torch

    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    with (run_dir / "save_dict.pkl").open("rb") as file_obj:
        save_dict = pickle.load(file_obj)
    result = pd.read_csv(run_dir / "result.csv")
    best = _best_row(result)
    checkpoint_index = int(best["checkpoint_index"])
    checkpoint = torch.load(
        run_dir / f"{checkpoint_index}.pt",
        map_location=device_name,
        weights_only=True,
    )
    return config, save_dict, checkpoint, best


def build_model_args(state_dict: dict, config: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        attr_num=int(state_dict["attr_matrix"].shape[0]),
        attr_present_dim=int(state_dict["attr_matrix"].shape[1]),
        implicit_dim=int(state_dict["user_embedding"].shape[1]),
        cat_implicit_dim=int(state_dict["gen_layer1.weight"].shape[0]),
        user_number=int(state_dict["user_embedding"].shape[0]),
        item_number=int(state_dict["item_embedding"].shape[0]),
        pretrain=False,
        pretrain_update=False,
        method_variant=str(config.get("method_variant", "baseline")),
        category_conf_dim=int(config.get("category_conf_dim", 16)),
        category_conf_max_count=int(config.get("category_conf_max_count", 5)),
        category_gate_scale=float(config.get("category_gate_scale", 0.5)),
        m11r2_feature_dim=int(config.get("m11r2_feature_dim", 16)),
        m11r3_residual_max_ratio=float(config.get("m11r3_residual_max_ratio", 0.15)),
        m11r3_film_strength=float(config.get("m11r3_film_strength", 0.1)),
        m11r4_expert_film_strength=float(config.get("m11r4_expert_film_strength", 0.2)),
        m11r4_fusion_strength=float(config.get("m11r4_fusion_strength", 0.25)),
        cicp_feature_dim=int(config.get("cicp_feature_dim", 16)),
        cicp_residual_max_ratio=float(config.get("cicp_residual_max_ratio", 0.15)),
        cicp_modality_strength=float(config.get("cicp_modality_strength", 0.25)),
        cicp_expert_strength=float(config.get("cicp_expert_strength", 0.20)),
        cicp_attention_strength=float(config.get("cicp_attention_strength", 0.50)),
        cicpr2_residual_max_ratio=float(config.get("cicpr2_residual_max_ratio", 0.15)),
        cicpr2_increment_strength=float(config.get("cicpr2_increment_strength", 0.50)),
        cicpr2_cross_attention_strength=float(
            config.get("cicpr2_cross_attention_strength", 0.50)
        ),
        cicpr2_cross_attention_temperature=float(
            config.get("cicpr2_cross_attention_temperature", 0.25)
        ),
        cicpr2_distillation_weight=float(config.get("cicpr2_distillation_weight", 0.05)),
        cicpr2_ordinal_weight=float(config.get("cicpr2_ordinal_weight", 0.05)),
        cicpr2_ordinal_margin=float(config.get("cicpr2_ordinal_margin", 0.02)),
        cicpr2_category_dropout_max=float(
            config.get("cicpr2_category_dropout_max", 0.50)
        ),
    )


def load_model(
    code_root: Path,
    state_dict: dict,
    config: dict[str, Any],
    device_name: str,
):
    import torch

    os.environ["CCFCREC_DEVICE"] = device_name
    amazon_code_dir = code_root / "Amazon VG"
    if str(amazon_code_dir) not in sys.path:
        sys.path.insert(0, str(amazon_code_dir))
    from model import CCFCRec

    model = CCFCRec(build_model_args(state_dict, config))
    model.load_state_dict(state_dict)
    model.to(torch.device(device_name))
    model.eval()
    return model


def _dcg(hits: list[float]) -> float:
    values = np.asarray(hits, dtype=float)
    if values.size == 0:
        return 0.0
    return float(
        np.sum((np.power(2.0, values) - 1.0) / np.log2(np.arange(2, values.size + 2)))
    )


def _item_metrics(recommended: list[int], targets: list[int]) -> tuple[float, float]:
    hits = [1.0 if user in set(targets) else 0.0 for user in recommended[:20]]
    ideal = _dcg(sorted(hits, reverse=True))
    return float(sum(hits) / 20.0), float(_dcg(hits) / ideal if ideal > 0 else 0.0)


def _validation_targets(
    validate_csv: Path,
    user_map: dict,
) -> tuple[list[str], dict[str, list[int]]]:
    validation = pd.read_csv(validate_csv, dtype={"asin": str, "reviewerID": str})
    validation["asin"] = validation["asin"].astype(str)
    validation["reviewerID"] = validation["reviewerID"].astype(str)
    normalized_user_map = {str(key): int(value) for key, value in user_map.items()}
    items = list(dict.fromkeys(validation["asin"].tolist()))
    targets = {
        str(asin): sorted(
            {
                normalized_user_map[user]
                for user in frame["reviewerID"].tolist()
                if user in normalized_user_map
            }
        )
        for asin, frame in validation.groupby("asin", sort=False)
    }
    return items, targets


def load_validation_profile(profile_csv: Path, items: list[str]) -> pd.DataFrame:
    profile = pd.read_csv(profile_csv, dtype={"raw_asin": str}, low_memory=False)
    normalized_columns = {str(column).strip().lower() for column in profile.columns}
    forbidden = sorted(normalized_columns & FORBIDDEN_EVALUATION_COLUMNS)
    if forbidden:
        raise ValueError(f"CICP profile contains forbidden evaluation columns: {forbidden}")
    required = {"raw_asin", "split", "cicp_score"}
    missing_columns = sorted(required - set(profile.columns))
    if missing_columns:
        raise ValueError(f"CICP profile is missing columns: {missing_columns}")
    validation = profile[profile["split"].astype(str).eq("validate")].copy()
    validation["raw_asin"] = validation["raw_asin"].astype(str)
    validation["cicp_score"] = pd.to_numeric(validation["cicp_score"], errors="raise")
    if validation["raw_asin"].duplicated().any():
        raise ValueError("validation profile contains duplicate raw_asin values")
    missing_items = sorted(set(items) - set(validation["raw_asin"]))
    if missing_items:
        raise ValueError(f"validation profile is missing {len(missing_items)} requested items")
    validation["cicp_group"] = pd.cut(
        validation["cicp_score"],
        bins=[-np.inf, 1.0 / 3.0, 2.0 / 3.0, np.inf],
        labels=["low", "mid", "high"],
        right=False,
    ).astype(str)
    return validation.set_index("raw_asin").loc[items].reset_index()


def evaluate_validation_items(
    *,
    code_root: Path,
    model,
    save_dict: dict,
    config: dict[str, Any],
    best: pd.Series,
    validate_csv: Path,
    profile_csv: Path,
    method_label: str,
    device_name: str,
    batch_size: int,
) -> pd.DataFrame:
    import torch

    amazon_code_dir = code_root / "Amazon VG"
    if str(amazon_code_dir) not in sys.path:
        sys.path.insert(0, str(amazon_code_dir))
    from cicp_features import load_cicp_feature_tensor
    from model import CICP_METHOD_VARIANTS
    from support import build_item_feature_tensors

    items, targets = _validation_targets(validate_csv, save_dict["user_ser_dict"])
    item_map = {item: index for index, item in enumerate(items)}
    categories, images = build_item_feature_tensors(
        item_serialize_dict=item_map,
        img_features=save_dict["img_feature_dict"],
        genres=save_dict["asin_category_int_map"],
        category_num=int(save_dict["category_ser_map_len"]),
    )
    profile = load_validation_profile(profile_csv, items)
    profile_by_item = profile.set_index("raw_asin")
    method_variant = str(config.get("method_variant", "baseline"))
    cicp_features = None
    if method_variant in CICP_METHOD_VARIANTS:
        cicp_features = load_cicp_feature_tensor(
            profile_csv,
            item_map,
            item_number=len(items),
            reject_evaluation_columns=True,
        )

    device = torch.device(device_name)
    user_embedding = model.user_embedding.to(device)
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            end = min(start + batch_size, len(items))
            item_batch = items[start:end]
            category_batch = categories[start:end].to(device)
            image_batch = images[start:end].to(device)
            if cicp_features is None:
                embeddings = model(category_batch, image_batch, len(item_batch))
            else:
                embeddings = model(
                    category_batch,
                    image_batch,
                    len(item_batch),
                    cicp_features=cicp_features[start:end].to(device),
                )
            recommended_batches = torch.topk(
                torch.matmul(embeddings, user_embedding.T),
                k=20,
                dim=1,
            ).indices.cpu().tolist()
            for raw_asin, recommended in zip(item_batch, recommended_batches):
                hr20, ndcg20 = _item_metrics(recommended, targets[raw_asin])
                rows.append(
                    {
                        "method_variant": method_variant,
                        "method_label": method_label,
                        "checkpoint_index": int(best["checkpoint_index"]),
                        "epoch": int(best["epoch"]),
                        "raw_asin": raw_asin,
                        "cicp_score": float(profile_by_item.loc[raw_asin, "cicp_score"]),
                        "cicp_group": str(profile_by_item.loc[raw_asin, "cicp_group"]),
                        "hr@20": hr20,
                        "ndcg@20": ndcg20,
                    }
                )
            print(
                f"item-eval progress method={method_label} items={end}/{len(items)}",
                flush=True,
            )
    return pd.DataFrame(rows)


def build_group_summary(item_metrics: pd.DataFrame) -> pd.DataFrame:
    baseline = item_metrics[item_metrics["method_label"].eq(BASELINE_LABEL)].set_index(
        "raw_asin"
    )
    rows: list[dict[str, Any]] = []
    groups = [("overall", item_metrics["raw_asin"].notna())]
    groups.extend(
        (group, item_metrics["cicp_group"].eq(group)) for group in ("low", "mid", "high")
    )
    for group_name, group_mask in groups:
        group = item_metrics[group_mask]
        for method_label, frame in group.groupby("method_label", sort=False):
            paired = frame.set_index("raw_asin").join(
                baseline[["hr@20", "ndcg@20"]],
                how="left",
                rsuffix="_baseline",
                validate="one_to_one",
            )
            delta_ndcg = paired["ndcg@20"] - paired["ndcg@20_baseline"]
            delta_hr = paired["hr@20"] - paired["hr@20_baseline"]
            baseline_ndcg = float(paired["ndcg@20_baseline"].mean())
            baseline_hr = float(paired["hr@20_baseline"].mean())
            method_ndcg = float(paired["ndcg@20"].mean())
            method_hr = float(paired["hr@20"].mean())
            rows.append(
                {
                    "cicp_group": group_name,
                    "method_label": method_label,
                    "method_variant": str(frame["method_variant"].iloc[0]),
                    "item_count": int(len(paired)),
                    "baseline_ndcg@20": baseline_ndcg,
                    "method_ndcg@20": method_ndcg,
                    "absolute_delta_ndcg@20": method_ndcg - baseline_ndcg,
                    "relative_pct_ndcg@20": (
                        (method_ndcg / baseline_ndcg - 1.0) * 100.0
                        if baseline_ndcg != 0.0
                        else np.nan
                    ),
                    "baseline_hr@20": baseline_hr,
                    "method_hr@20": method_hr,
                    "absolute_delta_hr@20": method_hr - baseline_hr,
                    "relative_pct_hr@20": (
                        (method_hr / baseline_hr - 1.0) * 100.0
                        if baseline_hr != 0.0
                        else np.nan
                    ),
                    "helped_ndcg_item_count": int((delta_ndcg > 0.0).sum()),
                    "harmed_ndcg_item_count": int((delta_ndcg < 0.0).sum()),
                    "equal_ndcg_item_count": int((delta_ndcg == 0.0).sum()),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    code_root = args.code_root.resolve()
    result_root = args.result_root.resolve()
    profile_csv = args.profile_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = select_device(args.device)

    frames: list[pd.DataFrame] = []
    curve_audit: list[dict[str, Any]] = []
    packages: list[tuple[str, dict[str, Any], dict, dict, pd.Series]] = []
    baseline_config, baseline_save, baseline_state, baseline_best = load_baseline_package(
        args.baseline_result.resolve(), device_name
    )
    packages.append(
        (BASELINE_LABEL, baseline_config, baseline_save, baseline_state, baseline_best)
    )
    run_dirs = sorted(
        path.parent
        for path in result_root.rglob("result.csv")
        if (path.parent / "run_config.json").exists()
    )
    if len(run_dirs) != 6:
        raise ValueError(f"expected six CICP-R1 run directories, got {len(run_dirs)}")
    for run_dir in run_dirs:
        config, save_dict, state_dict, best = load_run_package(run_dir, device_name)
        method_variant = str(config.get("method_variant", ""))
        if method_variant not in METHOD_LABELS:
            raise ValueError(f"unexpected method_variant: {method_variant}")
        packages.append((METHOD_LABELS[method_variant], config, save_dict, state_dict, best))

    for method_label, config, save_dict, state_dict, best in packages:
        print(
            f"item-eval start method={method_label} checkpoint={int(best['checkpoint_index'])} "
            f"device={device_name}",
            flush=True,
        )
        model = load_model(code_root, state_dict, config, device_name)
        frame = evaluate_validation_items(
            code_root=code_root,
            model=model,
            save_dict=save_dict,
            config=config,
            best=best,
            validate_csv=args.validate_csv.resolve(),
            profile_csv=profile_csv,
            method_label=method_label,
            device_name=device_name,
            batch_size=args.batch_size,
        )
        frames.append(frame)
        evaluated_ndcg = float(frame["ndcg@20"].mean())
        evaluated_hr = float(frame["hr@20"].mean())
        curve_audit.append(
            {
                "method_label": method_label,
                "method_variant": str(config.get("method_variant", "baseline")),
                "checkpoint_index": int(best["checkpoint_index"]),
                "epoch": int(best["epoch"]),
                "curve_ndcg@20": float(best["ndcg@20"]),
                "evaluated_ndcg@20": evaluated_ndcg,
                "absolute_error_ndcg@20": evaluated_ndcg - float(best["ndcg@20"]),
                "curve_hr@20": float(best["hr@20"]),
                "evaluated_hr@20": evaluated_hr,
                "absolute_error_hr@20": evaluated_hr - float(best["hr@20"]),
            }
        )
        del model
        if device_name == "mps":
            torch.mps.empty_cache()
        elif device_name == "cuda":
            torch.cuda.empty_cache()

    item_metrics = pd.concat(frames, ignore_index=True)
    group_summary = build_group_summary(item_metrics)
    audit = {
        "protocol": "cicpr1_validation_groups_v1",
        "device": device_name,
        "method_count": int(item_metrics["method_label"].nunique()),
        "items_per_method": int(item_metrics.groupby("method_label").size().iloc[0]),
        "item_row_count": int(len(item_metrics)),
        "evaluated_split": "validate",
        "validation_item_count": int(item_metrics["raw_asin"].nunique()),
        "full_dataset_item_count": 35322,
        "validation_coverage_pct_of_full_dataset": (
            item_metrics["raw_asin"].nunique() / 35322.0 * 100.0
        ),
        "train_recommendation_metrics_evaluated": False,
        "test_recommendation_metrics_read_or_generated": False,
        "full_mixed_recommendation_metrics_evaluated": False,
        "cicp_group_boundaries": {
            "low": "[0,1/3)",
            "mid": "[1/3,2/3)",
            "high": "[2/3,1]",
        },
        "curve_reaggregation_max_abs_error_ndcg@20": max(
            abs(row["absolute_error_ndcg@20"]) for row in curve_audit
        ),
        "curve_reaggregation_max_abs_error_hr@20": max(
            abs(row["absolute_error_hr@20"]) for row in curve_audit
        ),
        "curve_audit": curve_audit,
    }
    item_metrics.to_csv(output_dir / "cicpr1_validation_item_metrics.csv", index=False)
    group_summary.to_csv(output_dir / "cicpr1_validation_cicp_group_summary.csv", index=False)
    (output_dir / "cicpr1_validation_evaluation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--validate-csv", type=Path, required=True)
    parser.add_argument("--profile-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
