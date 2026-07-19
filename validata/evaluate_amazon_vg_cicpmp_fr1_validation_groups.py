#!/usr/bin/env python3
"""Re-evaluate CICP-MP-FR1 checkpoints on validation cold items only."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from evaluate_amazon_vg_cicpmp_r1_validation_groups import (
    build_model_args as build_r1_model_args,
)
from evaluate_amazon_vg_cicpr1_validation_groups import (
    BASELINE_LABEL,
    _item_metrics,
    _validation_targets,
    load_baseline_package,
    load_run_package,
    select_device,
)


METHOD_LABELS = {
    "cicpmp_fr1_scalar_residual_reference": "CICP-MP-FR1-E1-SRR",
    "cicpmp_fr1_modality_film": "CICP-MP-FR1-E2-MFM",
    "cicpmp_fr1_content_expert_routing": "CICP-MP-FR1-E3-CER",
    "cicpmp_fr1_cross_modal_attention": "CICP-MP-FR1-E4-CMA",
    "cicpmp_fr1_modality_film_shuffle": "CICP-MP-FR1-E5-MFS",
}
VALIDATION_ITEM_COUNT = 5298
FULL_ITEM_COUNT = 35322


def build_model_args(state_dict: dict, config: dict[str, Any]) -> SimpleNamespace:
    args = build_r1_model_args(state_dict, config)
    args.seed = int(config.get("seed", 43))
    args.cicpmp_fr1_block_dim = int(config.get("cicpmp_fr1_block_dim", 8))
    args.cicpmp_fr1_residual_max_ratio = float(
        config.get("cicpmp_fr1_residual_max_ratio", 0.15)
    )
    args.cicpmp_fr1_method_weight_decay = float(
        config.get("optimizer_parameter_groups", {}).get("method_weight_decay", 0.0)
    )
    return args


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


def _tertile(values: pd.Series) -> tuple[pd.Series, list[float]]:
    lower, upper = values.quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
    groups = pd.cut(
        values,
        bins=[-np.inf, lower, upper, np.inf],
        labels=["low", "mid", "high"],
        include_lowest=True,
    ).astype(str)
    return groups, [float(lower), float(upper)]


def load_analysis_profile(
    code_root: Path,
    raw_mp_profile_csv: Path,
    scalar_profile_csv: Path,
    items: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    amazon_code_dir = code_root / "Amazon VG"
    if str(amazon_code_dir) not in sys.path:
        sys.path.insert(0, str(amazon_code_dir))
    from cicp_mp_features import (
        CICP_MP_DIRECTION_FEATURE_NAMES,
        CICP_MP_FEATURE_NAMES,
        build_cicp_mp_feature_frame,
    )

    raw = pd.read_csv(raw_mp_profile_csv, dtype={"raw_asin": str}, low_memory=False)
    build_cicp_mp_feature_frame(raw, reject_evaluation_columns=True)
    raw = raw[raw["split"].astype(str).eq("validate")].copy()
    raw["raw_asin"] = raw["raw_asin"].astype(str)
    if raw["raw_asin"].duplicated().any():
        raise ValueError("raw CICP-MP validation profile contains duplicate items")
    missing = sorted(set(items) - set(raw["raw_asin"]))
    if missing:
        raise ValueError(f"raw CICP-MP profile is missing {len(missing)} validation items")
    raw = raw.set_index("raw_asin").loc[items].reset_index()
    values = raw.loc[:, CICP_MP_FEATURE_NAMES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("raw CICP-MP validation profile contains non-finite values")

    scalar = pd.read_csv(
        scalar_profile_csv,
        dtype={"raw_asin": str},
        low_memory=False,
    )
    forbidden = {
        "hr@5", "hr@10", "hr@20", "ndcg@5", "ndcg@10", "ndcg@20",
        "baseline_hr@20", "baseline_ndcg@20", "delta_hr@20", "delta_ndcg@20",
    }
    leaked = sorted({str(column).strip().lower() for column in scalar.columns} & forbidden)
    if leaked:
        raise ValueError(f"scalar CICP profile contains evaluation columns: {leaked}")
    scalar = scalar[scalar["split"].astype(str).eq("validate")].copy()
    scalar["raw_asin"] = scalar["raw_asin"].astype(str)
    scalar["cicp_score"] = pd.to_numeric(scalar["cicp_score"], errors="raise")
    scalar = scalar.set_index("raw_asin").loc[items, ["cicp_score"]]
    raw = raw.join(scalar, on="raw_asin")

    raw["mp_direction_norm"] = np.linalg.norm(
        values.loc[:, CICP_MP_DIRECTION_FEATURE_NAMES].to_numpy(dtype=float),
        axis=1,
    )
    group_sources = {
        "cicp_score_group": "cicp_score",
        "raw_increment_group": "mp_raw_predicted_increment",
        "semantic_increment_group": "mp_category_semantic_increment_prediction",
        "total_increment_group": "mp_category_total_increment_prediction",
        "attribution_share_group": "mp_category_attribution_positive_share_prediction",
        "attribution_entropy_group": "mp_category_attribution_entropy_prediction",
        "uncertainty_group": "mp_fold_prediction_uncertainty",
        "disagreement_group": "mp_hgb_ridge_disagreement",
        "direction_norm_group": "mp_direction_norm",
    }
    thresholds: dict[str, list[float]] = {}
    for group_column, source_column in group_sources.items():
        raw[group_column], thresholds[group_column] = _tertile(raw[source_column])

    audit = {
        "profile_feature_count": len(CICP_MP_FEATURE_NAMES),
        "validation_item_count": len(raw),
        "group_sources": group_sources,
        "validation_tertile_thresholds": thresholds,
        "group_counts": {
            column: {
                str(key): int(value)
                for key, value in raw[column].value_counts().sort_index().items()
            }
            for column in group_sources
        },
    }
    keep = ["raw_asin", "cicp_score", "mp_direction_norm"]
    keep.extend(CICP_MP_FEATURE_NAMES)
    keep.extend(group_sources)
    return raw.loc[:, keep], audit


def evaluate_validation_items(
    *,
    code_root: Path,
    model,
    save_dict: dict,
    config: dict[str, Any],
    best: pd.Series,
    validate_csv: Path,
    profile: pd.DataFrame,
    method_label: str,
    device_name: str,
    batch_size: int,
) -> pd.DataFrame:
    import torch

    amazon_code_dir = code_root / "Amazon VG"
    if str(amazon_code_dir) not in sys.path:
        sys.path.insert(0, str(amazon_code_dir))
    from cicp_features import load_cicp_feature_tensor
    from cicp_mp_features import load_cicp_mp_feature_tensor
    from model import (
        CICPMP_FR1_MP_METHOD_VARIANTS,
        CICPMP_FR1_SCALAR_REFERENCE_METHOD_VARIANTS,
    )
    from support import build_item_feature_tensors

    items, targets = _validation_targets(validate_csv, save_dict["user_ser_dict"])
    if items != profile["raw_asin"].tolist():
        raise ValueError("validation item order does not match analysis profile")
    item_map = {item: index for index, item in enumerate(items)}
    categories, images = build_item_feature_tensors(
        item_serialize_dict=item_map,
        img_features=save_dict["img_feature_dict"],
        genres=save_dict["asin_category_int_map"],
        category_num=int(save_dict["category_ser_map_len"]),
    )
    method_variant = str(config.get("method_variant", "baseline"))
    cicp_features = None
    mp_features = None
    if method_variant in CICPMP_FR1_SCALAR_REFERENCE_METHOD_VARIANTS:
        profile_path = Path(str(config.get("cicp_profile_path", "")))
        if not profile_path.is_file():
            raise ValueError(f"missing scalar CICP profile for {method_variant}: {profile_path}")
        cicp_features = load_cicp_feature_tensor(
            profile_path,
            item_map,
            item_number=len(items),
            reject_evaluation_columns=True,
        )
    elif method_variant in CICPMP_FR1_MP_METHOD_VARIANTS:
        profile_path = Path(str(config.get("cicp_mp_profile_path", "")))
        if not profile_path.is_file():
            raise ValueError(f"missing CICP-MP profile for {method_variant}: {profile_path}")
        mp_features = load_cicp_mp_feature_tensor(
            profile_path,
            item_map,
            item_number=len(items),
            reject_evaluation_columns=True,
        )

    device = torch.device(device_name)
    user_embedding = model.user_embedding.to(device)
    profile_by_item = profile.set_index("raw_asin")
    profile_columns = [column for column in profile.columns if column != "raw_asin"]
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            end = min(start + batch_size, len(items))
            item_batch = items[start:end]
            kwargs: dict[str, Any] = {}
            if cicp_features is not None:
                kwargs["cicp_features"] = cicp_features[start:end].to(device)
            if mp_features is not None:
                kwargs["cicp_mp_features"] = mp_features[start:end].to(device)
            embeddings = model(
                categories[start:end].to(device),
                images[start:end].to(device),
                len(item_batch),
                **kwargs,
            )
            recommended_batches = torch.topk(
                torch.matmul(embeddings, user_embedding.T),
                k=20,
                dim=1,
            ).indices.cpu().tolist()
            for raw_asin, recommended in zip(item_batch, recommended_batches):
                hr20, ndcg20 = _item_metrics(recommended, targets[raw_asin])
                row = {
                    "method_variant": method_variant,
                    "method_label": method_label,
                    "checkpoint_index": int(best["checkpoint_index"]),
                    "epoch": int(best["epoch"]),
                    "raw_asin": raw_asin,
                    "hr@20": hr20,
                    "ndcg@20": ndcg20,
                }
                item_profile = profile_by_item.loc[raw_asin]
                row.update({column: item_profile[column] for column in profile_columns})
                rows.append(row)
            print(
                f"item-eval progress method={method_label} items={end}/{len(items)}",
                flush=True,
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    code_root = args.code_root.resolve()
    result_root = args.result_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = select_device(args.device)
    run_dirs = sorted(
        path.parent
        for path in result_root.rglob("result.csv")
        if (path.parent / "run_config.json").exists()
    )
    if len(run_dirs) != 5:
        raise ValueError(f"expected five CICP-MP-FR1 run directories, got {len(run_dirs)}")

    baseline_config, baseline_save, baseline_state, baseline_best = load_baseline_package(
        args.baseline_result.resolve(), device_name
    )
    packages = [
        (BASELINE_LABEL, baseline_config, baseline_save, baseline_state, baseline_best)
    ]
    configs: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        config, save_dict, state_dict, best = load_run_package(run_dir, device_name)
        method_variant = str(config.get("method_variant", ""))
        if method_variant not in METHOD_LABELS:
            raise ValueError(f"unexpected method_variant: {method_variant}")
        packages.append((METHOD_LABELS[method_variant], config, save_dict, state_dict, best))
        configs.append(config)
    common_hashes = {
        str(config.get("ccfcrec_common_parameter_sha256", "")) for config in configs
    }
    if len(common_hashes) != 1:
        raise ValueError(f"actual common parameter hash mismatch: {common_hashes}")

    items, _ = _validation_targets(
        args.validate_csv.resolve(), baseline_save["user_ser_dict"]
    )
    profile, profile_audit = load_analysis_profile(
        code_root,
        args.raw_mp_profile_csv.resolve(),
        args.scalar_profile_csv.resolve(),
        items,
    )
    frames: list[pd.DataFrame] = []
    curve_audit: list[dict[str, Any]] = []
    for method_label, config, save_dict, state_dict, best in packages:
        print(
            f"item-eval start method={method_label} epoch={int(best['epoch'])} "
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
            profile=profile,
            method_label=method_label,
            device_name=device_name,
            batch_size=args.batch_size,
        )
        frames.append(frame)
        curve_audit.append(
            {
                "method_label": method_label,
                "method_variant": str(config.get("method_variant", "baseline")),
                "checkpoint_index": int(best["checkpoint_index"]),
                "epoch": int(best["epoch"]),
                "curve_ndcg@20": float(best["ndcg@20"]),
                "evaluated_ndcg@20": float(frame["ndcg@20"].mean()),
                "absolute_error_ndcg@20": float(frame["ndcg@20"].mean())
                - float(best["ndcg@20"]),
                "curve_hr@20": float(best["hr@20"]),
                "evaluated_hr@20": float(frame["hr@20"].mean()),
                "absolute_error_hr@20": float(frame["hr@20"].mean())
                - float(best["hr@20"]),
            }
        )
        del model
        if device_name == "mps":
            torch.mps.empty_cache()
        elif device_name == "cuda":
            torch.cuda.empty_cache()

    item_metrics = pd.concat(frames, ignore_index=True)
    counts = item_metrics.groupby("method_label").size()
    if len(counts) != 6 or not counts.eq(VALIDATION_ITEM_COUNT).all():
        raise ValueError(f"unexpected validation coverage: {counts.to_dict()}")
    max_ndcg_error = max(abs(row["absolute_error_ndcg@20"]) for row in curve_audit)
    max_hr_error = max(abs(row["absolute_error_hr@20"]) for row in curve_audit)
    if max_ndcg_error > 1e-10 or max_hr_error > 1e-10:
        raise ValueError(
            "validation item metrics do not reproduce saved curves: "
            f"ndcg_error={max_ndcg_error}, hr_error={max_hr_error}"
        )
    audit = {
        "protocol": "cicpmp_fr1_validation_groups_v1",
        "device": device_name,
        "method_count": int(item_metrics["method_label"].nunique()),
        "items_per_method": int(counts.iloc[0]),
        "item_row_count": int(len(item_metrics)),
        "evaluated_split": "validate",
        "validation_item_count": int(item_metrics["raw_asin"].nunique()),
        "full_dataset_item_count": FULL_ITEM_COUNT,
        "validation_coverage_pct_of_full_dataset": VALIDATION_ITEM_COUNT
        / FULL_ITEM_COUNT
        * 100.0,
        "train_recommendation_metrics_evaluated": False,
        "test_recommendation_metrics_read_or_generated": False,
        "full_mixed_recommendation_metrics_evaluated": False,
        "actual_common_parameter_sha256": next(iter(common_hashes)),
        "profile_audit": profile_audit,
        "curve_reaggregation_max_abs_error_ndcg@20": max_ndcg_error,
        "curve_reaggregation_max_abs_error_hr@20": max_hr_error,
        "curve_audit": curve_audit,
    }
    item_metrics.to_csv(output_dir / "cicpmp_fr1_validation_item_metrics.csv", index=False)
    (output_dir / "cicpmp_fr1_validation_evaluation_audit.json").write_text(
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
    parser.add_argument("--raw-mp-profile-csv", type=Path, required=True)
    parser.add_argument("--scalar-profile-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
