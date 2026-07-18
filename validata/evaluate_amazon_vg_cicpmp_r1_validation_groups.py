#!/usr/bin/env python3
"""Evaluate CICP-MP-R1 best checkpoints on validation cold items only."""

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

from evaluate_amazon_vg_cicpr1_validation_groups import (
    BASELINE_LABEL,
    _item_metrics,
    _validation_targets,
    build_model_args as build_legacy_model_args,
    load_baseline_package,
    load_run_package,
    select_device,
)


METHOD_LABELS = {
    "cicpmp_r1_reliable_residual": "CICP-MP-R1-E1-RRA",
    "cicpmp_r1_direction_alignment": "CICP-MP-R1-E2-DTA",
    "cicpmp_r1_attention_entropy": "CICP-MP-R1-E3-AEC",
    "cicpmp_r1_reliable_expert": "CICP-MP-R1-E4-RCE",
    "cicpmp_r1_counterfactual_calibration": "CICP-MP-R1-E5-CCI",
    "cicpmp_r1_direction_hard_negative": "CICP-MP-R1-E6-DHN",
}
VALIDATION_ITEM_COUNT = 5298
FULL_ITEM_COUNT = 35322


def build_model_args(state_dict: dict, config: dict[str, Any]) -> SimpleNamespace:
    args = build_legacy_model_args(state_dict, config)
    args.seed = int(config.get("seed", 43))
    args.cicpmp_hidden_dim = int(config.get("cicpmp_hidden_dim", 32))
    args.cicpmp_residual_max_ratio = float(
        config.get("cicpmp_residual_max_ratio", 0.15)
    )
    args.cicpmp_reliability_scale = float(
        config.get("cicpmp_reliability_scale", 50.0)
    )
    args.cicpmp_direction_weight = float(config.get("cicpmp_direction_weight", 0.05))
    args.cicpmp_entropy_weight = float(config.get("cicpmp_entropy_weight", 0.02))
    args.cicpmp_expert_strength = float(config.get("cicpmp_expert_strength", 0.20))
    args.cicpmp_counterfactual_weight = float(
        config.get("cicpmp_counterfactual_weight", 0.05)
    )
    args.cicpmp_hard_negative_strength = float(
        config.get("cicpmp_hard_negative_strength", 0.50)
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
    labels = pd.cut(
        values,
        bins=[-np.inf, lower, upper, np.inf],
        labels=["low", "mid", "high"],
        include_lowest=True,
    ).astype(str)
    return labels, [float(lower), float(upper)]


def load_validation_profile(
    code_root: Path,
    profile_csv: Path,
    items: list[str],
    reliability_scale: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    amazon_code_dir = code_root / "Amazon VG"
    if str(amazon_code_dir) not in sys.path:
        sys.path.insert(0, str(amazon_code_dir))
    from cicp_mp_features import (
        CICP_MP_ATTRIBUTION_ENTROPY_INDEX,
        CICP_MP_DIRECTION_FEATURE_NAMES,
        CICP_MP_FEATURE_NAMES,
        CICP_MP_SEMANTIC_INCREMENT_INDEX,
        CICP_MP_UNCERTAINTY_INDEX,
        CICP_MP_DISAGREEMENT_INDEX,
        build_cicp_mp_feature_frame,
    )

    profile = pd.read_csv(profile_csv, dtype={"raw_asin": str}, low_memory=False)
    build_cicp_mp_feature_frame(profile, reject_evaluation_columns=True)
    if "split" not in profile.columns:
        raise ValueError("CICP-MP profile is missing split")
    validation = profile[profile["split"].astype(str).eq("validate")].copy()
    validation["raw_asin"] = validation["raw_asin"].astype(str)
    if validation["raw_asin"].duplicated().any():
        raise ValueError("CICP-MP validation profile contains duplicate raw_asin")
    missing_items = sorted(set(items) - set(validation["raw_asin"]))
    if missing_items:
        raise ValueError(
            f"CICP-MP validation profile is missing {len(missing_items)} requested items"
        )
    validation = validation.set_index("raw_asin").loc[items].reset_index()
    values = validation.loc[:, CICP_MP_FEATURE_NAMES].apply(
        pd.to_numeric,
        errors="raise",
    )
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError("CICP-MP validation profile contains non-finite values")

    uncertainty = values.iloc[:, CICP_MP_UNCERTAINTY_INDEX].clip(lower=0.0)
    disagreement = values.iloc[:, CICP_MP_DISAGREEMENT_INDEX].clip(lower=0.0)
    validation["mp_reliability"] = np.exp(
        -float(reliability_scale) * (uncertainty + disagreement)
    ).clip(0.05, 1.0)
    validation["mp_direction_norm"] = np.linalg.norm(
        values.loc[:, CICP_MP_DIRECTION_FEATURE_NAMES].to_numpy(dtype=float),
        axis=1,
    )
    validation["mp_semantic_increment"] = values.iloc[
        :, CICP_MP_SEMANTIC_INCREMENT_INDEX
    ]
    validation["mp_attribution_entropy"] = values.iloc[
        :, CICP_MP_ATTRIBUTION_ENTROPY_INDEX
    ].clip(0.0, 1.0)

    validation["reliability_group"], reliability_thresholds = _tertile(
        validation["mp_reliability"]
    )
    entropy_median = float(validation["mp_attribution_entropy"].median())
    direction_median = float(validation["mp_direction_norm"].median())
    validation["semantic_sign_group"] = np.where(
        validation["mp_semantic_increment"] > 0.0,
        "positive",
        "non_positive",
    )
    validation["entropy_group"] = np.where(
        validation["mp_attribution_entropy"] > entropy_median,
        "high",
        "low",
    )
    validation["direction_norm_group"] = np.where(
        validation["mp_direction_norm"] > direction_median,
        "high",
        "low",
    )
    audit = {
        "reliability_scale": float(reliability_scale),
        "reliability_validation_tertile_thresholds": reliability_thresholds,
        "attribution_entropy_validation_median": entropy_median,
        "direction_norm_validation_median": direction_median,
        "semantic_sign_boundary": "positive > 0; non_positive <= 0",
        "group_counts": {
            column: {
                str(key): int(value)
                for key, value in validation[column].value_counts().sort_index().items()
            }
            for column in (
                "reliability_group",
                "semantic_sign_group",
                "entropy_group",
                "direction_norm_group",
            )
        },
    }
    keep = [
        "raw_asin",
        "mp_reliability",
        "mp_direction_norm",
        "mp_semantic_increment",
        "mp_attribution_entropy",
        "reliability_group",
        "semantic_sign_group",
        "entropy_group",
        "direction_norm_group",
    ]
    keep.extend(name for name in CICP_MP_FEATURE_NAMES if name not in keep)
    return validation.loc[:, keep], audit


def evaluate_validation_items(
    *,
    code_root: Path,
    model,
    save_dict: dict,
    config: dict[str, Any],
    best: pd.Series,
    validate_csv: Path,
    profile_csv: Path,
    profile: pd.DataFrame,
    method_label: str,
    device_name: str,
    batch_size: int,
) -> pd.DataFrame:
    import torch

    amazon_code_dir = code_root / "Amazon VG"
    if str(amazon_code_dir) not in sys.path:
        sys.path.insert(0, str(amazon_code_dir))
    from cicp_mp_features import load_cicp_mp_feature_tensor
    from model import CICPMP_R1_METHOD_VARIANTS
    from support import build_item_feature_tensors

    items, targets = _validation_targets(validate_csv, save_dict["user_ser_dict"])
    if items != profile["raw_asin"].tolist():
        raise ValueError("validation item order does not match prepared CICP-MP profile")
    item_map = {item: index for index, item in enumerate(items)}
    categories, images = build_item_feature_tensors(
        item_serialize_dict=item_map,
        img_features=save_dict["img_feature_dict"],
        genres=save_dict["asin_category_int_map"],
        category_num=int(save_dict["category_ser_map_len"]),
    )
    profile_by_item = profile.set_index("raw_asin")
    method_variant = str(config.get("method_variant", "baseline"))
    mp_features = None
    if method_variant in CICPMP_R1_METHOD_VARIANTS:
        mp_features = load_cicp_mp_feature_tensor(
            profile_csv,
            item_map,
            item_number=len(items),
            reject_evaluation_columns=True,
        )

    device = torch.device(device_name)
    user_embedding = model.user_embedding.to(device)
    rows: list[dict[str, Any]] = []
    profile_columns = [column for column in profile.columns if column != "raw_asin"]
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            end = min(start + batch_size, len(items))
            item_batch = items[start:end]
            category_batch = categories[start:end].to(device)
            image_batch = images[start:end].to(device)
            if mp_features is None:
                embeddings = model(category_batch, image_batch, len(item_batch))
            else:
                embeddings = model(
                    category_batch,
                    image_batch,
                    len(item_batch),
                    cicp_mp_features=mp_features[start:end].to(device),
                )
            recommended_batches = torch.topk(
                torch.matmul(embeddings, user_embedding.T),
                k=20,
                dim=1,
            ).indices.cpu().tolist()
            for raw_asin, recommended in zip(item_batch, recommended_batches):
                hr20, ndcg20 = _item_metrics(recommended, targets[raw_asin])
                profile_row = profile_by_item.loc[raw_asin]
                row = {
                    "method_variant": method_variant,
                    "method_label": method_label,
                    "checkpoint_index": int(best["checkpoint_index"]),
                    "epoch": int(best["epoch"]),
                    "raw_asin": raw_asin,
                    "hr@20": hr20,
                    "ndcg@20": ndcg20,
                }
                row.update({column: profile_row[column] for column in profile_columns})
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
    profile_csv = args.profile_csv.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = select_device(args.device)

    run_dirs = sorted(
        path.parent
        for path in result_root.rglob("result.csv")
        if (path.parent / "run_config.json").exists()
    )
    if len(run_dirs) != 6:
        raise ValueError(f"expected six CICP-MP-R1 run directories, got {len(run_dirs)}")

    packages: list[tuple[str, dict[str, Any], dict, dict, pd.Series]] = []
    baseline_config, baseline_save, baseline_state, baseline_best = load_baseline_package(
        args.baseline_result.resolve(), device_name
    )
    packages.append(
        (BASELINE_LABEL, baseline_config, baseline_save, baseline_state, baseline_best)
    )
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
        raise ValueError(f"CICP-MP-R1 actual common parameter hash mismatch: {common_hashes}")
    reliability_scales = {float(config["cicpmp_reliability_scale"]) for config in configs}
    if len(reliability_scales) != 1:
        raise ValueError(f"CICP-MP-R1 reliability scale mismatch: {reliability_scales}")

    items, _ = _validation_targets(args.validate_csv.resolve(), baseline_save["user_ser_dict"])
    profile, group_audit = load_validation_profile(
        code_root,
        profile_csv,
        items,
        next(iter(reliability_scales)),
    )
    frames: list[pd.DataFrame] = []
    curve_audit: list[dict[str, Any]] = []
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
    per_method = item_metrics.groupby("method_label").size()
    if len(per_method) != 7 or not per_method.eq(VALIDATION_ITEM_COUNT).all():
        raise ValueError(f"unexpected validation coverage: {per_method.to_dict()}")
    max_ndcg_error = max(abs(row["absolute_error_ndcg@20"]) for row in curve_audit)
    max_hr_error = max(abs(row["absolute_error_hr@20"]) for row in curve_audit)
    if max_ndcg_error > 1e-10 or max_hr_error > 1e-10:
        raise ValueError(
            "validation item metrics do not reproduce saved curves: "
            f"ndcg_error={max_ndcg_error}, hr_error={max_hr_error}"
        )
    audit = {
        "protocol": "cicpmp_r1_validation_groups_v1",
        "device": device_name,
        "method_count": int(item_metrics["method_label"].nunique()),
        "items_per_method": int(per_method.iloc[0]),
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
        "group_audit": group_audit,
        "curve_reaggregation_max_abs_error_ndcg@20": max_ndcg_error,
        "curve_reaggregation_max_abs_error_hr@20": max_hr_error,
        "curve_audit": curve_audit,
    }
    item_metrics.to_csv(output_dir / "cicpmp_r1_validation_item_metrics.csv", index=False)
    (output_dir / "cicpmp_r1_validation_evaluation_audit.json").write_text(
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
