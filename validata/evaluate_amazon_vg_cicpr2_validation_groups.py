#!/usr/bin/env python3
"""Evaluate CICP-R2 checkpoints on all validation cold items only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from evaluate_amazon_vg_cicpr1_validation_groups import (
    BASELINE_LABEL,
    build_group_summary,
    evaluate_validation_items,
    load_baseline_package,
    load_model,
    load_run_package,
    select_device,
)


METHOD_LABELS = {
    "cicpr2_content_direction_residual": "CICP-R2-E1-CDR",
    "cicpr2_category_increment_gate": "CICP-R2-E2-CID",
    "cicpr2_cross_modal_attention": "CICP-R2-E3-CMA",
    "cicpr2_score_distillation": "CICP-R2-E4-SD",
    "cicpr2_ordinal_counterfactual": "CICP-R2-E5-OCS",
    "cicpr2_reliability_dropout": "CICP-R2-E6-RCD",
}
VALIDATION_ITEM_COUNT = 5298
FULL_ITEM_COUNT = 35322


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
        raise ValueError(f"expected six CICP-R2 run directories, got {len(run_dirs)}")
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
    per_method = item_metrics.groupby("method_label").size()
    if len(per_method) != 7 or not per_method.eq(VALIDATION_ITEM_COUNT).all():
        raise ValueError(f"unexpected validation coverage: {per_method.to_dict()}")
    group_summary = build_group_summary(item_metrics)
    audit = {
        "protocol": "cicpr2_validation_groups_v1",
        "device": device_name,
        "method_count": int(item_metrics["method_label"].nunique()),
        "items_per_method": int(per_method.iloc[0]),
        "item_row_count": int(len(item_metrics)),
        "evaluated_split": "validate",
        "validation_item_count": int(item_metrics["raw_asin"].nunique()),
        "full_dataset_item_count": FULL_ITEM_COUNT,
        "validation_coverage_pct_of_full_dataset": (
            item_metrics["raw_asin"].nunique() / FULL_ITEM_COUNT * 100.0
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
    item_metrics.to_csv(output_dir / "cicpr2_validation_item_metrics.csv", index=False)
    group_summary.to_csv(
        output_dir / "cicpr2_validation_cicp_group_summary.csv", index=False
    )
    (output_dir / "cicpr2_validation_evaluation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
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
