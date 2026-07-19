#!/usr/bin/env python3
"""Audit CICP-MP-FR1 gate activity without reading validation outcomes."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evaluate_amazon_vg_cicpmp_fr1_validation_groups import (
    METHOD_LABELS,
    load_model,
)
from evaluate_amazon_vg_cicpr1_validation_groups import select_device


FIXED_EPOCHS = (1, 20, 40, 74, 100)


def tensor_summary(values) -> dict[str, float]:
    array = values.detach().float().cpu().numpy().reshape(-1)
    return {
        "mean": float(np.mean(array)),
        "abs_mean": float(np.mean(np.abs(array))),
        "rms": float(np.sqrt(np.mean(np.square(array)))),
        "abs_p95": float(np.quantile(np.abs(array), 0.95)),
        "abs_max": float(np.max(np.abs(array))),
        "nonzero_fraction": float(np.mean(np.abs(array) > 1e-6)),
    }


def embedding_relative_l2(actual, reference) -> np.ndarray:
    numerator = (actual - reference).float().norm(dim=1)
    denominator = actual.float().norm(dim=1).clamp_min(1e-12)
    return (numerator / denominator).detach().cpu().numpy()


def selected_epoch_rows(result: pd.DataFrame) -> pd.DataFrame:
    best = result.sort_values(
        ["ndcg@20", "hr@20", "epoch", "checkpoint_index"],
        ascending=[False, False, True, True],
    ).iloc[0]
    epochs = sorted(set(FIXED_EPOCHS) | {int(best["epoch"])})
    selected = result[result["epoch"].astype(int).isin(epochs)].copy()
    if len(selected) != len(epochs):
        raise ValueError(f"missing requested checkpoint epochs: {epochs}")
    selected["is_best_epoch"] = selected["epoch"].astype(int).eq(int(best["epoch"]))
    return selected.sort_values("epoch")


def _method_parameter_stats(model) -> dict[str, float | int]:
    values = [
        parameter.detach().float().reshape(-1).cpu()
        for name, parameter in model.named_parameters()
        if name.startswith("cicpmp_fr1_")
    ]
    if not values:
        raise ValueError("model has no CICP-MP-FR1 method parameters")
    import torch

    joined = torch.cat(values)
    return {
        "method_parameter_count": int(joined.numel()),
        "method_parameter_l2": float(joined.norm().item()),
        "method_parameter_abs_mean": float(joined.abs().mean().item()),
    }


def _forward(model, categories, images, cicp, mp, batch_size: int):
    import torch

    outputs = []
    gates = []
    residuals = []
    router_entropies = []
    with torch.no_grad():
        for start in range(0, len(categories), batch_size):
            end = min(start + batch_size, len(categories))
            kwargs: dict[str, Any] = {}
            if cicp is not None:
                kwargs["cicp_features"] = cicp[start:end]
            if mp is not None:
                kwargs["cicp_mp_features"] = mp[start:end]
            outputs.append(
                model(categories[start:end], images[start:end], end - start, **kwargs)
            )
            if model._last_cicpmp_gate is not None:
                gates.append(model._last_cicpmp_gate.detach())
            if model._last_cicpmp_residual is not None:
                residuals.append(model._last_cicpmp_residual.detach())
            if model.uses_cicpmp_fr1_content_expert():
                condition = model.build_cicpmp_fr1_condition(mp[start:end])
                weights = torch.softmax(model.cicpmp_fr1_expert_router(condition), dim=1)
                entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
                router_entropies.append(entropy.detach())
    return (
        torch.cat(outputs),
        torch.cat(gates) if gates else None,
        torch.cat(residuals) if residuals else None,
        torch.cat(router_entropies) if router_entropies else None,
    )


def evaluate_checkpoint(
    *,
    code_root: Path,
    run_dir: Path,
    config: dict[str, Any],
    save_dict: dict,
    checkpoint_row: pd.Series,
    validate_csv: Path,
    device_name: str,
    sample_size: int,
    batch_size: int,
) -> dict[str, Any]:
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

    validation = pd.read_csv(validate_csv, dtype={"asin": str}, usecols=["asin"])
    items = list(dict.fromkeys(validation["asin"].astype(str).tolist()))[:sample_size]
    item_map = {item: index for index, item in enumerate(items)}
    categories, images = build_item_feature_tensors(
        item_serialize_dict=item_map,
        img_features=save_dict["img_feature_dict"],
        genres=save_dict["asin_category_int_map"],
        category_num=int(save_dict["category_ser_map_len"]),
    )
    device = torch.device(device_name)
    categories = categories.to(device)
    images = images.to(device)
    method_variant = str(config["method_variant"])
    cicp = None
    mp = None
    if method_variant in CICPMP_FR1_SCALAR_REFERENCE_METHOD_VARIANTS:
        cicp = load_cicp_feature_tensor(
            Path(config["cicp_profile_path"]),
            item_map,
            item_number=len(items),
            reject_evaluation_columns=True,
        ).to(device)
    elif method_variant in CICPMP_FR1_MP_METHOD_VARIANTS:
        mp = load_cicp_mp_feature_tensor(
            Path(config["cicp_mp_profile_path"]),
            item_map,
            item_number=len(items),
            reject_evaluation_columns=True,
        ).to(device)

    checkpoint_index = int(checkpoint_row["checkpoint_index"])
    state_dict = torch.load(
        run_dir / f"{checkpoint_index}.pt",
        map_location=device_name,
        weights_only=True,
    )
    model = load_model(code_root, state_dict, config, device_name)
    actual, gate, residual, router_entropy = _forward(
        model, categories, images, cicp, mp, batch_size
    )
    signal = cicp if cicp is not None else mp
    zero_signal = torch.zeros_like(signal)
    permuted_signal = torch.roll(signal, shifts=1, dims=0)
    zero_cicp = zero_signal if cicp is not None else None
    zero_mp = zero_signal if mp is not None else None
    perm_cicp = permuted_signal if cicp is not None else None
    perm_mp = permuted_signal if mp is not None else None
    zero_output, _, _, _ = _forward(
        model, categories, images, zero_cicp, zero_mp, batch_size
    )
    permuted_output, _, _, _ = _forward(
        model, categories, images, perm_cicp, perm_mp, batch_size
    )
    zero_delta = embedding_relative_l2(actual, zero_output)
    permuted_delta = embedding_relative_l2(actual, permuted_output)

    row: dict[str, Any] = {
        "method_variant": method_variant,
        "method_label": METHOD_LABELS[method_variant],
        "epoch": int(checkpoint_row["epoch"]),
        "checkpoint_index": checkpoint_index,
        "is_best_epoch": bool(checkpoint_row["is_best_epoch"]),
        "sample_item_count": len(items),
        "actual_vs_zero_embedding_relative_l2_mean": float(np.mean(zero_delta)),
        "actual_vs_zero_embedding_relative_l2_p95": float(np.quantile(zero_delta, 0.95)),
        "actual_vs_permuted_embedding_relative_l2_mean": float(np.mean(permuted_delta)),
        "actual_vs_permuted_embedding_relative_l2_p95": float(np.quantile(permuted_delta, 0.95)),
        "gate_available": gate is not None,
        "residual_available": residual is not None,
    }
    row.update(_method_parameter_stats(model))
    if gate is not None:
        row.update({f"gate_{key}": value for key, value in tensor_summary(gate).items()})
    if residual is not None:
        residual_norm = residual.float().norm(dim=1).cpu().numpy()
        row.update(
            {
                "residual_norm_mean": float(np.mean(residual_norm)),
                "residual_norm_p95": float(np.quantile(residual_norm, 0.95)),
                "residual_nonzero_fraction": float(np.mean(residual_norm > 1e-6)),
            }
        )
    if router_entropy is not None:
        entropy = router_entropy.float().cpu().numpy()
        row.update(
            {
                "router_entropy_mean": float(np.mean(entropy)),
                "router_entropy_p05": float(np.quantile(entropy, 0.05)),
                "router_entropy_maximum": float(np.log(3.0)),
            }
        )
    del model
    if device_name == "mps":
        torch.mps.empty_cache()
    elif device_name == "cuda":
        torch.cuda.empty_cache()
    return row


def run(args: argparse.Namespace) -> dict[str, Any]:
    result_root = args.result_root.resolve()
    code_root = args.code_root.resolve()
    device_name = select_device(args.device)
    run_dirs = sorted(
        path.parent
        for path in result_root.rglob("result.csv")
        if (path.parent / "run_config.json").exists()
    )
    if len(run_dirs) != 5:
        raise ValueError(f"expected five run directories, got {len(run_dirs)}")
    rows: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        if config.get("method_variant") not in METHOD_LABELS:
            raise ValueError(f"unexpected method variant in {run_dir}")
        with (run_dir / "save_dict.pkl").open("rb") as file_obj:
            save_dict = pickle.load(file_obj)
        result = pd.read_csv(run_dir / "result.csv")
        for _, checkpoint_row in selected_epoch_rows(result).iterrows():
            print(
                f"mechanism-audit method={config['method_variant']} "
                f"epoch={int(checkpoint_row['epoch'])}",
                flush=True,
            )
            rows.append(
                evaluate_checkpoint(
                    code_root=code_root,
                    run_dir=run_dir,
                    config=config,
                    save_dict=save_dict,
                    checkpoint_row=checkpoint_row,
                    validate_csv=args.validate_csv.resolve(),
                    device_name=device_name,
                    sample_size=args.sample_size,
                    batch_size=args.batch_size,
                )
            )
    frame = pd.DataFrame(rows)
    args.output_csv.resolve().parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_csv.resolve(), index=False)
    audit = {
        "protocol": "cicpmp_fr1_mechanism_activity_v1",
        "device": device_name,
        "checkpoint_row_count": len(frame),
        "method_count": int(frame["method_label"].nunique()),
        "sample_item_count": int(frame["sample_item_count"].iloc[0]),
        "sample_split": "validate",
        "validation_outcomes_read_or_generated": False,
        "test_data_read_or_generated": False,
        "fixed_epochs": list(FIXED_EPOCHS),
        "best_epoch_also_included": True,
    }
    audit_path = args.output_csv.resolve().with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--validate-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
