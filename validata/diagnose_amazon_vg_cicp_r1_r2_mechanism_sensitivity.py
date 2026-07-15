#!/usr/bin/env python3
"""Diagnose CICP-R1-E1 and CICP-R2 mechanisms without retraining or test data."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

from evaluate_amazon_vg_cicpr1_validation_groups import (
    _item_metrics,
    _validation_targets,
    load_baseline_package,
    load_model,
    load_run_package,
    select_device,
)


BASELINE_LABEL = "baseline_seed43_workers8_fast_uniform"
BASELINE_NDCG20 = 0.1238145211709585
BASELINE_HR20 = 0.0206209890524726
FULL_ITEM_COUNT = 35322
VALIDATION_ITEM_COUNT = 5298
METHOD_LABELS = {
    "baseline": BASELINE_LABEL,
    "cicpr1_e4_residual": "CICP-R1-E1",
    "cicpr2_content_direction_residual": "CICP-R2-E1-CDR",
    "cicpr2_category_increment_gate": "CICP-R2-E2-CID",
    "cicpr2_cross_modal_attention": "CICP-R2-E3-CMA",
    "cicpr2_score_distillation": "CICP-R2-E4-SD",
    "cicpr2_ordinal_counterfactual": "CICP-R2-E5-OCS",
    "cicpr2_reliability_dropout": "CICP-R2-E6-RCD",
}
DIRECT_SCORE_METHODS = {
    "cicpr1_e4_residual",
    "cicpr2_content_direction_residual",
    "cicpr2_category_increment_gate",
    "cicpr2_cross_modal_attention",
}
TRAIN_ONLY_SCORE_METHODS = {
    "cicpr2_score_distillation",
    "cicpr2_ordinal_counterfactual",
    "cicpr2_reliability_dropout",
}
COMMON_PARAMETER_NAMES = (
    "attr_matrix",
    "attr_W1",
    "attr_b1",
    "attr_W2",
    "image_projection",
    "user_embedding",
    "item_embedding",
    "gen_layer1.weight",
    "gen_layer1.bias",
    "gen_layer2.weight",
    "gen_layer2.bias",
)


@dataclass
class ValidationContext:
    items: list[str]
    targets: dict[str, list[int]]
    categories: torch.Tensor
    images: torch.Tensor
    scores: torch.Tensor


def relative_pct(value: float, baseline: float) -> float:
    return (float(value) / float(baseline) - 1.0) * 100.0


def discover_run(result_root: Path, method_variant: str) -> Path:
    matches = []
    for config_path in result_root.rglob("run_config.json"):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if str(config.get("method_variant", "")) == method_variant:
            matches.append(config_path.parent)
    if len(matches) != 1:
        raise ValueError(
            f"expected one run for {method_variant} under {result_root}, got {len(matches)}"
        )
    return matches[0]


def build_context(
    *,
    code_root: Path,
    save_dict: dict,
    validate_csv: Path,
    profile_csv: Path,
) -> ValidationContext:
    amazon_code_dir = code_root / "Amazon VG"
    if str(amazon_code_dir) not in sys.path:
        sys.path.insert(0, str(amazon_code_dir))
    from support import build_item_feature_tensors

    items, targets = _validation_targets(validate_csv, save_dict["user_ser_dict"])
    item_map = {item: index for index, item in enumerate(items)}
    categories, images = build_item_feature_tensors(
        item_serialize_dict=item_map,
        img_features=save_dict["img_feature_dict"],
        genres=save_dict["asin_category_int_map"],
        category_num=int(save_dict["category_ser_map_len"]),
    )
    profile = pd.read_csv(profile_csv, dtype={"raw_asin": str}, low_memory=False)
    required = {"raw_asin", "split", "cicp_score"}
    missing = sorted(required - set(profile.columns))
    if missing:
        raise ValueError(f"profile missing columns: {missing}")
    validation = profile[profile["split"].astype(str).eq("validate")].copy()
    validation["raw_asin"] = validation["raw_asin"].astype(str)
    validation = validation.set_index("raw_asin").loc[items]
    scores = torch.as_tensor(
        pd.to_numeric(validation["cicp_score"], errors="raise").to_numpy(dtype="float32")
    )
    if len(items) != VALIDATION_ITEM_COUNT or len(scores) != VALIDATION_ITEM_COUNT:
        raise ValueError(f"unexpected validation coverage: {len(items)}/{len(scores)}")
    return ValidationContext(items, targets, categories, images, scores)


def feature_tensor(score: torch.Tensor) -> torch.Tensor:
    score = score.to(dtype=torch.float32).clamp(0.0, 1.0)
    return torch.stack((score, 1.0 - score, 4.0 * score * (1.0 - score)), dim=1)


def score_modes(scores: torch.Tensor) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(43)
    permutation = torch.randperm(len(scores), generator=generator)
    return {
        "true": scores.clone(),
        "neutral": torch.full_like(scores, 0.5),
        "shuffle": scores[permutation],
        "invert": 1.0 - scores,
        "zero": torch.zeros_like(scores),
        "one": torch.ones_like(scores),
    }


def _attention_statistics(model, categories, images, scores) -> dict[str, np.ndarray]:
    with torch.no_grad():
        z_v = torch.matmul(
            torch.matmul(model.attr_matrix, model.attr_W1) + model.attr_b1.squeeze(),
            model.attr_W2,
        ).squeeze(dim=1)
        mask = categories != -1
        neg_inf = torch.full_like(categories, -1e6, dtype=model.attr_matrix.dtype)
        base_logits = torch.where(mask, z_v.unsqueeze(0), neg_inf)
        base_attention = torch.softmax(base_logits, dim=1)
        p_v = torch.matmul(images, model.image_projection)
        image_query = F.normalize(model.cicpr2_attention_image_query(p_v), dim=1, eps=1e-6)
        category_key = F.normalize(
            model.cicpr2_attention_category_key(model.attr_matrix), dim=1, eps=1e-6
        )
        content_logits = torch.matmul(image_query, category_key.t()) / (
            model.cicpr2_cross_attention_temperature
        )
        content_logits = torch.where(mask, content_logits, neg_inf)
        content_attention = torch.softmax(content_logits, dim=1)
        counts = mask.sum(dim=1).clamp_min(1)
        denominator = torch.log(counts.to(dtype=torch.float32)).clamp_min(1e-6)

        def normalized_entropy(weights):
            entropy = -(weights * weights.clamp_min(1e-12).log()).sum(dim=1)
            return torch.where(counts > 1, entropy / denominator, torch.zeros_like(entropy))

        base_attr = torch.matmul(base_attention, model.attr_matrix)
        content_attr = torch.matmul(content_attention, model.attr_matrix)
        return {
            "base_attention_entropy": normalized_entropy(base_attention).cpu().numpy(),
            "content_attention_entropy": normalized_entropy(content_attention).cpu().numpy(),
            "base_attention_max": base_attention.max(dim=1).values.cpu().numpy(),
            "content_attention_max": content_attention.max(dim=1).values.cpu().numpy(),
            "base_content_attr_cosine": F.cosine_similarity(
                base_attr, content_attr, dim=1, eps=1e-6
            ).cpu().numpy(),
            "score_gate": (
                model.cicpr2_cross_attention_strength * scores.clamp(0.0, 1.0)
            ).cpu().numpy(),
        }


def evaluate_mode(
    *,
    model,
    method_variant: str,
    mode: str,
    scores: torch.Tensor,
    context: ValidationContext,
    device_name: str,
    batch_size: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    device = torch.device(device_name)
    original_variant = model.method_variant
    model.method_variant = "baseline" if mode == "off" else method_variant
    model.eval()
    features = None if mode == "off" else feature_tensor(scores)
    rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []
    residual_ratios: list[np.ndarray] = []
    residual_norms: list[np.ndarray] = []
    hidden_outputs: list[torch.Tensor] = []
    predicted_scores: list[np.ndarray] = []

    hook = None
    if method_variant in {"cicpr1_e4_residual", "cicpr2_content_direction_residual"}:
        hook = model.gen_layer1.register_forward_hook(
            lambda _module, _inputs, output: hidden_outputs.append(output.detach())
        )

    user_embedding = model.user_embedding.to(device)
    with torch.no_grad():
        for start in range(0, len(context.items), batch_size):
            end = min(start + batch_size, len(context.items))
            item_batch = context.items[start:end]
            category_batch = context.categories[start:end].to(device)
            image_batch = context.images[start:end].to(device)
            hidden_outputs.clear()
            if features is None:
                generated = model(category_batch, image_batch, len(item_batch))
            else:
                generated = model(
                    category_batch,
                    image_batch,
                    len(item_batch),
                    cicp_features=features[start:end].to(device),
                )
            embeddings.append(generated.detach().cpu().numpy())
            recommended = torch.topk(
                torch.matmul(generated, user_embedding.T), k=20, dim=1
            ).indices.cpu().tolist()
            for raw_asin, users in zip(item_batch, recommended):
                hr20, ndcg20 = _item_metrics(users, context.targets[raw_asin])
                rows.append(
                    {
                        "raw_asin": raw_asin,
                        "method_variant": method_variant,
                        "method_label": METHOD_LABELS[method_variant],
                        "intervention_mode": mode,
                        "cicp_score": 0.0,
                        "hr@20": hr20,
                        "ndcg@20": ndcg20,
                    }
                )
            if mode != "off" and model._last_cicp_residual is not None and hidden_outputs:
                residual = model._last_cicp_residual.detach()
                hidden = hidden_outputs[0]
                residual_norm = residual.norm(dim=1)
                hidden_norm = hidden.norm(dim=1).clamp_min(1e-12)
                residual_norms.append(residual_norm.cpu().numpy())
                residual_ratios.append((residual_norm / hidden_norm).cpu().numpy())
            if mode != "off" and method_variant == "cicpr2_score_distillation":
                predicted_scores.append(
                    model.predict_cicpr2_score(generated).detach().cpu().numpy()
                )
            print(
                f"intervention progress method={METHOD_LABELS[method_variant]} "
                f"mode={mode} items={end}/{len(context.items)}",
                flush=True,
            )
    if hook is not None:
        hook.remove()
    model.method_variant = original_variant
    frame = pd.DataFrame(rows)
    frame["cicp_score"] = context.scores.numpy()
    diagnostics: dict[str, Any] = {}
    if residual_ratios:
        ratios = np.concatenate(residual_ratios)
        norms = np.concatenate(residual_norms)
        cap = (
            model.cicp_residual_max_ratio
            if method_variant == "cicpr1_e4_residual"
            else model.cicpr2_residual_max_ratio
        )
        diagnostics.update(
            {
                "residual_ratio_mean": float(ratios.mean()),
                "residual_ratio_p50": float(np.quantile(ratios, 0.5)),
                "residual_ratio_p95": float(np.quantile(ratios, 0.95)),
                "residual_cap_saturation_share": float((ratios >= cap * 0.999).mean()),
                "residual_norm_mean": float(norms.mean()),
            }
        )
    if predicted_scores:
        predicted = np.concatenate(predicted_scores)
        target = context.scores.numpy()
        diagnostics.update(
            {
                "distilled_score_mae": float(np.abs(predicted - target).mean()),
                "distilled_score_spearman": float(spearmanr(predicted, target).statistic),
                "distilled_score_pearson": float(pearsonr(predicted, target).statistic),
                "distilled_score_prediction_std": float(predicted.std()),
            }
        )
    return frame, np.concatenate(embeddings), diagnostics


def summarize_mode(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    off_embeddings: np.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    difference = embeddings - off_embeddings
    off_norm = np.linalg.norm(off_embeddings, axis=1)
    relative_drift = np.linalg.norm(difference, axis=1) / np.clip(off_norm, 1e-12, None)
    cosine = np.sum(embeddings * off_embeddings, axis=1) / np.clip(
        np.linalg.norm(embeddings, axis=1) * off_norm, 1e-12, None
    )
    ndcg = float(frame["ndcg@20"].mean())
    hr = float(frame["hr@20"].mean())
    return {
        "method_variant": str(frame["method_variant"].iloc[0]),
        "method_label": str(frame["method_label"].iloc[0]),
        "intervention_mode": str(frame["intervention_mode"].iloc[0]),
        "item_count": int(len(frame)),
        "ndcg@20": ndcg,
        "relative_pct_ndcg@20_vs_official_baseline": relative_pct(
            ndcg, BASELINE_NDCG20
        ),
        "hr@20": hr,
        "relative_pct_hr@20_vs_official_baseline": relative_pct(hr, BASELINE_HR20),
        "embedding_relative_drift_vs_off_mean": float(relative_drift.mean()),
        "embedding_relative_drift_vs_off_p95": float(np.quantile(relative_drift, 0.95)),
        "embedding_cosine_vs_off_mean": float(cosine.mean()),
        **diagnostics,
    }


def evaluate_package(
    *,
    code_root: Path,
    method_variant: str,
    config: dict[str, Any],
    state_dict: dict,
    context: ValidationContext,
    device_name: str,
    batch_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    model = load_model(code_root, state_dict, config, device_name)
    modes = score_modes(context.scores)
    item_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    mechanism_rows: list[dict[str, Any]] = []

    off_frame, off_embeddings, off_diag = evaluate_mode(
        model=model,
        method_variant=method_variant,
        mode="off",
        scores=context.scores,
        context=context,
        device_name=device_name,
        batch_size=batch_size,
    )
    item_frames.append(off_frame)
    summaries.append(summarize_mode(off_frame, off_embeddings, off_embeddings, off_diag))
    selected_modes = modes if method_variant in DIRECT_SCORE_METHODS else {"true": modes["true"]}
    for mode, scores in selected_modes.items():
        frame, embeddings, diagnostics = evaluate_mode(
            model=model,
            method_variant=method_variant,
            mode=mode,
            scores=scores,
            context=context,
            device_name=device_name,
            batch_size=batch_size,
        )
        item_frames.append(frame)
        summaries.append(summarize_mode(frame, embeddings, off_embeddings, diagnostics))
        if mode == "true" and method_variant == "cicpr2_cross_modal_attention":
            values: dict[str, list[np.ndarray]] = {}
            device = torch.device(device_name)
            for start in range(0, len(context.items), batch_size):
                end = min(start + batch_size, len(context.items))
                stats = _attention_statistics(
                    model,
                    context.categories[start:end].to(device),
                    context.images[start:end].to(device),
                    context.scores[start:end].to(device),
                )
                for name, array in stats.items():
                    values.setdefault(name, []).append(array)
            for name, arrays in values.items():
                array = np.concatenate(arrays)
                mechanism_rows.append(
                    {
                        "method_variant": method_variant,
                        "method_label": METHOD_LABELS[method_variant],
                        "diagnostic": name,
                        "mean": float(array.mean()),
                        "p05": float(np.quantile(array, 0.05)),
                        "p50": float(np.quantile(array, 0.50)),
                        "p95": float(np.quantile(array, 0.95)),
                    }
                )

    dose_specs: list[tuple[str, str, list[float]]] = []
    if method_variant == "cicpr1_e4_residual":
        dose_specs.append(("cicp_residual_max_ratio", "cap", [0.05, 0.10, 0.15, 0.25, 0.40]))
    elif method_variant == "cicpr2_content_direction_residual":
        dose_specs.append(("cicpr2_residual_max_ratio", "cap", [0.05, 0.10, 0.15, 0.25, 0.40]))
    elif method_variant == "cicpr2_category_increment_gate":
        dose_specs.append(("cicpr2_increment_strength", "strength", [0.0, 0.25, 0.50, 0.75, 1.0]))
    elif method_variant == "cicpr2_cross_modal_attention":
        dose_specs.extend(
            [
                ("cicpr2_cross_attention_strength", "strength", [0.0, 0.25, 0.50, 0.75, 1.0]),
                ("cicpr2_cross_attention_temperature", "temperature", [0.125, 0.25, 0.50, 1.0]),
            ]
        )
    for attribute, label, values in dose_specs:
        original = float(getattr(model, attribute))
        for value in values:
            if np.isclose(value, original):
                continue
            setattr(model, attribute, value)
            mode = f"dose_{label}_{value:g}"
            frame, embeddings, diagnostics = evaluate_mode(
                model=model,
                method_variant=method_variant,
                mode=mode,
                scores=context.scores,
                context=context,
                device_name=device_name,
                batch_size=batch_size,
            )
            item_frames.append(frame)
            summaries.append(summarize_mode(frame, embeddings, off_embeddings, diagnostics))
        setattr(model, attribute, original)

    del model
    gc.collect()
    if device_name == "mps":
        torch.mps.empty_cache()
    elif device_name == "cuda":
        torch.cuda.empty_cache()
    return pd.concat(item_frames, ignore_index=True), pd.DataFrame(summaries), mechanism_rows


def initial_model_args(method_variant: str) -> SimpleNamespace:
    return SimpleNamespace(
        attr_num=3127,
        attr_present_dim=256,
        implicit_dim=256,
        cat_implicit_dim=256,
        user_number=52884,
        item_number=24726,
        pretrain=False,
        pretrain_update=False,
        method_variant=method_variant,
        category_conf_dim=16,
        category_conf_max_count=5,
        category_gate_scale=0.5,
        m11r2_feature_dim=16,
        m11r3_residual_max_ratio=0.15,
        m11r3_film_strength=0.1,
        m11r4_expert_film_strength=0.2,
        m11r4_fusion_strength=0.25,
        cicp_feature_dim=16,
        cicp_residual_max_ratio=0.15,
        cicp_modality_strength=0.25,
        cicp_expert_strength=0.20,
        cicp_attention_strength=0.50,
        cicpr2_residual_max_ratio=0.15,
        cicpr2_increment_strength=0.50,
        cicpr2_cross_attention_strength=0.50,
        cicpr2_cross_attention_temperature=0.25,
        cicpr2_category_dropout_max=0.50,
    )


def build_initialization_audit(code_root: Path) -> pd.DataFrame:
    amazon_code_dir = code_root / "Amazon VG"
    if str(amazon_code_dir) not in sys.path:
        sys.path.insert(0, str(amazon_code_dir))
    from model import CCFCRec, set_random_seed

    rows = []
    baseline_digest = None
    baseline_count = None
    for method_variant in METHOD_LABELS:
        set_random_seed(43)
        model = CCFCRec(initial_model_args(method_variant))
        state = model.state_dict()
        digest = hashlib.sha256()
        for name in COMMON_PARAMETER_NAMES:
            digest.update(state[name].detach().cpu().numpy().tobytes())
        common_digest = digest.hexdigest()
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        if method_variant == "baseline":
            baseline_digest = common_digest
            baseline_count = parameter_count
        rows.append(
            {
                "method_variant": method_variant,
                "method_label": METHOD_LABELS[method_variant],
                "total_parameter_count": parameter_count,
                "extra_parameter_count_vs_baseline": parameter_count - int(baseline_count or parameter_count),
                "common_initialization_sha256": common_digest,
                "common_initialization_equal_to_baseline": common_digest == baseline_digest,
            }
        )
        del model, state
        gc.collect()
    return pd.DataFrame(rows)


def build_signal_dose_audit(profile_csv: Path) -> pd.DataFrame:
    profile = pd.read_csv(profile_csv, low_memory=False)
    rows = []
    for split in ("train", "validate"):
        score = pd.to_numeric(
            profile.loc[profile["split"].astype(str).eq(split), "cicp_score"],
            errors="raise",
        ).to_numpy(dtype=float)
        implied = {
            "cicp_score": score,
            "CICP-R2-E1-CDR direct_scale": score,
            "CICP-R2-E2-CID category_gate": 1.0 + 0.5 * (2.0 * score - 1.0),
            "CICP-R2-E3-CMA blend_gate": 0.5 * score,
            "CICP-R2-E6-RCD whole_category_drop_probability": 0.5 * (1.0 - score),
        }
        for name, values in implied.items():
            rows.append(
                {
                    "split": split,
                    "quantity": name,
                    "item_count": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std()),
                    "min": float(values.min()),
                    "p10": float(np.quantile(values, 0.10)),
                    "p50": float(np.quantile(values, 0.50)),
                    "p90": float(np.quantile(values, 0.90)),
                    "max": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    code_root = args.code_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device_name = select_device(args.device)

    baseline_config, baseline_save, baseline_state, baseline_best = load_baseline_package(
        args.baseline_result.resolve(), device_name
    )
    context = build_context(
        code_root=code_root,
        save_dict=baseline_save,
        validate_csv=args.validate_csv.resolve(),
        profile_csv=args.profile_csv.resolve(),
    )
    packages = []
    packages.append(("baseline", baseline_config, baseline_state, baseline_best))
    for method_variant, root in [
        ("cicpr1_e4_residual", args.cicpr1_result_root.resolve()),
        *[(variant, args.cicpr2_result_root.resolve()) for variant in METHOD_LABELS if variant.startswith("cicpr2_")],
    ]:
        run_dir = discover_run(root, method_variant)
        config, save_dict, state_dict, best = load_run_package(run_dir, device_name)
        if save_dict["category_ser_map_len"] != baseline_save["category_ser_map_len"]:
            raise ValueError(f"category map mismatch for {method_variant}")
        packages.append((method_variant, config, state_dict, best))

    all_item_frames = []
    all_summaries = []
    mechanism_rows = []
    curve_audit = []
    for method_variant, config, state_dict, best in packages:
        if method_variant == "baseline":
            model = load_model(code_root, state_dict, config, device_name)
            frame, embeddings, diagnostics = evaluate_mode(
                model=model,
                method_variant="baseline",
                mode="off",
                scores=context.scores,
                context=context,
                device_name=device_name,
                batch_size=args.batch_size,
            )
            summary = summarize_mode(frame, embeddings, embeddings, diagnostics)
            del model
            item_frames = frame
            summaries = pd.DataFrame([summary])
            rows = []
        else:
            item_frames, summaries, rows = evaluate_package(
                code_root=code_root,
                method_variant=method_variant,
                config=config,
                state_dict=state_dict,
                context=context,
                device_name=device_name,
                batch_size=args.batch_size,
            )
        all_item_frames.append(item_frames)
        all_summaries.append(summaries)
        mechanism_rows.extend(rows)
        true_row = summaries[
            summaries["intervention_mode"].eq("true")
            if method_variant != "baseline"
            else summaries["intervention_mode"].eq("off")
        ].iloc[0]
        curve_audit.append(
            {
                "method_variant": method_variant,
                "method_label": METHOD_LABELS[method_variant],
                "best_epoch": int(best["epoch"]),
                "curve_ndcg@20": float(best["ndcg@20"]),
                "reaggregated_ndcg@20": float(true_row["ndcg@20"]),
                "absolute_error_ndcg@20": float(true_row["ndcg@20"] - best["ndcg@20"]),
                "curve_hr@20": float(best["hr@20"]),
                "reaggregated_hr@20": float(true_row["hr@20"]),
                "absolute_error_hr@20": float(true_row["hr@20"] - best["hr@20"]),
            }
        )

    item_metrics = pd.concat(all_item_frames, ignore_index=True)
    intervention_summary = pd.concat(all_summaries, ignore_index=True)
    initialization = build_initialization_audit(code_root)
    dose_audit = build_signal_dose_audit(args.profile_csv.resolve())
    mechanism_summary = pd.DataFrame(mechanism_rows)
    curve_audit_frame = pd.DataFrame(curve_audit)
    item_metrics.to_csv(output_dir / "cicp_root_intervention_item_metrics.csv", index=False)
    intervention_summary.to_csv(output_dir / "cicp_root_intervention_summary.csv", index=False)
    initialization.to_csv(output_dir / "cicp_root_initialization_audit.csv", index=False)
    dose_audit.to_csv(output_dir / "cicp_root_signal_dose_audit.csv", index=False)
    mechanism_summary.to_csv(output_dir / "cicp_root_mechanism_statistics.csv", index=False)
    curve_audit_frame.to_csv(output_dir / "cicp_root_curve_reaggregation_audit.csv", index=False)
    audit = {
        "protocol": "cicp_r1_r2_mechanism_sensitivity_v1",
        "device": device_name,
        "evaluated_split": "validate",
        "validation_item_count": len(context.items),
        "validation_coverage_pct_of_full_profile": len(context.items) / FULL_ITEM_COUNT * 100.0,
        "model_count": len(packages),
        "intervention_condition_count": int(len(intervention_summary)),
        "item_metric_row_count": int(len(item_metrics)),
        "max_curve_reaggregation_error_ndcg@20": float(
            curve_audit_frame["absolute_error_ndcg@20"].abs().max()
        ),
        "max_curve_reaggregation_error_hr@20": float(
            curve_audit_frame["absolute_error_hr@20"].abs().max()
        ),
        "test_item_metrics_read_or_generated": False,
        "train_recommendation_metrics_evaluated": False,
        "dose_sweeps_are_posthoc_diagnostic_only": True,
        "score_interventions_are_inference_only": True,
        "outputs": [
            "cicp_root_intervention_item_metrics.csv",
            "cicp_root_intervention_summary.csv",
            "cicp_root_initialization_audit.csv",
            "cicp_root_signal_dose_audit.csv",
            "cicp_root_mechanism_statistics.csv",
            "cicp_root_curve_reaggregation_audit.csv",
        ],
    }
    (output_dir / "cicp_root_diagnostic_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    return audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--cicpr1-result-root", type=Path, required=True)
    parser.add_argument("--cicpr2-result-root", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--validate-csv", type=Path, required=True)
    parser.add_argument("--profile-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
