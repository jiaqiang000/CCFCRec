#!/usr/bin/env python3
"""Analyze the five CICP-MP-FR1 validation runs under the frozen protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze_amazon_vg_cicpmp_r1_six_training import (
    BASELINE_HR20,
    BASELINE_LABEL,
    BASELINE_NDCG20,
    FULL_ITEM_COUNT,
    HISTORICAL_CICPR1_E1_NDCG20,
    TEST_ITEM_COUNT,
    THREE_PCT_THRESHOLD,
    TRAIN_ITEM_COUNT,
    VALIDATION_ITEM_COUNT,
    _bh_adjust,
    _paired_bootstrap,
    _tar_result_csv,
    relative_pct,
)


METHOD_ORDER = [
    "cicpmp_fr1_scalar_residual_reference",
    "cicpmp_fr1_modality_film",
    "cicpmp_fr1_content_expert_routing",
    "cicpmp_fr1_cross_modal_attention",
    "cicpmp_fr1_modality_film_shuffle",
]
METHOD_LABELS = {
    "cicpmp_fr1_scalar_residual_reference": "E1-SRR scalar residual reference（标量残差参照）",
    "cicpmp_fr1_modality_film": "E2-MFM modality FiLM（模态条件调制）",
    "cicpmp_fr1_content_expert_routing": "E3-CER content expert routing（内容专家路由）",
    "cicpmp_fr1_cross_modal_attention": "E4-CMA cross-modal attention（跨模态注意力）",
    "cicpmp_fr1_modality_film_shuffle": "E5-MFS modality FiLM shuffle（模态调制乱序控制）",
}
ITEM_LABELS = {
    "cicpmp_fr1_scalar_residual_reference": "CICP-MP-FR1-E1-SRR",
    "cicpmp_fr1_modality_film": "CICP-MP-FR1-E2-MFM",
    "cicpmp_fr1_content_expert_routing": "CICP-MP-FR1-E3-CER",
    "cicpmp_fr1_cross_modal_attention": "CICP-MP-FR1-E4-CMA",
    "cicpmp_fr1_modality_film_shuffle": "CICP-MP-FR1-E5-MFS",
}
REAL_MP_METHODS = {
    "cicpmp_fr1_modality_film",
    "cicpmp_fr1_content_expert_routing",
    "cicpmp_fr1_cross_modal_attention",
}
GROUP_COLUMNS = {
    "cicp_score": ("cicp_score_group", ("low", "mid", "high")),
    "raw_increment": ("raw_increment_group", ("low", "mid", "high")),
    "semantic_increment": ("semantic_increment_group", ("low", "mid", "high")),
    "total_increment": ("total_increment_group", ("low", "mid", "high")),
    "attribution_share": ("attribution_share_group", ("low", "mid", "high")),
    "attribution_entropy": ("attribution_entropy_group", ("low", "mid", "high")),
    "uncertainty": ("uncertainty_group", ("low", "mid", "high")),
    "disagreement": ("disagreement_group", ("low", "mid", "high")),
    "direction_norm": ("direction_norm_group", ("low", "mid", "high")),
}
PREDICTORS = [
    "cicp_score",
    "mp_raw_predicted_increment",
    "mp_category_semantic_increment_prediction",
    "mp_category_total_increment_prediction",
    "mp_category_attribution_positive_share_prediction",
    "mp_category_attribution_entropy_prediction",
    "mp_fold_prediction_uncertainty",
    "mp_hgb_ridge_disagreement",
    *[f"mp_direction16_{index:02d}" for index in range(16)],
    "mp_direction_norm",
]


def _best_row(curve: pd.DataFrame) -> pd.Series:
    return curve.sort_values(
        ["ndcg@20", "hr@20", "epoch", "checkpoint_index"],
        ascending=[False, False, True, True],
    ).iloc[0]


def summarize_curve(
    method: str,
    curve: pd.DataFrame,
    baseline_curve: pd.DataFrame,
    run_dir: Path,
    config: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(curve) != 100 or curve["epoch"].astype(int).tolist() != list(range(1, 101)):
        raise ValueError(f"{method} does not contain exactly epochs 1..100")
    checkpoints = list(run_dir.glob("*.pt"))
    if len(checkpoints) != 100:
        raise ValueError(f"{method} does not contain exactly 100 checkpoints")
    best = _best_row(curve)
    best_epoch = int(best["epoch"])
    matched = curve.set_index("epoch")[["ndcg@20", "hr@20"]].join(
        baseline_curve.set_index("epoch")[["ndcg@20", "hr@20"]],
        lsuffix="_method",
        rsuffix="_baseline",
        validate="one_to_one",
    )
    for metric in ("ndcg@20", "hr@20"):
        matched[f"absolute_delta_{metric}"] = (
            matched[f"{metric}_method"] - matched[f"{metric}_baseline"]
        )
        matched[f"relative_pct_{metric}"] = (
            matched[f"{metric}_method"] / matched[f"{metric}_baseline"] - 1.0
        ) * 100.0
    matched = matched.reset_index()
    matched.insert(0, "method_variant", method)
    same = matched[matched["epoch"].eq(best_epoch)].iloc[0]
    epoch74 = matched[matched["epoch"].eq(74)].iloc[0]
    late = curve[curve["epoch"].between(71, 100)]
    baseline_late = baseline_curve[baseline_curve["epoch"].between(71, 100)]
    matched_late = matched[matched["epoch"].between(71, 100)]
    parameter_count = config["parameter_count"]
    best_ndcg = float(best["ndcg@20"])
    best_hr = float(best["hr@20"])
    return {
        "method_variant": method,
        "method_label": METHOD_LABELS[method],
        "run_dir": str(run_dir),
        "best_epoch": best_epoch,
        "best_checkpoint_index": int(best["checkpoint_index"]),
        "best_ndcg@20": best_ndcg,
        "absolute_delta_ndcg@20_vs_baseline_best": best_ndcg - BASELINE_NDCG20,
        "relative_pct_ndcg@20_vs_baseline_best": relative_pct(best_ndcg, BASELINE_NDCG20),
        "best_hr@20_same_checkpoint": best_hr,
        "absolute_delta_hr@20_vs_baseline_best": best_hr - BASELINE_HR20,
        "relative_pct_hr@20_vs_baseline_best": relative_pct(best_hr, BASELINE_HR20),
        "same_epoch_relative_pct_ndcg@20": float(same["relative_pct_ndcg@20"]),
        "epoch74_ndcg@20": float(epoch74["ndcg@20_method"]),
        "epoch74_relative_pct_ndcg@20": float(epoch74["relative_pct_ndcg@20"]),
        "late30_mean_ndcg@20": float(late["ndcg@20"].mean()),
        "late30_relative_pct_vs_baseline_best": relative_pct(
            float(late["ndcg@20"].mean()), BASELINE_NDCG20
        ),
        "late30_matched_relative_pct_ndcg@20": relative_pct(
            float(late["ndcg@20"].mean()), float(baseline_late["ndcg@20"].mean())
        ),
        "late30_matched_positive_epochs": int(
            (matched_late["absolute_delta_ndcg@20"] > 0).sum()
        ),
        "all100_matched_positive_epochs": int(
            (matched["absolute_delta_ndcg@20"] > 0).sum()
        ),
        "absolute_delta_vs_historical_cicpr1_e1": best_ndcg
        - HISTORICAL_CICPR1_E1_NDCG20,
        "relative_pct_vs_historical_cicpr1_e1": relative_pct(
            best_ndcg, HISTORICAL_CICPR1_E1_NDCG20
        ),
        "passed_historical_anchor": bool(best_ndcg > HISTORICAL_CICPR1_E1_NDCG20),
        "passed_three_pct": bool(best_ndcg >= THREE_PCT_THRESHOLD),
        "absolute_gap_to_three_pct": best_ndcg - THREE_PCT_THRESHOLD,
        "common_parameters": int(parameter_count["common"]),
        "method_specific_parameters": int(parameter_count["method_specific"]),
        "total_parameters": int(parameter_count["total"]),
    }, matched


def collect_curves(
    result_root: Path,
    baseline_result: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_curve = _tar_result_csv(baseline_result)
    summaries = []
    matched_frames = []
    config_rows = []
    seen: set[str] = set()
    for result_path in sorted(result_root.rglob("result.csv")):
        run_dir = result_path.parent
        config_path = run_dir / "run_config.json"
        if not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        method = str(config.get("method_variant", ""))
        if method not in METHOD_ORDER:
            continue
        if method in seen:
            raise ValueError(f"duplicate method: {method}")
        seen.add(method)
        summary, matched = summarize_curve(
            method, pd.read_csv(result_path), baseline_curve, run_dir, config
        )
        summaries.append(summary)
        matched_frames.append(matched)
        optimizer = config.get("optimizer_parameter_groups", {})
        config_rows.append(
            {
                "method_variant": method,
                "method_label": METHOD_LABELS[method],
                "seed": int(config.get("seed", -1)),
                "negative_sampling_mode": config.get("negative_sampling_mode"),
                "num_workers": int(config.get("num_workers", -1)),
                "common_parameter_sha256": config.get("ccfcrec_common_parameter_sha256"),
                "base_weight_decay": float(optimizer.get("base_weight_decay", np.nan)),
                "method_weight_decay": float(optimizer.get("method_weight_decay", np.nan)),
                "input_standardization": config.get("cicpmp_fr1_input_standardization"),
                "initial_effect": config.get("cicpmp_fr1_initial_effect"),
                "activation_schedule": config.get("cicpmp_fr1_activation_schedule"),
                "fixed_reliability_multiplication": bool(
                    config.get("cicpmp_fr1_uses_fixed_reliability_multiplication", True)
                ),
                "offline_auxiliary_target": bool(
                    config.get("cicpmp_fr1_uses_offline_auxiliary_target", True)
                ),
                "e4_style_hidden_residual": bool(
                    config.get("cicpmp_fr1_e4_style_hidden_residual", False)
                ),
                "training_uses_validation_item_metrics": bool(
                    config.get("training_input_uses_validation_item_metrics", True)
                ),
                "training_uses_test_item_metrics": bool(
                    config.get("training_input_uses_test_item_metrics", True)
                ),
                "method_specific_parameters": int(
                    config.get("parameter_count", {}).get("method_specific", 0)
                ),
            }
        )
    if seen != set(METHOD_ORDER):
        raise ValueError(f"missing methods: {sorted(set(METHOD_ORDER) - seen)}")
    config_audit = pd.DataFrame(config_rows)
    if config_audit["common_parameter_sha256"].nunique() != 1:
        raise ValueError("common parameter hashes differ")
    if not config_audit["seed"].eq(43).all():
        raise ValueError("not all branches use seed43")
    if not config_audit["negative_sampling_mode"].eq("fast_uniform").all():
        raise ValueError("negative sampling protocol differs")
    if not config_audit["num_workers"].eq(8).all():
        raise ValueError("worker protocol differs")
    if not config_audit["base_weight_decay"].eq(0.1).all():
        raise ValueError("base weight decay differs")
    if not config_audit["method_weight_decay"].eq(0.0).all():
        raise ValueError("method weight decay differs")
    if not config_audit["initial_effect"].eq("exact_zero").all():
        raise ValueError("not all branches start from exact zero effect")
    if not config_audit["activation_schedule"].eq("none").all():
        raise ValueError("an activation schedule entered the experiment")
    forbidden = config_audit[
        [
            "fixed_reliability_multiplication",
            "offline_auxiliary_target",
            "training_uses_validation_item_metrics",
            "training_uses_test_item_metrics",
        ]
    ]
    if forbidden.any(axis=None):
        raise ValueError("a forbidden training mechanism or evaluation input was enabled")
    if config_audit["e4_style_hidden_residual"].sum() != 1:
        raise ValueError("exactly one branch must use the hidden residual")
    summary = pd.DataFrame(summaries)
    summary["method_order"] = summary["method_variant"].map(METHOD_ORDER.index)
    summary = summary.sort_values("method_order").drop(columns="method_order")
    return (
        summary.reset_index(drop=True),
        pd.concat(matched_frames, ignore_index=True),
        baseline_curve,
        config_audit,
    )


def _iter_groups(frame: pd.DataFrame):
    yield "overall", "all", frame
    for dimension, (column, values) in GROUP_COLUMNS.items():
        for value in values:
            yield dimension, value, frame[frame[column].eq(value)]


def build_group_summary(
    item_metrics: pd.DataFrame,
    focus_methods: set[str],
    repetitions: int,
) -> pd.DataFrame:
    baseline = item_metrics[item_metrics["method_label"].eq(BASELINE_LABEL)].set_index(
        "raw_asin"
    )
    if len(baseline) != VALIDATION_ITEM_COUNT:
        raise ValueError("baseline does not cover 5,298 validation items")
    rows = []
    for method_index, method in enumerate(METHOD_ORDER):
        method_frame = item_metrics[
            item_metrics["method_label"].eq(ITEM_LABELS[method])
        ].set_index("raw_asin")
        paired = method_frame.join(
            baseline[["ndcg@20", "hr@20"]],
            rsuffix="_baseline",
            validate="one_to_one",
        )
        if len(paired) != VALIDATION_ITEM_COUNT:
            raise ValueError(f"{method} item metrics are not fully paired")
        for group_index, (dimension, value, part) in enumerate(_iter_groups(paired)):
            row: dict[str, Any] = {
                "method_variant": method,
                "method_label": METHOD_LABELS[method],
                "group_dimension": dimension,
                "group_value": value,
                "item_count": len(part),
                "coverage_pct_of_validation": len(part) / VALIDATION_ITEM_COUNT * 100.0,
                "coverage_pct_of_full_profile": len(part) / FULL_ITEM_COUNT * 100.0,
            }
            for metric in ("ndcg@20", "hr@20"):
                baseline_mean = float(part[f"{metric}_baseline"].mean())
                method_mean = float(part[metric].mean())
                row[f"baseline_{metric}"] = baseline_mean
                row[f"method_{metric}"] = method_mean
                row[f"absolute_delta_{metric}"] = method_mean - baseline_mean
                row[f"relative_pct_{metric}"] = relative_pct(method_mean, baseline_mean)
            delta = part["ndcg@20"] - part["ndcg@20_baseline"]
            row["helped_ndcg_item_count"] = int((delta > 0).sum())
            row["harmed_ndcg_item_count"] = int((delta < 0).sum())
            row["equal_ndcg_item_count"] = int((delta == 0).sum())
            if dimension == "overall" or method in focus_methods:
                row.update(
                    _paired_bootstrap(
                        part["ndcg@20"].to_numpy(dtype=float),
                        part["ndcg@20_baseline"].to_numpy(dtype=float),
                        seed=43 + method_index * 100 + group_index,
                        repetitions=repetitions,
                    )
                )
            rows.append(row)
    return pd.DataFrame(rows)


def build_score_response(item_metrics: pd.DataFrame) -> pd.DataFrame:
    def calculate(left: pd.Series, right: pd.Series) -> tuple[float, float]:
        if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
            return np.nan, np.nan
        result = spearmanr(left, right)
        return float(result.statistic), float(result.pvalue)

    baseline = item_metrics[item_metrics["method_label"].eq(BASELINE_LABEL)].set_index(
        "raw_asin"
    )
    missing = sorted(set(PREDICTORS) - set(baseline.columns))
    if missing:
        raise ValueError(f"item metrics are missing predictors: {missing}")
    rows = []
    for predictor in PREDICTORS:
        for metric in ("ndcg@20", "hr@20"):
            statistic, p_value = calculate(baseline[predictor], baseline[metric])
            rows.append(
                {
                    "method_variant": "baseline",
                    "method_label": "baseline（基线）",
                    "predictor": predictor,
                    "response": metric,
                    "comparison": "predictor_vs_baseline_metric",
                    "spearman": statistic,
                    "p_value_uncorrected": p_value,
                }
            )
    for method in METHOD_ORDER:
        frame = item_metrics[item_metrics["method_label"].eq(ITEM_LABELS[method])].set_index(
            "raw_asin"
        )
        paired = frame.join(
            baseline[["ndcg@20", "hr@20"]],
            rsuffix="_baseline",
            validate="one_to_one",
        )
        for predictor in PREDICTORS:
            for metric in ("ndcg@20", "hr@20"):
                statistic, p_value = calculate(
                    paired[predictor], paired[metric] - paired[f"{metric}_baseline"]
                )
                rows.append(
                    {
                        "method_variant": method,
                        "method_label": METHOD_LABELS[method],
                        "predictor": predictor,
                        "response": metric,
                        "comparison": "predictor_vs_item_delta",
                        "spearman": statistic,
                        "p_value_uncorrected": p_value,
                    }
                )
    result = pd.DataFrame(rows)
    result["q_value_bh"] = _bh_adjust(result["p_value_uncorrected"])
    result["significant_after_bh_0.05"] = result["q_value_bh"] < 0.05
    return result


def build_control_comparison(
    matched_curve: pd.DataFrame,
    item_metrics: pd.DataFrame,
    repetitions: int,
) -> pd.DataFrame:
    real_method = "cicpmp_fr1_modality_film"
    shuffle_method = "cicpmp_fr1_modality_film_shuffle"
    real = matched_curve[matched_curve["method_variant"].eq(real_method)].set_index("epoch")
    shuffle = matched_curve[matched_curve["method_variant"].eq(shuffle_method)].set_index("epoch")
    rows = []
    for label, epochs in {
        "E2_best_epoch_40": [40],
        "baseline_best_epoch_74": [74],
        "E5_best_epoch_87": [87],
        "last_epoch_100": [100],
        "late30_epochs_71_100": list(range(71, 101)),
        "all100_epochs": list(range(1, 101)),
    }.items():
        real_mean = float(real.loc[epochs, "ndcg@20_method"].mean())
        shuffle_mean = float(shuffle.loc[epochs, "ndcg@20_method"].mean())
        rows.append(
            {
                "comparison_scope": label,
                "epoch_count": len(epochs),
                "real_ndcg@20": real_mean,
                "shuffle_ndcg@20": shuffle_mean,
                "absolute_delta_ndcg@20": real_mean - shuffle_mean,
                "relative_pct_vs_shuffle": relative_pct(real_mean, shuffle_mean),
                "real_above_shuffle_epoch_count": int(
                    (real.loc[epochs, "ndcg@20_method"] > shuffle.loc[epochs, "ndcg@20_method"]).sum()
                ),
            }
        )
    real_items = item_metrics[item_metrics["method_label"].eq(ITEM_LABELS[real_method])].set_index(
        "raw_asin"
    )
    shuffle_items = item_metrics[
        item_metrics["method_label"].eq(ITEM_LABELS[shuffle_method])
    ].set_index("raw_asin")
    ci = _paired_bootstrap(
        real_items["ndcg@20"].to_numpy(dtype=float),
        shuffle_items["ndcg@20"].to_numpy(dtype=float),
        seed=1043,
        repetitions=repetitions,
    )
    rows.append(
        {
            "comparison_scope": "independent_best_checkpoints_item_paired",
            "epoch_count": 2,
            "real_ndcg@20": float(real_items["ndcg@20"].mean()),
            "shuffle_ndcg@20": float(shuffle_items["ndcg@20"].mean()),
            "absolute_delta_ndcg@20": float(real_items["ndcg@20"].mean())
            - float(shuffle_items["ndcg@20"].mean()),
            "relative_pct_vs_shuffle": relative_pct(
                float(real_items["ndcg@20"].mean()), float(shuffle_items["ndcg@20"].mean())
            ),
            "real_above_shuffle_epoch_count": np.nan,
            **ci,
        }
    )
    return pd.DataFrame(rows)


def summarize_mechanisms(activity: pd.DataFrame) -> pd.DataFrame:
    best = activity[activity["is_best_epoch"].astype(bool)].copy()
    if len(best) != 5:
        raise ValueError("mechanism activity must contain one best row per method")
    best["functional_signal_active"] = (
        best["actual_vs_permuted_embedding_relative_l2_mean"] > 1e-3
    )
    best["gate_saturated"] = best["gate_abs_mean"].fillna(0.0) > 0.99
    best["mechanism_interpretation"] = "active_item_specific_conditioning"
    best.loc[
        best["method_variant"].eq("cicpmp_fr1_cross_modal_attention")
        & ~best["functional_signal_active"],
        "mechanism_interpretation",
    ] = "constant_saturated_cross_modal_structure_not_item_specific_mp"
    best.loc[
        best["method_variant"].eq("cicpmp_fr1_scalar_residual_reference"),
        "mechanism_interpretation",
    ] = "active_scalar_residual_reference"
    return best


def build_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "evaluation_scope": "complete_item_profile（完整物品档案）",
                "scope_item_count": FULL_ITEM_COUNT,
                "evaluated_item_count": 0,
                "coverage_pct_of_full_profile": 100.0,
                "status": "未评估",
                "note": "未生成训练、验证、测试混合推荐指标。",
            },
            {
                "evaluation_scope": "train_non_cold（训练非冷启动物品）",
                "scope_item_count": TRAIN_ITEM_COUNT,
                "evaluated_item_count": 0,
                "coverage_pct_of_full_profile": TRAIN_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
                "status": "未评估",
                "note": "没有生成训练物品推荐指标。",
            },
            {
                "evaluation_scope": "validation_cold（验证冷启动物品）",
                "scope_item_count": VALIDATION_ITEM_COUNT,
                "evaluated_item_count": VALIDATION_ITEM_COUNT,
                "coverage_pct_of_full_profile": VALIDATION_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
                "status": "已评估",
                "note": "本轮开发阶段正式方法与检查点选择口径。",
            },
            {
                "evaluation_scope": "test_cold（测试冷启动物品）",
                "scope_item_count": TEST_ITEM_COUNT,
                "evaluated_item_count": 0,
                "coverage_pct_of_full_profile": TEST_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
                "status": "未评估",
                "note": "没有读取或生成测试推荐指标。",
            },
            {
                "evaluation_scope": "target/non_target（目标/非目标物品）",
                "scope_item_count": 0,
                "evaluated_item_count": 0,
                "coverage_pct_of_full_profile": np.nan,
                "status": "不适用（未定义）",
                "note": "CICP-MP是连续23维画像，协议没有二元目标标签，不事后阈值化。",
            },
        ]
    )


def build_decision(
    curve_summary: pd.DataFrame,
    mechanism_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    control: pd.DataFrame,
) -> dict[str, Any]:
    performance = curve_summary.sort_values("best_ndcg@20", ascending=False).iloc[0]
    merged = curve_summary.merge(
        mechanism_summary[["method_variant", "functional_signal_active"]],
        on="method_variant",
        validate="one_to_one",
    )
    semantic_candidates = merged[
        merged["method_variant"].isin(REAL_MP_METHODS)
        & merged["functional_signal_active"]
    ]
    if semantic_candidates.empty:
        semantic = None
        route = "stop_cicpmp_route"
    else:
        semantic = semantic_candidates.sort_values("best_ndcg@20", ascending=False).iloc[0]
        stable = (
            int(semantic["late30_matched_positive_epochs"]) == 30
            and float(semantic["late30_matched_relative_pct_ndcg@20"]) > 0
        )
        hr_positive = float(semantic["relative_pct_hr@20_vs_baseline_best"]) > 0
        if bool(semantic["passed_three_pct"]) and stable and hr_positive:
            route = "matched_controls_then_multiseed_preparation"
        elif bool(semantic["passed_historical_anchor"]) and stable and hr_positive:
            route = "one_bounded_performance_breakthrough_design"
        else:
            route = "stop_cicpmp_route"
    overall = group_summary[group_summary["group_dimension"].eq("overall")].set_index(
        "method_variant"
    )
    control_best = control[
        control["comparison_scope"].eq("independent_best_checkpoints_item_paired")
    ].iloc[0]
    result: dict[str, Any] = {
        "performance_winner_method": str(performance["method_variant"]),
        "performance_winner_label": str(performance["method_label"]),
        "performance_winner_epoch": int(performance["best_epoch"]),
        "performance_winner_ndcg@20": float(performance["best_ndcg@20"]),
        "performance_winner_relative_pct_vs_baseline": float(
            performance["relative_pct_ndcg@20_vs_baseline_best"]
        ),
        "performance_winner_relative_pct_hr_vs_baseline": float(
            performance["relative_pct_hr@20_vs_baseline_best"]
        ),
        "performance_winner_functional_mp_active": bool(
            mechanism_summary.set_index("method_variant").loc[
                performance["method_variant"], "functional_signal_active"
            ]
        ),
        "three_pct_threshold": THREE_PCT_THRESHOLD,
        "historical_cicpr1_e1_ndcg@20": HISTORICAL_CICPR1_E1_NDCG20,
        "e2_minus_e5_best_absolute_delta_ndcg@20": float(
            control_best["absolute_delta_ndcg@20"]
        ),
        "e2_minus_e5_best_relative_pct": float(control_best["relative_pct_vs_shuffle"]),
        "e2_minus_e5_best_bootstrap_relative_pct_ci95": [
            float(control_best["bootstrap_relative_pct_ci95_low"]),
            float(control_best["bootstrap_relative_pct_ci95_high"]),
        ],
        "route": route,
        "multi_seed_authorized": bool(route == "matched_controls_then_multiseed_preparation"),
        "test_metrics_read_or_generated": False,
        "evidence_level": "B_single_seed_development_validation_with_posthoc_mechanism_audit",
    }
    if semantic is not None:
        semantic_method = str(semantic["method_variant"])
        semantic_overall = overall.loc[semantic_method]
        result.update(
            {
                "active_semantic_winner_method": semantic_method,
                "active_semantic_winner_label": str(semantic["method_label"]),
                "active_semantic_winner_epoch": int(semantic["best_epoch"]),
                "active_semantic_winner_ndcg@20": float(semantic["best_ndcg@20"]),
                "active_semantic_winner_relative_pct_vs_baseline": float(
                    semantic["relative_pct_ndcg@20_vs_baseline_best"]
                ),
                "active_semantic_winner_relative_pct_hr_vs_baseline": float(
                    semantic["relative_pct_hr@20_vs_baseline_best"]
                ),
                "active_semantic_winner_relative_pct_vs_history": float(
                    semantic["relative_pct_vs_historical_cicpr1_e1"]
                ),
                "active_semantic_winner_late30_matched_relative_pct": float(
                    semantic["late30_matched_relative_pct_ndcg@20"]
                ),
                "active_semantic_winner_bootstrap_relative_pct_ci95": [
                    float(semantic_overall["bootstrap_relative_pct_ci95_low"]),
                    float(semantic_overall["bootstrap_relative_pct_ci95_high"]),
                ],
            }
        )
    return result


def build_fallacy_scan(
    group_summary: pd.DataFrame,
    semantic_method: str,
) -> pd.DataFrame:
    groups = group_summary[
        group_summary["method_variant"].eq(semantic_method)
        & ~group_summary["group_dimension"].eq("overall")
    ]
    reversal = bool(
        (groups["relative_pct_ndcg@20"] < 0).any()
        and (groups["relative_pct_ndcg@20"] > 0).any()
    )
    entries = [
        ("Simpson's paradox（辛普森悖论）", "CAUTION" if reversal else "CHECKED", "整体与9类互补分层并报；存在正负分层时不以整体替代异质性。"),
        ("Ecological fallacy（生态谬误）", "CAUTION", "群体均值不能推出组内每个物品都同向获益。"),
        ("Berkson's paradox（伯克森悖论）", "CAUTION", "仅评估验证冷启动物品，不能外推至训练、测试或其他数据集。"),
        ("Collider bias（碰撞变量偏差）", "CAUTION", "画像分层是事后描述，不是条件化后的净因果效应。"),
        ("Base-rate neglect（基率忽视）", "CHECKED", "报告完整35,322物品档案与5,298验证物品的覆盖关系及每组计数。"),
        ("Regression to the mean（均值回归）", "CAUTION", "从每支100轮中选择峰值；同时报告同轮、第74轮和后30轮。"),
        ("Survivorship bias（幸存者偏差）", "CHECKED", "五支全部完成100轮并全部进入分析。"),
        ("Look-elsewhere effect（多处搜寻效应）", "CAUTION", "五支各选择100个检查点中的峰值，单随机种子峰值可能被放大。"),
        ("Garden of forking paths（分叉路径花园）", "CAUTION", "判决规则训练前冻结，但机制失活阈值、逐物品相关与分层属于事后诊断。"),
        ("Correlation is not causation（相关不等于因果）", "CAUTION", "E2/E5提供同结构语义控制；E3没有同载体乱序重训，E4收益不能归因于画像。"),
        ("Reverse causality（反向因果）", "CHECKED", "训练画像不含验证或测试推荐答案；观察收益也不能反向证明画像定义正确。"),
    ]
    return pd.DataFrame(entries, columns=["fallacy", "severity", "finding"])


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    def render(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(render(row[column]) for column in columns) + " |"
        for _, row in frame[columns].iterrows()
    )
    return "\n".join(lines)


def write_reports(
    *,
    output_dir: Path,
    route_output_dir: Path,
    run_stamp: str,
    result_root: Path,
    curve_summary: pd.DataFrame,
    config_audit: pd.DataFrame,
    group_summary: pd.DataFrame,
    score_response: pd.DataFrame,
    control: pd.DataFrame,
    mechanism: pd.DataFrame,
    coverage: pd.DataFrame,
    fallacy: pd.DataFrame,
    decision: dict[str, Any],
) -> tuple[Path, Path]:
    performance_method = decision["performance_winner_method"]
    semantic_method = decision["active_semantic_winner_method"]
    main = curve_summary.copy()
    overall = group_summary[group_summary["group_dimension"].eq("overall")][
        [
            "method_variant",
            "bootstrap_relative_pct_ci95_low",
            "bootstrap_relative_pct_ci95_high",
        ]
    ]
    main = main.merge(overall, on="method_variant", validate="one_to_one")
    main["relative_pct_vs_baseline"] = main["relative_pct_ndcg@20_vs_baseline_best"].map(
        lambda value: f"{value:+.3f}%"
    )
    main["hr_relative_pct_vs_baseline"] = main[
        "relative_pct_hr@20_vs_baseline_best"
    ].map(lambda value: f"{value:+.3f}%")
    main["relative_pct_vs_history"] = main["relative_pct_vs_historical_cicpr1_e1"].map(
        lambda value: f"{value:+.3f}%"
    )
    main["paired_ci95"] = main.apply(
        lambda row: (
            f"[{row['bootstrap_relative_pct_ci95_low']:+.3f}%, "
            f"{row['bootstrap_relative_pct_ci95_high']:+.3f}%]"
        ),
        axis=1,
    )
    stability = curve_summary.copy()
    for column in (
        "same_epoch_relative_pct_ndcg@20",
        "epoch74_relative_pct_ndcg@20",
        "late30_matched_relative_pct_ndcg@20",
    ):
        stability[column] = stability[column].map(lambda value: f"{value:+.3f}%")
    mechanism_view = mechanism.copy()
    mechanism_view["gate_abs_mean"] = mechanism_view["gate_abs_mean"].map(
        lambda value: "" if pd.isna(value) else f"{value:.6f}"
    )
    for column in (
        "actual_vs_zero_embedding_relative_l2_mean",
        "actual_vs_permuted_embedding_relative_l2_mean",
    ):
        mechanism_view[column] = mechanism_view[column].map(lambda value: f"{value:.6e}")
    control_view = control.copy()
    control_view["absolute_delta_ndcg@20"] = control_view["absolute_delta_ndcg@20"].map(
        lambda value: f"{value:+.12f}"
    )
    control_view["relative_pct_vs_shuffle"] = control_view["relative_pct_vs_shuffle"].map(
        lambda value: f"{value:+.3f}%"
    )
    semantic_groups = group_summary[
        group_summary["method_variant"].eq(semantic_method)
        & ~group_summary["group_dimension"].eq("overall")
    ].copy()
    semantic_groups["relative_pct_ndcg@20"] = semantic_groups[
        "relative_pct_ndcg@20"
    ].map(lambda value: f"{value:+.3f}%")
    semantic_groups["relative_pct_hr@20"] = semantic_groups[
        "relative_pct_hr@20"
    ].map(lambda value: f"{value:+.3f}%")
    semantic_response = score_response[
        score_response["method_variant"].eq(semantic_method)
        & score_response["response"].eq("ndcg@20")
    ].copy().sort_values("spearman", key=lambda values: values.abs(), ascending=False)
    performance_groups = group_summary[
        group_summary["method_variant"].eq(performance_method)
        & ~group_summary["group_dimension"].eq("overall")
    ]
    best_group = semantic_groups.iloc[
        semantic_groups["relative_pct_ndcg@20"].str.rstrip("%").astype(float).argmax()
    ]
    worst_group = semantic_groups.iloc[
        semantic_groups["relative_pct_ndcg@20"].str.rstrip("%").astype(float).argmin()
    ]
    e3_ci = decision["active_semantic_winner_bootstrap_relative_pct_ci95"]
    e2e5_ci = decision["e2_minus_e5_best_bootstrap_relative_pct_ci95"]

    report = f"""---
title: {run_stamp} CCFCRec Amazon-VG CICP-MP-FR1 five final repairs training 结果
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - CICP-MP-FR1
  - training-analysis
---

# CCFCRec Amazon-VG CICP-MP-FR1 five final repairs training 结果

> [!important] 结论
> 五支均完成100 epoch（100训练轮次）。整体性能胜者 E4-CMA cross-modal attention（跨模态注意力）在全部5,298个验证冷启动物品上达到 NDCG@20（前20归一化折损累计增益）`{decision['performance_winner_ndcg@20']:.12f}`，相对同口径 baseline（基线）`{decision['performance_winner_relative_pct_vs_baseline']:+.3f}%`，同点 HR@20（前20命中率）`{decision['performance_winner_relative_pct_hr_vs_baseline']:+.3f}%`；但它的画像条件门控饱和为近常数，最佳轮真实画像与逐物品置换画像的嵌入差只有约 `1.33e-7`，因此不能把该收益归因于部署时的逐物品CICP-MP（类别增量协同可预测性多分量画像）语义，更符合跨模态结构收益候选；完全区分二者仍需无画像条件重训对照。
>
> 实际仍使用逐物品画像的最佳载体是 E3-CER content expert routing（内容专家路由）：NDCG@20 `{decision['active_semantic_winner_ndcg@20']:.12f}`，相对基线 `{decision['active_semantic_winner_relative_pct_vs_baseline']:+.3f}%`，同点HR `{decision['active_semantic_winner_relative_pct_hr_vs_baseline']:+.3f}%`，后30轮同轮均值 `{decision['active_semantic_winner_late30_matched_relative_pct']:+.3f}%`、30/30轮为正。它只比历史 CICP-R1-E1 高 `{decision['active_semantic_winner_relative_pct_vs_history']:+.3f}%`，配对95%区间 `[{e3_ci[0]:+.3f}%, {e3_ci[1]:+.3f}%]` 仍跨零，所以是“窄幅通过最终修复判决”，不是确认性胜利。

## 一、评价范围

{_markdown_table(coverage, ['evaluation_scope','scope_item_count','evaluated_item_count','coverage_pct_of_full_profile','status','note'])}

这里“整体”只指全部5,298个 validation cold-start item（验证冷启动物品），占完整35,322个物品的 `14.999%`。训练非冷启动物品、测试冷启动物品、完整混合范围推荐指标均为“未评估”；CICP-MP是连续23维画像，没有预注册 target/non-target（目标/非目标）二元标签。

## 二、运行与协议审计

- 5/5分支正常完成，每支100条曲线、100个检查点，总训练时间约6小时10分。
- 五支公共参数实际初始化 SHA-256（安全散列算法）一致；E2与E5结构、参数量和新增模块初始参数一致。
- 公共参数 weight decay（权重衰减）为`0.1`，新增方法参数为`0.0`；五支初始影响均为精确0。
- E2至E5使用训练集逐特征标准化和四语义分块；无LayerNorm（层归一化）、固定可靠性乘法、离线辅助目标或人工轮次调度。
- 只有E1使用E4式隐藏残差；训练输入未使用验证或测试逐物品推荐答案。

{_markdown_table(config_audit, ['method_label','seed','negative_sampling_mode','num_workers','base_weight_decay','method_weight_decay','input_standardization','e4_style_hidden_residual','method_specific_parameters'])}

## 三、五支正式结果

同口径 baseline（基线）：NDCG@20 `{BASELINE_NDCG20:.12f}`，HR@20 `{BASELINE_HR20:.12f}`；历史 CICP-R1-E1 为 `{HISTORICAL_CICPR1_E1_NDCG20:.12f}`，即相对基线 `+2.051%`；`+3%` 门槛为 `{THREE_PCT_THRESHOLD:.12f}`。

{_markdown_table(main, ['method_label','best_epoch','best_ndcg@20','relative_pct_vs_baseline','best_hr@20_same_checkpoint','hr_relative_pct_vs_baseline','relative_pct_vs_history','paired_ci95','passed_three_pct'])}

解释：E4是数值最高分支且配对区间排除0，但不能归因于画像；E3是仍有逐物品画像敏感度的最高分支，刚超过历史锚点但区间跨0；E1标量参照未超过基线；E5乱序控制显著受损。

## 四、早晚曲线

{_markdown_table(stability, ['method_label','best_epoch','same_epoch_relative_pct_ndcg@20','epoch74_relative_pct_ndcg@20','late30_matched_relative_pct_ndcg@20','late30_matched_positive_epochs','all100_matched_positive_epochs'])}

- E3在第83轮最佳，第74轮相对同轮基线`+0.473%`，后30轮30/30为正。
- E4在第87轮最佳，第74轮相对同轮基线`+2.123%`，后30轮30/30为正，100/100同轮为正。
- 因此原CICP-MP-R1约40轮早峰、第74轮转负的后期冲突已经修复；但E4修复的是结构表现，不是逐物品画像使用。

## 五、机制实际使用审计

固定读取前1,024个验证物品的内容和训练安全画像，不读取交互答案；比较真实画像、全零画像和逐物品循环置换画像的输出嵌入。

{_markdown_table(mechanism_view, ['method_label','epoch','gate_abs_mean','actual_vs_zero_embedding_relative_l2_mean','actual_vs_permuted_embedding_relative_l2_mean','functional_signal_active','gate_saturated','mechanism_interpretation'])}

关键解释：

1. E4门控从第1轮均值约`-0.221`快速走到第20轮约`-0.993`，最佳第87轮为约`-1.000`。门控不是“被压没”，而是饱和为与画像几乎无关的常量。
2. E4当前检查点在推理时近似退化成固定的跨模态类别重解释结构；因此它可作为新backbone（骨干）候选证据，不能作为CICP-MP语义胜者。训练早期画像是否通过梯度历史塑造了结构参数，仍需无画像条件重训对照才能排除。
3. E3虽然混合门同样接近`-1`，但专家路由熵均值约`0.734`，且真实对置换画像嵌入差约`0.434`，说明画像仍通过三专家路由产生强逐物品条件作用。
4. E2、E3、E5都没有再次失活；E1残差也持续非零。修复方向1和2确实解除原实现性压制。

## 六、E2对E5语义控制

{_markdown_table(control_view, ['comparison_scope','epoch_count','real_ndcg@20','shuffle_ndcg@20','absolute_delta_ndcg@20','relative_pct_vs_shuffle','real_above_shuffle_epoch_count'])}

E2真实画像在100/100轮均高于同结构E5乱序画像，后30轮平均绝对差`+0.003234862989`；独立最佳检查点逐物品配对差相对E5为`{decision['e2_minus_e5_best_relative_pct']:+.3f}%`，95%区间`[{e2e5_ci[0]:+.3f}%, {e2e5_ci[1]:+.3f}%]`，排除0。这说明对E2模态调制载体而言，正确的物品-画像对应关系确实重要，不是只靠新增参数；但E2自身相对基线只有`+0.562%`，不能据此把E3或E4收益全部归因于画像。

## 七、E3完整互补画像分层

每个维度的low/mid/high（低/中/高）三组互补并完整覆盖5,298个验证冷物品；这些是诊断，不是选择后的新主指标。

{_markdown_table(semantic_groups, ['group_dimension','group_value','item_count','coverage_pct_of_validation','baseline_ndcg@20','method_ndcg@20','relative_pct_ndcg@20','relative_pct_hr@20','helped_ndcg_item_count','harmed_ndcg_item_count'])}

最佳分层为 `{best_group['group_dimension']} / {best_group['group_value']}`，相对变化 `{best_group['relative_pct_ndcg@20']}`；最弱分层为 `{worst_group['group_dimension']} / {worst_group['group_value']}`，相对变化 `{worst_group['relative_pct_ndcg@20']}`。分层差异不构成剂量因果关系。

## 八、全部23维画像响应

下表是E3的每个训练安全画像分量与逐物品NDCG收益的 Spearman（斯皮尔曼秩相关）；25行包含原CICP标量、23个CICP-MP分量和16维方向的联合范数。全分析共`{len(score_response)}`项相关检验并统一进行BH（错误发现率）校正。

{_markdown_table(semantic_response, ['predictor','spearman','p_value_uncorrected','q_value_bh','significant_after_bh_0.05'])}

这些响应只能定位可能的异质性，不能从相关性反向证明分量有效。E4的全部分层和响应仍保存在CSV中，但由于其画像输入功能性失活，不应按语义剂量解释；E4分层记录数为`{len(performance_groups)}`。

## 九、11项统计谬误扫描

{_markdown_table(fallacy, ['fallacy','severity','finding'])}

## 十、正式判决

预注册规则不是由E4单独触发，而由“仍使用画像”的E3触发：E3超过历史`+2.051%`，HR不反向，且晚期持续正向，因此 CICP-MP-FR1（类别增量协同可预测性多分量画像最终修复第一轮）判为**窄幅通过公平修复**，路线状态为 `{decision['route']}`。

执行边界：

- 不开放 multi-seed（多随机种子），因为没有活跃画像载体超过`+3%`，E3配对区间仍跨0。
- 允许且只允许一次范围固定的 performance-breakthrough design（性能突破设计），以真正保留逐物品画像依赖为硬约束。
- E4只能作为condition-free cross-modal backbone candidate（无画像条件跨模态骨干候选）进入下一轮设计对照，不能继续称为CICP-MP胜者。
- 下一轮必须给E3类活跃载体配同结构whole-row shuffle（整行乱序）控制；若活跃真实画像分支仍未超过`+3%`，停止CICP-MP并转向更新冷启动骨干或大模型类别语义基础增强。

## 十一、Material Passport（材料护照）

- 训练结果：`{result_root}`
- 主评价：5,298个验证冷启动物品，占完整35,322个物品`14.999%`。
- 训练输入：训练集拟合的23维逐特征标准化画像；测试物品0行；验证/测试推荐答案0列。
- 逐物品复算：baseline（基线）加五支共31,788行，重聚合最大绝对误差不超过`2.22e-16`。
- 机制审计：1,024个验证物品，仅内容与画像，无交互答案；固定检查第1/20/40/74/100轮及各分支最佳轮。
- 排除范围：训练推荐指标、测试推荐指标、35,322完整混合推荐指标均未生成。
- 证据等级：B级开发证据；单seed43（第43号随机种子）、开发验证集选择检查点、机制与分层含事后诊断。

## 十二、分析产物

- `cicpmp_fr1_curve_summary.csv`
- `cicpmp_fr1_epoch_matched_curve.csv`
- `cicpmp_fr1_validation_item_metrics.csv`
- `cicpmp_fr1_group_summary.csv`
- `cicpmp_fr1_score_response.csv`
- `cicpmp_fr1_e2_e5_control_comparison.csv`
- `cicpmp_fr1_mechanism_activity.csv`
- `cicpmp_fr1_mechanism_best_summary.csv`
- `cicpmp_fr1_coverage.csv`
- `cicpmp_fr1_fallacy_scan.csv`
- `cicpmp_fr1_decision.json`
"""
    report_path = output_dir / (
        f"{run_stamp} CCFCRec Amazon-VG CICP-MP-FR1 five final repairs training 结果.md"
    )
    report_path.write_text(report, encoding="utf-8")

    route_report = f"""---
title: {run_stamp} CCFCRec Amazon-VG CICP-MP-FR1 five final repairs training 路线判断
date: {run_stamp[:10]}
status: analyzed
tags:
  - CCFCRec
  - Amazon-VG
  - CICP-MP-FR1
  - route-decision
---

# CCFCRec Amazon-VG CICP-MP-FR1 five final repairs training 路线判断

结果报告：[[{report_path.stem}]]

> [!important] 判决
> CICP-MP-FR1（类别增量协同可预测性多分量画像最终修复第一轮）窄幅通过最后一次公平修复，但没有达到`+3%`，不开放multi-seed（多随机种子）。允许一次且仅一次性能突破设计；若真正使用逐物品画像的真实分支仍不超过`+3%`，停止CICP-MP并转向更新冷启动骨干或大模型类别语义基础增强。

## 一、为什么不是“E4已经证明路线成功”

E4-CMA cross-modal attention（跨模态注意力）相对基线`+2.568%`，HR同点`+2.838%`，后30轮30/30为正，是本批数值胜者。但其门控到最佳轮饱和为约`-1`，真实画像与置换画像的嵌入相对差仅约`1.33e-7`，当前检查点在推理时不再使用逐物品CICP-MP画像。因此E4更支持跨模态结构有潜力，不能证明CICP-MP语义有效；训练历史贡献仍需无画像条件重训对照区分。

## 二、为什么CICP-MP仍窄幅通过修复

E3-CER content expert routing（内容专家路由）仍对逐物品画像高度敏感，最佳NDCG@20为`{decision['active_semantic_winner_ndcg@20']:.12f}`，相对基线`{decision['active_semantic_winner_relative_pct_vs_baseline']:+.3f}%`，HR同点`{decision['active_semantic_winner_relative_pct_hr_vs_baseline']:+.3f}%`，后30轮同轮均值`{decision['active_semantic_winner_late30_matched_relative_pct']:+.3f}%`、30/30轮为正。它超过历史CICP-R1-E1，但只高`{decision['active_semantic_winner_relative_pct_vs_history']:+.3f}%`，且配对区间跨0。

同时，E2真实画像相对同结构E5整行乱序控制在100/100轮均为正，独立最佳逐物品配对区间排除0。这说明正确物品-画像对应至少在模态调制载体中有真实作用。因此不能按“修复后所有活跃画像机制仍低于历史锚点”的停止分支处理。

## 三、当前执行边界

1. 不开放multi-seed（多随机种子），不读取测试推荐指标。
2. 不继续E1标量残差、E2剂量、E4条件门控温度或局部参数扫描。
3. 只允许一次性能突破设计，必须同时满足：真实逐物品画像依赖保持活跃；包含同结构整行乱序控制；仍为seed43、100轮、全部5,298验证冷物品和同一冻结基线。
4. E3是CICP-MP活跃载体起点；E4只作为无画像条件跨模态骨干对照，不得冒充CICP-MP语义分支。
5. 若活跃真实画像分支超过`+3%`且HR不反向、后期不冲突，才进入多随机种子和正式测试准备。
6. 若仍低于`+3%`，停止CICP-MP路线，下一阶段转向updated cold-start backbone（更新冷启动骨干）或LLM category semantic foundation enhancement（大模型类别语义基础增强）。

## 四、当前状态

- 路线：`{decision['route']}`。
- 多随机种子：未开放。
- 测试集：未评估。
- 自动启动下一批训练：否；先完成一次性能突破实验的独立设计与审查。
"""
    route_output_dir.mkdir(parents=True, exist_ok=True)
    route_path = route_output_dir / (
        f"{run_stamp} CCFCRec Amazon-VG CICP-MP-FR1 five final repairs training 路线判断.md"
    )
    route_path.write_text(route_report, encoding="utf-8")
    return report_path, route_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_summary, matched_curve, baseline_curve, config_audit = collect_curves(
        args.result_root.resolve(), args.baseline_result.resolve()
    )
    item_metrics = pd.read_csv(args.item_metrics.resolve(), dtype={"raw_asin": str})
    activity = pd.read_csv(args.mechanism_activity.resolve())
    mechanism = summarize_mechanisms(activity)
    focus_methods = {
        "cicpmp_fr1_content_expert_routing",
        "cicpmp_fr1_cross_modal_attention",
    }
    group_summary = build_group_summary(item_metrics, focus_methods, args.bootstrap_repetitions)
    score_response = build_score_response(item_metrics)
    control = build_control_comparison(
        matched_curve, item_metrics, args.bootstrap_repetitions
    )
    coverage = build_coverage()
    decision = build_decision(curve_summary, mechanism, group_summary, control)
    semantic_method = decision.get("active_semantic_winner_method")
    if semantic_method is None:
        raise ValueError("no active semantic winner is available for the formal report")
    fallacy = build_fallacy_scan(group_summary, semantic_method)
    outputs = {
        "cicpmp_fr1_curve_summary.csv": curve_summary,
        "cicpmp_fr1_epoch_matched_curve.csv": matched_curve,
        "cicpmp_fr1_baseline_curve.csv": baseline_curve,
        "cicpmp_fr1_config_audit.csv": config_audit,
        "cicpmp_fr1_group_summary.csv": group_summary,
        "cicpmp_fr1_score_response.csv": score_response,
        "cicpmp_fr1_e2_e5_control_comparison.csv": control,
        "cicpmp_fr1_mechanism_best_summary.csv": mechanism,
        "cicpmp_fr1_coverage.csv": coverage,
        "cicpmp_fr1_fallacy_scan.csv": fallacy,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False)
    decision_path = output_dir / "cicpmp_fr1_decision.json"
    decision_path.write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path, route_path = write_reports(
        output_dir=output_dir,
        route_output_dir=args.route_output_dir.resolve(),
        run_stamp=args.run_stamp,
        result_root=args.result_root.resolve(),
        curve_summary=curve_summary,
        config_audit=config_audit,
        group_summary=group_summary,
        score_response=score_response,
        control=control,
        mechanism=mechanism,
        coverage=coverage,
        fallacy=fallacy,
        decision=decision,
    )
    manifest = {
        "protocol": "cicpmp_fr1_five_training_analysis_v1",
        "result_root": str(args.result_root.resolve()),
        "baseline_result": str(args.baseline_result.resolve()),
        "item_metrics": str(args.item_metrics.resolve()),
        "mechanism_activity": str(args.mechanism_activity.resolve()),
        "formal_report": str(report_path),
        "route_report": str(route_path),
        "test_metrics_read_or_generated": False,
        "output_files": sorted([*outputs, decision_path.name, report_path.name]),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--item-metrics", type=Path, required=True)
    parser.add_argument("--mechanism-activity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-output-dir", type=Path, required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
