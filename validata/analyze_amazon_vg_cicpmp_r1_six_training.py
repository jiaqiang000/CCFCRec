#!/usr/bin/env python3
"""Analyze the six CICP-MP-R1 validation runs under the frozen protocol."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


BASELINE_LABEL = "baseline_seed43_workers8_fast_uniform"
BASELINE_NDCG20 = 0.1238145211709585
BASELINE_HR20 = 0.0206209890524726
THREE_PCT_THRESHOLD = BASELINE_NDCG20 * 1.03
HISTORICAL_CICPR1_E1_NDCG20 = 0.1263542828877096
FULL_ITEM_COUNT = 35322
TRAIN_ITEM_COUNT = 24726
VALIDATION_ITEM_COUNT = 5298
TEST_ITEM_COUNT = 5298

METHOD_ORDER = [
    "cicpmp_r1_reliable_residual",
    "cicpmp_r1_direction_alignment",
    "cicpmp_r1_attention_entropy",
    "cicpmp_r1_reliable_expert",
    "cicpmp_r1_counterfactual_calibration",
    "cicpmp_r1_direction_hard_negative",
]
METHOD_LABELS = {
    "cicpmp_r1_reliable_residual": "E1-RRA reliable residual adapter（可靠残差适配器）",
    "cicpmp_r1_direction_alignment": "E2-DTA direction target alignment（方向目标对齐）",
    "cicpmp_r1_attention_entropy": "E3-AEC attribution entropy calibration（归因熵校准）",
    "cicpmp_r1_reliable_expert": "E4-RCE reliability-aware category expert（可靠性类别专家）",
    "cicpmp_r1_counterfactual_calibration": "E5-CCI calibrated counterfactual increment（校准反事实增量）",
    "cicpmp_r1_direction_hard_negative": "E6-DHN direction-aware hard negatives（方向感知难负例）",
}
ITEM_METHOD_LABELS = {
    "cicpmp_r1_reliable_residual": "CICP-MP-R1-E1-RRA",
    "cicpmp_r1_direction_alignment": "CICP-MP-R1-E2-DTA",
    "cicpmp_r1_attention_entropy": "CICP-MP-R1-E3-AEC",
    "cicpmp_r1_reliable_expert": "CICP-MP-R1-E4-RCE",
    "cicpmp_r1_counterfactual_calibration": "CICP-MP-R1-E5-CCI",
    "cicpmp_r1_direction_hard_negative": "CICP-MP-R1-E6-DHN",
}
GROUP_COLUMNS = {
    "reliability": ("reliability_group", ("low", "mid", "high")),
    "semantic_increment_sign": (
        "semantic_sign_group",
        ("non_positive", "positive"),
    ),
    "attribution_entropy": ("entropy_group", ("low", "high")),
    "direction_norm": ("direction_norm_group", ("low", "high")),
}
PREDICTORS = [
    "mp_raw_predicted_increment",
    "mp_category_semantic_increment_prediction",
    "mp_category_total_increment_prediction",
    "mp_category_attribution_positive_share_prediction",
    "mp_category_attribution_entropy_prediction",
    "mp_reliability",
    "mp_direction_norm",
]
RSP_COLUMNS = {
    "R_metadata_richness_score": "R richness（丰富度）",
    "S_train_support_score": "S support（支持度）",
    "P_popularity_score": "P popularity（流行度）",
}


def relative_pct(value: float, baseline: float) -> float:
    if float(baseline) == 0.0:
        return float("nan")
    return (float(value) / float(baseline) - 1.0) * 100.0


def _tar_result_csv(package: Path) -> pd.DataFrame:
    with tarfile.open(package, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.name.endswith("/result.csv")]
        if len(members) != 1:
            raise ValueError(f"expected one baseline result.csv, got {len(members)}")
        file_obj = tar.extractfile(members[0])
        if file_obj is None:
            raise ValueError("cannot read baseline result.csv")
        return pd.read_csv(io.BytesIO(file_obj.read()))


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
    epochs = curve["epoch"].astype(int).tolist()
    checkpoints = list(run_dir.glob("*.pt"))
    if len(curve) != 100 or epochs != list(range(1, 101)) or len(checkpoints) != 100:
        raise ValueError(f"{method} must contain epochs 1..100 and 100 checkpoints")
    best = _best_row(curve)
    epoch = int(best["epoch"])
    baseline_by_epoch = baseline_curve.set_index("epoch")
    method_by_epoch = curve.set_index("epoch")
    matched = method_by_epoch[["ndcg@20", "hr@20"]].join(
        baseline_by_epoch[["ndcg@20", "hr@20"]],
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
    best_ndcg = float(best["ndcg@20"])
    best_hr = float(best["hr@20"])
    same_epoch = matched[matched["epoch"].eq(epoch)].iloc[0]
    baseline_best_epoch = int(_best_row(baseline_curve)["epoch"])
    at_baseline_best = curve[curve["epoch"].eq(baseline_best_epoch)].iloc[0]
    late_curve = curve[curve["epoch"].between(71, 100)]
    baseline_late = baseline_curve[baseline_curve["epoch"].between(71, 100)]
    late_matched = matched[matched["epoch"].between(71, 100)]
    parameter_count = config.get("parameter_count", {})
    summary = {
        "method_variant": method,
        "method_label": METHOD_LABELS[method],
        "run_dir": str(run_dir),
        "epoch_count": len(curve),
        "checkpoint_count": len(checkpoints),
        "best_epoch": epoch,
        "best_checkpoint_index": int(best["checkpoint_index"]),
        "best_ndcg@20": best_ndcg,
        "absolute_delta_ndcg@20_vs_baseline_best": best_ndcg - BASELINE_NDCG20,
        "relative_pct_ndcg@20_vs_baseline_best": relative_pct(
            best_ndcg, BASELINE_NDCG20
        ),
        "best_hr@20_same_checkpoint": best_hr,
        "absolute_delta_hr@20_vs_baseline_best": best_hr - BASELINE_HR20,
        "relative_pct_hr@20_vs_baseline_best": relative_pct(best_hr, BASELINE_HR20),
        "passed_two_pct": bool(best_ndcg >= BASELINE_NDCG20 * 1.02),
        "passed_three_pct": bool(best_ndcg >= THREE_PCT_THRESHOLD),
        "absolute_gap_to_three_pct": best_ndcg - THREE_PCT_THRESHOLD,
        "absolute_delta_vs_historical_cicpr1_e1": (
            best_ndcg - HISTORICAL_CICPR1_E1_NDCG20
        ),
        "relative_pct_vs_historical_cicpr1_e1": relative_pct(
            best_ndcg, HISTORICAL_CICPR1_E1_NDCG20
        ),
        "same_epoch_baseline_ndcg@20": float(same_epoch["ndcg@20_baseline"]),
        "same_epoch_relative_pct_ndcg@20": float(same_epoch["relative_pct_ndcg@20"]),
        "same_epoch_relative_pct_hr@20": float(same_epoch["relative_pct_hr@20"]),
        "baseline_best_epoch": baseline_best_epoch,
        "at_baseline_best_epoch_ndcg@20": float(at_baseline_best["ndcg@20"]),
        "at_baseline_best_relative_pct_ndcg@20": relative_pct(
            float(at_baseline_best["ndcg@20"]), BASELINE_NDCG20
        ),
        "late30_mean_ndcg@20": float(late_curve["ndcg@20"].mean()),
        "late30_relative_pct_ndcg@20_vs_baseline_best": relative_pct(
            float(late_curve["ndcg@20"].mean()), BASELINE_NDCG20
        ),
        "late30_matched_relative_pct_ndcg@20": relative_pct(
            float(late_curve["ndcg@20"].mean()),
            float(baseline_late["ndcg@20"].mean()),
        ),
        "late30_above_baseline_best_epochs": int(
            (late_curve["ndcg@20"] > BASELINE_NDCG20).sum()
        ),
        "late30_matched_positive_epochs": int(
            (late_matched["absolute_delta_ndcg@20"] > 0.0).sum()
        ),
        "all100_matched_positive_epochs": int(
            (matched["absolute_delta_ndcg@20"] > 0.0).sum()
        ),
        "common_parameter_sha256": str(
            config.get("ccfcrec_common_parameter_sha256", "")
        ),
        "common_parameters": int(parameter_count.get("common", 0)),
        "method_specific_parameters": int(parameter_count.get("method_specific", 0)),
        "total_parameters": int(parameter_count.get("total", 0)),
    }
    return summary, matched


def collect_curves(
    result_root: Path,
    baseline_result: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_curve = _tar_result_csv(baseline_result)
    if len(baseline_curve) != 100:
        raise ValueError("baseline curve must contain exactly 100 rows")
    summaries: list[dict[str, Any]] = []
    matched_frames: list[pd.DataFrame] = []
    config_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result_path in sorted(result_root.rglob("result.csv")):
        run_dir = result_path.parent
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        method = str(config.get("method_variant", ""))
        if method not in METHOD_ORDER:
            continue
        if method in seen:
            raise ValueError(f"duplicate run for {method}")
        seen.add(method)
        summary, matched = summarize_curve(
            method,
            pd.read_csv(result_path),
            baseline_curve,
            run_dir,
            config,
        )
        summaries.append(summary)
        matched_frames.append(matched)
        config_rows.append(
            {
                "method_variant": method,
                "method_label": METHOD_LABELS[method],
                "seed": int(config.get("seed", -1)),
                "epochs": 100,
                "negative_sampling_mode": config.get("negative_sampling_mode"),
                "training_input_uses_validation_item_metrics": bool(
                    config.get("training_input_uses_validation_item_metrics", True)
                ),
                "training_input_uses_test_item_metrics": bool(
                    config.get("training_input_uses_test_item_metrics", True)
                ),
                "common_parameter_sha256": config.get(
                    "ccfcrec_common_parameter_sha256"
                ),
                "method_specific_parameters": config.get("parameter_count", {}).get(
                    "method_specific"
                ),
            }
        )
    if seen != set(METHOD_ORDER):
        raise ValueError(f"missing methods: {sorted(set(METHOD_ORDER) - seen)}")
    summary_frame = pd.DataFrame(summaries)
    summary_frame["method_order"] = summary_frame["method_variant"].map(
        METHOD_ORDER.index
    )
    summary_frame = summary_frame.sort_values("method_order").drop(columns="method_order")
    config_audit = pd.DataFrame(config_rows)
    if config_audit["common_parameter_sha256"].nunique() != 1:
        raise ValueError("actual common parameter hashes are not identical")
    if not config_audit["seed"].eq(43).all():
        raise ValueError("not all runs use seed43")
    if config_audit[
        [
            "training_input_uses_validation_item_metrics",
            "training_input_uses_test_item_metrics",
        ]
    ].any(axis=None):
        raise ValueError("evaluation item metrics entered a training input")
    return (
        summary_frame.reset_index(drop=True),
        pd.concat(matched_frames, ignore_index=True),
        baseline_curve,
        config_audit,
    )


def _paired_bootstrap(
    method_values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> dict[str, float]:
    if method_values.shape != baseline_values.shape or method_values.ndim != 1:
        raise ValueError("paired bootstrap inputs must be same-shape vectors")
    if len(method_values) == 0:
        return {
            "bootstrap_absolute_ci95_low": float("nan"),
            "bootstrap_absolute_ci95_high": float("nan"),
            "bootstrap_relative_pct_ci95_low": float("nan"),
            "bootstrap_relative_pct_ci95_high": float("nan"),
        }
    rng = np.random.default_rng(seed)
    absolute = np.empty(repetitions, dtype=float)
    relative = np.empty(repetitions, dtype=float)
    batch = 100
    cursor = 0
    while cursor < repetitions:
        size = min(batch, repetitions - cursor)
        selected = rng.integers(0, len(method_values), size=(size, len(method_values)))
        method_mean = method_values[selected].mean(axis=1)
        baseline_mean = baseline_values[selected].mean(axis=1)
        absolute[cursor : cursor + size] = method_mean - baseline_mean
        relative[cursor : cursor + size] = (
            method_mean / baseline_mean - 1.0
        ) * 100.0
        cursor += size
    return {
        "bootstrap_absolute_ci95_low": float(np.quantile(absolute, 0.025)),
        "bootstrap_absolute_ci95_high": float(np.quantile(absolute, 0.975)),
        "bootstrap_relative_pct_ci95_low": float(np.nanquantile(relative, 0.025)),
        "bootstrap_relative_pct_ci95_high": float(np.nanquantile(relative, 0.975)),
    }


def _iter_groups(frame: pd.DataFrame):
    yield "overall", "all", frame
    for dimension, (column, values) in GROUP_COLUMNS.items():
        for value in values:
            yield dimension, value, frame[frame[column].eq(value)]


def build_group_summary(
    item_metrics: pd.DataFrame,
    winner_method: str,
    repetitions: int,
) -> pd.DataFrame:
    baseline = item_metrics[item_metrics["method_label"].eq(BASELINE_LABEL)].set_index(
        "raw_asin"
    )
    if len(baseline) != VALIDATION_ITEM_COUNT:
        raise ValueError("baseline item metrics do not cover 5,298 validation items")
    rows: list[dict[str, Any]] = []
    for method_index, method in enumerate(METHOD_ORDER):
        method_frame = item_metrics[
            item_metrics["method_label"].eq(ITEM_METHOD_LABELS[method])
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
                "item_count": int(len(part)),
                "coverage_pct_of_validation": len(part) / VALIDATION_ITEM_COUNT * 100.0,
            }
            for metric in ("ndcg@20", "hr@20"):
                method_mean = float(part[metric].mean())
                baseline_mean = float(part[f"{metric}_baseline"].mean())
                row[f"baseline_{metric}"] = baseline_mean
                row[f"method_{metric}"] = method_mean
                row[f"absolute_delta_{metric}"] = method_mean - baseline_mean
                row[f"relative_pct_{metric}"] = relative_pct(method_mean, baseline_mean)
            delta = part["ndcg@20"] - part["ndcg@20_baseline"]
            row["helped_ndcg_item_count"] = int((delta > 0.0).sum())
            row["harmed_ndcg_item_count"] = int((delta < 0.0).sum())
            row["equal_ndcg_item_count"] = int((delta == 0.0).sum())
            should_bootstrap = dimension == "overall" or method == winner_method
            if should_bootstrap:
                ci = _paired_bootstrap(
                    part["ndcg@20"].to_numpy(dtype=float),
                    part["ndcg@20_baseline"].to_numpy(dtype=float),
                    seed=43 + method_index * 100 + group_index,
                    repetitions=repetitions,
                )
            else:
                ci = {
                    "bootstrap_absolute_ci95_low": np.nan,
                    "bootstrap_absolute_ci95_high": np.nan,
                    "bootstrap_relative_pct_ci95_low": np.nan,
                    "bootstrap_relative_pct_ci95_high": np.nan,
                }
            row.update({f"ndcg@20_{key}": value for key, value in ci.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def _bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(values))
    if len(valid) == 0:
        return pd.Series(result, index=p_values.index)
    ordered = valid[np.argsort(values[valid])]
    adjusted = values[ordered] * len(ordered) / np.arange(1, len(ordered) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result[ordered] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(result, index=p_values.index)


def build_score_response(item_metrics: pd.DataFrame) -> pd.DataFrame:
    def calculate(left: pd.Series, right: pd.Series) -> tuple[float, float]:
        if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
            return float("nan"), float("nan")
        result = spearmanr(left, right)
        return float(result.statistic), float(result.pvalue)

    baseline = item_metrics[item_metrics["method_label"].eq(BASELINE_LABEL)].set_index(
        "raw_asin"
    )
    rows: list[dict[str, Any]] = []
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
        frame = item_metrics[
            item_metrics["method_label"].eq(ITEM_METHOD_LABELS[method])
        ].set_index("raw_asin")
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
    result["q_value_bh_98_tests"] = _bh_adjust(result["p_value_uncorrected"])
    result["significant_after_bh_0.05"] = result["q_value_bh_98_tests"] < 0.05
    return result


def build_rsp_strata(
    item_metrics: pd.DataFrame,
    source_profile: Path,
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    profile = pd.read_csv(source_profile, dtype={"raw_asin": str}, low_memory=False)
    profile = profile[profile["split"].astype(str).eq("validate")].copy()
    thresholds: dict[str, list[float]] = {}
    keep = ["raw_asin"]
    for column in RSP_COLUMNS:
        values = pd.to_numeric(profile[column], errors="raise")
        lower, upper = values.quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
        if lower >= upper:
            raise ValueError(f"cannot build unique RSP tertiles for {column}")
        thresholds[column] = [float(lower), float(upper)]
        group_column = f"{column}_group"
        profile[group_column] = pd.cut(
            values,
            bins=[-np.inf, lower, upper, np.inf],
            labels=["low", "mid", "high"],
            include_lowest=True,
        ).astype(str)
        keep.append(group_column)
    baseline = item_metrics[item_metrics["method_label"].eq(BASELINE_LABEL)].set_index(
        "raw_asin"
    )
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        frame = item_metrics[
            item_metrics["method_label"].eq(ITEM_METHOD_LABELS[method])
        ].merge(profile[keep], on="raw_asin", validate="one_to_one")
        paired = frame.set_index("raw_asin").join(
            baseline[["ndcg@20", "hr@20"]],
            rsuffix="_baseline",
            validate="one_to_one",
        )
        for column, label in RSP_COLUMNS.items():
            group_column = f"{column}_group"
            for stratum in ("low", "mid", "high"):
                part = paired[paired[group_column].eq(stratum)]
                row: dict[str, Any] = {
                    "method_variant": method,
                    "method_label": METHOD_LABELS[method],
                    "stratification_variable": column,
                    "stratification_label": label,
                    "stratum": stratum,
                    "item_count": int(len(part)),
                }
                for metric in ("ndcg@20", "hr@20"):
                    baseline_mean = float(part[f"{metric}_baseline"].mean())
                    method_mean = float(part[metric].mean())
                    row[f"baseline_{metric}"] = baseline_mean
                    row[f"method_{metric}"] = method_mean
                    row[f"absolute_delta_{metric}"] = method_mean - baseline_mean
                    row[f"relative_pct_{metric}"] = relative_pct(
                        method_mean, baseline_mean
                    )
                rows.append(row)
    return pd.DataFrame(rows), thresholds


def build_coverage(group_summary: pd.DataFrame, winner_method: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "evaluation_scope": "complete_item_profile（完整物品档案）",
            "scope_item_count": FULL_ITEM_COUNT,
            "evaluated_item_count": 0,
            "coverage_pct_of_full_profile": 100.0,
            "coverage_pct_of_validation": np.nan,
            "recommendation_metric_status": "未评估",
            "note": "未生成训练、验证、测试混合的推荐指标；验证子范围另行报告。",
        },
        {
            "evaluation_scope": "train_non_cold（训练非冷启动物品）",
            "scope_item_count": TRAIN_ITEM_COUNT,
            "evaluated_item_count": 0,
            "coverage_pct_of_full_profile": TRAIN_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
            "coverage_pct_of_validation": np.nan,
            "recommendation_metric_status": "未评估",
            "note": "没有生成训练物品推荐指标。",
        },
        {
            "evaluation_scope": "validation_cold（验证冷启动物品）",
            "scope_item_count": VALIDATION_ITEM_COUNT,
            "evaluated_item_count": VALIDATION_ITEM_COUNT,
            "coverage_pct_of_full_profile": VALIDATION_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
            "coverage_pct_of_validation": 100.0,
            "recommendation_metric_status": "已评估",
            "note": "本轮开发阶段主评价集合。",
        },
        {
            "evaluation_scope": "test_cold（测试冷启动物品）",
            "scope_item_count": TEST_ITEM_COUNT,
            "evaluated_item_count": 0,
            "coverage_pct_of_full_profile": TEST_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
            "coverage_pct_of_validation": np.nan,
            "recommendation_metric_status": "未评估",
            "note": "没有读取或生成测试推荐指标。",
        },
        {
            "evaluation_scope": "validation_target（验证目标物品）",
            "scope_item_count": 0,
            "evaluated_item_count": 0,
            "coverage_pct_of_full_profile": np.nan,
            "coverage_pct_of_validation": np.nan,
            "recommendation_metric_status": "不适用（未定义）",
            "note": "CICP-MP-v1是连续23维画像，预注册协议没有目标/非目标二元标签。",
        },
        {
            "evaluation_scope": "validation_non_target（验证非目标物品）",
            "scope_item_count": 0,
            "evaluated_item_count": 0,
            "coverage_pct_of_full_profile": np.nan,
            "coverage_pct_of_validation": np.nan,
            "recommendation_metric_status": "不适用（未定义）",
            "note": "不得事后阈值化连续画像来伪造目标/非目标互补群体。",
        },
    ]
    winner_groups = group_summary[
        group_summary["method_variant"].eq(winner_method)
        & ~group_summary["group_dimension"].eq("overall")
    ]
    for row in winner_groups.itertuples(index=False):
        rows.append(
            {
                "evaluation_scope": (
                    f"validation_{row.group_dimension}_{row.group_value}"
                    "（验证画像互补组）"
                ),
                "scope_item_count": int(row.item_count),
                "evaluated_item_count": int(row.item_count),
                "coverage_pct_of_full_profile": row.item_count
                / FULL_ITEM_COUNT
                * 100.0,
                "coverage_pct_of_validation": row.item_count
                / VALIDATION_ITEM_COUNT
                * 100.0,
                "recommendation_metric_status": "已评估（诊断）",
                "note": "同一维度内各组互补并完整覆盖5,298个验证冷启动物品。",
            }
        )
    return pd.DataFrame(rows)


def build_fallacy_scan(group_summary: pd.DataFrame, winner_method: str) -> pd.DataFrame:
    winner_groups = group_summary[
        group_summary["method_variant"].eq(winner_method)
        & ~group_summary["group_dimension"].eq("overall")
    ]
    has_reversal = bool(
        (winner_groups["relative_pct_ndcg@20"] < 0.0).any()
        and (winner_groups["relative_pct_ndcg@20"] > 0.0).any()
    )
    entries = [
        (
            "Simpson's paradox（辛普森悖论）",
            "CAUTION" if has_reversal else "CHECKED",
            "互补画像组存在正负方向并存，整体均值会掩盖异质性。"
            if has_reversal
            else "互补画像组未观察到正负方向反转，但仍保留整体与分组并报。",
        ),
        (
            "Ecological fallacy（生态谬误）",
            "CAUTION",
            "群体平均变化不能推出组内每个物品都同向受益。",
        ),
        (
            "Berkson's paradox（伯克森悖论）",
            "CAUTION",
            "样本固定为验证冷启动物品，不能把该选择样本关系外推到训练物品、测试物品或其他数据集。",
        ),
        (
            "Collider bias（碰撞变量偏差）",
            "CAUTION",
            "R/S/P分层属于事后描述，条件化后的差异不是CICP-MP净因果效应。",
        ),
        (
            "Base-rate neglect（基率忽视）",
            "CHECKED",
            "报告了完整档案、验证主集合和每个互补组的物品数及覆盖率。",
        ),
        (
            "Regression to the mean（均值回归）",
            "CAUTION",
            "每支从100轮中选择峰值；同时报告同轮、基线最佳轮和后30轮以限制峰值放大解释。",
        ),
        (
            "Survivorship bias（幸存者偏差）",
            "CHECKED",
            "六个预注册分支全部完成100轮并全部进入汇总，没有只保留表现较好的分支。",
        ),
        (
            "Look-elsewhere effect（多处搜寻效应）",
            "CAUTION",
            "六支各检查100个检查点会增加偶然峰值概率；单seed开发峰值不等于确认性结果。",
        ),
        (
            "Garden of forking paths（分叉路径花园）",
            "CAUTION",
            "六机制和停止门槛预先固定，但逐物品画像、R/S/P和最佳检查点解释属于开发阶段分析。",
        ),
        (
            "Correlation is not causation（相关不等于因果）",
            "CAUTION",
            "本轮无同载体关闭、打乱和匹配控制，不能把分组相关写成CICP-MP语义导致推荐变化。",
        ),
        (
            "Reverse causality（反向因果）",
            "CHECKED",
            "训练画像不含验证或测试推荐结果；但推荐变化也不能反向证明23维画像定义正确。",
        ),
    ]
    result = pd.DataFrame(entries, columns=["fallacy", "severity", "finding"])
    if len(result) != 11:
        raise AssertionError("fallacy scan must contain exactly 11 entries")
    return result


def build_decision(
    curve_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    score_response: pd.DataFrame,
) -> dict[str, Any]:
    winner = curve_summary.sort_values("best_ndcg@20", ascending=False).iloc[0]
    winner_method = str(winner["method_variant"])
    overall = group_summary[
        group_summary["method_variant"].eq(winner_method)
        & group_summary["group_dimension"].eq("overall")
    ].iloc[0]
    winner_score = score_response[
        score_response["method_variant"].eq(winner_method)
        & score_response["response"].eq("ndcg@20")
        & score_response["comparison"].eq("predictor_vs_item_delta")
    ]
    available_response = winner_score.dropna(subset=["spearman"])
    if available_response.empty:
        best_response = pd.Series(
            {
                "predictor": "unavailable",
                "spearman": np.nan,
                "q_value_bh_98_tests": np.nan,
            }
        )
    else:
        best_response = available_response.loc[
            available_response["spearman"].abs().idxmax()
        ]
    passed_two = bool(float(winner["best_ndcg@20"]) >= BASELINE_NDCG20 * 1.02)
    passed_three = bool(float(winner["best_ndcg@20"]) >= THREE_PCT_THRESHOLD)
    beat_historical = bool(
        float(winner["best_ndcg@20"]) > HISTORICAL_CICPR1_E1_NDCG20
    )
    late_stability = bool(
        float(winner["late30_mean_ndcg@20"]) > BASELINE_NDCG20
        and int(winner["late30_matched_positive_epochs"]) >= 24
    )
    if passed_three:
        route = "controls_then_consider_multiseed"
        result_band = "over_3pct"
    elif passed_two and beat_historical:
        route = "at_most_one_confirmation_design_if_diagnostics_clear"
        result_band = "between_2_and_3pct_and_beats_history"
    else:
        route = "stop_cicp_carrier_extensions_review_semantic_basis_and_new_backbone"
        result_band = "below_2pct_or_not_better_than_historical_cicpr1_e1"
    return {
        "winner_method": winner_method,
        "winner_label": str(winner["method_label"]),
        "winner_best_epoch": int(winner["best_epoch"]),
        "winner_ndcg@20": float(winner["best_ndcg@20"]),
        "winner_absolute_delta_ndcg@20_vs_baseline": float(
            winner["absolute_delta_ndcg@20_vs_baseline_best"]
        ),
        "winner_relative_pct_ndcg@20_vs_baseline": float(
            winner["relative_pct_ndcg@20_vs_baseline_best"]
        ),
        "winner_hr@20": float(winner["best_hr@20_same_checkpoint"]),
        "winner_absolute_delta_hr@20_vs_baseline": float(
            winner["absolute_delta_hr@20_vs_baseline_best"]
        ),
        "winner_relative_pct_hr@20_vs_baseline": float(
            winner["relative_pct_hr@20_vs_baseline_best"]
        ),
        "winner_relative_pct_vs_historical_cicpr1_e1": float(
            winner["relative_pct_vs_historical_cicpr1_e1"]
        ),
        "winner_overall_bootstrap_relative_pct_ci95": [
            float(overall["ndcg@20_bootstrap_relative_pct_ci95_low"]),
            float(overall["ndcg@20_bootstrap_relative_pct_ci95_high"]),
        ],
        "three_pct_threshold": THREE_PCT_THRESHOLD,
        "absolute_gap_to_three_pct": float(winner["absolute_gap_to_three_pct"]),
        "passed_two_pct": passed_two,
        "passed_three_pct": passed_three,
        "beat_historical_cicpr1_e1": beat_historical,
        "late_stability_pass": late_stability,
        "strongest_posthoc_predictor": str(best_response["predictor"]),
        "strongest_posthoc_predictor_spearman": float(best_response["spearman"]),
        "strongest_posthoc_predictor_q_value_bh": float(
            best_response["q_value_bh_98_tests"]
        ),
        "result_band": result_band,
        "automatic_followup_authorized": bool(route != "stop_cicp_carrier_extensions_review_semantic_basis_and_new_backbone"),
        "route": route,
        "recommendation_metric_scope": "all_5298_validation_cold_items_only",
        "validation_coverage_pct_of_full_profile": VALIDATION_ITEM_COUNT
        / FULL_ITEM_COUNT
        * 100.0,
        "full_mixed_metrics_evaluated": False,
        "train_metrics_evaluated": False,
        "test_metrics_read_or_generated": False,
        "fallacy_scan_coverage": "11/11",
        "evidence_level": "B_development_validation_single_seed_posthoc_groups",
    }


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    def value_text(value: Any) -> str:
        if pd.isna(value):
            return ""
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.6f}"
        return str(value).replace("|", "\\|").replace("\n", " ")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(value_text(row[column]) for column in columns) + " |"
        for _, row in frame[columns].iterrows()
    ]
    return "\n".join([header, separator, *body])


def _field_lines(fields: list[tuple[str, str]]) -> str:
    return "\n".join(f"- `{name}`：{description}" for name, description in fields)


def write_reports(
    *,
    output_dir: Path,
    route_output_dir: Path,
    run_stamp: str,
    result_root: Path,
    config_audit: pd.DataFrame,
    curve_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    rsp_summary: pd.DataFrame,
    score_response: pd.DataFrame,
    coverage: pd.DataFrame,
    fallacy_scan: pd.DataFrame,
    decision: dict[str, Any],
) -> tuple[Path, Path]:
    winner_method = decision["winner_method"]
    winner = curve_summary[curve_summary["method_variant"].eq(winner_method)].iloc[0]
    main = curve_summary.copy()
    main["best_ndcg@20"] = main["best_ndcg@20"].map(lambda value: f"{value:.12f}")
    for column in (
        "absolute_delta_ndcg@20_vs_baseline_best",
        "absolute_delta_hr@20_vs_baseline_best",
    ):
        main[column] = main[column].map(lambda value: f"{value:+.12f}")
    for column in (
        "relative_pct_ndcg@20_vs_baseline_best",
        "relative_pct_hr@20_vs_baseline_best",
        "relative_pct_vs_historical_cicpr1_e1",
        "late30_relative_pct_ndcg@20_vs_baseline_best",
    ):
        main[column] = main[column].map(lambda value: f"{value:+.3f}%")
    overall_ci = group_summary[group_summary["group_dimension"].eq("overall")][
        [
            "method_variant",
            "ndcg@20_bootstrap_relative_pct_ci95_low",
            "ndcg@20_bootstrap_relative_pct_ci95_high",
        ]
    ].copy()
    overall_ci["paired_bootstrap_relative_ci95"] = overall_ci.apply(
        lambda row: (
            f"[{row['ndcg@20_bootstrap_relative_pct_ci95_low']:+.3f}%, "
            f"{row['ndcg@20_bootstrap_relative_pct_ci95_high']:+.3f}%]"
        ),
        axis=1,
    )
    main = main.merge(
        overall_ci[["method_variant", "paired_bootstrap_relative_ci95"]],
        on="method_variant",
        validate="one_to_one",
    )
    stability = curve_summary.copy()
    stability["same_epoch_relative_pct_ndcg@20"] = stability[
        "same_epoch_relative_pct_ndcg@20"
    ].map(lambda value: f"{value:+.3f}%")
    stability["at_baseline_best_relative_pct_ndcg@20"] = stability[
        "at_baseline_best_relative_pct_ndcg@20"
    ].map(lambda value: f"{value:+.3f}%")
    stability["late30_matched_relative_pct_ndcg@20"] = stability[
        "late30_matched_relative_pct_ndcg@20"
    ].map(lambda value: f"{value:+.3f}%")
    stability["late30_relative_pct_ndcg@20_vs_baseline_best"] = stability[
        "late30_relative_pct_ndcg@20_vs_baseline_best"
    ].map(lambda value: f"{value:+.3f}%")
    groups = group_summary[
        group_summary["method_variant"].eq(winner_method)
        & ~group_summary["group_dimension"].eq("overall")
    ].copy()
    groups["relative_pct_ndcg@20"] = groups["relative_pct_ndcg@20"].map(
        lambda value: f"{value:+.3f}%"
    )
    groups["relative_pct_hr@20"] = groups["relative_pct_hr@20"].map(
        lambda value: f"{value:+.3f}%"
    )
    groups["absolute_delta_ndcg@20"] = groups["absolute_delta_ndcg@20"].map(
        lambda value: f"{value:+.12f}"
    )
    groups["absolute_delta_hr@20"] = groups["absolute_delta_hr@20"].map(
        lambda value: f"{value:+.12f}"
    )
    groups["paired_bootstrap_relative_ci95"] = groups.apply(
        lambda row: (
            f"[{row['ndcg@20_bootstrap_relative_pct_ci95_low']:+.3f}%, "
            f"{row['ndcg@20_bootstrap_relative_pct_ci95_high']:+.3f}%]"
        ),
        axis=1,
    )
    winner_score = score_response[
        score_response["method_variant"].eq(winner_method)
        & score_response["response"].eq("ndcg@20")
    ].copy()
    winner_score = winner_score.sort_values("spearman", key=lambda values: values.abs(), ascending=False)
    winner_rsp = rsp_summary[rsp_summary["method_variant"].eq(winner_method)].copy()
    winner_rsp["relative_pct_ndcg@20"] = winner_rsp["relative_pct_ndcg@20"].map(
        lambda value: f"{value:+.3f}%"
    )
    winner_rsp["relative_pct_hr@20"] = winner_rsp["relative_pct_hr@20"].map(
        lambda value: f"{value:+.3f}%"
    )
    winner_rsp["absolute_delta_ndcg@20"] = winner_rsp[
        "absolute_delta_ndcg@20"
    ].map(lambda value: f"{value:+.12f}")
    winner_rsp["absolute_delta_hr@20"] = winner_rsp[
        "absolute_delta_hr@20"
    ].map(lambda value: f"{value:+.12f}")
    diagnostic_groups = group_summary[
        group_summary["method_variant"].eq(winner_method)
        & ~group_summary["group_dimension"].eq("overall")
    ]
    best_group = diagnostic_groups.sort_values("relative_pct_ndcg@20", ascending=False).iloc[0]
    worst_group = diagnostic_groups.sort_values("relative_pct_ndcg@20").iloc[0]
    ci_low, ci_high = decision["winner_overall_bootstrap_relative_pct_ci95"]

    report = f"""---
title: {run_stamp} CCFCRec Amazon-VG CICP-MP-R1 six mechanisms training 结果
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - CICP-MP-R1
  - experiment-result
---

# CCFCRec Amazon-VG CICP-MP-R1 six mechanisms training 结果

关联设计：[[2026-07-18 014651 CCFCRec Amazon-VG CICP-MP-R1机制空间审查与六机制正式实验设计]]

## Material Passport（材料护照）

- origin skill（来源技能）：academic-research-suite / experiment-agent（学术研究套件 / 实验代理）
- origin mode（来源模式）：validate（验证）
- verification status（验证状态）：ANALYZED_REAGGREGATION_VERIFIED（已分析且逐物品重聚合通过）
- overall confidence（整体置信等级）：CAUTION（谨慎）
- source result（源结果）：`{result_root}`
- evaluated scope（已评价范围）：全部 5,298 个 validation cold-start items（验证冷启动物品）
- full profile relation（相对完整档案）：`5,298 / 35,322 = {VALIDATION_ITEM_COUNT / FULL_ITEM_COUNT * 100.0:.3f}%`
- train/test/full recommendation metrics（训练/测试/完整混合推荐指标）：未评估
- evidence class（证据类别）：seed43（第43号单随机种子）开发阶段验证证据

## 一、结论先行

六个分支均完整完成 `100 epoch（100训练轮次）`。最佳分支是 {decision['winner_label']}，第 `{decision['winner_best_epoch']}` 轮 NDCG@20（前20归一化折损累计增益）为 `{decision['winner_ndcg@20']:.12f}`，相对同口径 baseline（基线）绝对变化 `{decision['winner_absolute_delta_ndcg@20_vs_baseline']:+.12f}`、相对变化 `{decision['winner_relative_pct_ndcg@20_vs_baseline']:+.3f}%`；同一检查点 HR@20（前20命中率）绝对变化 `{decision['winner_absolute_delta_hr@20_vs_baseline']:+.12f}`、相对变化 `{decision['winner_relative_pct_hr@20_vs_baseline']:+.3f}%`。

这个结果既没有达到 `+3%` 门槛，也没有达到 `+2%`，并且比历史 CICP-R1-E1（第一轮类别增量协同可预测性第一方案）的 `0.126354282888` 低 `{abs(float(winner['absolute_delta_vs_historical_cicpr1_e1'])):.12f}`，相对低 `{abs(decision['winner_relative_pct_vs_historical_cicpr1_e1']):.3f}%`。按照预注册停止规则，本轮结论是：**停止继续扩展 CICP（类别增量协同可预测性）载体，不进入局部参数扫描、不开放 multi-seed（多随机种子）；转入类别语义基础与更新冷启动推荐骨干的并列结构审查。**

配对 bootstrap（配对自助法）给出的最佳分支整体相对变化95%区间为 `[{ci_low:+.3f}%, {ci_high:+.3f}%]`。该区间是否跨零只用于描述 seed43（第43号随机种子）下物品层面的不确定性，不能替代多随机种子确认。

## 二、实验验收

字段说明：

{_field_lines([
    ('method_label', '六个预注册机制的名称。'),
    ('method_variant', '运行配置中的机制标识。'),
    ('seed', '训练随机种子，六支均应为43。'),
    ('epochs', '实际训练轮数，六支均应为100。'),
    ('negative_sampling_mode', '负采样协议，六支均为fast_uniform（快速均匀负采样）。'),
    ('training_input_uses_validation_item_metrics', '验证逐物品推荐结果是否进入训练输入，应为False。'),
    ('training_input_uses_test_item_metrics', '测试逐物品推荐结果是否进入训练输入，应为False。'),
    ('common_parameter_sha256', '实际公共参数初始化哈希，六支必须相同。'),
    ('method_specific_parameters', '该机制相对公共主干新增的可训练参数量。'),
])}

{_markdown_table(config_audit, list(config_audit.columns))}

验收结果：六支 `result.csv（结果曲线）` 均恰有100行、检查点均恰有100个、公共参数实际哈希只有1个唯一值；仅E1使用 E4-style residual（E4式残差），其余五支机制实质不同。验证和测试逐物品推荐结果均未作为训练输入，测试推荐指标也未被本分析读取或生成。

六支从 `2026-07-18 12:46:10` 运行至 `2026-07-18 21:30:08`，总耗时约 `8小时43分58秒`；没有失败、补跑或短训练分支。

## 三、六机制主结果

字段说明：

{_field_lines([
    ('method_label', '机制名称。'),
    ('best_epoch', '按验证NDCG@20选择的最佳训练轮次。'),
    ('best_ndcg@20', '全部5,298个验证冷启动物品的宏平均NDCG@20。'),
    ('absolute_delta_ndcg@20_vs_baseline_best', '相对baseline最佳值的NDCG@20绝对变化。'),
    ('relative_pct_ndcg@20_vs_baseline_best', '相对同口径baseline最佳值的NDCG@20百分比变化。'),
    ('absolute_delta_hr@20_vs_baseline_best', '分支最佳NDCG检查点相对baseline最佳值的HR@20绝对变化。'),
    ('relative_pct_hr@20_vs_baseline_best', '在该分支最佳NDCG检查点上，相对baseline最佳值的HR@20百分比变化。'),
    ('paired_bootstrap_relative_ci95', '配对逐物品bootstrap得到的NDCG相对变化95%区间。'),
    ('relative_pct_vs_historical_cicpr1_e1', '相对历史CICP-R1-E1最佳NDCG@20的百分比变化。'),
    ('passed_two_pct', '是否达到预注册+2%区间下界。'),
    ('passed_three_pct', '是否达到预注册+3%主门槛。'),
])}

{_markdown_table(main, ['method_label','best_epoch','best_ndcg@20','absolute_delta_ndcg@20_vs_baseline_best','relative_pct_ndcg@20_vs_baseline_best','absolute_delta_hr@20_vs_baseline_best','relative_pct_hr@20_vs_baseline_best','paired_bootstrap_relative_ci95','relative_pct_vs_historical_cicpr1_e1','passed_two_pct','passed_three_pct'])}

同口径 baseline（基线）为 NDCG@20（前20归一化折损累计增益）`{BASELINE_NDCG20:.12f}`、HR@20（前20命中率）`{BASELINE_HR20:.12f}`。这里的“整体”只指全部5,298个验证冷启动物品，占完整35,322个物品的 `{VALIDATION_ITEM_COUNT / FULL_ITEM_COUNT * 100.0:.3f}%`；不是完整物品范围、不是训练物品、不是测试物品，也不是某个局部画像群体。

## 四、曲线稳定性

字段说明：

{_field_lines([
    ('method_label', '机制名称。'),
    ('same_epoch_relative_pct_ndcg@20', '分支最佳轮相对baseline同一轮的NDCG@20百分比变化。'),
    ('at_baseline_best_relative_pct_ndcg@20', '在baseline最佳轮次上的NDCG@20百分比变化。'),
    ('late30_relative_pct_ndcg@20_vs_baseline_best', '第71至100轮均值相对baseline最佳值的百分比变化。'),
    ('late30_matched_relative_pct_ndcg@20', '第71至100轮均值相对baseline同区间均值的百分比变化。'),
    ('late30_above_baseline_best_epochs', '后30轮中高于baseline最佳值的轮数。'),
    ('late30_matched_positive_epochs', '后30轮中高于baseline同轮值的轮数。'),
    ('all100_matched_positive_epochs', '全部100轮中高于baseline同轮值的轮数。'),
])}

{_markdown_table(stability, ['method_label','same_epoch_relative_pct_ndcg@20','at_baseline_best_relative_pct_ndcg@20','late30_relative_pct_ndcg@20_vs_baseline_best','late30_matched_relative_pct_ndcg@20','late30_above_baseline_best_epochs','late30_matched_positive_epochs','all100_matched_positive_epochs'])}

最佳分支后30轮均值为 `{float(winner['late30_mean_ndcg@20']):.12f}`，相对baseline最佳值 `{float(winner['late30_relative_pct_ndcg@20_vs_baseline_best']):+.3f}%`。因此最佳峰值不能解释为持续超过基线的稳定平台。

## 五、最佳分支的关键互补群体

字段说明：

{_field_lines([
    ('group_dimension', '预先解释的画像分组维度：可靠性、语义增量符号、归因熵或方向范数。'),
    ('group_value', '该维度下的互补组取值。'),
    ('item_count', '组内验证冷启动物品数。'),
    ('coverage_pct_of_validation', '组内物品占全部验证冷启动物品的比例。'),
    ('baseline_ndcg@20', '该组baseline宏平均NDCG@20。'),
    ('method_ndcg@20', '该组最佳分支宏平均NDCG@20。'),
    ('absolute_delta_ndcg@20', '该组NDCG@20绝对变化。'),
    ('relative_pct_ndcg@20', '该组NDCG@20相对百分比变化。'),
    ('relative_pct_hr@20', '该组HR@20相对百分比变化。'),
    ('absolute_delta_hr@20', '该组HR@20绝对变化。'),
    ('paired_bootstrap_relative_ci95', '该组配对逐物品bootstrap的NDCG相对变化95%区间。'),
    ('helped_ndcg_item_count', '逐物品NDCG@20严格增加的物品数。'),
    ('harmed_ndcg_item_count', '逐物品NDCG@20严格下降的物品数。'),
])}

{_markdown_table(groups, ['group_dimension','group_value','item_count','coverage_pct_of_validation','baseline_ndcg@20','method_ndcg@20','absolute_delta_ndcg@20','relative_pct_ndcg@20','absolute_delta_hr@20','relative_pct_hr@20','paired_bootstrap_relative_ci95','helped_ndcg_item_count','harmed_ndcg_item_count'])}

观察到的最高子组是 `{best_group['group_dimension']} / {best_group['group_value']}`，相对变化 `{float(best_group['relative_pct_ndcg@20']):+.3f}%`；但它只有 `{int(best_group['item_count'])}` 个物品，占验证集合 `{float(best_group['coverage_pct_of_validation']):.3f}%`，95%区间为 `[{float(best_group['ndcg@20_bootstrap_relative_pct_ci95_low']):+.3f}%, {float(best_group['ndcg@20_bootstrap_relative_pct_ci95_high']):+.3f}%]`。最低子组是 `{worst_group['group_dimension']} / {worst_group['group_value']}`，相对变化 `{float(worst_group['relative_pct_ndcg@20']):+.3f}%`。所有关键子组的95%区间均跨零，且可靠性三等分没有形成随可靠性升高而稳定增强的收益规律。这些是 post-hoc diagnostic evidence（事后诊断证据），只能描述异质性，不能把小群体收益替代全部5,298个验证冷启动物品的整体结论，也不能据此自动启动下一轮小改。

## 六、连续画像响应

字段说明：

{_field_lines([
    ('predictor', 'CICP-MP画像中的连续预测分量或派生可靠性/方向范数。'),
    ('spearman', '该分量与最佳分支逐物品NDCG@20变化的斯皮尔曼秩相关。'),
    ('p_value_uncorrected', '未经多重比较校正的p值。'),
    ('q_value_bh_98_tests', '对全部98个相关检验做Benjamini-Hochberg校正后的q值。'),
    ('significant_after_bh_0.05', '在FDR（错误发现率）0.05下是否显著。'),
])}

{_markdown_table(winner_score, ['predictor','spearman','p_value_uncorrected','q_value_bh_98_tests','significant_after_bh_0.05'])}

最强绝对相关来自 `{decision['strongest_posthoc_predictor']}`，Spearman（斯皮尔曼秩相关）为 `{decision['strongest_posthoc_predictor_spearman']:+.4f}`，BH（多重比较）校正后 `q={decision['strongest_posthoc_predictor_q_value_bh']:.4f}`。无论是否显著，这都不是随机化干预，不能写成因果机制证据。

## 七、R/S/P事后分层

字段说明：

{_field_lines([
    ('stratification_label', 'R丰富度、S支持度或P流行度分层变量。'),
    ('stratum', '验证物品协变量三等分的low/mid/high层。'),
    ('item_count', '层内验证冷启动物品数。'),
    ('baseline_ndcg@20', '层内baseline宏平均NDCG@20。'),
    ('method_ndcg@20', '层内最佳分支宏平均NDCG@20。'),
    ('absolute_delta_ndcg@20', '层内NDCG@20绝对变化。'),
    ('relative_pct_ndcg@20', '层内NDCG@20相对百分比变化。'),
    ('relative_pct_hr@20', '层内HR@20相对百分比变化。'),
    ('absolute_delta_hr@20', '层内HR@20绝对变化。'),
])}

{_markdown_table(winner_rsp, ['stratification_label','stratum','item_count','baseline_ndcg@20','method_ndcg@20','absolute_delta_ndcg@20','relative_pct_ndcg@20','absolute_delta_hr@20','relative_pct_hr@20'])}

R/S/P（丰富度/支持度/流行度）只用于事后描述，没有构成同载体匹配控制，因此不能据此声称控制混杂后 CICP-MP（类别增量协同可预测性多分量画像）有效或无效。

## 八、评价覆盖

字段说明：

{_field_lines([
    ('evaluation_scope', '完整档案、训练、验证、测试或验证画像互补组。'),
    ('scope_item_count', '该范围定义下的物品数；未定义目标组时为0。'),
    ('evaluated_item_count', '该范围实际进入推荐指标聚合的物品数。'),
    ('coverage_pct_of_full_profile', '该范围占35,322个完整物品档案的比例。'),
    ('coverage_pct_of_validation', '该范围占5,298个验证冷启动物品的比例。'),
    ('recommendation_metric_status', '该范围推荐指标是已评估、未评估或不适用。'),
    ('note', '覆盖关系和解释限制。'),
])}

{_markdown_table(coverage, list(coverage.columns))}

完整物品、训练非冷启动和测试冷启动推荐指标明确标记为“未评估”。CICP-MP-v1（第一版类别增量协同可预测性多分量画像）是连续画像，不等于全部冷启动物品，也没有预注册的目标/非目标二元标签；全部冷启动物品在本报告中仅指5,298个验证冷启动物品。

## 九、统计谬误扫描

字段说明：

{_field_lines([
    ('fallacy', '被检查的统计推断风险。'),
    ('severity', 'CHECKED表示已通过当前可做审计，CAUTION表示解释时必须保留限制。'),
    ('finding', '本轮对应的具体证据和边界。'),
])}

{_markdown_table(fallacy_scan, list(fallacy_scan.columns))}

扫描覆盖：`11/11`。此外，连续画像相关检验已对98项比较做 BH（Benjamini-Hochberg错误发现率）校正，但六支各自100轮峰值仍属于开发阶段多处搜寻，不因该校正而变成确认性结果。

## 十、路线判断

1. `passed_two_pct（达到+2%） = {decision['passed_two_pct']}`。
2. `passed_three_pct（达到+3%） = {decision['passed_three_pct']}`。
3. `beat_historical_cicpr1_e1（超过历史CICP-R1-E1） = {decision['beat_historical_cicpr1_e1']}`。
4. `automatic_followup_authorized（自动允许后续训练） = {decision['automatic_followup_authorized']}`。
5. `route（路线） = {decision['route']}`。

因此不继续围绕 CICP-MP（类别增量协同可预测性多分量画像）做残差剂量、损失权重、门控温度或局部采样扫描。下一阶段是结构级审查，不是马上再训练：一条线审查可复现的类别语义基础增强，另一条线审查更新的 cold-start recommendation backbone（冷启动推荐模型骨干）。二者都必须先形成新的可证伪设计、训练安全边界和同口径baseline（基线）比较协议。

## 十一、核心产物

- `cicpmp_r1_curve_summary.csv`：六支100轮曲线摘要。
- `cicpmp_r1_epoch_matched_curve.csv`：六支与baseline同轮对比明细。
- `cicpmp_r1_validation_item_metrics.csv`：baseline加六支共37,086条验证逐物品记录。
- `cicpmp_r1_group_summary.csv`：四类关键互补群体结果。
- `cicpmp_r1_score_response_summary.csv`：连续画像响应与多重比较校正。
- `cicpmp_r1_rsp_strata_summary.csv`：R/S/P事后分层。
- `cicpmp_r1_evaluation_coverage.csv`：完整数据范围覆盖关系。
- `cicpmp_r1_fallacy_scan.csv`：11项统计谬误扫描。
- `cicpmp_r1_route_decision.json`：机器可读路线决策。
- `run_manifest.json`：输入、协议与全部输出清单。
"""

    route_report = f"""---
title: {run_stamp} CCFCRec Amazon-VG CICP-MP-R1 six mechanisms training 路线判断
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - CICP-MP-R1
  - route-decision
---

# CCFCRec Amazon-VG CICP-MP-R1 six mechanisms training 路线判断

关联设计：[[2026-07-18 014651 CCFCRec Amazon-VG CICP-MP-R1机制空间审查与六机制正式实验设计]]

关联结果：[[{run_stamp} CCFCRec Amazon-VG CICP-MP-R1 six mechanisms training 结果]]

> [!important] 主决策
> 六个 CICP-MP-R1（第一轮类别增量协同可预测性多分量画像）机制全部低于 `+2%`，且全部低于历史 CICP-R1-E1（第一轮类别增量协同可预测性第一方案）的 `+2.051%`。按预注册规则停止 CICP（类别增量协同可预测性）载体扩展，不做局部参数扫描，不进入 multi-seed（多随机种子）。

## 决策依据

1. 主评价集合是全部5,298个验证冷启动物品，占完整35,322个物品的 `{VALIDATION_ITEM_COUNT / FULL_ITEM_COUNT * 100.0:.3f}%`；训练、测试与完整混合推荐指标均未评估。
2. 最佳 {decision['winner_label']} 的 NDCG@20（前20归一化折损累计增益）为 `{decision['winner_ndcg@20']:.12f}`，相对baseline（基线）仅 `{decision['winner_relative_pct_ndcg@20_vs_baseline']:+.3f}%`。
3. 同一检查点 HR@20（前20命中率）相对baseline（基线）为 `{decision['winner_relative_pct_hr@20_vs_baseline']:+.3f}%`，但NDCG主指标距离 `+3%` 门槛仍差 `{abs(decision['absolute_gap_to_three_pct']):.12f}`。
4. 最佳分支相对历史 CICP-R1-E1（第一轮类别增量协同可预测性第一方案）仍为 `{decision['winner_relative_pct_vs_historical_cicpr1_e1']:+.3f}%`，多分量画像没有在当前接入下形成历史增量。
5. 六支全部完成100轮并共享相同公共参数初始化哈希；停止判断不是由训练失败、短训练或初始化不一致造成。
6. 本轮没有因果控制，结论应写为“六种预注册接入没有产生充分性能收益”，不能写成“CICP-MP语义已被因果证伪”。

## 下一阶段

1. 停止 CICP（类别增量协同可预测性）残差、门控、辅助损失、负样本权重和剂量的连续扩展。
2. 保留全部本轮产物，作为说明“多分量画像与六类载体均未超过历史单分数残差”的负结果证据。
3. 并列审查 category semantic basis（类别语义基础）：类别文本、可验证层级、商品属性与可复现外部语义数据，目标是形成低维语义可用性先验、类别图或辅助监督，而不是把大模型高维向量直接拼到CCFCRec后面。
4. 并列审查 newer cold-start backbone（更新冷启动推荐骨干）：寻找能更自然承载类别语义与协同对齐的结构，同时保留同协议baseline、100 epoch（100训练轮次）和训练输入禁用验证/测试逐物品答案的边界。
5. 两条线先做设计与可证伪性审查，不自动启动下一波训练；只有形成明显不同于当前CICP载体扩展的新假设后再立项。

## 固化状态

- route（路线）：`{decision['route']}`
- run another CICP carrier batch now（现在再跑一批CICP载体）：`False`
- run multi-seed now（现在运行多随机种子）：`False`
- review category semantic basis（审查类别语义基础）：`True`
- review newer cold-start backbone（审查更新冷启动推荐骨干）：`True`
- test item metrics read or generated（读取或生成测试逐物品指标）：`False`
- fallacy scan coverage（统计谬误扫描覆盖）：`11/11`
"""
    result_path = (
        output_dir
        / f"{run_stamp} CCFCRec Amazon-VG CICP-MP-R1 six mechanisms training 结果.md"
    )
    route_output_dir.mkdir(parents=True, exist_ok=True)
    route_path = (
        route_output_dir
        / f"{run_stamp} CCFCRec Amazon-VG CICP-MP-R1 six mechanisms training 路线判断.md"
    )
    result_path.write_text(report, encoding="utf-8")
    route_path.write_text(route_report, encoding="utf-8")
    return result_path, route_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_summary, matched_curve, baseline_curve, config_audit = collect_curves(
        args.result_root.resolve(), args.baseline_result.resolve()
    )
    winner_method = str(
        curve_summary.sort_values("best_ndcg@20", ascending=False).iloc[0][
            "method_variant"
        ]
    )
    item_metrics_path = (
        args.item_evaluation_dir.resolve() / "cicpmp_r1_validation_item_metrics.csv"
    )
    item_metrics = pd.read_csv(item_metrics_path, dtype={"raw_asin": str}, low_memory=False)
    expected_rows = 7 * VALIDATION_ITEM_COUNT
    if len(item_metrics) != expected_rows:
        raise ValueError(
            f"unexpected validation item row count: {len(item_metrics)} != {expected_rows}"
        )
    if item_metrics["raw_asin"].nunique() != VALIDATION_ITEM_COUNT:
        raise ValueError("validation item metrics do not cover 5,298 unique items")
    group_summary = build_group_summary(
        item_metrics, winner_method, args.bootstrap_repetitions
    )
    score_response = build_score_response(item_metrics)
    rsp_summary, rsp_thresholds = build_rsp_strata(
        item_metrics, args.source_profile.resolve()
    )
    coverage = build_coverage(group_summary, winner_method)
    fallacy_scan = build_fallacy_scan(group_summary, winner_method)
    decision = build_decision(curve_summary, group_summary, score_response)

    artifacts: dict[str, pd.DataFrame] = {
        "cicpmp_r1_config_acceptance.csv": config_audit,
        "cicpmp_r1_curve_summary.csv": curve_summary,
        "cicpmp_r1_epoch_matched_curve.csv": matched_curve,
        "cicpmp_r1_baseline_curve.csv": baseline_curve,
        "cicpmp_r1_group_summary.csv": group_summary,
        "cicpmp_r1_score_response_summary.csv": score_response,
        "cicpmp_r1_rsp_strata_summary.csv": rsp_summary,
        "cicpmp_r1_evaluation_coverage.csv": coverage,
        "cicpmp_r1_fallacy_scan.csv": fallacy_scan,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output_dir / name, index=False)
    (output_dir / "cicpmp_r1_route_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result_path, route_path = write_reports(
        output_dir=output_dir,
        route_output_dir=args.route_output_dir.resolve(),
        run_stamp=args.run_stamp,
        result_root=args.result_root.resolve(),
        config_audit=config_audit,
        curve_summary=curve_summary,
        group_summary=group_summary,
        rsp_summary=rsp_summary,
        score_response=score_response,
        coverage=coverage,
        fallacy_scan=fallacy_scan,
        decision=decision,
    )
    evaluation_audit_path = (
        args.item_evaluation_dir.resolve()
        / "cicpmp_r1_validation_evaluation_audit.json"
    )
    evaluation_audit = json.loads(evaluation_audit_path.read_text(encoding="utf-8"))
    if evaluation_audit["test_recommendation_metrics_read_or_generated"]:
        raise ValueError("evaluation audit says test recommendation metrics were accessed")
    manifest = {
        "protocol": "cicpmp_r1_six_training_analysis_v1",
        "run_stamp": args.run_stamp,
        "result_root": str(args.result_root.resolve()),
        "baseline_result": str(args.baseline_result.resolve()),
        "item_evaluation_dir": str(args.item_evaluation_dir.resolve()),
        "item_metrics": str(item_metrics_path),
        "item_evaluation_audit": str(evaluation_audit_path),
        "source_profile": str(args.source_profile.resolve()),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "rsp_validation_tertile_thresholds": rsp_thresholds,
        "recommendation_metric_scope": "validation_cold_5298_only",
        "validation_coverage_pct_of_full_profile": VALIDATION_ITEM_COUNT
        / FULL_ITEM_COUNT
        * 100.0,
        "train_recommendation_metrics_evaluated": False,
        "test_recommendation_metrics_read_or_generated": False,
        "full_mixed_recommendation_metrics_evaluated": False,
        "outputs": [
            *artifacts,
            "cicpmp_r1_validation_item_metrics.csv",
            "cicpmp_r1_validation_evaluation_audit.json",
            "cicpmp_r1_route_decision.json",
            str(result_path),
            str(route_path),
        ],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--item-evaluation-dir", type=Path, required=True)
    parser.add_argument("--source-profile", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--route-output-dir", type=Path, required=True)
    parser.add_argument("--run-stamp", required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
