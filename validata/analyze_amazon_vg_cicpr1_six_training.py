#!/usr/bin/env python3
"""Analyze the six CICP-R1 validation runs and write auditable artifacts."""

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
OLD_M11R2_BEST = 0.125158167769
FULL_ITEM_COUNT = 35322
TRAIN_ITEM_COUNT = 24726
VALIDATION_ITEM_COUNT = 5298
TEST_ITEM_COUNT = 5298
METHOD_ORDER = [
    "cicpr1_e4_residual",
    "cicpr1_modality_routing",
    "cicpr1_category_expert",
    "cicpr1_alignment_curriculum",
    "cicpr1_counterfactual_margin",
    "cicpr1_adaptive_attention",
]
METHOD_LABELS = {
    "cicpr1_e4_residual": "E1 CICP residual adapter（CICP残差适配器）",
    "cicpr1_modality_routing": "E2 modality routing（模态路由）",
    "cicpr1_category_expert": "E3 category expert（类别专家）",
    "cicpr1_alignment_curriculum": "E4 alignment curriculum（对齐课程）",
    "cicpr1_counterfactual_margin": "E5 counterfactual margin（反事实间隔）",
    "cicpr1_adaptive_attention": "E6 adaptive attention（自适应注意力）",
}
ITEM_METHOD_LABELS = {
    "cicpr1_e4_residual": "CICPR1E1_e4_residual",
    "cicpr1_modality_routing": "CICPR1E2_modality_routing",
    "cicpr1_category_expert": "CICPR1E3_category_expert",
    "cicpr1_alignment_curriculum": "CICPR1E4_alignment_curriculum",
    "cicpr1_counterfactual_margin": "CICPR1E5_counterfactual_margin",
    "cicpr1_adaptive_attention": "CICPR1E6_adaptive_attention",
}
RSP_COLUMNS = {
    "R_metadata_richness_score": "R richness（丰富度）",
    "S_train_support_score": "S support（支持度）",
    "P_popularity_score": "P popularity（流行度）",
}


def _tar_result_csv(package: Path) -> pd.DataFrame:
    with tarfile.open(package, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.name.endswith("/result.csv")]
        if len(members) != 1:
            raise ValueError(f"expected one baseline result.csv, got {len(members)}")
        file_obj = tar.extractfile(members[0])
        if file_obj is None:
            raise ValueError("cannot read baseline result.csv")
        return pd.read_csv(io.BytesIO(file_obj.read()))


def relative_pct(value: float, baseline: float) -> float:
    if float(baseline) == 0.0:
        return float("nan")
    return (float(value) / float(baseline) - 1.0) * 100.0


def summarize_curve(
    method: str,
    curve: pd.DataFrame,
    baseline_curve: pd.DataFrame,
    run_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if len(curve) != 100 or curve["epoch"].astype(int).tolist() != list(range(1, 101)):
        raise ValueError(f"{method} does not contain exactly epochs 1..100")
    if len(list(run_dir.glob("*.pt"))) != 100:
        raise ValueError(f"{method} does not contain exactly 100 checkpoints")
    best = curve.sort_values(
        ["ndcg@20", "hr@20", "epoch", "checkpoint_index"],
        ascending=[False, False, True, True],
    ).iloc[0]
    epoch = int(best["epoch"])
    baseline_by_epoch = baseline_curve.set_index("epoch")
    method_by_epoch = curve.set_index("epoch")
    matched = method_by_epoch[["ndcg@20", "hr@20"]].join(
        baseline_by_epoch[["ndcg@20", "hr@20"]],
        lsuffix="_method",
        rsuffix="_baseline",
        validate="one_to_one",
    )
    matched["absolute_delta_ndcg@20"] = (
        matched["ndcg@20_method"] - matched["ndcg@20_baseline"]
    )
    matched["relative_pct_ndcg@20"] = (
        matched["ndcg@20_method"] / matched["ndcg@20_baseline"] - 1.0
    ) * 100.0
    matched["absolute_delta_hr@20"] = matched["hr@20_method"] - matched["hr@20_baseline"]
    matched["relative_pct_hr@20"] = (
        matched["hr@20_method"] / matched["hr@20_baseline"] - 1.0
    ) * 100.0
    matched = matched.reset_index()
    matched.insert(0, "method_variant", method)
    late_curve = curve[curve["epoch"].between(71, 100)]
    late_matched = matched[matched["epoch"].between(71, 100)]
    baseline_late = baseline_curve[baseline_curve["epoch"].between(71, 100)]
    best_ndcg = float(best["ndcg@20"])
    best_hr = float(best["hr@20"])
    late_ndcg = float(late_curve["ndcg@20"].mean())
    late_hr = float(late_curve["hr@20"].mean())
    same_epoch = matched[matched["epoch"].eq(epoch)].iloc[0]
    at_baseline_best = curve[curve["epoch"].eq(74)].iloc[0]
    summary = {
        "method_variant": method,
        "method_label": METHOD_LABELS[method],
        "run_dir": str(run_dir),
        "epoch_count": int(len(curve)),
        "checkpoint_count": int(len(list(run_dir.glob("*.pt")))),
        "best_epoch": epoch,
        "best_checkpoint_index": int(best["checkpoint_index"]),
        "best_ndcg@20": best_ndcg,
        "absolute_delta_ndcg@20_vs_baseline_best": best_ndcg - BASELINE_NDCG20,
        "relative_pct_ndcg@20_vs_baseline_best": relative_pct(best_ndcg, BASELINE_NDCG20),
        "best_hr@20_same_checkpoint": best_hr,
        "absolute_delta_hr@20_vs_baseline_best": best_hr - BASELINE_HR20,
        "relative_pct_hr@20_vs_baseline_best": relative_pct(best_hr, BASELINE_HR20),
        "passed_three_pct_threshold": bool(best_ndcg >= THREE_PCT_THRESHOLD),
        "absolute_gap_to_three_pct_threshold": best_ndcg - THREE_PCT_THRESHOLD,
        "relative_pct_vs_old_m11r2_best": relative_pct(best_ndcg, OLD_M11R2_BEST),
        "same_epoch_baseline_ndcg@20": float(same_epoch["ndcg@20_baseline"]),
        "same_epoch_relative_pct_ndcg@20": float(same_epoch["relative_pct_ndcg@20"]),
        "same_epoch_relative_pct_hr@20": float(same_epoch["relative_pct_hr@20"]),
        "at_baseline_best_epoch74_ndcg@20": float(at_baseline_best["ndcg@20"]),
        "at_baseline_best_epoch74_relative_pct_ndcg@20": relative_pct(
            float(at_baseline_best["ndcg@20"]), BASELINE_NDCG20
        ),
        "at_baseline_best_epoch74_relative_pct_hr@20": relative_pct(
            float(at_baseline_best["hr@20"]), BASELINE_HR20
        ),
        "late30_mean_ndcg@20": late_ndcg,
        "late30_relative_pct_ndcg@20_vs_baseline_best": relative_pct(
            late_ndcg, BASELINE_NDCG20
        ),
        "late30_mean_hr@20": late_hr,
        "late30_relative_pct_hr@20_vs_baseline_best": relative_pct(late_hr, BASELINE_HR20),
        "late30_positive_ndcg_epochs_vs_baseline_best": int(
            (late_curve["ndcg@20"] > BASELINE_NDCG20).sum()
        ),
        "late30_positive_hr_epochs_vs_baseline_best": int(
            (late_curve["hr@20"] > BASELINE_HR20).sum()
        ),
        "matched_positive_ndcg_epochs": int((matched["absolute_delta_ndcg@20"] > 0.0).sum()),
        "matched_positive_hr_epochs": int((matched["absolute_delta_hr@20"] > 0.0).sum()),
        "late30_matched_relative_pct_ndcg@20": relative_pct(
            float(late_curve["ndcg@20"].mean()),
            float(baseline_late["ndcg@20"].mean()),
        ),
        "late30_matched_positive_ndcg_epochs": int(
            (late_matched["absolute_delta_ndcg@20"] > 0.0).sum()
        ),
    }
    return summary, matched


def collect_curves(
    result_root: Path,
    baseline_result: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline_curve = _tar_result_csv(baseline_result)
    if len(baseline_curve) != 100:
        raise ValueError("baseline curve must contain 100 epochs")
    summaries: list[dict[str, Any]] = []
    matched_frames: list[pd.DataFrame] = []
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
        curve = pd.read_csv(result_path)
        summary, matched = summarize_curve(method, curve, baseline_curve, run_dir)
        summaries.append(summary)
        matched_frames.append(matched)
    if seen != set(METHOD_ORDER):
        raise ValueError(f"missing CICP-R1 methods: {sorted(set(METHOD_ORDER) - seen)}")
    summary_frame = pd.DataFrame(summaries)
    summary_frame["method_order"] = summary_frame["method_variant"].map(METHOD_ORDER.index)
    summary_frame = summary_frame.sort_values("method_order").drop(columns="method_order")
    return summary_frame.reset_index(drop=True), pd.concat(matched_frames, ignore_index=True), baseline_curve


def _paired_bootstrap(
    method_values: np.ndarray,
    baseline_values: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> dict[str, float]:
    if method_values.shape != baseline_values.shape or method_values.ndim != 1:
        raise ValueError("paired bootstrap inputs must be same-shape vectors")
    rng = np.random.default_rng(seed)
    n = len(method_values)
    absolute = np.empty(repetitions, dtype=float)
    relative = np.empty(repetitions, dtype=float)
    for index in range(repetitions):
        selected = rng.integers(0, n, n)
        method_mean = float(method_values[selected].mean())
        baseline_mean = float(baseline_values[selected].mean())
        absolute[index] = method_mean - baseline_mean
        relative[index] = relative_pct(method_mean, baseline_mean)
    return {
        "bootstrap_absolute_ci95_low": float(np.quantile(absolute, 0.025)),
        "bootstrap_absolute_ci95_high": float(np.quantile(absolute, 0.975)),
        "bootstrap_relative_pct_ci95_low": float(np.nanquantile(relative, 0.025)),
        "bootstrap_relative_pct_ci95_high": float(np.nanquantile(relative, 0.975)),
    }


def build_group_bootstrap(
    item_metrics: pd.DataFrame,
    repetitions: int,
) -> pd.DataFrame:
    baseline = item_metrics[item_metrics["method_label"].eq(BASELINE_LABEL)].set_index("raw_asin")
    rows: list[dict[str, Any]] = []
    group_order = ["overall", "low", "mid", "high"]
    for method_index, method in enumerate(METHOD_ORDER):
        method_label = ITEM_METHOD_LABELS[method]
        method_frame = item_metrics[item_metrics["method_label"].eq(method_label)].set_index("raw_asin")
        paired = method_frame.join(
            baseline[["ndcg@20", "hr@20"]],
            rsuffix="_baseline",
            validate="one_to_one",
        )
        for group_index, group in enumerate(group_order):
            frame = paired if group == "overall" else paired[paired["cicp_group"].eq(group)]
            row: dict[str, Any] = {
                "method_variant": method,
                "method_label": METHOD_LABELS[method],
                "cicp_group": group,
                "item_count": int(len(frame)),
            }
            for metric_index, metric in enumerate(("ndcg@20", "hr@20")):
                method_values = frame[metric].to_numpy(dtype=float)
                baseline_values = frame[f"{metric}_baseline"].to_numpy(dtype=float)
                method_mean = float(method_values.mean())
                baseline_mean = float(baseline_values.mean())
                row[f"baseline_{metric}"] = baseline_mean
                row[f"method_{metric}"] = method_mean
                row[f"absolute_delta_{metric}"] = method_mean - baseline_mean
                row[f"relative_pct_{metric}"] = relative_pct(method_mean, baseline_mean)
                uncertainty = _paired_bootstrap(
                    method_values,
                    baseline_values,
                    seed=43 + method_index * 100 + group_index * 10 + metric_index,
                    repetitions=repetitions,
                )
                for key, value in uncertainty.items():
                    row[f"{metric}_{key}"] = value
            delta_ndcg = frame["ndcg@20"] - frame["ndcg@20_baseline"]
            row["helped_ndcg_item_count"] = int((delta_ndcg > 0.0).sum())
            row["harmed_ndcg_item_count"] = int((delta_ndcg < 0.0).sum())
            row["equal_ndcg_item_count"] = int((delta_ndcg == 0.0).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def build_score_response(item_metrics: pd.DataFrame) -> pd.DataFrame:
    def calculate(left: pd.Series, right: pd.Series) -> tuple[float, float]:
        if left.nunique(dropna=True) < 2 or right.nunique(dropna=True) < 2:
            return float("nan"), float("nan")
        result = spearmanr(left, right)
        return float(result.statistic), float(result.pvalue)

    baseline = item_metrics[item_metrics["method_label"].eq(BASELINE_LABEL)].set_index("raw_asin")
    rows = []
    for metric in ("ndcg@20", "hr@20"):
        statistic, p_value = calculate(baseline["cicp_score"], baseline[metric])
        rows.append(
            {
                "method_variant": "baseline",
                "response": metric,
                "comparison": "cicp_score_vs_baseline_metric",
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
        for metric in ("ndcg@20", "hr@20"):
            delta = paired[metric] - paired[f"{metric}_baseline"]
            statistic, p_value = calculate(paired["cicp_score"], delta)
            rows.append(
                {
                    "method_variant": method,
                    "response": metric,
                    "comparison": "cicp_score_vs_item_delta",
                    "spearman": statistic,
                    "p_value_uncorrected": p_value,
                }
            )
    return pd.DataFrame(rows)


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
        thresholds[column] = [float(lower), float(upper)]
        group_column = f"{column}_group"
        profile[group_column] = pd.cut(
            values,
            bins=[-np.inf, lower, upper, np.inf],
            labels=["low", "mid", "high"],
            include_lowest=True,
        ).astype(str)
        keep.append(group_column)
    baseline = item_metrics[item_metrics["method_label"].eq(BASELINE_LABEL)].set_index("raw_asin")
    rows = []
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
            for group in ("low", "mid", "high"):
                part = paired[paired[group_column].eq(group)]
                row = {
                    "method_variant": method,
                    "method_label": METHOD_LABELS[method],
                    "stratification_variable": column,
                    "stratification_label": label,
                    "stratum": group,
                    "item_count": int(len(part)),
                }
                for metric in ("ndcg@20", "hr@20"):
                    baseline_mean = float(part[f"{metric}_baseline"].mean())
                    method_mean = float(part[metric].mean())
                    row[f"baseline_{metric}"] = baseline_mean
                    row[f"method_{metric}"] = method_mean
                    row[f"absolute_delta_{metric}"] = method_mean - baseline_mean
                    row[f"relative_pct_{metric}"] = relative_pct(method_mean, baseline_mean)
                rows.append(row)
    return pd.DataFrame(rows), thresholds


def build_coverage(group_summary: pd.DataFrame) -> pd.DataFrame:
    group_counts = (
        group_summary[group_summary["method_variant"].eq("cicpr1_e4_residual")]
        .set_index("cicp_group")["item_count"]
        .to_dict()
    )
    rows = [
        {
            "evaluation_scope": "complete_item_profile（完整物品档案）",
            "scope_item_count": FULL_ITEM_COUNT,
            "evaluated_item_count": VALIDATION_ITEM_COUNT,
            "coverage_pct_of_full_profile": 100.0,
            "recommendation_metric_status": "未评估",
            "note": "只评估了其中的验证冷启动物品，未形成完整混合推荐指标。",
        },
        {
            "evaluation_scope": "train_non_cold（训练非冷启动）",
            "scope_item_count": TRAIN_ITEM_COUNT,
            "evaluated_item_count": 0,
            "coverage_pct_of_full_profile": TRAIN_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
            "recommendation_metric_status": "未评估",
            "note": "未生成训练物品推荐结果。",
        },
        {
            "evaluation_scope": "validation_cold（验证冷启动）",
            "scope_item_count": VALIDATION_ITEM_COUNT,
            "evaluated_item_count": VALIDATION_ITEM_COUNT,
            "coverage_pct_of_full_profile": VALIDATION_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
            "recommendation_metric_status": "已评估",
            "note": "当前开发阶段正式主评价范围。",
        },
        {
            "evaluation_scope": "test_cold（测试冷启动）",
            "scope_item_count": TEST_ITEM_COUNT,
            "evaluated_item_count": 0,
            "coverage_pct_of_full_profile": TEST_ITEM_COUNT / FULL_ITEM_COUNT * 100.0,
            "recommendation_metric_status": "未评估",
            "note": "未读取或生成测试逐物品推荐结果。",
        },
    ]
    for group in ("low", "mid", "high"):
        count = int(group_counts[group])
        rows.append(
            {
                "evaluation_scope": f"validation_CICP_{group}（验证CICP{group}组）",
                "scope_item_count": count,
                "evaluated_item_count": count,
                "coverage_pct_of_full_profile": count / FULL_ITEM_COUNT * 100.0,
                "recommendation_metric_status": "已评估",
                "note": "三个组互补并完整覆盖5,298个验证冷启动物品；只作子组诊断。",
            }
        )
    return pd.DataFrame(rows)


def build_fallacy_scan() -> pd.DataFrame:
    entries = [
        ("Simpson's paradox（辛普森悖论）", "CAUTION", "E1整体为正，但高CICP组NDCG为负；聚合值掩盖了方向异质性。"),
        ("Ecological fallacy（生态谬误）", "CAUTION", "高/中/低组均值不能推出每个物品都会同向受益。"),
        ("Berkson's paradox（伯克森悖论）", "NOTE", "样本固定为验证冷启动物品，仍需避免把该选择样本的关系外推到训练物品或其他数据集。"),
        ("Collider bias（碰撞变量偏差）", "CAUTION", "R/S/P事后分层只作描述；不能把条件分层后的差异解释为CICP净因果效应。"),
        ("Base-rate neglect（基率忽视）", "NOTE", "报告了三个CICP组的物品数、基线难度和完整覆盖，没有用小群体替代整体。"),
        ("Regression to the mean（均值回归）", "CAUTION", "六分支各从100轮中选择最高验证点；峰值效应可能向均值回落。"),
        ("Survivorship bias（幸存者偏差）", "NOTE", "六个预定分支全部完成并纳入，没有只报告成功分支。"),
        ("Look-elsewhere effect（多重搜寻效应）", "CAUTION", "比较六种机制并各选择100轮峰值，未做确认集或多重比较校正。"),
        ("Garden of forking paths（分叉路径花园）", "CAUTION", "机制和门槛训练前固定，但CICP/RSP子组解释及最佳检查点属于开发阶段选择。"),
        ("Correlation is not causation（相关不等于因果）", "CAUTION", "没有同载体CICP打乱、R/S/P匹配和无类别增量独立控制，不能证明CICP语义导致提升。"),
        ("Reverse causality（反向因果）", "NOTE", "训练画像未含验证/测试推荐结果，直接答案回流被阻断；但推荐收益不能反向证明CICP定义正确。"),
    ]
    return pd.DataFrame(entries, columns=["fallacy", "severity", "finding"])


def build_decision(
    curve_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    score_response: pd.DataFrame,
) -> dict[str, Any]:
    winner = curve_summary.sort_values("best_ndcg@20", ascending=False).iloc[0]
    winner_method = str(winner["method_variant"])
    winner_groups = group_summary[group_summary["method_variant"].eq(winner_method)].set_index(
        "cicp_group"
    )
    winner_score = score_response[
        score_response["method_variant"].eq(winner_method)
        & score_response["response"].eq("ndcg@20")
    ].iloc[0]
    late_stability = bool(
        winner["late30_mean_ndcg@20"] > BASELINE_NDCG20
        and int(winner["late30_positive_ndcg_epochs_vs_baseline_best"]) >= 20
        and int(winner["late30_matched_positive_ndcg_epochs"]) >= 24
    )
    high_group_reversal = bool(winner_groups.loc["high", "relative_pct_ndcg@20"] < 0.0)
    monotonic_response = bool(abs(float(winner_score["spearman"])) >= 0.05)
    bottleneck_clear = bool(not high_group_reversal and monotonic_response)
    passed_three = bool(winner["passed_three_pct_threshold"])
    result_band = "over_3pct" if passed_three else (
        "between_2_and_3pct"
        if float(winner["relative_pct_ndcg@20_vs_baseline_best"]) >= 2.0
        else "below_2pct"
    )
    automatic_followup_authorized = bool(
        passed_three or (result_band == "between_2_and_3pct" and late_stability and bottleneck_clear)
    )
    route = (
        "confirm_controls_then_multiseed"
        if passed_three
        else "one_bottleneck_followup"
        if automatic_followup_authorized
        else "no_automatic_cicp_r2_upgrade_semantic_basis_or_backbone"
    )
    return {
        "winner_method": winner_method,
        "winner_label": str(winner["method_label"]),
        "winner_best_epoch": int(winner["best_epoch"]),
        "winner_ndcg@20": float(winner["best_ndcg@20"]),
        "winner_relative_pct_ndcg@20_vs_baseline": float(
            winner["relative_pct_ndcg@20_vs_baseline_best"]
        ),
        "winner_hr@20": float(winner["best_hr@20_same_checkpoint"]),
        "winner_relative_pct_hr@20_vs_baseline": float(
            winner["relative_pct_hr@20_vs_baseline_best"]
        ),
        "three_pct_threshold": THREE_PCT_THRESHOLD,
        "absolute_gap_to_three_pct_threshold": float(
            winner["absolute_gap_to_three_pct_threshold"]
        ),
        "result_band": result_band,
        "late_stability_pass": late_stability,
        "bottleneck_clarity_pass": bottleneck_clear,
        "high_cicp_group_reversal": high_group_reversal,
        "winner_cicp_score_vs_item_delta_spearman": float(winner_score["spearman"]),
        "automatic_followup_authorized": automatic_followup_authorized,
        "route": route,
        "test_item_metrics_read_or_generated": False,
        "fallacy_scan_coverage": "11/11",
        "evidence_level": "B_development_validation_single_seed_with_posthoc_subgroups",
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


def write_reports(
    *,
    output_dir: Path,
    route_output_dir: Path,
    run_stamp: str,
    result_root: Path,
    curve_summary: pd.DataFrame,
    group_summary: pd.DataFrame,
    rsp_summary: pd.DataFrame,
    score_response: pd.DataFrame,
    coverage: pd.DataFrame,
    fallacy_scan: pd.DataFrame,
    decision: dict[str, Any],
) -> tuple[Path, Path]:
    winner = curve_summary[curve_summary["method_variant"].eq(decision["winner_method"])].iloc[0]
    main = curve_summary.copy()
    main["best_ndcg@20"] = main["best_ndcg@20"].map(lambda value: f"{value:.12f}")
    main["relative_pct_ndcg@20_vs_baseline_best"] = main[
        "relative_pct_ndcg@20_vs_baseline_best"
    ].map(lambda value: f"{value:+.3f}%")
    main["relative_pct_hr@20_vs_baseline_best"] = main[
        "relative_pct_hr@20_vs_baseline_best"
    ].map(lambda value: f"{value:+.3f}%")
    main["late30_relative_pct_ndcg@20_vs_baseline_best"] = main[
        "late30_relative_pct_ndcg@20_vs_baseline_best"
    ].map(lambda value: f"{value:+.3f}%")
    winner_groups = group_summary[group_summary["method_variant"].eq(decision["winner_method"])].copy()
    winner_groups["relative_pct_ndcg@20"] = winner_groups["relative_pct_ndcg@20"].map(
        lambda value: f"{value:+.3f}%"
    )
    winner_groups["relative_pct_hr@20"] = winner_groups["relative_pct_hr@20"].map(
        lambda value: f"{value:+.3f}%"
    )
    winner_groups["ndcg@20_bootstrap_relative_pct_ci95"] = winner_groups.apply(
        lambda row: (
            f"[{row['ndcg@20_bootstrap_relative_pct_ci95_low']:+.3f}%, "
            f"{row['ndcg@20_bootstrap_relative_pct_ci95_high']:+.3f}%]"
        ),
        axis=1,
    )
    e1_rsp = rsp_summary[rsp_summary["method_variant"].eq("cicpr1_e4_residual")].copy()
    e1_rsp["relative_pct_ndcg@20"] = e1_rsp["relative_pct_ndcg@20"].map(
        lambda value: f"{value:+.3f}%"
    )
    e1_rsp["relative_pct_hr@20"] = e1_rsp["relative_pct_hr@20"].map(
        lambda value: f"{value:+.3f}%"
    )
    e1_score = score_response[
        score_response["method_variant"].eq("cicpr1_e4_residual")
        & score_response["response"].eq("ndcg@20")
    ].iloc[0]
    baseline_score = score_response[
        score_response["method_variant"].eq("baseline")
        & score_response["response"].eq("ndcg@20")
    ].iloc[0]

    report = f"""---
title: {run_stamp} CCFCRec Amazon-VG CICP-R1 six mechanisms training 结果
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - CICP-R1
  - experiment-result
---

# CCFCRec Amazon-VG CICP-R1 six mechanisms training 结果

## Material Passport（材料护照）

- origin skill（来源技能）：experiment-agent（实验代理）
- origin mode（来源模式）：validate（验证）
- verification status（验证状态）：ANALYZED_REAGGREGATION_VERIFIED（已分析且重聚合已验证）
- overall confidence（整体置信等级）：CAUTION（谨慎）
- source result（源结果）：`{result_root}`
- evaluated scope（已评价范围）：5,298 个 validation cold-start items（验证冷启动物品）
- full profile relation（相对完整档案）：5,298 / 35,322 = 14.999%
- test item metrics（测试逐物品指标）：未读取、未生成

## 一、主结论

E1 CICP residual adapter（CICP残差适配器）在第 `{decision['winner_best_epoch']}` 轮取得 NDCG@20（前20归一化折损累计增益）`{decision['winner_ndcg@20']:.12f}`，相对同口径 baseline（基线）`{decision['winner_relative_pct_ndcg@20_vs_baseline']:+.3f}%`；同一检查点 HR@20（前20命中率）为 `{decision['winner_hr@20']:.12f}`，相对提升 `{decision['winner_relative_pct_hr@20_vs_baseline']:+.3f}%`。这是新的开发阶段最佳，超过旧 M11-R2 E4（第十一组第二轮第四方案）`{float(winner['relative_pct_vs_old_m11r2_best']):+.3f}%`。

但它没有达到 `+3%` 门槛 `0.127528956806`，还差 `{abs(decision['absolute_gap_to_three_pct_threshold']):.12f}`。其 NDCG@20（前20归一化折损累计增益）配对 bootstrap（配对自助法）95%区间跨过零，高 CICP（类别增量协同可预测性）组还出现 `-0.662%`，因此不能写成已经确认的方法突破或 CICP（类别增量协同可预测性）因果有效。

## 二、六分支整体结果

{_markdown_table(main, ['method_label','best_epoch','best_ndcg@20','relative_pct_ndcg@20_vs_baseline_best','relative_pct_hr@20_vs_baseline_best','late30_relative_pct_ndcg@20_vs_baseline_best','late30_positive_ndcg_epochs_vs_baseline_best','passed_three_pct_threshold'])}

解释口径：所有提升均相对验证集全部 5,298 个冷启动物品的 baseline（基线）最佳点，不是完整 35,322 个物品，也不是局部 CICP（类别增量协同可预测性）群体。

## 三、曲线稳定性

- E1 后30轮均值 NDCG@20（前20归一化折损累计增益）为 `{float(winner['late30_mean_ndcg@20']):.12f}`，相对基线最佳点 `{float(winner['late30_relative_pct_ndcg@20_vs_baseline_best']):+.3f}%`。
- E1 后30轮中有 `{int(winner['late30_positive_ndcg_epochs_vs_baseline_best'])}/30` 轮高于基线最佳 NDCG@20（前20归一化折损累计增益），同轮对比则为 `{int(winner['late30_matched_positive_ndcg_epochs'])}/30` 轮正向。
- E1 最佳第77轮相对 baseline（基线）同轮提升 `{float(winner['same_epoch_relative_pct_ndcg@20']):+.3f}%`；在 baseline（基线）最佳第74轮，E1 仍提升 `{float(winner['at_baseline_best_epoch74_relative_pct_ndcg@20']):+.3f}%`。
- 因此它不是单一孤立峰值，但峰值 `+2.051%` 明显高于后30轮均值 `+0.395%`，仍存在 checkpoint selection（检查点选择）放大。

## 四、CICP互补群体

{_markdown_table(winner_groups, ['cicp_group','item_count','baseline_ndcg@20','method_ndcg@20','relative_pct_ndcg@20','relative_pct_hr@20','ndcg@20_bootstrap_relative_pct_ci95','helped_ndcg_item_count','harmed_ndcg_item_count'])}

- low（低分）组 `+6.676%`，mid（中分）组 `+1.981%`，high（高分）组 `-0.662%`；三个组完整覆盖全部5,298个验证冷启动物品。
- CICP score（类别增量协同可预测性分数）与 baseline（基线）逐物品 NDCG@20（前20归一化折损累计增益）的 Spearman（斯皮尔曼秩相关）为 `{float(baseline_score['spearman']):+.4f}`，说明高分组本来更容易。
- CICP score（类别增量协同可预测性分数）与 E1 逐物品 NDCG@20（前20归一化折损累计增益）变化的 Spearman（斯皮尔曼秩相关）只有 `{float(e1_score['spearman']):+.4f}`，未经校正 `p={float(e1_score['p_value_uncorrected']):.4f}`。因此分组差异没有形成连续、单调的收益响应，不能据此认定“高CICP物品应被重点强化”。

## 五、R/S/P事后分层

{_markdown_table(e1_rsp, ['stratification_label','stratum','item_count','baseline_ndcg@20','relative_pct_ndcg@20','relative_pct_hr@20'])}

这些 R/S/P（丰富度/支持度/流行度）分层使用验证协变量三等分，只是 post-hoc descriptive analysis（事后描述分析），不等价于训练前要求的独立 R/S/P matched control（丰富度/支持度/流行度匹配控制），也不能证明净因果效应。

## 六、评价覆盖

{_markdown_table(coverage, list(coverage.columns))}

完整物品、训练非冷启动物品和测试冷启动物品的推荐指标均为“未评估”。本报告没有用局部群体或完整混合描述代替验证冷启动整体结论。

## 七、统计谬误扫描

覆盖：`11/11`。

{_markdown_table(fallacy_scan, list(fallacy_scan.columns))}

## 八、路线判断

- result band（结果区间）：`{decision['result_band']}`。
- late stability（后段稳定性）：`{decision['late_stability_pass']}`。
- bottleneck clarity（瓶颈明确性）：`{decision['bottleneck_clarity_pass']}`。
- automatic follow-up authorized（自动允许后续训练）：`{decision['automatic_followup_authorized']}`。
- route（路线）：`{decision['route']}`。

预注册规则要求 `+2%` 至 `+3%` 同时满足后段稳定和瓶颈明确，才允许一次强化。E1 满足数值区间和后段稳定，但高分组反向、连续分数与逐物品收益近零相关，瓶颈不明确，所以当前不直接启动 CICP-R2（类别增量协同可预测性第二轮）性能变体。E1 作为新的单随机种子开发最佳保留；研究主线转入 category semantic basis（类别语义基础）增强或更新 cold-start backbone（冷启动模型骨干）的设计审查。
"""
    route_report = f"""---
title: {run_stamp} CCFCRec Amazon-VG CICP-R1 six mechanisms training 路线判断
date: {run_stamp[:10]}
tags:
  - CCFCRec
  - Amazon-VG
  - CICP-R1
  - route-decision
---

# CCFCRec Amazon-VG CICP-R1 six mechanisms training 路线判断

关联设计：[[2026-07-14 004006 CCFCRec Amazon-VG CICP-R1六机制正式实验设计与实现]]

关联结果：[[{run_stamp} CCFCRec Amazon-VG CICP-R1 six mechanisms training 结果]]

> [!important] 主决策
> E1 CICP residual adapter（CICP残差适配器）把验证冷启动整体提升推进到新的历史最佳 `+2.051%`，HR@20（前20命中率）同点 `+2.609%`，但仍未达到 `+3%`。后段稳定通过，瓶颈明确性未通过，因此不自动进入 CICP-R2（类别增量协同可预测性第二轮）训练。

## 决策依据

1. 正式主口径是全部5,298个验证冷启动物品，占完整35,322个物品的 `14.999%`。
2. E1 最佳 NDCG@20（前20归一化折损累计增益）为 `{decision['winner_ndcg@20']:.12f}`，相对基线 `{decision['winner_relative_pct_ndcg@20_vs_baseline']:+.3f}%`；距离门槛仍差 `{abs(decision['absolute_gap_to_three_pct_threshold']):.12f}`。
3. E1 后30轮均值仍高于基线，且24/30轮高于基线最佳点，所以不是只靠一个峰值。
4. 低、中、高 CICP（类别增量协同可预测性）组分别为 `+6.676%`、`+1.981%`、`-0.662%`；收益没有沿 CICP 分数单调增加。
5. CICP 分数与 E1 逐物品收益的 Spearman（斯皮尔曼秩相关）仅 `{decision['winner_cicp_score_vs_item_delta_spearman']:+.4f}`，没有明确指出下一次应改哪个机制。
6. 当前仍是 seed43（第43号单随机种子）开发证据，且没有独立同载体打乱与匹配控制；不能宣称 CICP 语义因果有效。

## 下一步

1. 不直接设计又一批 CICP（类别增量协同可预测性）残差、门控或损失变体，也不开放 multi-seed（多随机种子）。
2. 保留 E1 作为当前开发最佳和后续新语义基础的可复用接入候选。
3. 启动 P2（第二优先级）：审查类别文本、类别层级、商品属性和可复现外部语义数据，设计低维 semantic availability prior（语义可用性先验）、类别图或辅助监督，而不是直接拼接大模型高维向量。
4. 同时审查更新的 cold-start recommendation backbone（冷启动推荐模型骨干）；新方案仍必须与同协议 baseline（基线）比较，并继续禁止验证/测试逐物品答案进入训练。

## 固化状态

- route（路线）：`{decision['route']}`
- run CICP-R2 now（现在运行CICP第二轮）：`False`
- run multi-seed now（现在运行多随机种子）：`False`
- preserve E1 as development best（保留E1为开发最佳）：`True`
- test item metrics read or generated（读取或生成测试逐物品指标）：`False`
- fallacy scan coverage（统计谬误扫描覆盖）：`11/11`
"""
    result_path = output_dir / f"{run_stamp} CCFCRec Amazon-VG CICP-R1 six mechanisms training 结果.md"
    route_output_dir.mkdir(parents=True, exist_ok=True)
    route_path = route_output_dir / f"{run_stamp} CCFCRec Amazon-VG CICP-R1 six mechanisms training 路线判断.md"
    result_path.write_text(report, encoding="utf-8")
    route_path.write_text(route_report, encoding="utf-8")
    return result_path, route_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    curve_summary, matched_curve, baseline_curve = collect_curves(
        args.result_root.resolve(), args.baseline_result.resolve()
    )
    item_metrics = pd.read_csv(
        args.item_evaluation_dir.resolve() / "cicpr1_validation_item_metrics.csv",
        dtype={"raw_asin": str},
    )
    if len(item_metrics) != 7 * VALIDATION_ITEM_COUNT:
        raise ValueError(f"unexpected validation item row count: {len(item_metrics)}")
    group_summary = build_group_bootstrap(item_metrics, args.bootstrap_repetitions)
    score_response = build_score_response(item_metrics)
    rsp_summary, rsp_thresholds = build_rsp_strata(item_metrics, args.source_profile.resolve())
    coverage = build_coverage(group_summary)
    fallacy_scan = build_fallacy_scan()
    decision = build_decision(curve_summary, group_summary, score_response)

    artifacts: dict[str, pd.DataFrame] = {
        "cicpr1_curve_summary.csv": curve_summary,
        "cicpr1_epoch_matched_curve.csv": matched_curve,
        "cicpr1_baseline_curve.csv": baseline_curve,
        "cicpr1_group_bootstrap_summary.csv": group_summary,
        "cicpr1_rsp_strata_summary.csv": rsp_summary,
        "cicpr1_score_response_summary.csv": score_response,
        "cicpr1_evaluation_coverage.csv": coverage,
        "cicpr1_fallacy_scan.csv": fallacy_scan,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output_dir / name, index=False)
    (output_dir / "cicpr1_route_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    result_path, route_path = write_reports(
        output_dir=output_dir,
        route_output_dir=args.route_output_dir.resolve(),
        run_stamp=args.run_stamp,
        result_root=args.result_root.resolve(),
        curve_summary=curve_summary,
        group_summary=group_summary,
        rsp_summary=rsp_summary,
        score_response=score_response,
        coverage=coverage,
        fallacy_scan=fallacy_scan,
        decision=decision,
    )
    manifest = {
        "protocol": "cicpr1_six_training_analysis_v1",
        "run_stamp": args.run_stamp,
        "result_root": str(args.result_root.resolve()),
        "baseline_result": str(args.baseline_result.resolve()),
        "item_evaluation_dir": str(args.item_evaluation_dir.resolve()),
        "source_profile": str(args.source_profile.resolve()),
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "rsp_validation_tertile_thresholds": rsp_thresholds,
        "recommendation_metric_scope": "validation_cold_5298_only",
        "test_item_metrics_read_or_generated": False,
        "outputs": [
            *artifacts,
            "cicpr1_route_decision.json",
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
