#!/usr/bin/env python3
"""Re-audit the M11 target signal without test-item performance diagnostics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr


PROJECT_ROOT = Path("/Users/luojiaqiang/Documents/Obsidian Vault/科研/CCFCRec对比学习思路")
DEFAULT_PROFILE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录/temp_20260710"
    / "2026-07-10 234142 m11r2-seven-run-design-audit"
    / "m11r2_seven_run_profile.csv"
)
DEFAULT_VALIDATION_ITEM_EVAL = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录/temp_20260713"
    / "2026-07-13 112318 m11r4-four-performance-first-analysis"
    / "m11r4_validation_item_eval.csv"
)
DEFAULT_TRAIN_SAFE_SOURCE = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录/temp_20260706"
    / "2026-07-06 004222 task4-pre3-train-safe-hard-proxy"
    / "task4_train_safe_hard_proxy_profile.csv"
)
DEFAULT_HISTORICAL_CANDIDATES = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录/temp_20260710"
    / "2026-07-10 021731 m11-target-construction-audit"
    / "m11_target_candidate_summary.csv"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "temp_202607_实验文件记录/temp_20260713"
    / "2026-07-13 113936 m11-signal-reaudit"
)

TARGET_COLUMN = "m11_high_acat_low_rsp_neighbor_support_flag"
ALLOWED_SPLITS = ("train", "validate")
SIGNALS = (
    "s_cat_v3",
    "RSP_score",
    "category_neighbor_mismatch_proxy_score",
    "support_tail_proxy_score",
    "m11_target_score",
)
CONFOUNDERS = (
    "s_cat_v3",
    "RSP_score",
    "support_tail_proxy_score",
    "category_neighbor_mismatch_proxy_score",
    "category_count",
    "A_collab_user_set_jaccard_mean",
    "A_collab_support_entropy_mean",
    "R_metadata_richness_score",
    "S_train_token_interaction_support_mean",
    "S_train_token_user_support_mean",
    "S_train_support_score",
    "P_popularity_score",
    "Acat_v3_disc_residual_pct",
    "Acat_v3_collab_residual_pct",
)
METHOD_BASELINE = "baseline"
METHOD_E4 = "m11r2_target_feature_fusion"
METHOD_E1 = "m11r4_protected_experts"
HISTORICAL_CANDIDATE = "high_acat_low_rsp_neighbor_support"
HISTORICAL_TEST_TARGET_COUNT = 988


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _bool(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].map(_truthy).fillna(False).astype(bool)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} missing columns: {missing}")


def load_profile(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"raw_asin": str})
    required = {
        "raw_asin",
        "split",
        "high_acat_flag",
        "RSP_group",
        "RSP_score",
        "support_tail_proxy_score",
        "support_tail_proxy_high_flag",
        "category_neighbor_mismatch_proxy_score",
        "category_neighbor_mismatch_proxy_high_flag",
        "train_safe_hard_proxy_score",
        TARGET_COLUMN,
        "m11_target_score",
        "s_cat_v3",
        "category_count",
    }
    _require_columns(frame, required, str(path))
    frame = frame[frame["split"].isin(ALLOWED_SPLITS)].copy()
    if frame["raw_asin"].duplicated().any():
        raise ValueError("clean profile contains duplicate train/validation raw_asin values")

    high_acat = _bool(frame, "high_acat_flag")
    not_rsp_high = ~frame["RSP_group"].astype(str).eq("RSP_high")
    neighbor = _bool(frame, "category_neighbor_mismatch_proxy_high_flag")
    support = _bool(frame, "support_tail_proxy_high_flag")
    expected_target = high_acat & not_rsp_high & neighbor & support
    observed_target = _bool(frame, TARGET_COLUMN)
    if not expected_target.equals(observed_target):
        raise ValueError("stored M11 target flag does not match its four-condition definition")

    expected_score = (
        0.45 * _numeric(frame, "s_cat_v3")
        + 0.35 * _numeric(frame, "train_safe_hard_proxy_score")
        + 0.20 * (1.0 - _numeric(frame, "RSP_score"))
    )
    if not np.allclose(expected_score.round(6), _numeric(frame, "m11_target_score"), rtol=0, atol=1e-12):
        raise ValueError("stored m11_target_score does not match its documented formula")
    if not np.allclose(
        _numeric(frame, "train_safe_hard_proxy_score"),
        _numeric(frame, "category_neighbor_mismatch_proxy_score"),
        rtol=0,
        atol=1e-12,
    ):
        raise ValueError("train-safe hard proxy is no longer identical to neighbor mismatch proxy")
    return frame


def component_masks(frame: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "high_acat": _bool(frame, "high_acat_flag"),
        "low_rsp_name_but_not_high": ~frame["RSP_group"].astype(str).eq("RSP_high"),
        "neighbor_high": _bool(frame, "category_neighbor_mismatch_proxy_high_flag"),
        "support_tail_high": _bool(frame, "support_tail_proxy_high_flag"),
        "target_flag": _bool(frame, TARGET_COLUMN),
    }


def build_composition(profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        frame = profile[profile["split"].eq(split)]
        masks = component_masks(frame)
        target = masks["target_flag"]
        target_rsp = frame.loc[target, "RSP_group"].value_counts()
        rows.append(
            {
                "split": split,
                "item_count": len(frame),
                "target_count": int(target.sum()),
                "target_share": target.mean(),
                "high_acat_share": masks["high_acat"].mean(),
                "not_rsp_high_share": masks["low_rsp_name_but_not_high"].mean(),
                "neighbor_high_share": masks["neighbor_high"].mean(),
                "support_tail_high_share": masks["support_tail_high"].mean(),
                "target_rsp_low_count": int(target_rsp.get("RSP_low", 0)),
                "target_rsp_mid_count": int(target_rsp.get("RSP_mid", 0)),
                "target_rsp_high_count": int(target_rsp.get("RSP_high", 0)),
                "target_score_mean": _numeric(frame, "m11_target_score").mean(),
                "target_score_std": _numeric(frame, "m11_target_score").std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def _mask_metrics(candidate: pd.Series, target: pd.Series) -> tuple[int, float, float, float]:
    selected = int(candidate.sum())
    intersection = int((candidate & target).sum())
    union = int((candidate | target).sum())
    precision = intersection / selected if selected else float("nan")
    recall = intersection / int(target.sum()) if target.any() else float("nan")
    jaccard = intersection / union if union else float("nan")
    return selected, precision, recall, jaccard


def build_component_ablation(profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        frame = profile[profile["split"].eq(split)]
        m = component_masks(frame)
        h, r, n, s, target = (
            m["high_acat"],
            m["low_rsp_name_but_not_high"],
            m["neighbor_high"],
            m["support_tail_high"],
            m["target_flag"],
        )
        candidates = {
            "high_acat_and_support": ("high_acat & support_tail_high", h & s),
            "high_acat_neighbor_support": ("high_acat & neighbor_high & support_tail_high", h & n & s),
            "high_acat_notrsphigh_support": (
                "high_acat & low_rsp_name_but_not_high & support_tail_high",
                h & r & s,
            ),
            "full_target": (
                "high_acat & low_rsp_name_but_not_high & neighbor_high & support_tail_high",
                h & r & n & s,
            ),
            "leave_out_high_acat": (
                "low_rsp_name_but_not_high & neighbor_high & support_tail_high",
                r & n & s,
            ),
            "leave_out_low_rsp_name_but_not_high": (
                "high_acat & neighbor_high & support_tail_high",
                h & n & s,
            ),
            "leave_out_neighbor_high": (
                "high_acat & low_rsp_name_but_not_high & support_tail_high",
                h & r & s,
            ),
            "leave_out_support_tail_high": (
                "high_acat & low_rsp_name_but_not_high & neighbor_high",
                h & r & n,
            ),
        }
        for name, (conditions, candidate) in candidates.items():
            selected, precision, recall, jaccard = _mask_metrics(candidate, target)
            rows.append(
                {
                    "split": split,
                    "candidate": name,
                    "conditions": conditions,
                    "selected_count": selected,
                    "target_count": int(target.sum()),
                    "precision_vs_full_target": precision,
                    "recall_vs_full_target": recall,
                    "jaccard_vs_full_target": jaccard,
                }
            )
    return pd.DataFrame(rows)


def build_component_phi(profile: pd.DataFrame) -> pd.DataFrame:
    frame = profile[profile["split"].eq("validate")]
    masks = pd.DataFrame(component_masks(frame), dtype=float)
    corr = masks.corr()
    return pd.DataFrame(
        [
            {"left": left, "right": right, "phi_correlation": corr.loc[left, right]}
            for left in corr.index
            for right in corr.columns
        ]
    )


def _auc_for_binary_target(target: pd.Series, score: pd.Series) -> float:
    target = target.astype(bool)
    positives = int(target.sum())
    negatives = int((~target).sum())
    if not positives or not negatives:
        return float("nan")
    ranks = score.rank(method="average")
    return float((ranks[target].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def build_distribution_stability(profile: pd.DataFrame) -> pd.DataFrame:
    train = profile[profile["split"].eq("train")]
    validate = profile[profile["split"].eq("validate")]
    target = _bool(validate, TARGET_COLUMN).astype(int)
    rows: list[dict[str, Any]] = []
    for signal in SIGNALS:
        train_values = _numeric(train, signal)
        validate_values = _numeric(validate, signal)
        ks = ks_2samp(train_values, validate_values)
        rows.append(
            {
                "signal": signal,
                "train_mean": train_values.mean(),
                "validate_mean": validate_values.mean(),
                "mean_shift": validate_values.mean() - train_values.mean(),
                "ks_statistic": ks.statistic,
                "ks_pvalue": ks.pvalue,
                "spearman_vs_validate_target_flag": spearmanr(validate_values, target).statistic,
            }
        )
    rows.append(
        {
            "signal": "m11_target_score_auc_for_target_membership",
            "train_mean": np.nan,
            "validate_mean": _auc_for_binary_target(target.astype(bool), _numeric(validate, "m11_target_score")),
            "mean_shift": np.nan,
            "ks_statistic": np.nan,
            "ks_pvalue": np.nan,
            "spearman_vs_validate_target_flag": np.nan,
        }
    )
    return pd.DataFrame(rows)


def load_validation_benefit(path: Path, profile: pd.DataFrame) -> pd.DataFrame:
    item_eval = pd.read_csv(path, dtype={"raw_asin": str})
    _require_columns(
        item_eval,
        {"method_variant", "split", "raw_asin", "target_flag", "ndcg@20"},
        str(path),
    )
    if not item_eval["split"].eq("validate").all():
        raise ValueError("benefit audit accepts validation item rows only")
    wanted = item_eval[item_eval["method_variant"].isin({METHOD_BASELINE, METHOD_E4, METHOD_E1})]
    pivot = wanted.pivot(index="raw_asin", columns="method_variant", values="ndcg@20").reset_index()
    _require_columns(pivot, {METHOD_BASELINE, METHOD_E4, METHOD_E1}, "validation item pivot")
    validate_profile = profile[profile["split"].eq("validate")].copy()
    merged = validate_profile.merge(pivot, on="raw_asin", how="inner", validate="one_to_one")
    if len(merged) != len(validate_profile):
        raise ValueError("validation item evaluation does not cover every validation profile item")
    merged["baseline_ndcg20"] = _numeric(merged, METHOD_BASELINE)
    merged["e4_delta"] = _numeric(merged, METHOD_E4) - merged["baseline_ndcg20"]
    merged["e1_delta"] = _numeric(merged, METHOD_E1) - merged["baseline_ndcg20"]
    merged["target_flag"] = _bool(merged, TARGET_COLUMN)
    return merged


def _benefit_groups(frame: pd.DataFrame) -> tuple[tuple[str, pd.DataFrame], ...]:
    return (
        ("validation_all", frame),
        ("validation_target", frame[frame["target_flag"]]),
        ("validation_non_target", frame[~frame["target_flag"]]),
    )


def build_benefit_correlation(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, part in _benefit_groups(frame):
        for signal in ("m11_target_score", *SIGNALS[:-1]):
            for outcome in ("e4_delta", "e1_delta"):
                rows.append(
                    {
                        "group": group,
                        "item_count": len(part),
                        "signal": signal,
                        "outcome": outcome,
                        "spearman": spearmanr(_numeric(part, signal), _numeric(part, outcome)).statistic,
                    }
                )
    return pd.DataFrame(rows)


def build_group_performance(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, part in _benefit_groups(frame):
        rows.append(
            {
                "group": group,
                "item_count": len(part),
                "baseline_ndcg20": part["baseline_ndcg20"].mean(),
                "e4_delta_mean": part["e4_delta"].mean(),
                "e1_delta_mean": part["e1_delta"].mean(),
                "e4_improved_share": part["e4_delta"].gt(0).mean(),
                "e4_harmed_share": part["e4_delta"].lt(0).mean(),
                "e1_improved_share": part["e1_delta"].gt(0).mean(),
                "e1_harmed_share": part["e1_delta"].lt(0).mean(),
            }
        )
    return pd.DataFrame(rows)


def build_score_bin_performance(frame: pd.DataFrame, train_scores: pd.Series) -> pd.DataFrame:
    work = frame.copy()
    train_quantiles = pd.Series(train_scores).quantile(np.linspace(0.0, 1.0, 11)).drop_duplicates()
    work["score_bin"] = pd.cut(
        work["m11_target_score"], bins=train_quantiles, include_lowest=True, duplicates="drop"
    )
    rows: list[dict[str, Any]] = []
    for score_bin, part in work.groupby("score_bin", observed=True, sort=True):
        rows.append(
            {
                "score_bin": str(score_bin),
                "item_count": len(part),
                "target_share": part["target_flag"].mean(),
                "baseline_ndcg20": part["baseline_ndcg20"].mean(),
                "e4_delta_mean": part["e4_delta"].mean(),
                "e1_delta_mean": part["e1_delta"].mean(),
                "e4_improved_share": part["e4_delta"].gt(0).mean(),
                "e4_harmed_share": part["e4_delta"].lt(0).mean(),
            }
        )
    return pd.DataFrame(rows)


def load_structural_source(path: Path) -> pd.DataFrame:
    usecols = ["raw_asin", "split", *[c for c in CONFOUNDERS if c not in {"s_cat_v3", "RSP_score", "support_tail_proxy_score", "category_neighbor_mismatch_proxy_score", "category_count"}]]
    frame = pd.read_csv(path, usecols=usecols, dtype={"raw_asin": str})
    frame = frame[frame["split"].isin(ALLOWED_SPLITS)].copy()
    if frame["raw_asin"].duplicated().any():
        raise ValueError("train-safe source contains duplicate train/validation raw_asin values")
    return frame.drop(columns="split")


def _standardized_mean_difference(target: pd.Series, non_target: pd.Series) -> float:
    pooled = math.sqrt((target.var(ddof=1) + non_target.var(ddof=1)) / 2.0)
    return float((target.mean() - non_target.mean()) / pooled) if pooled else float("nan")


def build_confounding(profile: pd.DataFrame, structural: pd.DataFrame) -> pd.DataFrame:
    merged = profile.merge(structural, on="raw_asin", how="left", validate="one_to_one")
    rows: list[dict[str, Any]] = []
    for split in ALLOWED_SPLITS:
        part = merged[merged["split"].eq(split)]
        target_flag = _bool(part, TARGET_COLUMN)
        for variable in CONFOUNDERS:
            values = _numeric(part, variable)
            target = values[target_flag].dropna()
            non_target = values[~target_flag].dropna()
            rows.append(
                {
                    "split": split,
                    "variable": variable,
                    "target_mean": target.mean(),
                    "non_target_mean": non_target.mean(),
                    "standardized_mean_difference": _standardized_mean_difference(target, non_target),
                }
            )
    return pd.DataFrame(rows)


def build_headroom_realization(frame: pd.DataFrame) -> pd.DataFrame:
    baseline = frame["baseline_ndcg20"].mean()
    target = frame["target_flag"]
    zero_target_count = int((target & frame["baseline_ndcg20"].eq(0)).sum())
    single_hit_floor = 1.0 / math.log2(21)
    upper_gain = zero_target_count * single_hit_floor / len(frame)
    e4_contribution = frame.loc[target, "e4_delta"].sum() / len(frame)
    e1_contribution = frame.loc[target, "e1_delta"].sum() / len(frame)
    return pd.DataFrame(
        [
            {
                "validation_item_count": len(frame),
                "target_count": int(target.sum()),
                "single_hit_floor": single_hit_floor,
                "single_hit_upper_gain_abs_overall": upper_gain,
                "single_hit_upper_gain_pct_vs_baseline": upper_gain / baseline * 100.0,
                "e4_target_contribution_abs_overall": e4_contribution,
                "e4_headroom_capture_fraction": e4_contribution / upper_gain,
                "e1_target_contribution_abs_overall": e1_contribution,
                "e1_headroom_capture_fraction": e1_contribution / upper_gain,
            }
        ]
    )


def build_provenance_audit(path: Path, validation_target_count: int) -> pd.DataFrame:
    historical = pd.read_csv(path)
    _require_columns(historical, {"candidate", "selected_count", "selected_baseline_ndcg@20_mean"}, str(path))
    row = historical[historical["candidate"].eq(HISTORICAL_CANDIDATE)]
    if len(row) != 1:
        raise ValueError("historical candidate summary must contain exactly one selected M11 candidate")
    selected_count = int(row.iloc[0]["selected_count"])
    baseline_mean = float(row.iloc[0]["selected_baseline_ndcg@20_mean"])
    count_assessment = (
        f"matches historical test target count {HISTORICAL_TEST_TARGET_COUNT}, "
        f"not current validation target count {validation_target_count}"
        if selected_count == HISTORICAL_TEST_TARGET_COUNT
        else f"does not match expected historical test target count {HISTORICAL_TEST_TARGET_COUNT}"
    )
    return pd.DataFrame(
        [
            {
                "check": "historical_candidate_selected_count",
                "value": selected_count,
                "assessment": count_assessment,
            },
            {
                "check": "historical_candidate_baseline_mean",
                "value": baseline_mean,
                "assessment": "candidate ranking used item-level baseline outcomes in historical scope",
            },
            {"check": "current_validation_target_count", "value": validation_target_count, "assessment": validation_target_count},
            {
                "check": "historical_test_informed_provenance",
                "value": True,
                "assessment": "cannot be erased by later clean training inputs",
            },
            {
                "check": "current_training_input_uses_eval_metrics",
                "value": False,
                "assessment": "clean M11-R2/R3/R4 profiles exclude evaluation-result columns",
            },
            {
                "check": "historical_test_informed_aggregate_read_for_provenance",
                "value": True,
                "assessment": "aggregate only; no test item rows were loaded or used for a new performance diagnostic",
            },
        ]
    )


def route_decision() -> dict[str, Any]:
    return {
        "route": "retire_m11_signal_as_primary_keep_diagnostic_subgroup",
        "primary_signal_status": "retire",
        "diagnostic_subgroup_status": "retain",
        "continuous_m11_target_score_status": "do_not_continue_as_primary",
        "reason_codes": [
            "historical_selection_used_test_item_outcomes",
            "four_way_name_is_effectively_high_acat_and_support_tail",
            "low_rsp_and_neighbor_conditions_are_nearly_redundant",
            "target_score_replicates_membership_but_not_method_benefit",
            "single_hit_headroom_realization_below_5pct",
            "multiple_distinct_carriers_failed_overall_2pct_gate",
        ],
        "run_training_now": False,
        "run_multi_seed_now": False,
        "construct_new_signal_next": True,
        "new_signal_must_be_train_only": True,
        "new_signal_must_cover_most_cold_items": True,
        "test_items_analyzed": 0,
        "test_item_level_metrics_read_or_generated": False,
        "historical_test_informed_aggregate_read_for_provenance": True,
        "test_used_for_new_signal_or_performance_diagnostic": False,
    }


def write_outputs(
    output_dir: Path,
    profile_path: Path,
    validation_path: Path,
    structural_path: Path,
    historical_path: Path,
) -> None:
    profile = load_profile(profile_path)
    benefit = load_validation_benefit(validation_path, profile)
    structural = load_structural_source(structural_path)
    outputs = {
        "m11_signal_composition.csv": build_composition(profile),
        "m11_signal_component_ablation.csv": build_component_ablation(profile),
        "m11_signal_component_phi.csv": build_component_phi(profile),
        "m11_signal_distribution_stability.csv": build_distribution_stability(profile),
        "m11_signal_benefit_correlation.csv": build_benefit_correlation(benefit),
        "m11_signal_group_performance.csv": build_group_performance(benefit),
        "m11_signal_score_bin_performance.csv": build_score_bin_performance(
            benefit, profile.loc[profile["split"].eq("train"), "m11_target_score"]
        ),
        "m11_signal_confounding.csv": build_confounding(profile, structural),
        "m11_signal_headroom_realization.csv": build_headroom_realization(benefit),
        "m11_signal_provenance_audit.csv": build_provenance_audit(
            historical_path, int(benefit["target_flag"].sum())
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)

    decision = route_decision()
    (output_dir / "m11_signal_route_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "analysis_scope": [
            "train_structural",
            "validation_structural",
            "validation_existing_method_benefit_audit",
            "historical_test_informed_aggregate_provenance_audit",
        ],
        "excluded_scope": ["test_item_rows", "test_item_level_metrics", "complete_profile_mixed_metric"],
        "complete_profile_item_count": 35322,
        "train_items_analyzed": int(profile["split"].eq("train").sum()),
        "validation_items_analyzed": int(profile["split"].eq("validate").sum()),
        "test_items_analyzed": 0,
        "test_item_level_metrics_read_or_generated": False,
        "historical_test_informed_aggregate_read_for_provenance": True,
        "test_used_for_new_signal_or_performance_diagnostic": False,
        "new_signal_constructed": False,
        "inputs": {
            "clean_profile": str(profile_path),
            "validation_item_eval": str(validation_path),
            "train_safe_source_columns": str(structural_path),
            "historical_candidate_aggregate_summary": str(historical_path),
        },
        "outputs": [*outputs, "m11_signal_route_decision.json", "run_manifest.json"],
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--validation-item-eval", type=Path, default=DEFAULT_VALIDATION_ITEM_EVAL)
    parser.add_argument("--train-safe-source", type=Path, default=DEFAULT_TRAIN_SAFE_SOURCE)
    parser.add_argument("--historical-candidates", type=Path, default=DEFAULT_HISTORICAL_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_outputs(
        args.output_dir,
        args.profile,
        args.validation_item_eval,
        args.train_safe_source,
        args.historical_candidates,
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
