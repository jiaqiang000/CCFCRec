import math

import pandas as pd

from analyze_amazon_vg_task4_post1_failure_layer import (
    attach_detail_groups,
    build_baseline_failure_layer_summary,
    build_candidate_layer_recommendation,
    build_m3_m6_delta_profile,
    build_trainhard_layer_delta,
)


def test_candidate_layer_recommendation_selects_stable_high_detail_layer() -> None:
    delta = pd.DataFrame(
        [
            {"split": "validate", "detail_group": "high_detail", "high_acat_train_safe_hard_flag": "True", "item_count": 100, "m3_minus_m6_ndcg@20_mean": 0.0020, "m3_minus_m6_hr@20_mean": 0.0010},
            {"split": "test", "detail_group": "high_detail", "high_acat_train_safe_hard_flag": "True", "item_count": 100, "m3_minus_m6_ndcg@20_mean": 0.0015, "m3_minus_m6_hr@20_mean": 0.0005},
            {"split": "validate", "detail_group": "low_detail", "high_acat_train_safe_hard_flag": "True", "item_count": 100, "m3_minus_m6_ndcg@20_mean": 0.0020, "m3_minus_m6_hr@20_mean": 0.0010},
            {"split": "test", "detail_group": "low_detail", "high_acat_train_safe_hard_flag": "True", "item_count": 100, "m3_minus_m6_ndcg@20_mean": -0.0010, "m3_minus_m6_hr@20_mean": 0.0001},
        ]
    )

    rec = build_candidate_layer_recommendation(delta, min_item_count=50, min_ndcg_delta=0.0005)
    top = rec.sort_values(["recommended_target_layer", "target_score"], ascending=[False, False]).iloc[0]

    assert top["detail_group"] == "high_detail"
    assert bool(top["recommended_target_layer"]) is True
    assert math.isclose(top["test_m3_minus_m6_ndcg@20_mean"], 0.0015)


def test_delta_profile_pairs_m3_and_shuffle_on_same_item() -> None:
    item_eval = pd.DataFrame(
        [
            {"split": "test", "raw_asin": "a", "method_variant": "task4_acat_trainhard_weight", "ndcg@20": 0.20, "hr@20": 0.02, "high_acat_train_safe_hard_flag": True, "high_acat_flag": True, "eval_baseline_hard_flag": True},
            {"split": "test", "raw_asin": "a", "method_variant": "task4_acat_shuffle_high_weight", "ndcg@20": 0.15, "hr@20": 0.01, "high_acat_train_safe_hard_flag": True, "high_acat_flag": True, "eval_baseline_hard_flag": True},
        ]
    )

    delta = build_m3_m6_delta_profile(item_eval)
    row = delta.iloc[0]

    assert row["raw_asin"] == "a"
    assert math.isclose(row["delta_ndcg@20"], 0.05)
    assert math.isclose(row["delta_hr@20"], 0.01)


def test_trainhard_layer_delta_groups_by_detail_and_trainhard_flag() -> None:
    delta = pd.DataFrame(
        [
            {"split": "test", "raw_asin": "a", "detail_group": "high_detail", "detail_order": 3, "high_acat_train_safe_hard_flag": True, "delta_ndcg@20": 0.04, "delta_hr@20": 0.01},
            {"split": "test", "raw_asin": "b", "detail_group": "high_detail", "detail_order": 3, "high_acat_train_safe_hard_flag": True, "delta_ndcg@20": 0.00, "delta_hr@20": 0.00},
            {"split": "test", "raw_asin": "c", "detail_group": "high_detail", "detail_order": 3, "high_acat_train_safe_hard_flag": False, "delta_ndcg@20": -0.02, "delta_hr@20": 0.00},
        ]
    )

    summary = build_trainhard_layer_delta(delta)
    target = summary[summary["high_acat_train_safe_hard_flag"].eq("True")].iloc[0]

    assert target["item_count"] == 2
    assert math.isclose(target["m3_minus_m6_ndcg@20_mean"], 0.02)
    assert math.isclose(target["positive_delta_ndcg_rate"], 0.5)


def test_trainhard_layer_delta_leaves_baseline_hard_rate_empty_without_baseline_metric() -> None:
    delta = pd.DataFrame(
        [
            {"split": "validate", "raw_asin": "a", "detail_group": "high_detail", "detail_order": 3, "high_acat_train_safe_hard_flag": True, "eval_baseline_hard_flag": False, "delta_ndcg@20": 0.04, "delta_hr@20": 0.01, "baseline_ndcg@20": float("nan")},
        ]
    )

    summary = build_trainhard_layer_delta(delta)
    row = summary.iloc[0]

    assert math.isnan(row["baseline_hard_rate"])


def test_baseline_failure_layer_summary_reports_hard_rate_after_detail_merge() -> None:
    item_eval = pd.DataFrame(
        [
            {"split": "test", "raw_asin": "a", "method_variant": "task4_acat_trainhard_weight", "baseline_ndcg@20": 0.0, "high_acat_flag": True, "eval_baseline_hard_flag": True, "high_acat_train_safe_hard_flag": True},
            {"split": "test", "raw_asin": "b", "method_variant": "task4_acat_trainhard_weight", "baseline_ndcg@20": 0.2, "high_acat_flag": True, "eval_baseline_hard_flag": False, "high_acat_train_safe_hard_flag": False},
        ]
    )
    availability = pd.DataFrame(
        [
            {"split": "test", "raw_asin": "a", "cat_count_bin": "cat_count_5_plus"},
            {"split": "test", "raw_asin": "b", "cat_count_bin": "cat_count_5_plus"},
        ]
    )

    merged = attach_detail_groups(item_eval, availability)
    summary = build_baseline_failure_layer_summary(merged)
    row = summary.iloc[0]

    assert row["detail_group"] == "high_detail"
    assert row["item_count"] == 2
    assert math.isclose(row["baseline_hard_rate"], 0.5)
