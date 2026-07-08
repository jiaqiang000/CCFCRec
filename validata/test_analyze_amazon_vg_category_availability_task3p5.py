#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.5 机制锁定诊断测试。
"""

import json
import math

import pandas as pd

from analyze_amazon_vg_category_availability_task3p5 import (
    build_group_mechanism_summary,
    build_item_delta_profile,
    build_route_decision,
    build_weak_bucket_summary,
    md_table,
)


def test_build_item_delta_profile_merges_sources_and_computes_deltas() -> None:
    item_profile = pd.DataFrame(
        [
            {"asin": "a", "category_group": "s_cat_v2_weak", "s_cat": 0.1},
            {"asin": "b", "category_group": "s_cat_v2_strong", "s_cat": 0.9},
        ]
    )
    baseline_score = pd.DataFrame(
        [
            {"asin": "a", "category_group": "s_cat_v2_weak", "hr@20": 0.0, "ndcg@20": 0.0, "q_norm": 10.0, "top20_user_norm_mean": 0.20, "target_score_max": 2.0, "margin_to_top20_cutoff": -1.0, "best_target_rank": 30.0},
            {"asin": "b", "category_group": "s_cat_v2_strong", "hr@20": 0.2, "ndcg@20": 0.3, "q_norm": 12.0, "top20_user_norm_mean": 0.25, "target_score_max": 3.0, "margin_to_top20_cutoff": -0.5, "best_target_rank": 10.0},
        ]
    )
    category_score = pd.DataFrame(
        [
            {"asin": "a", "category_group": "s_cat_v2_weak", "hr@20": 0.1, "ndcg@20": 0.2, "q_norm": 8.0, "top20_user_norm_mean": 0.30, "target_score_max": 1.5, "margin_to_top20_cutoff": -0.8, "best_target_rank": 20.0},
            {"asin": "b", "category_group": "s_cat_v2_strong", "hr@20": 0.1, "ndcg@20": 0.1, "q_norm": 9.0, "top20_user_norm_mean": 0.20, "target_score_max": 2.8, "margin_to_top20_cutoff": -0.6, "best_target_rank": 12.0},
        ]
    )
    baseline_target = pd.DataFrame(
        [
            {"asin": "a", "target_cosine_at_score_max": 0.20, "target_minus_top20_cosine_mean": -0.10, "target_minus_top20_user_norm_mean": -0.20},
            {"asin": "b", "target_cosine_at_score_max": 0.30, "target_minus_top20_cosine_mean": -0.20, "target_minus_top20_user_norm_mean": -0.10},
        ]
    )
    category_target = pd.DataFrame(
        [
            {"asin": "a", "target_cosine_at_score_max": 0.25, "target_minus_top20_cosine_mean": -0.05, "target_minus_top20_user_norm_mean": -0.30},
            {"asin": "b", "target_cosine_at_score_max": 0.32, "target_minus_top20_cosine_mean": -0.25, "target_minus_top20_user_norm_mean": -0.20},
        ]
    )
    baseline_content = pd.DataFrame(
        [
            {"asin": "a", "target_history_q_cosine_mean": 0.40, "top20_history_q_cosine_mean": 0.50, "target_minus_top20_history_q_cosine_mean": -0.10, "target_history_attr_cosine_mean": 0.30, "target_history_img_cosine_mean": 0.20, "top20_history_interaction_count_mean": 100.0},
            {"asin": "b", "target_history_q_cosine_mean": 0.45, "top20_history_q_cosine_mean": 0.55, "target_minus_top20_history_q_cosine_mean": -0.10, "target_history_attr_cosine_mean": 0.35, "target_history_img_cosine_mean": 0.25, "top20_history_interaction_count_mean": 120.0},
        ]
    )
    category_content = pd.DataFrame(
        [
            {"asin": "a", "target_history_q_cosine_mean": 0.45, "top20_history_q_cosine_mean": 0.60, "target_minus_top20_history_q_cosine_mean": -0.15, "target_history_attr_cosine_mean": 0.25, "target_history_img_cosine_mean": 0.15, "top20_history_interaction_count_mean": 130.0},
            {"asin": "b", "target_history_q_cosine_mean": 0.47, "top20_history_q_cosine_mean": 0.58, "target_minus_top20_history_q_cosine_mean": -0.11, "target_history_attr_cosine_mean": 0.33, "target_history_img_cosine_mean": 0.20, "top20_history_interaction_count_mean": 110.0},
        ]
    )

    profile = build_item_delta_profile(
        item_profile=item_profile,
        baseline_score=baseline_score,
        category_score=category_score,
        baseline_target=baseline_target,
        category_target=category_target,
        baseline_content=baseline_content,
        category_content=category_content,
    )

    weak = profile[profile["asin"].eq("a")].iloc[0]
    assert math.isclose(weak["delta_ndcg@20"], 0.2)
    assert math.isclose(weak["delta_q_norm"], -2.0)
    assert math.isclose(weak["delta_top20_user_norm_mean"], 0.10)
    assert math.isclose(weak["delta_top20_history_interaction_count_mean"], 30.0)
    assert math.isclose(weak["delta_target_minus_top20_history_q_cosine_mean"], -0.05)
    assert weak["norm_activity_pressure_score"] > 0


def test_group_summary_and_weak_buckets_expose_ranking_failure_pattern() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "category_group": "s_cat_v2_weak", "delta_ndcg@20": -0.10, "delta_hr@20": -0.05, "delta_top20_user_norm_mean": 0.20, "delta_top20_history_interaction_count_mean": 30.0, "delta_target_minus_top20_history_q_cosine_mean": -0.02, "delta_margin_to_top20_cutoff": 0.10, "norm_activity_pressure_score": 1.0},
            {"asin": "b", "category_group": "s_cat_v2_weak", "delta_ndcg@20": 0.05, "delta_hr@20": 0.02, "delta_top20_user_norm_mean": -0.05, "delta_top20_history_interaction_count_mean": -10.0, "delta_target_minus_top20_history_q_cosine_mean": 0.04, "delta_margin_to_top20_cutoff": 0.20, "norm_activity_pressure_score": -1.0},
            {"asin": "c", "category_group": "s_cat_v2_strong", "delta_ndcg@20": 0.03, "delta_hr@20": 0.01, "delta_top20_user_norm_mean": 0.10, "delta_top20_history_interaction_count_mean": 20.0, "delta_target_minus_top20_history_q_cosine_mean": 0.01, "delta_margin_to_top20_cutoff": 0.05, "norm_activity_pressure_score": 0.5},
        ]
    )

    group_summary = build_group_mechanism_summary(profile)
    bucket = build_weak_bucket_summary(profile, "norm_activity_pressure_score", "norm_activity_bucket")
    decision = build_route_decision(profile, group_summary, {"weak_pressure_high_minus_low_ndcg": -0.15})

    weak_summary = group_summary[group_summary["category_group"].eq("s_cat_v2_weak")].iloc[0]
    assert weak_summary["item_count"] == 2
    assert math.isclose(weak_summary["delta_ndcg@20_mean"], -0.025)
    assert set(bucket["norm_activity_bucket"]) == {"low", "high"}
    assert decision["route"] == "norm_activity_bias_supported"
    json.dumps(decision, ensure_ascii=False)


def test_md_table_renders_without_optional_tabulate_dependency() -> None:
    table = md_table(pd.DataFrame([{"name": "weak", "value": 0.1234567}]), ["name", "value"])

    assert "| name | value |" in table
    assert "| weak | 0.123457 |" in table
