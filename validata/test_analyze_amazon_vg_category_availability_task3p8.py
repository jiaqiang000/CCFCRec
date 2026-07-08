#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.8 rank-margin 诊断测试。
"""

import json
import math

import pandas as pd

from analyze_amazon_vg_category_availability_task3p8 import (
    build_controlled_margin_gap,
    build_margin_bucket_summary,
    build_rank_margin_profile,
    build_route_decision,
)


def test_rank_margin_profile_merges_margin_competitor_and_activity_signals() -> None:
    item_profile = pd.DataFrame(
        [
            {"asin": "a", "category_group": "s_cat_v2_weak", "category_count": 2, "gt_group": "gt_1", "gt_user_count": 1},
            {"asin": "b", "category_group": "s_cat_v2_strong", "category_count": 6, "gt_group": "gt_2_3", "gt_user_count": 3},
        ]
    )
    score = pd.DataFrame(
        [
            {"asin": "a", "hr@20": 0.0, "ndcg@20": 0.0, "margin_to_top20_cutoff": -0.4, "best_target_rank": 80.0, "best_target_rank_percentile": 0.08, "top20_user_norm_mean": 0.40, "q_norm": 10.0},
            {"asin": "b", "hr@20": 0.1, "ndcg@20": 0.2, "margin_to_top20_cutoff": 0.3, "best_target_rank": 8.0, "best_target_rank_percentile": 0.01, "top20_user_norm_mean": 0.20, "q_norm": 11.0},
        ]
    )
    target = pd.DataFrame(
        [
            {"asin": "a", "top20_cosine_mean": 0.50, "target_cosine_at_score_max": 0.20, "target_minus_top20_cosine_mean": -0.30, "target_minus_top20_user_norm_mean": -0.20},
            {"asin": "b", "top20_cosine_mean": 0.30, "target_cosine_at_score_max": 0.35, "target_minus_top20_cosine_mean": 0.05, "target_minus_top20_user_norm_mean": 0.10},
        ]
    )
    content = pd.DataFrame(
        [
            {"asin": "a", "target_history_interaction_count_mean": 5.0, "top20_history_interaction_count_mean": 100.0, "target_minus_top20_history_interaction_count_mean": -95.0, "target_minus_top20_history_q_cosine_mean": -0.10},
            {"asin": "b", "target_history_interaction_count_mean": 20.0, "top20_history_interaction_count_mean": 30.0, "target_minus_top20_history_interaction_count_mean": -10.0, "target_minus_top20_history_q_cosine_mean": 0.05},
        ]
    )

    profile = build_rank_margin_profile(item_profile, score, target, content)

    hard = profile[profile["asin"].eq("a")].iloc[0]
    easy = profile[profile["asin"].eq("b")].iloc[0]
    assert hard["target_rank_near_cutoff"]
    assert not easy["target_rank_near_cutoff"]
    assert hard["hard_negative_pressure_score"] > easy["hard_negative_pressure_score"]
    assert hard["margin_bucket"] == "low"
    assert easy["margin_bucket"] == "high"


def test_margin_summary_and_route_detect_rank_margin_hard_negative() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "margin_bucket": "low", "category_group": "s_cat_v2_weak", "gt_group": "gt_1", "target_activity_bucket": "low", "ndcg@20": 0.00, "hr@20": 0.00, "margin_to_top20_cutoff": -0.50, "best_target_rank": 80.0, "target_rank_near_cutoff": True, "hard_negative_pressure_score": 2.0, "top20_user_norm_mean": 0.40, "top20_history_interaction_count_mean": 100.0, "target_minus_top20_cosine_mean": -0.30},
            {"asin": "b", "margin_bucket": "low", "category_group": "s_cat_v2_mid", "gt_group": "gt_2_3", "target_activity_bucket": "high", "ndcg@20": 0.05, "hr@20": 0.01, "margin_to_top20_cutoff": -0.20, "best_target_rank": 40.0, "target_rank_near_cutoff": True, "hard_negative_pressure_score": 1.0, "top20_user_norm_mean": 0.35, "top20_history_interaction_count_mean": 80.0, "target_minus_top20_cosine_mean": -0.20},
            {"asin": "c", "margin_bucket": "high", "category_group": "s_cat_v2_strong", "gt_group": "gt_1", "target_activity_bucket": "low", "ndcg@20": 0.40, "hr@20": 0.08, "margin_to_top20_cutoff": 0.40, "best_target_rank": 5.0, "target_rank_near_cutoff": False, "hard_negative_pressure_score": -1.0, "top20_user_norm_mean": 0.20, "top20_history_interaction_count_mean": 20.0, "target_minus_top20_cosine_mean": 0.05},
            {"asin": "d", "margin_bucket": "high", "category_group": "s_cat_v2_strong", "gt_group": "gt_2_3", "target_activity_bucket": "high", "ndcg@20": 0.45, "hr@20": 0.09, "margin_to_top20_cutoff": 0.60, "best_target_rank": 4.0, "target_rank_near_cutoff": False, "hard_negative_pressure_score": -2.0, "top20_user_norm_mean": 0.15, "top20_history_interaction_count_mean": 15.0, "target_minus_top20_cosine_mean": 0.10},
        ]
    )

    bucket = build_margin_bucket_summary(profile)
    controlled = build_controlled_margin_gap(profile)
    decision = build_route_decision(bucket, controlled)

    low = bucket[bucket["margin_bucket"].eq("low")].iloc[0]
    assert math.isclose(low["ndcg@20_mean"], 0.025)
    assert math.isclose(low["target_rank_near_cutoff_rate"], 1.0)
    assert decision["route"] == "rank_margin_hard_negative_supported"
    json.dumps(decision, ensure_ascii=False)
