#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.19 proxy metric 诊断测试。
"""

import json

import pandas as pd

from analyze_amazon_vg_category_availability_task3p19 import (
    build_proxy_controlled_stability,
    build_proxy_metric_profile,
    build_proxy_vs_rank_summary,
    build_route_decision,
)


def test_proxy_metric_profile_builds_task4_candidate_proxies() -> None:
    item_profile = pd.DataFrame(
        [
            {"asin": "a", "category_group": "s_cat_v2_weak", "category_count": 2, "gt_group": "gt_1"},
            {"asin": "b", "category_group": "s_cat_v2_strong", "category_count": 6, "gt_group": "gt_1"},
        ]
    )
    score = pd.DataFrame(
        [
            {"asin": "a", "hr@20": 0.0, "ndcg@20": 0.0, "margin_to_top20_cutoff": -0.4, "best_target_rank": 80.0, "q_norm": 10.0, "top20_user_norm_mean": 0.4},
            {"asin": "b", "hr@20": 0.1, "ndcg@20": 0.4, "margin_to_top20_cutoff": 0.3, "best_target_rank": 8.0, "q_norm": 11.0, "top20_user_norm_mean": 0.2},
        ]
    )
    target = pd.DataFrame(
        [
            {"asin": "a", "target_minus_top20_cosine_mean": -0.30, "target_minus_top20_user_norm_mean": -0.20},
            {"asin": "b", "target_minus_top20_cosine_mean": 0.05, "target_minus_top20_user_norm_mean": 0.10},
        ]
    )
    content = pd.DataFrame(
        [
            {"asin": "a", "target_history_interaction_count_mean": 10.0, "target_minus_top20_history_q_cosine_mean": -0.10, "target_minus_top20_history_attr_cosine_mean": -0.20, "target_minus_top20_history_img_cosine_mean": -0.30},
            {"asin": "b", "target_history_interaction_count_mean": 12.0, "target_minus_top20_history_q_cosine_mean": 0.05, "target_minus_top20_history_attr_cosine_mean": 0.10, "target_minus_top20_history_img_cosine_mean": 0.20},
        ]
    )

    profile = build_proxy_metric_profile(item_profile, score, target, content)

    hard = profile[profile["asin"].eq("a")].iloc[0]
    easy = profile[profile["asin"].eq("b")].iloc[0]
    assert easy["margin_proxy"] > hard["margin_proxy"]
    assert easy["target_competitor_gap_proxy"] > hard["target_competitor_gap_proxy"]
    assert easy["modality_alignment_proxy"] > hard["modality_alignment_proxy"]
    assert easy["calibration_proxy"] > hard["calibration_proxy"]


def test_proxy_summary_stability_and_route_recommend_margin_proxy() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "gt_group": "gt_1", "target_activity_bucket": "low", "ndcg@20": 0.00, "best_target_rank": 80.0, "margin_to_top20_cutoff": -0.50, "margin_proxy": -0.50, "target_competitor_gap_proxy": -0.20, "modality_alignment_proxy": -0.10, "calibration_proxy": -1.0},
            {"asin": "b", "gt_group": "gt_2_3", "target_activity_bucket": "high", "ndcg@20": 0.05, "best_target_rank": 40.0, "margin_to_top20_cutoff": -0.20, "margin_proxy": -0.20, "target_competitor_gap_proxy": -0.10, "modality_alignment_proxy": -0.05, "calibration_proxy": -0.5},
            {"asin": "c", "gt_group": "gt_1", "target_activity_bucket": "low", "ndcg@20": 0.40, "best_target_rank": 5.0, "margin_to_top20_cutoff": 0.40, "margin_proxy": 0.40, "target_competitor_gap_proxy": 0.05, "modality_alignment_proxy": 0.10, "calibration_proxy": 0.5},
            {"asin": "d", "gt_group": "gt_2_3", "target_activity_bucket": "high", "ndcg@20": 0.45, "best_target_rank": 4.0, "margin_to_top20_cutoff": 0.60, "margin_proxy": 0.60, "target_competitor_gap_proxy": 0.10, "modality_alignment_proxy": 0.20, "calibration_proxy": 1.0},
        ]
    )

    proxy_summary = build_proxy_vs_rank_summary(profile)
    stability = build_proxy_controlled_stability(profile)
    decision = build_route_decision(proxy_summary, stability)

    margin = proxy_summary[proxy_summary["proxy_metric"].eq("margin_proxy")].iloc[0]
    assert margin["spearman_vs_ndcg@20"] > 0.9
    assert margin["spearman_vs_best_target_rank"] < -0.9
    assert decision["route"] == "margin_proxy_recommended"
    json.dumps(decision, ensure_ascii=False)
