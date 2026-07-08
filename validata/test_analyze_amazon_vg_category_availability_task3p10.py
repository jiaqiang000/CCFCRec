#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.10 norm/activity matched residual 诊断测试。
"""

import json
import math

import pandas as pd

from analyze_amazon_vg_category_availability_task3p10 import (
    build_hard_subgroup_candidates,
    build_matched_residual_summary,
    build_norm_activity_matched_profile,
    build_route_decision,
)


def test_norm_activity_matched_profile_computes_controlled_residuals() -> None:
    item_profile = pd.DataFrame(
        [
            {"asin": "a", "category_group": "s_cat_v2_weak", "category_count": 2, "gt_group": "gt_1", "gt_user_count": 1},
            {"asin": "b", "category_group": "s_cat_v2_strong", "category_count": 6, "gt_group": "gt_1", "gt_user_count": 1},
        ]
    )
    score = pd.DataFrame(
        [
            {"asin": "a", "hr@20": 0.0, "ndcg@20": 0.0, "margin_to_top20_cutoff": -0.4, "best_target_rank": 80.0, "top20_user_norm_mean": 0.30, "target_user_norm_mean": 0.10},
            {"asin": "b", "hr@20": 0.1, "ndcg@20": 0.4, "margin_to_top20_cutoff": 0.3, "best_target_rank": 8.0, "top20_user_norm_mean": 0.32, "target_user_norm_mean": 0.11},
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
            {"asin": "a", "target_history_interaction_count_mean": 10.0, "top20_history_interaction_count_mean": 60.0, "target_minus_top20_history_q_cosine_mean": -0.10},
            {"asin": "b", "target_history_interaction_count_mean": 11.0, "top20_history_interaction_count_mean": 58.0, "target_minus_top20_history_q_cosine_mean": 0.05},
        ]
    )

    profile = build_norm_activity_matched_profile(item_profile, score, target, content)

    hard = profile[profile["asin"].eq("a")].iloc[0]
    easy = profile[profile["asin"].eq("b")].iloc[0]
    assert hard["matched_control_key"] == easy["matched_control_key"]
    assert math.isclose(hard["ndcg@20_control_residual"], -0.2)
    assert math.isclose(easy["ndcg@20_control_residual"], 0.2)
    assert hard["residual_bucket"] == "hard"
    assert easy["residual_bucket"] == "easy"


def test_residual_summary_candidates_and_route_find_matched_hard_subgroup() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "residual_bucket": "hard", "category_group": "s_cat_v2_weak", "margin_bucket": "low", "ndcg@20": 0.00, "hr@20": 0.00, "ndcg@20_control_residual": -0.30, "hr@20_control_residual": -0.04, "margin_to_top20_cutoff": -0.50, "hard_negative_pressure_score": 1.0},
            {"asin": "b", "residual_bucket": "hard", "category_group": "s_cat_v2_mid", "margin_bucket": "low", "ndcg@20": 0.05, "hr@20": 0.01, "ndcg@20_control_residual": -0.20, "hr@20_control_residual": -0.03, "margin_to_top20_cutoff": -0.20, "hard_negative_pressure_score": 1.5},
            {"asin": "c", "residual_bucket": "easy", "category_group": "s_cat_v2_strong", "margin_bucket": "high", "ndcg@20": 0.40, "hr@20": 0.08, "ndcg@20_control_residual": 0.20, "hr@20_control_residual": 0.03, "margin_to_top20_cutoff": 0.40, "hard_negative_pressure_score": -1.0},
            {"asin": "d", "residual_bucket": "easy", "category_group": "s_cat_v2_strong", "margin_bucket": "high", "ndcg@20": 0.45, "hr@20": 0.09, "ndcg@20_control_residual": 0.30, "hr@20_control_residual": 0.04, "margin_to_top20_cutoff": 0.60, "hard_negative_pressure_score": -1.5},
        ]
    )

    summary = build_matched_residual_summary(profile)
    candidates = build_hard_subgroup_candidates(profile)
    decision = build_route_decision(summary, candidates)

    hard_summary = summary[summary["residual_bucket"].eq("hard")].iloc[0]
    assert math.isclose(hard_summary["ndcg@20_control_residual_mean"], -0.25)
    top_candidate = candidates.sort_values("hard_rate_enrichment", ascending=False).iloc[0]
    assert top_candidate["candidate_col"] == "margin_bucket"
    assert top_candidate["candidate_value"] == "low"
    assert decision["route"] == "matched_hard_subgroup_found"
    json.dumps(decision, ensure_ascii=False)
