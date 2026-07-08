#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 remaining Task3.x 诊断测试。
"""

import json

import pandas as pd

from analyze_amazon_vg_category_availability_task3p_remaining import (
    build_alt_availability_profile,
    build_alt_group_validity_summary,
    build_case_archetype_summary,
    build_consensus_signal_profile,
    build_modality_failure_summary,
    build_placebo_summary,
    build_target_competitor_gap_summary,
    decide_alt_availability_route,
    decide_modality_route,
    decide_placebo_route,
    decide_target_competitor_route,
)


def test_target_competitor_gap_summary_detects_competitor_overpower() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "gap_bucket": "gap_low", "ndcg@20": 0.0, "margin_to_top20_cutoff": -1.0, "best_target_rank": 100.0, "target_competitor_gap_proxy": -0.5, "top20_user_norm_mean": 0.5, "top20_history_interaction_count_mean": 100.0},
            {"asin": "b", "gap_bucket": "gap_high", "ndcg@20": 0.5, "margin_to_top20_cutoff": 0.5, "best_target_rank": 5.0, "target_competitor_gap_proxy": 0.2, "top20_user_norm_mean": 0.2, "top20_history_interaction_count_mean": 20.0},
        ]
    )

    summary = build_target_competitor_gap_summary(profile)
    decision = decide_target_competitor_route(summary)

    assert decision["route"] == "competitor_overpower_supported"
    json.dumps(decision, ensure_ascii=False)


def test_modality_summary_routes_to_strongest_modality() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "ndcg@20": 0.0, "best_target_rank": 100.0, "target_minus_top20_history_q_cosine_mean": -0.1, "target_minus_top20_history_attr_cosine_mean": -0.8, "target_minus_top20_history_img_cosine_mean": -0.2},
            {"asin": "b", "ndcg@20": 0.5, "best_target_rank": 5.0, "target_minus_top20_history_q_cosine_mean": 0.0, "target_minus_top20_history_attr_cosine_mean": 0.8, "target_minus_top20_history_img_cosine_mean": 0.1},
            {"asin": "c", "ndcg@20": 0.3, "best_target_rank": 10.0, "target_minus_top20_history_q_cosine_mean": 0.1, "target_minus_top20_history_attr_cosine_mean": 0.4, "target_minus_top20_history_img_cosine_mean": 0.2},
        ]
    )

    summary = build_modality_failure_summary(profile)
    decision = decide_modality_route(summary)

    assert decision["route"] == "attr_path_failure_supported"
    json.dumps(decision, ensure_ascii=False)


def test_alt_availability_and_placebo_support_rank_aware_signal() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "ndcg@20": 0.0, "margin_proxy": -1.0, "target_competitor_gap_proxy": -0.5, "gt_group": "gt_1", "category_count": 2, "target_activity_bucket": "low", "s_cat_group": "s_cat_v2_weak"},
            {"asin": "b", "ndcg@20": 0.1, "margin_proxy": -0.5, "target_competitor_gap_proxy": -0.2, "gt_group": "gt_1", "category_count": 3, "target_activity_bucket": "low", "s_cat_group": "s_cat_v2_mid"},
            {"asin": "c", "ndcg@20": 0.4, "margin_proxy": 0.4, "target_competitor_gap_proxy": 0.1, "gt_group": "gt_2_3", "category_count": 6, "target_activity_bucket": "high", "s_cat_group": "s_cat_v2_strong"},
            {"asin": "d", "ndcg@20": 0.6, "margin_proxy": 0.8, "target_competitor_gap_proxy": 0.3, "gt_group": "gt_2_3", "category_count": 7, "target_activity_bucket": "high", "s_cat_group": "s_cat_v2_strong"},
        ]
    )

    alt = build_alt_availability_profile(profile)
    validity = build_alt_group_validity_summary(alt)
    alt_decision = decide_alt_availability_route(validity)
    placebo = build_placebo_summary(alt, shuffle_count=8, random_seed=7)
    placebo_decision = decide_placebo_route(placebo)

    assert alt_decision["route"] == "rank_aware_availability_promising"
    assert placebo_decision["route"] == "availability_beats_placebo"
    json.dumps(alt_decision, ensure_ascii=False)
    json.dumps(placebo_decision, ensure_ascii=False)


def test_case_archetype_and_consensus_profiles_are_actionable() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "ndcg@20": 0.0, "margin_bucket": "mid", "residual_bucket": "hard", "hard_negative_pressure_score": 1.0, "category_group": "s_cat_v2_weak", "target_competitor_gap_proxy": -0.4, "modality_alignment_proxy": -0.3},
            {"asin": "b", "ndcg@20": 0.0, "margin_bucket": "mid", "residual_bucket": "hard", "hard_negative_pressure_score": 1.2, "category_group": "s_cat_v2_mid", "target_competitor_gap_proxy": -0.3, "modality_alignment_proxy": -0.2},
            {"asin": "c", "ndcg@20": 0.5, "margin_bucket": "high", "residual_bucket": "easy", "hard_negative_pressure_score": -1.0, "category_group": "s_cat_v2_strong", "target_competitor_gap_proxy": 0.2, "modality_alignment_proxy": 0.1},
        ]
    )

    archetypes = build_case_archetype_summary(profile)
    consensus = build_consensus_signal_profile(profile)

    assert "margin_mid_hard_residual" in set(archetypes["archetype"])
    assert consensus["consensus_score"].max() > consensus["consensus_score"].min()
