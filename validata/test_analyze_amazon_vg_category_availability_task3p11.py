#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.11 category-CF interaction 诊断测试。
"""

import json
import math

import pandas as pd

from analyze_amazon_vg_category_availability_task3p11 import (
    build_category_cf_interaction_profile,
    build_interaction_grid_summary,
    build_route_decision,
    build_within_cf_category_effect,
)


def test_interaction_profile_builds_category_and_cf_evidence_buckets() -> None:
    item_profile = pd.DataFrame(
        [
            {"asin": "a", "category_group": "s_cat_v2_weak", "category_count": 2, "gt_group": "gt_1", "s_cat": 0.1, "s_cat_v2_disc_within_control": 0.1, "s_cat_v2_collab_within_control": 0.2},
            {"asin": "b", "category_group": "s_cat_v2_strong", "category_count": 6, "gt_group": "gt_1", "s_cat": 0.9, "s_cat_v2_disc_within_control": 0.8, "s_cat_v2_collab_within_control": 0.9},
        ]
    )
    score = pd.DataFrame(
        [
            {"asin": "a", "hr@20": 0.0, "ndcg@20": 0.0, "margin_to_top20_cutoff": -0.4},
            {"asin": "b", "hr@20": 0.1, "ndcg@20": 0.4, "margin_to_top20_cutoff": 0.3},
        ]
    )
    target = pd.DataFrame(
        [
            {"asin": "a", "target_minus_top20_cosine_mean": -0.20},
            {"asin": "b", "target_minus_top20_cosine_mean": 0.10},
        ]
    )
    content = pd.DataFrame(
        [
            {"asin": "a", "target_history_interaction_count_mean": 10.0, "target_history_q_cosine_mean": 0.2, "target_minus_top20_history_q_cosine_mean": -0.2},
            {"asin": "b", "target_history_interaction_count_mean": 12.0, "target_history_q_cosine_mean": 0.6, "target_minus_top20_history_q_cosine_mean": 0.1},
        ]
    )

    profile = build_category_cf_interaction_profile(item_profile, score, target, content)

    low = profile[profile["asin"].eq("a")].iloc[0]
    high = profile[profile["asin"].eq("b")].iloc[0]
    assert low["category_evidence_bucket"] == "cat_low"
    assert high["category_evidence_bucket"] == "cat_high"
    assert low["cf_evidence_bucket"] == "cf_low"
    assert high["cf_evidence_bucket"] == "cf_high"
    assert high["category_evidence_score"] > low["category_evidence_score"]
    assert high["cf_evidence_score"] > low["cf_evidence_score"]


def test_interaction_effect_and_route_detect_category_helps_when_cf_weak() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "category_evidence_bucket": "cat_low", "cf_evidence_bucket": "cf_low", "ndcg@20": 0.05, "hr@20": 0.01, "ndcg@20_control_residual": -0.20, "margin_to_top20_cutoff": -0.5},
            {"asin": "b", "category_evidence_bucket": "cat_high", "cf_evidence_bucket": "cf_low", "ndcg@20": 0.20, "hr@20": 0.04, "ndcg@20_control_residual": 0.05, "margin_to_top20_cutoff": -0.2},
            {"asin": "c", "category_evidence_bucket": "cat_low", "cf_evidence_bucket": "cf_high", "ndcg@20": 0.30, "hr@20": 0.06, "ndcg@20_control_residual": 0.02, "margin_to_top20_cutoff": 0.3},
            {"asin": "d", "category_evidence_bucket": "cat_high", "cf_evidence_bucket": "cf_high", "ndcg@20": 0.32, "hr@20": 0.06, "ndcg@20_control_residual": 0.03, "margin_to_top20_cutoff": 0.4},
        ]
    )

    grid = build_interaction_grid_summary(profile)
    effect = build_within_cf_category_effect(grid)
    decision = build_route_decision(grid, effect)

    cf_low = effect[effect["cf_evidence_bucket"].eq("cf_low")].iloc[0]
    assert math.isclose(cf_low["high_minus_low_cat_residual_ndcg@20"], 0.25)
    assert decision["route"] == "category_helps_when_cf_weak"
    json.dumps(decision, ensure_ascii=False)
