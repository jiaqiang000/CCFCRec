#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.24-3.43 预防性诊断测试。
"""

import json

import pandas as pd

from analyze_amazon_vg_category_availability_task3p_extra import (
    build_checkpoint_group_stability,
    build_cross_stability_profile,
    build_modality_conflict_profile,
    build_modality_conflict_summary,
    build_proxy_ensemble_profile,
    build_proxy_ensemble_summary,
    build_threshold_sensitivity_summary,
    build_tradeoff_summary,
    decide_checkpoint_group_stability_route,
    decide_modality_conflict_route,
    decide_proxy_ensemble_route,
    decide_threshold_sensitivity_route,
    decide_tradeoff_route,
)


def test_threshold_sensitivity_requires_consistent_margin_gaps() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "margin_proxy": -1.5, "ndcg@20": 0.0},
            {"asin": "b", "margin_proxy": -1.0, "ndcg@20": 0.0},
            {"asin": "c", "margin_proxy": -0.2, "ndcg@20": 0.1},
            {"asin": "d", "margin_proxy": 0.2, "ndcg@20": 0.2},
            {"asin": "e", "margin_proxy": 1.0, "ndcg@20": 0.7},
            {"asin": "f", "margin_proxy": 1.5, "ndcg@20": 0.8},
        ]
    )

    summary = build_threshold_sensitivity_summary(profile, metric="margin_proxy", cuts=(0.33, 0.4))
    decision = decide_threshold_sensitivity_route(summary)

    assert (summary["high_minus_low_ndcg@20"] > 0).all()
    assert decision["route"] == "margin_threshold_robust"
    json.dumps(decision, ensure_ascii=False)


def test_cross_stability_keeps_existing_residual_bucket_when_merging() -> None:
    rank_profile = pd.DataFrame(
        [
            {"asin": "a", "rank_aware_group": "rank_low", "residual_bucket": "hard", "ndcg@20": 0.0},
            {"asin": "b", "rank_aware_group": "rank_high", "residual_bucket": "easy", "ndcg@20": 0.5},
        ]
    )
    residual_profile = pd.DataFrame(
        [
            {"asin": "a", "residual_bucket": "hard", "ndcg@20_control_residual": -0.2},
            {"asin": "b", "residual_bucket": "easy", "ndcg@20_control_residual": 0.2},
        ]
    )

    merged = build_cross_stability_profile(rank_profile, residual_profile)

    assert "residual_bucket" in merged.columns
    assert "rank_low__hard" in set(merged["rank_residual_group"])


def test_tradeoff_summary_detects_hard_gain_easy_loss() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "delta_ndcg@20": 0.05, "consensus_group": "consensus_high", "rank_aware_group": "rank_low", "residual_bucket": "hard"},
            {"asin": "b", "delta_ndcg@20": 0.04, "consensus_group": "consensus_high", "rank_aware_group": "rank_low", "residual_bucket": "hard"},
            {"asin": "c", "delta_ndcg@20": -0.04, "consensus_group": "consensus_low", "rank_aware_group": "rank_high", "residual_bucket": "easy"},
            {"asin": "d", "delta_ndcg@20": -0.05, "consensus_group": "consensus_low", "rank_aware_group": "rank_high", "residual_bucket": "easy"},
        ]
    )

    summary = build_tradeoff_summary(profile)
    decision = decide_tradeoff_route(summary)

    assert decision["route"] == "hard_gain_easy_loss_tradeoff"
    assert decision["evidence"]["hard_like_delta_ndcg@20"] > 0
    assert decision["evidence"]["easy_like_delta_ndcg@20"] < 0


def test_modality_conflict_routes_when_conflict_group_is_hard() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "ndcg@20": 0.0, "target_minus_top20_history_q_cosine_mean": -0.8, "target_minus_top20_history_attr_cosine_mean": 0.7, "target_minus_top20_history_img_cosine_mean": 0.6, "consensus_group": "consensus_high"},
            {"asin": "b", "ndcg@20": 0.1, "target_minus_top20_history_q_cosine_mean": -0.7, "target_minus_top20_history_attr_cosine_mean": 0.8, "target_minus_top20_history_img_cosine_mean": 0.5, "consensus_group": "consensus_high"},
            {"asin": "c", "ndcg@20": 0.5, "target_minus_top20_history_q_cosine_mean": 0.4, "target_minus_top20_history_attr_cosine_mean": 0.5, "target_minus_top20_history_img_cosine_mean": 0.4, "consensus_group": "consensus_low"},
            {"asin": "d", "ndcg@20": 0.6, "target_minus_top20_history_q_cosine_mean": 0.5, "target_minus_top20_history_attr_cosine_mean": 0.6, "target_minus_top20_history_img_cosine_mean": 0.5, "consensus_group": "consensus_low"},
        ]
    )

    conflict = build_modality_conflict_profile(profile)
    summary = build_modality_conflict_summary(conflict)
    decision = decide_modality_conflict_route(summary)

    assert decision["route"] == "modality_conflict_actionable"
    assert conflict["modality_conflict_score"].max() > conflict["modality_conflict_score"].min()


def test_checkpoint_group_stability_detects_stable_hard_gain() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "checkpoint_label": "baseline_best", "ndcg@20": 0.01, "consensus_group": "consensus_high"},
            {"asin": "a", "checkpoint_label": "baseline_last", "ndcg@20": 0.02, "consensus_group": "consensus_high"},
            {"asin": "a", "checkpoint_label": "category_conf_best", "ndcg@20": 0.04, "consensus_group": "consensus_high"},
            {"asin": "a", "checkpoint_label": "category_conf_last", "ndcg@20": 0.05, "consensus_group": "consensus_high"},
            {"asin": "b", "checkpoint_label": "baseline_best", "ndcg@20": 0.40, "consensus_group": "consensus_low"},
            {"asin": "b", "checkpoint_label": "baseline_last", "ndcg@20": 0.39, "consensus_group": "consensus_low"},
            {"asin": "b", "checkpoint_label": "category_conf_best", "ndcg@20": 0.38, "consensus_group": "consensus_low"},
            {"asin": "b", "checkpoint_label": "category_conf_last", "ndcg@20": 0.37, "consensus_group": "consensus_low"},
        ]
    )

    stability = build_checkpoint_group_stability(profile, group_col="consensus_group")
    decision = decide_checkpoint_group_stability_route(stability)

    assert decision["route"] == "new_groups_checkpoint_stable"
    assert decision["evidence"]["stable_positive_group_count"] >= 1


def test_proxy_ensemble_can_outperform_single_margin() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "ndcg@20": 0.0, "margin_proxy": -1.0, "residual_bucket": "hard", "target_competitor_gap_proxy": -0.4, "modality_conflict_score": 1.0, "coverage_score": 0.0, "delta_ndcg@20": 0.05},
            {"asin": "b", "ndcg@20": 0.1, "margin_proxy": -0.8, "residual_bucket": "hard", "target_competitor_gap_proxy": -0.2, "modality_conflict_score": 0.8, "coverage_score": 0.1, "delta_ndcg@20": 0.04},
            {"asin": "c", "ndcg@20": 0.7, "margin_proxy": 0.8, "residual_bucket": "easy", "target_competitor_gap_proxy": 0.2, "modality_conflict_score": 0.0, "coverage_score": 1.0, "delta_ndcg@20": -0.02},
            {"asin": "d", "ndcg@20": 0.8, "margin_proxy": 1.0, "residual_bucket": "easy", "target_competitor_gap_proxy": 0.3, "modality_conflict_score": 0.0, "coverage_score": 1.0, "delta_ndcg@20": -0.03},
        ]
    )

    ensemble = build_proxy_ensemble_profile(profile)
    summary = build_proxy_ensemble_summary(ensemble)
    decision = decide_proxy_ensemble_route(summary)

    assert decision["route"] == "proxy_ensemble_ready"
    assert summary.iloc[0]["ensemble_high_minus_low_ndcg@20"] > 0.5
