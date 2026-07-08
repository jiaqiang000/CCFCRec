#!/usr/bin/env python3
"""
CCFCRec Amazon-VG Task4-pre-2 Acat_v3 target profile 测试。
"""

from __future__ import annotations

import json

import pandas as pd

from build_amazon_vg_task4_acat_v3_target_profile import (
    build_category_concentration,
    build_rsp_score,
    build_target_profile,
    build_trainability_summary,
    decide_trainability_route,
)


def make_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    acat = pd.DataFrame(
        [
            {"raw_asin": "a", "split": "test", "category_raw": "Video Games,PC,Games", "category_tokens": "Video Games|PC|Games", "s_cat_v3": 0.90, "s_cat_v3_group": "s_cat_v3_strong", "category_count": 3, "R_metadata_richness_score": 0.2, "S_train_support_score": 0.2, "P_popularity_score": 0.2},
            {"raw_asin": "b", "split": "test", "category_raw": "Video Games,PC,Games", "category_tokens": "Video Games|PC|Games", "s_cat_v3": 0.80, "s_cat_v3_group": "s_cat_v3_strong", "category_count": 3, "R_metadata_richness_score": 0.3, "S_train_support_score": 0.3, "P_popularity_score": 0.3},
            {"raw_asin": "c", "split": "test", "category_raw": "Video Games,Xbox 360,Games", "category_tokens": "Video Games|Xbox 360|Games", "s_cat_v3": 0.20, "s_cat_v3_group": "s_cat_v3_weak", "category_count": 6, "R_metadata_richness_score": 0.8, "S_train_support_score": 0.8, "P_popularity_score": 0.8},
            {"raw_asin": "d", "split": "train", "category_raw": "Video Games,Wii,Games", "category_tokens": "Video Games|Wii|Games", "s_cat_v3": 0.85, "s_cat_v3_group": "s_cat_v3_strong", "category_count": 4, "R_metadata_richness_score": 0.4, "S_train_support_score": 0.4, "P_popularity_score": 0.4},
        ]
    )
    recoverability = pd.DataFrame(
        [
            {"asin": "a", "ndcg@20": 0.00, "margin_proxy": -1.2, "best_target_rank": 1000, "gt_user_count": 1, "target_activity_bucket": "low", "target_history_interaction_count_mean": 5},
            {"asin": "b", "ndcg@20": 0.00, "margin_proxy": -1.1, "best_target_rank": 950, "gt_user_count": 1, "target_activity_bucket": "low", "target_history_interaction_count_mean": 6},
            {"asin": "c", "ndcg@20": 0.50, "margin_proxy": 0.5, "best_target_rank": 20, "gt_user_count": 3, "target_activity_bucket": "high", "target_history_interaction_count_mean": 100},
        ]
    )
    return acat, recoverability


def test_rsp_score_builds_score_and_group() -> None:
    acat, _ = make_inputs()
    scored = build_rsp_score(acat)

    assert "RSP_score" in scored.columns
    assert "RSP_group" in scored.columns
    assert scored["RSP_score"].between(0.0, 1.0).all()
    assert set(scored["RSP_group"]).issubset({"RSP_low", "RSP_mid", "RSP_high"})


def test_target_profile_uses_eval_hard_names_not_training_hard_name() -> None:
    acat, recoverability = make_inputs()
    profile = build_target_profile(acat, recoverability)

    assert "high_acat_flag" in profile.columns
    assert "eval_baseline_hard_flag" in profile.columns
    assert "high_acat_eval_hard_flag" in profile.columns
    assert "baseline_hard_flag" not in profile.columns
    assert profile.loc[profile["raw_asin"].eq("a"), "high_acat_eval_hard_flag"].iloc[0]
    assert not profile.loc[profile["raw_asin"].eq("d"), "eval_baseline_hard_flag"].iloc[0]


def test_trainability_summary_reports_counts_overlap_and_train_proxy_gap() -> None:
    acat, recoverability = make_inputs()
    profile = build_target_profile(acat, recoverability)

    summary = build_trainability_summary(profile)

    assert summary.loc[0, "total_item_count"] == 4
    assert summary.loc[0, "eval_item_count"] == 3
    assert summary.loc[0, "high_acat_eval_hard_count"] == 2
    assert bool(summary.loc[0, "train_safe_hard_proxy_available"]) is False
    assert "high_acat_eval_hard_rsp_high_share" in summary.columns
    json.dumps(summary.to_dict(orient="records"), ensure_ascii=False)


def test_category_concentration_finds_top_category_share() -> None:
    acat, recoverability = make_inputs()
    profile = build_target_profile(acat, recoverability)
    concentration = build_category_concentration(profile)

    assert not concentration.empty
    assert concentration.iloc[0]["item_count"] >= 2
    assert concentration.iloc[0]["share"] >= 0.5


def test_route_detects_train_proxy_needed_for_sufficient_group() -> None:
    summary = pd.DataFrame(
        [
            {
                "high_acat_eval_hard_count": 300,
                "high_acat_eval_hard_share_eval": 0.20,
                "high_acat_eval_hard_top_category_share": 0.20,
                "high_acat_eval_hard_rsp_high_share": 0.30,
                "train_safe_hard_proxy_available": False,
            }
        ]
    )

    decision = decide_trainability_route(summary)

    assert decision["route"] == "target_profile_ready_train_proxy_needed"
    json.dumps(decision, ensure_ascii=False)


def test_route_rejects_tiny_or_concentrated_group() -> None:
    summary = pd.DataFrame(
        [
            {
                "high_acat_eval_hard_count": 20,
                "high_acat_eval_hard_share_eval": 0.02,
                "high_acat_eval_hard_top_category_share": 0.80,
                "high_acat_eval_hard_rsp_high_share": 0.30,
                "train_safe_hard_proxy_available": False,
            }
        ]
    )

    decision = decide_trainability_route(summary)

    assert decision["route"] == "target_profile_too_small_or_concentrated"
