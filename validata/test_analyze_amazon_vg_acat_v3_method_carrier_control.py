#!/usr/bin/env python3
"""
CCFCRec Amazon-VG Task4-pre Acat v3 方法载体对照审计测试。
"""

from __future__ import annotations

import json

import pandas as pd

from analyze_amazon_vg_acat_v3_method_carrier_control import (
    build_carrier_correlation,
    build_carrier_profile,
    build_carrier_summary,
    build_placebo_summary,
    decide_carrier_route,
)


def make_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    v3 = pd.DataFrame(
        [
            {"raw_asin": "a", "s_cat_v3": 0.90, "s_cat_v3_group": "s_cat_v3_strong", "category_count": 3, "R_metadata_richness_score": 0.2, "S_train_support_score": 0.2, "P_popularity_score": 0.2, "s_cat_v2": 0.4},
            {"raw_asin": "b", "s_cat_v3": 0.80, "s_cat_v3_group": "s_cat_v3_strong", "category_count": 3, "R_metadata_richness_score": 0.2, "S_train_support_score": 0.3, "P_popularity_score": 0.3, "s_cat_v2": 0.5},
            {"raw_asin": "c", "s_cat_v3": 0.20, "s_cat_v3_group": "s_cat_v3_weak", "category_count": 8, "R_metadata_richness_score": 0.8, "S_train_support_score": 0.8, "P_popularity_score": 0.8, "s_cat_v2": 0.6},
            {"raw_asin": "d", "s_cat_v3": 0.10, "s_cat_v3_group": "s_cat_v3_weak", "category_count": 8, "R_metadata_richness_score": 0.9, "S_train_support_score": 0.9, "P_popularity_score": 0.9, "s_cat_v2": 0.7},
        ]
    )
    delta = pd.DataFrame(
        [
            {"asin": "a", "baseline_ndcg@20": 0.05, "delta_ndcg@20": 0.05, "delta_margin_to_top20_cutoff": 0.40, "delta_q_norm": 0.10},
            {"asin": "b", "baseline_ndcg@20": 0.10, "delta_ndcg@20": 0.04, "delta_margin_to_top20_cutoff": 0.30, "delta_q_norm": 0.20},
            {"asin": "c", "baseline_ndcg@20": 0.40, "delta_ndcg@20": -0.01, "delta_margin_to_top20_cutoff": -0.10, "delta_q_norm": 0.30},
            {"asin": "d", "baseline_ndcg@20": 0.35, "delta_ndcg@20": -0.02, "delta_margin_to_top20_cutoff": -0.20, "delta_q_norm": 0.40},
        ]
    )
    recoverability = pd.DataFrame(
        [
            {"asin": "a", "proxy_ensemble_score": 0.9, "margin_proxy": -1.0},
            {"asin": "b", "proxy_ensemble_score": 0.8, "margin_proxy": -0.8},
            {"asin": "c", "proxy_ensemble_score": 0.2, "margin_proxy": 0.4},
            {"asin": "d", "proxy_ensemble_score": 0.1, "margin_proxy": 0.5},
        ]
    )
    return v3, delta, recoverability


def test_carrier_profile_builds_all_control_scores() -> None:
    v3, delta, recoverability = make_inputs()

    profile = build_carrier_profile(v3, delta, recoverability, random_seed=7)

    assert {"acat_v3_score", "acat_v3_shuffle_score", "rsp_only_score", "recoverability_proxy_score", "v2_s_cat_score"}.issubset(profile.columns)
    assert profile["acat_v3_score"].max() > profile["acat_v3_score"].min()
    assert set(profile["acat_v3_bucket"]).issubset({"low", "mid", "high"})
    assert len(profile) == 4


def test_carrier_summary_detects_acat_response_gap() -> None:
    v3, delta, recoverability = make_inputs()
    profile = build_carrier_profile(v3, delta, recoverability, random_seed=7)

    summary = build_carrier_summary(profile)
    acat = summary[summary["carrier"].eq("acat_v3")].iloc[0]

    assert acat["high_minus_low_delta_ndcg@20"] > 0.05
    assert acat["high_minus_low_delta_margin"] > 0.5


def test_carrier_correlation_and_placebo_are_jsonable() -> None:
    v3, delta, recoverability = make_inputs()
    profile = build_carrier_profile(v3, delta, recoverability, random_seed=7)

    corr = build_carrier_correlation(profile)
    placebo = build_placebo_summary(profile, shuffle_count=8, random_seed=7)

    assert "spearman_vs_delta_ndcg@20" in corr.columns
    assert placebo["shuffle_count"].iloc[0] == 8
    json.dumps(corr.to_dict(orient="records"), ensure_ascii=False)
    json.dumps(placebo.to_dict(orient="records"), ensure_ascii=False)


def test_decision_routes_ready_when_acat_beats_controls() -> None:
    summary = pd.DataFrame(
        [
            {"carrier": "acat_v3", "high_minus_low_delta_ndcg@20": 0.08, "high_minus_low_delta_margin": 0.50},
            {"carrier": "acat_v3_shuffle", "high_minus_low_delta_ndcg@20": 0.00, "high_minus_low_delta_margin": 0.05},
            {"carrier": "rsp_only", "high_minus_low_delta_ndcg@20": -0.02, "high_minus_low_delta_margin": 0.00},
            {"carrier": "recoverability_proxy", "high_minus_low_delta_ndcg@20": 0.03, "high_minus_low_delta_margin": 0.30},
        ]
    )
    placebo = pd.DataFrame([{"shuffle_delta_ndcg@20_mean": 0.0, "shuffle_delta_ndcg@20_p95": 0.02}])

    decision = decide_carrier_route(summary, placebo)

    assert decision["route"] == "acat_v3_carrier_ready"
    json.dumps(decision, ensure_ascii=False)


def test_decision_routes_new_method_when_acat_not_helped_by_old_category_conf() -> None:
    summary = pd.DataFrame(
        [
            {"carrier": "acat_v3", "high_minus_low_delta_ndcg@20": -0.03, "high_minus_low_delta_margin": -0.20},
            {"carrier": "acat_v3_shuffle", "high_minus_low_delta_ndcg@20": 0.00, "high_minus_low_delta_margin": 0.00},
            {"carrier": "rsp_only", "high_minus_low_delta_ndcg@20": -0.01, "high_minus_low_delta_margin": -0.05},
            {"carrier": "recoverability_proxy", "high_minus_low_delta_ndcg@20": 0.10, "high_minus_low_delta_margin": 0.80},
        ]
    )
    placebo = pd.DataFrame([{"shuffle_delta_ndcg@20_mean": 0.0, "shuffle_delta_ndcg@20_p95": 0.02}])

    decision = decide_carrier_route(summary, placebo)

    assert decision["route"] == "acat_v3_needs_new_method_not_category_conf"
    assert decision["evidence"]["recoverability_upper_bound_gap"] > 0


def test_decision_routes_rsp_stronger_when_old_method_tracks_controls() -> None:
    summary = pd.DataFrame(
        [
            {"carrier": "acat_v3", "high_minus_low_delta_ndcg@20": -0.01, "high_minus_low_delta_margin": 0.00},
            {"carrier": "acat_v3_shuffle", "high_minus_low_delta_ndcg@20": -0.01, "high_minus_low_delta_margin": 0.00},
            {"carrier": "rsp_only", "high_minus_low_delta_ndcg@20": 0.02, "high_minus_low_delta_margin": 0.10},
            {"carrier": "recoverability_proxy", "high_minus_low_delta_ndcg@20": -0.03, "high_minus_low_delta_margin": -0.20},
        ]
    )
    placebo = pd.DataFrame([{"shuffle_delta_ndcg@20_mean": 0.0, "shuffle_delta_ndcg@20_p95": 0.01}])

    decision = decide_carrier_route(summary, placebo)

    assert decision["route"] == "category_conf_response_rsp_only_stronger"
