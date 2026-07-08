#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.7 方法信号匹配诊断测试。
"""

import json
import math

import pandas as pd

from analyze_amazon_vg_category_availability_task3p7 import (
    build_component_vs_method_delta,
    build_method_fit_decision,
    build_method_signal_profile,
    build_norm_vs_direction_shift,
    build_representation_shift_by_v2_group,
)


def test_method_signal_profile_computes_norm_direction_ratio() -> None:
    delta = pd.DataFrame(
        [
            {
                "asin": "a",
                "category_group": "s_cat_v2_weak",
                "s_cat": 0.2,
                "s_cat_v2_disc_within_control": 0.1,
                "s_cat_v2_collab_within_control": 0.3,
                "delta_q_norm": -4.0,
                "delta_target_cosine_at_score_max": 0.01,
                "delta_target_history_q_cosine_mean": 0.02,
                "delta_target_minus_top20_history_q_cosine_mean": 0.01,
                "delta_target_minus_top20_cosine_mean": 0.00,
                "delta_margin_to_top20_cutoff": 0.1,
                "delta_ndcg@20": -0.02,
            }
        ]
    )

    profile = build_method_signal_profile(delta)
    row = profile.iloc[0]

    assert math.isclose(row["norm_shift_score"], 4.0)
    assert math.isclose(row["direction_shift_score"], 0.04)
    assert row["norm_direction_ratio"] > 90
    assert row["norm_dominant"]


def test_method_summaries_and_decision_detect_norm_not_direction() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "category_group": "s_cat_v2_weak", "s_cat": 0.1, "s_cat_v2_disc_within_control": 0.1, "s_cat_v2_collab_within_control": 0.2, "delta_ndcg@20": -0.04, "delta_q_norm": -4.0, "delta_target_cosine_at_score_max": 0.01, "delta_target_history_q_cosine_mean": 0.02, "delta_target_minus_top20_history_q_cosine_mean": 0.01, "delta_target_minus_top20_cosine_mean": 0.00, "delta_margin_to_top20_cutoff": 0.1, "norm_shift_score": 4.0, "direction_shift_score": 0.04, "norm_direction_ratio": 100.0, "norm_dominant": True},
            {"asin": "b", "category_group": "s_cat_v2_mid", "s_cat": 0.5, "s_cat_v2_disc_within_control": 0.4, "s_cat_v2_collab_within_control": 0.6, "delta_ndcg@20": -0.01, "delta_q_norm": -4.2, "delta_target_cosine_at_score_max": 0.02, "delta_target_history_q_cosine_mean": 0.01, "delta_target_minus_top20_history_q_cosine_mean": 0.01, "delta_target_minus_top20_cosine_mean": 0.00, "delta_margin_to_top20_cutoff": 0.1, "norm_shift_score": 4.2, "direction_shift_score": 0.04, "norm_direction_ratio": 105.0, "norm_dominant": True},
            {"asin": "c", "category_group": "s_cat_v2_strong", "s_cat": 0.9, "s_cat_v2_disc_within_control": 0.8, "s_cat_v2_collab_within_control": 0.9, "delta_ndcg@20": 0.02, "delta_q_norm": -4.1, "delta_target_cosine_at_score_max": 0.01, "delta_target_history_q_cosine_mean": 0.02, "delta_target_minus_top20_history_q_cosine_mean": 0.01, "delta_target_minus_top20_cosine_mean": 0.00, "delta_margin_to_top20_cutoff": 0.1, "norm_shift_score": 4.1, "direction_shift_score": 0.04, "norm_direction_ratio": 102.5, "norm_dominant": True},
        ]
    )

    rep = build_representation_shift_by_v2_group(profile)
    norm = build_norm_vs_direction_shift(profile)
    corr = build_component_vs_method_delta(profile)
    decision = build_method_fit_decision(rep, norm, corr)

    weak_rep = rep[rep["category_group"].eq("s_cat_v2_weak")].iloc[0]
    assert math.isclose(weak_rep["delta_ndcg@20_mean"], -0.04)
    assert norm["norm_dominant_rate"].min() == 1.0
    assert "s_cat" in set(corr["v2_component"])
    assert decision["route"] == "method_changes_norm_not_direction"
    json.dumps(decision, ensure_ascii=False)
