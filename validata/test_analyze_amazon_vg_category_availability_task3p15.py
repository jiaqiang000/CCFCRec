#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.15 score calibration / norm-control 诊断测试。
"""

import json
import math

import pandas as pd

from analyze_amazon_vg_category_availability_task3p15 import (
    build_calibration_profile,
    build_norm_control_summary,
    build_route_decision,
    build_scale_direction_summary,
)


def test_calibration_profile_computes_norm_and_direction_shift() -> None:
    delta = pd.DataFrame(
        [
            {
                "asin": "a",
                "category_group": "s_cat_v2_weak",
                "delta_ndcg@20": -0.02,
                "delta_q_norm": -4.0,
                "delta_top20_user_norm_mean": 0.05,
                "delta_target_cosine_at_score_max": 0.01,
                "delta_target_history_q_cosine_mean": 0.02,
                "delta_target_minus_top20_history_q_cosine_mean": 0.01,
                "delta_target_minus_top20_cosine_mean": 0.00,
                "delta_margin_to_top20_cutoff": 0.1,
            }
        ]
    )

    profile = build_calibration_profile(delta)
    row = profile.iloc[0]

    assert math.isclose(row["norm_shift_score"], 4.0)
    assert math.isclose(row["direction_shift_score"], 0.04)
    assert row["norm_direction_ratio"] > 90
    assert row["norm_shift_bucket"] == "all"


def test_norm_control_summary_and_decision_detect_scale_artifact() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "category_group": "s_cat_v2_weak", "norm_shift_bucket": "high", "delta_ndcg@20": -0.05, "delta_margin_to_top20_cutoff": 0.05, "delta_top20_user_norm_mean": 0.04, "norm_shift_score": 4.0, "direction_shift_score": 0.04, "norm_direction_ratio": 100.0, "norm_dominant": True},
            {"asin": "b", "category_group": "s_cat_v2_mid", "norm_shift_bucket": "high", "delta_ndcg@20": -0.01, "delta_margin_to_top20_cutoff": 0.03, "delta_top20_user_norm_mean": 0.03, "norm_shift_score": 4.2, "direction_shift_score": 0.04, "norm_direction_ratio": 105.0, "norm_dominant": True},
            {"asin": "c", "category_group": "s_cat_v2_strong", "norm_shift_bucket": "low", "delta_ndcg@20": 0.01, "delta_margin_to_top20_cutoff": 0.02, "delta_top20_user_norm_mean": 0.01, "norm_shift_score": 0.2, "direction_shift_score": 0.10, "norm_direction_ratio": 2.0, "norm_dominant": False},
            {"asin": "d", "category_group": "s_cat_v2_strong", "norm_shift_bucket": "low", "delta_ndcg@20": 0.02, "delta_margin_to_top20_cutoff": 0.03, "delta_top20_user_norm_mean": 0.00, "norm_shift_score": 0.3, "direction_shift_score": 0.10, "norm_direction_ratio": 3.0, "norm_dominant": False},
        ]
    )

    norm_summary = build_norm_control_summary(profile)
    scale_summary = build_scale_direction_summary(profile)
    decision = build_route_decision(norm_summary, scale_summary)

    high = norm_summary[norm_summary["norm_shift_bucket"].eq("high")].iloc[0]
    assert math.isclose(high["delta_ndcg@20_mean"], -0.03)
    assert decision["route"] == "signal_is_scale_artifact"
    json.dumps(decision, ensure_ascii=False)
