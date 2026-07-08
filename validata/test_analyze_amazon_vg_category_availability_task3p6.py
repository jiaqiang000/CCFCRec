#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3.6 分组有效性诊断测试。
"""

import json
import math

import pandas as pd

from analyze_amazon_vg_category_availability_task3p6 import (
    build_baseline_metric_by_v2_group,
    build_controlled_metric_gap,
    build_cross_distribution,
    build_group_validity_decision,
    build_group_validity_profile,
)


def test_group_validity_profile_adds_old_group_and_activity_controls() -> None:
    item_profile = pd.DataFrame(
        [
            {"asin": "a", "category_group": "s_cat_v2_weak", "category_count": 2, "gt_group": "gt_1", "gt_user_count": 1, "s_cat": 0.2, "s_cat_group_v1": "s_cat_weak"},
            {"asin": "b", "category_group": "s_cat_v2_strong", "category_count": 7, "gt_group": "gt_4_plus", "gt_user_count": 5, "s_cat": 0.8, "s_cat_group_v1": "s_cat_strong"},
        ]
    )
    baseline = pd.DataFrame(
        [
            {"asin": "a", "hr@20": 0.2, "ndcg@20": 0.4},
            {"asin": "b", "hr@20": 0.1, "ndcg@20": 0.2},
        ]
    )
    content = pd.DataFrame(
        [
            {"asin": "a", "target_history_interaction_count_mean": 10.0},
            {"asin": "b", "target_history_interaction_count_mean": 200.0},
        ]
    )

    profile = build_group_validity_profile(item_profile, baseline, content)

    weak = profile[profile["asin"].eq("a")].iloc[0]
    strong = profile[profile["asin"].eq("b")].iloc[0]
    assert weak["category_count_bucket"] == "cat_weak_1_3"
    assert strong["category_count_bucket"] == "cat_strong_6_plus"
    assert weak["target_activity_bucket"] == "low"
    assert strong["target_activity_bucket"] == "high"
    assert math.isclose(weak["baseline_ndcg@20"], 0.4)


def test_cross_summary_controlled_gap_and_decision_detect_not_task_relevant() -> None:
    profile = pd.DataFrame(
        [
            {"asin": "a", "category_group": "s_cat_v2_weak", "gt_group": "gt_1", "category_count_bucket": "cat_weak_1_3", "target_activity_bucket": "low", "baseline_hr@20": 0.10, "baseline_ndcg@20": 0.30, "s_cat": 0.2},
            {"asin": "b", "category_group": "s_cat_v2_weak", "gt_group": "gt_2_3", "category_count_bucket": "cat_mid_4_5", "target_activity_bucket": "high", "baseline_hr@20": 0.20, "baseline_ndcg@20": 0.40, "s_cat": 0.3},
            {"asin": "c", "category_group": "s_cat_v2_strong", "gt_group": "gt_1", "category_count_bucket": "cat_strong_6_plus", "target_activity_bucket": "low", "baseline_hr@20": 0.05, "baseline_ndcg@20": 0.10, "s_cat": 0.8},
            {"asin": "d", "category_group": "s_cat_v2_strong", "gt_group": "gt_2_3", "category_count_bucket": "cat_strong_6_plus", "target_activity_bucket": "high", "baseline_hr@20": 0.10, "baseline_ndcg@20": 0.20, "s_cat": 0.9},
        ]
    )

    metric_summary = build_baseline_metric_by_v2_group(profile)
    cross = build_cross_distribution(profile, "gt_group")
    controlled = build_controlled_metric_gap(profile)
    decision = build_group_validity_decision(metric_summary, controlled)

    weak_summary = metric_summary[metric_summary["category_group"].eq("s_cat_v2_weak")].iloc[0]
    assert math.isclose(weak_summary["baseline_ndcg@20_mean"], 0.35)
    assert cross["count"].sum() == 4
    assert "gt_group" in set(controlled["control_set"])
    assert decision["route"] == "v2_group_not_task_relevant"
    json.dumps(decision, ensure_ascii=False)
