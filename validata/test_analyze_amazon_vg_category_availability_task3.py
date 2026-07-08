#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 Task3 复评适配测试。

测试只覆盖稳定口径：
1. v2 availability 表转换为旧诊断脚本可读的 item_profile；
2. baseline 与 category_conf_input 的 v2 group delta 汇总。
"""

import math

import pandas as pd

from analyze_amazon_vg_category_availability_task3 import (
    build_task3_comparison_summary,
    build_task3_item_profile,
)


def test_build_task3_item_profile_uses_v2_group_and_test_items_only() -> None:
    availability = pd.DataFrame(
        [
            {
                "raw_asin": "train_a",
                "split": "train",
                "category_count": 3,
                "s_cat": 0.6,
                "s_cat_group": "s_cat_v2_strong",
                "s_cat_v1": 0.4,
                "s_cat_group_v1": "s_cat_weak",
                "s_cat_v2_disc_within_control": 0.7,
                "s_cat_v2_collab_within_control": 0.5,
            },
            {
                "raw_asin": "test_a",
                "split": "test",
                "category_count": 4,
                "s_cat": 0.2,
                "s_cat_group": "s_cat_v2_weak",
                "s_cat_v1": 0.8,
                "s_cat_group_v1": "s_cat_strong",
                "s_cat_v2_disc_within_control": 0.1,
                "s_cat_v2_collab_within_control": 0.3,
            },
        ]
    )
    test_rating = pd.DataFrame(
        [
            {"reviewerID": "u1", "asin": "test_a"},
            {"reviewerID": "u2", "asin": "test_a"},
            {"reviewerID": "u3", "asin": "other"},
        ]
    )

    profile = build_task3_item_profile(availability, test_rating)

    assert profile["asin"].tolist() == ["test_a"]
    row = profile.iloc[0]
    assert row["category_group"] == "s_cat_v2_weak"
    assert row["gt_user_count"] == 2
    assert row["gt_group"] == "gt_2_3"
    assert math.isclose(row["s_cat"], 0.2)
    assert row["s_cat_group_v1"] == "s_cat_strong"


def test_build_task3_comparison_summary_computes_group_delta() -> None:
    baseline = pd.DataFrame(
        [
            {"category_group": "s_cat_v2_weak", "item_count": 10, "hr@20_mean": 0.10, "ndcg@20_mean": 0.05},
            {"category_group": "s_cat_v2_strong", "item_count": 12, "hr@20_mean": 0.30, "ndcg@20_mean": 0.20},
        ]
    )
    category_conf = pd.DataFrame(
        [
            {"category_group": "s_cat_v2_weak", "item_count": 10, "hr@20_mean": 0.12, "ndcg@20_mean": 0.08},
            {"category_group": "s_cat_v2_strong", "item_count": 12, "hr@20_mean": 0.25, "ndcg@20_mean": 0.18},
        ]
    )

    summary = build_task3_comparison_summary(baseline, category_conf)

    weak = summary[summary["category_group"].eq("s_cat_v2_weak")].iloc[0]
    strong = summary[summary["category_group"].eq("s_cat_v2_strong")].iloc[0]

    assert math.isclose(weak["delta_ndcg@20_mean"], 0.03)
    assert math.isclose(weak["delta_hr@20_mean"], 0.02)
    assert math.isclose(strong["delta_ndcg@20_mean"], -0.02)
    assert math.isclose(strong["delta_hr@20_mean"], -0.05)
