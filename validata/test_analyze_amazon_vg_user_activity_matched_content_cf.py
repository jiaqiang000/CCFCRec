#!/usr/bin/env python3
"""
CCFCRec Amazon-VG user-activity matched content-CF 诊断脚本的最小测试。

测试只覆盖稳定口径：
1. activity 分桶字段是否产生；
2. matched bucket 内 strong/mid 减 weak 的 gap 是否正确；
3. 控制方式汇总是否按最小桶内样本阈值过滤。
"""

import math

import pandas as pd

from analyze_amazon_vg_user_activity_matched_content_cf import (
    build_matched_bucket_gap,
    prepare_profile,
    summarize_gap_by_control,
)


def make_profile() -> pd.DataFrame:
    rows = []
    for group, q_mean, ndcg in [
        ("cat_weak_1_3", 0.2, 0.1),
        ("cat_weak_1_3", 0.4, 0.2),
        ("cat_mid_4", 0.5, 0.3),
        ("cat_mid_4", 0.7, 0.4),
        ("cat_strong_5_plus", 0.8, 0.5),
        ("cat_strong_5_plus", 1.0, 0.6),
    ]:
        rows.append(
            {
                "category_group": group,
                "gt_group": "gt_1",
                "ndcg@20": ndcg,
                "target_history_q_cosine_mean": q_mean,
                "target_minus_top20_history_q_cosine_mean": q_mean - 0.1,
                "target_history_attr_cosine_mean": q_mean + 0.1,
                "target_history_img_cosine_mean": q_mean + 0.2,
                "target_history_interaction_count_mean": 10,
                "target_known_history_user_count": 2,
            }
        )
    return pd.DataFrame(rows)


def test_prepare_profile_adds_activity_buckets() -> None:
    profile = prepare_profile(make_profile())

    assert "target_history_interaction_bucket" in profile.columns
    assert "target_known_history_user_bucket" in profile.columns
    assert profile["target_history_interaction_bucket"].str.startswith("target_hist_interaction_").all()


def test_build_matched_bucket_gap_computes_group_differences() -> None:
    profile = prepare_profile(make_profile())
    matched = build_matched_bucket_gap(profile)
    row = matched[matched["control"].eq("gt_group")].iloc[0]

    assert row["cat_weak_1_3_count"] == 2
    assert row["cat_mid_4_count"] == 2
    assert row["cat_strong_5_plus_count"] == 2
    assert math.isclose(row["mid_minus_weak_target_history_q_cosine_mean"], 0.3)
    assert math.isclose(row["strong_minus_weak_target_history_q_cosine_mean"], 0.6)


def test_summarize_gap_by_control_filters_by_min_count() -> None:
    profile = prepare_profile(make_profile())
    matched = build_matched_bucket_gap(profile)
    summary = summarize_gap_by_control(matched, min_bucket_group_count=2)
    row = summary[
        summary["control"].eq("gt_group")
        & summary["metric"].eq("target_history_q_cosine_mean")
    ].iloc[0]

    assert row["valid_bucket_count"] == 1
    assert row["valid_item_count"] == 6
    assert math.isclose(row["mid_minus_weak_mean"], 0.3)
    assert math.isclose(row["strong_minus_weak_mean"], 0.6)
