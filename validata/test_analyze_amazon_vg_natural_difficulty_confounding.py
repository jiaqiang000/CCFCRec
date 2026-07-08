#!/usr/bin/env python3
"""
CCFCRec Amazon-VG 自然难度分层脚本的最小测试。

测试只覆盖稳定口径：
1. category_count 到 weak/mid/strong 的映射；
2. gt_user_count 分桶；
3. raw category 字符串解析。
4. 聚合时保留评估口径 gt_user_count 和原始 test 行数 raw_test_user_count。
"""

import pandas as pd

from analyze_amazon_vg_natural_difficulty_confounding import (
    METRICS,
    aggregate_metrics,
    category_group,
    gt_group,
    split_raw_category,
)


def test_category_group() -> None:
    assert category_group(1) == "cat_weak_1_3"
    assert category_group(3) == "cat_weak_1_3"
    assert category_group(4) == "cat_mid_4"
    assert category_group(5) == "cat_strong_5_plus"


def test_gt_group() -> None:
    assert gt_group(1) == "gt_1"
    assert gt_group(2) == "gt_2_3"
    assert gt_group(3) == "gt_2_3"
    assert gt_group(11) == "gt_4_11"
    assert gt_group(12) == "gt_12_plus"


def test_split_raw_category() -> None:
    assert split_raw_category("Video Games,PC,Games") == ["Video Games", "PC", "Games"]
    assert split_raw_category(" Video Games, , Accessories ") == ["Video Games", "Accessories"]
    assert split_raw_category("") == []


def test_aggregate_keeps_metric_gt_and_raw_test_count() -> None:
    rows = []
    for idx, raw_count in enumerate([3, 7]):
        row = {
            "category_group": "cat_weak_1_3",
            "gt_user_count": 2,
            "raw_test_user_count": raw_count,
            "raw_minus_metric_gt_user_count": raw_count - 2,
        }
        for metric in METRICS:
            row[metric] = float(idx)
        rows.append(row)
    result = aggregate_metrics(pd.DataFrame(rows), ["category_group"])

    assert result.loc[0, "gt_user_count_mean"] == 2
    assert result.loc[0, "raw_test_user_count_mean"] == 5
    assert result.loc[0, "raw_minus_metric_gt_user_count_mean"] == 3
