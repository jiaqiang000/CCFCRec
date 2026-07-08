#!/usr/bin/env python3
"""
CCFCRec Amazon-VG checkpoint selection 诊断脚本的最小测试。

测试只覆盖稳定口径：
1. 官方风格 item-level HR/NDCG；
2. checkpoint delta 汇总；
3. 分组 best checkpoint 选择。
"""

import math

import pandas as pd

from analyze_amazon_vg_checkpoint_selection import (
    METRICS,
    build_checkpoint_delta_summary,
    build_checkpoint_selection_summary,
    build_group_summary,
    build_overall_summary,
    compute_item_ranking_metrics,
    dcg_k,
)


def test_compute_item_ranking_metrics() -> None:
    recommended = [1, 2, 3, 4, 5]
    targets = [2, 4]

    result = compute_item_ranking_metrics(recommended, targets, ks=(2, 3, 5))

    assert result["hr@2"] == 0.5
    assert result["hr@3"] == 1 / 3
    assert result["hr@5"] == 2 / 5
    expected_ndcg_2 = (1 / math.log2(3)) / 1.0
    assert math.isclose(result["ndcg@2"], expected_ndcg_2)
    assert math.isclose(dcg_k([1.0, 0.0]), 1.0)


def make_profile() -> pd.DataFrame:
    rows = []
    for checkpoint_order, checkpoint_label, weak_ndcg, strong_ndcg in [
        (0, "epoch64_best_ndcg20", 0.10, 0.30),
        (1, "epoch100_last", 0.12, 0.25),
    ]:
        for asin, group, ndcg20 in [
            ("a", "cat_weak_1_3", weak_ndcg),
            ("b", "cat_strong_5_plus", strong_ndcg),
        ]:
            row = {
                "checkpoint_order": checkpoint_order,
                "checkpoint_label": checkpoint_label,
                "checkpoint_member": f"{checkpoint_label}.pt",
                "asin": asin,
                "category_count": 2 if group == "cat_weak_1_3" else 5,
                "category_group": group,
                "gt_user_count": 1,
                "gt_group": "gt_1",
                "raw_target_user_count": 1,
                "mapped_target_user_count": 1,
                "unknown_target_user_count": 0,
            }
            for metric in METRICS:
                row[metric] = ndcg20 if metric == "ndcg@20" else 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_build_checkpoint_delta_summary() -> None:
    delta = build_checkpoint_delta_summary(make_profile(), "epoch64_best_ndcg20", "epoch100_last")
    weak = delta[delta["group_value"].eq("cat_weak_1_3")].iloc[0]
    strong = delta[delta["group_value"].eq("cat_strong_5_plus")].iloc[0]

    assert math.isclose(weak["delta_ndcg@20_mean"], 0.02)
    assert math.isclose(strong["delta_ndcg@20_mean"], -0.05)
    assert weak["ndcg@20_improved_rate"] == 1.0
    assert strong["ndcg@20_declined_rate"] == 1.0


def test_build_checkpoint_selection_summary() -> None:
    profile = make_profile()
    overall = build_overall_summary(profile)
    category = build_group_summary(profile, "category_group")
    gt = build_group_summary(profile, "gt_group")
    selection = build_checkpoint_selection_summary(overall, category, gt)

    weak_ndcg = selection[
        selection["scope"].eq("category_group")
        & selection["group_value"].eq("cat_weak_1_3")
        & selection["metric"].eq("ndcg@20")
    ].iloc[0]
    strong_ndcg = selection[
        selection["scope"].eq("category_group")
        & selection["group_value"].eq("cat_strong_5_plus")
        & selection["metric"].eq("ndcg@20")
    ].iloc[0]

    assert weak_ndcg["best_checkpoint"] == "epoch100_last"
    assert strong_ndcg["best_checkpoint"] == "epoch64_best_ndcg20"
