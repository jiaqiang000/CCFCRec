#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category ablation 诊断脚本的最小测试。

测试只覆盖稳定口径：
1. category ablation variant 输入构造；
2. weak-mid/strong gap 计算；
3. variant 相对 original 的 delta 计算。
"""

import math

import numpy as np
import pandas as pd
import torch

from analyze_amazon_vg_category_ablation import (
    build_delta_summary,
    build_variant_attribute_and_image_batch,
    build_variant_gap_summary,
    build_variant_group_summary,
)


def test_build_variant_attribute_and_image_batch() -> None:
    asins = ["a"]
    category_map = {"a": [1, 3]}
    img_feature_dict = {"a": np.ones(4096, dtype=np.float32)}
    device = torch.device("cpu")

    original_attr, original_img = build_variant_attribute_and_image_batch(
        asins,
        "original",
        5,
        category_map,
        img_feature_dict,
        device,
    )
    image_only_attr, image_only_img = build_variant_attribute_and_image_batch(
        asins,
        "image_only",
        5,
        category_map,
        img_feature_dict,
        device,
    )
    category_only_attr, category_only_img = build_variant_attribute_and_image_batch(
        asins,
        "category_only",
        5,
        category_map,
        img_feature_dict,
        device,
    )
    all_attr, _ = build_variant_attribute_and_image_batch(
        asins,
        "all_category_upper",
        5,
        category_map,
        img_feature_dict,
        device,
    )

    assert original_attr.tolist() == [[-1.0, 1.0, -1.0, 1.0, -1.0]]
    assert torch.all(image_only_attr.eq(-1.0))
    assert torch.allclose(image_only_img, original_img)
    assert torch.allclose(category_only_attr, original_attr)
    assert torch.all(category_only_img.eq(0.0))
    assert torch.all(all_attr.eq(1.0))


def make_profile() -> pd.DataFrame:
    rows = []
    for variant, offset in [("original", 0.0), ("image_only", -0.1)]:
        for group, score in [
            ("cat_weak_1_3", 1.0),
            ("cat_mid_4", 2.0),
            ("cat_strong_5_plus", 3.0),
        ]:
            rows.append(
                {
                    "asin": f"{variant}_{group}",
                    "variant": variant,
                    "category_group": group,
                    "ndcg@20": score / 10,
                    "target_score_max": score + offset,
                    "margin_to_top20_cutoff": score + offset - 0.5,
                    "best_target_rank": 10 - score,
                    "target_history_q_cosine_mean": score / 10 + offset,
                    "top20_history_q_cosine_mean": score / 10,
                    "target_minus_top20_history_q_cosine_mean": offset,
                    "q_norm": score,
                }
            )
    return pd.DataFrame(rows)


def test_build_variant_gap_summary() -> None:
    group_summary = build_variant_group_summary(make_profile())
    gap = build_variant_gap_summary(group_summary)
    original = gap[gap["variant"].eq("original")].iloc[0]

    assert math.isclose(original["mid_minus_weak_target_score_max"], 1.0)
    assert math.isclose(original["strong_minus_weak_target_score_max"], 2.0)
    assert math.isclose(original["mid_minus_weak_margin_to_top20_cutoff"], 1.0)


def test_build_delta_summary() -> None:
    profile = pd.DataFrame(
        [
            {
                "asin": "a",
                "variant": "original",
                "category_group": "cat_weak_1_3",
                "target_score_max": 1.0,
                "margin_to_top20_cutoff": 0.5,
                "best_target_rank": 10.0,
                "target_history_q_cosine_mean": 0.2,
                "top20_history_q_cosine_mean": 0.3,
                "target_minus_top20_history_q_cosine_mean": -0.1,
                "q_norm": 2.0,
            },
            {
                "asin": "a",
                "variant": "image_only",
                "category_group": "cat_weak_1_3",
                "target_score_max": 0.8,
                "margin_to_top20_cutoff": 0.1,
                "best_target_rank": 15.0,
                "target_history_q_cosine_mean": 0.1,
                "top20_history_q_cosine_mean": 0.3,
                "target_minus_top20_history_q_cosine_mean": -0.2,
                "q_norm": 1.5,
            },
        ]
    )

    delta = build_delta_summary(profile)
    image_only = delta[delta["variant"].eq("image_only")].iloc[0]

    assert math.isclose(image_only["delta_target_score_max_mean"], -0.2)
    assert math.isclose(image_only["delta_margin_to_top20_cutoff_mean"], -0.4)
    assert math.isclose(image_only["delta_best_target_rank_mean"], 5.0)
    assert math.isclose(image_only["delta_target_history_q_cosine_mean_mean"], -0.1)
