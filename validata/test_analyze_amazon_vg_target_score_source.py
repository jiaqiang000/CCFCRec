#!/usr/bin/env python3
"""
CCFCRec Amazon-VG target-score 来源诊断脚本的最小测试。

测试只覆盖稳定口径：
1. target_score 的 dot/cos/norm 分解；
2. 目标用户历史类别与当前 item 类别的 overlap / jaccard。
"""

import math

import torch

from analyze_amazon_vg_target_score_source import category_overlap_stats, compute_alignment_stats


def test_compute_alignment_stats_decomposes_score_norm_cosine() -> None:
    q_vec = torch.tensor([1.0, 0.0])
    user_embedding = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
        ]
    )
    user_norms = torch.norm(user_embedding, dim=1)

    result, top_indices = compute_alignment_stats(q_vec, user_embedding, [1, 2], user_norms, top_k=2)

    assert top_indices == [2, 0]
    assert result["target_score_max"] == 2.0
    assert result["target_score_mean"] == 1.0
    assert result["target_cosine_at_score_max"] == 1.0
    assert result["target_cosine_max"] == 1.0
    assert result["target_cosine_mean"] == 0.5
    assert result["target_user_norm_at_score_max"] == 2.0
    assert result["target_user_norm_mean"] == 1.5
    assert result["top20_cosine_mean"] == 1.0
    assert result["top20_user_norm_mean"] == 1.5
    assert result["target_minus_top20_cosine_mean"] == -0.5


def test_category_overlap_stats() -> None:
    histories = {
        "u1": {"items": {"a", "b"}, "categories": {1, 3}, "interaction_count": 2},
        "u2": {"items": {"c"}, "categories": {4}, "interaction_count": 1},
    }
    result = category_overlap_stats(["u1", "u2"], {1, 2}, histories)

    assert result["history_known_user_count"] == 2
    assert math.isclose(result["history_category_overlap_rate_mean"], 0.25)
    assert math.isclose(result["history_category_overlap_rate_max"], 0.5)
    assert math.isclose(result["history_category_jaccard_mean"], 1 / 6)
    assert math.isclose(result["history_category_jaccard_max"], 1 / 3)
    assert math.isclose(result["history_item_count_mean"], 1.5)
    assert math.isclose(result["history_interaction_count_mean"], 1.5)
