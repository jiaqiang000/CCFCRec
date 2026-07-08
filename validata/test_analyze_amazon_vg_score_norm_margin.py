#!/usr/bin/env python3
"""
CCFCRec Amazon-VG score/norm/margin 诊断脚本的最小测试。

测试只覆盖稳定口径：
1. target_score_max 与 top-k cutoff margin；
2. best_target_rank 的 1-based 排名；
3. 无可序列化 target user 时的 NaN 与 0 hit 处理。
"""

import math

import torch

from analyze_amazon_vg_score_norm_margin import compute_single_score_margin


def test_compute_single_score_margin_with_target() -> None:
    scores = torch.tensor([0.1, 0.5, 0.4, 0.2])
    user_norms = torch.tensor([1.0, 2.0, 3.0, 4.0])

    result = compute_single_score_margin(scores, [2], user_norms, top_k=2, analysis_top_k=3)

    assert math.isclose(result["top1_score"], 0.5, abs_tol=1e-6)
    assert math.isclose(result["score_at_20"], 0.4, abs_tol=1e-6)
    assert math.isclose(result["score_at_21"], 0.2, abs_tol=1e-6)
    assert math.isclose(result["target_score_max"], 0.4, abs_tol=1e-6)
    assert math.isclose(result["margin_to_top20_cutoff"], 0.0, abs_tol=1e-6)
    assert result["best_target_rank"] == 2
    assert result["target_hit_count_at20"] == 1
    assert math.isclose(result["target_hr_at20_like"], 0.5, abs_tol=1e-6)
    assert math.isclose(result["target_user_norm_mean"], 3.0, abs_tol=1e-6)


def test_compute_single_score_margin_without_target() -> None:
    scores = torch.tensor([0.1, 0.5, 0.4, 0.2])
    user_norms = torch.tensor([1.0, 2.0, 3.0, 4.0])

    result = compute_single_score_margin(scores, [], user_norms, top_k=2, analysis_top_k=3)

    assert math.isnan(result["target_score_max"])
    assert math.isnan(result["margin_to_top20_cutoff"])
    assert math.isnan(result["best_target_rank"])
    assert result["target_hit_count_at20"] == 0
    assert result["target_hr_at20_like"] == 0
