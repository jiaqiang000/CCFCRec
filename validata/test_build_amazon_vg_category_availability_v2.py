#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v2 within-control 构造脚本测试。

测试只覆盖稳定口径：
1. 控制桶只从 train split 拟合；
2. within-control percentile 生成 0-1 分数；
3. v2 标准列覆盖 s_cat/s_cat_group，同时保留 v1。
"""

from __future__ import annotations

import pandas as pd

from build_amazon_vg_category_availability_v2 import (
    build_category_availability_v2,
    fit_control_bins,
    percentile_against_train_cells,
)


def make_v1() -> pd.DataFrame:
    rows = []
    for idx in range(12):
        support = idx / 11
        rows.append(
            {
                "raw_asin": f"item_{idx}",
                "split": "train" if idx < 8 else "test",
                "category_count": idx % 4 + 1,
                "R_metadata_richness_score": 0.2 + 0.03 * (idx % 3),
                "S_train_support_score": support,
                "P_popularity_score": support,
                "s_cat_gran": 1.0 - support,
                "s_cat_disc": 1.0 - support + 0.01 * (idx % 2),
                "s_cat_collab": 0.2 + 0.05 * (idx % 5),
                "s_cat": (1.0 - support + 1.0 - support + 0.01 * (idx % 2) + 0.2 + 0.05 * (idx % 5)) / 3,
                "s_cat_group": "old_group",
            }
        )
    return pd.DataFrame(rows)


def test_fit_control_bins_returns_labels_for_all_rows() -> None:
    frame = make_v1()
    bins, meta = fit_control_bins(frame)

    assert set(bins.columns) == {"category_count_bin", "R_bin", "S_bin", "P_bin"}
    assert len(bins) == len(frame)
    assert meta["fit_scope"] == "train_control_bins"


def test_percentile_against_train_cells_uses_train_distribution() -> None:
    frame = make_v1()
    bins, _ = fit_control_bins(frame)
    percentile = percentile_against_train_cells(frame, bins, score_col="s_cat_disc")

    assert percentile.between(0.0, 1.0).all()
    assert percentile.nunique() > 1


def test_build_category_availability_v2_preserves_v1_and_writes_standard_columns() -> None:
    frame = make_v1()
    result, meta = build_category_availability_v2(frame)

    assert "s_cat_v1" in result.columns
    assert "s_cat_group_v1" in result.columns
    assert "s_cat_v2" in result.columns
    assert "s_cat_v2_disc_within_control" in result.columns
    assert "s_cat_v2_collab_within_control" in result.columns
    assert result["s_cat"].equals(result["s_cat_v2"])
    assert set(result["s_cat_group"]).issubset({"s_cat_v2_weak", "s_cat_v2_mid", "s_cat_v2_strong"})
    assert meta["s_cat_policy"] == "v2_within_control_disc_collab_mean"
