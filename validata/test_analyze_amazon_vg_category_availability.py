#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability 变量审查脚本的最小测试。

测试只覆盖稳定口径：
1. 输出 long-format Pearson / Spearman 相关性；
2. 按 split 和 s_cat_group 聚合数量；
3. 生成审查 MD、CSV 和 run_manifest.json。
"""

from __future__ import annotations

import json

import pandas as pd

from analyze_amazon_vg_category_availability import (
    build_group_summary,
    compute_correlations,
    write_analysis_outputs,
)


def make_availability() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "raw_asin": "a",
                "split": "train",
                "category_count": 1,
                "s_cat_group": "s_cat_weak",
                "s_cat": 0.10,
                "s_cat_gran": 0.10,
                "s_cat_disc": 0.10,
                "s_cat_collab": 0.10,
                "R_metadata_richness_score": 0.20,
                "S_train_support_score": 0.10,
                "P_popularity_score": 0.10,
            },
            {
                "raw_asin": "b",
                "split": "train",
                "category_count": 2,
                "s_cat_group": "s_cat_mid",
                "s_cat": 0.20,
                "s_cat_gran": 0.20,
                "s_cat_disc": 0.20,
                "s_cat_collab": 0.20,
                "R_metadata_richness_score": 0.30,
                "S_train_support_score": 0.20,
                "P_popularity_score": 0.20,
            },
            {
                "raw_asin": "c",
                "split": "validate",
                "category_count": 9,
                "s_cat_group": "s_cat_strong",
                "s_cat": 0.90,
                "s_cat_gran": 0.90,
                "s_cat_disc": 0.90,
                "s_cat_collab": 0.90,
                "R_metadata_richness_score": 0.40,
                "S_train_support_score": 0.90,
                "P_popularity_score": 0.90,
            },
        ]
    )


def test_compute_correlations_includes_controls_and_components() -> None:
    result = compute_correlations(make_availability())

    pairs = {(row.left, row.right) for row in result.itertuples()}

    assert ("s_cat", "category_count") in pairs
    assert ("s_cat", "R_metadata_richness_score") in pairs
    assert ("s_cat", "S_train_support_score") in pairs
    assert ("s_cat", "P_popularity_score") in pairs
    assert result["pearson"].notna().any()
    assert result["spearman"].notna().any()


def test_build_group_summary_counts_split_distribution() -> None:
    result = build_group_summary(make_availability())

    weak_train = result[result["s_cat_group"].eq("s_cat_weak")].iloc[0]
    strong_validate = result[result["s_cat_group"].eq("s_cat_strong")].iloc[0]

    assert weak_train["item_count"] == 1
    assert weak_train["train_item_count"] == 1
    assert strong_validate["validate_item_count"] == 1


def test_write_analysis_outputs(tmp_path) -> None:
    availability = make_availability()

    outputs = write_analysis_outputs(
        availability=availability,
        output_dir=tmp_path,
        source_path=tmp_path / "category_availability_item.csv",
        corr_warning_threshold=0.85,
        min_group_share=0.05,
    )

    assert outputs.audit_md.exists()
    assert outputs.audit_md.name.endswith("CCFCRec Amazon-VG category availability 变量审查结果.md")
    assert outputs.correlations_csv.exists()
    assert outputs.group_summary_csv.exists()
    assert outputs.run_manifest_json.exists()

    audit_text = outputs.audit_md.read_text(encoding="utf-8")
    assert "字段说明" in audit_text
    assert "category_count" in audit_text
    assert "warning" in audit_text.lower()

    manifest = json.loads(outputs.run_manifest_json.read_text(encoding="utf-8"))
    assert manifest["script"] == "analyze_amazon_vg_category_availability.py"
    assert manifest["input_rows"] == 3
