#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability 敏感性审查脚本的最小测试。

测试只覆盖稳定口径：
1. 生成原始候选和 residual 候选分数；
2. residual 候选降低与控制变量的相关性；
3. 输出 decision、CSV、run_manifest 和带来源说明的结果 MD。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analyze_amazon_vg_category_availability_sensitivity import (
    build_sensitivity_scores,
    evaluate_candidates,
    residualize_score,
    write_sensitivity_outputs,
)


def make_availability() -> pd.DataFrame:
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
            }
        )
    return pd.DataFrame(rows)


def test_residualize_score_scales_residual_to_unit_interval() -> None:
    frame = make_availability()
    residual = residualize_score(
        frame,
        score_col="s_cat",
        control_cols=["category_count", "R_metadata_richness_score", "S_train_support_score", "P_popularity_score"],
    )

    assert residual.between(0.0, 1.0).all()
    assert residual.nunique() > 1


def test_build_sensitivity_scores_adds_expected_candidates() -> None:
    result = build_sensitivity_scores(make_availability())

    expected = {
        "score_s_cat_current",
        "score_s_cat_no_disc",
        "score_s_cat_no_gran",
        "score_s_cat_no_collab",
        "score_s_cat_resid_controls",
        "score_s_cat_no_disc_resid_controls",
        "score_s_cat_no_gran_resid_controls",
        "score_s_cat_component_resid_mean",
    }

    assert expected.issubset(set(result.columns))
    assert result[list(expected)].notna().all().all()


def test_evaluate_candidates_marks_residual_candidate_as_diagnostic_ready() -> None:
    scores = build_sensitivity_scores(make_availability())
    evaluation = evaluate_candidates(scores, control_threshold=0.70, component_threshold=0.90, min_group_share=0.10)

    decision = evaluation["decision"]
    candidates = evaluation["candidate_summary"]

    assert decision["route"] in {"ready_for_task3_diagnostic_only", "needs_variable_v2", "ready_for_task3_raw"}
    assert "score_s_cat_resid_controls" in set(candidates["candidate"])
    residual = candidates[candidates["candidate"].eq("score_s_cat_resid_controls")].iloc[0]
    assert residual["control_max_abs_spearman"] < 0.70


def test_write_sensitivity_outputs(tmp_path) -> None:
    availability = make_availability()
    source = tmp_path / "category_availability_item.csv"
    source.write_text("placeholder", encoding="utf-8")

    outputs = write_sensitivity_outputs(
        availability=availability,
        source_path=source,
        output_dir=tmp_path,
        upstream_design_links=[
            "2026-07-04 233711 CCFCRec Amazon-VG category availability 敏感性审查诊断设计"
        ],
    )

    assert outputs.scores_csv.exists()
    assert outputs.correlations_csv.exists()
    assert outputs.group_summary_csv.exists()
    assert outputs.decision_json.exists()
    assert outputs.result_md.exists()
    assert outputs.run_manifest_json.exists()
    assert outputs.result_md.name.endswith("CCFCRec Amazon-VG category availability 敏感性审查结果.md")

    decision = json.loads(outputs.decision_json.read_text(encoding="utf-8"))
    assert "route" in decision
    result_text = outputs.result_md.read_text(encoding="utf-8")
    assert "来源说明" in result_text
    assert "[[2026-07-04 233711 CCFCRec Amazon-VG category availability 敏感性审查诊断设计]]" in result_text
