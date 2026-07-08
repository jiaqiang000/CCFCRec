#!/usr/bin/env python3
"""
CCFCRec Amazon-VG category availability v3 purity audit 测试。

v3 的关键约束：
1. s_cat_v3 只能来自 metadata/train-graph category evidence；
2. 不允许 rank/NDCG/margin/recoverability 列影响变量定义；
3. v3 必须审计自己是否退化为 R/S/P。
"""

from __future__ import annotations

import json

import pandas as pd

from build_amazon_vg_category_availability_v3 import (
    _md_table,
    build_category_availability_v3,
    build_group_summary,
    build_purity_correlations,
    decide_v3_route,
    train_percentile,
)


def make_v2_like_frame() -> pd.DataFrame:
    rows = []
    category_signal = [0.05, 0.90, 0.15, 0.85, 0.20, 0.80, 0.25, 0.75, 0.30, 0.70, 0.35, 0.65]
    for idx, signal in enumerate(category_signal):
        support = idx / (len(category_signal) - 1)
        rows.append(
            {
                "raw_asin": f"item_{idx}",
                "split": "train" if idx < 9 else "test",
                "category_count": 2 + (idx % 4),
                "R_metadata_richness_score": 0.2 + 0.03 * (idx % 3),
                "S_train_support_score": support,
                "P_popularity_score": support,
                "A_gran_specific_ratio": 0.75 * support + 0.25 * signal,
                "A_gran_idf_mean": 1.0 + 2.5 * support + signal,
                "A_gran_idf_max": 2.0 + 3.0 * support + signal,
                "A_disc_combo_rarity": signal,
                "A_disc_generic_token_ratio": 1.0 - signal,
                "A_disc_specificity_score": 0.20 * support + 0.80 * signal,
                "A_collab_user_set_jaccard_mean": signal,
                "A_collab_support_entropy_mean": 1.0 - signal,
                "s_cat": support,
                "s_cat_group": "s_cat_v2_mid",
                "ndcg@20": 999.0 - idx,
                "margin_proxy": 888.0 - idx,
                "proxy_ensemble_score": 777.0 - idx,
            }
        )
    return pd.DataFrame(rows)


def test_train_percentile_uses_train_distribution_and_bounds_all_rows() -> None:
    frame = make_v2_like_frame()
    pct = train_percentile(frame["A_disc_combo_rarity"], frame["split"].eq("train"))

    assert pct.between(0.0, 1.0).all()
    assert pct.iloc[1] > pct.iloc[0]
    assert pct.iloc[-1] <= 1.0


def test_build_category_availability_v3_residualizes_against_rsp_controls() -> None:
    frame = make_v2_like_frame()
    result, meta = build_category_availability_v3(frame)

    assert "s_cat_v3" in result.columns
    assert "Acat_v3_gran_residual_pct" in result.columns
    assert "Acat_v3_disc_residual_pct" in result.columns
    assert "Acat_v3_collab_residual_pct" in result.columns
    assert result["s_cat_v3"].between(0.0, 1.0).all()
    assert set(result["s_cat_v3_group"]).issubset({"s_cat_v3_weak", "s_cat_v3_mid", "s_cat_v3_strong"})
    assert meta["s_cat_policy"] == "v3_rsp_residualized_category_evidence_mean"
    assert meta["leakage_policy"]["recoverability_columns"] == "not_used_for_feature_construction"

    raw_corr = result["Acat_v3_gran_raw"].corr(result["S_train_support_score"], method="spearman")
    residual_corr = result["Acat_v3_gran_residual_pct"].corr(result["S_train_support_score"], method="spearman")
    assert abs(residual_corr) < abs(raw_corr)


def test_recoverability_columns_do_not_change_v3_scores() -> None:
    frame = make_v2_like_frame()
    changed = frame.copy()
    changed["ndcg@20"] = list(reversed(frame["ndcg@20"].to_list()))
    changed["margin_proxy"] = list(reversed(frame["margin_proxy"].to_list()))
    changed["proxy_ensemble_score"] = list(reversed(frame["proxy_ensemble_score"].to_list()))

    left, _ = build_category_availability_v3(frame)
    right, _ = build_category_availability_v3(changed)

    pd.testing.assert_series_equal(left["s_cat_v3"], right["s_cat_v3"], check_names=False)
    pd.testing.assert_series_equal(left["s_cat_v3_group"], right["s_cat_v3_group"], check_names=False)


def test_purity_correlation_and_decision_detect_rsp_collapse() -> None:
    frame = make_v2_like_frame()
    frame["s_cat_v3"] = frame["S_train_support_score"]
    correlations = build_purity_correlations(frame)
    summary = pd.DataFrame(
        [
            {"s_cat_v3_group": "s_cat_v3_weak", "item_count": 4, "ndcg@20_mean": 0.1, "margin_proxy_mean": -1.0},
            {"s_cat_v3_group": "s_cat_v3_strong", "item_count": 4, "ndcg@20_mean": 0.5, "margin_proxy_mean": 1.0},
        ]
    )

    decision = decide_v3_route(correlations, summary)

    assert decision["route"] == "acat_v3_collapses_to_rsp"
    assert decision["evidence"]["max_abs_spearman_control"] >= 0.70
    json.dumps(decision, ensure_ascii=False)


def test_group_summary_and_decision_accept_pure_easy_relevant_signal() -> None:
    frame = pd.DataFrame(
        [
            {"raw_asin": "a", "s_cat_v3_group": "s_cat_v3_weak", "ndcg@20": 0.05, "margin_proxy": -1.2, "proxy_ensemble_score": 0.8},
            {"raw_asin": "b", "s_cat_v3_group": "s_cat_v3_weak", "ndcg@20": 0.10, "margin_proxy": -1.0, "proxy_ensemble_score": 0.7},
            {"raw_asin": "c", "s_cat_v3_group": "s_cat_v3_strong", "ndcg@20": 0.35, "margin_proxy": 0.7, "proxy_ensemble_score": 0.2},
            {"raw_asin": "d", "s_cat_v3_group": "s_cat_v3_strong", "ndcg@20": 0.40, "margin_proxy": 0.9, "proxy_ensemble_score": 0.1},
        ]
    )
    correlations = pd.DataFrame(
        [
            {"left": "s_cat_v3", "right": "category_count", "abs_spearman": 0.10, "right_family": "control"},
            {"left": "s_cat_v3", "right": "R_metadata_richness_score", "abs_spearman": 0.12, "right_family": "control"},
            {"left": "s_cat_v3", "right": "S_train_support_score", "abs_spearman": 0.08, "right_family": "control"},
            {"left": "s_cat_v3", "right": "P_popularity_score", "abs_spearman": 0.11, "right_family": "control"},
        ]
    )
    summary = build_group_summary(frame)

    assert "ndcg20_count" in summary.columns
    assert int(summary["ndcg20_count"].sum()) == 4
    decision = decide_v3_route(correlations, summary)

    assert decision["route"] == "acat_v3_pure_baseline_easy_relevant"
    assert decision["evidence"]["strong_minus_weak_ndcg@20"] > 0.20


def test_decision_separates_high_acat_baseline_failure_direction() -> None:
    correlations = pd.DataFrame(
        [
            {"left": "s_cat_v3", "right": "category_count", "abs_spearman": 0.10, "right_family": "control"},
            {"left": "s_cat_v3", "right": "R_metadata_richness_score", "abs_spearman": 0.12, "right_family": "control"},
            {"left": "s_cat_v3", "right": "S_train_support_score", "abs_spearman": 0.08, "right_family": "control"},
            {"left": "s_cat_v3", "right": "P_popularity_score", "abs_spearman": 0.11, "right_family": "control"},
        ]
    )
    summary = pd.DataFrame(
        [
            {"s_cat_v3_group": "s_cat_v3_weak", "item_count": 10, "ndcg20_mean": 0.30, "margin_proxy_mean": -0.2},
            {"s_cat_v3_group": "s_cat_v3_strong", "item_count": 10, "ndcg20_mean": 0.10, "margin_proxy_mean": -0.8},
        ]
    )

    decision = decide_v3_route(correlations, summary)

    assert decision["route"] == "acat_v3_pure_baseline_failure_relevant"
    assert decision["evidence"]["relevance_direction"] == "high_acat_baseline_harder"


def test_md_table_does_not_require_optional_tabulate_dependency() -> None:
    table = _md_table(pd.DataFrame([{"a": 1, "b": "x"}]), ["a", "b"])

    assert "| a | b |" in table
    assert "| 1 | x |" in table
