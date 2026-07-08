#!/usr/bin/env python3
"""
CCFCRec Amazon-VG Task4-pre-3 train-safe hard proxy 构造审计测试。
"""

from __future__ import annotations

import json

import pandas as pd

from build_amazon_vg_task4_train_safe_hard_proxy import (
    CANDIDATE_SCORE_COLUMNS,
    SELECTED_CANDIDATE,
    build_candidate_summary,
    build_correlation_summary,
    build_overlap_summary,
    build_proxy_profile,
    decide_proxy_route,
)


def make_profile() -> pd.DataFrame:
    rows = []
    for idx in range(9):
        split = "train" if idx < 5 else ("validate" if idx < 7 else "test")
        high_acat = idx in {1, 3, 4, 6, 7, 8}
        eval_available = split == "test"
        rows.append(
            {
                "raw_asin": f"asin_{idx}",
                "split": split,
                "category_raw": f"cat_{idx % 3}",
                "category_tokens": f"Video Games|cat_{idx % 3}",
                "category_count": 3 + idx % 4,
                "s_cat_v3": 0.80 if high_acat else 0.20 + idx * 0.01,
                "s_cat_v3_group": "s_cat_v3_strong" if high_acat else "s_cat_v3_weak",
                "Acat_v3_disc_residual_pct": 0.78 if high_acat else 0.25,
                "Acat_v3_collab_residual_pct": 0.76 if high_acat else 0.30,
                "RSP_score": 0.25 if high_acat else 0.70,
                "RSP_group": "RSP_low" if high_acat else "RSP_high",
                "R_metadata_richness_score": 0.60,
                "S_train_support_score": 0.20 if high_acat else 0.85,
                "P_popularity_score": 0.22 if high_acat else 0.82,
                "S_train_token_user_support_mean": 10 if high_acat else 200,
                "S_train_token_interaction_support_mean": 20 if high_acat else 400,
                "A_collab_user_set_jaccard_mean": 0.10 if high_acat else 0.70,
                "A_collab_support_entropy_mean": 0.90 if high_acat else 0.10,
                "A_collab_train_token_user_support_mean": 9 if high_acat else 180,
                "baseline_ndcg@20": 0.0 if eval_available and high_acat else (0.4 if eval_available else pd.NA),
                "baseline_margin_proxy": -1.0 if eval_available and high_acat else (0.5 if eval_available else pd.NA),
                "baseline_best_target_rank": 900 if eval_available and high_acat else (20 if eval_available else pd.NA),
                "eval_metric_available_flag": eval_available,
                "eval_baseline_hard_flag": bool(eval_available and high_acat),
                "high_acat_flag": high_acat,
                "high_acat_eval_hard_flag": bool(eval_available and high_acat),
            }
        )
    return pd.DataFrame(rows)


def test_proxy_scores_do_not_change_when_eval_columns_are_perturbed() -> None:
    profile = make_profile()
    built = build_proxy_profile(profile)

    perturbed = profile.copy()
    perturbed["baseline_ndcg@20"] = 1.0
    perturbed["baseline_margin_proxy"] = 99.0
    perturbed["baseline_best_target_rank"] = 1
    perturbed["eval_baseline_hard_flag"] = False
    perturbed["high_acat_eval_hard_flag"] = False
    rebuilt = build_proxy_profile(perturbed)

    for column in CANDIDATE_SCORE_COLUMNS:
        assert built[column].round(12).tolist() == rebuilt[column].round(12).tolist()
    assert "train_safe_hard_proxy_score" in built.columns
    assert "high_acat_train_safe_hard_flag" in built.columns


def test_candidate_summary_and_overlap_expose_selected_proxy_gate_metrics() -> None:
    proxy_profile = build_proxy_profile(make_profile())
    candidate_summary = build_candidate_summary(proxy_profile)
    overlap = build_overlap_summary(proxy_profile)
    corr = build_correlation_summary(proxy_profile)

    selected = candidate_summary[candidate_summary["candidate"].eq(SELECTED_CANDIDATE)].iloc[0]
    selected_overlap = overlap[overlap["candidate"].eq(SELECTED_CANDIDATE)].iloc[0]

    assert selected["train_high_count"] > 0
    assert selected["high_acat_train_proxy_hard_count"] > 0
    assert selected_overlap["high_acat_eval_hard_capture_rate"] >= 0.5
    assert SELECTED_CANDIDATE in set(corr["candidate"])
    json.dumps(candidate_summary.to_dict(orient="records"), ensure_ascii=False)


def test_route_ready_for_m3_when_selected_proxy_is_large_and_not_rsp_copy() -> None:
    candidate_summary = pd.DataFrame(
        [
            {
                "candidate": SELECTED_CANDIDATE,
                "high_acat_train_proxy_hard_count": 800,
                "high_acat_eval_proxy_high_count": 500,
                "high_acat_proxy_high_eval_hard_rate": 0.85,
                "high_acat_eval_hard_base_rate": 0.80,
                "high_acat_eval_hard_capture_rate": 0.60,
                "spearman_vs_RSP_score": -0.40,
                "high_acat_proxy_high_rsp_high_share": 0.20,
            }
        ]
    )

    decision = decide_proxy_route(candidate_summary)

    assert decision["route"] == "train_safe_proxy_ready_for_m3"
    json.dumps(decision, ensure_ascii=False)


def test_route_rejects_rsp_copy() -> None:
    candidate_summary = pd.DataFrame(
        [
            {
                "candidate": SELECTED_CANDIDATE,
                "high_acat_train_proxy_hard_count": 800,
                "high_acat_eval_proxy_high_count": 500,
                "high_acat_proxy_high_eval_hard_rate": 0.85,
                "high_acat_eval_hard_base_rate": 0.80,
                "high_acat_eval_hard_capture_rate": 0.60,
                "spearman_vs_RSP_score": -0.90,
                "high_acat_proxy_high_rsp_high_share": 0.20,
            }
        ]
    )

    decision = decide_proxy_route(candidate_summary)

    assert decision["route"] == "train_safe_proxy_rsp_overlap_too_high"
