from __future__ import annotations

import pandas as pd

from analyze_amazon_vg_cicpr2_six_embedding_generation import (
    BASELINE_HR20,
    BASELINE_NDCG20,
    METHOD_LABELS,
    METHOD_ORDER,
    build_decision,
    build_fallacy_scan,
)


def test_formal_method_names_are_round_qualified() -> None:
    assert len(METHOD_ORDER) == 6
    assert all(label.startswith("CICP-R2-E") for label in METHOD_LABELS.values())


def test_fallacy_scan_covers_all_eleven_types() -> None:
    scan = build_fallacy_scan()
    assert len(scan) == 11
    assert scan["fallacy"].nunique() == 11


def test_sub_three_pct_result_does_not_authorize_cicpr3() -> None:
    method = METHOD_ORDER[2]
    curve_summary = pd.DataFrame(
        [
            {
                "method_variant": method,
                "method_label": METHOD_LABELS[method],
                "best_epoch": 92,
                "best_ndcg@20": BASELINE_NDCG20 * 1.01,
                "best_hr@20_same_checkpoint": BASELINE_HR20 * 1.02,
                "relative_pct_ndcg@20_vs_baseline_best": 1.0,
                "relative_pct_hr@20_vs_baseline_best": 2.0,
                "late30_mean_ndcg@20": BASELINE_NDCG20 * 0.999,
                "late30_positive_ndcg_epochs_vs_baseline_best": 15,
                "late30_matched_positive_ndcg_epochs": 30,
            }
        ]
    )
    group_summary = pd.DataFrame(
        {
            "method_variant": [method] * 4,
            "cicp_group": ["overall", "low", "mid", "high"],
            "relative_pct_ndcg@20": [1.0, 2.0, 1.0, -0.5],
        }
    )
    score_response = pd.DataFrame(
        [
            {
                "method_variant": method,
                "response": "ndcg@20",
                "spearman": 0.01,
            }
        ]
    )
    decision = build_decision(curve_summary, group_summary, score_response)
    assert decision["passed_three_pct_threshold"] is False
    assert decision["beat_cicpr1_best"] is False
    assert decision["automatic_cicpr3_authorized"] is False
    assert decision["route"] == "stop_local_cicp_variants_rebuild_semantic_basis_or_backbone"
