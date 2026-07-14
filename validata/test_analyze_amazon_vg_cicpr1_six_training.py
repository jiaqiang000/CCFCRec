from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_amazon_vg_cicpr1_six_training import (
    BASELINE_LABEL,
    ITEM_METHOD_LABELS,
    _paired_bootstrap,
    build_decision,
    build_group_bootstrap,
    build_score_response,
    relative_pct,
)


def _item_fixture() -> pd.DataFrame:
    rows = []
    groups = ["low", "low", "mid", "mid", "high", "high"]
    scores = [0.1, 0.2, 0.4, 0.5, 0.8, 0.9]
    baseline = [0.05, 0.1, 0.2, 0.05, 0.3, 0.2]
    for index, (group, score, value) in enumerate(zip(groups, scores, baseline)):
        rows.append(
            {
                "raw_asin": str(index),
                "method_label": BASELINE_LABEL,
                "cicp_group": group,
                "cicp_score": score,
                "ndcg@20": value,
                "hr@20": value / 20.0,
            }
        )
    for method_index, (method, label) in enumerate(ITEM_METHOD_LABELS.items()):
        for index, (group, score, value) in enumerate(zip(groups, scores, baseline)):
            delta = 0.01 if method_index == 0 and group != "high" else 0.0
            rows.append(
                {
                    "raw_asin": str(index),
                    "method_label": label,
                    "cicp_group": group,
                    "cicp_score": score,
                    "ndcg@20": value + delta,
                    "hr@20": value / 20.0 + delta / 20.0,
                }
            )
    return pd.DataFrame(rows)


def test_relative_pct() -> None:
    assert np.isclose(relative_pct(1.03, 1.0), 3.0)
    assert np.isnan(relative_pct(1.0, 0.0))


def test_paired_bootstrap_is_deterministic_and_paired() -> None:
    method = np.asarray([0.2, 0.4, 0.6])
    baseline = np.asarray([0.1, 0.3, 0.5])
    first = _paired_bootstrap(method, baseline, seed=43, repetitions=100)
    second = _paired_bootstrap(method, baseline, seed=43, repetitions=100)
    assert first == second
    assert first["bootstrap_absolute_ci95_low"] > 0.09
    assert first["bootstrap_absolute_ci95_high"] < 0.11


def test_group_bootstrap_uses_complete_complementary_groups() -> None:
    summary = build_group_bootstrap(_item_fixture(), repetitions=100)
    e1 = summary[summary["method_variant"].eq("cicpr1_e4_residual")].set_index("cicp_group")
    assert e1.loc["overall", "item_count"] == 6
    assert e1.loc[["low", "mid", "high"], "item_count"].sum() == 6
    assert e1.loc["low", "absolute_delta_ndcg@20"] > 0
    assert e1.loc["high", "absolute_delta_ndcg@20"] == 0


def test_score_response_contains_baseline_and_all_methods() -> None:
    response = build_score_response(_item_fixture())
    assert set(response["method_variant"]) == {"baseline", *ITEM_METHOD_LABELS}
    assert len(response) == 14


def test_decision_requires_stability_and_clear_bottleneck_for_two_to_three_band() -> None:
    curve = pd.DataFrame(
        [
            {
                "method_variant": "cicpr1_e4_residual",
                "method_label": "E1",
                "best_epoch": 77,
                "best_ndcg@20": 0.12635,
                "best_hr@20_same_checkpoint": 0.0211,
                "relative_pct_ndcg@20_vs_baseline_best": 2.05,
                "relative_pct_hr@20_vs_baseline_best": 2.6,
                "absolute_gap_to_three_pct_threshold": -0.0011,
                "passed_three_pct_threshold": False,
                "late30_mean_ndcg@20": 0.1243,
                "late30_positive_ndcg_epochs_vs_baseline_best": 24,
                "late30_matched_positive_ndcg_epochs": 30,
            }
        ]
    )
    groups = pd.DataFrame(
        [
            {"method_variant": "cicpr1_e4_residual", "cicp_group": group, "relative_pct_ndcg@20": value}
            for group, value in (("overall", 2.05), ("low", 6.6), ("mid", 2.0), ("high", -0.6))
        ]
    )
    response = pd.DataFrame(
        [
            {
                "method_variant": "cicpr1_e4_residual",
                "response": "ndcg@20",
                "spearman": -0.01,
            }
        ]
    )
    decision = build_decision(curve, groups, response)
    assert decision["result_band"] == "between_2_and_3pct"
    assert decision["late_stability_pass"] is True
    assert decision["bottleneck_clarity_pass"] is False
    assert decision["automatic_followup_authorized"] is False
