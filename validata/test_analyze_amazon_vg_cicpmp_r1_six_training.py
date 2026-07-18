from __future__ import annotations

import numpy as np
import pandas as pd

import analyze_amazon_vg_cicpmp_r1_six_training as analysis


def _item_metrics() -> pd.DataFrame:
    rows = []
    items = ["a", "b", "c", "d"]
    profiles = {
        "reliability_group": ["low", "mid", "high", "high"],
        "semantic_sign_group": ["non_positive", "positive", "positive", "positive"],
        "entropy_group": ["low", "low", "high", "high"],
        "direction_norm_group": ["low", "high", "low", "high"],
        "mp_raw_predicted_increment": [0.1, 0.2, 0.3, 0.4],
        "mp_category_semantic_increment_prediction": [-0.1, 0.1, 0.2, 0.3],
        "mp_category_total_increment_prediction": [0.1, 0.3, 0.5, 0.7],
        "mp_category_attribution_positive_share_prediction": [0.2, 0.4, 0.6, 0.8],
        "mp_category_attribution_entropy_prediction": [0.1, 0.2, 0.7, 0.8],
        "mp_reliability": [0.2, 0.5, 0.8, 0.9],
        "mp_direction_norm": [0.1, 0.8, 0.2, 0.9],
    }
    labels = [analysis.BASELINE_LABEL, *analysis.ITEM_METHOD_LABELS.values()]
    for label_index, label in enumerate(labels):
        for index, raw_asin in enumerate(items):
            row = {
                "raw_asin": raw_asin,
                "method_label": label,
                "ndcg@20": 0.1 + index * 0.01 + label_index * 0.001,
                "hr@20": 0.02 + index * 0.001 + label_index * 0.0001,
            }
            row.update({key: values[index] for key, values in profiles.items()})
            rows.append(row)
    return pd.DataFrame(rows)


def test_bh_adjust_is_monotonic_in_p_value_order() -> None:
    p_values = pd.Series([0.04, 0.001, 0.02, np.nan])
    adjusted = analysis._bh_adjust(p_values)
    ordered = adjusted.iloc[[1, 2, 0]].to_numpy()
    assert np.all(np.diff(ordered) >= 0)
    assert np.isnan(adjusted.iloc[3])


def test_group_summary_uses_complete_complementary_groups(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "VALIDATION_ITEM_COUNT", 4)
    frame = analysis.build_group_summary(
        _item_metrics(), analysis.METHOD_ORDER[0], repetitions=20
    )
    winner = frame[frame["method_variant"].eq(analysis.METHOD_ORDER[0])]
    assert winner[winner["group_dimension"].eq("overall")]["item_count"].item() == 4
    for dimension in analysis.GROUP_COLUMNS:
        assert winner[winner["group_dimension"].eq(dimension)]["item_count"].sum() == 4
    overall = winner[winner["group_dimension"].eq("overall")].iloc[0]
    assert np.isclose(overall["absolute_delta_ndcg@20"], 0.001)
    assert np.isfinite(overall["ndcg@20_bootstrap_absolute_ci95_low"])


def test_decision_stops_when_all_methods_are_below_two_pct(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "VALIDATION_ITEM_COUNT", 4)
    group_summary = analysis.build_group_summary(
        _item_metrics(), analysis.METHOD_ORDER[0], repetitions=20
    )
    curves = []
    for index, method in enumerate(analysis.METHOD_ORDER):
        curves.append(
            {
                "method_variant": method,
                "method_label": analysis.METHOD_LABELS[method],
                "best_epoch": 40,
                "best_ndcg@20": analysis.BASELINE_NDCG20 * (1.003 - index * 0.001),
                "absolute_delta_ndcg@20_vs_baseline_best": analysis.BASELINE_NDCG20 * (0.003 - index * 0.001),
                "relative_pct_ndcg@20_vs_baseline_best": 0.3 - index * 0.1,
                "best_hr@20_same_checkpoint": analysis.BASELINE_HR20 * 1.01,
                "absolute_delta_hr@20_vs_baseline_best": analysis.BASELINE_HR20 * 0.01,
                "relative_pct_hr@20_vs_baseline_best": 1.0,
                "relative_pct_vs_historical_cicpr1_e1": -1.0 - index,
                "late30_mean_ndcg@20": analysis.BASELINE_NDCG20 * 0.99,
                "late30_matched_positive_epochs": 10,
                "absolute_gap_to_three_pct": -0.003,
            }
        )
    score_response = analysis.build_score_response(_item_metrics())
    decision = analysis.build_decision(
        pd.DataFrame(curves), group_summary, score_response
    )
    assert decision["passed_two_pct"] is False
    assert decision["beat_historical_cicpr1_e1"] is False
    assert decision["automatic_followup_authorized"] is False
    assert decision["route"].startswith("stop_cicp_carrier_extensions")


def test_coverage_and_fallacy_scan_keep_protocol_boundaries(monkeypatch) -> None:
    monkeypatch.setattr(analysis, "VALIDATION_ITEM_COUNT", 4)
    group_summary = analysis.build_group_summary(
        _item_metrics(), analysis.METHOD_ORDER[0], repetitions=20
    )
    coverage = analysis.build_coverage(group_summary, analysis.METHOD_ORDER[0])
    full = coverage[coverage["evaluation_scope"].str.startswith("complete")].iloc[0]
    test = coverage[coverage["evaluation_scope"].str.startswith("test_cold")].iloc[0]
    target = coverage[coverage["evaluation_scope"].str.startswith("validation_target")].iloc[0]
    assert full["recommendation_metric_status"] == "未评估"
    assert test["recommendation_metric_status"] == "未评估"
    assert "未定义" in target["recommendation_metric_status"]
    assert len(analysis.build_fallacy_scan(group_summary, analysis.METHOD_ORDER[0])) == 11
