from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analyze_amazon_vg_cicp_mp_v1 import (
    DirectionResult,
    assert_no_recommendation_metrics,
    build_fallacy_scan,
    build_protocol_audit,
    build_shuffle_permutations,
    effective_dimension_audit,
    decide_route,
    finalize_component_catalog,
    uncertainty_calibration,
)


def test_shuffle_permutations_preserve_items_and_within_count_groups() -> None:
    counts = np.asarray([1, 1, 1, 2, 2, 2, 3, 3])
    permutations = build_shuffle_permutations(counts)
    expected = np.arange(len(counts))
    assert set(permutations) == {
        "global_seed43",
        "global_seed44",
        "within_category_count_seed43",
    }
    for values in permutations.values():
        assert np.array_equal(np.sort(values), expected)
        assert not np.any(values == expected)
    within = permutations["within_category_count_seed43"]
    assert np.array_equal(counts[within], counts)


def test_effective_dimension_detects_score_transforms_as_redundant() -> None:
    score = np.linspace(0.01, 0.99, 200)
    frame = pd.DataFrame(
        {
            "score": score,
            "inverse": 1.0 - score,
            "independent": np.sin(np.arange(len(score))),
        }
    )
    summary, correlation = effective_dimension_audit(
        frame, ["score", "inverse", "independent"], label="synthetic"
    )
    assert summary.iloc[0]["numerical_rank"] == 2
    assert np.isclose(abs(correlation.loc["score", "inverse"]), 1.0)
    assert summary.iloc[0]["effective_rank"] < 3.0


def test_uncertainty_calibration_reports_monotonic_error_signal() -> None:
    target = np.zeros(100)
    uncertainty = np.linspace(0.01, 1.0, 100)
    prediction = uncertainty.copy()
    disagreement = uncertainty * 0.5
    summary, bins = uncertainty_calibration(
        target, prediction, uncertainty, disagreement
    )
    assert summary.iloc[0]["uncertainty_vs_absolute_error_spearman"] > 0.99
    assert bins["absolute_error_mean"].is_monotonic_increasing


def test_recommendation_metric_columns_are_rejected() -> None:
    safe = pd.DataFrame({"raw_asin": ["a"], "cicp_score": [0.5]})
    assert_no_recommendation_metrics(safe, "safe")
    unsafe = safe.assign(**{"ndcg@20": [0.1]})
    with pytest.raises(ValueError, match="forbidden recommendation"):
        assert_no_recommendation_metrics(unsafe, "unsafe")


def test_fallacy_scan_covers_all_eleven_items() -> None:
    assert len(build_fallacy_scan()) == 11


def test_route_shift_gate_ignores_rejected_candidates() -> None:
    profile = pd.DataFrame(
        {
            "split": ["train"] * 24726 + ["validate"] * 5298,
            "mp_retained": np.zeros(30024),
        }
    )
    effective = pd.DataFrame(
        [
            {
                "component_count": 3,
                "effective_rank": 2.4,
                "pair_share_abs_spearman_ge_0_95": 0.0,
            }
        ]
    )
    mapping = pd.DataFrame(
        [
            {
                "target": "category_semantic_residual_ridge",
                "model": "hist_gradient_boosting",
                "oof_spearman": 0.25,
            }
        ]
    )
    uncertainty = pd.DataFrame(
        [{"uncertainty_vs_absolute_error_spearman": 0.3}]
    )
    direction = DirectionResult(
        train_compressed=np.empty((0, 0)),
        train_mapped_oof=np.empty((0, 0)),
        validation_mapped=np.empty((0, 0)),
        profile_summary=pd.DataFrame(),
        fold_summary=pd.DataFrame(),
        feasibility={"feasible": False},
    )
    shift = pd.DataFrame(
        [
            {"signal": "mp_retained", "ks_statistic": 0.02},
            {"signal": "mp_rejected", "ks_statistic": 0.40},
        ]
    )
    decision = decide_route(
        profile,
        effective,
        mapping,
        uncertainty,
        direction,
        shift,
        ["mp_retained"],
        True,
    )
    assert decision["route_number"] == 1
    assert decision["gates"]["train_validation_shift"] is True


def test_protocol_audit_records_coverage_and_nonzero_oof_uncertainty() -> None:
    profile = pd.DataFrame(
        {
            "split": ["train"] * 24726 + ["validate"] * 5298,
            "mp_retained": np.ones(30024),
            "mp_fold_prediction_uncertainty": np.linspace(0.01, 0.10, 30024),
        }
    )
    decision = {
        "route_number": 1,
        "gates": {"main_component_deployability": True},
    }
    audit, coverage = build_protocol_audit(profile, ["mp_retained"], decision)
    assert audit["coverage_pass"] is True
    assert audit["leakage_pass"] is True
    assert audit["train_uncertainty_zero_share"] == 0.0
    assert audit["recommendation_metrics_read_or_generated"] is False
    assert coverage["all_retained_values_finite"].all()


def test_component_catalog_records_final_independence_status() -> None:
    catalog = pd.DataFrame(
        {
            "component": ["mp_raw_predicted_increment", "1-s / s^2 / 4s(1-s)"],
            "definition": ["raw", "transforms"],
        }
    )
    decisions = pd.DataFrame(
        [
            {
                "component": "mp_raw_predicted_increment",
                "decision": "retain",
                "reason": "anchor",
            },
            {
                "component": "1-s / s^2 / 4s(1-s)",
                "decision": "reject_no_new_information",
                "reason": "deterministic transforms",
            },
        ]
    )
    result = finalize_component_catalog(catalog, decisions)
    assert result.loc[0, "independence_status"] == "retained_after_redundancy_audit"
    assert result.loc[1, "independence_status"] == "not_independent_deterministic_transform"
