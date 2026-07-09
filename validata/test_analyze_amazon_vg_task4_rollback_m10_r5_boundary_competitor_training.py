import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from analyze_amazon_vg_task4_rollback_m10_r5_boundary_competitor_training import (
    build_baseline_comparison,
    build_baseline_summary,
    build_pair_comparison,
    build_validation_curve,
    build_validation_summary,
    decide_route,
)


def _write_run(root: Path, dirname: str, method_variant: str, ndcg_values: list[float], hr_values: list[float]) -> None:
    run_dir = root / dirname
    run_dir.mkdir(parents=True)
    (run_dir / "run_config.json").write_text(
        json.dumps(
            {
                "method_variant": method_variant,
                "task4_competitor_alpha": 0.25,
                "task4_competitor_margin": 0.1,
                "task4_competitor_k": 20,
                "negative_sampling_mode": "fast_uniform",
                "seed": 43,
                "num_workers": 8,
                "epoch": len(ndcg_values),
                "batch_size": 1024,
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "checkpoint_index": idx,
                "epoch": idx,
                "batch": 300,
                "total_batches": 326,
                "elapsed_s": 40 + idx,
                "loss": 1200.0 - idx,
                "contrast_sum": 1300.0 + idx,
                "hr@5": 0.01,
                "hr@10": 0.02,
                "hr@20": hr,
                "ndcg@5": 0.09,
                "ndcg@10": 0.10,
                "ndcg@20": ndcg,
            }
            for idx, (ndcg, hr) in enumerate(zip(ndcg_values, hr_values), start=1)
        ]
    ).to_csv(run_dir / "result.csv", index=False)


def _fake_root_ready() -> TemporaryDirectory:
    tmp = TemporaryDirectory()
    root = Path(tmp.name)
    _write_run(root, "real", "task4_boundary_competitor_pair", [0.120, 0.126, 0.124], [0.019, 0.023, 0.020])
    _write_run(root, "shuffle", "task4_boundary_competitor_pair_shuffle", [0.119, 0.122, 0.123], [0.018, 0.019, 0.020])
    _write_run(root, "rsp", "task4_boundary_competitor_pair_rsp_control", [0.119, 0.123, 0.122], [0.018, 0.020, 0.019])
    _write_run(root, "acat", "task4_boundary_competitor_pair_acat_control", [0.119, 0.124, 0.122], [0.018, 0.021, 0.019])
    return tmp


def test_validation_curve_maps_r5_roles() -> None:
    with _fake_root_ready() as root_text:
        curve = build_validation_curve(Path(root_text))

    assert set(curve["role"]) == {"real", "shuffle", "rsp_control", "acat_control"}
    assert set(curve["run_label"]) == {"R5a025_real", "R5a025_shuffle", "R5a025_rsp_control", "R5a025_acat_control"}


def test_route_opens_multiseed_only_when_real_beats_controls_and_peak_not_early() -> None:
    summary = pd.DataFrame(
        [
            {"role": "real", "best_ndcg@20": 0.126, "best_hr@20": 0.023, "best_ndcg_epoch": 30, "max_epoch": 100, "last_ndcg@20": 0.124, "completed_expected_epoch": True, "early_transient_peak_flag": False},
            {"role": "shuffle", "best_ndcg@20": 0.123, "best_hr@20": 0.020, "best_ndcg_epoch": 30, "max_epoch": 100, "last_ndcg@20": 0.121, "completed_expected_epoch": True, "early_transient_peak_flag": False},
            {"role": "rsp_control", "best_ndcg@20": 0.124, "best_hr@20": 0.021, "best_ndcg_epoch": 30, "max_epoch": 100, "last_ndcg@20": 0.120, "completed_expected_epoch": True, "early_transient_peak_flag": False},
            {"role": "acat_control", "best_ndcg@20": 0.124, "best_hr@20": 0.021, "best_ndcg_epoch": 30, "max_epoch": 100, "last_ndcg@20": 0.120, "completed_expected_epoch": True, "early_transient_peak_flag": False},
        ]
    )
    decision = decide_route(summary, build_pair_comparison(summary))

    assert decision["route"] == "r5_boundary_competitor_ready_for_multi_seed"
    assert decision["open_multi_seed"] is True


def test_route_marks_current_shape_as_rsp_control_dominated() -> None:
    summary = pd.DataFrame(
        [
            {"role": "real", "best_ndcg@20": 0.0902441629793257, "best_hr@20": 0.016, "best_ndcg_epoch": 18, "max_epoch": 100, "last_ndcg@20": 0.0864128803896509, "completed_expected_epoch": True, "early_transient_peak_flag": True},
            {"role": "shuffle", "best_ndcg@20": 0.0859686171470794, "best_hr@20": 0.015, "best_ndcg_epoch": 16, "max_epoch": 100, "last_ndcg@20": 0.0838482586521712, "completed_expected_epoch": True, "early_transient_peak_flag": True},
            {"role": "rsp_control", "best_ndcg@20": 0.0954757633587378, "best_hr@20": 0.017, "best_ndcg_epoch": 48, "max_epoch": 100, "last_ndcg@20": 0.0938842935429026, "completed_expected_epoch": True, "early_transient_peak_flag": False},
            {"role": "acat_control", "best_ndcg@20": 0.0890701801806842, "best_hr@20": 0.016, "best_ndcg_epoch": 16, "max_epoch": 100, "last_ndcg@20": 0.0849536394352615, "completed_expected_epoch": True, "early_transient_peak_flag": True},
        ]
    )
    comparison = build_pair_comparison(summary)
    decision = decide_route(summary, comparison)

    assert decision["passes_target_net_gate"] is True
    assert decision["not_losing_rsp_control"] is False
    assert decision["not_losing_acat_control"] is True
    assert decision["route"] == "r5_boundary_competitor_rsp_control_dominated"
    assert decision["open_multi_seed"] is False


def test_baseline_comparison_reports_absolute_and_percent_delta() -> None:
    with TemporaryDirectory() as tmpdir:
        baseline_csv = Path(tmpdir) / "baseline_result.csv"
        pd.DataFrame(
            [
                {"checkpoint_index": 1, "epoch": 1, "hr@20": 0.10, "ndcg@20": 0.20},
                {"checkpoint_index": 2, "epoch": 2, "hr@20": 0.12, "ndcg@20": 0.25},
            ]
        ).to_csv(baseline_csv, index=False)
        baseline = build_baseline_summary(baseline_csv)

    summary = pd.DataFrame(
        [
            {
                "run_label": "R5a025_real",
                "role": "real",
                "method_variant": "task4_boundary_competitor_pair",
                "best_ndcg@20": 0.20,
                "last_ndcg@20": 0.18,
                "best_hr@20": 0.09,
                "last_hr@20": 0.08,
            }
        ]
    )
    comparison = build_baseline_comparison(summary, baseline)
    row = comparison.iloc[0]
    decision = decide_route(
        pd.DataFrame(
            [
                {"role": "real", "best_ndcg@20": 0.20, "best_hr@20": 0.09, "best_ndcg_epoch": 30, "last_ndcg@20": 0.18, "completed_expected_epoch": True, "early_transient_peak_flag": False},
                {"role": "shuffle", "best_ndcg@20": 0.10, "best_hr@20": 0.08, "best_ndcg_epoch": 30, "last_ndcg@20": 0.10, "completed_expected_epoch": True, "early_transient_peak_flag": False},
                {"role": "rsp_control", "best_ndcg@20": 0.19, "best_hr@20": 0.08, "best_ndcg_epoch": 30, "last_ndcg@20": 0.10, "completed_expected_epoch": True, "early_transient_peak_flag": False},
                {"role": "acat_control", "best_ndcg@20": 0.19, "best_hr@20": 0.08, "best_ndcg_epoch": 30, "last_ndcg@20": 0.10, "completed_expected_epoch": True, "early_transient_peak_flag": False},
            ]
        ),
        pd.DataFrame(
            [
                {"comparison": "real_minus_shuffle", "delta_best_ndcg@20": 0.10},
                {"comparison": "real_minus_rsp_control", "delta_best_ndcg@20": 0.01},
                {"comparison": "real_minus_acat_control", "delta_best_ndcg@20": 0.01},
            ]
        ),
        comparison,
    )

    assert baseline["best_ndcg@20"] == 0.25
    assert row["delta_best_ndcg@20_vs_baseline"] == -0.05
    assert row["pct_best_ndcg@20_vs_baseline"] == -20.0
    assert decision["real_minus_baseline_best_ndcg@20"] == -0.05
    assert decision["real_pct_best_ndcg@20_vs_baseline"] == -20.0


def test_baseline_summary_records_validation_cold_start_scope() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        baseline_csv = root / "baseline_result.csv"
        validate_csv = root / "validate_rating.csv"
        train_csv = root / "train_rating.csv"
        pd.DataFrame(
            [
                {"checkpoint_index": 1, "epoch": 1, "hr@20": 0.10, "ndcg@20": 0.20},
                {"checkpoint_index": 2, "epoch": 2, "hr@20": 0.12, "ndcg@20": 0.25},
            ]
        ).to_csv(baseline_csv, index=False)
        pd.DataFrame(
            [
                {"reviewerID": "u1", "asin": "val-a", "rating": 5.0},
                {"reviewerID": "u2", "asin": "val-a", "rating": 4.0},
                {"reviewerID": "u3", "asin": "val-b", "rating": 5.0},
            ]
        ).to_csv(validate_csv, index=False)
        pd.DataFrame(
            [
                {"reviewerID": "u1", "asin": "train-a", "rating": 5.0},
                {"reviewerID": "u2", "asin": "train-b", "rating": 4.0},
            ]
        ).to_csv(train_csv, index=False)

        baseline = build_baseline_summary(baseline_csv, validate_csv, train_csv)

    comparison = build_baseline_comparison(
        pd.DataFrame(
            [
                {
                    "run_label": "R5a025_real",
                    "role": "real",
                    "method_variant": "task4_boundary_competitor_pair",
                    "best_ndcg@20": 0.20,
                    "last_ndcg@20": 0.18,
                    "best_hr@20": 0.09,
                    "last_hr@20": 0.08,
                }
            ]
        ),
        baseline,
    )

    assert baseline["metric_scope"] == "validation_split_all_unique_cold_start_items"
    assert baseline["evaluation_item_count"] == 2
    assert baseline["evaluation_interaction_count"] == 3
    assert baseline["train_item_overlap_count"] == 0
    assert baseline["is_cold_start_item_split"] is True
    assert comparison.iloc[0]["metric_scope"] == "validation_split_all_unique_cold_start_items"
    assert comparison.iloc[0]["evaluation_item_count"] == 2


def test_route_marks_incomplete_when_required_role_missing() -> None:
    summary = pd.DataFrame(
        [
            {"role": "real", "best_ndcg@20": 0.1},
            {"role": "shuffle", "best_ndcg@20": 0.09},
        ]
    )
    decision = decide_route(summary, pd.DataFrame())

    assert decision["route"] == "r5_incomplete_or_failed"
    assert decision["missing_roles"] == ["rsp_control", "acat_control"]
