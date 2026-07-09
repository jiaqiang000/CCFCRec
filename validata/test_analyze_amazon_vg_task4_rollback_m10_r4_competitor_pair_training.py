import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from analyze_amazon_vg_task4_rollback_m10_r4_competitor_pair_training import (
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


def _fake_root() -> TemporaryDirectory:
    tmp = TemporaryDirectory()
    root = Path(tmp.name)
    _write_run(root, "2026-07-09_02_20_08", "task4_competitor_pair", [0.120, 0.125, 0.123], [0.019, 0.021, 0.020])
    _write_run(root, "2026-07-09_03_42_38", "task4_competitor_pair_shuffle", [0.119, 0.122, 0.123], [0.018, 0.019, 0.020])
    _write_run(root, "2026-07-09_05_01_23", "task4_competitor_pair_rsp_control", [0.119, 0.124, 0.122], [0.018, 0.020, 0.019])
    _write_run(root, "2026-07-09_06_19_21", "task4_competitor_pair_acat_control", [0.119, 0.1235, 0.122], [0.018, 0.020, 0.019])
    return tmp


def test_validation_curve_and_summary_preserve_run_roles_and_best_metrics() -> None:
    with _fake_root() as root_text:
        curve = build_validation_curve(Path(root_text))
        summary = build_validation_summary(curve)

    assert set(curve["role"]) == {"real", "shuffle", "rsp_control", "acat_control"}
    real = summary[summary["role"].eq("real")].iloc[0]
    assert real["best_ndcg@20"] == 0.125
    assert real["best_ndcg_epoch"] == 2
    assert real["best_ndcg_peak_minus_last"] == 0.002


def test_pair_comparison_applies_m9_and_target_gates() -> None:
    with _fake_root() as root_text:
        summary = build_validation_summary(build_validation_curve(Path(root_text)))
        comparison = build_pair_comparison(summary)

    real_minus_shuffle = comparison[comparison["comparison"].eq("real_minus_shuffle")].iloc[0]
    assert real_minus_shuffle["delta_best_ndcg@20"] == 0.002
    assert real_minus_shuffle["passes_m9_net_gate"] is True
    assert real_minus_shuffle["passes_target_net_gate"] is True


def test_route_opens_multiseed_only_when_shuffle_target_and_controls_pass() -> None:
    with _fake_root() as root_text:
        summary = build_validation_summary(build_validation_curve(Path(root_text)))
        comparison = build_pair_comparison(summary)
        decision = decide_route(summary, comparison)

    assert decision["route"] == "r4_competitor_pair_ready_for_multi_seed"
    assert decision["open_multi_seed"] is True


def test_route_marks_current_shape_as_weak_positive_not_multiseed_ready() -> None:
    summary = pd.DataFrame(
        [
            {"role": "real", "best_ndcg@20": 0.1234194167381099, "best_hr@20": 0.0206398640996602, "best_ndcg_epoch": 36, "max_epoch": 100, "last_ndcg@20": 0.1204533377998631},
            {"role": "shuffle", "best_ndcg@20": 0.1227076703545805, "best_hr@20": 0.0207153642884107, "best_ndcg_epoch": 83, "max_epoch": 100, "last_ndcg@20": 0.1200884604247730},
            {"role": "rsp_control", "best_ndcg@20": 0.1233655287408778, "best_hr@20": 0.0207153642884107, "best_ndcg_epoch": 40, "max_epoch": 100, "last_ndcg@20": 0.1195248562396895},
            {"role": "acat_control", "best_ndcg@20": 0.1232621332319969, "best_hr@20": 0.0206776141940354, "best_ndcg_epoch": 40, "max_epoch": 100, "last_ndcg@20": 0.1202216899599482},
        ]
    )
    comparison = build_pair_comparison(summary)
    decision = decide_route(summary, comparison)

    assert decision["route"] == "r4_competitor_pair_weak_positive_not_multiseed_ready"
    assert decision["open_multi_seed"] is False
    assert decision["real_minus_shuffle_best_ndcg@20"] > decision["m9_real_shuffle_gate_abs"]
    assert decision["real_minus_shuffle_best_ndcg@20"] < decision["target_net_gate_abs"]
