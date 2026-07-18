import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/run_cicpmp_r1_six_mechanisms_seed43_fast_uniform_mps_100epoch.sh"
WATCHER = REPO_ROOT / "scripts/watch_cicpmp_r1_six_mechanisms_logs.sh"
PYTHON_BIN = "/opt/anaconda3/envs/ccfcrec-py3.11/bin/python"
EXPECTED_LABELS = [
    "CICP-MP-R1-E1-RRA",
    "CICP-MP-R1-E2-DTA",
    "CICP-MP-R1-E3-AEC",
    "CICP-MP-R1-E4-RCE",
    "CICP-MP-R1-E5-CCI",
    "CICP-MP-R1-E6-DHN",
]
EXPECTED_VARIANTS = [
    "cicpmp_r1_reliable_residual",
    "cicpmp_r1_direction_alignment",
    "cicpmp_r1_attention_entropy",
    "cicpmp_r1_reliable_expert",
    "cicpmp_r1_counterfactual_calibration",
    "cicpmp_r1_direction_hard_negative",
]


def _write_profile(path: Path, *, include_forbidden: bool = False) -> None:
    rows = []
    for index, split in enumerate(["train", "train", "validate", "validate"]):
        row = {
            "raw_asin": f"item_{index}",
            "split": split,
            "mp_raw_predicted_increment": 0.01 * index,
            "mp_category_semantic_increment_prediction": 0.02 * index,
            "mp_category_total_increment_prediction": 0.03 * index,
            "mp_category_attribution_positive_share_prediction": 0.5 + 0.1 * index,
            "mp_category_attribution_entropy_prediction": 0.4 + 0.1 * index,
            "mp_fold_prediction_uncertainty": 0.001 + 0.001 * index,
            "mp_hgb_ridge_disagreement": 0.003 + 0.001 * index,
        }
        for direction_index in range(16):
            row[f"mp_direction16_{direction_index:02d}"] = (
                (index + 1) * (direction_index + 1) * 0.0001
            )
        if include_forbidden:
            row["baseline_ndcg@20"] = 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_launcher_has_six_scoped_mechanisms_and_exactly_one_e4_style_residual():
    text = LAUNCHER.read_text(encoding="utf-8")
    run_lines = re.findall(r"^run_one [1-6] .+$", text, flags=re.MULTILINE)
    assert len(run_lines) == 6
    assert [line.split()[2] for line in run_lines] == EXPECTED_LABELS
    assert [line.split()[3] for line in run_lines] == EXPECTED_VARIANTS
    assert "E4_STYLE_RESIDUAL_BRANCH_COUNT=1" in text
    assert "E4_STYLE_RESIDUAL_BRANCH=CICP-MP-R1-E1-RRA" in text
    assert sum("residual" in variant for variant in EXPECTED_VARIANTS) == 1


def test_launcher_and_watcher_have_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER), str(WATCHER)], check=True)


def test_launcher_rejects_protocol_changes_before_profile_processing():
    for key, value, expected in [
        ("EPOCH", "40", "requires EPOCH=100"),
        ("SEED", "44", "requires SEED=43"),
        ("NUM_WORKERS", "4", "requires NUM_WORKERS=8"),
        ("BATCH_SIZE", "512", "requires BATCH_SIZE=1024"),
        (
            "NEGATIVE_SAMPLING_MODE",
            "legacy_cached",
            "requires NEGATIVE_SAMPLING_MODE=fast_uniform",
        ),
        (
            "CICPMP_RESIDUAL_MAX_RATIO",
            "0.20",
            "requires CICPMP_RESIDUAL_MAX_RATIO=0.15",
        ),
        (
            "CICPMP_DIRECTION_WEIGHT",
            "0.10",
            "requires CICPMP_DIRECTION_WEIGHT=0.05",
        ),
    ]:
        env = os.environ.copy()
        env.update(
            {
                "CICPMP_R1_UNDER_CAFFEINATE": "1",
                "CCFCREC_DEVICE": "mps",
                key: value,
            }
        )
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        assert completed.returncode != 0
        assert expected in completed.stderr


def test_dry_run_materializes_profile_initialization_audit_and_six_commands():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / "source.csv"
        result = root / "result"
        _write_profile(source)
        env = os.environ.copy()
        env.update(
            {
                "CICPMP_R1_UNDER_CAFFEINATE": "1",
                "SOURCE_PROFILE": str(source),
                "RESULT_ROOT": str(result),
                "CICPMP_R1_LATEST_POINTER": str(root / "latest_result_root.txt"),
                "DRY_RUN": "1",
                "CCFCREC_DEVICE": "mps",
                "PYTHON_BIN": PYTHON_BIN,
            }
        )
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        status = pd.read_csv(result / "status.tsv", sep="\t")
        clean_profile = pd.read_csv(
            result / "protocol/cicpmp_v1_train_validate_23d_profile.csv"
        )
        profile_audit = json.loads(
            (result / "protocol/cicpmp_v1_profile_audit.json").read_text(
                encoding="utf-8"
            )
        )
        init_audit = json.loads(
            (result / "protocol/cicpmp_r1_common_initialization_audit.json").read_text(
                encoding="utf-8"
            )
        )
        manifest = (result / "launcher_manifest.env").read_text(encoding="utf-8")
        command_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((result / "logs").glob("[1-6]_CICP-MP-R1-*.log"))
        )
        watcher = subprocess.run(
            ["bash", str(WATCHER)],
            cwd=REPO_ROOT,
            env={**env, "RESULT_ROOT": str(result)},
            text=True,
            capture_output=True,
            check=True,
        )

    assert "DRY_RUN_DONE [6/6]" in completed.stdout
    assert status["run_label"].tolist() == EXPECTED_LABELS
    assert status["method_variant"].tolist() == EXPECTED_VARIANTS
    assert status["state"].tolist() == ["dry_run"] * 6
    assert clean_profile.shape == (4, 25)
    assert list(clean_profile.columns[:2]) == ["raw_asin", "split"]
    assert profile_audit["retained_feature_count"] == 23
    assert profile_audit["test_rows_passed_to_training"] == 0
    assert profile_audit["validation_item_outcomes_passed_to_training"] is False
    assert init_audit["common_hash_unique_count"] == 1
    assert init_audit["post_initialization_rng_hash_unique_count"] == 1
    assert init_audit["common_parameters_elementwise_equal"] is True
    assert "FROZEN_SIGNAL_DIMENSIONS=23" in manifest
    assert "PREFLIGHT_COMMON_PARAMETER_HASH_UNIQUE_COUNT=1" in manifest
    assert "PREFLIGHT_POST_INITIALIZATION_RNG_HASH_UNIQUE_COUNT=1" in manifest
    assert "ACTUAL_RUN_CONFIG_AUDIT=" in manifest
    assert "verify_actual_run_config" in LAUNCHER.read_text(encoding="utf-8")
    for label, variant in zip(EXPECTED_LABELS, EXPECTED_VARIANTS):
        assert label in manifest
        assert variant in command_text
    assert "--epoch 100" in command_text
    assert "--seed 43" in command_text
    assert "completed=0/6" in watcher.stdout
    assert "preflight common initialization audit" in watcher.stdout
    assert "completed-run actual initialization and parameter audit" in watcher.stdout


def test_launcher_rejects_evaluation_result_columns():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / "unsafe.csv"
        _write_profile(source, include_forbidden=True)
        env = os.environ.copy()
        env.update(
            {
                "CICPMP_R1_UNDER_CAFFEINATE": "1",
                "SOURCE_PROFILE": str(source),
                "RESULT_ROOT": str(root / "result"),
                "CICPMP_R1_LATEST_POINTER": str(root / "latest_result_root.txt"),
                "DRY_RUN": "1",
                "CCFCREC_DEVICE": "mps",
                "PYTHON_BIN": PYTHON_BIN,
            }
        )
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
    assert completed.returncode != 0
    assert "forbidden evaluation-result columns" in completed.stderr
