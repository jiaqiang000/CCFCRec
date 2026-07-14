import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/run_cicpr2_six_embedding_generation_seed43_fast_uniform_mps_100epoch.sh"
WATCHER = REPO_ROOT / "scripts/watch_cicpr2_six_embedding_generation_logs.sh"
EXPECTED_LABELS = [
    "CICP-R2-E1-CDR",
    "CICP-R2-E2-CID",
    "CICP-R2-E3-CMA",
    "CICP-R2-E4-SD",
    "CICP-R2-E5-OCS",
    "CICP-R2-E6-RCD",
]
EXPECTED_VARIANTS = [
    "cicpr2_content_direction_residual",
    "cicpr2_category_increment_gate",
    "cicpr2_cross_modal_attention",
    "cicpr2_score_distillation",
    "cicpr2_ordinal_counterfactual",
    "cicpr2_reliability_dropout",
]


def _write_profile(path: Path, *, include_forbidden: bool = False) -> None:
    rows = [
        {"raw_asin": "train_a", "split": "train", "cicp_score": 0.8},
        {"raw_asin": "train_b", "split": "train", "cicp_score": 0.2},
        {"raw_asin": "valid_a", "split": "validate", "cicp_score": 0.7},
        {"raw_asin": "test_a", "split": "test", "cicp_score": 0.6},
    ]
    if include_forbidden:
        for row in rows:
            row["baseline_ndcg@20"] = 0.0
    pd.DataFrame(rows).to_csv(path, index=False)


def test_launcher_uses_scoped_names_and_exactly_one_e4_style_residual():
    text = LAUNCHER.read_text(encoding="utf-8")
    run_lines = re.findall(r"^run_one [1-6] .+$", text, flags=re.MULTILINE)
    assert len(run_lines) == 6
    assert [line.split()[2] for line in run_lines] == EXPECTED_LABELS
    assert [line.split()[3] for line in run_lines] == EXPECTED_VARIANTS
    assert "E4_STYLE_RESIDUAL_BRANCH_COUNT=1" in text
    assert "E4_STYLE_RESIDUAL_BRANCH=CICP-R2-E1-CDR" in text
    assert sum("residual" in variant for variant in EXPECTED_VARIANTS) == 1


def test_launcher_and_watcher_have_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER), str(WATCHER)], check=True, cwd=REPO_ROOT)


def test_launcher_rejects_protocol_changes_before_profile_processing():
    for key, value, expected in [
        ("EPOCH", "40", "requires EPOCH=100"),
        ("SEED", "44", "requires SEED=43"),
        ("NUM_WORKERS", "4", "requires NUM_WORKERS=8"),
        ("BATCH_SIZE", "512", "requires BATCH_SIZE=1024"),
        ("NEGATIVE_SAMPLING_MODE", "legacy_cached", "requires NEGATIVE_SAMPLING_MODE=fast_uniform"),
    ]:
        env = os.environ.copy()
        env.update({"CICPR2_UNDER_CAFFEINATE": "1", key: value})
        completed = subprocess.run(
            ["bash", str(LAUNCHER)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
        )
        assert completed.returncode != 0
        assert expected in completed.stderr


def test_dry_run_materializes_six_named_commands_and_score_only_profile():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / "source.csv"
        result = root / "result"
        _write_profile(source)
        env = os.environ.copy()
        env.update(
            {
                "CICPR2_UNDER_CAFFEINATE": "1",
                "SOURCE_PROFILE": str(source),
                "RESULT_ROOT": str(result),
                "CICPR2_LATEST_POINTER": str(root / "latest_result_root.txt"),
                "DRY_RUN": "1",
                "CCFCREC_DEVICE": "mps",
                "PYTHON_BIN": "/opt/anaconda3/envs/ccfcrec-py3.11/bin/python",
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
            result / "protocol/cicpr2_train_validate_score_only_profile.csv"
        )
        audit = json.loads(
            (result / "protocol/cicpr2_profile_audit.json").read_text(encoding="utf-8")
        )
        manifest = (result / "launcher_manifest.env").read_text(encoding="utf-8")
        command_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((result / "logs").glob("[1-6]_CICP-R2-*.log"))
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
    assert set(clean_profile["split"]) == {"train", "validate"}
    assert clean_profile.columns.tolist() == ["raw_asin", "split", "cicp_score"]
    assert audit["independent_cicp_information_dimensions"] == 1
    assert audit["test_rows_passed_to_training"] == 0
    assert audit["validation_item_outcomes_passed_to_training"] is False
    assert "FROZEN_SIGNAL_INDEPENDENT_DIMENSIONS=1" in manifest
    for label, variant in zip(EXPECTED_LABELS, EXPECTED_VARIANTS):
        assert label in manifest
        assert variant in command_text
    assert "--epoch 100" in command_text
    assert "completed=0/6" in watcher.stdout


def test_launcher_rejects_evaluation_result_columns():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / "unsafe.csv"
        _write_profile(source, include_forbidden=True)
        env = os.environ.copy()
        env.update(
            {
                "CICPR2_UNDER_CAFFEINATE": "1",
                "SOURCE_PROFILE": str(source),
                "RESULT_ROOT": str(root / "result"),
                "CICPR2_LATEST_POINTER": str(root / "latest_result_root.txt"),
                "DRY_RUN": "1",
                "CCFCREC_DEVICE": "mps",
                "PYTHON_BIN": "/opt/anaconda3/envs/ccfcrec-py3.11/bin/python",
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
