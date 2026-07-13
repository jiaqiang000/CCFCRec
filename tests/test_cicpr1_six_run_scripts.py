import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/run_cicpr1_six_access_methods_seed43_fast_uniform_mps_100epoch.sh"
WATCHER = REPO_ROOT / "scripts/watch_cicpr1_six_access_methods_logs.sh"
EXPECTED_VARIANTS = [
    "cicpr1_e4_residual",
    "cicpr1_modality_routing",
    "cicpr1_category_expert",
    "cicpr1_alignment_curriculum",
    "cicpr1_counterfactual_margin",
    "cicpr1_adaptive_attention",
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


def test_launcher_contains_six_distinct_100_epoch_runs_and_one_e4_residual():
    text = LAUNCHER.read_text(encoding="utf-8")
    run_lines = re.findall(r"^run_one [1-6] .+$", text, flags=re.MULTILINE)
    assert len(run_lines) == 6
    assert [line.split()[3] for line in run_lines] == EXPECTED_VARIANTS
    assert len(set(EXPECTED_VARIANTS)) == 6
    assert sum("e4_residual" in variant for variant in EXPECTED_VARIANTS) == 1
    assert 'EPOCH="${EPOCH:-100}"' in text
    assert 'if [[ "${EPOCH}" != "100" ]]' in text
    assert "E4_STYLE_RESIDUAL_BRANCH_COUNT=1" in text


def test_launcher_and_watcher_have_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER), str(WATCHER)], check=True, cwd=REPO_ROOT)


def test_launcher_rejects_non_100_epoch_before_profile_processing():
    env = os.environ.copy()
    env.update({"CICPR1_UNDER_CAFFEINATE": "1", "EPOCH": "40"})
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert "requires EPOCH=100" in completed.stderr


def test_dry_run_excludes_test_rows_and_materializes_all_six_commands():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source = root / "source.csv"
        result = root / "result"
        _write_profile(source)
        env = os.environ.copy()
        env.update(
            {
                "CICPR1_UNDER_CAFFEINATE": "1",
                "SOURCE_PROFILE": str(source),
                "RESULT_ROOT": str(result),
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
            result / "protocol/cicpr1_train_validate_score_only_profile.csv"
        )
        audit = json.loads(
            (result / "protocol/cicpr1_profile_audit.json").read_text(encoding="utf-8")
        )
        command_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted((result / "logs").glob("[1-6]_*.log"))
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
    assert status["state"].tolist() == ["dry_run"] * 6
    assert status["method_variant"].tolist() == EXPECTED_VARIANTS
    assert set(clean_profile["split"]) == {"train", "validate"}
    assert clean_profile.columns.tolist() == ["raw_asin", "split", "cicp_score"]
    assert audit["test_rows_passed_to_training"] == 0
    assert audit["validation_item_outcomes_passed_to_training"] is False
    assert audit["test_item_outcomes_read_or_generated"] is False
    assert audit["m11_target_columns_passed_to_training"] is False
    for variant in EXPECTED_VARIANTS:
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
                "CICPR1_UNDER_CAFFEINATE": "1",
                "SOURCE_PROFILE": str(source),
                "RESULT_ROOT": str(root / "result"),
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
