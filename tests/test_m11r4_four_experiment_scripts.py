import json
import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/run_m11r4_four_performance_first_seed43_fast_uniform_mps_100epoch.sh"
WATCHER = REPO_ROOT / "scripts/watch_m11r4_four_performance_first_logs.sh"

EXPECTED_VARIANTS = [
    "m11r4_protected_experts",
    "m11r4_continuous_fusion",
    "m11r4_relational_alignment",
    "m11r4_continuous_focal",
]


def _write_clean_profile(path: Path, *, include_forbidden: bool = False) -> None:
    rows = []
    for index, split in enumerate(["train", "train", "validate", "test"]):
        row = {
            "raw_asin": f"item_{index}",
            "split": split,
            "s_cat_v3": 0.8,
            "RSP_score": 0.3,
            "category_neighbor_mismatch_proxy_score": 0.7,
            "support_tail_proxy_score": 0.9,
            "m11_target_score": 0.75 if index % 2 == 0 else 0.45,
            "m11r1_full_target_flag": index in {0, 2, 3},
            "m11r1_full_target_loss_score": 0.75 if index in {0, 2, 3} else 0.0,
        }
        if include_forbidden:
            row["baseline_ndcg@20"] = 0.0
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_launcher_contains_four_distinct_100_epoch_runs_and_forecasts():
    text = LAUNCHER.read_text(encoding="utf-8")
    run_lines = re.findall(r"^run_one [1-4] .+$", text, flags=re.MULTILINE)

    assert len(run_lines) == 4
    assert [line.split()[3] for line in run_lines] == EXPECTED_VARIANTS
    assert 'EPOCH="${EPOCH:-100}"' in text
    assert 'if [[ "${EPOCH}" != "100" ]]' in text
    for index in range(1, 5):
        match = re.search(rf"FORECAST_E{index}_OVERALL_NDCG_PCT=([0-9.]+)", text)
        assert match is not None
        assert float(match.group(1)) > 3.0


def test_launcher_and_watcher_have_valid_bash_syntax():
    subprocess.run(["bash", "-n", str(LAUNCHER), str(WATCHER)], check=True, cwd=REPO_ROOT)


def test_launcher_rejects_non_100_epoch_before_profile_or_training():
    env = os.environ.copy()
    env.update({"M11R4_UNDER_CAFFEINATE": "1", "EPOCH": "40"})
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "requires EPOCH=100" in completed.stderr


def test_launcher_dry_run_excludes_test_rows_and_materializes_all_commands():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source_profile = root / "source_profile.csv"
        result = root / "result"
        _write_clean_profile(source_profile)
        env = os.environ.copy()
        env.update(
            {
                "M11R4_UNDER_CAFFEINATE": "1",
                "SOURCE_PROFILE": str(source_profile),
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
        training_profile = pd.read_csv(result / "protocol/m11r4_train_validate_only_profile.csv")
        audit = json.loads((result / "protocol/m11r4_profile_audit.json").read_text(encoding="utf-8"))
        command_text = "\n".join(
            path.read_text(encoding="utf-8") for path in sorted((result / "logs").glob("[1-4]_*.log"))
        )

    assert "DRY_RUN_DONE [4/4]" in completed.stdout
    assert status["state"].tolist() == ["dry_run"] * 4
    assert status["method_variant"].tolist() == EXPECTED_VARIANTS
    assert set(training_profile["split"]) == {"train", "validate"}
    assert audit["test_rows_passed_to_training"] == 0
    assert audit["validation_item_outcomes_passed_to_training"] is False
    assert audit["test_item_outcomes_read_or_generated"] is False
    for variant in EXPECTED_VARIANTS:
        assert variant in command_text
    assert "--epoch 100" in command_text


def test_launcher_rejects_evaluation_result_columns():
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        source_profile = root / "forbidden_profile.csv"
        _write_clean_profile(source_profile, include_forbidden=True)
        env = os.environ.copy()
        env.update(
            {
                "M11R4_UNDER_CAFFEINATE": "1",
                "SOURCE_PROFILE": str(source_profile),
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
