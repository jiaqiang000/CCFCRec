import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/run_m11r3_four_e4_followups_seed43_fast_uniform_mps_100epoch.sh"
WATCHER = REPO_ROOT / "scripts/watch_m11r3_four_e4_followups_logs.sh"

EXPECTED_VARIANTS = [
    "m11r3_dual_residual",
    "m11r3_norm_capped_residual",
    "m11r3_neighbor_transfer",
    "m11r3_target_film",
]


def _write_clean_profile(path: Path) -> None:
    rows = []
    for index, split in enumerate(["train", "validate", "test"]):
        rows.append(
            {
                "raw_asin": f"item_{index}",
                "split": split,
                "s_cat_v3": 0.8,
                "RSP_score": 0.3,
                "category_neighbor_mismatch_proxy_score": 0.7,
                "support_tail_proxy_score": 0.9,
                "m11_target_score": 0.75,
                "m11r1_full_target_flag": index == 0,
                "m11r1_full_target_loss_score": 0.75,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_launcher_contains_four_mechanism_distinct_100_epoch_runs() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    run_lines = re.findall(r"^run_one [1-4] .+$", text, flags=re.MULTILINE)

    assert len(run_lines) == 4
    assert [line.split()[3] for line in run_lines] == EXPECTED_VARIANTS
    assert 'EPOCH="${EPOCH:-100}"' in text
    assert 'if [[ "${EPOCH}" != "100" ]]' in text
    assert "TRAINING_INPUT_USES_TEST_ITEM_METRICS=false" in text


def test_launcher_and_watcher_have_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER), str(WATCHER)], check=True, cwd=REPO_ROOT)


def test_launcher_rejects_non_100_epoch_before_profile_or_training() -> None:
    env = os.environ.copy()
    env.update({"M11R3_UNDER_CAFFEINATE": "1", "EPOCH": "40"})
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "requires EPOCH=100" in completed.stderr


def test_launcher_dry_run_preflights_clean_profile_and_materializes_all_commands() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        profile = root / "profile.csv"
        result = root / "result"
        _write_clean_profile(profile)
        env = os.environ.copy()
        env.update(
            {
                "M11R3_UNDER_CAFFEINATE": "1",
                "TASK4_PROFILE": str(profile),
                "RESULT_ROOT": str(result),
                "DRY_RUN": "1",
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
        command_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted((result / "logs").glob("[1-4]_*.log")))

    assert "DRY_RUN_DONE [4/4]" in completed.stdout
    assert status["state"].tolist() == ["dry_run"] * 4
    assert status["method_variant"].tolist() == EXPECTED_VARIANTS
    for variant in EXPECTED_VARIANTS:
        assert variant in command_text
    assert "--epoch 100" in command_text
