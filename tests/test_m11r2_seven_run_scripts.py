import os
import re
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/run_m11r2_seven_experiments_seed43_fast_uniform_mps_100epoch.sh"
WATCHER = REPO_ROOT / "scripts/watch_m11r2_seven_experiments_logs.sh"


EXPECTED_VARIANTS = [
    "m11r2_qbpr_score_weight",
    "m11r2_qbpr_focal",
    "m11r2_qbpr_curriculum",
    "m11r2_target_feature_fusion",
    "m11r1_full_target_competitor_pair",
    "m11r1_popmatch_competitor_pair_control",
    "m11r1_lowacat_competitor_pair_control",
]


def test_launcher_contains_exactly_four_performance_and_three_reference_control_runs() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    run_lines = re.findall(r"^run_one [1-7] .+$", text, flags=re.MULTILINE)

    assert len(run_lines) == 7
    assert [line.split()[3] for line in run_lines] == EXPECTED_VARIANTS
    assert 'EPOCH="${EPOCH:-100}"' in text
    assert 'if [[ "${EPOCH}" != "100" ]]' in text
    assert "m11r2_seven_run_100epoch_design_ready" in text


def test_launcher_and_watcher_have_valid_bash_syntax() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER), str(WATCHER)], check=True, cwd=REPO_ROOT)


def test_launcher_rejects_non_100_epoch_before_training() -> None:
    env = os.environ.copy()
    env.update({"M11R2_UNDER_CAFFEINATE": "1", "EPOCH": "40"})
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert completed.returncode != 0
    assert "requires EPOCH=100" in completed.stderr


def test_watcher_reports_status_and_latest_epoch_snapshot() -> None:
    with TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "result"
        logs = root / "logs"
        logs.mkdir(parents=True)
        (root / "status.tsv").write_text(
            "run_index\trun_label\tmethod_variant\tstate\tstarted_at\tended_at\n"
            "1\tM11R2E1_qbpr_score\tm11r2_qbpr_score_weight\trunning\tstart\t\n",
            encoding="utf-8",
        )
        (logs / "master.log").write_text(
            "START [1/7] M11R2E1_qbpr_score m11r2_qbpr_score_weight\n",
            encoding="utf-8",
        )
        (logs / "1_M11R2E1_qbpr_score_m11r2_qbpr_score_weight.log").write_text(
            "[epoch 12/100][batch 300/326][ckpt 12] total_loss:1.0, elapsed:10s\n",
            encoding="utf-8",
        )
        pointer = Path(tmpdir) / "latest.txt"
        pointer.write_text(str(root) + "\n", encoding="utf-8")
        env = os.environ.copy()
        env.update({"LATEST_FILE": str(pointer), "FOLLOW": "0"})

        completed = subprocess.run(
            ["bash", str(WATCHER)],
            cwd=REPO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

    assert "M11R2E1_qbpr_score" in completed.stdout
    assert "[epoch 12/100]" in completed.stdout
    assert "START [1/7]" in completed.stdout
