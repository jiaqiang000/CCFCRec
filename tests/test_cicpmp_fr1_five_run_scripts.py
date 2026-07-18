import os
import re
import subprocess
import sys
from pathlib import Path

from test_cicpmp_fr1_five_final_repairs import make_source_profiles


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_cicpmp_fr1_five_final_repairs_seed43_fast_uniform_mps_100epoch.sh"
WATCHER = REPO_ROOT / "scripts" / "watch_cicpmp_fr1_five_final_repairs_logs.sh"


def test_launcher_freezes_five_runs_without_validation_derived_schedule():
    text = LAUNCHER.read_text(encoding="utf-8")
    run_lines = re.findall(r"^run_one [1-5] ", text, flags=re.MULTILINE)
    assert len(run_lines) == 5
    assert "EPOCH=100" in text
    assert '[[ "${EPOCH}" == "100" ]]' in text
    assert "epoch74" not in text.lower()
    assert "schedule_start" not in text
    assert "schedule_end" not in text
    assert "ACTIVATION_SCHEDULE=none" in text
    assert "E4_STYLE_HIDDEN_RESIDUAL_BRANCH_COUNT=1" in text
    assert text.count("cicpmp_fr1_scalar_residual_reference") >= 2
    assert "cicpmp_fr1_modality_film_shuffle shuffle" in text
    assert "--cicpmp_fr1_method_weight_decay" in text
    assert "--weight_decay" in text


def test_launcher_dry_run_builds_profiles_and_all_five_commands(tmp_path):
    scalar, mp = make_source_profiles()
    scalar_source = tmp_path / "scalar_source.csv"
    mp_source = tmp_path / "mp_source.csv"
    scalar.to_csv(scalar_source, index=False)
    mp.to_csv(mp_source, index=False)
    result_root = tmp_path / "result"
    pointer = tmp_path / "pointer.txt"
    env = os.environ.copy()
    env.update(
        {
            "DRY_RUN": "1",
            "CICPMP_FR1_UNDER_CAFFEINATE": "1",
            "CCFCREC_DEVICE": "mps",
            "PYTHON_BIN": sys.executable,
            "SCALAR_SOURCE_PROFILE": str(scalar_source),
            "MP_SOURCE_PROFILE": str(mp_source),
            "RESULT_ROOT": str(result_root),
            "CICPMP_FR1_LATEST_POINTER": str(pointer),
        }
    )
    completed = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "DRY_RUN_DONE [5/5]" in completed.stdout
    assert pointer.read_text(encoding="utf-8").strip() == str(result_root)

    status = (result_root / "status.tsv").read_text(encoding="utf-8")
    assert status.count("\tdry_run\t") == 5
    commands = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((result_root / "logs").glob("[1-5]_*.log"))
    )
    assert commands.count("--epoch 100") == 5
    assert commands.count("--cicpmp_fr1_method_weight_decay 0.0") == 5
    assert commands.count("--weight_decay 0.1") == 5
    assert commands.count("--cicp_profile_path") == 1
    assert commands.count("--cicp_mp_profile_path") == 4

    watcher = subprocess.run(
        ["bash", str(WATCHER)],
        cwd=REPO_ROOT,
        env={**env, "RESULT_ROOT": str(result_root)},
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert watcher.returncode == 0
    assert "completed=0/5" in watcher.stdout
