import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_demonstrator_fronts_match_the_oracle():
    completed = subprocess.run(
        [
            sys.executable,
            "examples/polymer_electrolyte.py",
            "--check",
            "--budget",
            "8",
            "--seeds",
            "1",
            "--batch",
            "4",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(completed.stdout)
    assert "random" in summary["algorithms"]
    assert "nsga2" in summary["algorithms"]
    assert summary["algorithms"]["random"]["median_hypervolume"] >= 0.0
