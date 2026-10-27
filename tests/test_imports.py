import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_core_import_does_not_load_torch():
    script = """
import sys
import molopt
import molopt.algorithms as algorithms
from molopt.algorithms import GridSearch, LatinHypercube, NSGA2, RandomSearch
from molopt.algorithms.bayesian import BayesianOptimization
assert "torch" not in sys.modules
assert "botorch" not in sys.modules
assert molopt.Optimizer is not None
print("ok")
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout


def test_torch_is_mentioned_only_by_bayesian_optimization():
    package = ROOT / "molopt"
    pattern = re.compile(r"^\s*(?:import|from)\s+(?:torch|botorch|gpytorch)\b", re.MULTILINE)
    offenders = []
    for path in package.rglob("*.py"):
        if pattern.search(path.read_text()) and path.name != "bayesian.py":
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_package_does_not_reference_forbidden_stacks():
    package = ROOT / "molopt"
    banned = ("molnex", "molab", "molpy", "molq", "optuna", "pymoo", "sklearn")
    offenders = []
    for path in package.rglob("*.py"):
        lowered = path.read_text().lower()
        for name in banned:
            if name in lowered:
                offenders.append(f"{path.name}:{name}")
    assert offenders == []
