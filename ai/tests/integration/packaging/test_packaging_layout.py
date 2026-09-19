"""Packaging contract for the final MAT application layout.

Locks the console entry points, wheel contents, and mypy scope to the values
approved in the restructure plan, so later tasks cannot drift from them.
"""

import tomllib
from pathlib import Path

AI_ROOT = Path(__file__).resolve().parents[3]
PYPROJECT = tomllib.loads((AI_ROOT / "pyproject.toml").read_text(encoding="utf-8"))

EXPECTED_SCRIPTS = {
    "mat-validate-data": "scripts.data_validate:main",
    "mat-split-data": "scripts.data_split:main",
    "mat-train-baseline": "scripts.train_baseline:main",
    "mat-evaluate": "scripts.eval_predictions:main",
    "mat-demo-vietnamese": "scripts.demo_api_vietnamese:main",
    "mat-serve": "main:run",
}


def test_console_entry_points_target_final_scripts() -> None:
    assert PYPROJECT["project"]["scripts"] == EXPECTED_SCRIPTS


def test_wheel_installs_src_and_scripts_with_forced_root_modules() -> None:
    wheel = PYPROJECT["tool"]["hatch"]["build"]["targets"]["wheel"]
    assert wheel["packages"] == ["src", "scripts"]
    assert wheel["force-include"] == {"main.py": "main.py", "config.py": "config.py"}


def test_mypy_targets_final_packages_and_root_modules() -> None:
    mypy = PYPROJECT["tool"]["mypy"]
    # mypy rejects ``packages`` combined with ``files`` in one config, so the
    # same coverage (src + scripts packages and the root modules) is expressed
    # through ``files``.
    assert mypy["files"] == ["src", "scripts", "main.py", "config.py"]
    assert mypy["strict"] is True
