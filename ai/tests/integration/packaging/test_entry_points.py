"""Entry-point and root-composition contracts for the installed application.

Locks console scripts to their own CLI boundary, the root ``main``/``config``
modules to their single responsibilities, and proves the installed package
works from outside the repository working directory.
"""

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

AI_ROOT = Path(__file__).resolve().parents[3]

WRAPPER_TARGETS = (
    "data_validate",
    "data_split",
    "train_baseline",
    "eval_predictions",
    "demo_api_vietnamese",
)


@pytest.mark.parametrize("wrapper", WRAPPER_TARGETS)
def test_wrapper_delegates_once_to_relocated_implementation(
    wrapper: str,
) -> None:
    path = AI_ROOT / "scripts" / f"{wrapper}.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    main_definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "main"
    ]
    imported_nlp_mains = [
        alias
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module is not None
        and (node.module == "src.nlp" or node.module.startswith("src.nlp."))
        for alias in node.names
        if alias.name == "main"
    ]

    assert len(main_definitions) == 1, f"scripts/{wrapper}.py must define its own main"
    assert not imported_nlp_mains, f"scripts/{wrapper}.py must not import src.nlp main"


def test_root_main_exposes_asgi_app_and_run() -> None:
    main = importlib.import_module("main")
    config = importlib.import_module("config")
    settings = importlib.import_module("src.core.settings")
    server = importlib.import_module("src.api.server")

    assert hasattr(main, "app"), "main.py must expose the ASGI app"
    assert callable(main.run), "main.py must expose the uvicorn runner"
    assert main.create_app is server.create_app, "main.py must delegate composition"
    assert config.Settings is settings.Settings, "config.py must re-export Settings"


def test_installed_package_works_outside_repository(tmp_path: Path) -> None:
    """Subprocess proof from outside the repo: imports and console --help resolve."""
    assert tmp_path != AI_ROOT
    env_python = sys.executable
    imports = subprocess.run(
        [env_python, "-c", "import src, main, config; print(src.__version__)"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert imports.returncode == 0, imports.stderr
    assert imports.stdout.strip() == "0.1.0"

    for console_script in (
        "mat-validate-data",
        "mat-split-data",
        "mat-train-baseline",
        "mat-evaluate",
        "mat-demo-vietnamese",
    ):
        result = subprocess.run(
            [console_script, "--help"],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert result.returncode == 0, f"{console_script}: {result.stderr}"
        assert "usage" in result.stdout.lower(), console_script


def test_serve_entry_point_resolves_without_starting() -> None:
    """``mat-serve`` maps to ``main:run``; resolution is proven without calling it.

    Calling ``run()`` would start uvicorn, so only import resolution and the
    metadata entry point are asserted here (existing CLI behavior is kept).
    """
    from importlib.metadata import entry_points

    main = importlib.import_module("main")
    assert callable(main.run)
    matches = [ep for ep in entry_points(group="console_scripts") if ep.name == "mat-serve"]
    assert matches, "mat-serve console script must be installed"
    assert matches[0].value == "main:run"


def test_built_wheel_installs_and_answers_help_outside_repository(
    tmp_path: Path,
) -> None:
    """The published artifact is the wheel, not the working tree: build it,
    install it into a fresh virtualenv, and run every non-destructive
    console ``--help`` from a directory outside the repository with an
    environment that cannot see the repo. A failure here is a packaging
    gap (missing module or package data), never a cwd artifact."""
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    build = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(wheel_dir)],
        cwd=AI_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert build.returncode == 0, build.stderr or build.stdout
    wheels = list(wheel_dir.glob("mat_ai-*.whl"))
    assert len(wheels) == 1, wheels

    venv_dir = tmp_path / "venv"
    venv = subprocess.run(
        ["uv", "venv", str(venv_dir), "--python", "3.12"],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert venv.returncode == 0, venv.stderr or venv.stdout
    install = subprocess.run(
        ["uv", "pip", "install", "--python", str(venv_dir / "bin" / "python"), str(wheels[0])],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert install.returncode == 0, install.stderr or install.stdout

    installed_entry_points = subprocess.run(
        [
            str(venv_dir / "bin" / "python"),
            "-c",
            "from importlib.metadata import entry_points; "
            "print(chr(10).join(sorted(ep.name for ep in entry_points(group='console_scripts') "
            "if ep.name.startswith('mat-'))))",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert installed_entry_points.returncode == 0, installed_entry_points.stderr
    assert "mat-generate-data" not in installed_entry_points.stdout.splitlines()

    run_dir = tmp_path / "outside"
    run_dir.mkdir()
    assert run_dir != AI_ROOT and AI_ROOT not in run_dir.parents
    environment = dict(os.environ)
    for poisoned in ("PYTHONPATH", "VIRTUAL_ENV"):
        environment.pop(poisoned, None)

    for console_script in (
        "mat-validate-data",
        "mat-split-data",
        "mat-train-baseline",
        "mat-evaluate",
        "mat-demo-vietnamese",
    ):
        result = subprocess.run(
            [str(venv_dir / "bin" / console_script), "--help"],
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
            env=environment,
        )
        assert result.returncode == 0, f"{console_script}: {result.stderr}"
        assert "usage" in result.stdout.lower(), console_script
