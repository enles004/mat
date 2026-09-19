# ai/tests/integration/packaging/test_docker_contract.py
"""Container contract: the baseline-only serving image runs the clean tree.

Locks the Dockerfile to the relocated application (root ``main.py`` plus the
installed ``src``/``scripts`` packages, CMD on ``main:app``) and Compose to the
baseline backend, artifact path, and ``/health-check`` healthcheck.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[4]


def _dockerfile() -> str:
    return (REPO_ROOT / "ai" / "Dockerfile").read_text(encoding="utf-8")


def test_dockerfile_copies_relocated_application_tree() -> None:
    dockerfile = _dockerfile()
    for required in (
        "COPY ai/main.py ai/config.py ./",
        "COPY ai/scripts ./scripts",
        "COPY ai/src ./src",
        "COPY ai/configs ./configs",
        "COPY ai/artifacts ./artifacts",
    ):
        assert required in dockerfile, f"Dockerfile must contain: {required}"
    assert "ai/tests" not in dockerfile, "tests must not ship in the image"


def test_docker_cmd_runs_root_main_app() -> None:
    command_lines = [line for line in _dockerfile().splitlines() if line.startswith("CMD ")]
    assert len(command_lines) == 1, "Dockerfile must declare exactly one CMD"
    assert "main:app" in command_lines[0], command_lines[0]
    assert "mat_ai" not in command_lines[0], command_lines[0]


def test_compose_ai_healthcheck_targets_health_check() -> None:
    compose = yaml.safe_load((REPO_ROOT / "infra" / "docker-compose.yml").read_text())
    service = compose["services"]["ai"]
    command = " ".join(service["healthcheck"]["test"])
    assert "/health-check" in command
    assert "/readyz" not in command
    assert service["environment"]["MAT_MODEL_BACKEND"] == "baseline"
    assert service["environment"]["MAT_BASELINE_ARTIFACT_DIR"] == "/app/artifacts/baseline"
