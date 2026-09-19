"""Composition tests: routers wired by the factory, never loading at import time."""

import ast
from pathlib import Path

from fastapi.testclient import TestClient

import src.api.server as server
from src.api.dependencies import RuntimeState
from src.api.server import create_app
from src.core.settings import Settings
from tests.integration.api.fakes import successful_loader

ENDPOINTS_DIR = Path(__file__).parents[3] / "src" / "api" / "v1" / "endpoints"
CONCRETE_ML_ROOTS = {"sklearn", "joblib", "torch", "transformers"}


def test_endpoints_never_import_concrete_ml() -> None:
    assert ENDPOINTS_DIR.is_dir(), "v1 endpoints package is required"
    for path in ENDPOINTS_DIR.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                root = name.split(".")[0]
                assert root not in CONCRETE_ML_ROOTS, f"{path.name} imports {name}"


def test_create_app_does_not_load_model_until_lifespan() -> None:
    calls: list[str] = []

    def recording_loader(backend: str) -> object:
        calls.append(backend)
        return successful_loader(backend)

    app = create_app(Settings(), recording_loader)
    assert calls == [], "create_app must not load a model; loading happens in lifespan"
    with TestClient(app) as client:
        assert calls, "lifespan startup must load the model exactly once"
        response = client.get("/health-check")
        assert response.status_code == 200


def test_service_providers_resolve_the_current_app_state_runtime() -> None:
    """Request-facing services are constructed against the live runtime state.

    The provider chain is ``get_runtime_state -> service(state)`` per
    request: whatever ``app.state.runtime`` holds *now* is what the service
    sees. A provider that cached a service bound to an earlier runtime
    would keep serving stale readiness after the state object is replaced,
    so this test swaps the runtime under a live app and requires the
    health route to follow it.
    """
    app = create_app(Settings(), successful_loader)
    with TestClient(app) as client:
        assert client.get("/health-check").status_code == 200

        client.app.state.runtime = RuntimeState()  # fresh, unready runtime
        response = client.get("/health-check")
        assert response.status_code == 503
        assert response.json()["code"] == "MODEL_NOT_READY"


def test_server_stays_a_pure_factory() -> None:
    assert not hasattr(server, "app"), "server.py must not build a global app"
    assert not hasattr(server, "run"), "uvicorn.run belongs to the root entry point"
    assert not hasattr(server, "settings"), "server.py must not build global Settings"
