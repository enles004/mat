from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import load_model_from_artifact
from src.api.server import create_app
from src.core.settings import Settings
from src.domain.exceptions import ArtifactUnavailable
from tests.integration.api.fakes import (
    runtime_failing_transformer_loader,
    transformer_fails_baseline_succeeds_loader,
    unexpected_failure_transformer_loader,
)


def test_auto_falls_back_only_during_startup_and_marks_degraded() -> None:
    settings = Settings(model_backend="auto")
    app = create_app(settings, transformer_fails_baseline_succeeds_loader)
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Xe ổn"})
    assert response.status_code == 200
    assert response.json()["model"]["degraded"] is True


def test_runtime_inference_error_is_not_silently_retried_on_baseline() -> None:
    settings = Settings(model_backend="transformer")
    app = create_app(settings, runtime_failing_transformer_loader)
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Xe ổn"})
    assert response.status_code == 500
    assert response.json()["code"] == "INFERENCE_FAILED"


def test_auto_unexpected_loader_error_stays_unready() -> None:
    settings = Settings(model_backend="auto")
    app = create_app(settings, unexpected_failure_transformer_loader)
    with TestClient(app) as client:
        readiness = client.get("/health-check")
        prediction = client.post("/predict", json={"text": "Xe ổn"})
    assert readiness.status_code == 503
    assert prediction.status_code == 503
    assert prediction.json()["code"] == "MODEL_NOT_READY"


def test_missing_artifact_dir_is_artifact_unavailable(tmp_path: Path) -> None:
    settings = Settings(baseline_artifact_dir=tmp_path / "missing")
    with pytest.raises(ArtifactUnavailable):
        load_model_from_artifact("baseline", settings)
