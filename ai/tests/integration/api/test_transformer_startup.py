"""Startup semantics for ``MAT_MODEL_BACKEND=transformer`` against the real loader.

The transformer branch must serve the artifact directory directly — there is
no ``model.joblib`` payload — while keeping the same verify-then-load
contract as the baseline branch: a missing directory and a manifest whose
backend disagrees are both ``ArtifactUnavailable``, and a fixed
``transformer`` selection never silently substitutes the baseline.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import load_model_from_artifact
from src.api.server import create_app
from src.core.settings import Settings
from src.domain.entities import (
    ArtifactManifest,
    CalibrationManifest,
    PreprocessorManifest,
)
from src.domain.exceptions import ArtifactUnavailable
from src.nlp.modeling.registry import ArtifactRegistry
from src.nlp.modeling.transformer_model import TransformerArtifactModel
from tests.integration.api.fakes import successful_loader


def _write_transformer_artifact(tmp_path: Path, backend: str) -> Path:
    """A minimal on-disk artifact: one payload file plus a consistent manifest."""
    directory = tmp_path / "champion"
    directory.mkdir()
    (directory / "config.json").write_text('{"model_type": "fake"}', encoding="utf-8")
    payload_files = ArtifactRegistry.payload_files(directory)
    checksum = ArtifactRegistry.payload_checksum(payload_files, base=directory)
    manifest = ArtifactManifest(
        schema_version="1",
        model_name="bamibert-finetuned",
        model_version="test",
        backend=backend,
        labels=["negative", "neutral", "positive"],
        preprocessor=PreprocessorManifest(
            name="bamibert-autotokenizer (raw view, no normalization)",
            version="test",
            config_checksum="sha256:" + "a" * 64,
        ),
        calibration=CalibrationManifest(method="none (raw softmax)", fitted_on="not calibrated"),
        data_checksum="sha256:" + "b" * 64,
        payload_checksum=checksum,
        git_revision="c" * 40,
        license="test-only",
        evaluation_report="reports/transformer-champion.md",
    )
    (directory / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    return directory


def test_transformer_loader_serves_adapter_without_joblib_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _write_transformer_artifact(tmp_path, backend="transformer")
    # The real HF load is the adapter's own unit-test concern; here we pin
    # that the loader wires the adapter to the *verified* manifest and never
    # reaches for a joblib payload.
    monkeypatch.setattr(TransformerArtifactModel, "load", lambda self: None)
    settings = Settings(transformer_artifact_dir=directory)

    model = load_model_from_artifact("transformer", settings)

    assert isinstance(model, TransformerArtifactModel)
    assert model.metadata().backend == "transformer"
    assert model.metadata().model_name == "bamibert-finetuned"
    assert model.health() == "ready"


def test_transformer_missing_artifact_is_artifact_unavailable(tmp_path: Path) -> None:
    settings = Settings(transformer_artifact_dir=tmp_path / "missing")
    with pytest.raises(ArtifactUnavailable, match="transformer"):
        load_model_from_artifact("transformer", settings)


def test_transformer_loader_rejects_linear_manifest(tmp_path: Path) -> None:
    directory = _write_transformer_artifact(tmp_path, backend="linear")
    settings = Settings(transformer_artifact_dir=directory)
    with pytest.raises(ArtifactUnavailable, match="declares backend"):
        load_model_from_artifact("transformer", settings)


def test_fixed_transformer_never_falls_back_to_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # startup passes only the backend to the production loader, which reads
    # its own Settings from the environment — so the missing artifact must
    # be pinned through the env override, not the app's Settings object.
    monkeypatch.setenv("MAT_TRANSFORMER_ARTIFACT_DIR", str(tmp_path / "missing"))
    settings = Settings(model_backend="transformer")
    app = create_app(settings, load_model_from_artifact)
    with TestClient(app) as client:
        health = client.get("/health-check")
        prediction = client.post("/predict", json={"text": "Xe ổn"})
    assert health.status_code == 503
    assert prediction.status_code == 503
    assert prediction.json()["code"] == "MODEL_NOT_READY"


def test_wire_reports_transformer_backend() -> None:
    app = create_app(Settings(model_backend="transformer"), successful_loader)
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Xe ổn"})
    assert response.status_code == 200
    assert response.json()["model"]["backend"] == "transformer"
