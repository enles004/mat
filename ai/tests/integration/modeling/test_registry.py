from pathlib import Path

import pytest
from sklearn.dummy import DummyClassifier  # type: ignore[import-untyped]

from src.domain.entities import (
    ArtifactManifest,
    CalibrationManifest,
    PreprocessorManifest,
    SentimentLabel,
)
from src.nlp.modeling.registry import ArtifactRegistry


def test_registry_rejects_modified_payload(tmp_path: Path) -> None:
    payload = tmp_path / "model.joblib"
    payload.write_bytes(b"original")
    registry = ArtifactRegistry()
    expected = registry.verify_payload_checksum([payload])
    payload.write_bytes(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        registry.verify_payload_checksum([payload], expected=expected)


def _manifest_for(payload_checksum: str) -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1",
        model_name="tfidf-word-char-logreg",
        model_version="1.0.0",
        backend="linear",
        labels=[SentimentLabel(label) for label in ("negative", "neutral", "positive")],
        preprocessor=PreprocessorManifest(
            name="test-preprocessor",
            version="1.0.0",
            config_checksum="sha256:" + "a" * 64,
        ),
        calibration=CalibrationManifest(method="none", fitted_on="test"),
        data_checksum="sha256:" + "b" * 64,
        payload_checksum=payload_checksum,
        git_revision="c" * 40,
        license="internal-use-only",
        evaluation_report="reports/model-selection.md",
    )


def _fitted_model() -> DummyClassifier:
    model = DummyClassifier(strategy="prior")
    model.fit(
        [f"text {index}" for index in range(6)],
        ["negative"] * 2 + ["neutral"] * 2 + ["positive"] * 2,
    )
    return model


def test_artifact_manifest_round_trips_through_the_registry(tmp_path: Path) -> None:
    directory = tmp_path / "baseline"
    registry = ArtifactRegistry()
    saved = registry.save(_fitted_model(), directory, _manifest_for)

    verified = registry.verify(directory)
    assert verified == saved.manifest

    model, manifest = registry.load(directory)
    assert manifest.payload_checksum == saved.manifest.payload_checksum
    probabilities = model.predict_proba(["unseen text"])
    assert probabilities.shape == (1, 3)
    assert probabilities.sum() == pytest.approx(1.0)


def test_registry_rejects_a_single_flipped_payload_byte(tmp_path: Path) -> None:
    directory = tmp_path / "baseline"
    registry = ArtifactRegistry()
    registry.save(_fitted_model(), directory, _manifest_for)
    payload = directory / "model.joblib"
    flipped = bytearray(payload.read_bytes())
    flipped[-1] ^= 0xFF
    payload.write_bytes(bytes(flipped))

    with pytest.raises(ValueError, match="checksum"):
        registry.verify(directory)
    with pytest.raises(ValueError, match="checksum"):
        registry.load(directory)


def test_registry_rejects_a_tampered_manifest_checksum(tmp_path: Path) -> None:
    directory = tmp_path / "baseline"
    registry = ArtifactRegistry()
    saved = registry.save(_fitted_model(), directory, _manifest_for)
    manifest_path = directory / "manifest.json"
    forged = "sha256:" + "0" * 64
    assert forged != saved.manifest.payload_checksum
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            saved.manifest.payload_checksum, forged
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checksum"):
        registry.verify(directory)
