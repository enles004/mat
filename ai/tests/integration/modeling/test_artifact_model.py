"""Characterization tests for the relocated artifact model, limits, and loaders."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier  # type: ignore[import-untyped]

from src.domain.entities import (
    ArtifactManifest,
    CalibrationManifest,
    Prediction,
    PreprocessorManifest,
)
from src.domain.exceptions import TextExceedsModelLimit
from src.nlp.evaluation.behavioral import (
    BehavioralEvaluator,
    ChallengeCase,
    ChallengeKind,
)
from src.nlp.modeling.artifact_model import LinearArtifactModel
from src.nlp.modeling.limits import ModelInputLimiter
from src.nlp.modeling.registry import ArtifactRegistry


class _FakePayload:
    """Mimics a calibrated pipeline: ``predict_proba`` + ``classes_`` in raw order."""

    classes_ = np.array(["positive", "negative", "neutral"])

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        assert len(texts) == 1
        return np.array([[0.7, 0.2, 0.1]])


def _manifest(**overrides: object) -> ArtifactManifest:
    fields: dict[str, object] = {
        "schema_version": "1",
        "model_name": "linear-svc",
        "model_version": "test",
        "backend": "linear",
        "labels": ["negative", "neutral", "positive"],
        "preprocessor": PreprocessorManifest(name="normalizer", version="1", config_checksum="abc"),
        "calibration": CalibrationManifest(method="sigmoid", fitted_on="out-of-fold"),
        "data_checksum": "data",
        "payload_checksum": "sha256:stub",
        "git_revision": "5c7055c",
        "license": "MIT",
        "evaluation_report": "reports/baseline-evaluation.md",
    }
    fields.update(overrides)
    return ArtifactManifest(**fields)  # type: ignore[arg-type]


def _model(threshold: float = 0.6, **overrides: object) -> LinearArtifactModel:
    return LinearArtifactModel(_FakePayload(), _manifest(**overrides), threshold)


def test_predict_reindexes_scores_to_frozen_label_order() -> None:
    prediction = _model().predict("xe điện vinfast rất tốt")
    assert isinstance(prediction, Prediction)
    assert prediction.label == "positive"
    assert prediction.confidence == pytest.approx(0.7)
    assert list(prediction.scores) == ["negative", "neutral", "positive"]
    assert prediction.scores == {
        "negative": pytest.approx(0.2),
        "neutral": pytest.approx(0.1),
        "positive": pytest.approx(0.7),
    }
    assert prediction.uncertain is False


def test_predict_marks_top_score_below_threshold_uncertain() -> None:
    assert _model(threshold=0.75).predict("anything").uncertain is True


def test_model_surface_and_manifest_metadata() -> None:
    manifest = _manifest()
    model = LinearArtifactModel(_FakePayload(), manifest, 0.6)
    for method in ("load", "predict", "metadata", "health"):
        assert hasattr(model, method), method
    assert model.load() is None
    assert model.metadata() is manifest
    assert model.health() == "ready"


def test_token_count_uses_whitespace_splitting() -> None:
    limiter = ModelInputLimiter()
    limiter.enforce(_model(max_input_tokens=3), "xe  máy\ndien")
    limiter.enforce(_model(max_input_tokens=1), "")
    with pytest.raises(TextExceedsModelLimit):
        limiter.enforce(_model(max_input_tokens=2), "xe  máy\ndien")


def test_enforce_token_limit_rejects_over_limit_with_typed_error() -> None:
    model = _model(max_input_tokens=2)
    with pytest.raises(TextExceedsModelLimit) as excinfo:
        ModelInputLimiter().enforce(model, "một hai ba")
    assert excinfo.value.limit == 2


def test_enforce_token_limit_allows_within_and_unlimited() -> None:
    ModelInputLimiter().enforce(_model(max_input_tokens=2), "một hai")
    ModelInputLimiter().enforce(_model(), "một hai ba bốn năm")


def test_verify_artifact_rejects_tampered_payload(tmp_path: Path) -> None:
    directory = tmp_path / "artifact"
    registry = ArtifactRegistry()
    registry.save(
        DummyClassifier(strategy="prior"),
        directory,
        lambda checksum: _manifest(payload_checksum=checksum),
    )
    payload = directory / "model.joblib"
    payload.write_bytes(payload.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        registry.verify(directory)


def test_verify_artifact_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        ArtifactRegistry().verify(tmp_path / "does-not-exist")


def test_challenge_reexports_are_domain_types() -> None:
    from src.domain.entities import ChallengeCase as DomainCase
    from src.domain.entities import ChallengeKind as DomainKind

    assert ChallengeCase is DomainCase
    assert ChallengeKind is DomainKind
    assert callable(BehavioralEvaluator().load_cases)
