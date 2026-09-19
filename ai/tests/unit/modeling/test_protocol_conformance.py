"""Runtime protocol conformance of the real artifact adapter.

The serving boundary is typed by the ``SentimentModel`` protocol; the
deployed adapter must satisfy it as a real instance check, not just by
convention. Signatures are compared method-by-method and the adapter is
exercised over a minimal ``predict_proba`` payload whose ``classes_`` are
deliberately in non-canonical order.
"""

import inspect

import pytest

from src.domain.contracts import SentimentModel
from src.domain.entities import (
    ArtifactManifest,
    CalibrationManifest,
    Prediction,
    PreprocessorManifest,
    SentimentLabel,
)
from src.nlp.modeling.artifact_model import LinearArtifactModel

PROTOCOL_METHODS = ("load", "predict", "metadata", "health")


class _FixedProbabilities:
    """Minimal calibrated-pipeline stand-in with ``classes_`` out of order."""

    def __init__(self, classes: list[str], probabilities: list[float]) -> None:
        self._classes = classes
        self._probabilities = probabilities

    @property
    def classes_(self) -> list[str]:
        return self._classes

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
        assert len(texts) == 1
        return [self._probabilities]


def _manifest() -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1",
        model_name="linear-svc",
        model_version="conformance-test",
        backend="linear",
        labels=["negative", "neutral", "positive"],
        preprocessor=PreprocessorManifest(
            name="normalizer (raw view)", version="1", config_checksum="abc"
        ),
        calibration=CalibrationManifest(method="sigmoid", fitted_on="out-of-fold"),
        data_checksum="data",
        payload_checksum="payload",
        git_revision="5c7055c",
        license="MIT",
        evaluation_report="reports/baseline-evaluation.md",
    )


def test_linear_artifact_model_conforms_to_sentiment_model_protocol() -> None:
    """The deployed adapter is a runtime ``SentimentModel`` and behaves as one."""
    payload = _FixedProbabilities(
        classes=["positive", "negative", "neutral"],  # non-canonical on purpose
        probabilities=[0.7, 0.2, 0.1],
    )
    manifest = _manifest()
    model = LinearArtifactModel(payload, manifest, uncertain_threshold=0.60)

    # Runtime structural conformance (requires the protocol to be runtime_checkable).
    assert isinstance(model, SentimentModel)

    # The method surface matches the protocol exactly, method by method.
    for name in PROTOCOL_METHODS:
        protocol_signature = inspect.signature(getattr(SentimentModel, name))
        adapter_signature = inspect.signature(getattr(LinearArtifactModel, name))
        protocol_params = list(protocol_signature.parameters)[1:]  # drop ``self``
        adapter_params = list(adapter_signature.parameters)[1:]
        assert adapter_params == protocol_params, name
        assert adapter_signature.return_annotation == (
            protocol_signature.return_annotation
        ), name

    # Behavioral conformance: canonical score keys and order even though the
    # payload exposes ``classes_`` in a different order, plus the metadata and
    # health surfaces the endpoints rely on.
    prediction = model.predict("xe chạy tốt")
    assert isinstance(prediction, Prediction)
    assert [label.value for label in prediction.scores] == [
        label.value for label in SentimentLabel
    ]
    assert prediction.scores[SentimentLabel.POSITIVE] == pytest.approx(0.7)
    assert prediction.label == SentimentLabel.POSITIVE
    assert prediction.uncertain is False
    assert model.metadata() is manifest
    assert model.health() == "ready"
    assert model.load() is None
