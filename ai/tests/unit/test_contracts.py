"""Characterization tests for domain entities, contracts, and API wire schemas."""

from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from src.api.v1.schemas import PredictRequest, ProblemDetail
from src.domain.contracts import ModelLoader, SentimentModel
from src.domain.entities import (
    LABELS,
    ArtifactManifest,
    CalibrationManifest,
    DatasetRow,
    Prediction,
    PreprocessorManifest,
    SentimentLabel,
)
from src.domain.exceptions import ArtifactUnavailable


def test_nlp_value_objects_are_importable_from_cohesive_domain_modules() -> None:
    from src.domain.artifacts import CatalogActivation, ResolvedModelSource
    from src.domain.evaluation import (
        CalibrationDecision,
        CandidateEvidence,
        CandidateGates,
        ChallengeResult,
    )
    from src.domain.outcomes import HealthCheckOutcome, PredictionOutcome
    from src.domain.training import BaselineRun, FoldManifest, SplitManifest, ValidationReport

    assert all(
        (
            FoldManifest,
            SplitManifest,
            BaselineRun,
            ValidationReport,
            CalibrationDecision,
            ChallengeResult,
            CandidateEvidence,
            CandidateGates,
            ResolvedModelSource,
            CatalogActivation,
            PredictionOutcome,
            HealthCheckOutcome,
        )
    )


def test_saved_artifact_lives_in_the_modeling_layer() -> None:
    """The exported-artifact value is filesystem-bound, so it is a modeling value."""
    from dataclasses import FrozenInstanceError
    from pathlib import Path

    from src.nlp.modeling.saved_artifact import SavedArtifact

    assert SavedArtifact.__dataclass_params__.frozen
    artifact = SavedArtifact.__new__(SavedArtifact)
    object.__setattr__(artifact, "directory", Path("artifacts/baseline"))
    with pytest.raises(FrozenInstanceError):
        artifact.directory = Path("artifacts/other")  # type: ignore[misc]


def test_request_outcomes_are_immutable_domain_values() -> None:
    from src.domain.outcomes import HealthCheckOutcome

    outcome = HealthCheckOutcome(backend="linear", degraded=False, version="1.0.0")
    with pytest.raises(FrozenInstanceError):
        outcome.backend = "transformer"  # type: ignore[misc]


def _dataset_row(**overrides: object) -> DatasetRow:
    fields: dict[str, object] = {
        "id": "row-1",
        "raw_text": "Xe chạy ổn",
        "normalized_text": "Xe chạy ổn",
        "label": "neutral",
        "aspect": "engine",
        "style": "comment",
        "noise_types": [],
        "difficulty": "easy",
        "canonical_group_id": "group-1",
        "generation_batch": "batch-1",
        "review_status": "approved",
    }
    fields.update(overrides)
    return DatasetRow(**fields)  # type: ignore[arg-type]


def test_dataset_row_rejects_unknown_label() -> None:
    with pytest.raises(ValidationError):
        _dataset_row(label="mixed")


def test_dataset_row_rejects_raw_text_reassignment() -> None:
    row = _dataset_row()
    with pytest.raises(ValidationError):
        row.raw_text = "Xe chạy tệ"


def test_predict_request_rejects_whitespace() -> None:
    with pytest.raises(ValidationError):
        PredictRequest(text="   \n")


def test_sentiment_label_order_is_stable() -> None:
    assert [label.value for label in SentimentLabel] == ["negative", "neutral", "positive"]


def test_prediction_dumps_canonical_shape() -> None:
    prediction = Prediction(
        label="positive",
        confidence=0.9,
        scores={"negative": 0.05, "neutral": 0.05, "positive": 0.9},
        uncertain=False,
    )
    assert prediction.model_dump() == {
        "label": "positive",
        "confidence": 0.9,
        "scores": {"negative": 0.05, "neutral": 0.05, "positive": 0.9},
        "uncertain": False,
    }


def test_artifact_manifest_parses_and_forbids_extras() -> None:
    manifest = ArtifactManifest(
        schema_version="1",
        model_name="linear-svc",
        model_version="baseline",
        backend="linear",
        labels=["negative", "neutral", "positive"],
        preprocessor=PreprocessorManifest(name="normalizer", version="1", config_checksum="abc"),
        calibration=CalibrationManifest(method="sigmoid", fitted_on="out-of-fold"),
        data_checksum="data",
        payload_checksum="payload",
        git_revision="5c7055c",
        license="MIT",
        evaluation_report="reports/baseline-evaluation.md",
    )
    assert manifest.max_input_tokens is None
    assert manifest.preprocessor.name == "normalizer"
    with pytest.raises(ValidationError):
        ArtifactManifest(**{**manifest.model_dump(), "unexpected": "field"})


def test_problem_detail_dumps_wire_shape() -> None:
    problem = ProblemDetail(
        type="about:blank",
        title="Service Unavailable",
        status=503,
        detail="model not ready",
        instance="/api/v1/predict",
        code="MODEL_NOT_READY",
        request_id="req-1",
    )
    assert problem.model_dump() == {
        "type": "about:blank",
        "title": "Service Unavailable",
        "status": 503,
        "detail": "model not ready",
        "instance": "/api/v1/predict",
        "code": "MODEL_NOT_READY",
        "request_id": "req-1",
        "errors": [],
    }


def test_sentiment_model_protocol_surface_is_stable() -> None:
    assert issubclass(SentimentModel, __import__("typing").Protocol)
    for method in ("load", "predict", "metadata", "health"):
        assert hasattr(SentimentModel, method), method


def test_model_loader_alias_matches_backend_literal() -> None:
    from collections.abc import Callable
    from typing import Literal

    assert ModelLoader == Callable[[Literal["baseline", "transformer"]], SentimentModel]


def test_artifact_unavailable_is_a_typed_domain_failure() -> None:
    error = ArtifactUnavailable("baseline artifact missing")
    assert isinstance(error, Exception)
    assert "baseline artifact missing" in str(error)


def test_request_text_length_boundaries_are_explicit_and_untruncated() -> None:
    """Exact minimum text is accepted; anything shorter fails precisely.

    The wire schema carries no maximum of its own: the byte-budget
    middleware owns the upper bound, so the schema itself must not
    truncate, trim, or reject budget-sized text (never silently truncate
    input). ``DatasetRow`` text fields hold the same minimum contract.
    """
    # Exactly the minimum: one non-blank character.
    assert PredictRequest(text="x").text == "x"
    assert _dataset_row(raw_text="x", normalized_text="x").raw_text == "x"

    # Below the minimum: empty and whitespace-only are rejected on both models.
    for below_minimum in ("", " ", "\t\n "):
        with pytest.raises(ValidationError):
            PredictRequest(text=below_minimum)
    with pytest.raises(ValidationError):
        _dataset_row(raw_text="", normalized_text="x")
    with pytest.raises(ValidationError):
        _dataset_row(raw_text="x", normalized_text="   ")

    # No schema-level maximum: full byte-budget text stays valid, untruncated.
    budget_text = "a" * 65_536
    assert PredictRequest(text=budget_text).text == budget_text


def test_prediction_rejects_non_finite_confidence_and_scores() -> None:
    """NaN/Infinity must fail loudly, never pass through as a score."""
    valid_scores = {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
    for confidence in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            Prediction(
                label="positive",
                confidence=confidence,
                scores=valid_scores,
                uncertain=False,
            )
    for key in ("negative", "neutral", "positive"):
        for value in (float("nan"), float("inf"), float("-inf")):
            with pytest.raises(ValidationError):
                Prediction(
                    label="positive",
                    confidence=0.7,
                    scores={**valid_scores, key: value},
                    uncertain=False,
                )


def test_prediction_scores_cover_exactly_the_canonical_labels() -> None:
    """Every score vector is the complete canonical label vector, exactly."""
    prediction = Prediction(
        label="positive",
        confidence=0.7,
        scores={"negative": 0.1, "neutral": 0.2, "positive": 0.7},
        uncertain=False,
    )
    assert set(prediction.scores) == set(LABELS)

    incomplete_vectors = (
        {},
        {"negative": 0.5, "positive": 0.5},  # a canonical key is missing
        {"negative": 0.1, "neutral": 0.2, "positive": 0.7, "mixed": 0.0},  # unknown key
    )
    for scores in incomplete_vectors:
        with pytest.raises(ValidationError):
            Prediction(label="positive", confidence=0.7, scores=scores, uncertain=False)


def _artifact_manifest() -> ArtifactManifest:
    return ArtifactManifest(
        schema_version="1",
        model_name="linear-svc",
        model_version="baseline",
        backend="linear",
        labels=["negative", "neutral", "positive"],
        preprocessor=PreprocessorManifest(name="normalizer", version="1", config_checksum="abc"),
        calibration=CalibrationManifest(method="sigmoid", fitted_on="out-of-fold"),
        data_checksum="data",
        payload_checksum="payload",
        git_revision="5c7055c",
        license="MIT",
        evaluation_report="reports/baseline-evaluation.md",
    )


def test_manifest_and_nested_metadata_models_are_frozen() -> None:
    """Manifest metadata is immutable after construction, nesting included.

    A verified artifact's manifest is provenance evidence; mutation after
    verification would silently rewrite what was verified. The frozen rule
    matches the normalization catalog models.
    """
    manifest = _artifact_manifest()
    with pytest.raises(ValidationError):
        manifest.model_name = "renamed-model"
    with pytest.raises(ValidationError):
        manifest.payload_checksum = "sha256:rewritten"
    with pytest.raises(ValidationError):
        manifest.preprocessor.name = "renamed-normalizer"
    with pytest.raises(ValidationError):
        manifest.calibration.method = "isotonic"
