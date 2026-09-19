"""Unit tests for the Result-based prediction service."""

from src.api.dependencies import RuntimeState
from src.api.services.prediction_service import PredictionService
from src.domain.entities import (
    ArtifactManifest,
    CalibrationManifest,
    Prediction,
    PreprocessorManifest,
)
from src.domain.exceptions import TextExceedsModelLimit
from src.libs.result import Ok


def _manifest(max_input_tokens: int | None = None) -> ArtifactManifest:
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
        git_revision="a" * 40,
        license="MIT",
        evaluation_report="reports/baseline-evaluation.md",
        max_input_tokens=max_input_tokens,
    )


class _FakeModel:
    """SentimentModel double that counts calls and can be made to fail."""

    def __init__(self, manifest: ArtifactManifest, predict_error: Exception | None = None) -> None:
        self._manifest = manifest
        self._predict_error = predict_error
        self.predict_calls = 0
        self.load_calls = 0
        self.metadata_calls = 0

    def load(self) -> None:
        self.load_calls += 1

    def predict(self, text: str) -> Prediction:
        self.predict_calls += 1
        if self._predict_error is not None:
            raise self._predict_error
        return Prediction(
            label="positive",
            confidence=0.9,
            scores={"negative": 0.05, "neutral": 0.05, "positive": 0.9},
            uncertain=False,
        )

    def metadata(self) -> ArtifactManifest:
        self.metadata_calls += 1
        return self._manifest

    def health(self) -> str:
        return "ready"


def _service(model: _FakeModel, *, ready: bool = True, degraded: bool = False) -> PredictionService:
    state = RuntimeState(model=model, ready=ready, degraded=degraded, backend="linear")
    return PredictionService(state)


def test_success_returns_ok_prediction_outcome_with_single_predict_call() -> None:
    model = _FakeModel(_manifest())
    service = _service(model)

    result = service.execute("xe chạy tốt")

    assert isinstance(result, Ok)
    outcome = result.ok_value
    assert outcome.prediction.label == "positive"
    assert outcome.prediction.confidence == 0.9
    assert outcome.degraded is False
    assert outcome.manifest.backend == "linear"
    assert model.predict_calls == 1
    assert model.load_calls == 0
    # One metadata read for the token-limit check, one for the outcome manifest.
    assert model.metadata_calls == 2


def test_degraded_state_flows_into_the_outcome() -> None:
    model = _FakeModel(_manifest())

    result = _service(model, degraded=True).execute("xe chạy tốt")

    assert isinstance(result, Ok)
    assert result.ok_value.degraded is True


def test_not_ready_returns_retryable_model_not_ready_without_touching_the_model() -> None:
    model = _FakeModel(_manifest())

    result = _service(model, ready=False).execute("xe chạy tốt")

    assert result.is_err()
    error = result.err_value
    assert error.code == "MODEL_NOT_READY"
    assert error.retryable is True
    assert "internal marker" not in error.public().values()
    assert model.predict_calls == 0
    assert model.metadata_calls == 0


def test_token_limit_is_the_expected_client_failure_branch() -> None:
    model = _FakeModel(_manifest(max_input_tokens=3))

    result = _service(model).execute("một hai ba bốn năm sáu")

    assert result.is_err()
    error = result.err_value
    assert error.code == "TEXT_EXCEEDS_MODEL_LIMIT"
    assert isinstance(error.reason, TextExceedsModelLimit)
    assert error.retryable is False
    assert model.predict_calls == 0


def test_inference_failure_keeps_the_reason_and_never_tries_another_backend() -> None:
    operational = RuntimeError("tokeniser exploded — internal marker")
    model = _FakeModel(_manifest(), predict_error=operational)

    result = _service(model).execute("xe chạy tốt")

    assert result.is_err()
    error = result.err_value
    assert error.code == "INFERENCE_FAILED"
    assert error.reason is operational
    assert "internal marker" not in error.public().values()
    assert model.predict_calls == 1  # called exactly once; no retry, no fallback
