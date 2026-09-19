"""Unit tests for the Result-based health-check service."""

from src.api.dependencies import RuntimeState
from src.api.services.health_check_service import HealthCheckService
from src.domain.entities import (
    ArtifactManifest,
    CalibrationManifest,
    Prediction,
    PreprocessorManifest,
)
from src.libs.result import Ok


def _manifest() -> ArtifactManifest:
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
        max_input_tokens=None,
    )


class _FakeModel:
    """SentimentModel double whose metadata can be made to fail."""

    def __init__(self, manifest: ArtifactManifest, metadata_error: Exception | None = None) -> None:
        self._manifest = manifest
        self._metadata_error = metadata_error
        self.metadata_calls = 0
        self.predict_calls = 0

    def load(self) -> None:  # pragma: no cover - the service never loads
        raise AssertionError("health check must not load anything")

    def predict(self, text: str) -> Prediction:  # pragma: no cover - never called
        self.predict_calls += 1
        raise AssertionError("health check must not predict")

    def metadata(self) -> ArtifactManifest:
        self.metadata_calls += 1
        if self._metadata_error is not None:
            raise self._metadata_error
        return self._manifest

    def health(self) -> str:
        return "ready"


def _service(
    model: _FakeModel, *, ready: bool = True, degraded: bool = False
) -> HealthCheckService:
    state = RuntimeState(model=model, ready=ready, degraded=degraded, backend="linear")
    return HealthCheckService(state)


def test_ready_health_returns_ok_outcome_from_active_metadata() -> None:
    model = _FakeModel(_manifest())

    result = _service(model).execute()

    assert isinstance(result, Ok)
    outcome = result.ok_value
    assert outcome.backend == "linear"
    assert outcome.version == "baseline"
    assert outcome.degraded is False
    assert model.metadata_calls == 1
    assert model.predict_calls == 0


def test_degraded_state_flows_into_the_health_outcome() -> None:
    result = _service(_FakeModel(_manifest()), degraded=True).execute()

    assert isinstance(result, Ok)
    assert result.ok_value.degraded is True


def test_not_ready_health_is_retryable_model_not_ready_without_reading_metadata() -> None:
    model = _FakeModel(_manifest())

    result = _service(model, ready=False).execute()

    assert result.is_err()
    error = result.err_value
    assert error.code == "MODEL_NOT_READY"
    assert error.retryable is True
    assert "internal marker" not in error.public().values()
    assert model.metadata_calls == 0


def test_health_metadata_failure_fails_closed_without_public_detail() -> None:
    operational = RuntimeError("manifest decoder exploded — internal marker")
    model = _FakeModel(_manifest(), metadata_error=operational)

    result = _service(model).execute()

    assert result.is_err()
    error = result.err_value
    assert error.code == "HEALTH_CHECK_FAILED"
    assert error.reason is operational
    assert "internal marker" not in error.public().values()
    assert model.metadata_calls == 1
