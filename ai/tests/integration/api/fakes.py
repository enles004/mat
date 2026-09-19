from dataclasses import dataclass

from src.domain.entities import (
    ArtifactManifest,
    CalibrationManifest,
    Prediction,
    PreprocessorManifest,
)
from src.domain.exceptions import ArtifactUnavailable


@dataclass
class FakeModel:
    backend: str = "linear"
    version: str = "1.0.0"
    fail_predict: bool = False

    def load(self) -> None:
        return None

    def predict(self, text: str) -> Prediction:
        if self.fail_predict:
            raise RuntimeError("synthetic inference failure")
        return Prediction(
            label="positive",
            confidence=0.70,
            scores={"negative": 0.10, "neutral": 0.20, "positive": 0.70},
            uncertain=False,
        )

    def metadata(self) -> ArtifactManifest:
        return ArtifactManifest(
            schema_version="1",
            model_name="fake",
            model_version=self.version,
            backend=self.backend,
            labels=["negative", "neutral", "positive"],
            preprocessor=PreprocessorManifest(
                name="fake", version="1.0.0", config_checksum="sha256:" + "a" * 64
            ),
            calibration=CalibrationManifest(method="none", fitted_on="not-applicable"),
            data_checksum="sha256:" + "b" * 64,
            payload_checksum="sha256:" + "c" * 64,
            git_revision="d" * 40,
            license="test-only",
            evaluation_report="reports/test.md",
        )

    def health(self) -> str:
        return "ready"


def successful_loader(backend: str) -> FakeModel:
    return FakeModel(backend="linear" if backend == "baseline" else "transformer")


def transformer_fails_baseline_succeeds_loader(backend: str) -> FakeModel:
    if backend == "transformer":
        raise ArtifactUnavailable("transformer artifact unavailable")
    return FakeModel(backend="linear")


def runtime_failing_transformer_loader(backend: str) -> FakeModel:
    return FakeModel(backend="transformer", fail_predict=True)


def unexpected_failure_transformer_loader(backend: str) -> FakeModel:
    """An unexpected wiring bug in the transformer path — not a missing artifact."""
    if backend == "transformer":
        raise RuntimeError("unexpected wiring bug")
    return FakeModel(backend="linear")
