from dataclasses import dataclass

from src.domain.entities import ArtifactManifest, Prediction


@dataclass(frozen=True)
class PredictionOutcome:
    prediction: Prediction
    manifest: ArtifactManifest
    degraded: bool


@dataclass(frozen=True)
class HealthCheckOutcome:
    backend: str
    degraded: bool
    version: str
