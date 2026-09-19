from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Frozen sentiment label order shared across all components.
LABELS = ["negative", "neutral", "positive"]

type ChallengeKind = Literal["MFT", "INV", "DIR"]

# Model scores are probabilities: NaN and the infinities are validation
# failures, never values that pass through silently.
FiniteScore = Annotated[float, Field(allow_inf_nan=False)]


class SentimentLabel(StrEnum):
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class DatasetRow(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    id: str = Field(min_length=1)
    raw_text: str = Field(min_length=1, frozen=True)
    normalized_text: str = Field(min_length=1)
    label: SentimentLabel
    aspect: str = Field(min_length=1)
    style: str = Field(min_length=1)
    noise_types: list[str]
    difficulty: Difficulty
    canonical_group_id: str = Field(min_length=1)
    generation_batch: str = Field(min_length=1)
    review_status: ReviewStatus

    @field_validator("raw_text", "normalized_text")
    @classmethod
    def reject_blank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class Prediction(BaseModel):
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    scores: dict[SentimentLabel, FiniteScore]
    uncertain: bool

    @field_validator("scores")
    @classmethod
    def require_canonical_score_keys(
        cls, value: dict[SentimentLabel, float]
    ) -> dict[SentimentLabel, float]:
        """Every score vector is the complete canonical label vector, exactly."""
        if set(value) != set(SentimentLabel):
            raise ValueError("scores must cover exactly the canonical labels")
        return value


class PreprocessorManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    version: str
    config_checksum: str


class CalibrationManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    method: str
    fitted_on: str


class ArtifactManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["1"]
    model_name: str
    model_version: str
    backend: Literal["linear", "transformer"]
    labels: list[SentimentLabel]
    preprocessor: PreprocessorManifest
    calibration: CalibrationManifest
    data_checksum: str
    payload_checksum: str
    git_revision: str
    license: str
    evaluation_report: str
    max_input_tokens: int | None = Field(default=None, ge=1)


@dataclass(frozen=True)
class ChallengeCase:
    """One minimum-functionality, invariance, or directional challenge."""

    id: str
    kind: ChallengeKind
    text: str
    paired_text: str
    expected_label: str
    paired_expected_label: str
    minimum_delta: float
    phenomenon: str = "unspecified"


@dataclass(frozen=True)
class NormalizationResult:
    """Value result of normalizing one raw text (no rule knowledge here)."""

    normalized_text: str
    applied_rules: tuple[str, ...]
    warnings: tuple[str, ...]
