from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.domain.entities import SentimentLabel


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def reject_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty or whitespace-only")
        return value


class ModelResponse(BaseModel):
    backend: str
    version: str
    degraded: bool


class PredictResponse(BaseModel):
    label: SentimentLabel
    confidence: float
    scores: dict[SentimentLabel, float]
    uncertain: bool
    model: ModelResponse
    request_id: str


class ProblemDetail(BaseModel):
    type: str
    title: str
    status: int
    detail: str
    instance: str
    code: str
    request_id: str
    errors: list[dict[str, Any]] = Field(default_factory=list)
