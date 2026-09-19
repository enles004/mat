from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator

from src.domain.entities import NormalizationResult

_RULE_CATEGORIES = Literal["negation", "core", "teencode", "regional", "automotive", "brand"]


def _reject_blank(value: str, label: str) -> str:
    if not value.strip():
        raise ValueError(f"{label} must not be empty")
    return value


class CatalogSource(BaseModel):
    """Provenance of a catalog entry; only reviewed internal sources are valid."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["reviewed_internal"]
    reference: str

    @field_validator("reference")
    @classmethod
    def _reference_not_blank(cls, value: str) -> str:
        return _reject_blank(value, "reference")


class NormalizationRuleSpec(BaseModel):
    """One reviewed normalization rule with its exact regex payload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    category: _RULE_CATEGORIES
    pattern: str
    replacement: str
    context_any: tuple[str, ...]
    flags: Literal["IGNORECASE", "NONE"]
    source: CatalogSource

    @field_validator("id")
    @classmethod
    def _id_not_blank(cls, value: str) -> str:
        return _reject_blank(value, "id")

    @field_validator("pattern")
    @classmethod
    def _pattern_not_blank(cls, value: str, info: ValidationInfo) -> str:
        """Reject blank patterns; compilation is validated by the NLP loader."""
        if not value.strip():
            raise ValueError(f"rule '{info.data.get('id', '?')}': pattern must not be empty")
        return value

    @field_validator("replacement")
    @classmethod
    def _replacement_not_blank(cls, value: str, info: ValidationInfo) -> str:
        if not value.strip():
            raise ValueError(f"rule '{info.data.get('id', '?')}': replacement must not be empty")
        return value

    @field_validator("context_any")
    @classmethod
    def _context_items_not_blank(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if not item.strip():
                raise ValueError("context_any entries must not be empty")
        return value


@dataclass(frozen=True)
class RuleMatch:
    """One candidate span emitted by the compiled matcher."""

    start: int
    end: int
    replacement: str
    rule_id: str


class Normalizer(Protocol):
    def normalize(self, text: str) -> NormalizationResult:
        raise NotImplementedError
