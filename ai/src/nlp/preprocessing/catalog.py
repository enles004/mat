import re as _re
from typing import Literal as _Literal

import pydantic as _pydantic

from src.domain.normalization import NormalizationRuleSpec as _NormalizationRuleSpec


class NormalizationCatalog(_pydantic.BaseModel):
    """Frozen top-level normalization catalog document."""

    model_config = _pydantic.ConfigDict(frozen=True, extra="forbid")

    schema_version: _Literal["1"]
    dictionary_version: str
    rules: tuple[_NormalizationRuleSpec, ...] = _pydantic.Field(min_length=1)

    @staticmethod
    def _reject_blank(value: str, label: str) -> str:
        if not value.strip():
            raise ValueError(f"{label} must not be empty")
        return value

    @_pydantic.field_validator("dictionary_version")
    @classmethod
    def _dictionary_version_not_blank(cls, value: str) -> str:
        return cls._reject_blank(value, "dictionary_version")

    @_pydantic.field_validator("rules")
    @classmethod
    def _patterns_compile(
        cls, value: tuple[_NormalizationRuleSpec, ...]
    ) -> tuple[_NormalizationRuleSpec, ...]:
        """Compile every rule pattern; a broken regex fails validation loudly."""
        for rule in value:
            flags = _re.IGNORECASE if rule.flags == "IGNORECASE" else 0
            try:
                _re.compile(rule.pattern, flags)
            except _re.error as exc:
                raise ValueError(f"rule '{rule.id}': uncompilable pattern: {exc}") from exc
        return value

    @_pydantic.model_validator(mode="after")
    def _unique_rule_ids(self) -> "NormalizationCatalog":
        seen: set[str] = set()
        for rule in self.rules:
            if rule.id in seen:
                raise ValueError(f"duplicate rule id: {rule.id}")
            seen.add(rule.id)
        return self
