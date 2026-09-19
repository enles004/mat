import re
from dataclasses import dataclass as _dataclass

from src.domain.normalization import RuleMatch as _RuleMatch
from src.nlp.preprocessing.catalog import NormalizationCatalog as _NormalizationCatalog


@_dataclass(frozen=True)
class _Entry:
    rule_id: str
    pattern: re.Pattern[str]
    replacement: str
    context_any: tuple[str, ...]  # casefolded at build time


@_dataclass(frozen=True)
class RegexMatcher:
    """Immutable matcher compiling every rule pattern exactly once."""

    _entries: tuple[_Entry, ...]

    @classmethod
    def from_catalog(cls, catalog: _NormalizationCatalog) -> "RegexMatcher":
        if not catalog.rules:
            raise ValueError("catalog has no rules")
        entries = tuple(
            _Entry(
                rule_id=spec.id,
                pattern=re.compile(
                    spec.pattern,
                    re.IGNORECASE if spec.flags == "IGNORECASE" else 0,
                ),
                replacement=spec.replacement,
                context_any=tuple(value.casefold() for value in spec.context_any),
            )
            for spec in catalog.rules
        )
        return cls(_entries=entries)

    def find(self, text: str) -> tuple[_RuleMatch, ...]:
        folded = text.casefold()
        candidates: list[_RuleMatch] = []
        for entry in self._entries:
            if entry.context_any and not any(value in folded for value in entry.context_any):
                continue
            for match in entry.pattern.finditer(text):
                candidates.append(
                    _RuleMatch(match.start(), match.end(), entry.replacement, entry.rule_id)
                )
        candidates.sort(key=lambda match: (match.start, match.end))
        return tuple(candidates)
