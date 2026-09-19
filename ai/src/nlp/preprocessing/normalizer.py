from src.domain.entities import NormalizationResult as _NormalizationResult
from src.nlp.preprocessing.matcher import RegexMatcher as _RegexMatcher


class TextNormalizer:
    """Resolve matcher candidates into the legacy normalization contract."""

    def __init__(self, matcher: _RegexMatcher) -> None:
        self._matcher = matcher

    def normalize(self, text: str) -> _NormalizationResult:
        candidates: list[tuple[int, int, str, str]] = [
            (match.start, match.end, match.replacement, match.rule_id)
            for match in self._matcher.find(text)
        ]

        accepted: list[tuple[int, int, str, str]] = []
        warnings: list[str] = []
        for candidate in sorted(candidates, key=lambda value: (value[0], value[1])):
            if accepted and candidate[0] < accepted[-1][1]:
                warnings.append(f"overlap_skipped:{candidate[3]}")
                continue
            accepted.append(candidate)

        output = text
        for start, end, replacement, _ in reversed(accepted):
            output = output[:start] + replacement + output[end:]
        return _NormalizationResult(
            normalized_text=output,
            applied_rules=tuple(item[3] for item in accepted),
            warnings=tuple(warnings),
        )
