from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from src.nlp.constants import _REQUEST_ID_PATTERN, _SCORE_SUM_TOLERANCE, _SEMANTIC_FIELDS, LABELS


class DemoEvaluator:
    """Enforce the hard ``/predict`` response contract and summarize demo runs."""

    @staticmethod
    def _as_number(value: Any) -> float | None:
        """Return the value as a float, or None when it is not a JSON number."""
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    def check_response(
        self, payload: Mapping[str, Any], status_code: int | None = None
    ) -> list[str]:
        """Hard contract checks for one ``/predict`` response.

        Returns a list of human-readable violations; an empty list means the
        response passed every hard check.
        """
        violations: list[str] = []
        if status_code is not None and status_code != 200:
            violations.append(f"http status {status_code} != 200")
        label = payload.get("label")
        if label not in LABELS:
            violations.append(f"label {label!r} is not one of {LABELS}")
        confidence = self._as_number(payload.get("confidence"))
        if confidence is None or not 0.0 < confidence <= 1.0:
            violations.append(f"confidence {payload.get('confidence')!r} is not a number in (0, 1]")
        scores = payload.get("scores")
        if not isinstance(scores, Mapping):
            violations.append(f"scores {scores!r} is not an object")
        elif set(scores) != set(LABELS):
            violations.append(f"scores keys {sorted(scores)} != {sorted(LABELS)}")
        else:
            numeric_values: list[float] = []
            all_numeric = True
            for name in LABELS:
                value = self._as_number(scores[name])
                if value is None or not 0.0 <= value <= 1.0:
                    all_numeric = False
                    violations.append(f"scores[{name}] {scores[name]!r} is not a number in [0, 1]")
                else:
                    numeric_values.append(value)
            if all_numeric:
                total = sum(numeric_values)
                if abs(total - 1.0) > _SCORE_SUM_TOLERANCE:
                    violations.append(
                        f"scores sum to {total:.6f} instead of 1 "
                        f"(tolerance {_SCORE_SUM_TOLERANCE:g})"
                    )
        uncertain = payload.get("uncertain")
        if not isinstance(uncertain, bool):
            violations.append(f"uncertain {uncertain!r} is not a bool")
        model = payload.get("model")
        if not isinstance(model, Mapping):
            violations.append(f"model {model!r} is not an object")
        else:
            backend = model.get("backend")
            if backend != "linear":
                violations.append(f"model.backend {backend!r} != 'linear'")
            degraded = model.get("degraded")
            if degraded is not False:
                violations.append(f"model.degraded {degraded!r} is not False")
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not _REQUEST_ID_PATTERN.fullmatch(request_id):
            violations.append(
                f"request_id {request_id!r} does not match {_REQUEST_ID_PATTERN.pattern}"
            )
        return violations

    def responses_equal(self, first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
        """True when two responses are byte-equal on the semantic fields."""
        left = json.dumps({field: first.get(field) for field in _SEMANTIC_FIELDS}, sort_keys=True)
        right = json.dumps({field: second.get(field) for field in _SEMANTIC_FIELDS}, sort_keys=True)
        return left == right

    def summarize(self, results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Observational tallies: agreement per category plus hard-check counts."""
        categories: dict[str, dict[str, int]] = {}
        agreed = mismatched = hard_failures = repeat_mismatches = 0
        for result in results:
            category = str(result.get("category", "unknown"))
            bucket = categories.setdefault(
                category, {"total": 0, "agreed": 0, "mismatched": 0, "predicted_uncertain": 0}
            )
            bucket["total"] += 1
            if result.get("matches_expected"):
                agreed += 1
                bucket["agreed"] += 1
            else:
                mismatched += 1
                bucket["mismatched"] += 1
            if result.get("uncertain") is True:
                bucket["predicted_uncertain"] += 1
            if not result.get("hard_check_passed", False):
                hard_failures += 1
            if not result.get("repeat_equal", False):
                repeat_mismatches += 1
        return {
            "total_cases": len(results),
            "agreement": {"agreed": agreed, "mismatched": mismatched},
            "hard_check_failures": hard_failures,
            "repeat_mismatches": repeat_mismatches,
            "categories": dict(sorted(categories.items())),
        }
