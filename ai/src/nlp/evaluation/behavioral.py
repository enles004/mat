import csv
from pathlib import Path
from typing import cast

from src.domain.entities import LABELS, ChallengeCase, ChallengeKind
from src.domain.evaluation import ChallengeResult as _ChallengeResult
from src.nlp.constants import _CHALLENGE_COLUMNS

__all__ = [
    "BehavioralEvaluator",
    "ChallengeCase",
    "ChallengeKind",
    "PredictionMap",
]


type PredictionMap = dict[str, dict[str, float]]


class BehavioralEvaluator:
    """Load reviewed challenge cases and evaluate model predictions against them."""

    @staticmethod
    def _validate_case(case: ChallengeCase) -> None:
        if not case.id or not case.text or not case.phenomenon:
            raise ValueError("Challenge id, text, and phenomenon must be non-empty")
        if case.expected_label not in LABELS:
            raise ValueError(f"Unsupported expected label for {case.id}: {case.expected_label}")
        if case.kind == "MFT":
            if case.paired_text or case.paired_expected_label or case.minimum_delta != 0.0:
                raise ValueError(f"MFT case must not define a pair or delta: {case.id}")
            return
        if not case.paired_text:
            raise ValueError(f"Paired challenge text must be non-empty: {case.id}")
        if case.paired_expected_label not in LABELS:
            raise ValueError(
                f"Unsupported paired expected label for {case.id}: {case.paired_expected_label}"
            )
        if case.kind == "INV":
            if case.paired_expected_label != case.expected_label or case.minimum_delta != 0.0:
                raise ValueError(f"INV case must preserve its label and use zero delta: {case.id}")
        elif case.minimum_delta <= 0.0:
            raise ValueError(f"DIR case must define a positive minimum delta: {case.id}")

    @staticmethod
    def _predicted_label(scores: dict[str, float]) -> str:
        return max(LABELS, key=scores.__getitem__)

    def load_cases(self, path: Path) -> list[ChallengeCase]:
        """Load and validate reviewed challenge cases from CSV."""
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != _CHALLENGE_COLUMNS:
                raise ValueError("Challenge CSV columns do not match the required schema")
            records = list(reader)

        cases: list[ChallengeCase] = []
        seen_ids: set[str] = set()
        for record in records:
            kind = record["kind"]
            if kind not in {"MFT", "INV", "DIR"}:
                raise ValueError(f"Unsupported challenge kind: {kind}")
            case = ChallengeCase(
                id=record["id"],
                kind=cast(ChallengeKind, kind),
                text=record["text"],
                paired_text=record["paired_text"],
                expected_label=record["expected_label"],
                paired_expected_label=record["paired_expected_label"],
                minimum_delta=float(record["minimum_delta"]),
                phenomenon=record["phenomenon"],
            )
            if case.id in seen_ids:
                raise ValueError(f"Duplicate challenge ID: {case.id}")
            self._validate_case(case)
            seen_ids.add(case.id)
            cases.append(case)
        return cases

    def _evaluate_mft(self, case: ChallengeCase, predictions: PredictionMap) -> _ChallengeResult:
        """Require a minimum-functionality example to receive its expected label."""
        predicted = self._predicted_label(predictions[case.text])
        return _ChallengeResult(
            case.id,
            case.kind,
            predicted == case.expected_label,
            f"predicted={predicted}; expected={case.expected_label}",
        )

    def _evaluate_invariance(
        self, case: ChallengeCase, predictions: PredictionMap
    ) -> _ChallengeResult:
        """Require both meaning-preserving variants to receive the expected label."""
        left = self._predicted_label(predictions[case.text])
        right = self._predicted_label(predictions[case.paired_text])
        return _ChallengeResult(
            case.id,
            case.kind,
            left == right == case.expected_label,
            f"{left}->{right}",
        )

    def _evaluate_directional(
        self, case: ChallengeCase, predictions: PredictionMap
    ) -> _ChallengeResult:
        """Require the original label probability to fall and the paired label to win."""
        original = predictions[case.text][case.expected_label]
        paired = predictions[case.paired_text][case.expected_label]
        paired_label = self._predicted_label(predictions[case.paired_text])
        passed = (
            original - paired >= case.minimum_delta and paired_label == case.paired_expected_label
        )
        return _ChallengeResult(case.id, case.kind, passed, f"delta={original - paired:.4f}")

    def evaluate(
        self, cases: list[ChallengeCase], predictions: PredictionMap
    ) -> list[_ChallengeResult]:
        """Evaluate challenge cases in their reviewed file order."""
        evaluators = {
            "MFT": self._evaluate_mft,
            "INV": self._evaluate_invariance,
            "DIR": self._evaluate_directional,
        }
        return [evaluators[case.kind](case, predictions) for case in cases]
