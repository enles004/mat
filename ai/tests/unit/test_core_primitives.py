"""Characterization tests for domain primitives, values computed by hand."""

import csv

import numpy as np

from src.domain.entities import LABELS, ChallengeCase
from src.domain.scoring import multiclass_brier
from src.nlp.evaluation.behavioral import BehavioralEvaluator

CHALLENGE_COLUMNS = [
    "id",
    "kind",
    "text",
    "paired_text",
    "expected_label",
    "paired_expected_label",
    "minimum_delta",
    "phenomenon",
]


def test_labels_order_is_pinned() -> None:
    assert LABELS == ["negative", "neutral", "positive"]


def test_multiclass_brier_matches_hand_computed_value() -> None:
    """Row 1 (negative): (0.9-1)^2 + 0.05^2 + 0.05^2 = 0.015.

    Row 2 (positive): 0.1^2 + 0.2^2 + (0.7-1)^2 = 0.14.
    Mean = (0.015 + 0.14) / 2 = 0.0775; the pinned float is the exact IEEE-754
    value the frozen implementation produces for that arithmetic.
    """
    probabilities = np.array([[0.9, 0.05, 0.05], [0.1, 0.2, 0.7]])
    assert multiclass_brier(["negative", "positive"], probabilities) == 0.07750000000000001


def test_load_challenge_cases_returns_expected_frozen_rows(tmp_path) -> None:
    csv_path = tmp_path / "challenges.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(CHALLENGE_COLUMNS)
        writer.writerow(["mft-1", "MFT", "Xe này chạy ổn", "", "positive", "", "0.0", "baseline"])
        writer.writerow(
            [
                "inv-1",
                "INV",
                "Xe này chạy ổn",
                "xe này chạy ổn",
                "positive",
                "positive",
                "0.0",
                "casing",
            ]
        )
        writer.writerow(
            [
                "dir-1",
                "DIR",
                "Xe này tốt",
                "Xe này rất tốt",
                "positive",
                "positive",
                "0.05",
                "intensity",
            ]
        )

    cases = BehavioralEvaluator().load_cases(csv_path)

    assert cases == [
        ChallengeCase(
            id="mft-1",
            kind="MFT",
            text="Xe này chạy ổn",
            paired_text="",
            expected_label="positive",
            paired_expected_label="",
            minimum_delta=0.0,
            phenomenon="baseline",
        ),
        ChallengeCase(
            id="inv-1",
            kind="INV",
            text="Xe này chạy ổn",
            paired_text="xe này chạy ổn",
            expected_label="positive",
            paired_expected_label="positive",
            minimum_delta=0.0,
            phenomenon="casing",
        ),
        ChallengeCase(
            id="dir-1",
            kind="DIR",
            text="Xe này tốt",
            paired_text="Xe này rất tốt",
            expected_label="positive",
            paired_expected_label="positive",
            minimum_delta=0.05,
            phenomenon="intensity",
        ),
    ]
    assert [case.kind for case in cases] == ["MFT", "INV", "DIR"]
