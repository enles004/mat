from collections import Counter
from pathlib import Path

from src.nlp.evaluation.behavioral import BehavioralEvaluator, ChallengeCase
from src.nlp.training.dataset import DatasetStore

AI_ROOT = Path(__file__).parents[2]

_EVALUATOR = BehavioralEvaluator()


def test_directional_case_requires_positive_probability_to_drop() -> None:
    case = ChallengeCase(
        id="dir-1",
        kind="DIR",
        text="Xe chạy tốt",
        paired_text="Xe chạy không tốt",
        expected_label="positive",
        paired_expected_label="negative",
        minimum_delta=0.20,
    )
    predictions = {
        "Xe chạy tốt": {"negative": 0.1, "neutral": 0.1, "positive": 0.8},
        "Xe chạy không tốt": {"negative": 0.7, "neutral": 0.2, "positive": 0.1},
    }

    assert _EVALUATOR.evaluate([case], predictions)[0].passed


def test_directional_case_requires_paired_expected_label_to_win() -> None:
    case = ChallengeCase(
        id="dir-2",
        kind="DIR",
        text="Xe chạy tốt",
        paired_text="Xe chạy không tốt",
        expected_label="positive",
        paired_expected_label="negative",
        minimum_delta=0.20,
    )
    predictions = {
        case.text: {"negative": 0.1, "neutral": 0.1, "positive": 0.8},
        case.paired_text: {"negative": 0.3, "neutral": 0.5, "positive": 0.2},
    }

    assert not _EVALUATOR.evaluate([case], predictions)[0].passed


def test_minimum_functionality_requires_expected_label() -> None:
    case = ChallengeCase(
        id="mft-1",
        kind="MFT",
        text="Phanh phản hồi chậm",
        paired_text="",
        expected_label="negative",
        paired_expected_label="",
        minimum_delta=0.0,
    )
    predictions = {
        case.text: {"negative": 0.7, "neutral": 0.2, "positive": 0.1},
    }

    result = _EVALUATOR.evaluate([case], predictions)[0]

    assert result.passed
    assert result.detail == "predicted=negative; expected=negative"


def test_invariance_requires_both_labels_to_match_expectation() -> None:
    case = ChallengeCase(
        id="inv-1",
        kind="INV",
        text="Hyundai chạy êm",
        paired_text="H. chạy êm",
        expected_label="positive",
        paired_expected_label="positive",
        minimum_delta=0.0,
    )
    predictions = {
        case.text: {"negative": 0.7, "neutral": 0.2, "positive": 0.1},
        case.paired_text: {"negative": 0.6, "neutral": 0.3, "positive": 0.1},
    }

    result = _EVALUATOR.evaluate([case], predictions)[0]

    assert not result.passed
    assert result.detail == "negative->negative"


def test_reviewed_challenge_set_has_exact_counts_and_required_coverage() -> None:
    cases = _EVALUATOR.load_cases(AI_ROOT / "data" / "challenge_set.csv")
    required_phenomena = {
        "negation",
        "mixed_sentiment",
        "aspect_conflict",
        "comparison",
        "restrained_sarcasm",
        "brand_abbreviation",
        "slang_negation",
        "casing",
        "spacing",
        "punctuation",
        "emoji",
        "missing_diacritics",
        "unicode_normalization",
        "unseen_model_names",
    }

    assert Counter(case.kind for case in cases) == {"MFT": 29, "INV": 23, "DIR": 22}
    assert len({case.id for case in cases}) == 74
    assert required_phenomena <= {case.phenomenon for case in cases}

    invariance_pairs = {(case.text, case.paired_text) for case in cases if case.kind == "INV"}
    assert any("Hyundai" in left and "H." in right for left, right in invariance_pairs)
    assert any("không" in left and "ko" in right for left, right in invariance_pairs)
    assert any("không" in left and "khum" in right for left, right in invariance_pairs)


def test_challenge_text_is_outside_the_frozen_dataset() -> None:
    cases = _EVALUATOR.load_cases(AI_ROOT / "data" / "challenge_set.csv")
    rows = DatasetStore().read(AI_ROOT / "data" / "dataset.csv")
    dataset_texts = {text for row in rows for text in (row.raw_text, row.normalized_text)}
    challenge_texts = {text for case in cases for text in (case.text, case.paired_text) if text}

    assert len(challenge_texts) == 119
    assert challenge_texts.isdisjoint(dataset_texts)
