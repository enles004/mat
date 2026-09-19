"""Contract tests for the authored Vietnamese demo case file.

These cases exist only for the behavioral showcase: they are exploratory
examples, never a benchmark, and must never feed training, calibration,
selection, or any frozen artifact.
"""

import json
import re
import tomllib
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

AI_ROOT = Path(__file__).parents[2]
CASES_PATH = AI_ROOT / "data" / "vietnamese_demo_cases.json"
PYPROJECT_PATH = AI_ROOT / "pyproject.toml"

EXPECTED_CATEGORIES: dict[str, str] = {
    "automotive-specific": "AS",
    "code-switch-emoji": "CS",
    "mixed-aspects": "MX",
    "missing-diacritics": "MD",
    "negation": "NG",
    "regional": "RG",
    "sarcasm-idiom": "SC",
    "slang": "SL",
    "teencode": "TC",
}
VALID_LABELS = {"negative", "neutral", "positive"}
STRING_FIELDS = ("id", "category", "text", "human_interpretation", "expected_label", "notes")
# Sanity lexicon: ASCII-only spellings of common Vietnamese motoring words.
ASCII_ONLY_VIETNAMESE_WORDS = {
    "banh",
    "chay",
    "cua",
    "di",
    "em",
    "ko",
    "khong",
    "lanh",
    "may",
    "mat",
    "on",
    "roi",
    "tot",
    "xang",
    "xe",
}


def _load_document() -> dict[str, Any]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def _load_cases() -> list[dict[str, Any]]:
    return list(_load_document()["cases"])


def _has_diacritic(text: str) -> bool:
    for character in text:
        if character in "đĐ":
            return True
        if len(unicodedata.normalize("NFD", character)) > 1:
            return True
    return False


def test_cases_file_exists_and_parses_as_object() -> None:
    assert CASES_PATH.is_file()
    document = _load_document()
    assert isinstance(document, dict)


def test_version_is_one() -> None:
    assert _load_document()["version"] == 1


def test_exactly_54_cases_across_nine_categories_of_six() -> None:
    cases = _load_cases()
    assert len(cases) == 54
    counts = Counter(str(case["category"]) for case in cases)
    assert dict(counts) == {category: 6 for category in EXPECTED_CATEGORIES}


def test_ids_are_unique_with_their_category_prefix() -> None:
    cases = _load_cases()
    ids = [str(case["id"]) for case in cases]
    assert len(set(ids)) == 54
    for case in cases:
        prefix = EXPECTED_CATEGORIES[str(case["category"])]
        assert re.fullmatch(rf"{prefix}-\d{{2}}", str(case["id"])), case["id"]


def test_texts_are_unique_and_non_empty() -> None:
    texts = [str(case["text"]) for case in _load_cases()]
    assert len(set(texts)) == 54
    assert all(text.strip() for text in texts)


def test_expected_labels_valid_and_every_string_field_non_empty() -> None:
    for case in _load_cases():
        for field in STRING_FIELDS:
            value = case.get(field)
            assert isinstance(value, str) and value.strip(), (case["id"], field)
        assert case["expected_label"] in VALID_LABELS, case["id"]


def test_missing_diacritics_cases_are_ascii_with_vietnamese_words() -> None:
    for case in _load_cases():
        if case["category"] != "missing-diacritics":
            continue
        text = str(case["text"])
        assert text.isascii(), case["id"]
        words = set(re.findall(r"[a-z]+", text.lower()))
        assert words & ASCII_ONLY_VIETNAMESE_WORDS, case["id"]


def test_other_categories_carry_at_least_one_diacritic() -> None:
    for case in _load_cases():
        if case["category"] == "missing-diacritics":
            continue
        assert _has_diacritic(str(case["text"])), case["id"]


def test_pyproject_declares_demo_console_script() -> None:
    data = tomllib.loads(PYPROJECT_PATH.read_text(encoding="utf-8"))
    assert data["project"]["scripts"]["mat-demo-vietnamese"] == "scripts.demo_api_vietnamese:main"
