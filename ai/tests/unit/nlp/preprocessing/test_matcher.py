"""Compiled matcher tests (Plan 1 Task 3)."""

import builtins
import re
import time

import pytest

from src.domain.normalization import RuleMatch
from src.nlp.preprocessing.catalog import NormalizationCatalog
from src.nlp.preprocessing.matcher import RegexMatcher


def production_catalog() -> NormalizationCatalog:
    return NormalizationCatalog.model_validate(
        {
            "schema_version": "1",
            "dictionary_version": "1.0.0",
            "rules": [
                {
                    "id": "hyundai_h_dot",
                    "category": "brand",
                    "pattern": r"(?<!\w)H\.(?!\w)",
                    "replacement": "Hyundai",
                    "context_any": ["xe", "hãng", "nhà", "ô tô", "oto", "car"],
                    "flags": "IGNORECASE",
                    "source": {"kind": "reviewed_internal", "reference": "MAT-NORM-BRAND-001"},
                },
                {
                    "id": "ko_to_khong",
                    "category": "negation",
                    "pattern": r"(?<!\w)(ko|khum|k)(?!\w)",
                    "replacement": "không",
                    "context_any": [],
                    "flags": "IGNORECASE",
                    "source": {"kind": "reviewed_internal", "reference": "MAT-NORM-NEG-001"},
                },
                {
                    "id": "cx_to_cung",
                    "category": "teencode",
                    "pattern": r"(?<!\w)cx(?!\w)",
                    "replacement": "cũng",
                    "context_any": [],
                    "flags": "IGNORECASE",
                    "source": {"kind": "reviewed_internal", "reference": "MAT-NORM-TEEN-001"},
                },
                {
                    "id": "dc_to_duoc",
                    "category": "teencode",
                    "pattern": r"(?<!\w)(dc|đc)(?!\w)",
                    "replacement": "được",
                    "context_any": [],
                    "flags": "IGNORECASE",
                    "source": {"kind": "reviewed_internal", "reference": "MAT-NORM-TEEN-002"},
                },
            ],
        }
    )


def overlaps_catalog() -> NormalizationCatalog:
    return NormalizationCatalog.model_validate(
        {
            "schema_version": "1",
            "dictionary_version": "1.0.0",
            "rules": [
                {
                    "id": "r_long",
                    "category": "core",
                    "pattern": "aa",
                    "replacement": "X",
                    "context_any": [],
                    "flags": "NONE",
                    "source": {"kind": "reviewed_internal", "reference": "SYNTH-001"},
                },
                {
                    "id": "r_short",
                    "category": "core",
                    "pattern": "a",
                    "replacement": "Y",
                    "context_any": [],
                    "flags": "NONE",
                    "source": {"kind": "reviewed_internal", "reference": "SYNTH-002"},
                },
            ],
        }
    )


def spans(matcher: RegexMatcher, text: str) -> list[tuple[int, int, str, str]]:
    return [(m.start, m.end, m.replacement, m.rule_id) for m in matcher.find(text)]


def test_find_returns_position_ordered_matches() -> None:
    matcher = RegexMatcher.from_catalog(production_catalog())
    assert spans(matcher, "xe cx ko dc") == [
        (3, 5, "cũng", "cx_to_cung"),
        (6, 8, "không", "ko_to_khong"),
        (9, 11, "được", "dc_to_duoc"),
    ]


def test_find_respects_remote_context_any() -> None:
    matcher = RegexMatcher.from_catalog(production_catalog())
    assert matcher.find("Anh H. đang họp") == ()
    assert spans(matcher, "xe nhà H. chạy") == [(7, 9, "Hyundai", "hyundai_h_dot")]


def test_find_applies_ignorecase_and_boundaries() -> None:
    matcher = RegexMatcher.from_catalog(production_catalog())
    assert spans(matcher, "KO khum K") == [
        (0, 2, "không", "ko_to_khong"),
        (3, 7, "không", "ko_to_khong"),
        (8, 9, "không", "ko_to_khong"),
    ]
    assert matcher.find("khub") == ()
    assert spans(matcher, "k.") == [(0, 1, "không", "ko_to_khong")]
    assert matcher.find("kko") == ()


def test_find_repeated_and_no_match() -> None:
    matcher = RegexMatcher.from_catalog(production_catalog())
    assert spans(matcher, "ko ko") == [
        (0, 2, "không", "ko_to_khong"),
        (3, 5, "không", "ko_to_khong"),
    ]
    assert matcher.find("ổn") == ()
    assert matcher.find("") == ()


def test_find_marks_hyundai_inside_words_not_matched() -> None:
    matcher = RegexMatcher.from_catalog(production_catalog())
    assert matcher.find("xe H.Al") == ()
    assert spans(matcher, "xe (H.)") == [(4, 6, "Hyundai", "hyundai_h_dot")]


def test_overlapping_rules_expose_full_candidate_set() -> None:
    # find() returns every candidate scan hit, position-ordered; overlap
    # resolution stays in TextNormalizer (legacy semantics preserved there).
    matcher = RegexMatcher.from_catalog(overlaps_catalog())
    assert spans(matcher, "aaa") == [
        (0, 1, "Y", "r_short"),
        (0, 2, "X", "r_long"),
        (1, 2, "Y", "r_short"),
        (2, 3, "Y", "r_short"),
    ]


def test_matcher_is_immutable() -> None:
    matcher = RegexMatcher.from_catalog(production_catalog())
    with pytest.raises(Exception, match=""):
        matcher.find = lambda text: ()  # type: ignore[method-assign]


def test_compilation_happens_only_at_construction(monkeypatch: pytest.MonkeyPatch) -> None:
    catalog = production_catalog()
    counter = {"compile": 0}
    real_compile = re.compile

    def counting_compile(*args: object, **kwargs: object) -> re.Pattern[str]:
        counter["compile"] += 1
        return real_compile(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(re, "compile", counting_compile)
    matcher = RegexMatcher.from_catalog(catalog)
    at_construction = counter["compile"]
    assert at_construction > 0

    def no_io(*args: object, **kwargs: object) -> object:
        raise AssertionError("normalize must not touch the filesystem")

    monkeypatch.setattr(builtins, "open", no_io)
    for index in range(100):
        matcher.find(f"xe ko {index} cx H. dc")
    assert counter["compile"] == at_construction


def test_rule_match_is_frozen() -> None:
    match = RuleMatch(0, 1, "x", "r")
    with pytest.raises(Exception, match=""):
        match.start = 5  # type: ignore[misc]


def test_pathological_inputs_stay_within_budget() -> None:
    """Pathological shapes stay linear: one very long no-match string and one
    match-dense string both complete inside a generous wall-clock bound, with
    position-ordered, complete matches — compiled patterns never backtrack
    catastrophically and never rescan after construction."""
    matcher = RegexMatcher.from_catalog(production_catalog())

    no_match = "z" * 200_000
    started = time.perf_counter_ns()
    assert matcher.find(no_match) == ()
    no_match_elapsed_ns = time.perf_counter_ns() - started

    dense = " ".join(["ko"] * 25_000)
    started = time.perf_counter_ns()
    matches = matcher.find(dense)
    dense_elapsed_ns = time.perf_counter_ns() - started

    assert len(matches) == 25_000, "every token must match exactly once"
    assert [match.rule_id for match in matches] == ["ko_to_khong"] * 25_000
    assert [(matches[0].start, matches[0].end)] == [(0, 2)]
    assert (matches[-1].start, matches[-1].end) == (len(dense) - 2, len(dense))
    starts = [match.start for match in matches]
    assert starts == sorted(starts)
    assert len(set(starts)) == 25_000, "one match per token, no duplicates"

    wall_clock_budget_ns = 1_000_000_000  # 1.0 s: a ceiling on algorithmic blowups
    assert no_match_elapsed_ns < wall_clock_budget_ns
    assert dense_elapsed_ns < wall_clock_budget_ns
