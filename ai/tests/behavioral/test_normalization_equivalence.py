"""Rule characterization and old/new equivalence harness (Plan 1 Task 3).

Dictionary 2.0.0 expanded the catalog from 4 to 13 rules (brand aliases,
elongation flattening, code-switching); the matcher semantics below are
unchanged, so every corpus text must still match both implementations.

The reference implementation below is the verbatim pre-catalog algorithm
(per-rule scan over independently compiled patterns, whole-text casefolded
context check, (start, end) sort, first-overlap skip warnings, reverse
replacement). Every generated and real corpus text must produce an equal
``NormalizationResult`` — text, applied rule ids, and warnings — from both
implementations.
"""

import csv
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

from src.nlp.preprocessing.catalog import NormalizationCatalog
from src.nlp.preprocessing.catalog_loader import CatalogLoader
from src.nlp.preprocessing.matcher import RegexMatcher
from src.nlp.preprocessing.normalizer import TextNormalizer

DATA = Path("data")


@dataclass(frozen=True)
class ReferenceRule:
    rule_id: str
    pattern: re.Pattern[str]
    replacement: str
    context_any: tuple[str, ...]


def reference_normalize(
    text: str,
    rules: tuple[ReferenceRule, ...],
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    folded = text.casefold()
    candidates: list[tuple[int, int, str, str]] = []
    warnings: list[str] = []
    for rule in rules:
        if rule.context_any and not any(value in folded for value in rule.context_any):
            continue
        for match in rule.pattern.finditer(text):
            candidates.append((match.start(), match.end(), rule.replacement, rule.rule_id))
    accepted: list[tuple[int, int, str, str]] = []
    for candidate in sorted(candidates, key=lambda value: (value[0], value[1])):
        if accepted and candidate[0] < accepted[-1][1]:
            warnings.append(f"overlap_skipped:{candidate[3]}")
            continue
        accepted.append(candidate)
    output = text
    for start, end, replacement, _ in reversed(accepted):
        output = output[:start] + replacement + output[end:]
    return output, tuple(item[3] for item in accepted), tuple(warnings)


def reference_rules(catalog: NormalizationCatalog) -> tuple[ReferenceRule, ...]:
    rules: list[ReferenceRule] = []
    for spec in catalog.rules:
        flags = re.IGNORECASE if spec.flags == "IGNORECASE" else 0
        rules.append(
            ReferenceRule(
                rule_id=spec.id,
                pattern=re.compile(spec.pattern, flags),
                replacement=spec.replacement,
                context_any=tuple(value.casefold() for value in spec.context_any),
            )
        )
    return tuple(rules)


def production_catalog() -> NormalizationCatalog:
    return CatalogLoader().load(Path("configs/normalization.yaml")).catalog


def real_texts() -> list[str]:
    texts: list[str] = []
    showcase = json.loads((DATA / "vietnamese_demo_cases.json").read_text(encoding="utf-8"))
    texts.extend(case["text"] for case in showcase["cases"])
    with (DATA / "challenge_set.csv").open(newline="", encoding="utf-8") as handle:
        texts.extend(row["text"] for row in csv.DictReader(handle))
    with (DATA / "dataset.csv").open(newline="", encoding="utf-8") as handle:
        texts.extend(row["raw_text"] for row in csv.DictReader(handle))
    return texts


def generated_texts(count: int = 10_000) -> list[str]:
    rng = random.Random(20260918)
    tokens = [
        "xe",
        "nhà",
        "hãng",
        "H.",
        "h.",
        "ko",
        "KO",
        "Ko",
        "khum",
        "k",
        "k.",
        "cx",
        "CX",
        "dc",
        "đc",
        "ổn",
        "!",
        "?",
        ".",
        ",",
        ":))",
        "ô tô",
        "oto",
        "car",
        "",
    ]
    joins = [" ", ""]
    texts: list[str] = []
    for _ in range(count):
        length = rng.randint(0, 10)
        parts = [rng.choice(tokens) for _ in range(length)]
        texts.append(rng.choice(joins).join(parts))
    return texts


def new_result(
    catalog: NormalizationCatalog,
    text: str,
) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    normalizer = TextNormalizer(RegexMatcher.from_catalog(catalog))
    result = normalizer.normalize(text)
    return result.normalized_text, result.applied_rules, result.warnings


def assert_equivalent(catalog: NormalizationCatalog, texts: list[str]) -> None:
    rules = reference_rules(catalog)
    normalizer = TextNormalizer(RegexMatcher.from_catalog(catalog))
    for text in texts:
        expected = reference_normalize(text, rules)
        result = normalizer.normalize(text)
        actual = (result.normalized_text, result.applied_rules, result.warnings)
        assert actual == expected, f"divergence on {text!r}: reference {expected}, new {actual}"


def test_generated_corpus_is_equivalent() -> None:
    assert_equivalent(production_catalog(), generated_texts())


def test_real_corpus_is_equivalent() -> None:
    texts = real_texts()
    assert len(texts) == 1_028
    assert_equivalent(production_catalog(), texts)


def test_legacy_rule_characterization_exact_outputs() -> None:
    catalog = production_catalog()

    result = new_result(catalog, "Xe nhà H. chạy cx ổn :))")
    assert result == ("Xe nhà Hyundai chạy cũng ổn :))", ("hyundai_h_dot", "cx_to_cung"), ())

    kept = new_result(catalog, "Anh H. đang họp")
    assert kept == ("Anh H. đang họp", (), ())

    negation = new_result(catalog, "xe KO ổn 😡")
    assert negation == ("xe không ổn 😡", ("ko_to_khong",), ())

    mixed = new_result(catalog, "xe H. cx ko dc")
    assert mixed == (
        "xe Hyundai cũng không được",
        ("hyundai_h_dot", "cx_to_cung", "ko_to_khong", "dc_to_duoc"),
        (),
    )

    no_match = new_result(catalog, "ổn!")
    assert no_match == ("ổn!", (), ())


def test_dictionary_2_rule_characterization_exact_outputs() -> None:
    """Dictionary 2.0.0 additions: brand aliases, elongation, code-switching."""
    catalog = production_catalog()

    toy = new_result(catalog, "Toy chạy cx ổn")
    assert toy == ("Toyota chạy cũng ổn", ("toy_to_toyota", "cx_to_cung"), ())

    vf = new_result(catalog, "VF 8 thì ko tệ")
    assert vf == ("Vinfast 8 thì không tệ", ("vf_to_vinfast", "ko_to_khong"), ())

    mitsu = new_result(catalog, "Mitsu chạy dc")
    assert mitsu == ("Mitsubishi chạy được", ("mitsu_to_mitsubishi", "dc_to_duoc"), ())

    mec = new_result(catalog, "con Mẹc này đẹp")
    assert mec == ("con Mercedes này đẹp", ("mec_to_mercedes",), ())

    elongated = new_result(catalog, "xe quáaa đẹp")
    assert elongated == ("xe quá đẹp", ("elong_qua_to_qua",), ())

    loud = new_result(catalog, "rấttt ổn")
    assert loud == ("rất ổn", ("elong_rat_to_rat",), ())

    good = new_result(catalog, "xe này good")
    assert good == ("xe này tốt", ("overall_good_to_tot",), ())

    average = new_result(catalog, "chất lượng average")
    assert average == ("chất lượng tạm", ("overall_average_to_tam",), ())

    bad = new_result(catalog, "nội thất bad lắm")
    assert bad == ("nội thất tệ lắm", ("overall_bad_to_te",), ())

    full_brand_untouched = new_result(catalog, "Toyota cũ mà chậc, vinfast thì ko")
    assert full_brand_untouched == (
        "Toyota cũ mà chậc, vinfast thì không",
        ("ko_to_khong",),
        (),
    )


def test_punctuation_adjacent_and_repeated_matches() -> None:
    catalog = production_catalog()
    assert new_result(catalog, "ko,cx:dc") == (
        "không,cũng:được",
        ("ko_to_khong", "cx_to_cung", "dc_to_duoc"),
        (),
    )
    assert new_result(catalog, "ko ko ko") == (
        "không không không",
        ("ko_to_khong", "ko_to_khong", "ko_to_khong"),
        (),
    )


def test_normalizer_is_idempotent_through_new_path() -> None:
    normalizer = TextNormalizer(RegexMatcher.from_catalog(production_catalog()))
    once = normalizer.normalize("Xe nhà H. chạy cx ổn :))")
    twice = normalizer.normalize(once.normalized_text)
    assert twice.normalized_text == once.normalized_text


def test_overlapping_synthetic_rules_keep_legacy_resolution() -> None:
    """Pins the pre-catalog overlap/warning semantics the matcher must keep.

    A one-pass alternation would resolve "aaa" to "XY" with no warning
    (leftmost-longest wins); the legacy algorithm accepts the shorter
    earlier-ending candidate and warns on the overlapped longer one. Exact
    behavior outranks the optimization claim, so the legacy resolution is
    the contract.
    """
    catalog = NormalizationCatalog.model_validate(
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
    assert new_result(catalog, "aaa") == (
        "YYY",
        ("r_short", "r_short", "r_short"),
        ("overlap_skipped:r_long",),
    )
    assert_equivalent(catalog, ["aaa", "aa", "aaaa", "aXa", "aa aa aa"])
