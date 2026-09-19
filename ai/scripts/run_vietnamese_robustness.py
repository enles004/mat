#!/usr/bin/env python3
"""Runner and canonical comparison for the Vietnamese robustness suite.

Plan Task 7: executes the reviewed data-as-test suite
(``tests/fixtures/vietnamese_robustness_v1.json``) against a live
baseline-serving ``/predict`` endpoint, verifies every hard contract gate,
and writes the observational agreement report.

Hard gates — any violation exits 1 and no report is written:

- schema/version and the exact 15 x 6 category matrix with 2 positive /
  2 neutral / 2 negative human labels per category, unique IDs, unique
  texts, exactly one primary category per case;
- zero overlap with the frozen corpora (900 dataset + 60 challenge + 54
  demo) under the system's own normalization: no exact normalized text and
  no six-token n-gram in common;
- two real ``/predict`` requests per case with distinct, syntactically
  valid request IDs echoed by the service, payloads identical except the
  request ID;
- every response: finite three-label scores summing to one, exact canonical
  score-key order, baseline backend/version, ``degraded=false``;
- exact canonical match to the versioned baseline-response snapshot, where
  the request ID is excluded from equality only (both IDs must exist, be
  valid, be distinct, and be echoed).

Human/model semantic agreement is observational: it is reported, never
gated. Report rendering is deterministic (no clock input unless
``$MAT_REPORT_GENERATED_AT`` is set) and contains no reviewer personal
data.

Usage (release profile, from ``ai/`` with the baseline server up)::

    uv run --locked python scripts/run_vietnamese_robustness.py \\
        --cases tests/fixtures/vietnamese_robustness_v1.json \\
        --baseline tests/fixtures/vietnamese_robustness_v1_baseline.json
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from csv import DictReader
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

from src.nlp.preprocessing.catalog_loader import CatalogLoader
from src.nlp.preprocessing.matcher import RegexMatcher
from src.nlp.preprocessing.normalizer import TextNormalizer

AI_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "1"
SUITE_NAME = "vietnamese-robustness"
CASES_PER_CATEGORY = 6
LABELS_PER_CATEGORY = 2
REQUESTS_PER_CASE = 2
N_GRAM_SIZE = 6
SCORE_SUM_TOLERANCE = 1e-6
VALID_LABELS = ("negative", "neutral", "positive")
CANONICAL_SCORE_ORDER = ("negative", "neutral", "positive")
RESPONSE_KEYS = ("label", "confidence", "scores", "uncertain", "model", "request_id")
REQUIRED_CASE_FIELDS = ("id", "category", "text", "expected_label", "notes")
FULL_MATRIX_CATEGORIES = (
    "regional-north",
    "regional-central",
    "regional-south",
    "channel-automotive-forum",
    "channel-social-video-comment",
    "channel-ecommerce-review",
    "channel-sales-service-chat",
    "teencode-abbreviation",
    "missing-diacritics-typos",
    "code-switching",
    "negation-scope",
    "mixed-aspects",
    "sarcasm-idiom",
    "automotive-jargon",
    "target-ambiguity",
)
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_REPORT_PATH = Path("reports/vietnamese-robustness-v1.md")

_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,128}")
_DROP_KEY_RE = re.compile(r"request|timing|elapsed|latency|duration", re.IGNORECASE)
_WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)

Poster = Callable[[str, str], Mapping[str, Any]]


# --- shared text and normalization helpers ---------------------------------


def production_normalizer() -> TextNormalizer:
    """The locked production catalog compiled once through the shared engine."""
    loaded = CatalogLoader().load(AI_ROOT / "configs" / "normalization.yaml")
    return TextNormalizer(RegexMatcher.from_catalog(loaded.catalog))


def system_normalize(normalizer: TextNormalizer, text: str) -> str:
    """Normalize with the system's own catalog engine (the one algorithm)."""
    return normalizer.normalize(text).normalized_text


def tokenize(text: str) -> tuple[str, ...]:
    """Whitespace-free word tokens of the casefolded text (Unicode-aware)."""
    return tuple(_WORD_RE.findall(text.casefold()))


def ngrams_of(tokens: Sequence[str], size: int = N_GRAM_SIZE) -> set[tuple[str, ...]]:
    """Every consecutive ``size``-token window of the token sequence."""
    return {tuple(tokens[start : start + size]) for start in range(len(tokens) - size + 1)}


# --- schema and coverage gate -----------------------------------------------


def schema_violations(document: Mapping[str, Any], *, full_matrix: bool) -> list[str]:
    """Contract violations of the suite document itself.

    ``full_matrix`` enforces the canonical 15-category production matrix;
    the internal-consistency checks (6 cases and a 2/2/2 human label balance
    per declared category, unique IDs and texts, one primary category per
    case) always apply. Human labels live in ``expected_label`` and are kept
    strictly separate from observed model labels.
    """
    violations: list[str] = []
    if document.get("schema_version") != SCHEMA_VERSION:
        violations.append(
            f"schema_version must be {SCHEMA_VERSION!r}, got {document.get('schema_version')!r}"
        )
    if document.get("version") != 1:
        violations.append(f"version must be 1, got {document.get('version')!r}")

    categories_value = document.get("categories")
    if (
        not isinstance(categories_value, list)
        or not categories_value
        or not all(isinstance(item, str) and item.strip() for item in categories_value)
    ):
        violations.append("categories must be a non-empty list of non-empty strings")
        return violations
    categories = [str(item) for item in categories_value]
    if len(set(categories)) != len(categories):
        violations.append("categories must be unique")
    if full_matrix and tuple(categories) != FULL_MATRIX_CATEGORIES:
        violations.append(
            "full matrix must declare exactly the 15 canonical categories in order: "
            f"{list(FULL_MATRIX_CATEGORIES)}, got {categories}"
        )

    cases_value = document.get("cases")
    if not isinstance(cases_value, list) or not cases_value:
        violations.append("cases must be a non-empty list")
        return violations
    expected_total = CASES_PER_CATEGORY * len(categories)
    if len(cases_value) != expected_total:
        violations.append(
            f"expected {expected_total} cases ({len(categories)} categories x "
            f"{CASES_PER_CATEGORY}), got {len(cases_value)}"
        )

    seen_ids: list[str] = []
    seen_texts: list[str] = []
    by_category: dict[str, list[Mapping[str, Any]]] = {name: [] for name in categories}
    for position, case_value in enumerate(cases_value):
        if not isinstance(case_value, Mapping):
            violations.append(f"case #{position}: must be an object")
            continue
        case_id_value = case_value.get("id")
        label = f"case {case_id_value!r}" if isinstance(case_id_value, str) else f"case #{position}"
        for field in REQUIRED_CASE_FIELDS:
            value = case_value.get(field)
            if not isinstance(value, str) or not value.strip():
                violations.append(f"{label}: field {field!r} must be a non-empty string")
        if isinstance(case_id_value, str):
            seen_ids.append(case_id_value)
        text_value = case_value.get("text")
        if isinstance(text_value, str):
            seen_texts.append(text_value)
        category_value = case_value.get("category")
        if isinstance(category_value, str) and category_value in by_category:
            by_category[category_value].append(case_value)
        else:
            violations.append(
                f"{label}: primary category {category_value!r} is not in the declared set"
            )
        expected_label = case_value.get("expected_label")
        if expected_label not in VALID_LABELS:
            violations.append(
                f"{label}: expected_label {expected_label!r} not in {list(VALID_LABELS)}"
            )
        secondary = case_value.get("secondary_tags")
        if secondary is not None and (
            not isinstance(secondary, list) or not all(isinstance(tag, str) for tag in secondary)
        ):
            violations.append(f"{label}: secondary_tags must be a list of strings")

    if len(set(seen_ids)) != len(seen_ids):
        violations.append("case ids must be unique")
    if len(set(seen_texts)) != len(seen_texts):
        violations.append("case texts must be unique")

    for category in categories:
        group = by_category[category]
        if len(group) != CASES_PER_CATEGORY:
            violations.append(
                f"category {category!r} has {len(group)} cases, expected {CASES_PER_CATEGORY}"
            )
        label_counts = Counter(str(case.get("expected_label")) for case in group)
        for label_name in VALID_LABELS:
            if label_counts[label_name] != LABELS_PER_CATEGORY:
                violations.append(
                    f"category {category!r} has {label_counts[label_name]} {label_name} "
                    f"cases, expected {LABELS_PER_CATEGORY}"
                )
    return violations


# --- cross-corpus disjointness gate -----------------------------------------


@dataclass(frozen=True)
class CorpusIndex:
    """Frozen-corpus fingerprint: exact normalized texts plus six-token n-grams."""

    exact: frozenset[str]
    ngrams: frozenset[tuple[str, ...]]
    sizes: Mapping[str, int]


def _data_dir() -> Path:
    env = os.environ.get("MAT_AI_DATA")
    return Path(env) if env else AI_ROOT / "data"


def load_corpus_texts() -> dict[str, list[tuple[str, str]]]:
    """The byte-frozen corpora as (id, text): 54 demo + 60 challenge + 900 dataset."""
    data = _data_dir()
    showcase_document = json.loads(
        (data / "vietnamese_demo_cases.json").read_text(encoding="utf-8")
    )
    showcase = [(str(case["id"]), str(case["text"])) for case in showcase_document["cases"]]
    with (data / "challenge_set.csv").open(newline="", encoding="utf-8") as handle:
        challenge = [(str(row["id"]), str(row["text"])) for row in DictReader(handle)]
    with (data / "dataset.csv").open(newline="", encoding="utf-8") as handle:
        dataset = [(str(row["id"]), str(row["raw_text"])) for row in DictReader(handle)]
    return {"showcase": showcase, "challenge": challenge, "dataset": dataset}


def build_corpus_index(normalizer: TextNormalizer) -> CorpusIndex:
    """Normalize every frozen corpus text once and fingerprint both gates."""
    corpus = load_corpus_texts()
    exact: set[str] = set()
    grams: set[tuple[str, ...]] = set()
    sizes: dict[str, int] = {}
    for source, items in corpus.items():
        sizes[source] = len(items)
        for _, text in items:
            normalized = system_normalize(normalizer, text)
            exact.add(normalized)
            grams |= ngrams_of(tokenize(normalized))
    return CorpusIndex(exact=frozenset(exact), ngrams=frozenset(grams), sizes=sizes)


def corpus_overlap_violations(
    document: Mapping[str, Any],
    index: CorpusIndex,
    normalizer: TextNormalizer,
) -> list[str]:
    """Cases sharing an exact normalized text or any six-token n-gram."""
    violations: list[str] = []
    cases_value = document.get("cases")
    if not isinstance(cases_value, list):
        return ["cases must be a list before overlap checking"]
    for case_value in cases_value:
        if not isinstance(case_value, Mapping):
            continue
        case_id = case_value.get("id")
        text_value = case_value.get("text")
        if not isinstance(text_value, str):
            continue
        normalized = system_normalize(normalizer, text_value)
        if normalized in index.exact:
            violations.append(
                f"case {case_id!r}: normalized text exactly matches the frozen corpus"
            )
        hits = ngrams_of(tokenize(normalized)) & index.ngrams
        if hits:
            sample = " ".join(sorted(hits)[0])
            violations.append(
                f"case {case_id!r}: shares {len(hits)} six-token n-gram(s) with the "
                f"frozen corpus, e.g. {sample!r}"
            )
    return violations


# --- response contract and request-ID gates ---------------------------------


def request_ids_for(case_id: str) -> tuple[str, str]:
    """The deterministic distinct request-ID pair used for one case."""
    return (f"{case_id}-req-1", f"{case_id}-req-2")


def request_id_violations(request_id: str) -> list[str]:
    if not _REQUEST_ID_RE.fullmatch(request_id):
        return [f"invalid request id {request_id!r}: must match {_REQUEST_ID_RE.pattern!r}"]
    return []


def request_pair_violations(first: str, second: str) -> list[str]:
    """Both IDs must be syntactically valid and distinct — reuse is a failure."""
    violations = [*request_id_violations(first), *request_id_violations(second)]
    if first == second:
        violations.append(f"request ids must be distinct, got {first!r} twice")
    return violations


def canonical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """The response projection used for equality: request/timing keys dropped.

    Only request/timing-identifying keys are excluded; every model-bearing
    field (label, confidence, scores, uncertain, model metadata) must match
    exactly, including float formatting after JSON round-trip.
    """
    return {key: value for key, value in payload.items() if not _DROP_KEY_RE.search(key)}


def _finite_number(value: Any) -> TypeGuard[float]:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def response_contract_violations(
    payload: Mapping[str, Any], expected_model: Mapping[str, Any]
) -> list[str]:
    """One response against the hard API contract on the baseline path."""
    violations: list[str] = []
    if tuple(payload.keys()) != RESPONSE_KEYS:
        violations.append(
            f"response keys must be exactly {RESPONSE_KEYS} in order, got {tuple(payload.keys())}"
        )
    if payload.get("label") not in VALID_LABELS:
        violations.append(f"label {payload.get('label')!r} is not a canonical sentiment label")
    confidence = payload.get("confidence")
    if not _finite_number(confidence):
        violations.append(f"confidence must be a finite number, got {confidence!r}")
    elif not 0.0 <= confidence <= 1.0:
        violations.append(f"confidence must be within [0, 1], got {confidence!r}")
    if not isinstance(payload.get("uncertain"), bool):
        violations.append(f"uncertain must be a boolean, got {payload.get('uncertain')!r}")

    scores = payload.get("scores")
    if not isinstance(scores, Mapping):
        violations.append(f"scores must be an object, got {type(scores).__name__}")
        return violations
    keys = tuple(str(key) for key in scores)
    if set(keys) != set(CANONICAL_SCORE_ORDER):
        violations.append(
            f"scores must cover exactly {list(CANONICAL_SCORE_ORDER)}, got {sorted(keys)}"
        )
        return violations
    if keys != CANONICAL_SCORE_ORDER:
        violations.append(
            f"scores must be in canonical order {list(CANONICAL_SCORE_ORDER)}, got {keys}"
        )
    values = [scores[key] for key in CANONICAL_SCORE_ORDER]
    if not all(_finite_number(value) for value in values):
        violations.append(f"every score must be a finite number, got {values!r}")
        return violations
    total = sum(float(value) for value in values)
    if abs(total - 1.0) > SCORE_SUM_TOLERANCE:
        violations.append(f"scores must sum to 1 within {SCORE_SUM_TOLERANCE}, got {total!r}")

    model = payload.get("model")
    if not isinstance(model, Mapping):
        violations.append("model metadata block must be an object")
        return violations
    if model.get("backend") != expected_model.get("backend"):
        violations.append(
            f"model.backend must be {expected_model.get('backend')!r}, got {model.get('backend')!r}"
        )
    if model.get("version") != expected_model.get("version"):
        violations.append(
            f"model.version must be {expected_model.get('version')!r}, got {model.get('version')!r}"
        )
    if model.get("degraded") is not False:
        violations.append(
            f"model.degraded must be false on the baseline serving path, "
            f"got {model.get('degraded')!r}"
        )
    return violations


@dataclass(frozen=True)
class CaptureResult:
    """The two observed responses of one case plus any capture violations."""

    payloads: list[dict[str, Any]]
    violations: list[str]


def capture_case(post: Poster, text: str, case_id: str) -> CaptureResult:
    """Two real requests per case: identical payloads, distinct valid IDs."""
    first_id, second_id = request_ids_for(case_id)
    violations = request_pair_violations(first_id, second_id)
    payloads: list[dict[str, Any]] = []
    for request_id in (first_id, second_id):
        payload = dict(post(text, request_id))
        payloads.append(payload)
        echoed = payload.get("request_id")
        if echoed != request_id:
            violations.append(
                f"case {case_id!r}: response must echo request id {request_id!r}, got {echoed!r}"
            )
    return CaptureResult(payloads=payloads, violations=violations)


def http_poster(base_url: str) -> Poster:
    """POST one ``/predict`` request over real HTTP (release-profile path)."""

    def post(text: str, request_id: str) -> Mapping[str, Any]:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/predict",
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-request-id": request_id},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            payload: dict[str, Any] = json.load(response)
            return payload

    return post


# --- baseline snapshot gate --------------------------------------------------


def live_baseline_document(
    document: Mapping[str, Any], observed: Mapping[str, Sequence[Mapping[str, Any]]]
) -> dict[str, Any]:
    """Assemble a baseline snapshot document from one verified capture."""
    responses = {case_id: [dict(payload) for payload in pair] for case_id, pair in observed.items()}
    model_block: dict[str, Any] = {}
    for pair in responses.values():
        candidate = pair[0].get("model") if pair else None
        if isinstance(candidate, Mapping):
            model_block = {
                "backend": candidate.get("backend"),
                "version": candidate.get("version"),
            }
        break
    return {
        "schema_version": SCHEMA_VERSION,
        "suite": SUITE_NAME,
        "suite_version": document.get("version"),
        "model": model_block,
        "responses": responses,
    }


def baseline_document_violations(baseline: Mapping[str, Any]) -> list[str]:
    """Integrity of the stored snapshot itself, before any drift comparison."""
    violations: list[str] = []
    if baseline.get("schema_version") != SCHEMA_VERSION:
        violations.append(f"baseline schema_version must be {SCHEMA_VERSION!r}")
    if baseline.get("suite") != SUITE_NAME:
        violations.append(f"baseline suite must be {SUITE_NAME!r}")
    model_value = baseline.get("model")
    expected_model: Mapping[str, Any] = {}
    if (
        not isinstance(model_value, Mapping)
        or not model_value.get("backend")
        or not model_value.get("version")
    ):
        violations.append("baseline model block must record backend and version")
    else:
        expected_model = model_value

    responses = baseline.get("responses")
    if not isinstance(responses, Mapping) or not responses:
        violations.append("baseline responses must be a non-empty mapping")
        return violations
    for case_id, pair_value in responses.items():
        label = f"case {case_id!r}"
        if not isinstance(pair_value, list) or len(pair_value) != REQUESTS_PER_CASE:
            violations.append(
                f"{label}: baseline must store exactly {REQUESTS_PER_CASE} response objects"
            )
            continue
        typed_pair = [item for item in pair_value if isinstance(item, Mapping)]
        if len(typed_pair) != REQUESTS_PER_CASE:
            violations.append(f"{label}: baseline responses must be objects")
            continue
        for payload in typed_pair:
            violations.extend(
                f"{label}: {violation}"
                for violation in response_contract_violations(payload, expected_model)
            )
        first_id = str(typed_pair[0].get("request_id"))
        second_id = str(typed_pair[1].get("request_id"))
        violations.extend(
            f"{label}: {violation}" for violation in request_pair_violations(first_id, second_id)
        )
        if canonical_payload(typed_pair[0]) != canonical_payload(typed_pair[1]):
            violations.append(
                f"{label}: stored responses differ beyond the request id (nondeterministic capture)"
            )
    return violations


def canonical_snapshot(baseline: Mapping[str, Any]) -> dict[str, Any]:
    """The hash projection: request IDs stripped, everything else preserved."""
    responses = baseline.get("responses")
    canonical_responses: dict[str, Any] = {}
    if isinstance(responses, Mapping):
        for case_id, pair in responses.items():
            if isinstance(pair, list):
                canonical_responses[str(case_id)] = [
                    canonical_payload(item) for item in pair if isinstance(item, Mapping)
                ]
            else:
                canonical_responses[str(case_id)] = pair
    return {**baseline, "responses": canonical_responses}


def snapshot_hash(baseline: Mapping[str, Any]) -> str:
    """Canonical SHA-256: sort_keys JSON of the request-ID-free snapshot.

    Independent of dict insertion order and interpreter hash seed, so a
    fresh process hashing the same snapshot must produce the same digest.
    """
    canonical = json.dumps(
        canonical_snapshot(baseline),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _preview(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)[:400]


def snapshot_drift_violations(
    baseline: Mapping[str, Any], observed: Mapping[str, Sequence[Mapping[str, Any]]]
) -> list[str]:
    """Canonical drift between the stored snapshot and a fresh capture."""
    violations: list[str] = []
    responses = baseline.get("responses")
    stored: dict[str, list[Mapping[str, Any]]] = {}
    malformed: dict[str, str] = {}
    if isinstance(responses, Mapping):
        for case_id, pair in responses.items():
            key = str(case_id)
            candidates = (
                [item for item in pair if isinstance(item, Mapping)]
                if isinstance(pair, list)
                else []
            )
            typed_pair = candidates
            if len(typed_pair) != REQUESTS_PER_CASE:
                malformed[key] = f"baseline must store exactly {REQUESTS_PER_CASE} response objects"
                continue
            stored[key] = typed_pair

    for case_id in sorted(set(stored) | set(malformed) | set(observed)):
        if case_id in malformed:
            violations.append(f"case {case_id!r}: {malformed[case_id]}")
            continue
        if case_id not in observed:
            violations.append(f"case {case_id!r}: missing from the observed capture")
            continue
        if case_id not in stored:
            violations.append(f"case {case_id!r}: not present in the baseline snapshot")
            continue
        observed_pair = list(observed[case_id])
        if len(observed_pair) != REQUESTS_PER_CASE:
            violations.append(
                f"case {case_id!r}: observed capture must hold exactly "
                f"{REQUESTS_PER_CASE} responses"
            )
            continue
        for position, (stored_payload, observed_payload) in enumerate(
            zip(stored[case_id], observed_pair, strict=True)
        ):
            stored_canonical = canonical_payload(stored_payload)
            observed_canonical = canonical_payload(observed_payload)
            if stored_canonical != observed_canonical:
                violations.append(
                    f"case {case_id!r} response #{position + 1}: snapshot drift; "
                    f"stored={_preview(stored_canonical)} observed={_preview(observed_canonical)}"
                )
    return violations


# --- observational report ----------------------------------------------------


def _percent(part: int, whole: int) -> str:
    if whole == 0:
        return "n/a"
    return f"{100.0 * part / whole:.1f}%"


def render_report(
    document: Mapping[str, Any],
    baseline: Mapping[str, Any],
    observed_labels: Mapping[str, str],
) -> str:
    """Deterministic observational agreement report (markdown).

    Byte-identical for identical inputs: no clock, no locale, no dict-order
    dependence. An optional generation stamp comes only from the
    ``MAT_REPORT_GENERATED_AT`` environment variable. Agreement is counted
    between the human ``expected_label`` and the observed model label and is
    never a pass/fail input.
    """
    cases = [case for case in document.get("cases", []) if isinstance(case, Mapping)]
    categories_value = document.get("categories")
    categories = (
        [str(item) for item in categories_value] if isinstance(categories_value, list) else []
    )
    total = len(cases)
    disagreements: list[tuple[str, str, str, str]] = []
    agree_by_category: dict[str, int] = {category: 0 for category in categories}
    cases_by_category: dict[str, int] = {category: 0 for category in categories}
    for case in cases:
        case_id = str(case.get("id"))
        category = str(case.get("category"))
        human = str(case.get("expected_label"))
        observed = observed_labels.get(case_id, "missing")
        cases_by_category[category] = cases_by_category.get(category, 0) + 1
        if observed == human:
            agree_by_category[category] = agree_by_category.get(category, 0) + 1
        else:
            disagreements.append((case_id, category, human, observed))
    agreed = total - len(disagreements)

    model_value = baseline.get("model")
    model_block = model_value if isinstance(model_value, Mapping) else {}
    lines = [
        "# Vietnamese robustness suite — observational agreement report",
        "",
        f"- suite: {SUITE_NAME}",
        f"- suite version: {document.get('version')}",
        f"- cases: {total} ({len(categories)} categories x {CASES_PER_CATEGORY})",
        f"- model: backend={model_block.get('backend')!r} "
        f"version={model_block.get('version')!r} (baseline serving)",
        f"- snapshot hash (canonical, request ids excluded): {snapshot_hash(baseline)}",
    ]
    generated_at = os.environ.get("MAT_REPORT_GENERATED_AT")
    if generated_at:
        lines.append(f"- generated: {generated_at}")
    lines += [
        "",
        "## Overall agreement (observational)",
        "",
        f"- human/model label agreement: {agreed}/{total} ({_percent(agreed, total)})",
        "",
        "## Agreement by category",
        "",
        "| category | cases | agree | disagree | disagreement rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for category in categories:
        case_count = cases_by_category.get(category, 0)
        category_agree = agree_by_category.get(category, 0)
        lines.append(
            f"| {category} | {case_count} | {category_agree} "
            f"| {case_count - category_agree} "
            f"| {_percent(case_count - category_agree, case_count)} |"
        )
    lines += ["", "## Disagreement details (case ID only; no reviewer personal data)", ""]
    if disagreements:
        for case_id, category, human, observed in sorted(disagreements):
            lines.append(f"- {case_id} ({category}): human={human} model={observed}")
    else:
        lines.append("- none")
    lines += [
        "",
        "## Scope",
        "",
        "Semantic agreement is observational for the frozen baseline; this report is not",
        "an accuracy claim and never gates a release. Hard gates (schema, corpus overlap,",
        "API contract, snapshot equality) are enforced by the runner exit code.",
    ]
    return "\n".join(lines) + "\n"


# --- CLI ---------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """Run every hard gate; exit 0 only when all pass and the report is written."""
    parser = argparse.ArgumentParser(
        description="Run the Vietnamese robustness suite against a live baseline server."
    )
    parser.add_argument(
        "--cases", type=Path, required=True, help="Path to vietnamese_robustness_v1.json"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to vietnamese_robustness_v1_baseline.json",
    )
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help="Base URL of the serving /predict endpoint"
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help=f"Report output path (default: {AI_ROOT / DEFAULT_REPORT_PATH})",
    )
    args = parser.parse_args(argv)

    document: dict[str, Any] = json.loads(args.cases.read_text(encoding="utf-8"))
    baseline: dict[str, Any] = json.loads(args.baseline.read_text(encoding="utf-8"))

    violations = schema_violations(document, full_matrix=True)
    normalizer = production_normalizer()
    violations += corpus_overlap_violations(document, build_corpus_index(normalizer), normalizer)
    violations += baseline_document_violations(baseline)

    model_value = baseline.get("model")
    expected_model: Mapping[str, Any] = model_value if isinstance(model_value, Mapping) else {}
    post = http_poster(str(args.base_url))
    observed: dict[str, list[dict[str, Any]]] = {}
    cases_value = document.get("cases", [])
    for case_value in cases_value if isinstance(cases_value, list) else []:
        if not isinstance(case_value, Mapping):
            violations.append(f"case entry is not an object: {case_value!r}")
            continue
        case_id = str(case_value.get("id"))
        result = capture_case(post, str(case_value.get("text")), case_id)
        observed[case_id] = result.payloads
        violations += result.violations
        for payload in result.payloads:
            violations += response_contract_violations(payload, expected_model)

    violations += snapshot_drift_violations(baseline, observed)
    if violations:
        for violation in violations:
            print(f"HARD-FAIL: {violation}", file=sys.stderr)
        print(
            f"{len(violations)} hard-contract violation(s); no report was written",
            file=sys.stderr,
        )
        return 1

    observed_labels = {case_id: str(pair[0]["label"]) for case_id, pair in observed.items()}
    report = render_report(document, baseline, observed_labels)
    report_path = args.report if args.report is not None else AI_ROOT / DEFAULT_REPORT_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    agreed = sum(
        1
        for case_value in observed
        if observed_labels[case_value] == _expected_label_for(document, case_value)
    )
    print(f"snapshot sha256: {snapshot_hash(baseline)}")
    print(f"observational agreement: {agreed}/{len(observed)}")
    print(f"report: {report_path}")
    return 0


def _expected_label_for(document: Mapping[str, Any], case_id: str) -> str:
    cases_value = document.get("cases", [])
    for case_value in cases_value if isinstance(cases_value, list) else []:
        if isinstance(case_value, Mapping) and str(case_value.get("id")) == case_id:
            return str(case_value.get("expected_label"))
    return "missing"


if __name__ == "__main__":
    raise SystemExit(main())
