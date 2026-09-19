"""Harness for the Vietnamese robustness data-as-test suite (plan Task 7).

Four nodes, exactly:

1. ``test_suite_schema_and_coverage_contract`` — schema/version, unique
   IDs/texts, the exact category matrix (15 x 6 with 2 positive / 2 neutral /
   2 negative human labels per category), one primary category per case.
2. ``test_cases_are_disjoint_from_frozen_corpora`` — no exact normalized text
   and no six-token n-gram overlap with the 900 dataset + 74 challenge + 54
   demo texts, compared under the system's own normalization catalog.
3. ``test_two_real_predict_requests_satisfy_api_invariants`` — two real
   ``/predict`` requests per case with distinct valid request IDs, identical
   payloads except the request ID, finite three-label scores summing to one,
   canonical score-key order, baseline backend/version, ``degraded=false``.
4. ``test_baseline_snapshot_and_report_are_reproducible`` — canonical
   snapshot comparison (request ID excluded from equality only), snapshot
   hash reproduced by a fresh interpreter process, byte-deterministic report
   rendering.

Each sabotage mode (wrong category count, duplicated text, corpus copy,
invalid scores, reused request ID, nondeterministic response, snapshot
drift) is exercised inside its node through the shared contract helpers in
``scripts.run_vietnamese_robustness`` — the helpers stay in one place so the
pytest harness and the release-profile runner enforce the identical rules.

The reviewed fixture (``tests/fixtures/vietnamese_robustness_v1.json``) and
its baseline snapshot (``..._baseline.json``) are authored in plan Step 4-5.
Until those files exist, every node runs the same contract machinery against
a reduced 1-category x 6-case fixture built here; the reduced fixture is a
valid suite under the internal-consistency rules and must still fail the
full 15-category matrix check.
"""

import asyncio
import copy
import hashlib
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import FastAPI

from scripts.run_vietnamese_robustness import (
    baseline_document_violations,
    build_corpus_index,
    canonical_payload,
    capture_case,
    corpus_overlap_violations,
    live_baseline_document,
    load_corpus_texts,
    ngrams_of,
    production_normalizer,
    render_report,
    request_ids_for,
    request_pair_violations,
    response_contract_violations,
    schema_violations,
    snapshot_drift_violations,
    snapshot_hash,
    system_normalize,
    tokenize,
)
from src.api.dependencies import (
    RuntimeState,
    load_model_from_artifact,
    startup_load,
    startup_normalization,
)
from src.api.server import create_app
from src.core.settings import Settings

AI_ROOT = Path(__file__).resolve().parents[2]
REAL_CASES_PATH = AI_ROOT / "tests" / "fixtures" / "vietnamese_robustness_v1.json"
REAL_BASELINE_PATH = AI_ROOT / "tests" / "fixtures" / "vietnamese_robustness_v1_baseline.json"
BASELINE_MANIFEST_PATH = AI_ROOT / "artifacts" / "baseline" / "manifest.json"

TEMP_CATEGORY = "regional-north"
# Temporary harness rows (1 of the 15 categories, 2/2/2 human labels).
# Verified disjoint from the frozen corpora under the production catalog;
# they are stand-ins only and never enter the reviewed 90-case suite.
TEMP_CASES: tuple[tuple[str, str, str], ...] = (
    (
        "RN-01",
        "positive",
        "Hôm qua tôi đưa chú ruột ra bến xe Mỹ Đình lấy xe mới, chú khen chạy mượt hẳn.",
    ),
    (
        "RN-02",
        "positive",
        "Phiên bản điện bản giới hạn ở Hà Nội này sạc một tiếng là đủ đi cả tuần, quá tiện.",
    ),
    (
        "RN-03",
        "neutral",
        "Chị tôi vừa đổi từ xe số sàn sang xe số tự động, bảo là chạy trong phố đỡ mỏi chân.",
    ),
    (
        "RN-04",
        "neutral",
        "Cửa hàng ở Cầu Giấy báo giá lăn bánh xong là yên tâm, phụ kiện tặng kèm tùy thời điểm.",
    ),
    (
        "RN-05",
        "negative",
        "Ra đại lý phía Bắc xem xe, nhân viên nói leo đủ thứ khiến tôi thấy khó chịu hẳn.",
    ),
    (
        "RN-06",
        "negative",
        "Trời Hà Nội lạnh độ này để xe ngoài sân qua đêm, sáng ra đề máy ì ạch nghe rất ức chế.",
    ),
)


def temp_suite_document() -> dict[str, Any]:
    """The reduced stand-in fixture used until Step 4 authors the real 90."""
    return {
        "schema_version": "1",
        "suite": "vietnamese-robustness",
        "version": 1,
        "description": (
            "Temporary reduced harness fixture (1 of 15 categories); replaced "
            "by the reviewed 90-case suite in Task 7 Step 4."
        ),
        "categories": [TEMP_CATEGORY],
        "cases": [
            {
                "id": case_id,
                "category": TEMP_CATEGORY,
                "text": text,
                "expected_label": label,
                "secondary_tags": [],
                "notes": "Temporary harness row; not part of the reviewed suite.",
            }
            for case_id, label, text in TEMP_CASES
        ],
    }


class AsgiPredictor:
    """Real ``/predict`` over in-process ASGI transport (production app)."""

    def __init__(self, app: FastAPI) -> None:
        self._transport = httpx.ASGITransport(app=app)

    def post(self, text: str, request_id: str) -> dict[str, Any]:
        async def call() -> httpx.Response:
            async with httpx.AsyncClient(
                transport=self._transport, base_url="http://testserver"
            ) as client:
                return await client.post(
                    "/predict", json={"text": text}, headers={"x-request-id": request_id}
                )

        response = asyncio.run(call())
        assert response.status_code == 200, (response.status_code, response.text)
        return response.json()


@pytest.fixture(scope="module")
def suite_document() -> dict[str, Any]:
    """The reviewed suite once authored; the reduced fixture until then."""
    if REAL_CASES_PATH.is_file():
        return json.loads(REAL_CASES_PATH.read_text(encoding="utf-8"))
    return temp_suite_document()


@pytest.fixture(scope="module")
def baseline_manifest() -> dict[str, Any]:
    """The frozen serving artifact manifest (read-only reference)."""
    return json.loads(BASELINE_MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def baseline_app() -> FastAPI:
    """Production composition root, loaded exactly as the lifespan does."""
    app = create_app(Settings(), load_model_from_artifact)
    state: RuntimeState = app.state.runtime
    startup_load(state, Settings(), load_model_from_artifact)
    startup_normalization(state, Settings())
    assert state.ready, state.startup_error
    assert not state.degraded
    assert state.backend == "linear"
    return app


@pytest.fixture(scope="module")
def predictor(baseline_app: FastAPI) -> AsgiPredictor:
    return AsgiPredictor(baseline_app)


# --- Node 1: schema and coverage -------------------------------------------


def test_suite_schema_and_coverage_contract(suite_document: dict[str, Any]) -> None:
    using_real_fixture = REAL_CASES_PATH.is_file()
    assert schema_violations(suite_document, full_matrix=using_real_fixture) == []
    if not using_real_fixture:
        # The reduced stand-in is internally valid but must fail the full
        # 15-category matrix, proving the full-matrix branch is enforced.
        assert schema_violations(suite_document, full_matrix=True) != []

    def mutate_category_count(document: dict[str, Any]) -> dict[str, Any]:
        trimmed = copy.deepcopy(document)
        trimmed["cases"].pop()
        return trimmed

    def mutate_duplicated_text(document: dict[str, Any]) -> dict[str, Any]:
        duplicated = copy.deepcopy(document)
        duplicated["cases"][1]["text"] = duplicated["cases"][0]["text"]
        return duplicated

    def mutate_label_balance(document: dict[str, Any]) -> dict[str, Any]:
        skewed = copy.deepcopy(document)
        # Flip one negative row (cases[4]) so the negative quota breaks.
        skewed["cases"][4]["expected_label"] = "positive"
        return skewed

    sabotages: list[tuple[str, Callable[[dict[str, Any]], dict[str, Any]], str]] = [
        ("wrong-category-count", mutate_category_count, "cases, expected 6"),
        ("duplicated-text", mutate_duplicated_text, "case texts must be unique"),
        ("label-imbalance", mutate_label_balance, "negative cases, expected 2"),
    ]
    for name, mutator, fragment in sabotages:
        violations = schema_violations(mutator(suite_document), full_matrix=False)
        assert violations, f"sabotage {name} was not detected"
        assert any(fragment in violation for violation in violations), (
            name,
            violations,
        )


# --- Node 2: cross-corpus disjointness --------------------------------------


def test_cases_are_disjoint_from_frozen_corpora(suite_document: dict[str, Any]) -> None:
    normalizer = production_normalizer()
    index = build_corpus_index(normalizer)
    corpus = load_corpus_texts()
    assert {source: len(items) for source, items in corpus.items()} == {
        "showcase": 54,
        "challenge": 74,
        "dataset": 900,
    }
    assert len(index.exact) > 0 and len(index.ngrams) > 0

    assert corpus_overlap_violations(suite_document, index, normalizer) == []

    source_text = _long_dataset_text(corpus, normalizer)

    def mutate_corpus_copy(document: dict[str, Any]) -> dict[str, Any]:
        copied = copy.deepcopy(document)
        copied["cases"][0]["text"] = source_text
        return copied

    def mutate_ngram_retaining_copy(document: dict[str, Any]) -> dict[str, Any]:
        variant = copy.deepcopy(document)
        variant["cases"][1]["text"] = _ngram_retaining_variant(source_text, normalizer)
        return variant

    copied_violations = corpus_overlap_violations(
        mutate_corpus_copy(suite_document), index, normalizer
    )
    assert copied_violations, "verbatim corpus copy was not detected"
    assert any("exactly matches" in violation for violation in copied_violations)

    variant_violations = corpus_overlap_violations(
        mutate_ngram_retaining_copy(suite_document), index, normalizer
    )
    assert variant_violations, "six-token n-gram overlap was not detected"
    assert any("n-gram" in violation for violation in variant_violations)
    assert not any("exactly matches" in violation for violation in variant_violations), (
        "the n-gram sabotage must be caught by the n-gram gate, not the exact gate"
    )


def _long_dataset_text(corpus: dict[str, list[tuple[str, str]]], normalizer: Any) -> str:
    """First dataset row (frozen order) whose normalized form has at least
    eight tokens, so an eight-token n-gram window is available."""
    for _, text in corpus["dataset"]:
        if len(tokenize(system_normalize(normalizer, text))) >= 8:
            return text
    raise AssertionError("no dataset row with >= 8 normalized tokens")


def _ngram_retaining_variant(text: str, normalizer: Any) -> str:
    """Rewrite a corpus text so its normalized form differs while a run of
    eight tokens survives re-normalization — the pure n-gram sabotage."""
    tokens = tokenize(system_normalize(normalizer, text))
    window = tokens[2:10]
    assert len(window) == 8, tokens
    variant = (
        "Mình muốn bàn thêm một chút: "
        + " ".join(window)
        + " cũng là điểm được nhiều người nhắc lại gần đây."
    )
    survived = ngrams_of(tokenize(system_normalize(normalizer, variant)))
    assert any(tuple(window[start : start + 6]) in survived for start in range(3)), (
        "the eight-token window did not survive re-normalization",
        window,
    )
    return variant


# --- Node 3: two-request API invariants -------------------------------------


def test_two_real_predict_requests_satisfy_api_invariants(
    suite_document: dict[str, Any],
    predictor: AsgiPredictor,
    baseline_manifest: dict[str, Any],
) -> None:
    expected_model = {
        "backend": baseline_manifest["backend"],
        "version": baseline_manifest["model_version"],
    }
    last_payload: dict[str, Any] | None = None
    for case in suite_document["cases"]:
        result = capture_case(predictor.post, case["text"], case["id"])
        assert result.violations == [], (case["id"], result.violations)
        first_id, second_id = request_ids_for(case["id"])
        assert result.payloads[0]["request_id"] == first_id, case["id"]
        assert result.payloads[1]["request_id"] == second_id, case["id"]
        for payload in result.payloads:
            violations = response_contract_violations(payload, expected_model)
            assert violations == [], (case["id"], violations)
        assert canonical_payload(result.payloads[0]) == canonical_payload(result.payloads[1]), (
            f"case {case['id']}: identical payloads must produce identical responses"
        )
        last_payload = result.payloads[0]

    assert last_payload is not None
    template = copy.deepcopy(last_payload)

    def with_scores(payload: dict[str, Any], scores: dict[str, Any]) -> dict[str, Any]:
        mutant = copy.deepcopy(payload)
        mutant["scores"] = scores
        return mutant

    original_scores = template["scores"]
    shifted_sum = dict(original_scores)
    shifted_sum["positive"] = float(original_scores["positive"]) - 0.1
    reordered = {key: original_scores[key] for key in reversed(list(original_scores))}
    missing_key = {key: value for key, value in original_scores.items() if key != "neutral"}
    extra_key = {**original_scores, "unknown": 0.0}

    def with_model(payload: dict[str, Any], **updates: Any) -> dict[str, Any]:
        mutant = copy.deepcopy(payload)
        mutant["model"] = {**mutant["model"], **updates}
        return mutant

    sabotages: list[tuple[dict[str, Any], str]] = [
        (
            with_scores(template, {**original_scores, "positive": float("nan")}),
            "finite",
        ),
        (
            with_scores(template, {**original_scores, "positive": float("inf")}),
            "finite",
        ),
        (with_scores(template, shifted_sum), "sum to 1"),
        (with_scores(template, reordered), "canonical order"),
        (with_scores(template, missing_key), "must cover exactly"),
        (with_scores(template, extra_key), "must cover exactly"),
        (with_model(template, backend="transformer"), "backend"),
        (with_model(template, version="9.9.9"), "version"),
        (with_model(template, degraded=True), "degraded"),
    ]
    for mutant, fragment in sabotages:
        violations = response_contract_violations(mutant, expected_model)
        assert violations, f"sabotage {fragment!r} was not detected"
        assert any(fragment in violation for violation in violations), (
            fragment,
            violations,
        )

    # Reused request ID and malformed IDs must fail the pair gate.
    assert request_pair_violations("same-id", "same-id") != []
    assert request_pair_violations("bad id!", "other-id") != []
    assert request_pair_violations("x" * 129, "other-id") != []
    assert request_pair_violations("a-valid-id", "another-valid-id") == []

    # Nondeterministic-response detection: the canonical equality mechanism
    # exercised above on every real pair flags any mutated response.
    drifted = copy.deepcopy(template)
    drifted["confidence"] = float(template["confidence"]) + 1e-9
    assert canonical_payload(template) != canonical_payload(drifted)


# --- Node 4: snapshot and report reproducibility ----------------------------


@pytest.fixture(scope="module")
def baseline_document(suite_document: dict[str, Any], predictor: AsgiPredictor) -> dict[str, Any]:
    """The authored baseline snapshot once it exists; otherwise captured live."""
    if REAL_BASELINE_PATH.is_file():
        return json.loads(REAL_BASELINE_PATH.read_text(encoding="utf-8"))
    observed = {
        case["id"]: capture_case(predictor.post, case["text"], case["id"]).payloads
        for case in suite_document["cases"]
    }
    return live_baseline_document(suite_document, observed)


def test_baseline_snapshot_and_report_are_reproducible(
    suite_document: dict[str, Any],
    baseline_document: dict[str, Any],
    predictor: AsgiPredictor,
    tmp_path: Path,
) -> None:
    assert baseline_document_violations(baseline_document) == []
    for pair in baseline_document["responses"].values():
        assert request_pair_violations(pair[0]["request_id"], pair[1]["request_id"]) == []

    # A fresh capture (new request IDs) must match the snapshot canonically.
    fresh = {
        case["id"]: capture_case(predictor.post, case["text"], case["id"]).payloads
        for case in suite_document["cases"]
    }
    assert snapshot_drift_violations(baseline_document, fresh) == []
    reproduced = {**baseline_document, "responses": fresh}
    assert snapshot_hash(baseline_document) == snapshot_hash(reproduced)

    # The snapshot hash must reproduce identically in a fresh interpreter
    # process (sort_keys canonical JSON: independent of hash seed / order).
    baseline_path = tmp_path / "vietnamese_robustness_v1_baseline.json"
    baseline_path.write_text(json.dumps(baseline_document, ensure_ascii=False), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-c", _FRESH_PROCESS_HASH_SCRIPT, str(baseline_path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=AI_ROOT,
        timeout=120,
    )
    assert completed.stdout.strip() == snapshot_hash(baseline_document)

    # Report rendering is byte-deterministic across repeated calls.
    observed_labels = {case_id: pair[0]["label"] for case_id, pair in sorted(fresh.items())}
    report_one = render_report(suite_document, baseline_document, observed_labels)
    report_two = render_report(suite_document, baseline_document, observed_labels)
    assert report_one == report_two
    assert report_one.strip()
    assert (
        hashlib.sha256(report_one.encode("utf-8")).hexdigest()
        == hashlib.sha256(report_two.encode("utf-8")).hexdigest()
    )

    # Sabotage: snapshot drift (label, confidence, metadata) must be caught.
    first_case = suite_document["cases"][0]["id"]
    for name, mutator in _drift_sabotages(baseline_document, first_case):
        assert snapshot_drift_violations(mutator(), fresh), (
            f"snapshot drift sabotage {name} was not detected"
        )

    # Sabotage: a nondeterministic stored pair must fail snapshot integrity.
    nondeterministic = copy.deepcopy(baseline_document)
    pair = nondeterministic["responses"][first_case]
    pair[1] = {**pair[1], "confidence": float(pair[1]["confidence"]) + 1e-9}
    violations = baseline_document_violations(nondeterministic)
    assert violations, "nondeterministic stored pair was not detected"
    assert any("differ" in violation for violation in violations)


def _drift_sabotages(
    baseline_document: dict[str, Any], case_id: str
) -> list[tuple[str, Callable[[], dict[str, Any]]]]:
    def label_drift() -> dict[str, Any]:
        drifted = copy.deepcopy(baseline_document)
        stored = drifted["responses"][case_id][0]
        stored["label"] = "positive" if stored["label"] != "positive" else "neutral"
        return drifted

    def confidence_drift() -> dict[str, Any]:
        drifted = copy.deepcopy(baseline_document)
        stored = drifted["responses"][case_id][0]
        stored["confidence"] = float(stored["confidence"]) + 1e-9
        return drifted

    def metadata_drift() -> dict[str, Any]:
        drifted = copy.deepcopy(baseline_document)
        drifted["responses"][case_id][0]["model"] = {
            **drifted["responses"][case_id][0]["model"],
            "version": "0.0.0",
        }
        return drifted

    return [
        ("label", label_drift),
        ("confidence", confidence_drift),
        ("model-metadata", metadata_drift),
    ]


_FRESH_PROCESS_HASH_SCRIPT = """import json
import sys

from scripts.run_vietnamese_robustness import snapshot_hash

with open(sys.argv[1], encoding="utf-8") as handle:
    document = json.load(handle)
print(snapshot_hash(document))
"""
