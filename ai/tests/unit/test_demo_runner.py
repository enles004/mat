"""Unit tests for the demo runner's pure functions (no HTTP server)."""

from typing import Any

from scripts.demo_api_vietnamese import render_markdown
from src.nlp.evaluation.demo import DemoEvaluator


def _valid_payload() -> dict[str, Any]:
    return {
        "label": "positive",
        "confidence": 0.82,
        "scores": {"negative": 0.06, "neutral": 0.12, "positive": 0.82},
        "uncertain": False,
        "model": {"backend": "linear", "version": "0.1.0", "degraded": False},
        "request_id": "req_a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
    }


def test_valid_payload_passes_with_zero_violations() -> None:
    assert DemoEvaluator().check_response(_valid_payload()) == []
    assert DemoEvaluator().check_response(_valid_payload(), status_code=200) == []


def test_missing_score_key_is_a_violation() -> None:
    payload = _valid_payload()
    del payload["scores"]["neutral"]
    violations = DemoEvaluator().check_response(payload)
    assert any("scores keys" in violation for violation in violations)


def test_scores_not_summing_to_one_is_a_violation() -> None:
    payload = _valid_payload()
    payload["scores"] = {"negative": 0.05, "neutral": 0.05, "positive": 0.80}
    violations = DemoEvaluator().check_response(payload)
    assert any("sum to" in violation for violation in violations)


def test_wrong_backend_is_a_violation() -> None:
    payload = _valid_payload()
    payload["model"]["backend"] = "transformer"
    violations = DemoEvaluator().check_response(payload)
    assert any("model.backend" in violation for violation in violations)


def test_degraded_true_is_a_violation() -> None:
    payload = _valid_payload()
    payload["model"]["degraded"] = True
    violations = DemoEvaluator().check_response(payload)
    assert any("model.degraded" in violation for violation in violations)


def test_bad_request_id_is_a_violation() -> None:
    payload = _valid_payload()
    payload["request_id"] = "abc123"
    violations = DemoEvaluator().check_response(payload)
    assert any("request_id" in violation for violation in violations)


def test_label_typo_is_a_violation() -> None:
    payload = _valid_payload()
    payload["label"] = "positiv"
    violations = DemoEvaluator().check_response(payload)
    assert any("label" in violation for violation in violations)


def test_non_200_status_is_a_violation() -> None:
    violations = DemoEvaluator().check_response(_valid_payload(), status_code=503)
    assert any("http status" in violation for violation in violations)


def test_equal_responses_are_deterministic() -> None:
    assert DemoEvaluator().responses_equal(_valid_payload(), _valid_payload())


def test_differing_label_is_not_deterministic() -> None:
    second = _valid_payload()
    second["label"] = "neutral"
    assert not DemoEvaluator().responses_equal(_valid_payload(), second)


def _result(
    category: str,
    expected: str,
    predicted: str | None,
    *,
    uncertain: bool = False,
    hard_check_passed: bool = True,
    repeat_equal: bool = True,
) -> dict[str, Any]:
    return {
        "id": f"{category}-{expected}-{predicted}",
        "category": category,
        "expected_label": expected,
        "predicted_label": predicted,
        "uncertain": uncertain,
        "hard_check_passed": hard_check_passed,
        "repeat_equal": repeat_equal,
        "matches_expected": predicted == expected,
    }


def test_summarize_counts_categories_agreement_and_flags() -> None:
    results = [
        _result("teencode", "positive", "positive"),
        _result("teencode", "negative", "positive"),
        _result("negation", "neutral", "neutral", uncertain=True),
        _result("slang", "negative", None, hard_check_passed=False, repeat_equal=False),
    ]
    summary = DemoEvaluator().summarize(results)
    assert summary["total_cases"] == 4
    assert summary["agreement"] == {"agreed": 2, "mismatched": 2}
    assert summary["hard_check_failures"] == 1
    assert summary["repeat_mismatches"] == 1
    assert summary["categories"]["teencode"] == {
        "total": 2,
        "agreed": 1,
        "mismatched": 1,
        "predicted_uncertain": 0,
    }
    assert summary["categories"]["negation"]["predicted_uncertain"] == 1
    assert summary["categories"]["slang"]["mismatched"] == 1


def test_render_markdown_includes_disclaimer_and_every_case_row() -> None:
    results = [
        _result("teencode", "positive", "positive"),
        _result("slang", "negative", "neutral", uncertain=True),
    ]
    markdown = render_markdown(
        results,
        DemoEvaluator().summarize(results),
        command="mat-demo-vietnamese --cases data/vietnamese_demo_cases.json",
        timestamp="2026-09-17T00:00:00Z",
        base_url="http://127.0.0.1:8123",
    )
    assert "not a benchmark" in markdown
    assert "mat-demo-vietnamese --cases data/vietnamese_demo_cases.json" in markdown
    assert "2026-09-17T00:00:00Z" in markdown
    for row_marker in ("| teencode-", "| slang-"):
        assert row_marker in markdown
