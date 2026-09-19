"""Vietnamese behavioral showcase workflow.

Replays authored exploratory cases against a running ``/predict`` endpoint,
enforces the hard response contract and a deterministic-repeat check, and
writes observational evidence (JSON + optional Markdown).

This is a behavioral showcase only. It is not a benchmark, not an accuracy
estimate, and nothing it produces may ever feed training, calibration,
model selection, or any frozen artifact. Expected-vs-predicted tallies are
observational and never fail the run; only hard contract failures do.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from src.nlp.evaluation.demo import DemoEvaluator

DEFAULT_CASES_PATH = Path("data/vietnamese_demo_cases.json")
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT = httpx.Timeout(30.0)
DISCLAIMER = (
    "These cases are authored exploratory examples for a behavioral showcase. "
    "They are not a benchmark and not an accuracy estimate, and they must "
    "never influence training, calibration, model selection, or any frozen "
    "artifact. Mismatches are reported honestly and kept in the evidence."
)

_EVALUATOR = DemoEvaluator()


def _as_number(value: Any) -> float | None:
    """Return the value as a float, or None when it is not a JSON number."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _format_confidence(result: Mapping[str, Any]) -> str:
    confidence = _as_number(result.get("confidence"))
    if confidence is not None:
        return f"{confidence:.4f}"
    return "?"


def _cell(text: str) -> str:
    """Make arbitrary text safe for one Markdown table cell."""
    return text.replace("|", "\\|").replace("\n", " ")


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Read and minimally shape-check the authored cases JSON."""
    document: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("cases"), list):
        raise ValueError("cases file must be a JSON object with a 'cases' list")
    cases: list[dict[str, Any]] = []
    for item in document["cases"]:
        if not isinstance(item, dict):
            raise ValueError("every case must be a JSON object")
        cases.append(item)
    return cases


def fetch_prediction(
    client: httpx.Client, base_url: str, text: str
) -> tuple[int, dict[str, Any] | None, str | None]:
    """POST one ``/predict`` request; return (status, payload, error)."""
    try:
        response = client.post(f"{base_url.rstrip('/')}/predict", json={"text": text})
    except httpx.HTTPError as error:
        return -1, None, f"transport error: {type(error).__name__}: {error}"
    try:
        payload: Any = response.json()
    except ValueError as error:
        return response.status_code, None, f"non-JSON response body: {error}"
    if not isinstance(payload, dict):
        return response.status_code, None, "response body is not a JSON object"
    return response.status_code, payload, None


def run_case(client: httpx.Client, base_url: str, case: Mapping[str, Any]) -> dict[str, Any]:
    """Send one case twice, hard-check both responses, compare the repeat."""
    text = str(case.get("text", ""))
    first_status, first, first_error = fetch_prediction(client, base_url, text)
    second_status, second, second_error = fetch_prediction(client, base_url, text)

    violations: list[str] = []
    if first is None or first_error is not None:
        violations.append(f"first request failed: {first_error or 'missing payload'}")
    else:
        violations.extend(_EVALUATOR.check_response(first, first_status))
    if second is None or second_error is not None:
        violations.append(f"repeat request failed: {second_error or 'missing payload'}")
    else:
        violations.extend(_EVALUATOR.check_response(second, second_status))

    repeat_equal = (
        first is not None and second is not None and _EVALUATOR.responses_equal(first, second)
    )
    if not repeat_equal:
        violations.append(
            "deterministic-repeat check failed: semantic fields differ between responses"
        )

    source = first if first is not None else second
    expected_label = str(case.get("expected_label", ""))
    model_info = source.get("model") if source is not None else None
    model_block = dict(model_info) if isinstance(model_info, Mapping) else None
    predicted_label = source.get("label") if source is not None else None
    confidence = source.get("confidence") if source is not None else None
    uncertain_flag = source.get("uncertain") if source is not None else None
    return {
        "id": str(case.get("id", "?")),
        "category": str(case.get("category", "?")),
        "text": text,
        "human_interpretation": str(case.get("human_interpretation", "")),
        "expected_label": expected_label,
        "predicted_label": predicted_label,
        "confidence": confidence,
        "uncertain": uncertain_flag,
        "model": model_block,
        "first_status": first_status,
        "second_status": second_status,
        "first_request_id": first.get("request_id") if first is not None else None,
        "second_request_id": second.get("request_id") if second is not None else None,
        "repeat_equal": repeat_equal,
        "hard_check_violations": violations,
        "hard_check_passed": not violations,
        "matches_expected": predicted_label == expected_label,
    }


def render_markdown(
    results: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
    *,
    command: str,
    timestamp: str,
    base_url: str,
) -> str:
    """Render the observational evidence document."""
    agreement = summary.get("agreement", {})
    lines: list[str] = [
        "# Vietnamese behavioral demo — evidence",
        "",
        f"> {DISCLAIMER}",
        "",
        f"- Generated (UTC): {timestamp}",
        f"- Command: `{command}`",
        f"- Base URL: {base_url}",
        f"- Cases: {summary.get('total_cases', len(results))} "
        f"({len(summary.get('categories', {}))} categories)",
        "",
        "## Per-case results",
        "",
        "| id | category | text | expected | predicted | confidence | uncertain | repeat-equal |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for result in results:
        predicted = result.get("predicted_label")
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {} |".format(
                _cell(str(result.get("id", "?"))),
                _cell(str(result.get("category", "?"))),
                _cell(str(result.get("text", ""))),
                _cell(str(result.get("expected_label", "?"))),
                _cell(str(predicted if predicted is not None else "?")),
                _format_confidence(result),
                str(result.get("uncertain")),
                str(result.get("repeat_equal")),
            )
        )
    lines.extend(
        [
            "",
            "## Per-category agreement",
            "",
            "| category | cases | agreed | mismatched | predicted-uncertain |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    categories = summary.get("categories", {})
    if isinstance(categories, Mapping):
        for name, tallies in categories.items():
            if isinstance(tallies, Mapping):
                lines.append(
                    f"| {name} | {tallies.get('total')} | {tallies.get('agreed')} "
                    f"| {tallies.get('mismatched')} | {tallies.get('predicted_uncertain')} |"
                )
    lines.extend(
        [
            "",
            "## Overall",
            "",
            f"- Agreement: {agreement.get('agreed', 0)}/{summary.get('total_cases', 0)} "
            f"({agreement.get('mismatched', 0)} mismatches, kept deliberately)",
            f"- Hard-check failures: {summary.get('hard_check_failures', 0)}",
            f"- Deterministic-repeat mismatches: {summary.get('repeat_mismatches', 0)}",
            "",
            "Agreement here is observational commentary on authored examples, "
            "not an accuracy estimate: the sample is small, hand-picked to be "
            "hard, and every case was written by one author.",
            "",
            "This file is generated by `mat-demo-vietnamese`; do not hand-edit.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mat-demo-vietnamese",
        description=(
            "Behavioral showcase: replay authored Vietnamese demo cases "
            "against a running MAT /predict endpoint (not a benchmark)."
        ),
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output", type=Path, required=True, help="JSON results path")
    parser.add_argument("--markdown", type=Path, default=None, help="evidence table path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Replay authored Vietnamese demo cases against a running endpoint."""
    args = build_parser().parse_args(argv)
    try:
        cases = load_cases(args.cases)
    except (OSError, ValueError) as error:
        print(f"error: cannot load cases from {args.cases}: {error}", file=sys.stderr)
        return 2

    effective_argv = sys.argv[1:] if argv is None else list(argv)
    command = "mat-demo-vietnamese " + " ".join(effective_argv)
    timestamp = _utc_now_iso()
    print(f"Vietnamese behavioral demo: {len(cases)} cases against {args.base_url}")

    results: list[dict[str, Any]] = []
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        for case in cases:
            result = run_case(client, args.base_url, case)
            results.append(result)
            predicted = result["predicted_label"]
            marker = "ok  " if result["hard_check_passed"] else "FAIL"
            print(
                f"[{marker}] {result['id']} ({result['category']}): "
                f"expected={result['expected_label']} predicted={predicted} "
                f"confidence={_format_confidence(result)} "
                f"uncertain={result['uncertain']} repeat_equal={result['repeat_equal']}"
            )

    summary = _EVALUATOR.summarize(results)
    document = {
        "version": 1,
        "generated_at_utc": timestamp,
        "command": command,
        "base_url": args.base_url,
        "disclaimer": DISCLAIMER,
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(
            render_markdown(
                results, summary, command=command, timestamp=timestamp, base_url=args.base_url
            ),
            encoding="utf-8",
        )

    agreement = summary["agreement"]
    print(
        f"Agreement (observational): {agreement['agreed']}/{summary['total_cases']} "
        f"agreed, {agreement['mismatched']} mismatched"
    )
    for name, tallies in summary["categories"].items():
        print(
            f"  {name}: {tallies['agreed']}/{tallies['total']} agreed, "
            f"{tallies['mismatched']} mismatched, "
            f"{tallies['predicted_uncertain']} uncertain-flagged"
        )
    failed_ids = [str(result["id"]) for result in results if not result["hard_check_passed"]]
    if failed_ids:
        print(
            f"HARD CHECK FAILURES ({len(failed_ids)} case ids): {', '.join(failed_ids)}",
            file=sys.stderr,
        )
        return 1
    print("All hard checks passed (HTTP 200, response contract, deterministic repeat).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
