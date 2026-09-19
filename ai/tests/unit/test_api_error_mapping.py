"""Unit tests for the error-code mapping table in src.api.error_mapping."""

import re
from dataclasses import FrozenInstanceError

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api.error_mapping as error_mapping
from src.api.error_mapping import ERROR_POLICY, ErrorPolicy, classify
from src.api.error_types import ClientError, ServerError
from src.api.errors import install_error_handlers
from src.api.middlewares.request_context import RequestContextMiddleware
from src.libs.result import Error


def test_known_code_maps_to_configured_class_and_status() -> None:
    error = Error(code="TEXT_EXCEEDS_MODEL_LIMIT", message="input too long")

    mapped = classify(error)

    assert isinstance(mapped, ClientError)
    assert mapped.get_status_code() == 422
    policy = ERROR_POLICY["TEXT_EXCEEDS_MODEL_LIMIT"]
    assert policy.public_title  # the policy carries the public title used on the wire


def test_every_production_code_maps_to_its_server_or_client_family() -> None:
    cases = {
        "MODEL_NOT_READY": (ServerError, 503),
        "TEXT_EXCEEDS_MODEL_LIMIT": (ClientError, 422),
        "INFERENCE_FAILED": (ServerError, 500),
        "HEALTH_CHECK_FAILED": (ServerError, 500),
    }
    for code, (expected_cls, expected_status) in cases.items():
        mapped = classify(Error(code=code, message="operation failed"))

        assert isinstance(mapped, expected_cls), code
        assert mapped.get_status_code() == expected_status, code
        assert ERROR_POLICY[code].public_title


def test_server_error_codes_mask_their_message_on_the_wire() -> None:
    for code in ("MODEL_NOT_READY", "INFERENCE_FAILED", "HEALTH_CHECK_FAILED"):
        mapped = classify(Error(code=code, message="db password — internal marker"))

        body = mapped.to_dict()
        assert body == {"code": code, "message": "Internal server error"}
        assert "internal marker" not in body.values()


def test_unknown_code_fails_closed_to_server_error_500() -> None:
    mapped = classify(Error(code="NEVER_DEFINED_ANYWHERE", message="anything"))

    assert isinstance(mapped, ServerError)
    assert mapped.get_status_code() == 500
    assert mapped.to_dict() == {
        "code": "NEVER_DEFINED_ANYWHERE",
        "message": "Internal server error",
    }


def test_missing_code_fails_closed_to_server_error_500() -> None:
    blank = Error(code="", message="anything")

    assert isinstance(classify(blank), ServerError)


def test_empty_table_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(error_mapping, "ERROR_POLICY", {})

    mapped = classify(Error(code="MODEL_NOT_READY", message="mapped"))

    assert isinstance(mapped, ServerError)
    assert mapped.get_status_code() == 500


def test_error_policy_is_frozen_and_typed() -> None:
    policy = ErrorPolicy(cls=ClientError, status_code=400, public_title="Bad request")

    with pytest.raises(FrozenInstanceError):
        policy.status_code = 500  # type: ignore[misc]


def test_unknown_code_renders_as_sanitized_500_problem_json() -> None:
    """End to end: an unknown internal code classifies to a ServerError whose
    wire form is a masked, correlated RFC 9457 500 — the code may name the
    family, but no message or reason internals surface."""

    def raise_unknown() -> None:
        raise classify(Error(code="NEVER_DEFINED_ANYWHERE", message="db password", reason="stack"))

    app = FastAPI()
    install_error_handlers(app)
    app.add_middleware(RequestContextMiddleware, max_request_bytes=65_536)
    app.get("/failure")(raise_unknown)

    response = TestClient(app, raise_server_exceptions=False).get(
        "/failure", headers={"X-Request-ID": "req-unknown-1"}
    )

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 500
    assert body["title"] == "Internal Server Error"
    assert body["detail"] == "Internal server error"
    assert body["request_id"] == "req-unknown-1"
    assert "db password" not in response.text
    assert "stack" not in response.text
    assert re.fullmatch(r"https://mat\.local/problems/[\w-]+", body["type"])
