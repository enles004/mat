"""Handler-level tests for the additive APIError problem-details rendering."""

import logging
import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.error_types import ClientError, ServerError
from src.api.errors import install_error_handlers
from src.libs.result import Error


def test_client_error_renders_problem_json() -> None:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/failure")
    def failure() -> None:
        raise ClientError(
            Error(code="SOME_CODE", message="explainable client mistake", reason="internal")
        )

    response = TestClient(app).get("/failure")

    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 400
    assert body["code"] == "SOME_CODE"
    assert body["title"] == "Bad Request"
    assert body["detail"] == "explainable client mistake"
    assert body["instance"] == "/failure"
    assert "internal" not in response.text
    assert re.fullmatch(r"https://mat\.local/problems/[\w-]+", body["type"])


def test_server_error_renders_masked_problem_json() -> None:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise ServerError(
            Error(code="BACKEND_DIED", message="secret database password", reason="traceback")
        )

    response = TestClient(app, raise_server_exceptions=False).get("/boom")

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 500
    assert body["code"] == "BACKEND_DIED"
    assert body["title"] == "Internal Server Error"
    assert body["detail"] == "Internal server error"
    assert "secret database password" not in response.text
    assert "traceback" not in response.text


def test_normal_routes_are_unaffected() -> None:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/ok")
    def ok() -> dict[str, bool]:
        return {"ok": True}

    response = TestClient(app).get("/ok")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_internal_reason_is_logged_but_never_serialized_to_the_wire(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The internal reason stays server-side — including non-serializable
    ``Exception`` reasons: every reason must appear in the error log
    (observability) and never in the problem body, title, or detail."""

    def raise_client() -> None:
        raise ClientError(
            Error(
                code="CLIENT_MISTAKE",
                message="explainable client mistake",
                reason="db password rejected",
            )
        )

    def raise_server() -> None:
        raise ServerError(
            Error(code="BACKEND_DIED", message="public mask", reason="traceback frame #7")
        )

    def raise_object_reason() -> None:
        raise ServerError(
            Error(
                code="WIRING_BUG",
                message="internal failure",
                reason=RuntimeError("secret stack"),
            )
        )

    app = FastAPI()
    install_error_handlers(app)
    app.get("/client")(raise_client)
    app.get("/server")(raise_server)
    app.get("/object-reason")(raise_object_reason)
    client = TestClient(app, raise_server_exceptions=False)

    with caplog.at_level(logging.ERROR, logger="src.api.errors"):
        responses = [
            client.get("/client"),
            client.get("/server"),
            client.get("/object-reason"),
        ]

    for response in responses:
        assert response.status_code in (400, 500)
        assert response.headers["content-type"].startswith("application/problem+json")
        body = response.json()
        assert "reason" not in body
    for secret in ("db password rejected", "traceback frame #7", "secret stack"):
        for response in responses:
            assert secret not in response.text
        assert secret in " ".join(record.getMessage() for record in caplog.records), secret

    object_response = responses[2]
    assert object_response.json()["code"] == "WIRING_BUG"
    assert "RuntimeError" not in object_response.text


def test_handler_rendering_failure_returns_sanitized_rfc9457_500() -> None:
    """If rendering a problem body itself explodes, the client still receives a
    sanitized RFC 9457 500 — never a plain-text fallback or internal detail."""

    class ExplodingProjection(ServerError):
        def to_dict(self) -> dict[str, str]:
            raise RuntimeError("projection exploded")

    def broken() -> None:
        raise ExplodingProjection(
            Error(code="BROKEN_RENDERER", message="should never surface", reason="internal")
        )

    app = FastAPI()
    install_error_handlers(app)
    app.get("/broken")(broken)

    response = TestClient(app, raise_server_exceptions=False).get("/broken")

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 500
    assert body["code"] == "INTERNAL_ERROR"
    assert body["detail"] == "Unexpected internal error"
    assert "projection exploded" not in response.text
    assert "should never surface" not in response.text
