import asyncio
import re

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.middlewares.request_context import RequestContextMiddleware, RequestTooLarge


def test_body_larger_than_limit_returns_413() -> None:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware, max_request_bytes=32)

    @app.post("/predict")
    def predict() -> dict[str, bool]:
        return {"ok": True}

    response = TestClient(app).post("/predict", content=b"x" * 33)
    assert response.status_code == 413
    assert response.headers["content-type"].startswith("application/problem+json")


def _limited_app(max_request_bytes: int) -> FastAPI:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware, max_request_bytes=max_request_bytes)

    @app.post("/predict")
    async def predict(payload: dict[str, object]) -> dict[str, object]:
        return payload

    return app


def test_chunked_generator_body_crossing_limit_returns_413() -> None:
    app = _limited_app(max_request_bytes=32)
    chunks = [b"a" * 16, b"b" * 16, b"c" * 1]

    response = TestClient(app).post(
        "/predict", content=(chunk for chunk in chunks)
    )
    assert response.status_code == 413
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 413
    assert body["code"] == "CONTENT_TOO_LARGE"
    assert body["instance"] == "/predict"
    assert re.fullmatch(r"req_[0-9a-f]{32}", body["request_id"])
    assert response.headers["X-Request-ID"] == body["request_id"]


def test_content_length_over_limit_is_rejected_before_body_read() -> None:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware, max_request_bytes=32)
    bodies_read: list[bytes] = []

    @app.post("/predict")
    async def predict(request: Request) -> dict[str, bool]:
        bodies_read.append(await request.body())
        return {"ok": True}

    response = TestClient(app).post(
        "/predict", content=b"y" * 33, headers={"X-Request-ID": "client.413-echo"}
    )
    assert response.status_code == 413
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "CONTENT_TOO_LARGE"
    assert response.headers["X-Request-ID"] == "client.413-echo"
    assert re.fullmatch(r"total;dur=[0-9.]+", response.headers["Server-Timing"])
    assert bodies_read == [], "oversized Content-Length must be rejected before reading the body"


def test_chunked_body_read_directly_in_endpoint_returns_413() -> None:
    app = FastAPI()
    app.add_middleware(RequestContextMiddleware, max_request_bytes=32)

    @app.post("/predict")
    async def predict(request: Request) -> dict[str, int]:
        return {"length": len(await request.body())}

    chunks = [b"a" * 20, b"b" * 20]

    response = TestClient(app).post("/predict", content=(chunk for chunk in chunks))
    assert response.status_code == 413
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["code"] == "CONTENT_TOO_LARGE"
    assert body["instance"] == "/predict"


def test_request_too_large_after_response_start_propagates() -> None:
    # A body-limit trip after the response already started must abort the
    # connection, not be swallowed. Driven at the ASGI level (no TestClient)
    # on purpose: if the middleware ever swallows the exception again, this
    # fails as an error instead of hanging a real client until timeout.
    incoming: list[dict[str, object]] = [
        {"type": "http.request", "body": b"a" * 16, "more_body": True},
        {"type": "http.request", "body": b"b" * 17, "more_body": True},
        {"type": "http.disconnect"},
    ]

    async def receive() -> dict[str, object]:
        # 16 + 17 = 33 bytes crosses the 32-byte limit on the second message.
        if len(incoming) > 1:
            return incoming.pop(0)
        return incoming[0]  # keep serving http.disconnect; never block

    sent: list[dict[str, object]] = []

    async def send(message: dict[str, object]) -> None:
        sent.append(message)

    async def app(scope, receive_, send_):
        await send_({"type": "http.response.start", "status": 200, "headers": []})
        await receive_()
        await receive_()

    middleware = RequestContextMiddleware(app, max_request_bytes=32)
    scope: dict[str, object] = {
        "type": "http",
        "method": "POST",
        "path": "/predict",
        "headers": [],
        "query_string": b"",
    }

    with pytest.raises(RequestTooLarge):
        asyncio.run(middleware(scope, receive, send))
    assert sent[0]["type"] == "http.response.start"  # response really had started


def test_valid_request_id_is_preserved_and_context_headers_added() -> None:
    app = _limited_app(max_request_bytes=64)

    response = TestClient(app).post(
        "/predict", json={"text": "ok"}, headers={"X-Request-ID": "client.17-alpha_3"}
    )
    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "client.17-alpha_3"
    assert re.fullmatch(r"total;dur=\d+\.\d+", response.headers["Server-Timing"])


def test_invalid_request_ids_are_replaced_with_generated_ids() -> None:
    app = _limited_app(max_request_bytes=64)
    client = TestClient(app)

    invalid_values: list[str | bytes] = [
        "spaces in id",
        "bad/id",
        # Non-ASCII bytes are legal on the wire but must never be echoed.
        "中文".encode(),
        "a" * 129,
    ]
    for supplied in invalid_values:
        response = client.post(
            "/predict", json={"text": "ok"}, headers={"X-Request-ID": supplied}
        )
        assert response.status_code == 200
        echoed = response.headers["X-Request-ID"]
        raw = supplied.encode() if isinstance(supplied, str) else supplied
        assert echoed.encode("latin-1") != raw
        assert re.fullmatch(r"req_[0-9a-f]{32}", echoed)


def test_missing_request_id_is_generated() -> None:
    app = _limited_app(max_request_bytes=64)

    response = TestClient(app).post("/predict", json={"text": "ok"})
    assert response.status_code == 200
    assert re.fullmatch(r"req_[0-9a-f]{32}", response.headers["X-Request-ID"])


def _post_raw_text(app: FastAPI, text: str) -> httpx.Response:
    body = ('{"text": "' + text + '"}').encode("utf-8")
    return TestClient(app).post(
        "/predict", content=body, headers={"content-type": "application/json"}
    )


def test_multibyte_body_is_counted_in_bytes_not_characters() -> None:
    # The limit is a byte budget: each Vietnamese multibyte character costs 3
    # bytes, so a body whose character count is far below the limit can still
    # exceed it and must be rejected. Accepted text arrives untruncated.
    app = _limited_app(max_request_bytes=96)
    accepted = "ạ" * 28  # 84 text bytes + 12 envelope bytes = exactly 96
    rejected = "ạ" * 29  # 99 bytes, but only 29 characters

    under = _post_raw_text(app, accepted)
    assert under.status_code == 200
    assert under.json()["text"] == accepted, "accepted multibyte text must be untruncated"

    over = _post_raw_text(app, rejected)
    assert over.status_code == 413
    assert over.headers["content-type"].startswith("application/problem+json")
    body = over.json()
    assert body["code"] == "CONTENT_TOO_LARGE"
    assert len(rejected) < 96, "character count must not be the limiting dimension"
