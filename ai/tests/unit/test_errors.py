import json
import re

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from src.api.error_mapping import classify
from src.api.error_types import ClientError, ServerError
from src.api.errors import install_error_handlers
from src.api.middlewares.request_context import RequestContextMiddleware
from src.api.v1.schemas import PredictRequest
from src.libs.result import Error


def test_malformed_json_body_returns_400_problem_json() -> None:
    app = FastAPI()
    install_error_handlers(app)

    @app.post("/echo")
    async def echo(payload: dict[str, str]) -> dict[str, str]:
        return payload

    response = TestClient(app).post(
        "/echo", content=b"{not json", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 400
    assert body["code"] == "MALFORMED_JSON"
    assert body["instance"] == "/echo"
    assert "Traceback" not in response.text


def test_direct_json_read_of_malformed_body_returns_400_problem_json() -> None:
    app = FastAPI()
    install_error_handlers(app)

    @app.post("/raw")
    async def raw(request: Request) -> dict[str, str]:
        payload = json.loads(await request.body())
        return {"received": str(payload)}

    response = TestClient(app).post(
        "/raw", content=b"[truncated", headers={"content-type": "application/json"}
    )
    assert response.status_code == 400
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "MALFORMED_JSON"


def test_request_validation_failure_returns_422_problem_json() -> None:
    app = FastAPI()
    install_error_handlers(app)

    @app.post("/predict")
    async def predict(payload: PredictRequest) -> dict[str, str]:
        return {"text": payload.text}

    response = TestClient(app).post("/predict", json={"text": "   ", "unexpected": 1})
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 422
    assert body["code"] == "VALIDATION_ERROR"
    assert body["instance"] == "/predict"
    assert body["errors"], "validation failures must be enumerated in the problem body"
    assert "Traceback" not in response.text


def test_uncaught_exception_returns_sanitized_500_problem_json() -> None:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("secret internal detail")

    client = TestClient(app, raise_server_exceptions=False)
    response = client.get("/boom")
    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 500
    assert body["code"] == "INTERNAL_ERROR"
    assert body["detail"] == "Unexpected internal error"
    assert "secret internal detail" not in response.text
    assert "Traceback" not in response.text
    assert re.fullmatch(r"https://mat\.local/problems/[\w-]+", body["type"])


def test_request_id_propagates_to_every_error_family() -> None:
    """Validation, client, server, and unknown-code errors all carry the
    caller's request ID in both the X-Request-ID header and the problem body."""
    app = FastAPI()
    install_error_handlers(app)
    app.add_middleware(RequestContextMiddleware, max_request_bytes=65_536)

    @app.get("/client")
    def client_failure() -> None:
        raise ClientError(Error(code="SOME_CODE", message="client mistake", reason="internal"))

    @app.get("/server")
    def server_failure() -> None:
        raise ServerError(Error(code="BACKEND_DIED", message="mask me", reason="internal"))

    @app.get("/unknown")
    def unknown_code_failure() -> None:
        # Unknown internal codes fail closed: the mapping layer turns them
        # into ServerErrors (500), never client-visible 4xx guesses.
        raise classify(Error(code="NEVER_DEFINED_ANYWHERE", message="anything"))

    @app.post("/validate")
    async def validate(payload: PredictRequest) -> dict[str, str]:
        return {"text": payload.text}

    client = TestClient(app, raise_server_exceptions=False)
    supplied = "client.family-01"

    families = [
        (client.get, "/client", 400),
        (client.get, "/server", 500),
        (client.get, "/unknown", 500),
        (client.post, "/validate", 422),
    ]
    for method, path, expected_status in families:
        response = method(path, headers={"X-Request-ID": supplied})
        assert response.status_code == expected_status, path
        assert response.headers["content-type"].startswith("application/problem+json"), path
        assert response.headers["X-Request-ID"] == supplied, path
        body = response.json()
        assert body["request_id"] == supplied, path
