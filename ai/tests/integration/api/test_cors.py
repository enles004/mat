"""Browser-demo CORS contract: the demo UI (frontend :3000) calls this API
cross-origin (different port), so responses must carry the allow-list CORS
headers or the browser blocks every call with a "Failed to fetch" error."""

from fastapi.testclient import TestClient

from src.api.server import create_app
from src.core.settings import Settings
from tests.integration.api.fakes import successful_loader

UI_ORIGIN = "http://localhost:3000"


def _client() -> TestClient:
    return TestClient(create_app(Settings(), successful_loader))


def test_predict_preflight_from_ui_origin_is_allowed() -> None:
    with _client() as client:
        response = client.options(
            "/predict",
            headers={
                "Origin": UI_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == UI_ORIGIN
    assert "POST" in response.headers["access-control-allow-methods"]


def test_predict_response_carries_allow_origin_for_ui_origin() -> None:
    with _client() as client:
        response = client.post(
            "/predict",
            json={"text": "Xe chạy rất ổn"},
            headers={"Origin": UI_ORIGIN},
        )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == UI_ORIGIN


def test_origins_outside_the_allow_list_get_no_cors_headers() -> None:
    with _client() as client:
        response = client.get("/health-check", headers={"Origin": "http://evil.example"})
    assert response.status_code == 200
    assert "access-control-allow-origin" not in response.headers
