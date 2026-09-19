import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import RuntimeState, get_prediction_service
from src.api.server import create_app
from src.api.services.prediction_service import PredictionService
from src.core.settings import Settings
from src.domain.exceptions import ArtifactUnavailable
from tests.integration.api.fakes import FakeModel, successful_loader


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Settings(), successful_loader))


def test_health_check_is_503_before_model_and_200_after_lifespan() -> None:
    app = create_app(Settings(), successful_loader)
    client = TestClient(app)
    assert client.get("/livez").status_code == 200
    assert client.get("/health-check").status_code == 503
    with client:
        assert client.get("/health-check").status_code == 200


def test_readyz_is_removed_and_health_check_owns_readiness(client: TestClient) -> None:
    response = client.get("/readyz")
    assert response.status_code == 404
    # The removed route answers the same RFC 9457 contract as every failure.
    assert response.headers["content-type"].startswith("application/problem+json")
    body = response.json()
    assert body["status"] == 404
    assert body["code"] == "NOT_FOUND"
    assert body["title"] == "Not Found"
    assert body["instance"] == "/readyz"
    with client:
        assert client.get("/health-check").status_code == 200


def test_health_check_body_reports_backend_degraded_and_version() -> None:
    app = create_app(Settings(), successful_loader)
    with TestClient(app) as client:
        response = client.get("/health-check")
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "linear"
    assert body["degraded"] is False
    assert body["version"] == "1.0.0"


def test_readiness_stays_503_with_generic_detail_after_lifespan_load_failure() -> None:
    """A model-load failure inside the lifespan keeps the service unready; the
    sanitized reason stays in RuntimeState for operators and never reaches the
    readiness response."""

    def failing_loader(backend: str) -> FakeModel:
        raise ArtifactUnavailable("corrupt baseline artifact: checksum drift xyz")

    app = create_app(Settings(), failing_loader)
    with TestClient(app) as client:
        readiness = client.get("/health-check")
        liveness = client.get("/livez")
        prediction = client.post("/predict", json={"text": "Xe ổn"})

    assert liveness.status_code == 200
    assert readiness.status_code == 503
    assert readiness.headers["content-type"].startswith("application/problem+json")
    body = readiness.json()
    assert body["code"] == "MODEL_NOT_READY"
    assert "corrupt baseline artifact" not in readiness.text
    assert "xyz" not in readiness.text
    assert prediction.status_code == 503

    state = app.state.runtime
    assert state.ready is False
    assert state.startup_error is not None, "operators need the retained startup reason"


def test_two_independently_created_apps_have_isolated_readiness() -> None:
    """Two apps built from the same factory share no readiness: activating one
    lifespan never turns the other app ready."""
    ready_app = create_app(Settings(), successful_loader)
    unready_app = create_app(Settings(), successful_loader)

    with TestClient(ready_app) as ready_client:
        assert ready_client.get("/health-check").status_code == 200
        unready_client = TestClient(unready_app)
        assert unready_client.get("/health-check").status_code == 503
        assert ready_client.get("/health-check").status_code == 200


def test_prediction_service_dependency_override_controls_the_endpoint() -> None:
    """The endpoint resolves PredictionService through request DI: overriding
    the provider swaps the model the endpoint scores against, without any
    lifespan involvement."""
    app = create_app(Settings(), successful_loader)
    model = FakeModel(fail_predict=True)
    state = RuntimeState(model=model, ready=True, degraded=False, backend="linear")
    app.dependency_overrides[get_prediction_service] = lambda: PredictionService(state)

    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Xe ổn"})

    assert response.status_code == 500
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "INFERENCE_FAILED"


def test_unsupported_method_returns_405_problem_json_on_every_route() -> None:
    """Method mismatches answer exact RFC 9457 problem responses with the
    allowed-methods hint, on every public route."""
    app = create_app(Settings(), successful_loader)
    cases = (
        ("GET", "/predict", {"POST"}),
        ("PUT", "/predict", {"POST"}),
        ("POST", "/health-check", {"GET"}),
        ("POST", "/livez", {"GET"}),
    )
    with TestClient(app) as client:
        for method, path, allowed in cases:
            response = client.request(method, path)
            assert response.status_code == 405, (method, path)
            assert response.headers["content-type"].startswith("application/problem+json"), (
                method,
                path,
            )
            body = response.json()
            assert body["status"] == 405
            assert body["code"] == "METHOD_NOT_ALLOWED"
            assert body["instance"] == path
            assert body["request_id"]
            assert allowed <= set(response.headers["allow"].split(", ")), (method, path)
