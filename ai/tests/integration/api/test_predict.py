import asyncio
import re

import httpx
from fastapi.testclient import TestClient

from src.api.dependencies import startup_load, startup_normalization
from src.api.server import create_app
from src.core.settings import Settings
from src.domain.entities import ArtifactManifest, Prediction
from tests.integration.api.fakes import FakeModel, successful_loader


def test_predict_returns_three_scores_and_model_metadata() -> None:
    app = create_app(Settings(), successful_loader)
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Xe chạy rất ổn"})
    assert response.status_code == 200
    body = response.json()
    assert set(body["scores"]) == {"negative", "neutral", "positive"}
    assert body["model"] == {"backend": "linear", "version": "1.0.0", "degraded": False}
    assert body["request_id"].startswith("req_")


def test_predict_before_lifespan_returns_503_model_not_ready() -> None:
    app = create_app(Settings(), successful_loader)
    client = TestClient(app)
    response = client.post("/predict", json={"text": "Xe ổn"})
    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "MODEL_NOT_READY"


class TokenRecordingModel(FakeModel):
    """Fake with a one-token artifact limit that records every text it scores."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_texts: list[str] = []

    def predict(self, text: str) -> Prediction:
        self.seen_texts.append(text)
        return super().predict(text)

    def metadata(self) -> ArtifactManifest:
        return super().metadata().model_copy(update={"max_input_tokens": 1})


def test_text_over_token_limit_is_rejected_without_truncation() -> None:
    model = TokenRecordingModel()

    def tiny_limit_loader(backend: str) -> TokenRecordingModel:
        return model

    app = create_app(Settings(), tiny_limit_loader)
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Xe chạy rất ổn"})
    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/problem+json")
    assert response.json()["code"] == "TEXT_EXCEEDS_MODEL_LIMIT"
    # The model is never invoked: over-limit text is rejected, never truncated.
    assert model.seen_texts == []


def test_body_byte_boundary_at_limit_minus_one_limit_and_over() -> None:
    """The 64 KiB policy is a byte budget measured at limit-1/limit/limit+1."""
    app = create_app(Settings(), successful_loader)
    with TestClient(app) as client:
        for size, expected_status in ((65_535, 200), (65_536, 200), (65_537, 413)):
            body = b'{"text": "' + b"A" * (size - 12) + b'"}'
            assert len(body) == size
            response = client.post(
                "/predict", content=body, headers={"content-type": "application/json"}
            )
            assert response.status_code == expected_status, (size, response.status_code)
            if expected_status == 200:
                assert response.json()["label"] == "positive"
            else:
                assert response.headers["content-type"].startswith(
                    "application/problem+json"
                )
                assert response.json()["code"] == "CONTENT_TOO_LARGE"


def test_concurrent_requests_keep_request_ids_isolated() -> None:
    """Concurrent calls never observe, reuse, or overwrite another request's ID."""
    app = create_app(Settings(), successful_loader)
    state = app.state.runtime
    startup_load(state, Settings(), successful_loader)
    startup_normalization(state, Settings())
    supplied_ids = [f"client.conc-{index:02d}" for index in range(6)]

    async def gather_responses() -> list[httpx.Response]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver"
        ) as client:
            requests = []
            for index in range(12):
                headers = (
                    {"X-Request-ID": supplied_ids[index]} if index < len(supplied_ids) else {}
                )
                requests.append(
                    client.post("/predict", json={"text": f"xe chạy ổn {index}"}, headers=headers)
                )
            return list(await asyncio.gather(*requests))

    responses = asyncio.run(gather_responses())
    assert len(responses) == 12
    returned_ids = [response.json()["request_id"] for response in responses]
    assert len(set(returned_ids)) == 12, "request IDs must never collide across requests"
    for index, response in enumerate(responses[:6]):
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == supplied_ids[index]
        assert returned_ids[index] == supplied_ids[index]
    for response in responses[6:]:
        assert response.status_code == 200
        generated = response.headers["X-Request-ID"]
        assert re.fullmatch(r"req_[0-9a-f]{32}", generated)
        assert response.json()["request_id"] == generated


def test_scores_are_serialized_in_canonical_label_order() -> None:
    """Wire order of ``scores`` is always negative/neutral/positive, whatever the
    model's internal mapping order is."""

    class ShuffledScoreModel(FakeModel):
        def predict(self, text: str) -> Prediction:
            prediction = super().predict(text)
            return prediction.model_copy(
                update={
                    "scores": {
                        "positive": prediction.scores["positive"],
                        "neutral": prediction.scores["neutral"],
                        "negative": prediction.scores["negative"],
                    }
                }
            )

    app = create_app(Settings(), lambda backend: ShuffledScoreModel())
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "Xe chạy rất ổn"})
    assert response.status_code == 200
    body = response.json()
    assert list(body["scores"]) == ["negative", "neutral", "positive"]
    # Raw-wire order, measured inside the scores object only (the label field
    # itself can legitimately contain the word "positive").
    scores_wire = response.text[response.text.index('"scores"') :]
    assert (
        scores_wire.index('"negative"')
        < scores_wire.index('"neutral"')
        < scores_wire.index('"positive"')
    )


def test_predict_rejects_missing_and_wrong_content_type() -> None:
    """A JSON body sent without ``application/json`` is a validation failure,
    never a silent success; a ``charset`` parameter is still accepted."""
    app = create_app(Settings(), successful_loader)
    json_body = '{"text": "Xe ổn"}'.encode()
    with TestClient(app) as client:
        for headers in (
            {},
            {"content-type": "text/plain"},
            {"content-type": "application/x-www-form-urlencoded"},
        ):
            response = client.post("/predict", content=json_body, headers=headers)
            assert response.status_code == 422, headers
            assert response.headers["content-type"].startswith("application/problem+json")
            body = response.json()
            assert body["status"] == 422
            assert body["code"] == "VALIDATION_ERROR"
            assert body["request_id"], headers

        accepted = client.post(
            "/predict",
            content=json_body,
            headers={"content-type": "application/json; charset=utf-8"},
        )
        assert accepted.status_code == 200
