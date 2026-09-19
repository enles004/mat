import json
import logging

import pytest
from fastapi.testclient import TestClient

from src.api.server import create_app
from src.core.logging import (
    AuditFormatter,
    MonitorFormatter,
    stop_audit_listener,
)
from src.core.settings import Settings
from tests.integration.api.fakes import successful_loader

_CHANNEL_NAMES = ("src", "mat.audit", "mat.monitor")


class _Capture(logging.Handler):
    def __init__(self, formatter: logging.Formatter) -> None:
        super().__init__()
        self.setFormatter(formatter)
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))

    def payloads(self) -> list[dict[str, object]]:
        return [json.loads(line) for line in self.lines]


def _reset_channels() -> None:
    stop_audit_listener()
    for name in _CHANNEL_NAMES:
        channel = logging.getLogger(name)
        for handler in list(channel.handlers):
            channel.removeHandler(handler)
        channel.setLevel(logging.NOTSET)
        channel.propagate = True


@pytest.fixture
def audit_lines():
    _reset_channels()
    capture = _Capture(AuditFormatter("mat", "dev"))
    logging.getLogger("mat.audit").addHandler(capture)
    yield capture.payloads
    _reset_channels()


@pytest.fixture
def monitor_lines():
    capture = _Capture(MonitorFormatter("mat", "dev"))
    logging.getLogger("mat.monitor").addHandler(capture)
    yield capture.payloads
    _reset_channels()


def test_predict_writes_request_audit_response_audit_and_monitor_line(
    audit_lines, monitor_lines
) -> None:
    app = create_app(Settings(), successful_loader)
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "Xe này chạy rất tốt"},
            headers={"X-Request-ID": "req_obs1"},
        )
    assert response.status_code == 200

    assert len(audit_lines()) == 2
    request_line, response_line = audit_lines()
    assert request_line["type"] == "AUDIT"
    assert request_line["kind"] == "CLIENT_REQUEST"
    assert request_line["method"] == "POST"
    assert request_line["path"] == "/predict"
    assert request_line["request_id"] == "req_obs1"
    assert "Xe này chạy rất tốt" in str(request_line["body"])

    assert response_line["kind"] == "CLIENT_RESPONSE"
    assert response_line["status_code"] == 200
    assert response_line["request_id"] == "req_obs1"
    response_body = json.loads(str(response_line["body"]))
    assert response_body["label"] == "positive"
    assert response_body["confidence"] == pytest.approx(0.70)

    assert len(monitor_lines()) == 1
    monitor = monitor_lines()[0]
    assert monitor["type"] == "MON"
    assert monitor["path"] == "/predict"
    assert monitor["status_code"] == 200
    assert float(monitor["total_ms"]) > 0.0


def test_get_request_audit_has_no_body_field(audit_lines, monitor_lines) -> None:
    app = create_app(Settings(), successful_loader)
    with TestClient(app) as client:
        assert client.get("/health-check").status_code == 200

    assert len(audit_lines()) == 2
    request_line, response_line = audit_lines()
    assert request_line["kind"] == "CLIENT_REQUEST"
    assert "body" not in request_line
    assert response_line["kind"] == "CLIENT_RESPONSE"
    assert response_line["status_code"] == 200


def test_oversized_request_is_audited_with_413_response(audit_lines, monitor_lines) -> None:
    app = create_app(Settings(max_request_bytes=256), successful_loader)
    with TestClient(app) as client:
        response = client.post("/predict", json={"text": "x" * 1_000})
    assert response.status_code == 413

    response_lines = [line for line in audit_lines() if line["kind"] == "CLIENT_RESPONSE"]
    assert len(response_lines) == 1
    assert response_lines[0]["status_code"] == 413

    assert len(monitor_lines()) == 1
    assert monitor_lines()[0]["status_code"] == 413


def test_create_app_repeated_setup_keeps_single_channel_handler() -> None:
    _reset_channels()
    create_app(Settings(), successful_loader)
    create_app(Settings(), successful_loader)
    assert len(logging.getLogger("mat.audit").handlers) == 1
    assert len(logging.getLogger("src").handlers) == 1
    assert len(logging.getLogger("mat.monitor").handlers) == 1
    _reset_channels()
