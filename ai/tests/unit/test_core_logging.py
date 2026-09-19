import json
import logging

from src.core.logging import (
    ApplicationFormatter,
    AuditFormatter,
    MonitorFormatter,
    request_id_ctx,
    setup_logging,
    stop_audit_listener,
)


def _record(
    msg: str = "hello",
    *,
    extra: dict[str, object] | None = None,
    exc_info: BaseException | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name="src.api.errors",
        level=logging.INFO,
        pathname="path.py",
        lineno=1,
        msg=msg,
        args=(),
        exc_info=exc_info,
    )
    for key, value in (extra or {}).items():
        setattr(record, key, value)
    return record


def test_application_formatter_renders_one_json_app_line() -> None:
    line = ApplicationFormatter("mat", "dev").format(_record("service started"))
    payload = json.loads(line)
    assert payload["type"] == "APP"
    assert payload["service"] == "mat"
    assert payload["env"] == "dev"
    assert payload["msg"] == "service started"
    assert payload["name"] == "src.api.errors"
    assert payload["request_id"] == "unknown"
    assert isinstance(payload["ts"], int)
    # None fields are dropped, not null-filled.
    assert "stacktrace" not in payload


def test_application_formatter_renders_stacktrace_and_request_context() -> None:
    token = request_id_ctx.set("req_ctx1")
    error = ValueError("bad input")
    try:
        line = ApplicationFormatter("mat", "prod").format(
            _record("boom", exc_info=(type(error), error, error.__traceback__))
        )
    finally:
        request_id_ctx.reset(token)
    payload = json.loads(line)
    assert payload["request_id"] == "req_ctx1"
    assert "ValueError: bad input" in payload["stacktrace"]


def test_audit_formatter_reads_request_id_from_record_for_queue_safety() -> None:
    """Audit records travel through a QueueListener worker thread where the
    request ContextVar is invisible; the id must ride on the record itself."""
    token = request_id_ctx.set("req_from_ctx")
    try:
        line = AuditFormatter("mat", "dev").format(
            _record(
                "CLIENT_REQUEST",
                extra={"request_id": "req_from_record", "kind": "CLIENT_REQUEST"},
            )
        )
    finally:
        request_id_ctx.reset(token)
    assert json.loads(line)["request_id"] == "req_from_record"


def test_audit_formatter_falls_back_to_context_then_unknown() -> None:
    token = request_id_ctx.set("req_ctx2")
    try:
        from_context = json.loads(AuditFormatter("mat", "dev").format(_record()))
    finally:
        request_id_ctx.reset(token)
    assert from_context["request_id"] == "req_ctx2"
    assert json.loads(AuditFormatter("mat", "dev").format(_record()))["request_id"] == "unknown"


def test_audit_formatter_renders_wire_fields_and_drops_missing_ones() -> None:
    line = AuditFormatter("mat", "dev").format(
        _record(
            "CLIENT_RESPONSE",
            extra={
                "request_id": "req_w1",
                "kind": "CLIENT_RESPONSE",
                "status_code": 200,
                "body": '{"label": "positive"}',
            },
        )
    )
    payload = json.loads(line)
    assert payload["type"] == "AUDIT"
    assert payload["kind"] == "CLIENT_RESPONSE"
    assert payload["status_code"] == 200
    assert payload["body"] == '{"label": "positive"}'
    assert "method" not in payload
    assert "client_ip" not in payload


def test_audit_formatter_keeps_vietnamese_text_unescaped() -> None:
    line = AuditFormatter("mat", "dev").format(
        _record(
            "CLIENT_REQUEST",
            extra={"kind": "CLIENT_REQUEST", "body": "Xe này chạy rất tốt"},
        )
    )
    assert "Xe này chạy rất tốt" in line
    assert "\\u" not in line


def test_monitor_formatter_renders_latency_and_drops_absent_model_ms() -> None:
    formatted = MonitorFormatter("mat", "dev")
    with_model = json.loads(
        formatted.format(
            _record(
                "request",
                extra={
                    "request_id": "req_m1",
                    "method": "POST",
                    "path": "/predict",
                    "status_code": 200,
                    "total_ms": 1.5,
                    "model_ms": 0.25,
                },
            )
        )
    )
    assert with_model["type"] == "MON"
    assert with_model["total_ms"] == 1.5
    assert with_model["model_ms"] == 0.25

    without_model = json.loads(
        formatted.format(
            _record(
                "request",
                extra={"method": "GET", "path": "/livez", "status_code": 200, "total_ms": 0.2},
            )
        )
    )
    assert "model_ms" not in without_model


def test_setup_logging_attaches_each_channel_once(reset_log_channels) -> None:
    reset_log_channels()
    setup_logging("mat", "dev")
    setup_logging("mat", "dev")

    app_logger = logging.getLogger("src")
    audit_logger = logging.getLogger("mat.audit")
    monitor_logger = logging.getLogger("mat.monitor")
    assert len(app_logger.handlers) == 1
    assert len(audit_logger.handlers) == 1
    assert len(monitor_logger.handlers) == 1
    assert isinstance(app_logger.handlers[0].formatter, ApplicationFormatter)
    assert isinstance(audit_logger.handlers[0], logging.handlers.QueueHandler)
    assert isinstance(monitor_logger.handlers[0].formatter, MonitorFormatter)
    for channel in (app_logger, audit_logger, monitor_logger):
        assert channel.propagate is False

    stop_audit_listener()
