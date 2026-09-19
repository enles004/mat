import json
import logging
import queue
import time
import traceback
from contextvars import ContextVar
from logging.handlers import QueueHandler, QueueListener
from typing import Any, TextIO

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"

request_id_ctx: ContextVar[str | None] = ContextVar("request_id", default=None)
model_latency_ms_ctx: ContextVar[float | None] = ContextVar(
    "model_latency_ms", default=None
)

_UNKNOWN_REQUEST_ID = "unknown"
_APP_LOGGER_NAME = "src"
_AUDIT_LOGGER_NAME = "mat.audit"
_MONITOR_LOGGER_NAME = "mat.monitor"


class ApplicationFormatter(logging.Formatter):
    """One-line JSON application log (type APP) with an optional stacktrace."""

    def __init__(self, service_name: str, environment: str) -> None:
        super().__init__()
        self._service_name = service_name
        self._environment = environment

    def format(self, record: logging.LogRecord) -> str:
        stacktrace: str | None = None
        if record.exc_info:
            stacktrace = "".join(traceback.format_exception(*record.exc_info))
        payload = {
            "type": "APP",
            "ts": _epoch_ms(),
            "level": record.levelname,
            "service": self._service_name,
            "env": self._environment,
            "request_id": request_id_ctx.get() or _UNKNOWN_REQUEST_ID,
            "name": record.name,
            "msg": record.getMessage(),
            "stacktrace": stacktrace,
        }
        return _dump(payload)


class AuditFormatter(logging.Formatter):
    """One-line JSON wire audit log (type AUDIT) for request/response pairs."""

    def __init__(self, service_name: str, environment: str) -> None:
        super().__init__()
        self._service_name = service_name
        self._environment = environment

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "type": "AUDIT",
            "ts": _epoch_ms(),
            "level": record.levelname,
            "service": self._service_name,
            "env": self._environment,
            "request_id": _get(record, "request_id")
            or request_id_ctx.get()
            or _UNKNOWN_REQUEST_ID,
            "kind": _get(record, "kind"),
            "method": _get(record, "method"),
            "path": _get(record, "path"),
            "client_ip": _get(record, "client_ip"),
            "status_code": _get(record, "status_code"),
            "body": _get(record, "body"),
            "msg": record.getMessage(),
        }
        return _dump(payload)


class MonitorFormatter(logging.Formatter):
    """One-line JSON latency log (type MON) per completed request."""

    def __init__(self, service_name: str, environment: str) -> None:
        super().__init__()
        self._service_name = service_name
        self._environment = environment

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "type": "MON",
            "ts": _epoch_ms(),
            "level": record.levelname,
            "service": self._service_name,
            "env": self._environment,
            "request_id": _get(record, "request_id")
            or request_id_ctx.get()
            or _UNKNOWN_REQUEST_ID,
            "method": _get(record, "method"),
            "path": _get(record, "path"),
            "status_code": _get(record, "status_code"),
            "total_ms": _get(record, "total_ms"),
            "model_ms": _get(record, "model_ms"),
            "msg": record.getMessage(),
        }
        return _dump(payload)


def _epoch_ms() -> int:
    return int(time.time() * 1000)


def _get(record: logging.LogRecord, key: str) -> Any:
    return getattr(record, key, None)


def _dump(payload: dict[str, Any]) -> str:
    """Serialize the log record to one JSON line, dropping None fields."""
    present = {key: value for key, value in payload.items() if value is not None}
    return json.dumps(present, ensure_ascii=False, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once for console entry points."""
    logging.basicConfig(level=level, format=_LOG_FORMAT)


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger for MAT modules."""
    return logging.getLogger(name)


def audit_logger() -> logging.Logger:
    """Return the wire-audit channel logger."""
    return logging.getLogger(_AUDIT_LOGGER_NAME)


def monitor_logger() -> logging.Logger:
    """Return the latency-monitor channel logger."""
    return logging.getLogger(_MONITOR_LOGGER_NAME)


def _stream_handler(formatter: logging.Formatter) -> logging.StreamHandler[TextIO]:
    handler: logging.StreamHandler[TextIO] = logging.StreamHandler()
    handler.setFormatter(formatter)
    return handler


class _AuditChannel:
    """Own the background queue that keeps audit writes off the request hot path."""

    def __init__(self) -> None:
        self._listener: QueueListener | None = None
        self._queue: queue.Queue[logging.LogRecord] | None = None

    def attach(self, logger: logging.Logger, formatter: logging.Formatter) -> None:
        self._queue = queue.Queue()
        self._listener = QueueListener(self._queue, _stream_handler(formatter))
        self._listener.start()
        logger.addHandler(QueueHandler(self._queue))

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
            self._queue = None


_audit_channel = _AuditChannel()


def setup_logging(service_name: str, environment: str, level: int = logging.INFO) -> None:
    """Attach the APP/AUDIT/MON JSON channels once; later calls are no-ops.

    Channel state is checked on the loggers' own handlers, never
    ``hasHandlers()``: an ancestor handler (pytest's root capture, uvicorn's
    root config) must not make an unconfigured channel look configured.
    """
    app_logger = logging.getLogger(_APP_LOGGER_NAME)
    audit = logging.getLogger(_AUDIT_LOGGER_NAME)
    monitor = logging.getLogger(_MONITOR_LOGGER_NAME)

    for channel in (app_logger, audit, monitor):
        channel.setLevel(level)
        channel.propagate = False

    if not app_logger.handlers:
        app_logger.addHandler(
            _stream_handler(ApplicationFormatter(service_name, environment))
        )
    if not audit.handlers:
        _audit_channel.attach(audit, AuditFormatter(service_name, environment))
    if not monitor.handlers:
        monitor.addHandler(_stream_handler(MonitorFormatter(service_name, environment)))


def stop_audit_listener() -> None:
    """Stop the background audit listener at shutdown; safe to call twice."""
    _audit_channel.stop()
