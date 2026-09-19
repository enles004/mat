from typing import Any

from starlette.types import Message, Scope

from src.api.middlewares.request_context import peek_request_id
from src.core.logging import (
    audit_logger,
    model_latency_ms_ctx,
    monitor_logger,
    request_id_ctx,
)

_KIND_REQUEST = "CLIENT_REQUEST"
_KIND_RESPONSE = "CLIENT_RESPONSE"


def _body_text(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace") if raw else ""


class RequestCapture:
    """Accumulate one request's wire facts and emit its AUDIT/MON lines once.

    One instance per request; the observability middleware feeds it the raw
    receive/send messages and reports the final elapsed time.
    """

    def __init__(self, max_capture_bytes: int) -> None:
        self._max_capture_bytes = max_capture_bytes
        self._method: str | None = None
        self._path: str | None = None
        self._client_ip: str | None = None
        self._peeked_request_id: str | None = None
        self._request_body: list[bytes] = []
        self._request_captured = 0
        self._request_limit_hit = False
        self._request_emitted = False
        self._response_seen = False
        self._response_emitted = False
        self._monitor_emitted = False
        self._response_status: int | None = None
        self._response_body: list[bytes] = []
        self._response_captured = 0
        self._response_limit_hit = False

    def bind_scope(self, scope: Scope) -> None:
        client = scope.get("client")
        self._method = str(scope["method"])
        self._path = str(scope["path"])
        self._client_ip = str(client[0]) if client else None
        # The client-supplied id is known before the inner request-context
        # middleware runs and generates one when this is None.
        self._peeked_request_id = peek_request_id(scope)

    def observe_request_message(self, message: Message) -> bool:
        """Capture one received message; return True once the body is done."""
        if message["type"] == "http.request":
            chunk: bytes = message.get("body", b"")
            if self._request_captured + len(chunk) > self._max_capture_bytes:
                self._request_limit_hit = True
                # The notice must precede the request's own 413 response line.
                self.emit_request()
                return True
            self._request_body.append(chunk)
            self._request_captured += len(chunk)
            return not message.get("more_body", False)
        return True

    def observe_send(self, message: Message) -> None:
        if message["type"] == "http.response.start":
            self._response_seen = True
            self._response_status = int(message["status"])
        elif message["type"] == "http.response.body":
            self._response_seen = True
            chunk: bytes = message.get("body", b"")
            self._capture_response_chunk(chunk)
            if not message.get("more_body", False) and not self._response_emitted:
                self._response_emitted = True
                self._emit_response()

    def finish(self, elapsed_ms: float) -> None:
        """Emit whatever the wire interaction actually produced, once each."""
        self.emit_request()
        if self._response_seen and not self._monitor_emitted:
            self._monitor_emitted = True
            self._emit_monitor(elapsed_ms)

    def _capture_response_chunk(self, chunk: bytes) -> None:
        if self._response_limit_hit:
            return
        if self._response_captured + len(chunk) > self._max_capture_bytes:
            self._response_limit_hit = True
            self._response_body = []
            return
        self._response_body.append(chunk)
        self._response_captured += len(chunk)

    def emit_request(self) -> None:
        if self._request_emitted:
            return
        self._request_emitted = True
        if self._method is None:
            return
        if self._request_limit_hit:
            _audit(
                kind=_KIND_REQUEST,
                method=self._method,
                path=self._path,
                client_ip=self._client_ip,
                request_id=self._peeked_request_id,
                msg=(
                    "request body capture limit exceeded "
                    f"(limit {self._max_capture_bytes} bytes)"
                ),
            )
            return
        body = _body_text(b"".join(self._request_body))
        _audit(
            kind=_KIND_REQUEST,
            method=self._method,
            path=self._path,
            client_ip=self._client_ip,
            request_id=self._peeked_request_id,
            body=body or None,
        )

    def _emit_response(self) -> None:
        if self._response_limit_hit:
            _audit(
                kind=_KIND_RESPONSE,
                status_code=self._response_status,
                msg=(
                    "response body capture limit exceeded "
                    f"(limit {self._max_capture_bytes} bytes)"
                ),
            )
            return
        body = _body_text(b"".join(self._response_body))
        _audit(
            kind=_KIND_RESPONSE,
            status_code=self._response_status,
            body=body or None,
        )

    def _emit_monitor(self, elapsed_ms: float) -> None:
        extra: dict[str, Any] = {
            "method": self._method,
            "path": self._path,
            "status_code": self._response_status,
            "total_ms": round(elapsed_ms, 3),
        }
        model_ms = model_latency_ms_ctx.get()
        if model_ms is not None:
            extra["model_ms"] = round(model_ms, 3)
        monitor_logger().info("request", extra=extra)


def _audit(
    *,
    kind: str,
    method: str | None = None,
    path: str | None = None,
    client_ip: str | None = None,
    status_code: int | None = None,
    body: str | None = None,
    request_id: str | None = None,
    msg: str | None = None,
) -> None:
    extra: dict[str, Any] = {
        "kind": kind,
        "method": method,
        "path": path,
        "client_ip": client_ip,
        "status_code": status_code,
        "body": body,
    }
    # A peeked client-supplied id is baked in at log-call time; otherwise the
    # request-context ContextVar (set by the inner middleware) applies.
    resolved_request_id = request_id or request_id_ctx.get()
    if resolved_request_id is not None:
        extra["request_id"] = resolved_request_id
    audit_logger().info(msg if msg is not None else kind, extra=extra)
