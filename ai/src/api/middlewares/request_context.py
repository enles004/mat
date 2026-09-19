import re
import time
from uuid import uuid4

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.api.v1.schemas import ProblemDetail
from src.core.logging import request_id_ctx


class RequestTooLarge(Exception):
    """Raised by the receive wrapper once a streamed body crosses the limit."""


_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,128}")


def peek_request_id(scope: Scope) -> str | None:
    """Return the client-supplied request id if valid; None otherwise.

    Outer middleware (observability) reads this before this middleware runs;
    this module stays the owner of generating and binding the final id.
    """
    headers = Headers(raw=scope["headers"])
    supplied = headers.get("x-request-id", "")
    return supplied if _REQUEST_ID_PATTERN.fullmatch(supplied) else None


def _too_large_json_response(request_id: str, path: str) -> JSONResponse:
    body = ProblemDetail(
        type="https://mat.local/problems/content-too-large",
        title="Content Too Large",
        status=413,
        detail="Request body exceeds the configured maximum size.",
        instance=path,
        code="CONTENT_TOO_LARGE",
        request_id=request_id,
    )
    return JSONResponse(
        status_code=413,
        content=body.model_dump(),
        media_type="application/problem+json",
    )


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, max_request_bytes: int) -> None:
        self.app = app
        self.max_request_bytes = max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        supplied = peek_request_id(scope)
        request_id = (
            supplied
            if supplied is not None
            else f"req_{uuid4().hex}"
        )
        scope.setdefault("state", {})["request_id"] = request_id
        request_id_ctx.set(request_id)

        # Once the body limit trips, the request must end in a 413 problem
        # response no matter how the inner app reacts: FastAPI's routing turns
        # arbitrary body-read exceptions into a generic HTTPException(400),
        # so any response produced after the trip is rewritten into the 413
        # problem response; a raw RequestTooLarge is also caught below.
        limit_exceeded = False
        replacement: JSONResponse | None = None
        rewrite_started = False
        replacement_sent = False
        response_started = False

        async def send_with_context(message: Message) -> None:
            nonlocal rewrite_started, replacement_sent, response_started
            if message["type"] == "http.response.start":
                response_started = True
                if limit_exceeded and replacement is not None:
                    rewrite_started = True
                    message["status"] = replacement.status_code
                    message["headers"] = list(replacement.raw_headers)
                mutable = MutableHeaders(scope=message)
                mutable["X-Request-ID"] = request_id
                elapsed_ms = (time.perf_counter() - started) * 1_000
                mutable["Server-Timing"] = f"total;dur={elapsed_ms:.3f}"
            elif message["type"] == "http.response.body" and rewrite_started:
                if not replacement_sent:
                    assert replacement is not None
                    message["body"] = replacement.body
                    replacement_sent = True
                else:
                    message["body"] = b""
                message["more_body"] = False
            await send(message)

        content_length = Headers(raw=scope["headers"]).get("content-length")
        if content_length is not None and int(content_length) > self.max_request_bytes:
            await request_too_large_response(scope, receive, send_with_context, request_id)
            return

        received = 0

        async def limited_receive() -> Message:
            nonlocal received, limit_exceeded, replacement
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_request_bytes:
                    limit_exceeded = True
                    replacement = _too_large_json_response(request_id, str(scope["path"]))
                    raise RequestTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send_with_context)
        except RequestTooLarge:
            if not response_started:
                await request_too_large_response(
                    scope, receive, send_with_context, request_id
                )
            else:
                # The response already started: a second http.response.start
                # would violate the ASGI protocol and a trailing empty body
                # would fake a truncated 200. Re-raise so the server aborts
                # the connection instead of leaving the client hanging.
                raise


async def request_too_large_response(
    scope: Scope,
    receive: Receive,
    send: Send,
    request_id: str,
) -> None:
    response = _too_large_json_response(request_id, str(scope["path"]))
    await response(scope, receive, send)
