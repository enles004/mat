import time

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from src.api.middlewares.request_capture import RequestCapture


class ObservabilityMiddleware:
    """Emit wire audit and latency logs around each HTTP request.

    This is the outermost middleware: the request body is drained from the
    server receive up front and replayed to the app, which lets the
    CLIENT_REQUEST audit line precede the app and therefore the
    CLIENT_RESPONSE line; the body-limit wrapper still sees every byte
    through the replay. Bodies are captured up to ``max_capture_bytes``; a
    larger body is logged without its content and the capture limit is stated
    in the log message instead of a silent truncation. It must sit outside
    RequestContextMiddleware so request-id rewrites (413 responses included)
    pass through its capture.
    """

    def __init__(self, app: ASGIApp, max_capture_bytes: int) -> None:
        self.app = app
        self.max_capture_bytes = max_capture_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        state = RequestCapture(max_capture_bytes=self.max_capture_bytes)
        state.bind_scope(scope)
        replay: list[Message] = []

        async def replay_receive() -> Message:
            if replay:
                return replay.pop(0)
            return await receive()

        async def send_with_capture(message: Message) -> None:
            state.observe_send(message)
            await send(message)

        try:
            # Drain the body from the server receive up front so
            # CLIENT_REQUEST precedes the app and therefore the
            # CLIENT_RESPONSE line; the app re-reads it from the replay.
            body_done = False
            while not body_done:
                message = await receive()
                body_done = state.observe_request_message(message)
                replay.append(message)
            state.emit_request()
            await self.app(scope, replay_receive, send_with_capture)
        finally:
            state.finish((time.perf_counter() - started) * 1_000)
