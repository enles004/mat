import json
import logging
from http import HTTPStatus
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.error_mapping import ERROR_POLICY
from src.api.error_types import APIError
from src.api.v1.schemas import ProblemDetail

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unassigned")


def _problem_json_response(
    *,
    status: int,
    code: str,
    title: str,
    detail: str,
    request: Request,
    errors: list[dict[str, Any]] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    body = ProblemDetail(
        type=f"https://mat.local/problems/{code.lower().replace('_', '-')}",
        title=title,
        status=status,
        detail=detail,
        instance=str(request.url.path),
        code=code,
        request_id=_request_id(request),
        errors=errors if errors is not None else [],
    )
    return JSONResponse(
        status_code=status,
        content=body.model_dump(),
        media_type="application/problem+json",
        headers=headers,
    )


def _public_api_title(exc: APIError) -> str:
    """Prefer the policy title for a known code; otherwise the HTTP status phrase."""
    policy = ERROR_POLICY.get(exc.base_error.code)
    if policy is not None:
        return policy.public_title
    return _status_phrase(exc.get_status_code())


def _status_phrase(status: int) -> str:
    """The standard HTTP reason phrase, or a generic fallback for unknown codes."""
    try:
        return HTTPStatus(status).phrase
    except ValueError:
        return "Error"


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        # Routing-level failures (404 unknown route, 405 method mismatch) follow
        # the same RFC 9457 contract as every other API failure: the status and
        # any routing hint (e.g. ``Allow``) are preserved and nothing internal
        # is exposed. This replaces FastAPI's plain-JSON default handler.
        phrase = _status_phrase(exc.status_code)
        return _problem_json_response(
            status=exc.status_code,
            code=phrase.upper().replace(" ", "_").replace("-", "_"),
            title=phrase,
            detail=phrase,
            request=request,
            headers=dict(exc.headers) if exc.headers else None,
        )

    @app.exception_handler(APIError)
    async def handle_api_error(request: Request, exc: APIError) -> JSONResponse:
        # Internal diagnostics only: request id, error id, code, and reason
        # stay in server logs; the response carries the public projection.
        projection = exc.to_dict()
        logger.error(
            "API error (request_id=%s, error_id=%s, code=%s, reason=%r)",
            _request_id(request),
            exc.base_error.id,
            exc.base_error.code,
            exc.get_reason(),
        )
        return _problem_json_response(
            status=exc.get_status_code(),
            code=projection["code"],
            title=_public_api_title(exc),
            detail=projection["message"],
            request=request,
            headers=exc.get_headers() or None,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        errors = [
            {
                "loc": [str(item) for item in error.get("loc", ())],
                "msg": str(error.get("msg", "")),
                "type": str(error.get("type", "")),
            }
            for error in exc.errors()
        ]
        # FastAPI reports body JSON decode failures as RequestValidationError
        # entries of type "json_invalid". Those are malformed requests (400),
        # not field-level validation failures (422).
        if any(error["type"] == "json_invalid" for error in errors):
            return _problem_json_response(
                status=400,
                code="MALFORMED_JSON",
                title="Malformed JSON body",
                detail="Request body is not valid JSON.",
                request=request,
            )
        return _problem_json_response(
            status=422,
            code="VALIDATION_ERROR",
            title="Request validation failed",
            detail="One or more request fields failed validation.",
            request=request,
            errors=errors,
        )

    @app.exception_handler(json.JSONDecodeError)
    async def handle_malformed_json(request: Request, exc: json.JSONDecodeError) -> JSONResponse:
        return _problem_json_response(
            status=400,
            code="MALFORMED_JSON",
            title="Malformed JSON body",
            detail="Request body is not valid JSON.",
            request=request,
        )

    @app.exception_handler(Exception)
    async def handle_uncaught_exception(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        logger.exception("Unhandled API error (request_id=%s)", request_id)
        return _problem_json_response(
            status=500,
            code="INTERNAL_ERROR",
            title="Internal Server Error",
            detail="Unexpected internal error",
            request=request,
        )
