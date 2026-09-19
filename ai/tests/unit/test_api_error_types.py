"""Unit tests for the API error envelope types in src.api.error_types."""

from http import HTTPStatus

import pytest

from src.api.error_types import ClientError, ServerError
from src.libs.result import Error


def _error(reason: str = "internal diagnostics") -> Error:
    return Error(code="SOME_CODE", message="some message", reason=reason)


def test_client_error_accepts_4xx_statuses() -> None:
    for status in (HTTPStatus.BAD_REQUEST, HTTPStatus.NOT_FOUND, HTTPStatus.UNPROCESSABLE_ENTITY):
        assert ClientError(_error(), status_code=status).get_status_code() == status


@pytest.mark.parametrize("status", [500, 200, 302])
def test_client_error_rejects_non_4xx_statuses(status: int) -> None:
    with pytest.raises(ValueError):
        ClientError(_error(), status_code=status)


def test_server_error_accepts_5xx_statuses() -> None:
    assert ServerError(_error()).get_status_code() == HTTPStatus.INTERNAL_SERVER_ERROR
    assert ServerError(_error(), status_code=503).get_status_code() == 503


def test_server_error_rejects_4xx_statuses() -> None:
    with pytest.raises(ValueError):
        ServerError(_error(), status_code=404)


def test_server_error_to_dict_masks_message_but_reason_is_kept() -> None:
    error = ServerError(_error(reason="database credentials rejected"))

    assert error.to_dict() == {"code": "SOME_CODE", "message": "Internal server error"}
    assert error.get_reason() == "database credentials rejected"
    assert error.base_error.message == "some message"


def test_header_dict_is_copied_on_construction_and_read() -> None:
    source = {"X-Trace": "abc"}
    error = ClientError(_error(), headers=source)

    source["X-Trace"] = "mutated"
    assert error.get_headers() == {"X-Trace": "abc"}

    returned = error.get_headers()
    returned["X-Injected"] = "no"
    assert error.get_headers() == {"X-Trace": "abc"}


def test_retryable_flag_stays_internal() -> None:
    """``retryable`` is an internal scheduling hint: it appears in no public
    projection unless the contract explicitly adds one."""
    retriable = Error(
        code="BACKEND_BUSY", message="try again", reason="upstream 503", retryable=True
    )

    client = ClientError(retriable)
    server = ServerError(retriable)
    for projection in (client.to_dict(), server.to_dict(), retriable.to_dict(), retriable.public()):
        assert "retryable" not in projection

    assert retriable.retryable is True  # the flag survives internally for callers
