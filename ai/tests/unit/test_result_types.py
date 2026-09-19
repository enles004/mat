"""Unit tests for the Result port in src.libs.result."""

from typing import Literal, assert_type

from src.libs.result import Err, Error, Ok, Return


def test_ok_round_trip() -> None:
    result = Return.ok(41)

    assert isinstance(result, Ok)
    assert result.is_ok() is True
    assert result.is_err() is False
    assert result.unwrap() == 41


def test_err_round_trip() -> None:
    error = Error(code="MODEL_LOAD_FAILED", message="artifact unreadable")
    result = Return.err(error)

    assert isinstance(result, Err)
    assert result.is_err() is True
    assert result.is_ok() is False
    assert result.unwrap_err() is error


def test_is_ok_and_is_err_narrow_to_literals() -> None:
    ok_result = Return.ok("value")
    err_result = Return.err(Error(code="CODE", message="boom"))

    assert_type(ok_result.is_ok(), Literal[True])
    assert_type(ok_result.is_err(), Literal[False])
    assert_type(err_result.is_err(), Literal[True])
    assert_type(err_result.is_ok(), Literal[False])
    assert ok_result.is_ok() is True
    assert ok_result.is_err() is False
    assert err_result.is_err() is True
    assert err_result.is_ok() is False


def test_error_ids_are_unique_across_instances() -> None:
    first = Error(code="CODE", message="first")
    second = Error(code="CODE", message="second")

    assert first.id != second.id
    assert len(first.id) == 32  # uuid4 hex


def test_public_projection_never_contains_reason() -> None:
    error = Error(code="CODE", message="public message", reason="secret stack context")

    public = error.public()

    assert "reason" not in public
    assert "secret stack context" not in public.values()
    assert public == {
        "error_code": "CODE",
        "message": "public message",
        "error_id": error.id,
    }


def test_to_dict_carries_reason_for_internal_diagnostics() -> None:
    error = Error(code="CODE", message="message", reason="internal detail")

    assert error.to_dict() == {
        "error_code": "CODE",
        "message": "message",
        "error_id": error.id,
        "reason": "internal detail",
    }


def test_retryable_defaults_to_false() -> None:
    assert Error(code="CODE", message="message").retryable is False
    assert Error(code="CODE", message="message", retryable=True).retryable is True
