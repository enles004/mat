import uuid
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar

from result import Err as ResultError
from result import Ok as ResultOk
from result import Result, is_err, is_ok

T = TypeVar("T")

type ErrorReason = str | dict[str, Any] | Exception | Error


@dataclass(frozen=True, kw_only=True)
class Error:
    """Frozen, uniquely identified error whose reason stays internal."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    code: str
    message: str
    reason: ErrorReason = ""
    retryable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_code": self.code,
            "message": self.message,
            "error_id": self.id,
            "reason": self.reason,
        }

    def public(self) -> dict[str, Any]:
        return {
            "error_code": self.code,
            "message": self.message,
            "error_id": self.id,
        }

    def __repr__(self) -> str:
        return (
            f"Error(message={self.message}, id={self.id}, "
            f"code={self.code}, reason={self.reason})"
        )


class Ok(ResultOk[T]):
    """Successful result carrying a value, with literal is_ok/is_err returns."""

    def is_err(self) -> Literal[False]:
        return False

    def is_ok(self) -> Literal[True]:
        return True


class Err(ResultError[Error]):
    """Failed result carrying an Error, with literal is_ok/is_err returns."""

    def is_err(self) -> Literal[True]:
        return True

    def is_ok(self) -> Literal[False]:
        return False


class Return:
    """Construction helpers used at Result call sites."""

    @staticmethod
    def ok(value: T) -> Ok[T]:
        return Ok(value)

    @staticmethod
    def err(error: Error) -> Err:
        """Err is already concrete: its base is parameterized with Error."""
        return Err(error)


__all__ = [
    "Err",
    "Error",
    "ErrorReason",
    "Ok",
    "Result",
    "Return",
    "is_err",
    "is_ok",
]
