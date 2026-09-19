from abc import ABC, abstractmethod
from http import HTTPStatus

from src.libs.result import Error, ErrorReason


class APIError(Exception, ABC):
    """Base class for HTTP-mapped errors carrying an internal :class:`Error`.

    ``to_dict()`` is an INTERNAL projection used for diagnostics only; the only
    wire format is RFC 9457 problem+json rendered by ``src.api.errors``. The
    internal ``reason`` never becomes public ``detail``.
    """

    def __init__(
        self,
        base_error: Error,
        status_code: int,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.base_error = base_error
        self.status_code = status_code
        self.headers = dict(headers or {})
        super().__init__(base_error.message)

    @abstractmethod
    def get_status_code(self) -> int:
        """Return the HTTP status code this error maps to."""

    @abstractmethod
    def to_dict(self) -> dict[str, str]:
        """Return the internal projection; never the wire format."""

    def get_reason(self) -> ErrorReason:
        """Return the internal reason for diagnostics; never serialized publicly."""
        return self.base_error.reason

    def get_headers(self) -> dict[str, str]:
        """Return a copy of the response headers attached to this error."""
        return dict(self.headers)


class ClientError(APIError):
    """Client error (4xx) with a transparent code and message projection."""

    def __init__(
        self,
        base_error: Error,
        status_code: int = HTTPStatus.BAD_REQUEST,
        headers: dict[str, str] | None = None,
    ) -> None:
        if not 400 <= status_code < 500:
            raise ValueError(f"ClientError requires a 4xx status code, got {status_code}")
        super().__init__(base_error, status_code, headers)

    def get_status_code(self) -> int:
        return self.status_code

    def to_dict(self) -> dict[str, str]:
        return {"code": self.base_error.code, "message": self.base_error.message}


class ServerError(APIError):
    """Server error (5xx) whose message is masked in every public projection."""

    def __init__(
        self,
        base_error: Error,
        status_code: int = HTTPStatus.INTERNAL_SERVER_ERROR,
        headers: dict[str, str] | None = None,
    ) -> None:
        if not 500 <= status_code < 600:
            raise ValueError(f"ServerError requires a 5xx status code, got {status_code}")
        super().__init__(base_error, status_code, headers)

    def get_status_code(self) -> int:
        return self.status_code

    def to_dict(self) -> dict[str, str]:
        return {"code": self.base_error.code, "message": "Internal server error"}

