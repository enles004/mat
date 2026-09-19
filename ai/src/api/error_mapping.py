from collections.abc import Mapping
from dataclasses import dataclass
from http import HTTPStatus

from src.api.error_types import APIError, ClientError, ServerError
from src.libs.result import Error


@dataclass(frozen=True)
class ErrorPolicy:
    """How one internal error code is projected onto a public API error."""

    cls: type[APIError]
    status_code: int
    public_title: str


ERROR_POLICY: Mapping[str, ErrorPolicy] = {
    "MODEL_NOT_READY": ErrorPolicy(
        cls=ServerError,
        status_code=HTTPStatus.SERVICE_UNAVAILABLE,
        public_title="Service Unavailable",
    ),
    "TEXT_EXCEEDS_MODEL_LIMIT": ErrorPolicy(
        cls=ClientError,
        status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
        public_title="Unprocessable Entity",
    ),
    "INFERENCE_FAILED": ErrorPolicy(
        cls=ServerError,
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        public_title="Internal Server Error",
    ),
    "HEALTH_CHECK_FAILED": ErrorPolicy(
        cls=ServerError,
        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
        public_title="Internal Server Error",
    ),
}


def classify(error: Error) -> APIError:
    """Map an internal Error onto its public API error, failing closed to 500.

    Unknown or missing codes never leak internals: they classify as
    ``ServerError``. The status is taken from the policy table only, never
    inferred from the code prefix.
    """
    policy = ERROR_POLICY.get(error.code)
    if policy is None:
        return ServerError(error)
    return policy.cls(error, status_code=policy.status_code)
