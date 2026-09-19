import logging
from typing import TYPE_CHECKING

from src.domain.outcomes import HealthCheckOutcome
from src.libs.result import Err, Error, Ok, Return

if TYPE_CHECKING:
    from src.api.dependencies import RuntimeState

logger = logging.getLogger(__name__)


class HealthCheckService:
    """Read the active model's health and branch it into a Result outcome.

    An unready runtime is a retryable ``MODEL_NOT_READY`` failure; a failure
    to read the active model's metadata fails closed as
    ``HEALTH_CHECK_FAILED`` with the exception retained as the internal
    reason. The check never predicts and never loads anything.
    """

    def __init__(self, state: "RuntimeState") -> None:
        self._state = state

    def execute(self) -> Ok[HealthCheckOutcome] | Err:
        model = self._state.model
        if not self._state.ready or model is None:
            return Return.err(
                Error(
                    code="MODEL_NOT_READY",
                    message="The model is not ready; the check may be retried.",
                    retryable=True,
                )
            )
        try:
            manifest = model.metadata()
        except Exception as exc:
            logger.exception("health-check metadata read failed")
            return Return.err(
                Error(
                    code="HEALTH_CHECK_FAILED",
                    message="The health check could not read the active model state.",
                    reason=exc,
                    retryable=True,
                )
            )
        return Return.ok(
            HealthCheckOutcome(
                backend=manifest.backend,
                degraded=self._state.degraded,
                version=manifest.model_version,
            )
        )
