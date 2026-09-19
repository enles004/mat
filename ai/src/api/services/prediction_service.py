import logging
import time
from typing import TYPE_CHECKING

from src.core.logging import model_latency_ms_ctx
from src.domain.exceptions import TextExceedsModelLimit
from src.domain.outcomes import PredictionOutcome
from src.libs.result import Err, Error, Ok, Return
from src.nlp.modeling.limits import ModelInputLimiter

if TYPE_CHECKING:
    from src.api.dependencies import RuntimeState

logger = logging.getLogger(__name__)


class PredictionService:
    """Score one request and branch it into a Result of outcomes.

    The service owns the operational branching of a prediction: a not-ready
    runtime is a retryable ``MODEL_NOT_READY`` failure, an over-limit input
    is the expected ``TEXT_EXCEEDS_MODEL_LIMIT`` client failure, and an
    operational model failure is ``INFERENCE_FAILED`` with the exception
    retained as the internal reason. Exactly one backend is ever consulted:
    a failure is reported, never retried on another backend or loader.
    """

    def __init__(self, state: "RuntimeState", limiter: ModelInputLimiter | None = None) -> None:
        self._state = state
        self._limiter = limiter if limiter is not None else ModelInputLimiter()

    def execute(self, text: str) -> Ok[PredictionOutcome] | Err:
        model = self._state.model
        if not self._state.ready or model is None:
            return Return.err(
                Error(
                    code="MODEL_NOT_READY",
                    message="The model is not ready; the request may be retried.",
                    retryable=True,
                )
            )
        try:
            self._limiter.enforce(model, text)
            predict_started = time.perf_counter()
            prediction = model.predict(text)
            model_latency_ms_ctx.set((time.perf_counter() - predict_started) * 1_000)
            manifest = model.metadata()
        except TextExceedsModelLimit as exc:
            return Return.err(
                Error(
                    code="TEXT_EXCEEDS_MODEL_LIMIT",
                    message=str(exc),
                    reason=exc,
                    retryable=False,
                )
            )
        except Exception as exc:
            logger.exception("prediction inference failed")
            return Return.err(
                Error(
                    code="INFERENCE_FAILED",
                    message="The model failed to score the request.",
                    reason=exc,
                    retryable=False,
                )
            )
        return Return.ok(
            PredictionOutcome(
                prediction=prediction,
                manifest=manifest,
                degraded=self._state.degraded,
            )
        )
