from fastapi import APIRouter, Depends, Request

from src.api.dependencies import get_prediction_service
from src.api.error_mapping import classify
from src.api.services.prediction_service import PredictionService
from src.api.v1.schemas import ModelResponse, PredictRequest, PredictResponse
from src.domain.entities import SentimentLabel
from src.libs.result import Err

router = APIRouter()


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unassigned"))


@router.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    request: Request,
    service: PredictionService = Depends(get_prediction_service),
) -> PredictResponse:
    result = service.execute(payload.text)
    if isinstance(result, Err):
        raise classify(result.err_value)
    outcome = result.ok_value
    return PredictResponse(
        label=outcome.prediction.label,
        confidence=outcome.prediction.confidence,
        scores={label: outcome.prediction.scores[label] for label in SentimentLabel},
        uncertain=outcome.prediction.uncertain,
        model=ModelResponse(
            backend=outcome.manifest.backend,
            version=outcome.manifest.model_version,
            degraded=outcome.degraded,
        ),
        request_id=_request_id(request),
    )
