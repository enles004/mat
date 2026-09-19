from fastapi import APIRouter, Depends
from pydantic import BaseModel

from src.api.dependencies import get_health_check_service
from src.api.error_mapping import classify
from src.api.services.health_check_service import HealthCheckService
from src.libs.result import Err


class ReadinessStatus(BaseModel):
    backend: str
    degraded: bool
    version: str


router = APIRouter()


@router.get("/livez")
def livez() -> dict[str, str]:
    # Liveness answers 200 whenever the ASGI process can answer at all.
    return {"status": "ok"}


@router.get("/health-check")
def health_check(
    service: HealthCheckService = Depends(get_health_check_service),
) -> ReadinessStatus:
    result = service.execute()
    if isinstance(result, Err):
        raise classify(result.err_value)
    outcome = result.ok_value
    return ReadinessStatus(
        backend=outcome.backend,
        degraded=outcome.degraded,
        version=outcome.version,
    )
