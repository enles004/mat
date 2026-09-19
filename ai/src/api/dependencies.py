import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib  # type: ignore[import-untyped]
from fastapi import Depends, Request
from pydantic import ValidationError

from src.api.services.health_check_service import HealthCheckService
from src.api.services.prediction_service import PredictionService
from src.core.settings import Settings
from src.domain.artifacts import CatalogActivation
from src.domain.contracts import ModelLoader, Normalizer, SentimentModel
from src.domain.exceptions import ArtifactUnavailable, CatalogError
from src.nlp.constants import MODEL_PAYLOAD_NAME
from src.nlp.modeling.artifact_model import LinearArtifactModel
from src.nlp.modeling.registry import ArtifactRegistry
from src.nlp.preprocessing.provider import NormalizerProvider

logger = logging.getLogger(__name__)
_ARTIFACT_REGISTRY = ArtifactRegistry()


@dataclass
class RuntimeState:
    model: SentimentModel | None = None
    ready: bool = False
    degraded: bool = False
    backend: str | None = None
    startup_error: str | None = None
    normalization: CatalogActivation | None = None


def load_model_from_artifact(
    backend: Literal["baseline", "transformer"], settings: Settings | None = None
) -> SentimentModel:
    """Production ``ModelLoader``: verify then load a trusted local artifact.

    Only genuine artifact failures are reported as ``ArtifactUnavailable``:
    verification errors from ``verify_artifact`` (missing directory, missing
    manifest, empty payload, checksum mismatch, malformed manifest — raised
    as ``ValueError``/``ValidationError``), the backend cross-check, and the
    artifact read itself. Anything else — e.g. a bug while building the
    adapter — propagates unchanged so it is never mistaken for a missing
    artifact by startup fallback decisions.

    The baseline branch reads a joblib payload; the transformer branch hands
    the verified directory to the HF adapter, which loads it locally inside
    its own ``load()`` — a transformer artifact has no joblib payload.
    """
    resolved = settings if settings is not None else Settings()
    directory = (
        resolved.baseline_artifact_dir
        if backend == "baseline"
        else resolved.transformer_artifact_dir
    )
    artifact_dir = Path(directory)
    try:
        manifest = _ARTIFACT_REGISTRY.verify(artifact_dir)
    except (
        ValueError,
        ValidationError,
        FileNotFoundError,
        NotADirectoryError,
    ) as error:
        raise ArtifactUnavailable(
            f"artifact for backend {backend!r} is unavailable: {error}"
        ) from error
    expected_backend: Literal["linear", "transformer"] = (
        "linear" if backend == "baseline" else "transformer"
    )
    if manifest.backend != expected_backend:
        raise ArtifactUnavailable(
            f"artifact in {directory} declares backend {manifest.backend!r}; "
            f"expected {expected_backend!r}"
        )
    if backend == "baseline":
        try:
            payload = joblib.load(artifact_dir / MODEL_PAYLOAD_NAME)
        except Exception as error:
            raise ArtifactUnavailable(
                f"artifact for backend {backend!r} is unavailable: {error}"
            ) from error
        return LinearArtifactModel(payload, manifest, resolved.uncertain_threshold)
    from src.nlp.modeling.transformer_model import TransformerArtifactModel

    model = TransformerArtifactModel(artifact_dir, manifest, resolved.uncertain_threshold)
    try:
        model.load()
    except ArtifactUnavailable:
        raise
    except Exception as error:
        raise ArtifactUnavailable(
            f"artifact for backend {backend!r} is unavailable: {error}"
        ) from error
    return model


def _sanitized(error: BaseException) -> str:
    """A stack-trace-free startup reason for state retention; details go to logs."""
    return f"{type(error).__name__}: {error}"


def _load_model(loader: ModelLoader, backend: Literal["baseline", "transformer"]) -> SentimentModel:
    """Load and wake one backend; any load failure propagates to the caller."""
    model = loader(backend)
    model.load()
    model.metadata()
    return model


def _activate(state: RuntimeState, model: SentimentModel, *, degraded: bool) -> None:
    state.model = model
    state.ready = True
    state.degraded = degraded
    state.backend = model.metadata().backend
    state.startup_error = None


def startup_load(state: RuntimeState, settings: Settings, loader: ModelLoader) -> None:
    """Startup model selection: a fixed backend, or the one-shot auto fallback.

    ``baseline``/``transformer`` load their own backend or stay unready —
    no silent substitution. ``auto`` tries the transformer exactly once
    during startup; only a known artifact-unavailable failure
    (``ArtifactUnavailable``) loads the baseline and serves degraded. Any
    other failure is a wiring bug, not a missing artifact: the service
    stays unready exactly like a fixed-backend failure. Only ``Exception``
    is ever caught (never ``BaseException``); sanitized reasons are
    retained in ``state`` while full diagnostics go to the log.
    """
    if settings.model_backend != "auto":
        try:
            _activate(state, _load_model(loader, settings.model_backend), degraded=False)
        except Exception as error:
            logger.exception("Startup model load failed for backend %s", settings.model_backend)
            state.startup_error = _sanitized(error)
        return

    try:
        _activate(state, _load_model(loader, "transformer"), degraded=False)
        return
    except ArtifactUnavailable as error:
        logger.exception(
            "Auto startup: transformer load failed; falling back to baseline (degraded)"
        )
        transformer_error = _sanitized(error)
    except Exception as error:
        logger.exception("Startup model load failed for backend transformer")
        state.startup_error = _sanitized(error)
        return
    try:
        _activate(state, _load_model(loader, "baseline"), degraded=True)
        state.startup_error = (
            f"transformer unavailable ({transformer_error}); serving degraded baseline"
        )
    except Exception as error:
        logger.exception("Auto startup: baseline fallback also failed")
        state.startup_error = (
            f"transformer unavailable ({transformer_error}); baseline failed ({_sanitized(error)})"
        )


def _expected_checksum(lock_path: Path) -> str:
    """The checksum the catalog must hash to, read from the operator lock."""
    try:
        raw = lock_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"normalization lock unreadable: {lock_path}") from exc
    try:
        lock = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"normalization lock is malformed: {lock_path}: {exc}") from exc
    expected = lock.get("catalog_checksum") if isinstance(lock, dict) else None
    if not isinstance(expected, str):
        raise CatalogError(f"normalization lock is missing catalog_checksum: {lock_path}")
    return expected


def startup_normalization(state: RuntimeState, settings: Settings) -> None:
    """Activate the locked normalization catalog exactly once, during lifespan.

    Unlike model loading there is no fallback semantics: a drifted catalog,
    an unusable lock, or a compilation failure is a configuration failure
    whose exception propagates out of the lifespan — the service never
    becomes ready on a sanitized guess. The activation record (normalizer,
    dictionary version, checksum, rule count) is published atomically on
    ``RuntimeState``; the activation log carries metadata only, never rule
    contents or input text.
    """
    expected_checksum = _expected_checksum(settings.normalization_lock_path)
    provider = NormalizerProvider(lock_path=settings.normalization_lock_path)
    activation = provider.activate(settings.normalization_catalog_path, expected_checksum)
    state.normalization = activation
    logger.info(
        "normalization catalog activated (dictionary_version=%s, "
        "catalog_checksum=%s, rule_count=%d)",
        activation.dictionary_version,
        activation.catalog_checksum,
        activation.rule_count,
    )


def get_normalizer(state: RuntimeState) -> Normalizer:
    """Request-time access to the active normalizer; no filesystem I/O."""
    activation = state.normalization
    if activation is None:
        raise CatalogError("normalizer not activated: startup has not activated a catalog")
    return activation.normalizer


def get_runtime_state(request: Request) -> RuntimeState:
    """The live runtime state published by the application lifespan.

    Reads ``app.state.runtime`` only: request-time dependency resolution
    never loads an artifact or touches the filesystem. A mistyped runtime
    value is a wiring bug that fails loudly instead of degrading silently.
    """
    runtime = request.app.state.runtime
    if not isinstance(runtime, RuntimeState):
        raise RuntimeError(
            f"app.state.runtime must hold a RuntimeState; got {type(runtime).__name__}"
        )
    return runtime


def get_prediction_service(
    state: RuntimeState = Depends(get_runtime_state),
) -> PredictionService:
    """Request-scoped prediction service bound to the live runtime state."""
    return PredictionService(state)


def get_health_check_service(
    state: RuntimeState = Depends(get_runtime_state),
) -> HealthCheckService:
    """Request-scoped health-check service bound to the live runtime state."""
    return HealthCheckService(state)
