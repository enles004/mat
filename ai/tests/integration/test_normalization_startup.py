# ai/tests/integration/test_normalization_startup.py
"""Startup DI wiring for the locked normalization catalog (Plan 1 Task 5).

The lifespan activates the checked-in catalog exactly once against its
checksum lock; a drifted catalog or unusable lock is a configuration
failure that blocks startup instead of falling back to a sanitized-but-
ready service. Requests never reload the file.
"""

import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.api.dependencies import get_normalizer
from src.api.server import create_app
from src.core.settings import Settings
from src.domain.contracts import Normalizer
from src.domain.entities import ArtifactManifest, NormalizationResult, Prediction
from src.domain.exceptions import CatalogError
from src.nlp.preprocessing.catalog_loader import CatalogLoader
from tests.integration.api.fakes import FakeModel, successful_loader

PROD_CATALOG = Path("configs/normalization.yaml")
PROD_LOCK = Path("configs/normalization.lock.json")
ACTIVATED = "normalization catalog activated"

# Catalog contents that must never appear in the activation log.
RULE_CONTENTS = (r"(?<!\w)cx(?!\w)", "không", "hyundai")


def prod_checksum() -> str:
    return "sha256:" + hashlib.sha256(PROD_CATALOG.read_bytes()).hexdigest()


def drift_catalog(tmp_path: Path) -> Path:
    """A complete, valid, but unreviewed catalog variant."""
    body = PROD_CATALOG.read_text(encoding="utf-8").replace(
        "replacement: không",
        "replacement: khôngV2",
    )
    path = tmp_path / "normalization-drift.yaml"
    path.write_text(body, encoding="utf-8")
    return path


class _ChannelCapture(logging.Handler):
    """Capture records on the channel logger itself.

    ``setup_logging`` wires the APP channel to logger ``src`` with
    ``propagate = False`` (an ancestor handler must never make an
    unconfigured channel look configured), so pytest's root-level caplog
    cannot see channel records by design — this handler attaches where the
    record is actually handled.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_startup_activates_locked_catalog_once_and_logs_metadata() -> None:
    capture = _ChannelCapture()
    channel = logging.getLogger("src.api.dependencies")
    channel.addHandler(capture)
    try:
        app = create_app(Settings(), successful_loader)
        with TestClient(app):
            records = [r for r in capture.records if ACTIVATED in r.getMessage()]
    finally:
        channel.removeHandler(capture)
    assert len(records) == 1, [r.getMessage() for r in records]
    message = records[0].getMessage()
    assert "dictionary_version=2.0.0" in message
    assert f"catalog_checksum={prod_checksum()}" in message
    assert "rule_count=13" in message
    for content in RULE_CONTENTS:
        assert content not in message, f"activation log leaked rule content: {content}"

    activation = app.state.runtime.normalization
    assert activation is not None
    assert activation.dictionary_version == "2.0.0"
    assert activation.rule_count == 13
    assert activation.catalog_checksum == prod_checksum()


def test_get_normalizer_returns_one_stable_instance_across_requests() -> None:
    app = create_app(Settings(), successful_loader)
    with TestClient(app) as client:
        state = app.state.runtime
        assert get_normalizer(state) is state.normalization.normalizer
        assert client.get("/health-check").status_code == 200
        response = client.post("/predict", json={"text": "xe cx ổn"})
        assert response.status_code == 200
        assert get_normalizer(state) is state.normalization.normalizer


@pytest.mark.parametrize(
    ("scenario", "catalog", "lock"),
    [
        ("catalog_drift", drift_catalog, PROD_LOCK),
        ("missing_lock", PROD_CATALOG, Path("nonexistent/normalization.lock.json")),
        ("malformed_lock", PROD_CATALOG, "malformed-lock.json"),
    ],
)
def test_catalog_configuration_failure_blocks_startup_loudly(
    tmp_path: Path,
    scenario: str,
    catalog: Callable[[Path], Path] | Path,
    lock: Path | str,
) -> None:
    catalog_path = catalog(tmp_path) if callable(catalog) else catalog
    lock_path = tmp_path / lock if isinstance(lock, str) else lock
    if scenario == "malformed_lock":
        lock_path.write_text("{not json", encoding="utf-8")

    settings = Settings(
        normalization_catalog_path=catalog_path,
        normalization_lock_path=lock_path,
    )
    app = create_app(settings, successful_loader)
    with pytest.raises(CatalogError):
        with TestClient(app):
            pass
    assert app.state.runtime.normalization is None, "drift must not activate"


def test_requests_never_reload_the_catalog_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []

    real_load = CatalogLoader.load

    def counting_load(self: CatalogLoader, path: Path) -> object:
        calls.append(path)
        return real_load(self, path)

    monkeypatch.setattr("src.nlp.preprocessing.catalog_loader.CatalogLoader.load", counting_load)
    app = create_app(Settings(), successful_loader)
    with TestClient(app) as client:
        assert len(calls) == 1, "lifespan must load the catalog exactly once"
        client.get("/health-check")
        client.get("/livez")
        client.post("/predict", json={"text": "xe cx ổn"})
        assert len(calls) == 1, "requests must never reload the catalog"


class _RecordingModel:
    """SentimentModel stand-in that records exactly the text it is asked to score."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def load(self) -> None:
        return None

    def predict(self, text: str) -> Prediction:
        self.seen.append(text)
        return Prediction(
            label="positive",
            confidence=0.70,
            scores={"negative": 0.10, "neutral": 0.20, "positive": 0.70},
            uncertain=False,
        )

    def metadata(self) -> ArtifactManifest:
        return FakeModel().metadata()

    def health(self) -> str:
        return "ready"


def recording_loader(backend: str) -> _RecordingModel:
    return _RecordingModel()


class _CountingNormalizer:
    """Normalizer spy: records every text pushed through the active catalog."""

    def __init__(self, inner: Normalizer) -> None:
        self._inner = inner
        self.calls: list[str] = []

    def normalize(self, text: str) -> NormalizationResult:
        self.calls.append(text)
        return self._inner.normalize(text)


def test_predict_stays_on_the_artifacts_declared_raw_view() -> None:
    """The checked-in baseline artifact declares the raw view in its
    preprocessor manifest, so /predict must feed the request text to the
    model untouched: the runtime normalizer is never invoked on the serving
    path."""
    manifest = json.loads(
        (Path(Settings().baseline_artifact_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    assert "(raw view)" in manifest["preprocessor"]["name"]

    app = create_app(Settings(), recording_loader)
    with TestClient(app) as client:
        activation = app.state.runtime.normalization
        assert activation is not None
        spy = _CountingNormalizer(activation.normalizer)
        app.state.runtime.normalization = replace(activation, normalizer=spy)

        response = client.post("/predict", json={"text": "xe cx ko dc"})
        assert response.status_code == 200

    model = app.state.runtime.model
    assert isinstance(model, _RecordingModel)
    assert model.seen == ["xe cx ko dc"], "the model must score the raw request text"
    assert spy.calls == [], "the runtime normalizer must never be called on /predict"
