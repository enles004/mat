"""Unit tests for the transformer artifact adapter (fake HF stack, no network).

The adapter is the serving seam for ``MAT_MODEL_BACKEND=transformer``: it
must load a local HuggingFace directory offline, trust the artifact manifest
labels over the model config, reindex scores into the frozen label order,
and reject over-limit input without truncation — the same behavioral
contract as ``LinearArtifactModel``. Torch/transformers are faked through
``sys.modules`` so the tests never import the real stack.
"""

import contextlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.domain.contracts import SentimentModel
from src.domain.entities import (
    ArtifactManifest,
    CalibrationManifest,
    Prediction,
    PreprocessorManifest,
)
from src.domain.exceptions import ArtifactUnavailable, TextExceedsModelLimit
from src.nlp.modeling.transformer_model import TransformerArtifactModel


class _FakeLogits:
    def __init__(self, rows: list[list[float]]) -> None:
        self._rows = rows

    def tolist(self) -> list[list[float]]:
        return self._rows


class _FakeOutput:
    def __init__(self, logits: _FakeLogits) -> None:
        self.logits = logits


class _FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[list[int]]]:
        self.calls.append({"text": text, **kwargs})
        return {"input_ids": [[101, 5, 6, 7, 102]]}


class _FakeClassifier:
    def __init__(self) -> None:
        self.eval_calls = 0
        self.forward_calls = 0
        # Deliberately conflicting with the manifest labels order.
        self.id2label = {0: "positive", 1: "negative", 2: "neutral"}

    def eval(self) -> "_FakeClassifier":
        self.eval_calls += 1
        return self

    def __call__(self, **encoded: object) -> _FakeOutput:
        self.forward_calls += 1
        return _FakeOutput(_FakeLogits([[0.1, 2.0, 0.3]]))


class _FakeStack:
    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.classifier = _FakeClassifier()
        self.tokenizer_paths: list[tuple[str, dict[str, object]]] = []
        self.classifier_paths: list[tuple[str, dict[str, object]]] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stack = self

        class _FakeAutoTokenizer:
            @staticmethod
            def from_pretrained(path: str, **kwargs: object) -> _FakeTokenizer:
                stack.tokenizer_paths.append((path, kwargs))
                return stack.tokenizer

        class _FakeAutoModel:
            @staticmethod
            def from_pretrained(path: str, **kwargs: object) -> _FakeClassifier:
                stack.classifier_paths.append((path, kwargs))
                return stack.classifier

        monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=contextlib.nullcontext))
        monkeypatch.setitem(
            sys.modules,
            "transformers",
            SimpleNamespace(
                AutoTokenizer=_FakeAutoTokenizer,
                AutoModelForSequenceClassification=_FakeAutoModel,
            ),
        )


def _manifest(**overrides: object) -> ArtifactManifest:
    fields: dict[str, object] = {
        "schema_version": "1",
        "model_name": "bamibert-finetuned",
        "model_version": "test",
        "backend": "transformer",
        "labels": ["negative", "neutral", "positive"],
        "preprocessor": PreprocessorManifest(name="tokenizer", version="1", config_checksum="abc"),
        "calibration": CalibrationManifest(method="none (raw softmax)", fitted_on="not calibrated"),
        "data_checksum": "data",
        "payload_checksum": "sha256:stub",
        "git_revision": "5c7055c",
        "license": "bsd-3-clause-clear, other",
        "evaluation_report": "reports/transformer-champion.md",
    }
    fields.update(overrides)
    return ArtifactManifest.model_validate(fields)


def _model(
    directory: Path, manifest: ArtifactManifest, threshold: float = 0.6
) -> TransformerArtifactModel:
    return TransformerArtifactModel(directory, manifest, threshold)


def test_load_reads_local_files_only_and_switches_to_eval_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _FakeStack()
    stack.install(monkeypatch)
    manifest = _manifest()
    model = _model(tmp_path, manifest)

    assert model.load() is None
    assert len(stack.tokenizer_paths) == 1
    assert len(stack.classifier_paths) == 1
    assert stack.tokenizer_paths[0][0] == str(tmp_path)
    assert stack.classifier_paths[0][1] == {"local_files_only": True}
    assert stack.classifier.eval_calls == 1

    prediction = model.predict("xe chạy tốt")
    assert isinstance(prediction, Prediction)


def test_load_is_idempotent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _FakeStack()
    stack.install(monkeypatch)
    model = _model(tmp_path, _manifest())

    model.load()
    model.load()

    assert len(stack.classifier_paths) == 1


def test_predict_reindexes_scores_to_frozen_label_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _FakeStack()
    stack.install(monkeypatch)
    model = _model(tmp_path, _manifest())
    model.load()

    prediction = model.predict("xe này chạy rất tốt")

    # The head favors index 1; the manifest maps index 1 to "neutral".
    assert prediction.label == "neutral"
    assert prediction.confidence == pytest.approx(0.7508, abs=1e-3)
    assert list(prediction.scores) == ["negative", "neutral", "positive"]
    assert prediction.scores["negative"] == pytest.approx(0.1123, abs=1e-3)
    assert prediction.scores["positive"] == pytest.approx(0.1371, abs=1e-3)
    assert prediction.uncertain is False


def test_predict_trusts_manifest_labels_over_model_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _FakeStack()
    stack.install(monkeypatch)
    model = _model(tmp_path, _manifest())
    model.load()

    prediction = model.predict("anything")

    # The fake config.id2label claims index 1 is "negative"; the manifest
    # says "neutral" and the manifest wins.
    assert prediction.label == "neutral"


def test_predict_marks_top_score_below_threshold_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _FakeStack()
    stack.install(monkeypatch)
    model = _model(tmp_path, _manifest(), threshold=0.99)
    model.load()

    assert model.predict("anything").uncertain is True


def test_predict_rejects_over_limit_text_without_truncation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stack = _FakeStack()
    stack.install(monkeypatch)
    model = _model(tmp_path, _manifest(max_input_tokens=4))
    model.load()

    with pytest.raises(TextExceedsModelLimit) as excinfo:
        model.predict("năm token")
    assert excinfo.value.limit == 4
    assert stack.classifier.forward_calls == 0


def test_predict_requires_load_first(tmp_path: Path) -> None:
    model = _model(tmp_path, _manifest())
    with pytest.raises(RuntimeError, match="load"):
        model.predict("anything")


def test_missing_transformers_dependency_raises_actionable_artifact_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    model = _model(tmp_path, _manifest())

    with pytest.raises(ArtifactUnavailable, match=r"uv sync --locked --extra transformer"):
        model.load()


def test_load_failure_maps_to_artifact_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenAutoModel:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> object:
            raise OSError("corrupt checkpoint")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(no_grad=contextlib.nullcontext))
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=lambda path, **kw: _FakeTokenizer()),
            AutoModelForSequenceClassification=_BrokenAutoModel,
        ),
    )
    model = _model(tmp_path, _manifest())

    with pytest.raises(ArtifactUnavailable, match="failed to load"):
        model.load()


def test_protocol_conformance(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stack = _FakeStack()
    stack.install(monkeypatch)
    manifest = _manifest()
    model = _model(tmp_path, manifest)

    assert isinstance(model, SentimentModel)
    assert model.load() is None
    assert model.metadata() is manifest
    assert model.health() == "ready"
