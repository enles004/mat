"""Unit tests for the champion fine-tune and export pipeline (DI fakes).

The champion exporter refits the transformer on *all* development rows
(the same frozen protocol as the baseline champion: no epoch selection,
no frozen-test access) and exports a serving artifact whose manifest the
registry can verify. Fakes mirror ``tests/integration/modeling/
test_transformer.py``; no HuggingFace download happens here.
"""

import hashlib
import re
from pathlib import Path
from typing import Any

from src.domain.artifacts import ResolvedModelSource
from src.domain.entities import DatasetRow
from src.domain.training import FoldManifest, SplitManifest
from src.nlp.modeling.registry import ArtifactRegistry
from src.nlp.modeling.transformer_champion import TransformerChampionExporter

_LABELS = ["negative", "neutral", "positive"]


class _FakeTokenizer:
    model_max_length = 128

    def __init__(self) -> None:
        self.saved: list[Path] = []

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [1]}

    def save_pretrained(self, directory: Any) -> None:
        self.saved.append(Path(directory))
        (Path(directory) / "tokenizer.json").write_text("tokenizer", encoding="utf-8")


class _FakeModel:
    def __init__(self) -> None:
        self.saved: list[Path] = []

    def save_pretrained(self, directory: Any) -> None:
        self.saved.append(Path(directory))
        (Path(directory) / "model.safetensors").write_text("weights", encoding="utf-8")


class _FakeDataset:
    def __init__(self, texts: list[str], labels_: list[int] | None) -> None:
        self.texts = texts
        self.labels = labels_


def _rows() -> list[DatasetRow]:
    return [
        DatasetRow(
            id=f"{label}-{index}",
            raw_text=f"raw-{label}-{index}",
            normalized_text=f"normalized-{label}-{index}",
            label=label,
            aspect="engine",
            style="review",
            noise_types=[],
            difficulty="easy",
            canonical_group_id=f"group-{label}-{index}",
            generation_batch="test",
            review_status="approved",
        )
        for label in _LABELS
        for index in range(5)
    ] + [
        DatasetRow(
            id="frozen",
            raw_text="raw-frozen",
            normalized_text="normalized-frozen",
            label="positive",
            aspect="engine",
            style="review",
            noise_types=[],
            difficulty="easy",
            canonical_group_id="frozen-group",
            generation_batch="test",
            review_status="approved",
        )
    ]


def _split_manifest(rows: list[DatasetRow]) -> SplitManifest:
    development_ids = [row.id for row in rows if row.id != "frozen"]
    folds = [
        FoldManifest(
            fold=fold,
            validation_ids=[f"{label}-{fold}" for label in _LABELS],
            train_ids=[
                row_id
                for row_id in development_ids
                if row_id not in {f"{label}-{fold}" for label in _LABELS}
            ],
        )
        for fold in range(5)
    ]
    return SplitManifest(
        seed=42,
        dataset_checksum="frozen",
        development_ids=development_ids,
        test_ids=["frozen"],
        folds=folds,
    )


def _source() -> ResolvedModelSource:
    return ResolvedModelSource(
        repo_id="Qualcomm-AI-Research/BamiBERT",
        revision="a" * 40,
        license_id="bsd-3-clause-clear, other (qualcomm-responsible-ai-license)",
        intended_use="research and educational purposes",
        model_card_checksum="b" * 64,
    )


class _Harness:
    """Exporter with fake collaborators plus the records the tests assert on."""

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.models: list[_FakeModel] = []
        self.trainers: list[Any] = []
        self.seeds: list[int] = []
        self.load_revisions: list[str] = []

    def exporter(self) -> TransformerChampionExporter:
        harness = self

        class _FakeTrainer:
            def __init__(self, **kwargs: object) -> None:
                self.args = kwargs["args"]
                self.model = kwargs["model"]
                self.train_dataset = kwargs["train_dataset"]
                harness.trainers.append(self)

            def train(self) -> None:
                checkpoint = Path(str(self.args.output_dir)) / "checkpoint-1"
                checkpoint.mkdir(parents=True)
                (checkpoint / "payload").write_text("temporary", encoding="utf-8")

        def fake_model_loader(
            source: ResolvedModelSource,
        ) -> tuple[_FakeModel, dict[str, list[str]]]:
            harness.load_revisions.append(source.revision)
            model = _FakeModel()
            harness.models.append(model)
            return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}

        class _FakeConfig:
            model_type = "roberta"
            max_position_embeddings = 16
            pad_token_id = 1

        def fake_seed_initializer(seed: int) -> None:
            harness.seeds.append(seed)

        return TransformerChampionExporter(
            tokenizer_loader=lambda repo_id, revision: (
                harness.load_revisions.append(revision) or harness.tokenizer
            ),
            config_loader=lambda repo_id, revision: (
                harness.load_revisions.append(revision) or _FakeConfig()
            ),
            model_loader=fake_model_loader,
            dataset_builder=lambda tokenizer, texts, labels_: _FakeDataset(texts, labels_),
            collator_factory=lambda **kwargs: object(),
            trainer_factory=lambda **kwargs: _FakeTrainer(**kwargs),
            seed_initializer=fake_seed_initializer,
            device_probe=lambda: False,
        )

    def export(self, tmp_path: Path, data_path: Path, work_dir: Path) -> tuple[Any, Path]:
        rows = _rows()
        data_path.write_text("id,raw_text\nfrozen,raw-frozen\n", encoding="utf-8")
        artifact_dir = tmp_path / "champion"
        saved = self.exporter().export(
            rows=rows,
            manifest=_split_manifest(rows),
            source=_source(),
            data_path=data_path,
            artifact_dir=artifact_dir,
            work_dir=work_dir,
        )
        return saved, artifact_dir


def test_champion_arguments_mirror_folds_without_epoch_selection(tmp_path: Path) -> None:
    arguments = TransformerChampionExporter.build_champion_arguments(tmp_path, seed=42)

    assert arguments.learning_rate == 2e-5
    assert arguments.per_device_train_batch_size == 16
    assert arguments.per_device_eval_batch_size == 32
    assert arguments.num_train_epochs == 4
    assert arguments.weight_decay == 0.01
    assert arguments.warmup_ratio == 0.10
    assert arguments.seed == 42
    assert arguments.data_seed == 42
    # Fitting all development rows leaves no legitimate validation split and
    # no epoch checkpoint to select from: selection-free on purpose.
    assert arguments.eval_strategy.value == "no"
    assert arguments.save_strategy.value == "no"
    assert arguments.load_best_model_at_end is False
    assert arguments.save_total_limit is None


def test_export_trains_once_on_all_development_rows_and_excludes_frozen(
    tmp_path: Path,
) -> None:
    harness = _Harness()
    _, _ = harness.export(tmp_path, tmp_path / "dataset.csv", tmp_path / "work")

    assert len(harness.trainers) == 1
    dataset = harness.trainers[0].train_dataset
    expected = [f"raw-{label}-{index}" for label in _LABELS for index in range(5)]
    assert dataset.texts == expected
    assert dataset.labels == [0] * 5 + [1] * 5 + [2] * 5
    assert "raw-frozen" not in dataset.texts
    assert harness.seeds == [42]
    assert harness.load_revisions == ["a" * 40, "a" * 40, "a" * 40]


def test_export_writes_payload_then_manifest_and_round_trip_verifies(
    tmp_path: Path,
) -> None:
    harness = _Harness()
    saved, artifact_dir = harness.export(tmp_path, tmp_path / "dataset.csv", tmp_path / "work")

    assert harness.tokenizer.saved == [artifact_dir]
    assert harness.models[0].saved == [artifact_dir]
    # manifest.json is written last and never counted as payload.
    assert saved.payload_files == (
        artifact_dir / "model.safetensors",
        artifact_dir / "tokenizer.json",
    )
    manifest = ArtifactRegistry().verify(artifact_dir)
    assert manifest == saved.manifest
    assert manifest.schema_version == "1"
    assert manifest.model_name == "bamibert-finetuned"
    assert manifest.model_version == "a" * 40
    assert manifest.backend == "transformer"
    assert [label.value for label in manifest.labels] == _LABELS
    assert manifest.max_input_tokens == 14  # roberta: 16 positions - pad 1 - 1
    assert manifest.preprocessor.name == "bamibert-autotokenizer (raw view, no normalization)"
    assert manifest.preprocessor.version == "a" * 40
    assert manifest.preprocessor.config_checksum == f"sha256:{'b' * 64}"
    assert manifest.calibration.method == "none (raw softmax)"
    assert "not calibrated" in manifest.calibration.fitted_on
    data_bytes = (tmp_path / "dataset.csv").read_bytes()
    assert manifest.data_checksum == f"sha256:{hashlib.sha256(data_bytes).hexdigest()}"
    assert "qualcomm-responsible-ai-license" in manifest.license
    assert "coursework" in manifest.license
    assert manifest.evaluation_report == "reports/transformer-champion.md"
    assert re.fullmatch(r"[0-9a-f]{40}", manifest.git_revision)


def test_export_removes_work_dir_and_leaves_no_checkpoint_junk(tmp_path: Path) -> None:
    harness = _Harness()
    work_dir = tmp_path / "work"
    _, _ = harness.export(tmp_path, tmp_path / "dataset.csv", work_dir)

    assert not work_dir.exists()
    # The exported payload is exactly the two save_pretrained outputs.
    assert sorted(path.name for path in (tmp_path / "champion").iterdir()) == [
        "manifest.json",
        "model.safetensors",
        "tokenizer.json",
    ]
