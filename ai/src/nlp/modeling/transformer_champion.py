import json
import shutil
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch as _torch
import transformers as _transformers
from transformers import AutoConfig as _AutoConfig
from transformers import AutoModelForSequenceClassification as _AutoModelForSequenceClassification
from transformers import AutoTokenizer as _AutoTokenizer
from transformers import DataCollatorWithPadding as _DataCollatorWithPadding
from transformers import (
    TrainingArguments as _TrainingArguments,
)
from transformers import (
    set_seed as _set_seed,
)

from src.domain.artifacts import ResolvedModelSource as _ResolvedModelSource
from src.domain.entities import ArtifactManifest as _ArtifactManifest
from src.domain.entities import CalibrationManifest as _CalibrationManifest
from src.domain.entities import DatasetRow as _DatasetRow
from src.domain.entities import PreprocessorManifest as _PreprocessorManifest
from src.domain.entities import SentimentLabel as _SentimentLabel
from src.domain.exceptions import (
    ModelLoadingProvenanceError as _ModelLoadingProvenanceError,
)
from src.domain.training import SplitManifest as _SplitManifest
from src.nlp.constants import ID2LABEL, LABEL2ID, LABELS, MANIFEST_NAME
from src.nlp.modeling.audited_trainer import AuditedTrainer as _AuditedTrainer
from src.nlp.modeling.registry import ArtifactRegistry as _ArtifactRegistry
from src.nlp.modeling.saved_artifact import SavedArtifact as _SavedArtifact
from src.nlp.modeling.transformer import (
    ConfigLoader as _ConfigLoader,
)
from src.nlp.modeling.transformer import (
    PinnedModelLoader as _PinnedModelLoader,
)
from src.nlp.modeling.transformer import (
    TokenizedDatasetBuilder as _TokenizedDatasetBuilder,
)
from src.nlp.modeling.transformer import (
    TokenizerLoader as _TokenizerLoader,
)
from src.nlp.modeling.transformer import TransformerTrainer as _TransformerTrainer

_MODEL_NAME = "bamibert-finetuned"
_PREPROCESSOR_NAME = "bamibert-autotokenizer (raw view, no normalization)"
_CALIBRATION_METHOD = "none (raw softmax)"
_CALIBRATION_FITTED_ON = (
    "not calibrated; raw softmax of the fine-tuned classifier head served directly"
)
_LICENSE = (
    "bsd-3-clause-clear, other (qualcomm-responsible-ai-license); "
    "research/educational intended use; coursework submission, not production deployment"
)
_EVALUATION_REPORT_PATH = "reports/transformer-champion.md"


class TransformerChampionExporter:
    """Fine-tune BamiBERT on all development rows and export a serving artifact.

    The champion protocol mirrors the baseline champion: fit the selected
    configuration on the *entire* development set with no epoch selection
    (``eval_strategy="no"``, ``save_strategy="no"`` — there is no legitimate
    validation split left once selection is done) and never touch the frozen
    test set. Collaborators are injected exactly like ``TransformerTrainer``
    so tests drive the export without downloading models.
    """

    def __init__(
        self,
        *,
        tokenizer_loader: _TokenizerLoader | None = None,
        config_loader: _ConfigLoader | None = None,
        model_loader: _PinnedModelLoader | None = None,
        dataset_builder: _TokenizedDatasetBuilder | None = None,
        collator_factory: Callable[..., Any] | None = None,
        trainer_factory: Callable[..., Any] | None = None,
        seed_initializer: Callable[[int], None] | None = None,
        device_probe: Callable[[], bool] | None = None,
    ) -> None:
        self._tokenizer_loader: _TokenizerLoader = tokenizer_loader or (
            lambda repo_id, revision: _AutoTokenizer.from_pretrained(repo_id, revision=revision)
        )
        self._config_loader: _ConfigLoader = config_loader or (
            lambda repo_id, revision: _AutoConfig.from_pretrained(repo_id, revision=revision)
        )
        self._model_loader: _PinnedModelLoader = model_loader or self._load_pinned_classifier
        self._dataset_builder: _TokenizedDatasetBuilder = (
            dataset_builder or _TransformerTrainer._tokenize_dataset
        )
        self._collator_factory: Callable[..., Any] = collator_factory or (
            lambda **kwargs: _DataCollatorWithPadding(**kwargs)
        )
        self._trainer_factory: Callable[..., Any] = trainer_factory or (
            lambda **kwargs: _AuditedTrainer(
                loading_validator=lambda info: _TransformerTrainer.validate_loading_info(
                    info, phase="checkpoint"
                ),
                **kwargs,
            )
        )
        self._seed_initializer: Callable[[int], None] = seed_initializer or _set_seed
        self._device_probe: Callable[[], bool] = device_probe or _torch.cuda.is_available

    def _load_pinned_classifier(
        self, source: _ResolvedModelSource
    ) -> tuple[Any, dict[str, list[str]]]:
        loaded = _AutoModelForSequenceClassification.from_pretrained(
            source.repo_id,
            revision=source.revision,
            num_labels=len(LABEL2ID),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            output_loading_info=True,
        )
        if not isinstance(loaded, tuple) or len(loaded) != 2:
            raise _ModelLoadingProvenanceError("Source loader did not return loading diagnostics")
        model, loading_info = loaded
        if not isinstance(loading_info, Mapping):
            raise _ModelLoadingProvenanceError("Source loading diagnostics are not a mapping")
        return model, _TransformerTrainer.validate_loading_info(loading_info, phase="source")

    @staticmethod
    def build_champion_arguments(output_dir: Path, seed: int) -> _TrainingArguments:
        """Fold hyperparameters without epoch selection or checkpoint saving."""
        return _TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=4,
            weight_decay=0.01,
            warmup_ratio=0.10,
            eval_strategy="no",
            save_strategy="no",
            load_best_model_at_end=False,
            seed=seed,
            data_seed=seed,
            fp16=_torch.cuda.is_available(),
            report_to=[],
        )

    def export(
        self,
        *,
        rows: list[_DatasetRow],
        manifest: _SplitManifest,
        source: _ResolvedModelSource,
        data_path: Path,
        artifact_dir: Path,
        work_dir: Path,
        model_family: str = "bamibert",
    ) -> _SavedArtifact:
        """Train once on every development row, then save payload-then-manifest."""
        development_rows = _TransformerTrainer._development_rows(rows, manifest)
        _TransformerTrainer.assert_compatible(source, installed_version=_transformers.__version__)
        tokenizer = self._tokenizer_loader(source.repo_id, source.revision)
        config = self._config_loader(source.repo_id, source.revision)
        texts = [
            _TransformerTrainer.prepare_text(
                development_rows[row_id].raw_text, model_family=model_family
            )
            for row_id in manifest.development_ids
        ]
        labels = [
            LABEL2ID[development_rows[row_id].label.value] for row_id in manifest.development_ids
        ]
        _TransformerTrainer.audit_token_lengths(
            tokenizer, texts, config_limit=_TransformerTrainer.usable_sequence_limit(config)
        )
        train_dataset = self._dataset_builder(tokenizer, texts, labels)
        collator = self._collator_factory(tokenizer=tokenizer)
        self._seed_initializer(manifest.seed)
        if self._device_probe():
            _torch.cuda.empty_cache()
        model, _source_loading = self._model_loader(source)
        try:
            trainer = self._trainer_factory(
                model=model,
                source_revision=source.revision,
                args=self.build_champion_arguments(work_dir, seed=manifest.seed),
                train_dataset=train_dataset,
                data_collator=collator,
            )
            try:
                trainer.train()
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)
        finally:
            if self._device_probe():
                _torch.cuda.empty_cache()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(artifact_dir)
        model.save_pretrained(artifact_dir)
        payload_files = _ArtifactRegistry.payload_files(artifact_dir)
        champion_manifest = _ArtifactManifest(
            schema_version="1",
            model_name=_MODEL_NAME,
            model_version=source.revision,
            backend="transformer",
            labels=[_SentimentLabel(label) for label in LABELS],
            preprocessor=_PreprocessorManifest(
                name=_PREPROCESSOR_NAME,
                version=source.revision,
                config_checksum=f"sha256:{source.model_card_checksum}",
            ),
            calibration=_CalibrationManifest(
                method=_CALIBRATION_METHOD,
                fitted_on=_CALIBRATION_FITTED_ON,
            ),
            data_checksum=f"sha256:{_TransformerTrainer._checksum(data_path)}",
            payload_checksum=_ArtifactRegistry.payload_checksum(payload_files, base=artifact_dir),
            max_input_tokens=_TransformerTrainer.usable_sequence_limit(config),
            git_revision=self._git_revision(),
            license=_LICENSE,
            evaluation_report=_EVALUATION_REPORT_PATH,
        )
        (artifact_dir / MANIFEST_NAME).write_text(
            json.dumps(
                champion_manifest.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return _SavedArtifact(
            directory=artifact_dir,
            manifest=champion_manifest,
            payload_files=_ArtifactRegistry.payload_files(artifact_dir),
        )

    @staticmethod
    def _git_revision() -> str:
        repository = Path(__file__).resolve().parents[4]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        )
        revision = result.stdout.strip()
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ValueError(f"Unexpected git revision: {revision!r}")
        return revision
