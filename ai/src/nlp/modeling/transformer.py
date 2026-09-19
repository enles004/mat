import hashlib
import json
import shutil
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, cast

import numpy as _np
import numpy.typing as _npt
import pandas as _pd  # type: ignore[import-untyped]
import torch as _torch
import transformers as _transformers
from datasets import Dataset as _Dataset  # type: ignore[import-untyped]
from packaging.specifiers import SpecifierSet as _SpecifierSet
from packaging.version import Version as _Version
from sklearn.metrics import f1_score as _f1_score  # type: ignore[import-untyped]
from transformers import (
    AutoConfig as _AutoConfig,
)
from transformers import (
    AutoModelForSequenceClassification as _AutoModelForSequenceClassification,
)
from transformers import (
    AutoTokenizer as _AutoTokenizer,
)
from transformers import (
    DataCollatorWithPadding as _DataCollatorWithPadding,
)
from transformers import (
    EvalPrediction as _EvalPrediction,
)
from transformers import (
    TrainingArguments as _TrainingArguments,
)
from transformers import (
    set_seed as _set_seed,
)

from src.domain.artifacts import ResolvedModelSource as _ResolvedModelSource
from src.domain.entities import ChallengeCase as _ChallengeCase
from src.domain.entities import DatasetRow as _DatasetRow
from src.domain.entities import ReviewStatus as _ReviewStatus
from src.domain.exceptions import (
    ModelLoadingProvenanceError as _ModelLoadingProvenanceError,
)
from src.domain.exceptions import ProbabilityValidationError as _ProbabilityValidationError
from src.domain.exceptions import TokenLimitExceededError as _TokenLimitExceededError
from src.domain.exceptions import (
    TransformersVersionIncompatibleError as _TransformersVersionIncompatibleError,
)
from src.domain.training import FoldMeasurement as _FoldMeasurement
from src.domain.training import SplitManifest as _SplitManifest
from src.domain.training import TokenLengthAudit as _TokenLengthAudit
from src.nlp.constants import ID2LABEL, LABEL2ID, LABEL_ORDER, OOF_COLUMNS
from src.nlp.modeling.audited_trainer import AuditedTrainer as _AuditedTrainer

TokenizerLoader = Callable[[str, str], Any]
ConfigLoader = Callable[[str, str], Any]
PinnedModelLoader = Callable[[_ResolvedModelSource], tuple[Any, dict[str, list[str]]]]
TokenizedDatasetBuilder = Callable[[Any, list[str], list[int] | None], _Dataset]


class TransformerTrainer:
    """Cross-validate the revision-pinned transformer candidate on development folds.

    Training collaborators (tokenizer/config/model loading, dataset building,
    trainer construction, seeding, and device probing) are injected so tests can
    drive the full fold loop without downloading models; defaults bind the
    pinned Hugging Face loaders.
    """

    def __init__(
        self,
        *,
        tokenizer_loader: TokenizerLoader | None = None,
        config_loader: ConfigLoader | None = None,
        model_loader: PinnedModelLoader | None = None,
        dataset_builder: TokenizedDatasetBuilder | None = None,
        collator_factory: Callable[..., Any] | None = None,
        trainer_factory: Callable[..., Any] | None = None,
        seed_initializer: Callable[[int], None] | None = None,
        device_probe: Callable[[], bool] | None = None,
    ) -> None:
        self._tokenizer_loader: TokenizerLoader = tokenizer_loader or self._load_tokenizer
        self._config_loader: ConfigLoader = config_loader or self._load_config
        self._model_loader: PinnedModelLoader = model_loader or self._load_pinned_classifier
        self._dataset_builder: TokenizedDatasetBuilder = (
            dataset_builder or TransformerTrainer._tokenize_dataset
        )
        self._collator_factory: Callable[..., Any] = collator_factory or (
            lambda **kwargs: _DataCollatorWithPadding(**kwargs)
        )
        self._trainer_factory: Callable[..., Any] = trainer_factory or (
            lambda **kwargs: _AuditedTrainer(
                loading_validator=lambda info: self.validate_loading_info(info, phase="checkpoint"),
                **kwargs,
            )
        )
        self._seed_initializer: Callable[[int], None] = seed_initializer or _set_seed
        self._device_probe: Callable[[], bool] = device_probe or _torch.cuda.is_available

    @staticmethod
    def _loading_key_list(info: Mapping[str, object], name: str) -> list[str]:
        raw_keys = info.get(name, [])
        if not isinstance(raw_keys, Sequence | set) or isinstance(raw_keys, str):
            raise _ModelLoadingProvenanceError(f"Loader diagnostics {name} must be a key sequence")
        return sorted(str(key) for key in raw_keys)

    @staticmethod
    def _tokenize_dataset(tokenizer: Any, texts: list[str], labels: list[int] | None) -> _Dataset:
        payload: dict[str, list[str] | list[int]] = {"text": texts}
        if labels is not None:
            payload["labels"] = labels
        dataset = _Dataset.from_dict(payload)
        return dataset.map(
            lambda batch: tokenizer(batch["text"], truncation=False),
            batched=True,
            remove_columns=["text"],
        )

    @staticmethod
    def _checksum(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _remove_checkpoints(path: Path) -> None:
        if path.exists():
            shutil.rmtree(path)

    @staticmethod
    def _load_tokenizer(repo_id: str, revision: str) -> Any:
        return _AutoTokenizer.from_pretrained(repo_id, revision=revision)

    @staticmethod
    def _load_config(repo_id: str, revision: str) -> Any:
        return _AutoConfig.from_pretrained(repo_id, revision=revision)

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
        return model, self.validate_loading_info(loading_info, phase="source")

    @staticmethod
    def prepare_text(text: str, model_family: str) -> str:
        """Keep BamiBERT-compatible social signal unchanged."""
        if model_family in {"bamibert", "visobert"}:
            return text
        raise ValueError(f"unsupported model family: {model_family}")

    @staticmethod
    def assert_compatible(source: _ResolvedModelSource, installed_version: str) -> None:
        """Fail before tokenizer/model loading when the model card excludes this version."""
        if not source.transformers_requirement:
            return
        requirement = _SpecifierSet(source.transformers_requirement)
        if _Version(installed_version) not in requirement:
            raise _TransformersVersionIncompatibleError(
                f"Pinned model card requires transformers{source.transformers_requirement}; "
                f"installed transformers={installed_version}"
            )

    @staticmethod
    def build_training_arguments(output_dir: Path, seed: int) -> _TrainingArguments:
        """Return the fixed, reproducible training configuration."""
        return _TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            num_train_epochs=4,
            weight_decay=0.01,
            warmup_ratio=0.10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
            save_total_limit=1,
            seed=seed,
            data_seed=seed,
            fp16=_torch.cuda.is_available(),
            report_to=[],
        )

    @staticmethod
    def compute_trainer_metrics(evaluation: _EvalPrediction) -> dict[str, float]:
        """Calculate the macro-F1 used to select the epoch checkpoint."""
        predictions = evaluation.predictions
        logits = predictions[0] if isinstance(predictions, tuple) else predictions
        predicted = _np.asarray(logits).argmax(axis=1)
        return {"macro_f1": float(_f1_score(evaluation.label_ids, predicted, average="macro"))}

    @staticmethod
    def validate_loading_info(
        info: Mapping[str, object], *, phase: Literal["source", "checkpoint"]
    ) -> dict[str, list[str]]:
        """Validate model-load diagnostics, allowing only documented source head differences."""
        diagnostics = {
            "missing_keys": TransformerTrainer._loading_key_list(info, "missing_keys"),
            "unexpected_keys": TransformerTrainer._loading_key_list(info, "unexpected_keys"),
            "mismatched_keys": TransformerTrainer._loading_key_list(info, "mismatched_keys"),
            "error_msgs": TransformerTrainer._loading_key_list(info, "error_msgs"),
        }
        if diagnostics["mismatched_keys"] or diagnostics["error_msgs"]:
            raise _ModelLoadingProvenanceError(
                f"{phase} loader has mismatched keys or errors: {diagnostics}"
            )
        if phase == "source":
            bad_missing = [
                key for key in diagnostics["missing_keys"] if not key.startswith("classifier.")
            ]
            bad_unexpected = [
                key for key in diagnostics["unexpected_keys"] if not key.startswith("lm_head.")
            ]
            if bad_missing or bad_unexpected:
                raise _ModelLoadingProvenanceError(
                    f"source loader has backbone missing/unexpected keys: "
                    f"missing={bad_missing}, unexpected={bad_unexpected}"
                )
        elif diagnostics["missing_keys"] or diagnostics["unexpected_keys"]:
            raise _ModelLoadingProvenanceError(
                f"checkpoint loader has backbone missing/unexpected keys: {diagnostics}"
            )
        return diagnostics

    @staticmethod
    def _declared_limit(value: object) -> int | None:
        if isinstance(value, int) and 0 < value < 1_000_000:
            return value
        return None

    @staticmethod
    def usable_sequence_limit(config: object) -> int:
        """Return the effective non-padding sequence limit for the resolved architecture."""
        declared_limit = TransformerTrainer._declared_limit(
            getattr(config, "max_position_embeddings", None)
        )
        if declared_limit is None:
            raise ValueError("Resolved model does not declare max_position_embeddings")
        model_type = getattr(config, "model_type", "")
        padding_index = getattr(config, "pad_token_id", None)
        if model_type in {"roberta", "xlm-roberta", "camembert"}:
            if not isinstance(padding_index, int):
                raise ValueError(
                    "RoBERTa-compatible model does not declare an integer pad_token_id"
                )
            usable_limit = declared_limit - padding_index - 1
            if usable_limit <= 0:
                raise ValueError("RoBERTa-compatible model has no usable position embeddings")
            return usable_limit
        return declared_limit

    @staticmethod
    def audit_token_lengths(
        tokenizer: Any, texts: list[str], config_limit: int | None
    ) -> _TokenLengthAudit:
        """Inspect token lengths without truncation and fail on any overflow."""
        tokenizer_limit = TransformerTrainer._declared_limit(
            getattr(tokenizer, "model_max_length", None)
        )
        possible_limits = [limit for limit in (tokenizer_limit, config_limit) if limit is not None]
        if not possible_limits:
            raise ValueError("Resolved model does not declare a usable token limit")
        model_limit = min(possible_limits)
        lengths = [
            len(tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"])
            for text in texts
        ]
        maximum = max(lengths, default=0)
        audit = _TokenLengthAudit(
            row_count=len(texts),
            maximum_observed_tokens=maximum,
            model_limit=model_limit,
            tokenizer_limit=tokenizer_limit,
            config_limit=config_limit,
        )
        if maximum > model_limit:
            raise _TokenLimitExceededError(
                "Raw development text exceeds declared model limit: "
                f"maximum_observed_tokens={maximum}, model_limit={model_limit}"
            )
        return audit

    @staticmethod
    def _development_rows(
        rows: list[_DatasetRow], manifest: _SplitManifest
    ) -> dict[str, _DatasetRow]:
        development_ids = manifest.development_ids
        test_ids = manifest.test_ids
        if len(development_ids) != len(set(development_ids)):
            raise ValueError("Development IDs must be unique")
        if len(test_ids) != len(set(test_ids)):
            raise ValueError("Frozen test IDs must be unique")
        if not set(development_ids).isdisjoint(test_ids):
            raise ValueError("Development and frozen test IDs must be disjoint")
        development_id_set = set(development_ids)
        development_rows = {row.id: row for row in rows if row.id in development_id_set}
        missing_ids = development_id_set - development_rows.keys()
        if missing_ids:
            raise ValueError(f"Dataset is missing development IDs: {sorted(missing_ids)}")
        if any(
            row.review_status is not _ReviewStatus.APPROVED for row in development_rows.values()
        ):
            raise ValueError("Transformer training requires approved development rows")
        return development_rows

    @staticmethod
    def _softmax_probabilities(predictions: object) -> _npt.NDArray[_np.float64]:
        logits = predictions[0] if isinstance(predictions, tuple) else predictions
        values = _np.asarray(logits, dtype=float)
        shifted = values - values.max(axis=1, keepdims=True)
        numerator = _np.exp(shifted)
        return cast(_npt.NDArray[_np.float64], numerator / numerator.sum(axis=1, keepdims=True))

    @staticmethod
    def _challenge_texts(cases: list[_ChallengeCase]) -> list[str]:
        return list(
            dict.fromkeys(text for case in cases for text in (case.text, case.paired_text) if text)
        )

    @staticmethod
    def _probability_audit(predictions: _pd.DataFrame) -> dict[str, float]:
        probabilities = predictions[LABEL_ORDER].to_numpy(dtype=float)
        if probabilities.size == 0:
            raise _ProbabilityValidationError("OOF probabilities must not be empty")
        if not _np.isfinite(probabilities).all():
            raise _ProbabilityValidationError("OOF probabilities must be finite")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise _ProbabilityValidationError("OOF probabilities must be in [0, 1]")
        sum_error = _np.abs(probabilities.sum(axis=1) - 1.0)
        if not _np.isclose(probabilities.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6).all():
            raise _ProbabilityValidationError("OOF probabilities must sum to one")
        return {
            "minimum_probability": float(probabilities.min()),
            "maximum_probability": float(probabilities.max()),
            "maximum_sum_error": float(sum_error.max()),
        }

    @staticmethod
    def clear_run_artifacts(output_dir: Path, report_path: Path) -> None:
        """Remove prior prediction/failure evidence before a new reproducible attempt."""
        TransformerTrainer._remove_checkpoints(output_dir / ".checkpoints")
        for name in (
            "cv_predictions.csv",
            "challenge_predictions.json",
            "cv_metrics.json",
            "token-limit-failure.md",
        ):
            artifact_path = output_dir / name
            if artifact_path.exists():
                artifact_path.unlink()
        for artifact_path in (report_path, report_path.with_suffix(".json")):
            if artifact_path.exists():
                artifact_path.unlink()

    @staticmethod
    def write_token_limit_failure(output_dir: Path, error: _TokenLimitExceededError) -> None:
        (output_dir / "token-limit-failure.md").write_text(
            "# Transformer token-limit failure\n\n"
            "Tokenization was performed with `truncation=False`; "
            "no rows were silently shortened.\n\n"
            f"{error}\n",
            encoding="utf-8",
        )

    @staticmethod
    def write_failure_evidence(
        output_dir: Path,
        report_path: Path,
        data_path: Path,
        splits_path: Path,
        source: _ResolvedModelSource,
        error: Exception,
        installed_transformers_version: str,
    ) -> None:
        """Write explicit blocked-run evidence without creating fictional predictions."""
        TransformerTrainer.clear_run_artifacts(output_dir, report_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "schema_version": "1",
            "candidate_type": "transformer",
            "status": "blocked",
            "oof_predictions_generated": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "installed_transformers_version": installed_transformers_version,
            "model_source": source.as_dict(),
            "dataset_checksum": TransformerTrainer._checksum(data_path),
            "split_manifest_checksum": TransformerTrainer._checksum(splits_path),
        }
        (output_dir / "cv_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        lines = [
            "# Transformer evaluation blocked",
            "",
            "## Status",
            "",
            "No OOF or challenge predictions were generated; "
            "no metrics were inferred or substituted.",
            "",
            "## Reproducible failure",
            "",
            f"- Repository: `{source.repo_id}`",
            f"- Revision: `{source.revision}`",
            f"- Installed transformers: `transformers={installed_transformers_version}`",
            f"- Model-card requirement: `transformers{source.transformers_requirement}`",
            f"- Exception: `{type(error).__name__}: {error}`",
            "",
            "The run failed before valid prediction evidence could be produced. Re-run only after "
            "the recorded error is addressed.",
            "",
            "## Inputs",
            "",
            f"- Dataset SHA-256: `{metrics['dataset_checksum']}`",
            f"- Split manifest SHA-256: `{metrics['split_manifest_checksum']}`",
            f"- Model-card README SHA-256: `{source.model_card_checksum}`",
        ]
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def write_metrics(
        output_dir: Path,
        data_path: Path,
        splits_path: Path,
        source: _ResolvedModelSource,
        measurements: list[_FoldMeasurement],
        token_audit: _TokenLengthAudit,
        predictions: _pd.DataFrame,
    ) -> None:
        """Persist the complete transformer cross-validation evidence."""
        metrics: dict[str, object] = {
            "schema_version": "1",
            "candidate_type": "transformer",
            "model_family": "bamibert",
            "model_source": source.as_dict(),
            "dataset_checksum": TransformerTrainer._checksum(data_path),
            "split_manifest_checksum": TransformerTrainer._checksum(splits_path),
            "labels": LABEL_ORDER,
            "token_length_audit": asdict(token_audit),
            "folds": [asdict(measurement) for measurement in measurements],
            "mean_macro_f1": sum(measurement.macro_f1 for measurement in measurements)
            / len(measurements),
            "oof_row_count": len(predictions),
            "probability_audit": TransformerTrainer._probability_audit(predictions),
        }
        (output_dir / "cv_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def cross_validate(
        self,
        rows: list[_DatasetRow],
        manifest: _SplitManifest,
        source: _ResolvedModelSource,
        output_dir: Path,
        challenge_cases: list[_ChallengeCase],
        model_family: str = "bamibert",
    ) -> tuple[
        _pd.DataFrame, list[_FoldMeasurement], _TokenLengthAudit, dict[str, dict[str, float]]
    ]:
        """Train one fresh revision-pinned model per persisted validation fold."""
        if [fold.fold for fold in manifest.folds] != list(range(5)):
            raise ValueError("Transformer cross-validation requires five ordered persisted folds")
        development_rows = self._development_rows(rows, manifest)
        development_id_set = set(manifest.development_ids)
        source_revision = source.revision
        self.assert_compatible(source, installed_version=_transformers.__version__)
        tokenizer = self._tokenizer_loader(source.repo_id, source_revision)
        config = self._config_loader(source.repo_id, source_revision)
        raw_development_texts = [
            self.prepare_text(development_rows[row_id].raw_text, model_family=model_family)
            for row_id in manifest.development_ids
        ]
        token_audit = self.audit_token_lengths(
            tokenizer,
            raw_development_texts,
            config_limit=self.usable_sequence_limit(config),
        )
        challenge_texts = self._challenge_texts(challenge_cases)
        challenge_dataset = self._dataset_builder(
            tokenizer,
            [self.prepare_text(text, model_family=model_family) for text in challenge_texts],
            None,
        )
        collator = self._collator_factory(tokenizer=tokenizer)
        frames: list[_pd.DataFrame] = []
        measurements: list[_FoldMeasurement] = []
        challenge_probabilities: list[_npt.NDArray[_np.float64]] = []
        checkpoint_root = output_dir / ".checkpoints"

        try:
            for fold in manifest.folds:
                train_id_set = set(fold.train_ids)
                validation_id_set = set(fold.validation_ids)
                if train_id_set & validation_id_set:
                    raise ValueError(
                        f"Persisted fold {fold.fold} overlaps training and validation IDs"
                    )
                if train_id_set | validation_id_set != development_id_set:
                    raise ValueError(f"Persisted fold {fold.fold} does not cover development IDs")
                if len(train_id_set) != len(fold.train_ids) or len(validation_id_set) != len(
                    fold.validation_ids
                ):
                    raise ValueError(f"Persisted fold {fold.fold} contains duplicate IDs")

                train_rows = [development_rows[row_id] for row_id in fold.train_ids]
                validation_rows = [development_rows[row_id] for row_id in fold.validation_ids]
                train_dataset = self._dataset_builder(
                    tokenizer,
                    [
                        self.prepare_text(row.raw_text, model_family=model_family)
                        for row in train_rows
                    ],
                    [LABEL2ID[row.label.value] for row in train_rows],
                )
                validation_dataset = self._dataset_builder(
                    tokenizer,
                    [
                        self.prepare_text(row.raw_text, model_family=model_family)
                        for row in validation_rows
                    ],
                    [LABEL2ID[row.label.value] for row in validation_rows],
                )
                fold_output = checkpoint_root / f"fold-{fold.fold}"
                self._seed_initializer(manifest.seed)
                if self._device_probe():
                    _torch.cuda.empty_cache()
                    _torch.cuda.reset_peak_memory_stats()
                started_at = time.perf_counter()
                model, source_loading = self._model_loader(source)
                trainer = self._trainer_factory(
                    model=model,
                    source_revision=source.revision,
                    args=self.build_training_arguments(fold_output, seed=manifest.seed),
                    train_dataset=train_dataset,
                    eval_dataset=validation_dataset,
                    data_collator=collator,
                    compute_metrics=self.compute_trainer_metrics,
                )
                try:
                    trainer.train()
                    if trainer.best_checkpoint_diagnostics is None:
                        raise _ModelLoadingProvenanceError(
                            "Trainer did not validate its best checkpoint"
                        )
                    validation_output = trainer.predict(validation_dataset)
                    challenge_output = trainer.predict(challenge_dataset)
                    duration_seconds = time.perf_counter() - started_at
                    peak_allocated_bytes = (
                        int(_torch.cuda.max_memory_allocated()) if self._device_probe() else 0
                    )
                    probabilities = self._softmax_probabilities(validation_output.predictions)
                    challenge_probabilities.append(
                        self._softmax_probabilities(challenge_output.predictions)
                    )
                    labels = [row.label.value for row in validation_rows]
                    predicted = probabilities.argmax(axis=1)
                    measurements.append(
                        _FoldMeasurement(
                            fold=fold.fold,
                            duration_seconds=duration_seconds,
                            peak_allocated_bytes=peak_allocated_bytes,
                            macro_f1=float(
                                _f1_score(
                                    [LABEL2ID[label] for label in labels],
                                    predicted,
                                    average="macro",
                                )
                            ),
                            validation_row_count=len(validation_rows),
                            source_loading=source_loading,
                            best_checkpoint_loading=trainer.best_checkpoint_diagnostics,
                        )
                    )
                    frames.append(
                        _pd.concat(
                            [
                                _pd.DataFrame(
                                    {
                                        "id": fold.validation_ids,
                                        "fold": fold.fold,
                                        "y_true": labels,
                                    }
                                ),
                                _pd.DataFrame(probabilities, columns=LABEL_ORDER),
                            ],
                            axis="columns",
                        )
                    )
                finally:
                    del trainer
                    del model
                    self._remove_checkpoints(fold_output)
                    if self._device_probe():
                        _torch.cuda.empty_cache()
        finally:
            self._remove_checkpoints(checkpoint_root)

        predictions = _pd.concat(frames, ignore_index=True).loc[:, OOF_COLUMNS]
        predictions = predictions.sort_values(["fold", "id"], kind="stable").reset_index(drop=True)
        if Counter(predictions["id"].tolist()) != Counter(manifest.development_ids):
            raise AssertionError("OOF predictions must cover every development ID exactly once")
        self._probability_audit(predictions)
        mean_challenge_probabilities = _np.mean(_np.stack(challenge_probabilities, axis=0), axis=0)
        challenge_scores = {
            text: {
                label: float(mean_challenge_probabilities[index, class_index])
                for class_index, label in enumerate(LABEL_ORDER)
            }
            for index, text in enumerate(challenge_texts)
        }
        return predictions, measurements, token_audit, challenge_scores
