import json
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.domain.artifacts import ResolvedModelSource
from src.domain.entities import DatasetRow
from src.domain.exceptions import (
    ModelLoadingProvenanceError,
    ProbabilityValidationError,
    TokenLimitExceededError,
    TransformersVersionIncompatibleError,
)
from src.domain.training import FoldManifest, SplitManifest
from src.nlp.evaluation.behavioral import ChallengeCase
from src.nlp.modeling.audited_trainer import AuditedTrainer
from src.nlp.modeling.transformer import TransformerTrainer

AI_ROOT = Path(__file__).parents[3]


def test_transformer_extra_and_lock_match_bamibert_card_compatibility() -> None:
    project = tomllib.loads((AI_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((AI_ROOT / "uv.lock").read_text(encoding="utf-8"))
    transformer_extra = project["project"]["optional-dependencies"]["transformer"]
    locked_versions = {package["name"]: package["version"] for package in lock["package"]}

    assert "transformers>=4.49,<=5.5.0" in transformer_extra
    assert locked_versions["transformers"] == "5.5.0"
    assert locked_versions["tokenizers"] == "0.22.2"


def test_bamibert_adapter_preserves_social_signal() -> None:
    text = "Xe nhà H. chạy cx ổn :))"
    assert TransformerTrainer.prepare_text(text, model_family="bamibert") == text


def test_adapter_rejects_unknown_model_family() -> None:
    with pytest.raises(ValueError, match="unsupported model family"):
        TransformerTrainer.prepare_text("text", model_family="unknown")


def test_training_arguments_match_pinned_experiment_hyperparameters(tmp_path: Path) -> None:
    arguments = TransformerTrainer.build_training_arguments(tmp_path, seed=42)

    assert arguments.learning_rate == 2e-5
    assert arguments.per_device_train_batch_size == 16
    assert arguments.per_device_eval_batch_size == 32
    assert arguments.num_train_epochs == 4
    assert arguments.weight_decay == 0.01
    assert arguments.warmup_ratio == 0.10
    assert arguments.get_warmup_steps(100) == 10
    assert arguments.eval_strategy.value == "epoch"
    assert arguments.save_strategy.value == "epoch"
    assert arguments.load_best_model_at_end is True
    assert arguments.metric_for_best_model == "eval_macro_f1"
    assert arguments.save_total_limit == 1
    assert arguments.seed == 42
    assert arguments.data_seed == 42
    assert arguments.report_to == []


def test_token_audit_fails_when_raw_text_exceeds_declared_limit() -> None:
    class FakeTokenizer:
        model_max_length = 4

        def __call__(
            self, text: str, *, truncation: bool, add_special_tokens: bool
        ) -> dict[str, list[int]]:
            assert truncation is False
            assert add_special_tokens is True
            return {"input_ids": list(range(len(text.split())))}

    with pytest.raises(TokenLimitExceededError, match="exceeds declared model limit"):
        TransformerTrainer.audit_token_lengths(
            FakeTokenizer(), ["one two three four five"], config_limit=8
        )


def test_roberta_token_audit_uses_limit_after_padding_position_offset() -> None:
    class FakeConfig:
        model_type = "roberta"
        max_position_embeddings = 2050
        pad_token_id = 1

    class BoundaryTokenizer:
        model_max_length = 10**30

        def __call__(
            self, text: str, *, truncation: bool, add_special_tokens: bool
        ) -> dict[str, list[int]]:
            return {"input_ids": list(range(int(text)))}

    limit = TransformerTrainer.usable_sequence_limit(FakeConfig())

    assert limit == 2048
    audit = TransformerTrainer.audit_token_lengths(
        BoundaryTokenizer(), ["2048"], config_limit=limit
    )
    assert audit.model_limit == 2048
    with pytest.raises(TokenLimitExceededError, match="maximum_observed_tokens=2049"):
        TransformerTrainer.audit_token_lengths(BoundaryTokenizer(), ["2049"], config_limit=limit)


def test_loader_gate_allows_only_documented_source_head_differences() -> None:
    source_info = {
        "missing_keys": {
            "classifier.dense.bias",
            "classifier.dense.weight",
            "classifier.out_proj.bias",
            "classifier.out_proj.weight",
        },
        "unexpected_keys": {
            "lm_head.bias",
            "lm_head.decoder.bias",
            "lm_head.dense.bias",
            "lm_head.dense.weight",
            "lm_head.layer_norm.bias",
            "lm_head.layer_norm.weight",
        },
        "mismatched_keys": [],
        "error_msgs": [],
    }

    diagnostics = TransformerTrainer.validate_loading_info(source_info, phase="source")

    assert diagnostics["missing_keys"] == sorted(source_info["missing_keys"])
    source_info["missing_keys"] = {"roberta.encoder.layer.0.output.LayerNorm.weight"}
    with pytest.raises(ModelLoadingProvenanceError, match="backbone"):
        TransformerTrainer.validate_loading_info(source_info, phase="source")


def test_audited_trainer_reloads_checkpoint_with_strict_backbone_diagnostics(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeModel:
        def from_pretrained(
            self, path: str, **kwargs: object
        ) -> tuple[object, dict[str, list[str]]]:
            calls.append((path, kwargs))
            return object(), {
                "missing_keys": [],
                "unexpected_keys": [],
                "mismatched_keys": [],
                "error_msgs": [],
            }

        def load_state_dict(self, state: object, *, strict: bool) -> None:
            calls.append(("load_state_dict", {"state": state, "strict": strict}))

    class CheckpointModel:
        def state_dict(self) -> dict[str, str]:
            return {"backbone": "preserved"}

    model = FakeModel()
    trainer = object.__new__(AuditedTrainer)
    trainer.state = SimpleNamespace(best_model_checkpoint=str(checkpoint))
    trainer.model = model
    trainer.source_revision = "a" * 40
    trainer._loading_validator = lambda info: TransformerTrainer.validate_loading_info(
        info, phase="checkpoint"
    )

    # Substitute the checkpoint model only after the source loader call is observed.
    def load_checkpoint(
        path: str, **kwargs: object
    ) -> tuple[CheckpointModel, dict[str, list[str]]]:
        calls.append((path, kwargs))
        return CheckpointModel(), {
            "missing_keys": [],
            "unexpected_keys": [],
            "mismatched_keys": [],
            "error_msgs": [],
        }

    model.from_pretrained = load_checkpoint  # type: ignore[method-assign]
    trainer._load_best_model()

    assert calls[0] == (
        str(checkpoint),
        {
            "revision": "a" * 40,
            "output_loading_info": True,
            "local_files_only": True,
        },
    )
    assert calls[1] == (
        "load_state_dict",
        {"state": {"backbone": "preserved"}, "strict": True},
    )
    assert trainer.best_checkpoint_diagnostics == {
        "missing_keys": [],
        "unexpected_keys": [],
        "mismatched_keys": [],
        "error_msgs": [],
    }


def test_probability_audit_rejects_non_finite_out_of_range_and_non_unit_rows() -> None:
    valid = pd.DataFrame(
        {
            "negative": [0.2],
            "neutral": [0.3],
            "positive": [0.5],
        }
    )
    assert TransformerTrainer._probability_audit(valid)["maximum_sum_error"] == 0.0
    for invalid in (
        pd.DataFrame({"negative": [np.nan], "neutral": [0.3], "positive": [0.7]}),
        pd.DataFrame({"negative": [-0.1], "neutral": [0.3], "positive": [0.8]}),
        pd.DataFrame({"negative": [0.2], "neutral": [0.3], "positive": [0.6]}),
    ):
        with pytest.raises(ProbabilityValidationError):
            TransformerTrainer._probability_audit(invalid)


def test_challenge_texts_excludes_empty_paired_text() -> None:
    cases = [
        ChallengeCase(
            id="mft",
            kind="MFT",
            text="actual challenge",
            paired_text="",
            expected_label="positive",
            paired_expected_label="",
            minimum_delta=0.0,
        )
    ]

    assert TransformerTrainer._challenge_texts(cases) == ["actual challenge"]


def test_transformer_preflight_rejects_installed_version_outside_model_card_requirement() -> None:
    source = ResolvedModelSource(
        repo_id="Qualcomm-AI-Research/BamiBERT",
        revision="a" * 40,
        license_id="bsd-3-clause-clear, other",
        intended_use="The model is intended for research and educational purposes.",
        transformers_requirement="<=5.5.0",
    )

    with pytest.raises(TransformersVersionIncompatibleError, match=r"transformers<=5\.5\.0"):
        TransformerTrainer.assert_compatible(source, installed_version="5.17.0")


def test_failure_evidence_records_blocked_run_without_inventing_predictions(tmp_path: Path) -> None:
    source = ResolvedModelSource(
        repo_id="Qualcomm-AI-Research/BamiBERT",
        revision="a" * 40,
        license_id="bsd-3-clause-clear, other",
        intended_use="The model is intended for research and educational purposes.",
        transformers_requirement="<=5.5.0",
    )
    data_path = tmp_path / "dataset.csv"
    splits_path = tmp_path / "split_manifest.json"
    data_path.write_text("dataset", encoding="utf-8")
    splits_path.write_text("splits", encoding="utf-8")
    output_dir = tmp_path / "runs" / "transformer"
    report_path = tmp_path / "reports" / "transformer-evaluation.md"
    error = TransformersVersionIncompatibleError("Pinned model card requires transformers<=5.5.0")
    output_dir.mkdir(parents=True)
    for name in ("cv_predictions.csv", "challenge_predictions.json", "token-limit-failure.md"):
        (output_dir / name).write_text("stale", encoding="utf-8")
    report_path.parent.mkdir(parents=True)
    report_path.write_text("stale", encoding="utf-8")
    report_path.with_suffix(".json").write_text("stale", encoding="utf-8")

    TransformerTrainer.write_failure_evidence(
        output_dir=output_dir,
        report_path=report_path,
        data_path=data_path,
        splits_path=splits_path,
        source=source,
        error=error,
        installed_transformers_version="5.17.0",
    )

    metrics = json.loads((output_dir / "cv_metrics.json").read_text(encoding="utf-8"))
    assert metrics["status"] == "blocked"
    assert metrics["oof_predictions_generated"] is False
    assert metrics["error_type"] == "TransformersVersionIncompatibleError"
    assert metrics["model_source"]["revision"] == "a" * 40
    assert not (output_dir / "cv_predictions.csv").exists()
    assert not (output_dir / "challenge_predictions.json").exists()
    assert not (output_dir / "token-limit-failure.md").exists()
    markdown = report_path.read_text(encoding="utf-8")
    assert "No OOF or challenge predictions were generated" in markdown
    assert "transformers=5.17.0" in markdown
    assert not report_path.with_suffix(".json").exists()


def test_cross_validation_uses_raw_text_fresh_pinned_models_and_cleans_checkpoints(
    tmp_path: Path,
) -> None:
    labels = ["negative", "neutral", "positive"]
    development_rows = [
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
        for label in labels
        for index in range(5)
    ]
    frozen_row = DatasetRow(
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
    development_ids = [row.id for row in development_rows]
    folds = [
        FoldManifest(
            fold=fold,
            validation_ids=[f"{label}-{fold}" for label in labels],
            train_ids=[
                row_id
                for row_id in development_ids
                if row_id not in {f"{label}-{fold}" for label in labels}
            ],
        )
        for fold in range(5)
    ]
    manifest = SplitManifest(
        seed=42,
        dataset_checksum="frozen",
        development_ids=development_ids,
        test_ids=[frozen_row.id],
        folds=folds,
    )
    source = ResolvedModelSource(
        repo_id="Qualcomm-AI-Research/BamiBERT",
        revision="a" * 40,
        license_id="bsd-3-clause-clear",
        intended_use="research and educational purposes",
    )
    load_revisions: list[str] = []
    models: list[object] = []
    training_texts: list[list[str]] = []

    class FakeTokenizer:
        model_max_length = 128

        def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
            return {"input_ids": [1]}

    class FakeConfig:
        model_type = "roberta"
        max_position_embeddings = 16
        pad_token_id = 1

    class FakeDataset:
        def __init__(self, texts: list[str], labels_: list[int] | None) -> None:
            self.texts = texts
            self.labels = labels_

    def fake_tokenizer_loader(repo_id: str, revision: str) -> FakeTokenizer:
        assert repo_id == "Qualcomm-AI-Research/BamiBERT"
        load_revisions.append(revision)
        return FakeTokenizer()

    def fake_config_loader(repo_id: str, revision: str) -> FakeConfig:
        assert repo_id == "Qualcomm-AI-Research/BamiBERT"
        load_revisions.append(revision)
        return FakeConfig()

    def fake_model_loader(
        resolved: ResolvedModelSource,
    ) -> tuple[object, dict[str, list[str]]]:
        load_revisions.append(resolved.revision)
        model = object()
        models.append(model)
        return model, {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}

    def fake_dataset_builder(
        tokenizer: object, texts: list[str], labels_: list[int] | None
    ) -> FakeDataset:
        return FakeDataset(texts, labels_)

    class FakeTrainer:
        def __init__(self, **kwargs: object) -> None:
            self.args = kwargs["args"]
            self.train_dataset = kwargs["train_dataset"]
            self.best_checkpoint_diagnostics = {"missing_keys": [], "unexpected_keys": []}
            training_texts.append(self.train_dataset.texts)

        def train(self) -> None:
            checkpoint = Path(self.args.output_dir) / "checkpoint-1"
            checkpoint.mkdir(parents=True)
            (checkpoint / "payload").write_text("temporary", encoding="utf-8")

        def predict(self, dataset: FakeDataset) -> SimpleNamespace:
            if dataset.labels is None:
                return SimpleNamespace(predictions=np.zeros((len(dataset.texts), 3)))
            logits = np.full((len(dataset.labels), 3), -10.0)
            for index, label in enumerate(dataset.labels):
                logits[index, label] = 10.0
            return SimpleNamespace(predictions=logits)

    trainer = TransformerTrainer(
        tokenizer_loader=fake_tokenizer_loader,
        config_loader=fake_config_loader,
        model_loader=fake_model_loader,
        dataset_builder=fake_dataset_builder,
        collator_factory=lambda **_: object(),
        trainer_factory=FakeTrainer,
        seed_initializer=lambda _seed: None,
        device_probe=lambda: False,
    )

    predictions, _, _, challenge_scores = trainer.cross_validate(
        rows=[*development_rows, frozen_row],
        manifest=manifest,
        source=source,
        output_dir=tmp_path / "transformer",
        challenge_cases=[
            ChallengeCase(
                id="challenge",
                kind="MFT",
                text="challenge text",
                paired_text="",
                expected_label="positive",
                paired_expected_label="",
                minimum_delta=0.0,
            )
        ],
    )

    assert load_revisions == ["a" * 40] * 7
    assert len(models) == len({id(model) for model in models}) == 5
    assert all(text.startswith("raw-") for texts in training_texts for text in texts)
    assert all("normalized-" not in text for texts in training_texts for text in texts)
    assert frozen_row.id not in set(predictions["id"])
    assert len(predictions) == len(development_rows)
    assert set(challenge_scores) == {"challenge text"}
    assert not (tmp_path / "transformer" / ".checkpoints").exists()
