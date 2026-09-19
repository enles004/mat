import csv
import json
from pathlib import Path

import pytest

from scripts import data_validate
from src.domain.entities import DatasetRow
from src.domain.training import FoldManifest
from src.nlp.training.dataset import DatasetStore
from src.nlp.training.splitting import DatasetSplitter
from src.nlp.training.validation import DatasetValidator


def test_validation_rejects_conflicting_labels_in_one_family() -> None:
    common = {
        "normalized_text": "Xe chạy ổn",
        "aspect": "engine",
        "style": "comment",
        "noise_types": [],
        "difficulty": "easy",
        "canonical_group_id": "family-1",
        "generation_batch": "batch-1",
        "review_status": "approved",
    }
    rows = [
        DatasetRow(id="a", raw_text="Xe chạy ổn", label="positive", **common),
        DatasetRow(id="b", raw_text="Xe không ổn", label="negative", **common),
    ]
    report = DatasetValidator().validate_dataset(rows)
    assert "conflicting_group_labels:family-1" in report.errors


def test_validation_allows_pending_only_when_requested() -> None:
    dataset_path = Path("data/dataset.csv")
    approved = DatasetStore().read(dataset_path)
    pending_rows = [row.model_copy(update={"review_status": "pending"}) for row in approved]
    pending_report = DatasetValidator().validate_dataset(pending_rows, allow_pending=True)
    required_report = DatasetValidator().validate_dataset(pending_rows)
    assert pending_report.ok
    assert "pending_rows:900" in pending_report.warnings
    assert f"non_approved_row:{pending_rows[0].id}" in required_report.errors


def test_validation_accepts_a_balanced_approved_dataset() -> None:
    dataset_path = Path("data/dataset.csv")
    rows = DatasetStore().read(dataset_path)
    report = DatasetValidator().validate_dataset(rows)
    assert report.ok
    assert report.row_count == 900
    assert report.class_counts == {"negative": 300, "neutral": 300, "positive": 300}
    assert report.noise_ratio == 339 / 900


def test_split_manifest_validation_rejects_checksum_and_group_leakage() -> None:
    """A tampered freeze point must not be accepted for model evaluation."""
    dataset_path = Path("data/dataset.csv")
    rows = DatasetStore().read(dataset_path)
    manifest = DatasetSplitter().build(rows, dataset_path)
    leaked_manifest = manifest.model_copy(
        update={
            "dataset_checksum": "0" * 64,
            "test_ids": [*manifest.test_ids, manifest.development_ids[0]],
        }
    )

    errors = DatasetValidator(dataset_path).validate_split_manifest(rows, leaked_manifest).errors

    assert "split_manifest_checksum_mismatch" in errors
    assert "split_manifest_group_overlap:test_development" in errors


def test_split_manifest_validation_requires_each_development_row_once_in_oof() -> None:
    """Repeated validation groups would otherwise corrupt out-of-fold evaluation."""
    dataset_path = Path("data/dataset.csv")
    rows = DatasetStore().read(dataset_path)
    manifest = DatasetSplitter().build(rows, dataset_path)
    first_fold = manifest.folds[0]
    repeated_validation_ids = first_fold.validation_ids
    replacement_fold = FoldManifest(
        fold=manifest.folds[1].fold,
        train_ids=sorted(set(manifest.development_ids) - set(repeated_validation_ids)),
        validation_ids=repeated_validation_ids,
    )
    invalid_manifest = manifest.model_copy(
        update={"folds": [first_fold, replacement_fold, *manifest.folds[2:]]}
    )

    errors = DatasetValidator(dataset_path).validate_split_manifest(rows, invalid_manifest).errors

    assert "split_manifest_oof_coverage" in errors


def test_validation_flags_duplicated_row_ids() -> None:
    """One controlled row ID must never appear twice in a validated dataset."""
    shared = {
        "raw_text": "Xe chạy ổn",
        "normalized_text": "Xe chạy ổn",
        "aspect": "engine",
        "style": "review",
        "noise_types": [],
        "difficulty": "easy",
        "canonical_group_id": "family-dup",
        "generation_batch": "test",
        "review_status": "approved",
    }
    rows = [
        DatasetRow(id="dup-1", label="positive", **shared),
        DatasetRow(id="dup-1", label="positive", **shared),
    ]

    report = DatasetValidator().validate_dataset(rows)

    assert "duplicate_id:dup-1" in report.errors
    assert report.errors.count("duplicate_id:dup-1") == 1


def test_split_manifest_rejects_empty_and_single_class_development_folds() -> None:
    """A development fold with no rows or a single class must fail manifest validation."""
    dataset_path = Path("data/dataset.csv")
    rows = DatasetStore().read(dataset_path)
    manifest = DatasetSplitter().build(rows, dataset_path)
    labels_by_id = {row.id: row.label.value for row in rows}

    empty_fold = FoldManifest(fold=0, train_ids=list(manifest.development_ids), validation_ids=[])
    empty_errors = (
        DatasetValidator(dataset_path)
        .validate_split_manifest(
            rows, manifest.model_copy(update={"folds": [empty_fold, *manifest.folds[1:]]})
        )
        .errors
    )
    assert "split_manifest_class_missing:fold_0_validation" in empty_errors
    assert "split_manifest_oof_coverage" in empty_errors

    single_class_ids = sorted(
        row_id for row_id in manifest.folds[0].validation_ids if labels_by_id[row_id] == "negative"
    )
    assert single_class_ids, "fixture fold 0 must contain negative validation rows"
    single_class_fold = FoldManifest(
        fold=0,
        train_ids=sorted(set(manifest.development_ids) - set(single_class_ids)),
        validation_ids=single_class_ids,
    )
    single_class_errors = (
        DatasetValidator(dataset_path)
        .validate_split_manifest(
            rows,
            manifest.model_copy(update={"folds": [single_class_fold, *manifest.folds[1:]]}),
        )
        .errors
    )
    assert "split_manifest_class_missing:fold_0_validation" in single_class_errors


def test_validation_cli_reports_unsupported_label_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "unsupported-label.csv"
    with input_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "id",
                "raw_text",
                "normalized_text",
                "label",
                "aspect",
                "style",
                "noise_types",
                "difficulty",
                "canonical_group_id",
                "generation_batch",
                "review_status",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "invalid-label-1",
                "raw_text": "Xe chạy ổn",
                "normalized_text": "Xe chạy ổn",
                "label": "unsupported",
                "aspect": "engine",
                "style": "review",
                "noise_types": "[]",
                "difficulty": "easy",
                "canonical_group_id": "family-invalid",
                "generation_batch": "test",
                "review_status": "approved",
            }
        )
    assert data_validate.main(["--input", str(input_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["errors"] == ["unsupported_label:unsupported"]
    assert report["warnings"] == []
    assert report["row_count"] == 0


def test_validation_cli_reports_malformed_noise_json_as_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    input_path = tmp_path / "malformed-noise.csv"
    with input_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=(
                "id",
                "raw_text",
                "normalized_text",
                "label",
                "aspect",
                "style",
                "noise_types",
                "difficulty",
                "canonical_group_id",
                "generation_batch",
                "review_status",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "id": "malformed-noise-1",
                "raw_text": "Xe chạy ổn",
                "normalized_text": "Xe chạy ổn",
                "label": "positive",
                "aspect": "engine",
                "style": "review",
                "noise_types": "not-json",
                "difficulty": "easy",
                "canonical_group_id": "family-malformed",
                "generation_batch": "test",
                "review_status": "approved",
            }
        )
    assert data_validate.main(["--input", str(input_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["errors"] == ["malformed_input:noise_types"]
    assert report["warnings"] == []
    assert report["row_count"] == 0
