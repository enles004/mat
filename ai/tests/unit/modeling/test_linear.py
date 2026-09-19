import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from scripts.train_baseline import _load_seed
from src.domain.entities import DatasetRow, SentimentLabel
from src.domain.training import BaselineRun, FoldManifest, SplitManifest
from src.nlp.constants import _BASELINE_CONFIG, LABEL_ORDER
from src.nlp.modeling.baseline import BaselineTrainer
from src.nlp.training.dataset import DatasetStore
from src.nlp.training.validation import DatasetValidator


def _trainer() -> BaselineTrainer:
    return BaselineTrainer(dataset_store=DatasetStore(), validator=DatasetValidator())


def test_linear_pipeline_produces_ordered_probabilities() -> None:
    texts = [
        "xe chạy quá tệ",
        "thông số xe ở mức tiêu chuẩn",
        "xe chạy rất tốt",
    ]
    labels = ["negative", "neutral", "positive"]
    trainer = _trainer()
    model = trainer.build_pipeline(seed=42)
    model.fit(texts, labels)
    scores = trainer.predict_scores(model, ["xe chạy tốt"])
    assert list(scores.columns) == [label.value for label in SentimentLabel]
    assert np.isclose(scores.to_numpy().sum(axis=1), 1.0).all()


def test_cross_validation_emits_only_each_development_row_as_validation() -> None:
    rows = [
        DatasetRow(
            id=f"{label.value}-{index}",
            raw_text=f"raw {label.value} automotive text {index}",
            normalized_text=f"normalized {label.value} automotive text {index}",
            label=label,
            aspect="engine",
            style="review",
            noise_types=[],
            difficulty="easy",
            canonical_group_id=f"group-{label.value}-{index}",
            generation_batch="test",
            review_status="approved",
        )
        for label in SentimentLabel
        for index in range(10)
    ]
    frozen_test = DatasetRow(
        id="frozen-test",
        raw_text="held out text",
        normalized_text="held out text",
        label="positive",
        aspect="engine",
        style="review",
        noise_types=[],
        difficulty="easy",
        canonical_group_id="frozen-group",
        generation_batch="test",
        review_status="approved",
    )
    development_ids = sorted(row.id for row in rows)
    folds = [
        FoldManifest(
            fold=fold,
            validation_ids=sorted(
                f"{label.value}-{index}"
                for label in SentimentLabel
                for index in range(fold * 2, fold * 2 + 2)
            ),
            train_ids=sorted(
                row_id
                for row_id in development_ids
                if row_id
                not in {
                    f"{label.value}-{index}"
                    for label in SentimentLabel
                    for index in range(fold * 2, fold * 2 + 2)
                }
            ),
        )
        for fold in range(5)
    ]
    manifest = SplitManifest(
        seed=42,
        dataset_checksum="frozen",
        development_ids=development_ids,
        test_ids=[frozen_test.id],
        folds=folds,
    )

    run = _trainer().cross_validate([*rows, frozen_test], manifest, view="raw")

    assert list(run.oof_predictions.columns) == [
        "id",
        "fold",
        "y_true",
        *[label.value for label in SentimentLabel],
    ]
    assert set(run.oof_predictions["id"]) == set(development_ids)
    assert frozen_test.id not in set(run.oof_predictions["id"])
    assert len(run.oof_predictions) == len(development_ids)
    assert len(run.fold_macro_f1) == 5
    assert np.isclose(
        run.oof_predictions[[label.value for label in SentimentLabel]].sum(axis=1), 1.0
    ).all()


def test_view_selection_keeps_raw_when_macro_f1_means_tie() -> None:
    raw = BaselineRun(view="raw", oof_predictions=pd.DataFrame(), fold_macro_f1=(0.5,) * 5)
    normalized = BaselineRun(
        view="normalized", oof_predictions=pd.DataFrame(), fold_macro_f1=(0.5,) * 5
    )

    assert _trainer().select_view(raw, normalized) is raw


def test_cross_validation_rejects_fold_train_validation_overlap() -> None:
    """The fold-disjointness guard must survive optimized Python (no `assert`)."""
    rows = [
        DatasetRow(
            id=f"{label.value}-{index}",
            raw_text=f"raw {label.value} automotive text {index}",
            normalized_text=f"normalized {label.value} automotive text {index}",
            label=label,
            aspect="engine",
            style="review",
            noise_types=[],
            difficulty="easy",
            canonical_group_id=f"group-{label.value}-{index}",
            generation_batch="test",
            review_status="approved",
        )
        for label in SentimentLabel
        for index in range(10)
    ]
    frozen_test = DatasetRow(
        id="frozen-test",
        raw_text="held out text",
        normalized_text="held out text",
        label="positive",
        aspect="engine",
        style="review",
        noise_types=[],
        difficulty="easy",
        canonical_group_id="frozen-group",
        generation_batch="test",
        review_status="approved",
    )
    development_ids = sorted(row.id for row in rows)
    folds = [
        FoldManifest(
            fold=fold,
            validation_ids=sorted(
                f"{label.value}-{index}"
                for label in SentimentLabel
                for index in range(fold * 2, fold * 2 + 2)
            ),
            train_ids=sorted(
                row_id
                for row_id in development_ids
                if row_id
                not in {
                    f"{label.value}-{index}"
                    for label in SentimentLabel
                    for index in range(fold * 2, fold * 2 + 2)
                }
            ),
        )
        for fold in range(5)
    ]
    leaked_fold = folds[0].model_copy(
        update={"train_ids": sorted([*folds[0].train_ids, folds[0].validation_ids[0]])}
    )
    manifest = SplitManifest(
        seed=42,
        dataset_checksum="frozen",
        development_ids=development_ids,
        test_ids=[frozen_test.id],
        folds=[leaked_fold, *folds[1:]],
    )

    with pytest.raises(ValueError, match="overlaps training and validation IDs"):
        _trainer().cross_validate([*rows, frozen_test], manifest, view="raw")


def test_model_config_rejects_label_order_mismatch(tmp_path: Path) -> None:
    """Config labels must equal the canonical SentimentLabel order, not merely the set."""
    manifest = SplitManifest(
        seed=42, dataset_checksum="frozen", development_ids=[], test_ids=[], folds=[]
    )
    aligned_path = tmp_path / "aligned-models.yaml"
    aligned_path.write_text(
        yaml.safe_dump({"seed": 42, "labels": LABEL_ORDER, "baseline": _BASELINE_CONFIG}),
        encoding="utf-8",
    )

    assert _load_seed(aligned_path, manifest) == 42

    reordered_path = tmp_path / "reordered-models.yaml"
    reordered_path.write_text(
        yaml.safe_dump({"seed": 42, "labels": list(reversed(LABEL_ORDER))}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="labels must match the SentimentLabel enum order"):
        _load_seed(reordered_path, manifest)


def _deterministic_fixture() -> tuple[list[DatasetRow], SplitManifest]:
    """Small approved development set with five complete group-disjoint folds."""
    rows = [
        DatasetRow(
            id=f"{label.value}-{index}",
            raw_text=f"raw {label.value} automotive text {index}",
            normalized_text=f"normalized {label.value} automotive text {index}",
            label=label,
            aspect="engine",
            style="review",
            noise_types=[],
            difficulty="easy",
            canonical_group_id=f"group-{label.value}-{index}",
            generation_batch="test",
            review_status="approved",
        )
        for label in SentimentLabel
        for index in range(10)
    ]
    frozen_test = DatasetRow(
        id="frozen-test",
        raw_text="held out text",
        normalized_text="held out text",
        label="positive",
        aspect="engine",
        style="review",
        noise_types=[],
        difficulty="easy",
        canonical_group_id="frozen-group",
        generation_batch="test",
        review_status="approved",
    )
    development_ids = sorted(row.id for row in rows)
    folds = [
        FoldManifest(
            fold=fold,
            validation_ids=sorted(
                f"{label.value}-{index}"
                for label in SentimentLabel
                for index in range(fold * 2, fold * 2 + 2)
            ),
            train_ids=sorted(
                row_id
                for row_id in development_ids
                if row_id
                not in {
                    f"{label.value}-{index}"
                    for label in SentimentLabel
                    for index in range(fold * 2, fold * 2 + 2)
                }
            ),
        )
        for fold in range(5)
    ]
    manifest = SplitManifest(
        seed=42,
        dataset_checksum="frozen",
        development_ids=development_ids,
        test_ids=[frozen_test.id],
        folds=folds,
    )
    return [*rows, frozen_test], manifest


def test_small_fixture_rerun_is_byte_deterministic() -> None:
    """Two independent baseline reruns over one small fixture must be byte-identical."""

    def rerun_digest() -> str:
        rows, manifest = _deterministic_fixture()
        chunks: list[str] = []
        for view in ("raw", "normalized"):
            run = _trainer().cross_validate(rows, manifest, view=view)
            chunks.append(run.oof_predictions.to_csv(index=False, lineterminator="\n"))
            chunks.append(json.dumps(list(run.fold_macro_f1)) + "\n")
        return hashlib.sha256("\n".join(chunks).encode("utf-8")).hexdigest()

    first_digest = rerun_digest()
    second_digest = rerun_digest()
    print(f"fixture rerun digest: {first_digest}")
    assert first_digest == second_digest
    assert len(first_digest) == 64
