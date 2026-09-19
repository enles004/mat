from collections import Counter as _Counter
from collections.abc import Sequence as _Sequence
from typing import Literal as _Literal

import pandas as _pd  # type: ignore[import-untyped]
from sklearn.base import BaseEstimator as _BaseEstimator  # type: ignore[import-untyped]
from sklearn.base import TransformerMixin as _TransformerMixin
from sklearn.feature_extraction.text import (  # type: ignore[import-untyped]
    TfidfVectorizer as _TfidfVectorizer,
)
from sklearn.linear_model import (  # type: ignore[import-untyped]
    LogisticRegression as _LogisticRegression,
)
from sklearn.metrics import f1_score as _f1_score  # type: ignore[import-untyped]
from sklearn.pipeline import (  # type: ignore[import-untyped]
    FeatureUnion as _FeatureUnion,
)
from sklearn.pipeline import (
    Pipeline as _Pipeline,
)

from src.domain.entities import DatasetRow as _DatasetRow
from src.domain.entities import ReviewStatus as _ReviewStatus
from src.domain.training import BaselineRun as _BaselineRun
from src.domain.training import SplitManifest as _SplitManifest
from src.nlp.constants import _VIEWS, LABEL_ORDER, OOF_COLUMNS
from src.nlp.preprocessing.vietnamese_tokenizer import (
    VietnameseSegmenter as _VietnameseSegmenter,
)
from src.nlp.training.dataset import DatasetStore as _DatasetStore
from src.nlp.training.validation import DatasetValidator as _DatasetValidator


class _SegmentTexts(_BaseEstimator, _TransformerMixin):  # type: ignore[misc]
    """Apply explicit Vietnamese segmentation before vectorization."""

    def __init__(self, segmenter: _VietnameseSegmenter) -> None:
        self.segmenter = segmenter

    def fit(self, texts: _Sequence[str], labels: object = None) -> "_SegmentTexts":
        return self

    def transform(self, texts: _Sequence[str]) -> list[str]:
        return self.segmenter.segment_batch(texts)


class BaselineTrainer:
    """Build, score, cross-validate, and select the deterministic baseline."""

    def __init__(
        self,
        dataset_store: _DatasetStore | None = None,
        validator: _DatasetValidator | None = None,
        segmenter: _VietnameseSegmenter | None = None,
    ) -> None:
        self._dataset_store = dataset_store
        self._validator = validator
        self._segmenter = segmenter

    @staticmethod
    def build_pipeline(
        seed: int = 42,
        segmenter: _VietnameseSegmenter | None = None,
    ) -> _Pipeline:
        """Build an unfitted deterministic word/character TF-IDF classifier.

        Without ``segmenter`` the pipeline is exactly the v1 champion
        (implicit sklearn tokenization). Pass a ``VietnameseSegmenter`` to
        insert explicit segmentation before vectorization.
        """
        features = _FeatureUnion(
            [
                (
                    "word",
                    _TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
                (
                    "char",
                    _TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                        lowercase=True,
                    ),
                ),
            ]
        )
        classifier = _LogisticRegression(
            C=2.0,
            class_weight="balanced",
            max_iter=2_000,
            random_state=seed,
        )
        steps: list[tuple[str, object]] = []
        if segmenter is not None:
            steps.append(("segment", _SegmentTexts(segmenter=segmenter)))
        steps.extend([("features", features), ("classifier", classifier)])
        return _Pipeline(steps)

    @staticmethod
    def predict_scores(model: _Pipeline, texts: _Sequence[str]) -> _pd.DataFrame:
        """Return class probabilities in the frozen SentimentLabel enum order."""
        probabilities = model.predict_proba(list(texts))
        classes = list(model.named_steps["classifier"].classes_)
        frame = _pd.DataFrame(probabilities, columns=classes)
        return frame.reindex(columns=LABEL_ORDER)

    @staticmethod
    def _development_row_map(
        rows: list[_DatasetRow], manifest: _SplitManifest
    ) -> dict[str, _DatasetRow]:
        rows_by_id = {row.id: row for row in rows}
        if len(rows_by_id) != len(rows):
            raise ValueError("Dataset rows must have unique IDs")

        development_ids = manifest.development_ids
        test_ids = manifest.test_ids
        if len(development_ids) != len(set(development_ids)):
            raise ValueError("Development IDs must be unique")
        if len(test_ids) != len(set(test_ids)):
            raise ValueError("Frozen test IDs must be unique")
        if not set(development_ids).isdisjoint(test_ids):
            raise ValueError("Development and frozen test IDs must be disjoint")
        missing_ids = set(development_ids) - rows_by_id.keys()
        if missing_ids:
            raise ValueError(f"Dataset is missing development IDs: {sorted(missing_ids)}")

        development_rows = {row_id: rows_by_id[row_id] for row_id in development_ids}
        if any(
            row.review_status is not _ReviewStatus.APPROVED for row in development_rows.values()
        ):
            raise ValueError("Baseline training requires approved development rows")
        return development_rows

    def cross_validate(
        self,
        rows: list[_DatasetRow],
        manifest: _SplitManifest,
        view: _Literal["raw", "normalized"],
    ) -> _BaselineRun:
        """Fit one fresh pipeline per persisted fold and emit validation probabilities."""
        if view not in _VIEWS:
            raise ValueError(f"Unsupported baseline view: {view}")
        if [fold.fold for fold in manifest.folds] != list(range(5)):
            raise ValueError("Baseline cross-validation requires five ordered persisted folds")

        rows_by_id = self._development_row_map(rows, manifest)
        development_id_set = set(manifest.development_ids)
        frames: list[_pd.DataFrame] = []
        fold_macro_f1: list[float] = []

        for fold in manifest.folds:
            train_id_set = set(fold.train_ids)
            validation_id_set = set(fold.validation_ids)
            if not train_id_set.isdisjoint(validation_id_set):
                raise ValueError(f"Persisted fold {fold.fold} overlaps training and validation IDs")
            if train_id_set | validation_id_set != development_id_set:
                raise ValueError(f"Persisted fold {fold.fold} does not cover development IDs")
            if len(train_id_set) != len(fold.train_ids) or len(validation_id_set) != len(
                fold.validation_ids
            ):
                raise ValueError(f"Persisted fold {fold.fold} contains duplicate IDs")

            train_rows = [rows_by_id[row_id] for row_id in fold.train_ids]
            validation_rows = [rows_by_id[row_id] for row_id in fold.validation_ids]
            if view == "raw":
                train_texts = [row.raw_text for row in train_rows]
                validation_texts = [row.raw_text for row in validation_rows]
            else:
                train_texts = [row.normalized_text for row in train_rows]
                validation_texts = [row.normalized_text for row in validation_rows]
            train_labels = [row.label.value for row in train_rows]
            validation_labels = [row.label.value for row in validation_rows]

            model = self.build_pipeline(seed=manifest.seed, segmenter=self._segmenter)
            model.fit(train_texts, train_labels)
            scores = self.predict_scores(model, validation_texts)
            predicted_labels = scores.idxmax(axis="columns").tolist()
            fold_macro_f1.append(
                float(
                    _f1_score(
                        validation_labels,
                        predicted_labels,
                        labels=LABEL_ORDER,
                        average="macro",
                        zero_division=0,
                    )
                )
            )
            frames.append(
                _pd.concat(
                    [
                        _pd.DataFrame(
                            {
                                "id": fold.validation_ids,
                                "fold": fold.fold,
                                "y_true": validation_labels,
                            }
                        ),
                        scores.reset_index(drop=True),
                    ],
                    axis="columns",
                )
            )

        predictions = _pd.concat(frames, ignore_index=True).loc[:, OOF_COLUMNS]
        predictions = predictions.sort_values(["fold", "id"], kind="stable").reset_index(drop=True)
        if _Counter(predictions["id"].tolist()) != _Counter(manifest.development_ids):
            raise AssertionError("OOF predictions must cover every development ID exactly once")
        return _BaselineRun(
            view=view,
            oof_predictions=predictions,
            fold_macro_f1=tuple(fold_macro_f1),
        )

    @staticmethod
    def _mean_macro_f1(run: _BaselineRun) -> float:
        if not run.fold_macro_f1:
            raise ValueError("Baseline run has no fold metrics")
        return sum(run.fold_macro_f1) / len(run.fold_macro_f1)

    def select_view(self, raw: _BaselineRun, normalized: _BaselineRun) -> _BaselineRun:
        """Choose the stronger text view, keeping raw text when the means tie."""
        if raw.view != "raw" or normalized.view != "normalized":
            raise ValueError("Expected raw and normalized baseline runs")
        return raw if self._mean_macro_f1(raw) >= self._mean_macro_f1(normalized) else normalized
