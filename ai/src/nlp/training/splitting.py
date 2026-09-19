import hashlib as _hashlib
from collections import Counter as _Counter
from pathlib import Path as _Path

from sklearn.model_selection import (  # type: ignore[import-untyped]
    StratifiedGroupKFold as _StratifiedGroupKFold,
)

from src.domain.entities import DatasetRow as _DatasetRow
from src.domain.training import FoldManifest as _FoldManifest
from src.domain.training import SplitManifest as _SplitManifest


class DatasetSplitter:
    """Build deterministic group-aware split manifests."""

    @staticmethod
    def _dataset_checksum(path: _Path) -> str:
        return _hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _split_inputs(rows: list[_DatasetRow]) -> tuple[list[str], list[str], list[str]]:
        return (
            [row.id for row in rows],
            [row.label.value for row in rows],
            [row.canonical_group_id for row in rows],
        )

    @staticmethod
    def _test_fold_score(
        rows: list[_DatasetRow], test_indices: list[int], global_class_counts: _Counter[str]
    ) -> float:
        test_count = len(test_indices)
        test_class_counts = _Counter(rows[index].label.value for index in test_indices)
        test_ratio_error = abs(test_count / len(rows) - 0.20)
        class_ratio_error = sum(
            abs(test_class_counts[label] / test_count - count / len(rows))
            for label, count in global_class_counts.items()
        )
        return test_ratio_error + class_ratio_error

    @staticmethod
    def _sorted_ids(rows: list[_DatasetRow], indices: list[int]) -> list[str]:
        return sorted(rows[index].id for index in indices)

    def build(self, rows: list[_DatasetRow], dataset_path: _Path, seed: int = 42) -> _SplitManifest:
        """Select a deterministic test fold, then build five development folds."""
        ids, labels, groups = self._split_inputs(rows)
        splitter = _StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        global_class_counts = _Counter(labels)
        candidates = list(splitter.split(ids, labels, groups))
        _, (development_indices, test_indices) = min(
            enumerate(candidates),
            key=lambda candidate: (
                self._test_fold_score(rows, candidate[1][1].tolist(), global_class_counts),
                candidate[0],
            ),
        )
        development_rows = [rows[index] for index in development_indices]
        development_ids, development_labels, development_groups = self._split_inputs(
            development_rows
        )
        development_splitter = _StratifiedGroupKFold(
            n_splits=5, shuffle=True, random_state=seed + 1
        )
        folds = [
            _FoldManifest(
                fold=fold_index,
                train_ids=self._sorted_ids(development_rows, train_indices.tolist()),
                validation_ids=self._sorted_ids(development_rows, validation_indices.tolist()),
            )
            for fold_index, (train_indices, validation_indices) in enumerate(
                development_splitter.split(development_ids, development_labels, development_groups)
            )
        ]
        return _SplitManifest(
            seed=seed,
            dataset_checksum=self._dataset_checksum(dataset_path),
            development_ids=self._sorted_ids(rows, development_indices.tolist()),
            test_ids=self._sorted_ids(rows, test_indices.tolist()),
            folds=folds,
        )
