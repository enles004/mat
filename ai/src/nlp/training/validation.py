import hashlib as _hashlib
from collections import Counter as _Counter
from pathlib import Path as _Path

from src.domain.entities import DatasetRow as _DatasetRow
from src.domain.entities import ReviewStatus as _ReviewStatus
from src.domain.entities import SentimentLabel as _SentimentLabel
from src.domain.training import SplitManifest as _SplitManifest
from src.domain.training import ValidationReport as _ValidationReport


class DatasetValidator:
    """Validate typed datasets and caller-supplied split manifests."""

    def __init__(self, dataset_path: _Path | None = None) -> None:
        self._dataset_path = dataset_path

    @staticmethod
    def _report(rows: list[_DatasetRow], errors: list[str]) -> _ValidationReport:
        return _ValidationReport(
            errors=tuple(errors),
            warnings=(),
            row_count=len(rows),
            class_counts={
                label.value: sum(row.label is label for row in rows) for label in _SentimentLabel
            },
            noise_ratio=sum(bool(row.noise_types) for row in rows) / len(rows) if rows else 0.0,
        )

    def validate_dataset(
        self, rows: list[_DatasetRow], allow_pending: bool = False
    ) -> _ValidationReport:
        """Check row range, labels, balance, noise ratio, groups, and review state."""
        valid_labels = {label.value for label in _SentimentLabel}
        class_counts = {label.value: 0 for label in _SentimentLabel}
        groups_by_label = {label.value: set[str]() for label in _SentimentLabel}
        group_labels: dict[str, set[str]] = {}
        errors: list[str] = []
        warnings: list[str] = []
        seen_ids: set[str] = set()
        noisy_rows = 0
        pending_rows = 0
        rejected_rows = 0
        if not 500 <= len(rows) <= 1000:
            errors.append(f"row_count_out_of_range:{len(rows)}")
        for row in rows:
            row_id = str(row.id)
            label = str(row.label)
            group_id = str(row.canonical_group_id)
            if not row_id.strip():
                errors.append("blank_id")
            elif row_id in seen_ids:
                errors.append(f"duplicate_id:{row_id}")
            else:
                seen_ids.add(row_id)
            if not str(row.raw_text).strip() or not str(row.normalized_text).strip():
                errors.append(f"blank_text:{row_id}")
            if label not in valid_labels:
                errors.append(f"unsupported_label:{label}")
            else:
                class_counts[label] += 1
                groups_by_label[label].add(group_id)
            group_labels.setdefault(group_id, set()).add(label)
            noisy_rows += bool(row.noise_types)
            review_status = str(row.review_status)
            if review_status == _ReviewStatus.PENDING.value:
                pending_rows += 1
            elif review_status == _ReviewStatus.REJECTED.value:
                rejected_rows += 1
            if not allow_pending and review_status != _ReviewStatus.APPROVED.value:
                errors.append(f"non_approved_row:{row_id}")
        for group_id, labels in group_labels.items():
            if len(labels) > 1:
                errors.append(f"conflicting_group_labels:{group_id}")
        for label, groups in groups_by_label.items():
            if len(groups) < 5:
                errors.append(f"too_few_groups:{label}:{len(groups)}")
        largest_class = max(class_counts.values(), default=0)
        smallest_class = min(class_counts.values(), default=0)
        if largest_class and (largest_class - smallest_class) / largest_class > 0.10:
            errors.append(f"class_imbalance:min={smallest_class},max={largest_class}")
        noise_ratio = noisy_rows / len(rows) if rows else 0.0
        if not 0.35 <= noise_ratio <= 0.40:
            errors.append(f"noise_ratio_out_of_range:{noise_ratio:.4f}")
        if pending_rows and allow_pending:
            warnings.append(f"pending_rows:{pending_rows}")
        if rejected_rows and allow_pending:
            warnings.append(f"rejected_rows:{rejected_rows}")
        return _ValidationReport(
            errors=tuple(errors),
            warnings=tuple(warnings),
            row_count=len(rows),
            class_counts=class_counts,
            noise_ratio=noise_ratio,
        )

    @staticmethod
    def _duplicate_ids(ids: list[str]) -> bool:
        return len(ids) != len(set(ids))

    @staticmethod
    def _has_all_labels(ids: list[str], labels_by_id: dict[str, str]) -> bool:
        return {labels_by_id[row_id] for row_id in ids if row_id in labels_by_id} == {
            label.value for label in _SentimentLabel
        }

    def validate_split_manifest(
        self, rows: list[_DatasetRow], manifest: _SplitManifest
    ) -> _ValidationReport:
        """Validate complete, group-disjoint test and development partitions."""
        errors: list[str] = []
        if self._dataset_path is None:
            errors.append("split_manifest_dataset_path_missing")
        elif (
            manifest.dataset_checksum
            != _hashlib.sha256(self._dataset_path.read_bytes()).hexdigest()
        ):
            errors.append("split_manifest_checksum_mismatch")
        dataset_ids = {row.id for row in rows}
        groups_by_id = {row.id: row.canonical_group_id for row in rows}
        labels_by_id = {row.id: row.label.value for row in rows}
        test_ids = manifest.test_ids
        development_ids = manifest.development_ids
        all_partition_ids = [*test_ids, *development_ids]
        if self._duplicate_ids(all_partition_ids):
            errors.append("split_manifest_duplicate_ids")
        if set(all_partition_ids) != dataset_ids:
            errors.append("split_manifest_row_coverage")
        if set(all_partition_ids) - dataset_ids:
            errors.append("split_manifest_unknown_id")
        test_groups = {groups_by_id[row_id] for row_id in test_ids if row_id in groups_by_id}
        development_groups = {
            groups_by_id[row_id] for row_id in development_ids if row_id in groups_by_id
        }
        if not test_groups.isdisjoint(development_groups):
            errors.append("split_manifest_group_overlap:test_development")
        if not self._has_all_labels(test_ids, labels_by_id):
            errors.append("split_manifest_class_missing:test")
        if not self._has_all_labels(development_ids, labels_by_id):
            errors.append("split_manifest_class_missing:development")
        development_id_set = set(development_ids)
        if len(manifest.folds) != 5 or [fold.fold for fold in manifest.folds] != list(range(5)):
            errors.append("split_manifest_fold_count")
        oof_validation_ids: list[str] = []
        for fold in manifest.folds:
            train_ids = fold.train_ids
            validation_ids = fold.validation_ids
            oof_validation_ids.extend(validation_ids)
            fold_ids = [*train_ids, *validation_ids]
            if self._duplicate_ids(fold_ids) or set(fold_ids) != development_id_set:
                errors.append(f"split_manifest_row_coverage:fold_{fold.fold}")
            if set(fold_ids) - development_id_set:
                errors.append(f"split_manifest_unknown_id:fold_{fold.fold}")
            train_groups = {groups_by_id[row_id] for row_id in train_ids if row_id in groups_by_id}
            validation_groups = {
                groups_by_id[row_id] for row_id in validation_ids if row_id in groups_by_id
            }
            if not train_groups.isdisjoint(validation_groups):
                errors.append(f"split_manifest_group_overlap:fold_{fold.fold}")
            if not self._has_all_labels(train_ids, labels_by_id):
                errors.append(f"split_manifest_class_missing:fold_{fold.fold}_train")
            if not self._has_all_labels(validation_ids, labels_by_id):
                errors.append(f"split_manifest_class_missing:fold_{fold.fold}_validation")
        if _Counter(oof_validation_ids) != _Counter(development_ids):
            errors.append("split_manifest_oof_coverage")
        return self._report(rows, errors)
