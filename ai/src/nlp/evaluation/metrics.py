from collections import defaultdict
from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.metrics import (  # type: ignore[import-untyped]
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)

from src.domain.entities import LABELS, DatasetRow
from src.domain.scoring import multiclass_brier
from src.nlp.constants import _SLICE_FIELDS

__all__ = [
    "LABELS",
    "MetricsEvaluator",
    "multiclass_brier",
]


class MetricsEvaluator:
    """Compute aggregate and slice-level metrics from truth and probabilities."""

    def compute(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Compute aggregate metrics from truth and ordered class probabilities."""
        probabilities = frame[LABELS].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError("probabilities must be finite")
        predicted = [LABELS[index] for index in probabilities.argmax(axis=1)]
        per_class = cast(
            dict[str, Any],
            classification_report(
                frame["y_true"],
                predicted,
                labels=LABELS,
                output_dict=True,
                zero_division=0,
            ),
        )
        return {
            "macro_f1": float(
                f1_score(
                    frame["y_true"],
                    predicted,
                    labels=LABELS,
                    average="macro",
                    zero_division=0,
                )
            ),
            "per_class": per_class,
            "confusion_matrix": confusion_matrix(
                frame["y_true"], predicted, labels=LABELS
            ).tolist(),
            "log_loss": float(log_loss(frame["y_true"], probabilities, labels=LABELS)),
            "brier": multiclass_brier(frame["y_true"].tolist(), probabilities),
        }

    def compute_slices(
        self, predictions: pd.DataFrame, rows: Sequence[DatasetRow], field: str
    ) -> dict[str, dict[str, Any]]:
        """Compute metrics for metadata values represented by at least ten predictions."""
        if field not in _SLICE_FIELDS:
            raise ValueError(f"Unsupported slice field: {field}")
        if predictions["id"].duplicated().any():
            raise ValueError("Slice predictions must contain unique IDs")

        rows_by_id = {row.id: row for row in rows}
        member_indices: defaultdict[str, list[int]] = defaultdict(list)
        for position, prediction in predictions.reset_index(drop=True).iterrows():
            row_id = str(prediction["id"])
            if row_id not in rows_by_id:
                raise ValueError(f"Prediction ID is absent from dataset metadata: {row_id}")
            dataset_row = rows_by_id[row_id]
            if str(prediction["y_true"]) != dataset_row.label.value:
                raise ValueError(f"Prediction truth does not match dataset metadata: {row_id}")

            raw_value = getattr(dataset_row, field)
            values = raw_value if isinstance(raw_value, list) else [raw_value]
            for value in dict.fromkeys(str(item) for item in values):
                member_indices[value].append(position)

        slices: dict[str, dict[str, Any]] = {}
        indexed_predictions = predictions.reset_index(drop=True)
        for value in sorted(member_indices):
            indices = member_indices[value]
            if len(indices) < 10:
                continue
            metrics = self.compute(indexed_predictions.iloc[indices])
            slices[value] = {"count": len(indices), **metrics}
        return slices
