import pandas as pd
import pytest

from src.domain.entities import DatasetRow
from src.nlp.evaluation.metrics import MetricsEvaluator


def test_metrics_include_macro_f1_log_loss_and_multiclass_brier() -> None:
    frame = pd.DataFrame(
        {
            "y_true": ["negative", "neutral", "positive"],
            "negative": [0.8, 0.1, 0.1],
            "neutral": [0.1, 0.8, 0.1],
            "positive": [0.1, 0.1, 0.8],
        }
    )

    metrics = MetricsEvaluator().compute(frame)

    assert metrics["macro_f1"] == 1.0
    assert metrics["log_loss"] == pytest.approx(0.2231435513)
    assert metrics["brier"] == pytest.approx(0.06)
    assert metrics["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert metrics["per_class"]["neutral"]["f1-score"] == 1.0


def test_macro_f1_uses_all_three_labels_when_a_class_is_absent() -> None:
    frame = pd.DataFrame(
        {
            "y_true": ["negative"],
            "negative": [0.8],
            "neutral": [0.1],
            "positive": [0.1],
        }
    )

    assert MetricsEvaluator().compute(frame)["macro_f1"] == pytest.approx(1 / 3)


def test_slice_metrics_emit_only_slices_with_at_least_ten_examples() -> None:
    rows = [
        DatasetRow(
            id=f"row-{index}",
            raw_text=f"Xe số {index}",
            normalized_text=f"Xe số {index}",
            label="negative" if index % 2 == 0 else "positive",
            aspect="engine" if index < 10 else "battery",
            style="review",
            noise_types=["teencode", "emoji"] if index < 10 else ["emoji"],
            difficulty="hard",
            canonical_group_id=f"group-{index}",
            generation_batch="test",
            review_status="approved",
        )
        for index in range(12)
    ]
    predictions = pd.DataFrame(
        {
            "id": [row.id for row in rows],
            "y_true": [row.label.value for row in rows],
            "negative": [0.8 if row.label.value == "negative" else 0.1 for row in rows],
            "neutral": [0.1] * len(rows),
            "positive": [0.8 if row.label.value == "positive" else 0.1 for row in rows],
        }
    )

    aspect_slices = MetricsEvaluator().compute_slices(predictions, rows, "aspect")
    noise_slices = MetricsEvaluator().compute_slices(predictions, rows, "noise_types")

    assert list(aspect_slices) == ["engine"]
    assert aspect_slices["engine"]["count"] == 10
    assert aspect_slices["engine"]["macro_f1"] == pytest.approx(2 / 3)
    assert noise_slices["emoji"]["count"] == 12
    assert noise_slices["teencode"]["count"] == 10


def test_slice_metrics_reject_unsupported_fields() -> None:
    with pytest.raises(ValueError, match="Unsupported slice field"):
        MetricsEvaluator().compute_slices(pd.DataFrame(), [], "label")


def test_metrics_reject_non_finite_probability_input() -> None:
    """NaN or infinite probabilities must fail loud with the established typed error."""
    frame = pd.DataFrame(
        {
            "y_true": ["negative", "positive"],
            "negative": [float("nan"), 0.1],
            "neutral": [0.1, 0.1],
            "positive": [0.1, 0.8],
        }
    )

    with pytest.raises(ValueError, match="probabilities must be finite"):
        MetricsEvaluator().compute(frame)
