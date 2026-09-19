import json
from pathlib import Path

import pandas as pd

from src.nlp.evaluation.behavioral import BehavioralEvaluator
from src.nlp.evaluation.reporting import EvaluationReporter

AI_ROOT = Path(__file__).parents[3]


def test_selected_prediction_view_uses_metrics_choice_without_double_counting(
    tmp_path: Path,
) -> None:
    predictions_path = tmp_path / "cv_predictions.csv"
    pd.DataFrame(
        {
            "view": ["raw", "raw", "normalized", "normalized"],
            "id": ["one", "two", "one", "two"],
            "fold": [0, 1, 0, 1],
            "y_true": ["negative", "positive", "negative", "positive"],
            "negative": [0.9, 0.1, 0.7, 0.2],
            "neutral": [0.05, 0.1, 0.2, 0.2],
            "positive": [0.05, 0.8, 0.1, 0.6],
        }
    ).to_csv(predictions_path, index=False)
    (tmp_path / "cv_metrics.json").write_text(
        json.dumps({"chosen_view": "normalized"}), encoding="utf-8"
    )

    selected, view = EvaluationReporter().load_selected_predictions(predictions_path)

    assert view == "normalized"
    assert selected["id"].tolist() == ["one", "two"]
    assert selected["negative"].tolist() == [0.7, 0.2]


def test_baseline_report_contains_metrics_slices_and_challenge_failures(tmp_path: Path) -> None:
    output_path = tmp_path / "baseline-evaluation.md"

    report = EvaluationReporter().generate(
        predictions_path=AI_ROOT / "runs" / "baseline" / "cv_predictions.csv",
        data_path=AI_ROOT / "data" / "dataset.csv",
        challenge_path=AI_ROOT / "data" / "challenge_set.csv",
        output_path=output_path,
    )

    # The report must honor the pipeline's own view selection evidence
    # (cv_metrics.json: chosen_view), not a specific CV outcome.
    chosen_view = json.loads(
        (AI_ROOT / "runs" / "baseline" / "cv_metrics.json").read_text(encoding="utf-8")
    )["chosen_view"]
    assert report["selected_view"] == chosen_view
    assert report["row_count"] == 720
    assert report["unique_id_count"] == 720
    assert {fold["count"] for fold in report["fold_distribution"].values()} == {144}
    assert set(report["slices"]) == {"noise_types", "aspect", "style", "difficulty"}
    assert all(
        slice_metrics["count"] >= 10
        for family in report["slices"].values()
        for slice_metrics in family.values()
    )
    assert {kind: values["count"] for kind, values in report["challenge"]["by_kind"].items()} == {
        "DIR": 22,
        "INV": 23,
        "MFT": 29,
    }

    markdown = output_path.read_text(encoding="utf-8")
    assert f"Selected prediction view: `{chosen_view}`" in markdown
    assert "## Fold distribution" in markdown
    assert "## Aggregate metrics" in markdown
    assert "## Per-class metrics" in markdown
    assert "## Confusion matrix" in markdown
    assert "### noise_types" in markdown
    assert "### aspect" in markdown
    assert "### style" in markdown
    assert "### difficulty" in markdown
    assert "## Behavioral challenge suite" in markdown
    assert "## Challenge failures" in markdown

    json_report = json.loads(output_path.with_suffix(".json").read_text(encoding="utf-8"))
    failure_ids = {
        result["id"] for result in json_report["challenge"]["results"] if not result["passed"]
    }
    assert failure_ids
    assert all(failure_id in markdown for failure_id in failure_ids)


def test_transformer_report_uses_persisted_transformer_challenge_scores(tmp_path: Path) -> None:
    source_predictions = pd.read_csv(AI_ROOT / "runs" / "baseline" / "cv_predictions.csv")
    predictions_path = tmp_path / "cv_predictions.csv"
    source_predictions.loc[source_predictions["view"] == "raw"].drop(columns="view").to_csv(
        predictions_path, index=False
    )
    (tmp_path / "cv_metrics.json").write_text(
        json.dumps({"candidate_type": "transformer"}), encoding="utf-8"
    )
    cases = BehavioralEvaluator().load_cases(AI_ROOT / "data" / "challenge_set.csv")
    challenge_scores = {
        text: {"negative": 0.1, "neutral": 0.2, "positive": 0.7}
        for case in cases
        for text in (case.text, case.paired_text)
        if text is not None
    }
    (tmp_path / "challenge_predictions.json").write_text(
        json.dumps(challenge_scores), encoding="utf-8"
    )

    report = EvaluationReporter().generate(
        predictions_path=predictions_path,
        data_path=AI_ROOT / "data" / "dataset.csv",
        challenge_path=AI_ROOT / "data" / "challenge_set.csv",
        output_path=tmp_path / "transformer-evaluation.md",
    )

    assert report["selected_view"] == "single"
    assert report["challenge"]["count"] == 74
