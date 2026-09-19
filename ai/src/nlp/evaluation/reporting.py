import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from src.domain.entities import DatasetRow
from src.nlp.constants import _SERVICE_ROOT, _SLICE_FIELDS, FROZEN_OUTPUT_ROOTS
from src.nlp.evaluation.behavioral import (
    BehavioralEvaluator,
    ChallengeCase,
    PredictionMap,
)
from src.nlp.evaluation.metrics import LABELS, MetricsEvaluator
from src.nlp.modeling.baseline import BaselineTrainer
from src.nlp.training.dataset import DatasetStore


class EvaluationReporter:
    """Evaluate one prediction artifact and write Markdown plus JSON evidence."""

    def __init__(
        self,
        *,
        evaluator: BehavioralEvaluator | None = None,
        metrics: MetricsEvaluator | None = None,
        trainer: BaselineTrainer | None = None,
    ) -> None:
        self._evaluator = evaluator or BehavioralEvaluator()
        self._metrics_evaluator = metrics or MetricsEvaluator()
        self._baseline_trainer = trainer or BaselineTrainer()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        return cast(dict[str, Any], document)

    @staticmethod
    def ensure_output_allowed(path: Path) -> None:
        """Refuse evidence paths that resolve into a frozen data/artifact/run root."""
        resolved = path.resolve()
        for root_name in FROZEN_OUTPUT_ROOTS:
            frozen_root = (_SERVICE_ROOT / root_name).resolve()
            if resolved == frozen_root or frozen_root in resolved.parents:
                raise ValueError(
                    f"refusing to write evaluation evidence into the frozen "
                    f"{root_name}/ location: {path}"
                )

    @staticmethod
    def _validate_predictions(frame: pd.DataFrame) -> None:
        required = {"id", "fold", "y_true", *LABELS}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Prediction file is missing columns: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Prediction file must not be empty")
        if frame["id"].duplicated().any():
            raise ValueError("Selected predictions must contain one row per ID")
        unsupported_labels = set(frame["y_true"].astype(str)) - set(LABELS)
        if unsupported_labels:
            raise ValueError(
                f"Prediction file contains unsupported labels: {sorted(unsupported_labels)}"
            )
        probabilities = frame[LABELS].to_numpy(dtype=float)
        if not np.isfinite(probabilities).all():
            raise ValueError("Prediction probabilities must be finite")
        if ((probabilities < 0.0) | (probabilities > 1.0)).any():
            raise ValueError("Prediction probabilities must be in [0, 1]")
        if not np.isclose(probabilities.sum(axis=1), 1.0).all():
            raise ValueError("Prediction probabilities must sum to one")

    def load_selected_predictions(self, path: Path) -> tuple[pd.DataFrame, str]:
        """Load one candidate view, honoring the adjacent baseline selection evidence."""
        frame = pd.read_csv(path)
        selected_view = "single"
        if "view" in frame.columns:
            metrics_path = path.with_name("cv_metrics.json")
            metrics = self._read_json(metrics_path)
            chosen_view = metrics.get("chosen_view")
            if not isinstance(chosen_view, str):
                raise ValueError(f"Missing chosen_view in {metrics_path}")
            available_views = set(frame["view"].astype(str))
            if chosen_view not in available_views:
                raise ValueError(f"Chosen view {chosen_view!r} is absent from predictions")
            frame = (
                frame.loc[frame["view"] == chosen_view].drop(columns="view").reset_index(drop=True)
            )
            selected_view = chosen_view
        self._validate_predictions(frame)
        return frame, selected_view

    @staticmethod
    def _checksum(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _fold_distribution(self, predictions: pd.DataFrame) -> dict[str, dict[str, Any]]:
        distribution: dict[str, dict[str, Any]] = {}
        for fold, fold_predictions in predictions.groupby("fold", sort=True):
            fold_metrics = self._metrics_evaluator.compute(fold_predictions)
            distribution[str(fold)] = {
                "count": len(fold_predictions),
                "macro_f1": fold_metrics["macro_f1"],
            }
        return distribution

    @staticmethod
    def _challenge_texts(cases: list[ChallengeCase]) -> list[str]:
        return list(
            dict.fromkeys(text for case in cases for text in (case.text, case.paired_text) if text)
        )

    def _candidate_type(self, predictions_path: Path) -> str:
        metrics_path = predictions_path.with_name("cv_metrics.json")
        if not metrics_path.exists():
            return "baseline"
        candidate_type = self._read_json(metrics_path).get("candidate_type", "baseline")
        if not isinstance(candidate_type, str):
            raise ValueError(f"candidate_type must be a string in {metrics_path}")
        return candidate_type

    def _predict_transformer_challenges(
        self, predictions_path: Path, cases: list[ChallengeCase]
    ) -> PredictionMap:
        """Load probabilities emitted by the five pinned transformer fold models."""
        evidence_path = predictions_path.with_name("challenge_predictions.json")
        evidence = self._read_json(evidence_path)
        texts = self._challenge_texts(cases)
        missing_texts = set(texts) - evidence.keys()
        if missing_texts:
            raise ValueError(
                f"Transformer challenge evidence is missing texts: {sorted(missing_texts)}"
            )
        predictions: PredictionMap = {}
        for text in texts:
            raw_scores = evidence[text]
            if not isinstance(raw_scores, dict) or set(raw_scores) != set(LABELS):
                raise ValueError(
                    f"Transformer challenge evidence has invalid labels for text: {text!r}"
                )
            scores = {label: float(raw_scores[label]) for label in LABELS}
            values = np.asarray(list(scores.values()), dtype=float)
            if not np.isfinite(values).all() or (values < 0.0).any() or (values > 1.0).any():
                raise ValueError(
                    f"Transformer challenge evidence has invalid probabilities for text: {text!r}"
                )
            if not np.isclose(values.sum(), 1.0):
                raise ValueError(
                    f"Transformer challenge evidence does not sum to one for text: {text!r}"
                )
            predictions[text] = scores
        return predictions

    def _predict_baseline_challenges(
        self,
        predictions: pd.DataFrame,
        rows: list[DatasetRow],
        cases: list[ChallengeCase],
        selected_view: str,
    ) -> PredictionMap:
        if selected_view not in {"raw", "normalized"}:
            raise ValueError("Challenge inference currently requires a selected baseline text view")

        rows_by_id = {row.id: row for row in rows}
        missing_ids = set(predictions["id"].astype(str)) - rows_by_id.keys()
        if missing_ids:
            raise ValueError(f"Prediction IDs are absent from the dataset: {sorted(missing_ids)}")
        development_rows = [rows_by_id[str(row_id)] for row_id in predictions["id"]]
        train_texts = [
            row.raw_text if selected_view == "raw" else row.normalized_text
            for row in development_rows
        ]
        train_labels = [row.label.value for row in development_rows]
        model = self._baseline_trainer.build_pipeline()
        model.fit(train_texts, train_labels)

        texts = self._challenge_texts(cases)
        scores = self._baseline_trainer.predict_scores(model, texts)
        return {
            text: {label: float(scores.iloc[index][label]) for label in LABELS}
            for index, text in enumerate(texts)
        }

    def _summarize_challenges(
        self, cases: list[ChallengeCase], predictions: PredictionMap
    ) -> dict[str, Any]:
        results = self._evaluator.evaluate(cases, predictions)
        cases_by_id = {case.id: case for case in cases}
        by_kind_counts = Counter(result.kind for result in results)
        by_kind_passed = Counter(result.kind for result in results if result.passed)
        by_phenomenon_counts = Counter(case.phenomenon for case in cases)
        by_phenomenon_passed: Counter[str] = Counter()
        for result in results:
            if result.passed:
                by_phenomenon_passed[cases_by_id[result.id].phenomenon] += 1

        def summaries(
            counts: Counter[str], passed: Counter[str]
        ) -> dict[str, dict[str, int | float]]:
            return {
                name: {
                    "count": counts[name],
                    "passed": passed[name],
                    "pass_rate": passed[name] / counts[name],
                }
                for name in sorted(counts)
            }

        passed_count = sum(result.passed for result in results)
        return {
            "count": len(results),
            "passed": passed_count,
            "pass_rate": passed_count / len(results),
            "by_kind": summaries(by_kind_counts, by_kind_passed),
            "by_phenomenon": summaries(by_phenomenon_counts, by_phenomenon_passed),
            "results": [
                {
                    "id": result.id,
                    "kind": result.kind,
                    "phenomenon": cases_by_id[result.id].phenomenon,
                    "passed": result.passed,
                    "detail": result.detail,
                }
                for result in results
            ],
        }

    @staticmethod
    def _format_float(value: int | float) -> str:
        return f"{float(value):.6f}"

    def _render_markdown(self, candidate: str, report: dict[str, Any]) -> str:
        metrics = report["metrics"]
        lines = [
            f"# {candidate} evaluation",
            "",
            "## Evaluation scope",
            "",
            f"- Selected prediction view: `{report['selected_view']}`",
            f"- Challenge inference: {report['challenge_inference']}",
            f"- OOF rows: {report['row_count']}",
            f"- Unique OOF IDs: {report['unique_id_count']}",
            "- Behavioral challenge cases are excluded from aggregate metrics and model fitting.",
            "",
            "## Fold distribution",
            "",
            "| Fold | Count | Macro-F1 |",
            "| --- | ---: | ---: |",
        ]
        for fold, values in report["fold_distribution"].items():
            lines.append(
                f"| {fold} | {values['count']} | {self._format_float(values['macro_f1'])} |"
            )

        lines.extend(
            [
                "",
                "## Aggregate metrics",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Macro-F1 | {self._format_float(metrics['macro_f1'])} |",
                f"| Log loss | {self._format_float(metrics['log_loss'])} |",
                f"| Multiclass Brier | {self._format_float(metrics['brier'])} |",
                "",
                "## Per-class metrics",
                "",
                "| Label | Precision | Recall | F1 | Support |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for label in LABELS:
            values = metrics["per_class"][label]
            lines.append(
                f"| {label} | {self._format_float(values['precision'])} | "
                f"{self._format_float(values['recall'])} | "
                f"{self._format_float(values['f1-score'])} | "
                f"{int(values['support'])} |"
            )

        lines.extend(
            [
                "",
                "## Confusion matrix",
                "",
                "Rows are actual labels; columns are predictions in fixed order.",
                "",
                "| Actual \\ Predicted | negative | neutral | positive |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for label, values in zip(LABELS, metrics["confusion_matrix"], strict=True):
            lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")

        lines.extend(["", "## Slice metrics", ""])
        for field, slices in report["slices"].items():
            lines.extend(
                [
                    f"### {field}",
                    "",
                    "| Slice | Count | Macro-F1 | Log loss | Brier |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for name, values in slices.items():
                lines.append(
                    f"| {name} | {values['count']} | "
                    f"{self._format_float(values['macro_f1'])} | "
                    f"{self._format_float(values['log_loss'])} | "
                    f"{self._format_float(values['brier'])} |"
                )
            lines.append("")

        challenge = report["challenge"]
        lines.extend(
            [
                "## Behavioral challenge suite",
                "",
                f"Overall: {challenge['passed']}/{challenge['count']} "
                f"({self._format_float(challenge['pass_rate'])}).",
                "",
                "| Kind | Passed | Count | Pass rate |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for kind, values in challenge["by_kind"].items():
            lines.append(
                f"| {kind} | {values['passed']} | {values['count']} | "
                f"{self._format_float(values['pass_rate'])} |"
            )

        lines.extend(
            [
                "",
                "### Phenomenon results",
                "",
                "| Phenomenon | Passed | Count | Pass rate |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for phenomenon, values in challenge["by_phenomenon"].items():
            lines.append(
                f"| {phenomenon} | {values['passed']} | {values['count']} | "
                f"{self._format_float(values['pass_rate'])} |"
            )

        lines.extend(
            [
                "",
                "## Challenge failures",
                "",
                "| ID | Kind | Phenomenon | Detail |",
                "| --- | --- | --- | --- |",
            ]
        )
        failures = [result for result in challenge["results"] if not result["passed"]]
        if failures:
            for result in failures:
                detail = str(result["detail"]).replace("|", "\\|")
                lines.append(
                    f"| {result['id']} | {result['kind']} | {result['phenomenon']} | {detail} |"
                )
        else:
            lines.append("| — | — | — | No failures |")

        lines.extend(
            [
                "",
                "## Source checksums",
                "",
                "| Artifact | SHA-256 |",
                "| --- | --- |",
            ]
        )
        for name, checksum in report["source_checksums"].items():
            lines.append(f"| {name} | `{checksum}` |")
        return "\n".join(lines) + "\n"

    def generate(
        self,
        predictions_path: Path,
        data_path: Path,
        challenge_path: Path,
        output_path: Path,
    ) -> dict[str, Any]:
        """Evaluate one prediction artifact and write Markdown plus JSON evidence."""
        self.ensure_output_allowed(output_path)
        predictions, selected_view = self.load_selected_predictions(predictions_path)
        rows = DatasetStore().read(data_path)
        cases = self._evaluator.load_cases(challenge_path)
        slices = {
            field: self._metrics_evaluator.compute_slices(predictions, rows, field)
            for field in _SLICE_FIELDS
        }
        candidate_type = self._candidate_type(predictions_path)
        if candidate_type == "transformer":
            challenge_predictions = self._predict_transformer_challenges(predictions_path, cases)
            challenge_inference = (
                "mean probabilities persisted from the five revision-pinned transformer fold models"
            )
        else:
            challenge_predictions = self._predict_baseline_challenges(
                predictions, rows, cases, selected_view
            )
            challenge_inference = "baseline reconstructed from the selected text view"
        source_checksums = {
            "predictions": self._checksum(predictions_path),
            "dataset": self._checksum(data_path),
            "challenge_set": self._checksum(challenge_path),
        }
        if candidate_type == "transformer":
            source_checksums["transformer_challenge_predictions"] = self._checksum(
                predictions_path.with_name("challenge_predictions.json")
            )
        report: dict[str, Any] = {
            "schema_version": "1",
            "selected_view": selected_view,
            "candidate_type": candidate_type,
            "challenge_inference": challenge_inference,
            "row_count": len(predictions),
            "unique_id_count": predictions["id"].nunique(),
            "fold_distribution": self._fold_distribution(predictions),
            "metrics": self._metrics_evaluator.compute(predictions),
            "slices": slices,
            "challenge": self._summarize_challenges(cases, challenge_predictions),
            "source_checksums": source_checksums,
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        candidate = predictions_path.parent.name.capitalize()
        output_path.write_text(self._render_markdown(candidate, report), encoding="utf-8")
        output_path.with_suffix(".json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
