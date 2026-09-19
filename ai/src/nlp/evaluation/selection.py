import hashlib
import json
import os
import platform
import subprocess
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import sklearn  # type: ignore[import-untyped]
import yaml

from src.domain.entities import (
    ArtifactManifest,
    CalibrationManifest,
    PreprocessorManifest,
    SentimentLabel,
)
from src.domain.evaluation import (
    MACRO_F1_DROP_TOLERANCE as _MACRO_F1_DROP_TOLERANCE,
)
from src.domain.evaluation import (
    CalibrationDecision as _CalibrationDecision,
)
from src.domain.evaluation import (
    CandidateEvidence as _CandidateEvidence,
)
from src.domain.evaluation import (
    CandidateGates as _CandidateGates,
)
from src.domain.training import SplitManifest as _SplitManifest
from src.nlp.constants import (
    _GATE_TITLES,
    _SLICE_FIELDS,
    BASELINE_LICENSE,
    CHALLENGE_CASE_COUNT,
    EVALUATION_REPORT_PATH,
    MODEL_NAME,
    MODEL_VERSION,
)
from src.nlp.evaluation.behavioral import BehavioralEvaluator, PredictionMap
from src.nlp.evaluation.metrics import LABELS, MetricsEvaluator
from src.nlp.evaluation.reporting import EvaluationReporter
from src.nlp.modeling.baseline import BaselineTrainer
from src.nlp.modeling.calibration import ProbabilityCalibrator
from src.nlp.modeling.registry import ArtifactRegistry
from src.nlp.training.dataset import DatasetStore
from src.nlp.training.validation import DatasetValidator


class ChampionSelector:
    """Record champion gates, run the one-shot frozen test, and export the artifact."""

    def __init__(
        self,
        *,
        reporter: EvaluationReporter | None = None,
        evaluator: BehavioralEvaluator | None = None,
        metrics: MetricsEvaluator | None = None,
        trainer: BaselineTrainer | None = None,
        calibrator: ProbabilityCalibrator | None = None,
        registry: ArtifactRegistry | None = None,
    ) -> None:
        self._trainer = trainer or BaselineTrainer()
        self._evaluator = evaluator or BehavioralEvaluator()
        self._metrics_evaluator = metrics or MetricsEvaluator()
        self._reporter = reporter or EvaluationReporter(
            evaluator=self._evaluator,
            metrics=self._metrics_evaluator,
            trainer=self._trainer,
        )
        self._calibrator = calibrator or ProbabilityCalibrator(self._trainer)
        self._registry = registry or ArtifactRegistry()

    def select(self, candidates: list[_CandidateEvidence]) -> _CandidateEvidence:
        eligible = [candidate for candidate in candidates if candidate.eligible]
        if not eligible:
            raise ValueError("no deployable candidate passed all gates")
        return max(
            eligible,
            key=lambda item: (item.macro_f1, item.behavior_pass_rate, item.name == "baseline"),
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError(f"Expected a JSON object: {path}")
        return cast(dict[str, Any], document)

    @staticmethod
    def _git_revision() -> str:
        """Revision for provenance; containers without .git use MAT_GIT_REVISION."""
        fallback = os.environ.get("MAT_GIT_REVISION", "").strip()
        if fallback:
            if len(fallback) != 40 or any(
                character not in "0123456789abcdef" for character in fallback
            ):
                raise ValueError(f"Unexpected MAT_GIT_REVISION: {fallback!r}")
            return fallback
        repository = Path(__file__).resolve().parents[4]
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
        )
        revision = result.stdout.strip()
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ValueError(f"Unexpected git revision: {revision!r}")
        return revision

    @staticmethod
    def _selection_config(config_path: Path) -> dict[str, Any]:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("Model configuration must be a mapping")
        selection = document.get("selection")
        if not isinstance(selection, dict):
            raise ValueError("Model configuration is missing the selection block")
        if selection.get("calibration_max_macro_f1_drop") != _MACRO_F1_DROP_TOLERANCE:
            raise ValueError(
                "Model configuration calibration_max_macro_f1_drop must match "
                f"{_MACRO_F1_DROP_TOLERANCE}"
            )
        return cast(dict[str, Any], selection)

    @staticmethod
    def _ordered_oof(predictions: pd.DataFrame, manifest: _SplitManifest) -> pd.DataFrame:
        """Reorder OOF rows into manifest development order and attach fold IDs."""
        if Counter(predictions["id"].astype(str).tolist()) != Counter(manifest.development_ids):
            raise ValueError("OOF predictions must cover every development ID exactly once")
        fold_of_id: dict[str, int] = {}
        for fold in manifest.folds:
            for row_id in fold.validation_ids:
                fold_of_id[row_id] = fold.fold
        indexed = predictions.assign(id=predictions["id"].astype(str)).set_index("id")
        ordered = indexed.loc[manifest.development_ids].reset_index(drop=True)
        ordered["fold"] = [fold_of_id[row_id] for row_id in manifest.development_ids]
        return ordered

    def _baseline_calibration(
        self, predictions_path: Path, manifest: _SplitManifest
    ) -> tuple[pd.DataFrame, _CalibrationDecision]:
        """Return manifest-ordered OOF rows and the cross-fitted calibration decision."""
        predictions, _selected_view = self._reporter.load_selected_predictions(predictions_path)
        ordered = self._ordered_oof(predictions, manifest)
        native = ordered[LABELS].to_numpy(dtype=float)
        targets = ordered["y_true"].astype(str).to_numpy()
        folds = ordered["fold"].to_numpy(dtype=np.int64)
        calibrated = self._calibrator.cross_fitted_sigmoid_oof(native, targets, folds)
        return ordered, self._calibrator.decide(native, calibrated, targets)

    @staticmethod
    def _group_leakage_evidence(
        rows: list[Any], manifest: _SplitManifest, data_path: Path
    ) -> tuple[bool, str]:
        errors = DatasetValidator(data_path).validate_split_manifest(rows, manifest).errors
        rows_by_id = {row.id: row for row in rows}
        development_groups = {
            rows_by_id[row_id].canonical_group_id for row_id in manifest.development_ids
        }
        test_groups = {rows_by_id[row_id].canonical_group_id for row_id in manifest.test_ids}
        fold_disjoint = all(
            {rows_by_id[row_id].canonical_group_id for row_id in fold.train_ids}.isdisjoint(
                {rows_by_id[row_id].canonical_group_id for row_id in fold.validation_ids}
            )
            for fold in manifest.folds
        )
        passed = not errors and development_groups.isdisjoint(test_groups) and fold_disjoint
        detail = (
            f"{len(errors)} manifest validation errors; "
            f"{len(development_groups)} development groups disjoint from "
            f"{len(test_groups)} frozen-test groups; fold group isolation={fold_disjoint}"
        )
        return passed, detail

    @staticmethod
    def _behavior_evidence(report: dict[str, Any]) -> tuple[float, dict[str, Any]]:
        challenge = report.get("challenge")
        if not isinstance(challenge, dict) or challenge.get("count") != CHALLENGE_CASE_COUNT:
            raise ValueError("Behavior report must summarize the 74 reviewed challenge cases")
        pass_rate = challenge["pass_rate"]
        if not isinstance(pass_rate, (int, float)):
            raise ValueError("Behavior report pass rate must be numeric")
        return float(pass_rate), challenge

    def _baseline_gates(
        self,
        run_dir: Path,
        report_path: Path,
        manifest: _SplitManifest,
        leakage: tuple[bool, str],
        calibration: _CalibrationDecision,
        git_revision: str,
    ) -> _CandidateGates:
        metrics = self._read_json(run_dir / "cv_metrics.json")
        report = self._read_json(report_path)
        predictions, selected_view = self._reporter.load_selected_predictions(
            run_dir / "cv_predictions.csv"
        )
        ordered = self._ordered_oof(predictions, manifest)
        oof_metrics = self._metrics_evaluator.compute(ordered)
        reported_macro_f1 = float(report["metrics"]["macro_f1"])
        if abs(float(oof_metrics["macro_f1"]) - reported_macro_f1) > 1e-9:
            raise ValueError("Baseline OOF metrics disagree with the recorded evaluation report")

        five_fold = (
            sorted(set(ordered["fold"].tolist())) == [0, 1, 2, 3, 4]
            and len(metrics["views"][selected_view]["fold_macro_f1"]) == 5
            and len(ordered) == len(manifest.development_ids)
        )
        labels_ok = metrics.get("labels") == LABELS and set(ordered["y_true"].astype(str)) == set(
            LABELS
        )
        probabilities = ordered[LABELS].to_numpy(dtype=float)
        finite = bool(
            np.isfinite(probabilities).all()
            and (probabilities >= 0.0).all()
            and (probabilities <= 1.0).all()
            and np.isclose(probabilities.sum(axis=1), 1.0).all()
        )
        behavior_pass_rate, challenge = self._behavior_evidence(report)
        effective_macro_f1 = (
            calibration.calibrated_macro_f1
            if calibration.method == "sigmoid"
            else calibration.native_macro_f1
        )

        return _CandidateGates(
            name="baseline",
            reproducible_five_fold_run=five_fold,
            no_group_leakage=leakage[0],
            all_required_labels=labels_ok,
            finite_probabilities=finite,
            behavior_report_present=True,
            artifact_export_supported=True,
            source_revision_recorded=True,
            license_approved=True,
            macro_f1=effective_macro_f1,
            behavior_pass_rate=behavior_pass_rate,
            license_status=(
                f"approved: {BASELINE_LICENSE} (in-repo trained artifact, no external terms)"
            ),
            evidence={
                "reproducible_five_fold_run": (
                    f"five persisted folds over {len(ordered)} OOF rows, view={selected_view}"
                ),
                "no_group_leakage": leakage[1],
                "all_required_labels": "OOF truth and metrics cover negative/neutral/positive",
                "finite_probabilities": "OOF probabilities finite, in [0, 1], rows sum to one",
                "behavior_report_present": (
                    f"{challenge['passed']}/{challenge['count']} challenges passed "
                    f"({report_path.name})"
                ),
                "artifact_export_supported": "joblib-serializable scikit-learn pipeline",
                "source_revision_recorded": f"git revision {git_revision}",
                "license_approved": "trained in this repository on reviewed synthetic data",
                "macro_f1": (
                    f"native OOF {calibration.native_macro_f1:.6f} matches "
                    f"{reported_macro_f1:.6f} from {report_path.name}; cross-fitted calibrated "
                    f"{calibration.calibrated_macro_f1:.6f}; effective {effective_macro_f1:.6f}"
                ),
            },
        )

    def _transformer_gates(
        self,
        run_dir: Path,
        report_path: Path,
        manifest: _SplitManifest,
        leakage: tuple[bool, str],
    ) -> _CandidateGates:
        metrics = self._read_json(run_dir / "cv_metrics.json")
        source = self._read_json(run_dir / "source.json")
        report = self._read_json(report_path)
        predictions, _selected_view = self._reporter.load_selected_predictions(
            run_dir / "cv_predictions.csv"
        )
        ordered = self._ordered_oof(predictions, manifest)
        oof_metrics = self._metrics_evaluator.compute(ordered)
        folds = metrics.get("folds")
        revision = str(source.get("revision", ""))
        revision_ok = len(revision) == 40 and all(
            character in "0123456789abcdef" for character in revision
        )
        five_fold = (
            isinstance(folds, list)
            and [fold.get("fold") for fold in folds] == [0, 1, 2, 3, 4]
            and len(ordered) == len(manifest.development_ids)
        )
        labels_ok = metrics.get("labels") == LABELS and set(ordered["y_true"].astype(str)) == set(
            LABELS
        )
        probabilities = ordered[LABELS].to_numpy(dtype=float)
        finite = bool(
            np.isfinite(probabilities).all()
            and (probabilities >= 0.0).all()
            and (probabilities <= 1.0).all()
            and np.isclose(probabilities.sum(axis=1), 1.0).all()
        )
        behavior_pass_rate, challenge = self._behavior_evidence(report)
        intended_use = str(source.get("intended_use", "")).strip()
        macro_f1 = float(oof_metrics["macro_f1"])
        reported_macro_f1 = float(report["metrics"]["macro_f1"])
        if abs(macro_f1 - reported_macro_f1) > 1e-9:
            raise ValueError("Transformer OOF metrics disagree with the recorded evaluation report")

        return _CandidateGates(
            name="transformer",
            reproducible_five_fold_run=five_fold,
            no_group_leakage=leakage[0],
            all_required_labels=labels_ok,
            finite_probabilities=finite,
            behavior_report_present=True,
            artifact_export_supported=False,
            source_revision_recorded=revision_ok,
            license_approved=False,
            macro_f1=macro_f1,
            behavior_pass_rate=behavior_pass_rate,
            license_status=(
                "research-only: Qualcomm responsible AI license restricts to research "
                "and educational use"
            ),
            evidence={
                "reproducible_five_fold_run": (
                    f"five persisted folds over {len(ordered)} OOF rows, "
                    f"mean fold macro-F1 {float(metrics['mean_macro_f1']):.6f}"
                ),
                "no_group_leakage": leakage[1],
                "all_required_labels": "OOF truth and metrics cover negative/neutral/positive",
                "finite_probabilities": "OOF probabilities finite, in [0, 1], rows sum to one",
                "behavior_report_present": (
                    f"{challenge['passed']}/{challenge['count']} challenges passed "
                    f"({report_path.name})"
                ),
                "artifact_export_supported": (
                    "not demonstrated: no transformer payload exported "
                    "(Task 8 persisted metrics/predictions only)"
                ),
                "source_revision_recorded": f"BamiBERT revision {revision}",
                "license_approved": (
                    f"model card intended use: {intended_use!r}; deployment requires documented "
                    "approval of the exact terms (see reports/transformer-license-audit.md)"
                ),
                "macro_f1": (
                    f"OOF {macro_f1:.6f} matches {reported_macro_f1:.6f} from {report_path.name}"
                ),
            },
        )

    @staticmethod
    def _format_float(value: int | float) -> str:
        return f"{float(value):.6f}"

    def _render_gates(
        self,
        data_path: Path,
        splits_path: Path,
        config_path: Path,
        challenge_path: Path,
        calibration: _CalibrationDecision,
        baseline: _CandidateGates,
        transformer: _CandidateGates,
        champion: _CandidateEvidence,
        git_revision: str,
    ) -> str:
        lines = [
            "# Model selection",
            "",
            "## Scope and frozen inputs",
            "",
            f"- Recorded: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} UTC",
            f"- Git revision: `{git_revision}`",
            "- Champion gates below were written before any frozen-test label access.",
            "",
            "| Frozen input | SHA-256 |",
            "| --- | --- |",
            f"| `{data_path}` | `{self._sha256(data_path)}` |",
            f"| `{splits_path}` | `{self._sha256(splits_path)}` |",
            f"| `{challenge_path}` | `{self._sha256(challenge_path)}` |",
            "",
            f"- `configs/models.yaml` selection.calibration_max_macro_f1_drop confirmed as "
            f"{_MACRO_F1_DROP_TOLERANCE}.",
            "",
            "## Baseline calibration gate (cross-fitted over persisted folds)",
            "",
            "Per the binding ruling, calibrated OOF scores come from leave-one-fold-out",
            "calibration: for each held-out fold the sigmoid calibrator is fitted only on the",
            "other four folds' OOF outputs, and the held-out fold is scored by that calibrator.",
            "The final calibrator for the exported model is fitted on all OOF outputs via the",
            "five persisted fold pairs. No row is ever calibrated on itself.",
            "",
            "| Metric | Native OOF | Cross-fitted calibrated OOF |",
            "| --- | ---: | ---: |",
            f"| Log loss | {self._format_float(calibration.native_log_loss)} | "
            f"{self._format_float(calibration.calibrated_log_loss)} |",
            f"| Multiclass Brier | {self._format_float(calibration.native_brier)} | "
            f"{self._format_float(calibration.calibrated_brier)} |",
            f"| Macro-F1 | {self._format_float(calibration.native_macro_f1)} | "
            f"{self._format_float(calibration.calibrated_macro_f1)} |",
            "",
        ]
        if calibration.method == "sigmoid":
            lines.append(
                "- Decision: **keep sigmoid calibration** — at least one proper score improved "
                f"(log loss {calibration.calibrated_log_loss - calibration.native_log_loss:+.6f}, "
                f"Brier {calibration.calibrated_brier - calibration.native_brier:+.6f}) and "
                "macro-F1 dropped by at most "
                f"{_MACRO_F1_DROP_TOLERANCE} (change "
                f"{calibration.calibrated_macro_f1 - calibration.native_macro_f1:+.6f})."
            )
        else:
            lines.append(
                "- Decision: **no calibration** — the conservative gate failed: proper scores "
                f"(log loss {calibration.calibrated_log_loss - calibration.native_log_loss:+.6f}, "
                f"Brier {calibration.calibrated_brier - calibration.native_brier:+.6f}) and/or the "
                f"macro-F1 budget ({_MACRO_F1_DROP_TOLERANCE}, change "
                f"{calibration.calibrated_macro_f1 - calibration.native_macro_f1:+.6f})."
            )

        lines.extend(
            [
                "",
                "## Candidate gates",
                "",
                "| Gate | baseline | transformer |",
                "| --- | --- | --- |",
            ]
        )
        for field, title in _GATE_TITLES.items():
            cells = []
            for gates in (baseline, transformer):
                passed = getattr(gates, field)
                state = "PASS" if passed else "FAIL"
                cells.append(f"{state} — {gates.evidence[field]}")
            lines.append(f"| {title} | {cells[0]} | {cells[1]} |")
        lines.append(
            "| Macro-F1 (development OOF) | "
            f"{baseline.evidence['macro_f1']} | {transformer.evidence['macro_f1']} |"
        )
        lines.append(
            "| Behavior pass rate | "
            f"{self._format_float(baseline.behavior_pass_rate)} | "
            f"{self._format_float(transformer.behavior_pass_rate)} |"
        )
        lines.append(
            f"| License status | {baseline.license_status} | {transformer.license_status} |"
        )

        lines.extend(
            [
                "",
                "## Ranking and champion",
                "",
                "Ranking is macro-F1 first, behavior pass rate second, baseline operational",
                "simplicity as the final tie-break. Only eligible candidates may win.",
                "",
                "| Candidate | Eligible | Macro-F1 | Behavior pass rate | Ranking tuple |",
                "| --- | --- | ---: | ---: | --- |",
            ]
        )
        for gates in (baseline, transformer):
            evidence = gates.as_evidence()
            lines.append(
                f"| {evidence.name} | {evidence.eligible} | "
                f"{self._format_float(evidence.macro_f1)} | "
                f"{self._format_float(evidence.behavior_pass_rate)} | "
                f"({self._format_float(evidence.macro_f1)}, "
                f"{self._format_float(evidence.behavior_pass_rate)}, "
                f"{evidence.name == 'baseline'}) |"
            )
        lines.extend([""])
        for gates in (baseline, transformer):
            if not gates.eligible:
                lines.append(f"- `{gates.name}` is ineligible: {gates.license_status}.")
        lines.extend(
            [
                "",
                f"**Champion: {champion.name}** — selected under macro-F1 first, behavior pass",
                "rate second, baseline operational simplicity last, "
                "among eligible candidates only.",
                "",
            ]
        )
        return "\n".join(lines)

    def run_gates(
        self,
        data_path: Path,
        splits_path: Path,
        config_path: Path,
        challenge_path: Path,
        baseline_run: Path,
        baseline_report: Path,
        transformer_run: Path,
        transformer_report: Path,
        output_path: Path,
    ) -> dict[str, Any]:
        """Record every champion gate and the winner before frozen-test access."""
        self._reporter.ensure_output_allowed(output_path)
        self._selection_config(config_path)
        rows = DatasetStore().read(data_path)
        manifest = _SplitManifest.model_validate_json(splits_path.read_text(encoding="utf-8"))
        git_revision = self._git_revision()
        leakage = self._group_leakage_evidence(rows, manifest, data_path)
        _, calibration = self._baseline_calibration(baseline_run / "cv_predictions.csv", manifest)
        baseline = self._baseline_gates(
            baseline_run, baseline_report, manifest, leakage, calibration, git_revision
        )
        transformer = self._transformer_gates(
            transformer_run, transformer_report, manifest, leakage
        )
        champion = self.select([baseline.as_evidence(), transformer.as_evidence()])

        document = self._render_gates(
            data_path,
            splits_path,
            config_path,
            challenge_path,
            calibration,
            baseline,
            transformer,
            champion,
            git_revision,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(document, encoding="utf-8")
        return {
            "champion": champion,
            "calibration": calibration,
            "baseline": baseline,
            "transformer": transformer,
        }

    @staticmethod
    def _development_texts(rows: list[Any], manifest: _SplitManifest, view: str) -> list[str]:
        rows_by_id = {row.id: row for row in rows}
        development_rows = [rows_by_id[row_id] for row_id in manifest.development_ids]
        if view == "raw":
            return [row.raw_text for row in development_rows]
        return [row.normalized_text for row in development_rows]

    def _fit_final_model(
        self, rows: list[Any], manifest: _SplitManifest, decision: _CalibrationDecision, view: str
    ) -> Any:
        rows_by_id = {row.id: row for row in rows}
        development_rows = [rows_by_id[row_id] for row_id in manifest.development_ids]
        texts = self._development_texts(rows, manifest, view)
        labels = [row.label.value for row in development_rows]
        if decision.method == "sigmoid":
            model = self._calibrator.build_estimator(
                seed=manifest.seed,
                fold_pairs=self._calibrator.build_fold_pairs(manifest),
            )
        else:
            model = self._trainer.build_pipeline(seed=manifest.seed)
        model.fit(texts, labels)
        return model

    @staticmethod
    def _predict_probabilities(model: Any, texts: list[str]) -> pd.DataFrame:
        probabilities = np.asarray(model.predict_proba(list(texts)), dtype=float)
        classes = [str(value) for value in model.classes_]
        frame = pd.DataFrame(probabilities, columns=classes)
        return frame.reindex(columns=LABELS)

    @staticmethod
    def _cpu_model() -> str:
        try:
            for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            return "unknown"
        return "unknown"

    def _latency_environment(self, model: Any, texts: list[str]) -> dict[str, Any]:
        durations_ms: list[float] = []
        for text in texts:
            start = time.perf_counter()
            model.predict_proba([text])
            durations_ms.append((time.perf_counter() - start) * 1_000.0)
        batch_start = time.perf_counter()
        model.predict_proba(texts)
        batch_ms = (time.perf_counter() - batch_start) * 1_000.0
        return {
            "per_row_latency_ms": {
                "mean": float(np.mean(durations_ms)),
                "p50": float(np.percentile(durations_ms, 50)),
                "p95": float(np.percentile(durations_ms, 95)),
            },
            "batch_latency_ms": {"rows": len(texts), "total": batch_ms},
            "environment": {
                "python": platform.python_version(),
                "scikit_learn": sklearn.__version__,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "cpu": self._cpu_model(),
            },
        }

    def _challenge_recheck(self, model: Any, challenge_path: Path) -> dict[str, Any]:
        cases = self._evaluator.load_cases(challenge_path)
        texts = list(
            dict.fromkeys(text for case in cases for text in (case.text, case.paired_text) if text)
        )
        scores = self._predict_probabilities(model, texts)
        predictions: PredictionMap = {
            text: {label: float(scores.iloc[index][label]) for label in LABELS}
            for index, text in enumerate(texts)
        }
        results = self._evaluator.evaluate(cases, predictions)
        passed = sum(result.passed for result in results)
        return {
            "count": len(results),
            "passed": passed,
            "pass_rate": passed / len(results),
            "failures": [
                {"id": result.id, "kind": result.kind, "detail": result.detail}
                for result in results
                if not result.passed
            ],
        }

    def _manifest_for(
        self,
        dataset_checksum: str,
        normalization_version: str,
        normalization_checksum: str,
        decision: _CalibrationDecision,
        view: str,
        git_revision: str,
    ) -> Callable[[str], ArtifactManifest]:
        def manifest_for(payload_checksum: str) -> ArtifactManifest:
            if decision.method == "sigmoid":
                fitted_on = "720 development OOF rows; five persisted fold calibrators ensembled"
            else:
                fitted_on = "not-applicable (calibration rejected by the proper-score gate)"
            return ArtifactManifest(
                schema_version="1",
                model_name=MODEL_NAME,
                model_version=MODEL_VERSION,
                backend="linear",
                labels=[SentimentLabel(label) for label in LABELS],
                preprocessor=PreprocessorManifest(
                    name=f"tfidf-word-bigram-char_wb-trigram ({view} view)",
                    version=normalization_version,
                    config_checksum=f"sha256:{normalization_checksum}",
                ),
                calibration=CalibrationManifest(
                    method=decision.method,
                    fitted_on=fitted_on,
                ),
                data_checksum=f"sha256:{dataset_checksum}",
                payload_checksum=payload_checksum,
                git_revision=git_revision,
                license=BASELINE_LICENSE,
                evaluation_report=EVALUATION_REPORT_PATH,
            )

        return manifest_for

    def _render_frozen(
        self,
        metrics: dict[str, Any],
        slices: dict[str, dict[str, Any]],
        latency: dict[str, Any],
        behavior: dict[str, Any],
        errors: list[dict[str, Any]],
    ) -> str:
        lines = [
            "",
            "## Frozen-test evaluation (single run after the gate write)",
            "",
            "The final pipeline was fitted on all 720 development IDs with the selected view",
            "and calibration, then evaluated on the 180 frozen test rows exactly once. No",
            "model-selection decision was revisited from these numbers.",
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
                "| Actual \\\\ Predicted | negative | neutral | positive |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for label, values in zip(LABELS, metrics["confusion_matrix"], strict=True):
            lines.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")

        lines.extend(["", "## Slice metrics", ""])
        for field, field_slices in slices.items():
            lines.extend(
                [
                    f"### {field}",
                    "",
                    "| Slice | Count | Macro-F1 | Log loss | Brier |",
                    "| --- | ---: | ---: | ---: | ---: |",
                ]
            )
            for name, values in field_slices.items():
                lines.append(
                    f"| {name} | {values['count']} | "
                    f"{self._format_float(values['macro_f1'])} | "
                    f"{self._format_float(values['log_loss'])} | "
                    f"{self._format_float(values['brier'])} |"
                )
            lines.append("")

        lines.extend(
            [
                "## Error examples",
                "",
                "| ID | Actual | Predicted | negative | neutral | positive |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        if errors:
            for error in errors:
                lines.append(
                    f"| {error['id']} | {error['actual']} | {error['predicted']} | "
                    f"{error['negative']:.4f} | {error['neutral']:.4f} | "
                    f"{error['positive']:.4f} |"
                )
        else:
            lines.append("| — | — | — | — | — | No frozen-test errors |")

        per_row = latency["per_row_latency_ms"]
        batch = latency["batch_latency_ms"]
        environment = latency["environment"]
        lines.extend(
            [
                "",
                "## Latency and environment",
                "",
                "| Measurement | Value |",
                "| --- | ---: |",
                f"| Per-row predict latency mean | {self._format_float(per_row['mean'])} ms |",
                f"| Per-row predict latency p50 | {self._format_float(per_row['p50'])} ms |",
                f"| Per-row predict latency p95 | {self._format_float(per_row['p95'])} ms |",
                f"| Batch latency ({batch['rows']} rows) | "
                f"{self._format_float(batch['total'])} ms |",
                f"| Platform | {environment['platform']} |",
                f"| CPU | {environment['cpu']} |",
                f"| Python | {environment['python']} |",
                f"| scikit-learn | {environment['scikit_learn']} |",
                f"| NumPy | {environment['numpy']} |",
                "",
                "## Behavioral re-check on the exported model",
                "",
                f"{behavior['passed']}/{behavior['count']} challenges passed "
                f"({self._format_float(behavior['pass_rate'])}) when the exported artifact "
                "itself was re-run over the reviewed challenge set "
                "(development-only evidence, no frozen rows).",
                "",
            ]
        )
        if behavior["failures"]:
            lines.extend(
                [
                    "| ID | Kind | Detail |",
                    "| --- | --- | --- |",
                ]
            )
            for failure in behavior["failures"]:
                detail = str(failure["detail"]).replace("|", "\\|")
                lines.append(f"| {failure['id']} | {failure['kind']} | {detail} |")
            lines.append("")
        return "\n".join(lines)

    def _render_export(self, saved: Any) -> str:
        manifest = saved.manifest
        lines = [
            "## Exported artifact",
            "",
            f"- Directory: `artifacts/baseline` (payload `{saved.payload_files[0].name}` plus "
            "`manifest.json`)",
            f"- Payload checksum: `{manifest.payload_checksum}`",
            f"- Model: `{manifest.model_name}` version {manifest.model_version} "
            f"(backend `{manifest.backend}`)",
            f"- Labels: {', '.join(manifest.labels)}",
            f"- Preprocessor: {manifest.preprocessor.name} "
            f"(config checksum `{manifest.preprocessor.config_checksum}`)",
            f"- Calibration: method `{manifest.calibration.method}`, fitted on "
            f'"{manifest.calibration.fitted_on}"',
            f"- Dataset checksum: `{manifest.data_checksum}`",
            f"- Git revision: `{manifest.git_revision}`",
            f"- License: `{manifest.license}`",
            f"- Evaluation report: `{manifest.evaluation_report}`",
            "- Max input tokens: not applicable (the linear backend applies no tokenizer limit)",
            "",
            "Verify with: `cd ai && uv run --locked python "
            "scripts/release_verify_artifact.py verify artifacts/baseline`.",
            "",
        ]
        return "\n".join(lines)

    @staticmethod
    def _refuse_second_frozen_run(output_path: Path) -> None:
        """Refuse to run the frozen test twice against one selection report."""
        if output_path.exists() and "## Frozen-test evaluation" in output_path.read_text(
            encoding="utf-8"
        ):
            raise ValueError(
                f"{output_path} already records a frozen-test evaluation; "
                "refusing to run the frozen test a second time"
            )

    def run_frozen_test(
        self,
        data_path: Path,
        splits_path: Path,
        baseline_run: Path,
        challenge_path: Path,
        artifact_dir: Path,
        output_path: Path,
        git_revision: str | None = None,
    ) -> dict[str, Any]:
        """Fit on development, evaluate the frozen test once, then export the artifact."""
        if git_revision is None:
            git_revision = self._git_revision()
        self._reporter.ensure_output_allowed(output_path)
        self._refuse_second_frozen_run(output_path)
        dataset_checksum = self._sha256(data_path)
        split_checksum = self._sha256(splits_path)
        rows = DatasetStore().read(data_path)
        manifest = _SplitManifest.model_validate_json(splits_path.read_text(encoding="utf-8"))
        metrics_document = self._read_json(baseline_run / "cv_metrics.json")
        view = str(metrics_document["chosen_view"])
        _, calibration = self._baseline_calibration(baseline_run / "cv_predictions.csv", manifest)

        model = self._fit_final_model(rows, manifest, calibration, view)

        rows_by_id = {row.id: row for row in rows}
        test_rows = [rows_by_id[row_id] for row_id in sorted(manifest.test_ids)]
        texts = [row.raw_text if view == "raw" else row.normalized_text for row in test_rows]
        probabilities = self._predict_probabilities(model, texts)
        predictions = pd.concat(
            [
                pd.DataFrame(
                    {
                        "id": [row.id for row in test_rows],
                        "y_true": [row.label.value for row in test_rows],
                    }
                ),
                probabilities.reset_index(drop=True),
            ],
            axis="columns",
        )
        metrics = self._metrics_evaluator.compute(predictions)
        slices = {
            field: self._metrics_evaluator.compute_slices(predictions, rows, field)
            for field in _SLICE_FIELDS
        }
        latency = self._latency_environment(model, texts)
        behavior = self._challenge_recheck(model, challenge_path)
        errors = []
        for _, row in predictions.iterrows():
            predicted = LABELS[int(np.argmax([row[label] for label in LABELS]))]
            if predicted != str(row["y_true"]):
                errors.append(
                    {
                        "id": str(row["id"]),
                        "actual": str(row["y_true"]),
                        "predicted": predicted,
                        "negative": float(row["negative"]),
                        "neutral": float(row["neutral"]),
                        "positive": float(row["positive"]),
                    }
                )

        saved = self._registry.save(
            model,
            artifact_dir,
            self._manifest_for(
                dataset_checksum,
                self._normalization_version(),
                self._sha256(self._normalization_path()),
                calibration,
                view,
                git_revision,
            ),
        )

        document = self._render_frozen(metrics, slices, latency, behavior, errors)
        document += self._render_export(saved)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(document)
        return {
            "metrics": metrics,
            "row_count": len(predictions),
            "slices": slices,
            "latency": latency,
            "behavior": behavior,
            "errors": errors,
            "artifact": saved,
            "checksums": {
                "dataset": dataset_checksum,
                "split_manifest": split_checksum,
            },
        }

    @staticmethod
    def _normalization_path() -> Path:
        return Path("configs/normalization.yaml")

    @staticmethod
    def _normalization_version() -> str:
        document = yaml.safe_load(
            ChampionSelector._normalization_path().read_text(encoding="utf-8")
        )
        if not isinstance(document, dict) or not isinstance(
            document.get("dictionary_version"), str
        ):
            raise ValueError("Normalization config must declare a string version")
        return str(document["dictionary_version"])
