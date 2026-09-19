import json
from pathlib import Path

import pandas as pd  # type: ignore[import-untyped]
import pytest

from src.domain.evaluation import CandidateEvidence
from src.domain.training import FoldManifest, SplitManifest
from src.nlp.evaluation.metrics import LABELS
from src.nlp.evaluation.reporting import EvaluationReporter
from src.nlp.evaluation.selection import ChampionSelector

AI_ROOT = Path(__file__).parents[3]


def test_ineligible_transformer_cannot_replace_baseline() -> None:
    baseline = CandidateEvidence("baseline", True, 0.72, 0.80, "approved")
    transformer = CandidateEvidence("transformer", False, 0.80, 0.90, "research-only")
    assert ChampionSelector().select([baseline, transformer]).name == "baseline"


def test_select_champion_requires_one_eligible_candidate() -> None:
    baseline = CandidateEvidence("baseline", False, 0.72, 0.80, "approved")
    transformer = CandidateEvidence("transformer", False, 0.80, 0.90, "research-only")
    with pytest.raises(ValueError, match="no deployable candidate passed all gates"):
        ChampionSelector().select([baseline, transformer])


def test_ranking_breaks_exact_ties_toward_the_baseline() -> None:
    baseline = CandidateEvidence("baseline", True, 0.72, 0.80, "approved")
    transformer = CandidateEvidence("transformer", True, 0.72, 0.80, "approved")
    assert ChampionSelector().select([baseline, transformer]).name == "baseline"
    assert ChampionSelector().select([transformer, baseline]).name == "baseline"


def test_candidate_evidence_is_immutable() -> None:
    baseline = CandidateEvidence("baseline", True, 0.72, 0.80, "approved")
    with pytest.raises(AttributeError):
        baseline.macro_f1 = 0.99  # type: ignore[misc]


def _transformer_fixtures(
    tmp_path: Path, reported_macro_f1: float
) -> tuple[Path, Path, SplitManifest]:
    """Minimal transformer run/report fixtures whose OOF macro-F1 recomputes to 1.0."""
    development_ids = [f"dev-{index:02d}" for index in range(12)]
    labels = [LABELS[index % 3] for index in range(12)]
    folds: list[FoldManifest] = []
    start = 0
    for fold_index, size in enumerate((3, 3, 2, 2, 2)):
        validation = development_ids[start : start + size]
        train = [row_id for row_id in development_ids if row_id not in validation]
        folds.append(FoldManifest(fold=fold_index, train_ids=train, validation_ids=validation))
        start += size
    manifest = SplitManifest(
        seed=42,
        dataset_checksum="d" * 64,
        development_ids=development_ids,
        test_ids=["test-00", "test-01", "test-02"],
        folds=folds,
    )

    run_dir = tmp_path / "transformer"
    run_dir.mkdir()
    frame = pd.DataFrame(
        {
            "id": development_ids,
            "fold": [index % 5 for index in range(12)],
            "y_true": labels,
        }
    )
    for label in LABELS:
        frame[label] = [0.98 if row_label == label else 0.01 for row_label in labels]
    frame.to_csv(run_dir / "cv_predictions.csv", index=False)
    (run_dir / "cv_metrics.json").write_text(
        json.dumps(
            {
                "labels": LABELS,
                "folds": [{"fold": fold_index} for fold_index in range(5)],
                "mean_macro_f1": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "source.json").write_text(
        json.dumps(
            {
                "revision": "a" * 40,
                "license": "qualcomm-responsible-ai-license",
                "intended_use": "The model is intended for research and educational purposes.",
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "transformer-evaluation.json"
    report_path.write_text(
        json.dumps(
            {
                "metrics": {"macro_f1": reported_macro_f1},
                "challenge": {"count": 74, "passed": 40, "pass_rate": 40.0 / 74.0},
            }
        ),
        encoding="utf-8",
    )
    return run_dir, report_path, manifest


def test_transformer_gate_metrics_must_match_the_recorded_report(tmp_path: Path) -> None:
    run_dir, report_path, manifest = _transformer_fixtures(tmp_path, reported_macro_f1=0.9)
    with pytest.raises(ValueError, match="Transformer OOF metrics disagree"):
        ChampionSelector()._transformer_gates(
            run_dir, report_path, manifest, (True, "unit fixture")
        )


def test_transformer_export_support_is_not_published_as_demonstrated(
    tmp_path: Path,
) -> None:
    run_dir, report_path, manifest = _transformer_fixtures(tmp_path, reported_macro_f1=1.0)
    gates = ChampionSelector()._transformer_gates(
        run_dir, report_path, manifest, (True, "unit fixture")
    )
    assert gates.artifact_export_supported is False
    assert "not demonstrated" in gates.evidence["artifact_export_supported"]
    assert gates.license_approved is False
    assert gates.eligible is False


def test_second_frozen_test_run_is_refused(tmp_path: Path) -> None:
    report = tmp_path / "model-selection.md"
    report.write_text(
        "# Model selection\n\n## Frozen-test evaluation (single run after the gate write)\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="refusing to run the frozen test a second time"):
        ChampionSelector().run_frozen_test(
            data_path=tmp_path / "absent-dataset.csv",
            splits_path=tmp_path / "absent-splits.json",
            baseline_run=tmp_path / "absent-run",
            challenge_path=tmp_path / "absent-challenge.csv",
            artifact_dir=tmp_path / "absent-artifact",
            output_path=report,
            git_revision="a" * 40,
        )


def test_report_output_paths_resolving_into_frozen_locations_fail_closed(
    tmp_path: Path,
) -> None:
    """Evidence writers refuse any output path that resolves into a frozen root."""
    frozen_markdown = AI_ROOT / "data" / "nested" / ".." / "baseline-evaluation.md"

    with pytest.raises(ValueError, match="frozen"):
        EvaluationReporter().generate(
            predictions_path=tmp_path / "absent.csv",
            data_path=tmp_path / "absent-dataset.csv",
            challenge_path=tmp_path / "absent-challenge.csv",
            output_path=frozen_markdown,
        )
    with pytest.raises(ValueError, match="frozen"):
        ChampionSelector().run_gates(
            data_path=tmp_path / "absent-dataset.csv",
            splits_path=tmp_path / "absent-splits.json",
            config_path=tmp_path / "absent-config.yaml",
            challenge_path=tmp_path / "absent-challenge.csv",
            baseline_run=tmp_path / "absent-baseline",
            baseline_report=tmp_path / "absent-baseline.json",
            transformer_run=tmp_path / "absent-transformer",
            transformer_report=tmp_path / "absent-transformer.json",
            output_path=AI_ROOT / "runs" / "model-selection.md",
        )
    with pytest.raises(ValueError, match="frozen"):
        ChampionSelector().run_frozen_test(
            data_path=tmp_path / "absent-dataset.csv",
            splits_path=tmp_path / "absent-splits.json",
            baseline_run=tmp_path / "absent-baseline",
            challenge_path=tmp_path / "absent-challenge.csv",
            artifact_dir=tmp_path / "absent-artifact",
            output_path=AI_ROOT / "artifacts" / "baseline" / "model-selection.md",
            git_revision="a" * 40,
        )

    assert not (AI_ROOT / "data" / "nested").exists()
    assert not (AI_ROOT / "runs" / "model-selection.md").exists()
    assert not (AI_ROOT / "artifacts" / "baseline" / "model-selection.md").exists()
    EvaluationReporter().ensure_output_allowed(AI_ROOT / "reports" / "model-selection.md")
    EvaluationReporter().ensure_output_allowed(tmp_path / "model-selection.md")
