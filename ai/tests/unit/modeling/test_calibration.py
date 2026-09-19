import numpy as np
import pytest

from src.domain.evaluation import MACRO_F1_DROP_TOLERANCE
from src.domain.training import FoldManifest, SplitManifest
from src.nlp.evaluation.metrics import LABELS
from src.nlp.modeling.baseline import BaselineTrainer
from src.nlp.modeling.calibration import MACRO_F1_DROP_TOLERANCE as EXPORTED_TOLERANCE
from src.nlp.modeling.calibration import ProbabilityCalibrator


def _calibrator() -> ProbabilityCalibrator:
    return ProbabilityCalibrator(BaselineTrainer())


def test_calibration_module_exposes_domain_macro_f1_drop_tolerance() -> None:
    assert EXPORTED_TOLERANCE == MACRO_F1_DROP_TOLERANCE


def test_temperature_is_positive_and_changes_overconfident_logits() -> None:
    logits = np.array([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]])
    labels = np.array([0, 2, 1])
    temperature = _calibrator().fit_temperature(logits, labels)
    assert temperature > 0.0
    assert temperature != 1.0


def _confidently_wrong_logits(rows: int, wrong_rows: set[int]) -> tuple[np.ndarray, np.ndarray]:
    logits = np.zeros((rows, 3), dtype=np.float64)
    labels = np.fromiter((row % 3 for row in range(rows)), dtype=np.int64, count=rows)
    for row in range(rows):
        true_class = int(labels[row])
        if row in wrong_rows:
            wrong_class = (true_class + 1) % 3
            logits[row, wrong_class] = 9.0
            logits[row, true_class] = 6.0
        else:
            logits[row, true_class] = 9.0
    return logits, labels


def test_temperature_softens_overconfident_logits() -> None:
    wrong_rows = set(range(0, 30, 10))
    logits, labels = _confidently_wrong_logits(30, wrong_rows)
    temperature = _calibrator().fit_temperature(logits, labels)
    assert temperature > 1.0


def test_temperature_rejects_misaligned_inputs() -> None:
    logits = np.zeros((5, 3), dtype=np.float64)
    labels = np.zeros(4, dtype=np.int64)
    with pytest.raises(ValueError, match="aligned"):
        _calibrator().fit_temperature(logits, labels)


def test_cross_fitted_calibration_rejects_non_finite_probabilities() -> None:
    """NaN or infinite OOF probabilities must fail closed before any calibration."""
    rows = 10
    labels = [row % 3 for row in range(rows)]
    folds = [row % 5 for row in range(rows)]

    nan_probabilities = np.full((rows, 3), 1.0 / 3.0)
    nan_probabilities[0, 0] = float("nan")
    with pytest.raises(ValueError, match="probabilities must be finite"):
        _calibrator().cross_fitted_sigmoid_oof(nan_probabilities, labels, folds)

    infinite_probabilities = np.full((rows, 3), 1.0 / 3.0)
    infinite_probabilities[1, 2] = float("inf")
    with pytest.raises(ValueError, match="probabilities must be finite"):
        _calibrator().cross_fitted_sigmoid_oof(infinite_probabilities, labels, folds)


def _fold_manifest(fold_count: int) -> SplitManifest:
    development_ids = [f"dev-{index:02d}" for index in range(12)]
    base, remainder = divmod(len(development_ids), fold_count)
    folds: list[FoldManifest] = []
    start = 0
    for fold in range(fold_count):
        size = base + (1 if fold < remainder else 0)
        validation = development_ids[start : start + size]
        train = [row_id for row_id in development_ids if row_id not in validation]
        folds.append(FoldManifest(fold=fold, train_ids=train, validation_ids=validation))
        start += size
    return SplitManifest(
        seed=42,
        dataset_checksum="c" * 64,
        development_ids=development_ids,
        test_ids=["test-00"],
        folds=folds,
    )


def test_fold_calibrator_pairs_reject_missing_and_extra_persisted_folds() -> None:
    """The exported calibrator ensemble is built from exactly the five persisted folds."""
    with pytest.raises(ValueError, match="five ordered persisted folds"):
        _calibrator().build_fold_pairs(_fold_manifest(4))
    with pytest.raises(ValueError, match="five ordered persisted folds"):
        _calibrator().build_fold_pairs(_fold_manifest(6))


def test_cross_fitted_calibration_never_scores_its_own_training_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = 60
    labels = np.fromiter((row % 3 for row in range(rows)), dtype=np.int64, count=rows)
    folds = np.fromiter((row % 5 for row in range(rows)), dtype=np.int64, count=rows)
    probabilities = np.full((rows, 3), 0.01, dtype=np.float64)
    for row in range(rows):
        probabilities[row, labels[row]] = 0.98

    calibrator = _calibrator()
    original = calibrator._fit_sigmoid_calibrator
    fitted_indices: list[np.ndarray] = []

    def spy(
        native: np.ndarray,
        targets: np.ndarray,
        fit_indices: np.ndarray,
    ) -> list[tuple[float, float]]:
        fitted_indices.append(np.array(fit_indices, copy=True))
        return original(native, targets, fit_indices)

    monkeypatch.setattr(calibrator, "_fit_sigmoid_calibrator", spy)
    calibrated = calibrator.cross_fitted_sigmoid_oof(probabilities, labels, folds)

    assert len(fitted_indices) == 5
    for fold in range(5):
        held_out = np.flatnonzero(folds == fold)
        expected_fit = np.flatnonzero(folds != fold)
        assert not set(fitted_indices[fold].tolist()) & set(held_out.tolist())
        assert set(fitted_indices[fold].tolist()) == set(expected_fit.tolist())
    assert np.isclose(calibrated.sum(axis=1), 1.0).all()
    assert (calibrated >= 0.0).all() and (calibrated <= 1.0).all()


def test_cross_fitted_calibration_softens_overconfident_scores() -> None:
    from sklearn.metrics import log_loss  # type: ignore[import-untyped]

    rows = 90
    labels = np.fromiter((row % 3 for row in range(rows)), dtype=np.int64, count=rows)
    folds = np.fromiter((row % 5 for row in range(rows)), dtype=np.int64, count=rows)
    probabilities = np.full((rows, 3), 0.01, dtype=np.float64)
    for row in range(rows):
        true_class = row % 3
        if row % 4 == 0:
            winner, confidence = (true_class + 1) % 3, 0.99
        else:
            spread = ((row * 7) % 10) / 9.0
            winner, confidence = true_class, 0.60 + 0.35 * spread
        probabilities[row, winner] = confidence
        others = [c for c in range(3) if c != winner]
        remainder = (1.0 - confidence - 0.02) / 2.0
        probabilities[row, others[0]] += remainder
        probabilities[row, others[1]] += remainder

    calibrated = _calibrator().cross_fitted_sigmoid_oof(probabilities, labels, folds)
    truth = [LABELS[index] for index in labels]
    native_loss = float(log_loss(truth, probabilities, labels=LABELS))
    calibrated_loss = float(log_loss(truth, calibrated, labels=LABELS))
    assert calibrated_loss < native_loss


def _distribution(rows: int, winner_for_row: dict[int, tuple[int, float]]) -> np.ndarray:
    probabilities = np.full((rows, 3), 0.01, dtype=np.float64)
    for row, (winner, probability) in winner_for_row.items():
        probabilities[row, winner] = probability
    for row in range(rows):
        remaining = 1.0 - probabilities[row].sum()
        winner = winner_for_row[row][0]
        others = [c for c in range(3) if c != winner]
        probabilities[row, others[0]] += remaining / 2.0
        probabilities[row, others[1]] += remaining / 2.0
    assert np.isclose(probabilities.sum(axis=1), 1.0).all()
    return probabilities


def test_calibration_decision_keeps_a_proper_score_improvement() -> None:
    rows = 30
    labels = np.fromiter((row % 3 for row in range(rows)), dtype=np.int64, count=rows)
    truth = [LABELS[index] for index in labels]

    native_rows = {
        row: (row % 3 if row % 10 < 8 else (row % 3 + 1) % 3, 0.98) for row in range(rows)
    }
    native = _distribution(rows, native_rows)
    calibrated = _distribution(rows, {row: (row % 3, 0.90) for row in range(rows)})

    decision = _calibrator().decide(native, calibrated, truth)
    assert decision.method == "sigmoid"
    assert decision.keep


def test_calibration_decision_rejects_a_macro_f1_regression() -> None:
    rows = 30
    labels = np.fromiter((row % 3 for row in range(rows)), dtype=np.int64, count=rows)
    truth = [LABELS[index] for index in labels]

    native_rows = {
        row: (row % 3 if row % 10 < 9 else (row % 3 + 1) % 3, 0.99) for row in range(rows)
    }
    native = _distribution(rows, native_rows)
    calibrated_rows = {
        row: (row % 3 if row % 5 < 4 else (row % 3 + 1) % 3, 0.97)
        if row % 5 < 4
        else ((row % 3 + 1) % 3, 0.45)
        for row in range(rows)
    }
    calibrated = _distribution(rows, calibrated_rows)

    decision = _calibrator().decide(native, calibrated, truth)
    assert decision.method == "none"
    assert not decision.keep
    assert decision.calibrated_log_loss < decision.native_log_loss
    assert decision.calibrated_macro_f1 < decision.native_macro_f1 - 0.01
