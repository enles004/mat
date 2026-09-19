from collections.abc import Sequence as _Sequence

import numpy as _np
import numpy.typing as _npt
import torch as _torch
from sklearn.calibration import (  # type: ignore[import-untyped]
    CalibratedClassifierCV as _CalibratedClassifierCV,
)
from sklearn.linear_model import (  # type: ignore[import-untyped]
    LogisticRegression as _LogisticRegression,
)
from sklearn.metrics import (  # type: ignore[import-untyped]
    f1_score as _f1_score,
)
from sklearn.metrics import (
    log_loss as _log_loss,
)

from src.domain.entities import LABELS as _LABELS
from src.domain.evaluation import MACRO_F1_DROP_TOLERANCE as _MACRO_F1_DROP_TOLERANCE
from src.domain.evaluation import CalibrationDecision as _CalibrationDecision
from src.domain.scoring import multiclass_brier as _multiclass_brier
from src.domain.training import SplitManifest as _SplitManifest
from src.nlp.constants import _PROBABILITY_EPSILON
from src.nlp.modeling.baseline import BaselineTrainer as _BaselineTrainer

MACRO_F1_DROP_TOLERANCE = _MACRO_F1_DROP_TOLERANCE


class ProbabilityCalibrator:
    """Fit, apply, validate, and conservatively select probability calibration."""

    def __init__(self, baseline_trainer: _BaselineTrainer | None = None) -> None:
        self._baseline_trainer = baseline_trainer or _BaselineTrainer()

    @staticmethod
    def fit_temperature(
        logits: _npt.NDArray[_np.float64], labels: _npt.NDArray[_np.int64]
    ) -> float:
        """Return the scalar temperature fitted by LBFGS over log-temperature."""
        values = _torch.tensor(_np.asarray(logits, dtype=_np.float64))
        targets = _torch.tensor(_np.asarray(labels, dtype=_np.int64), dtype=_torch.int64)
        if values.ndim != 2 or values.shape[1] < 2 or values.shape[0] != targets.numel():
            raise ValueError("logits and labels must be aligned rows over two or more classes")

        log_temperature = _torch.zeros(1, dtype=_torch.float64, requires_grad=True)
        optimizer = _torch.optim.LBFGS(
            [log_temperature],
            lr=0.5,
            max_iter=200,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            line_search_fn="strong_wolfe",
        )

        def closure() -> _torch.Tensor:
            optimizer.zero_grad()
            loss = _torch.nn.functional.cross_entropy(values / log_temperature.exp(), targets)
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        optimizer.step(closure)  # type: ignore[no-untyped-call]
        return float(log_temperature.exp().item())

    @staticmethod
    def _log_odds(
        probabilities: _npt.NDArray[_np.float64],
    ) -> _npt.NDArray[_np.float64]:
        clipped = _np.clip(
            probabilities, _PROBABILITY_EPSILON, 1.0 - _PROBABILITY_EPSILON
        )
        return _np.log(clipped / (1.0 - clipped))

    @staticmethod
    def _fit_platt(
        scores: _npt.NDArray[_np.float64], positive: _npt.NDArray[_np.int64]
    ) -> tuple[float, float]:
        """Fit the two-parameter Platt sigmoid as a near-unregularized logistic fit."""
        model = _LogisticRegression(C=1e10, solver="lbfgs", max_iter=10_000)
        model.fit(scores.reshape(-1, 1), positive)
        return float(model.coef_[0, 0]), float(model.intercept_[0])

    @staticmethod
    def _apply_platt(
        scores: _npt.NDArray[_np.float64], slope: float, intercept: float
    ) -> _npt.NDArray[_np.float64]:
        return 1.0 / (1.0 + _np.exp(-(slope * scores + intercept)))

    def _fit_sigmoid_calibrator(
        self,
        native: _npt.NDArray[_np.float64],
        targets: _npt.NDArray[_np.int64],
        fit_indices: _npt.NDArray[_np.int64],
    ) -> list[tuple[float, float]]:
        """Fit one Platt sigmoid per class using only the selected OOF rows."""
        calibrators: list[tuple[float, float]] = []
        for class_index in range(len(_LABELS)):
            positive = (targets[fit_indices] == class_index).astype(_np.int64)
            if int(positive.min()) == int(positive.max()):
                raise ValueError("calibration rows must contain every sentiment class")
            scores = self._log_odds(native[fit_indices, class_index])
            calibrators.append(self._fit_platt(scores, positive))
        return calibrators

    @staticmethod
    def _encode_labels(y_true: _npt.ArrayLike) -> _npt.NDArray[_np.int64]:
        values = _np.asarray(y_true)
        if values.dtype.kind in "iub":
            return values.astype(_np.int64)
        try:
            return _np.array([_LABELS.index(str(value)) for value in values], dtype=_np.int64)
        except ValueError as error:
            raise ValueError("labels must match the frozen SentimentLabel order") from error

    @staticmethod
    def _validated_probabilities(
        probabilities: _npt.ArrayLike, expected_classes: int
    ) -> _npt.NDArray[_np.float64]:
        values = _np.asarray(probabilities, dtype=_np.float64)
        if values.ndim != 2 or values.shape[1] != expected_classes:
            raise ValueError(f"probabilities must be rows over {expected_classes} classes")
        if not _np.isfinite(values).all():
            raise ValueError("probabilities must be finite")
        if values.min() < 0.0 or values.max() > 1.0:
            raise ValueError("probabilities must be in [0, 1]")
        if not _np.isclose(values.sum(axis=1), 1.0).all():
            raise ValueError("probabilities must sum to one")
        return values

    def cross_fitted_sigmoid_oof(
        self,
        probabilities: _npt.ArrayLike,
        y_true: _npt.ArrayLike,
        folds: _npt.ArrayLike,
    ) -> _npt.NDArray[_np.float64]:
        """Leave-one-fold-out sigmoid calibration over persisted development folds."""
        native = self._validated_probabilities(probabilities, len(_LABELS))
        targets = self._encode_labels(y_true)
        fold_ids = _np.asarray(folds)
        if native.shape[0] != targets.shape[0] or fold_ids.shape[0] != native.shape[0]:
            raise ValueError("probabilities, labels, and folds must be aligned")
        unique_folds = sorted(set(fold_ids.tolist()))
        if len(unique_folds) < 2:
            raise ValueError("cross-fitted calibration requires at least two folds")

        calibrated = _np.empty_like(native)
        for fold in unique_folds:
            held_out = _np.flatnonzero(fold_ids == fold)
            fit_indices = _np.flatnonzero(fold_ids != fold)
            calibrators = self._fit_sigmoid_calibrator(native, targets, fit_indices)
            for class_index, (slope, intercept) in enumerate(calibrators):
                scores = self._log_odds(native[held_out, class_index])
                calibrated[held_out, class_index] = self._apply_platt(
                    scores, slope, intercept
                )
        row_sums = calibrated.sum(axis=1, keepdims=True)
        if not _np.isfinite(calibrated).all() or (row_sums <= 0.0).any():
            raise ValueError("cross-fitted calibration produced a degenerate distribution")
        return calibrated / row_sums

    def decide(
        self,
        native_probabilities: _npt.ArrayLike,
        calibrated_probabilities: _npt.ArrayLike,
        y_true: _npt.ArrayLike,
    ) -> _CalibrationDecision:
        """Apply the conservative proper-score and macro-F1 calibration gate."""
        native = self._validated_probabilities(native_probabilities, len(_LABELS))
        calibrated = self._validated_probabilities(calibrated_probabilities, len(_LABELS))
        if native.shape != calibrated.shape:
            raise ValueError("native and calibrated probabilities must share one shape")
        targets = self._encode_labels(y_true)
        labels = [_LABELS[index] for index in targets]

        def scores(
            probabilities: _npt.NDArray[_np.float64],
        ) -> tuple[float, float, float]:
            predicted = [_LABELS[index] for index in probabilities.argmax(axis=1)]
            return (
                float(_log_loss(labels, probabilities, labels=_LABELS)),
                float(_multiclass_brier(labels, probabilities)),
                float(
                    _f1_score(
                        labels,
                        predicted,
                        labels=_LABELS,
                        average="macro",
                        zero_division=0,
                    )
                ),
            )

        native_log_loss, native_brier, native_macro_f1 = scores(native)
        calibrated_log_loss, calibrated_brier, calibrated_macro_f1 = scores(calibrated)
        decision = _CalibrationDecision(
            method="sigmoid",
            native_log_loss=native_log_loss,
            calibrated_log_loss=calibrated_log_loss,
            native_brier=native_brier,
            calibrated_brier=calibrated_brier,
            native_macro_f1=native_macro_f1,
            calibrated_macro_f1=calibrated_macro_f1,
        )
        if decision.keep:
            return decision
        return _CalibrationDecision(
            method="none",
            native_log_loss=native_log_loss,
            calibrated_log_loss=calibrated_log_loss,
            native_brier=native_brier,
            calibrated_brier=calibrated_brier,
            native_macro_f1=native_macro_f1,
            calibrated_macro_f1=calibrated_macro_f1,
        )

    @staticmethod
    def build_fold_pairs(
        manifest: _SplitManifest,
    ) -> list[tuple[_npt.NDArray[_np.int64], _npt.NDArray[_np.int64]]]:
        """Positional pairs over manifest-ordered development rows."""
        if [fold.fold for fold in manifest.folds] != list(range(5)):
            raise ValueError("Calibration requires the five ordered persisted folds")
        position = {row_id: index for index, row_id in enumerate(manifest.development_ids)}
        pairs: list[tuple[_npt.NDArray[_np.int64], _npt.NDArray[_np.int64]]] = []
        for fold in manifest.folds:
            try:
                train = _np.array(
                    sorted(position[row_id] for row_id in fold.train_ids), dtype=_np.int64
                )
                validation = _np.array(
                    sorted(position[row_id] for row_id in fold.validation_ids),
                    dtype=_np.int64,
                )
            except KeyError as error:
                raise ValueError(
                    f"Persisted fold {fold.fold} references unknown development IDs"
                ) from error
            if set(train.tolist()) & set(validation.tolist()):
                raise ValueError(
                    f"Persisted fold {fold.fold} overlaps training and validation rows"
                )
            if set(train.tolist()) | set(validation.tolist()) != set(range(len(position))):
                raise ValueError(f"Persisted fold {fold.fold} does not cover development rows")
            pairs.append((train, validation))
        return pairs

    def build_estimator(
        self,
        seed: int,
        fold_pairs: _Sequence[
            tuple[_npt.NDArray[_np.int64], _npt.NDArray[_np.int64]]
        ],
    ) -> _CalibratedClassifierCV:
        """Build the final calibrated baseline over the persisted fold pairs."""
        return _CalibratedClassifierCV(
            estimator=self._baseline_trainer.build_pipeline(seed),
            method="sigmoid",
            cv=list(fold_pairs),
            ensemble=True,
        )
