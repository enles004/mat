from typing import Any

import numpy as np

from src.domain.entities import ArtifactManifest, Prediction, SentimentLabel
from src.nlp.constants import LABELS


class LinearArtifactModel:
    """``SentimentModel`` adapter over a verified linear (TF-IDF) artifact payload.

    The exported baseline payload is a calibrated estimator
    (``CalibratedClassifierCV``), not a bare ``Pipeline``, so scores are read
    from ``predict_proba`` columns reindexed by ``classes_`` into the frozen
    ``SentimentLabel`` order — the same technique as the frozen-test export
    in ``src.nlp.evaluation.selection``. This works for calibrated and plain
    pipeline payloads alike.
    """

    def __init__(
        self, payload: Any, manifest: ArtifactManifest, uncertain_threshold: float
    ) -> None:
        self._payload = payload
        self._manifest = manifest
        self._uncertain_threshold = uncertain_threshold

    def load(self) -> None:
        return None

    def predict(self, text: str) -> Prediction:
        probabilities = np.asarray(self._payload.predict_proba([text]), dtype=float)[0]
        classes = [str(value) for value in self._payload.classes_]
        by_class = dict(zip(classes, probabilities, strict=True))
        scores = {SentimentLabel(label): float(by_class[label]) for label in LABELS}
        best = max(LABELS, key=lambda label: scores[SentimentLabel(label)])
        confidence = scores[SentimentLabel(best)]
        # Empirical uncertainty: a top score below the configured threshold
        # (Settings.uncertain_threshold, default 0.60) is reported as
        # uncertain=True rather than hidden behind a confident label.
        uncertain = confidence < self._uncertain_threshold
        return Prediction(
            label=SentimentLabel(best),
            confidence=confidence,
            scores=scores,
            uncertain=uncertain,
        )

    def metadata(self) -> ArtifactManifest:
        return self._manifest

    def health(self) -> str:
        return "ready"
