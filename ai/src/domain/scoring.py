from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from src.domain.entities import LABELS


def multiclass_brier(
    y_true: Sequence[str], probabilities: npt.NDArray[np.float64]
) -> float:
    """Return the multiclass Brier score using the fixed label order."""
    encoded = np.eye(len(LABELS))[[LABELS.index(label) for label in y_true]]
    return float(np.mean(np.sum((probabilities - encoded) ** 2, axis=1)))
