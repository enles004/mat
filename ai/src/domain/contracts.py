from collections.abc import Callable
from typing import Literal, Protocol, runtime_checkable

from src.domain.entities import ArtifactManifest, Prediction
from src.domain.normalization import Normalizer


@runtime_checkable
class SentimentModel(Protocol):
    def load(self) -> None:
        raise NotImplementedError

    def predict(self, text: str) -> Prediction:
        raise NotImplementedError

    def metadata(self) -> ArtifactManifest:
        raise NotImplementedError

    def health(self) -> str:
        raise NotImplementedError


ModelLoader = Callable[[Literal["baseline", "transformer"]], SentimentModel]

__all__ = ["ModelLoader", "Normalizer", "SentimentModel"]
