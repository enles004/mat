import math
from pathlib import Path
from typing import Any

from src.domain.entities import ArtifactManifest, Prediction, SentimentLabel
from src.domain.exceptions import ArtifactUnavailable, TextExceedsModelLimit
from src.nlp.constants import LABELS

_DEPENDENCY_HINT = (
    "transformer backend requires missing dependencies; "
    "install the extra: uv sync --locked --extra transformer"
)


class TransformerArtifactModel:
    """``SentimentModel`` adapter over a local HuggingFace transformer directory.

    Serves the fine-tuned BamiBERT champion exported by
    ``src.nlp.modeling.transformer_champion``. Torch and transformers are
    imported lazily inside ``load()`` so a baseline-only deployment never
    needs them at import time, and the directory is read with
    ``local_files_only=True`` — artifacts load only from the trusted local
    path, never the network. Score handling mirrors ``LinearArtifactModel``:
    head indices map to labels through the *manifest* order (the model
    config's ``id2label`` is not trusted), scores are reindexed into the
    frozen canonical label order, and a top score below the uncertainty
    threshold is reported as ``uncertain=True`` rather than hidden.
    """

    def __init__(
        self, directory: Path, manifest: ArtifactManifest, uncertain_threshold: float
    ) -> None:
        self._directory = directory
        self._manifest = manifest
        self._uncertain_threshold = uncertain_threshold
        self._torch: Any = None
        self._tokenizer: Any = None
        self._classifier: Any = None

    def load(self) -> None:
        if self._classifier is not None:
            return
        try:
            import torch
            import transformers
        except ImportError as error:
            raise ArtifactUnavailable(_DEPENDENCY_HINT) from error
        try:
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(self._directory), local_files_only=True
            )
            classifier = transformers.AutoModelForSequenceClassification.from_pretrained(
                str(self._directory), local_files_only=True
            )
        except ArtifactUnavailable:
            raise
        except Exception as error:
            raise ArtifactUnavailable(
                f"transformer artifact in {self._directory} failed to load: {error}"
            ) from error
        classifier.eval()
        self._torch = torch
        self._tokenizer = tokenizer
        self._classifier = classifier

    def predict(self, text: str) -> Prediction:
        if self._classifier is None or self._tokenizer is None:
            raise RuntimeError("transformer model must be loaded before predict")
        # truncation=False: over-limit text raises instead of being silently
        # shortened — the token count here is the model's true subword count,
        # a second guard beyond the whitespace-based request limiter.
        encoded = self._tokenizer(text, return_tensors="pt", truncation=False)
        limit = self._manifest.max_input_tokens
        if limit is not None and len(encoded["input_ids"][0]) > limit:
            raise TextExceedsModelLimit(limit)
        with self._torch.no_grad():
            output = self._classifier(**encoded)
        raw = output.logits.tolist()[0]
        probabilities = self._softmax(raw)
        # Head index i means manifest.labels[i]; config.id2label is ignored.
        head_labels = dict(
            zip(range(len(self._manifest.labels)), self._manifest.labels, strict=True)
        )
        scores = {
            SentimentLabel(label): probabilities[index] for index, label in head_labels.items()
        }
        ordered = {SentimentLabel(label): scores[SentimentLabel(label)] for label in LABELS}
        best = max(LABELS, key=lambda label: ordered[SentimentLabel(label)])
        confidence = ordered[SentimentLabel(best)]
        uncertain = confidence < self._uncertain_threshold
        return Prediction(
            label=SentimentLabel(best),
            confidence=confidence,
            scores=ordered,
            uncertain=uncertain,
        )

    def _softmax(self, raw: list[float]) -> list[float]:
        largest = max(raw)
        exponentials = [math.exp(value - largest) for value in raw]
        total = sum(exponentials)
        return [value / total for value in exponentials]

    def metadata(self) -> ArtifactManifest:
        return self._manifest

    def health(self) -> str:
        return "ready"
