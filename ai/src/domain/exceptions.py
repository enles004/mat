class ArtifactUnavailable(Exception):
    """A model artifact directory is missing, fails verification, or will not load."""


class CatalogError(Exception):
    """The normalization catalog or its reviewed lock is unusable."""


class TokenLimitExceededError(ValueError):
    """Raised when raw text is too long for the resolved model."""


class TransformersVersionIncompatibleError(ValueError):
    """Raised when installed transformers violates the pinned model-card requirement."""


class ModelLoadingProvenanceError(ValueError):
    """Raised when a model load does not preserve the pinned checkpoint backbone."""


class ProbabilityValidationError(ValueError):
    """Raised when OOF probability evidence is not a valid distribution."""


class TextExceedsModelLimit(Exception):
    """Input text exceeds the model's maximum token count.

    Input is never truncated: the request is rejected, and the transport
    layer maps this failure onto its wire error shape.
    """

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(
            f"Input text exceeds the model's maximum of {limit} tokens; "
            "text is never truncated, so the request is rejected."
        )
