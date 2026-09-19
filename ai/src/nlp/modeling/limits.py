from src.domain.contracts import SentimentModel as _SentimentModel
from src.domain.exceptions import TextExceedsModelLimit as _TextExceedsModelLimit


class ModelInputLimiter:
    """Count original-text tokens and reject inputs above a model's limit."""

    @staticmethod
    def _token_count(text: str) -> int:
        """Count tokens with the linear backend's deterministic whitespace rule."""
        return len(text.split())

    def enforce(self, model: _SentimentModel, text: str) -> None:
        """Reject text above the artifact's token limit; input is never truncated."""
        limit = model.metadata().max_input_tokens
        if limit is not None and self._token_count(text) > limit:
            raise _TextExceedsModelLimit(limit)
