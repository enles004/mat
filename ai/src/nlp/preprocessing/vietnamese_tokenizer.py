import warnings as _warnings
from collections.abc import Sequence as _Sequence


class VietnameseSegmenter:
    """Deterministic pyvi wrapper with one segment method per text batch."""

    def __init__(self) -> None:
        try:
            from pyvi import ViTokenizer as _ViTokenizer  # type: ignore[import-untyped]
        except ImportError as error:
            raise ValueError(
                "pyvi is required for Vietnamese segmentation; install it with `uv sync --locked`."
            ) from error
        self._tokenizer = _ViTokenizer

    def segment(self, text: str) -> str:
        """Join compound words in one text with underscores."""
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore", SyntaxWarning)
            return str(self._tokenizer.tokenize(text))

    def segment_batch(self, texts: _Sequence[str]) -> list[str]:
        """Segment every text in order; the output aligns with the input."""
        return [self.segment(text) for text in texts]
