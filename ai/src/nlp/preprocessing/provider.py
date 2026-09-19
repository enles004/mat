import threading
from pathlib import Path as _Path

from src.domain.artifacts import CatalogActivation as _CatalogActivation
from src.domain.contracts import Normalizer as _Normalizer
from src.domain.exceptions import CatalogError as _CatalogError
from src.nlp.preprocessing.catalog_loader import CatalogLoader as _CatalogLoader
from src.nlp.preprocessing.matcher import RegexMatcher as _RegexMatcher
from src.nlp.preprocessing.normalizer import TextNormalizer as _TextNormalizer

__all__ = ["NormalizerProvider"]


class NormalizerProvider:
    """Holds the active normalizer and swaps it atomically on activation."""

    def __init__(self, lock_path: _Path) -> None:
        self._lock_path = lock_path
        self._swap_lock = threading.Lock()
        self._activation: _CatalogActivation | None = None

    def get(self) -> _Normalizer:
        with self._swap_lock:
            activation = self._activation
        if activation is None:
            raise _CatalogError("normalizer not activated: no catalog has been activated yet")
        return activation.normalizer

    def activate(self, path: _Path, expected_checksum: str) -> _CatalogActivation:
        """Build and validate a replacement off-path, then swap it in once."""
        loader = _CatalogLoader()
        loaded = loader.load(path)
        loader.verify_lock(loaded, self._lock_path)
        if loaded.checksum != expected_checksum:
            raise _CatalogError(
                f"expected catalog checksum mismatch for {path}: "
                f"loaded {loaded.checksum}, expected {expected_checksum}"
            )
        normalizer = _TextNormalizer(_RegexMatcher.from_catalog(loaded.catalog))
        smoke = normalizer.normalize("")
        if smoke.normalized_text != "":
            raise _CatalogError("smoke normalization corrupted empty input")
        activation = _CatalogActivation(
            normalizer=normalizer,
            dictionary_version=loaded.catalog.dictionary_version,
            catalog_checksum=loaded.checksum,
            rule_count=len(loaded.catalog.rules),
        )
        with self._swap_lock:
            self._activation = activation
        return activation
